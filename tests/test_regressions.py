"""Regression tests for the concrete failures found during architecture review."""

import json
import math
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from email.message import EmailMessage
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from nasdaq_agent import api
from nasdaq_agent.config import _deep_merge, load_settings
from nasdaq_agent.context import get_ctx, start_run
from nasdaq_agent.data import metrics, prices
from nasdaq_agent.infra import delivery, mailer
from nasdaq_agent.infra.jobs import JobStore, JobWorker
from nasdaq_agent.infra.schemas import AnalysisResult
from nasdaq_agent.sandbox import policy, runner
from tests.conftest import bars, SESSION, STOCK, SCREENER


def test_absolute_output_directory_for_host_docker_mounts(monkeypatch, tmp_path):
    shared = tmp_path / "host-shared-output"
    monkeypatch.setenv("RUN_OUTPUT_DIR", str(shared))
    assert load_settings().output_root == shared


def test_all_metrics_match_a_hand_calculated_example():
    actual = metrics.compute(pd.DataFrame({"close": [100, 110, 99, 108.9, 98.01, 107.811]}),
                             {"GSPC": pd.DataFrame({"close": [100, 101, 102, 103, 104, 105]})}, 5, "X")
    assert actual["daily_returns"] == pytest.approx([.1, -.1, .1, -.1, .1])
    assert actual["avg_daily_return"] == pytest.approx(.02)
    assert actual["cumulative_return"] == pytest.approx(.07811)
    assert actual["trend_slope"] == pytest.approx(.00371)
    assert actual["volatility_daily"] == pytest.approx(math.sqrt(.012))
    assert actual["volatility_annualized"] == pytest.approx(math.sqrt(3.024))
    assert actual["max_drawdown"] == pytest.approx(-.109)
    assert actual["benchmark"] == pytest.approx({"GSPC": .05})
    assert actual["excess_return"] == pytest.approx(.02811)


@pytest.mark.parametrize("field,value", [
    ("daily_returns", [0] * 5), ("volatility_daily", 99), ("volatility_annualized", 999),
    ("max_drawdown", -.99), ("trend_slope", -9), ("benchmark", {"GSPC": -.99}),
    ("excess_return", 88), ("symbol", "WRONG"), ("days", 3), ("trend", "sideways"),
])
def test_each_wrong_field_is_rejected(field, value):
    frame = pd.DataFrame({"close": [10, 11, 12, 13, 14, 15]})
    ref = metrics.compute(frame, {"GSPC": frame}, 5, "X")
    assert metrics.compare({**ref, field: value}, ref)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_analysis_rejects_non_finite_numbers(bad):
    ref = metrics.compute(pd.DataFrame({"close": [1, 2, 3, 4, 5, 6]}), {}, 5, "X")
    with pytest.raises(ValidationError):
        AnalysisResult.model_validate({**ref, "benchmark": {"GSPC": bad}})


def test_price_scale_does_not_change_returns_or_risk():
    frame = pd.DataFrame({"close": [10, 11, 9, 14, 12, 15]})
    assert metrics.compare(metrics.compute(frame * 1000, {}, 5, "X"), metrics.compute(frame, {}, 5, "X")) == []


@pytest.fixture
def client(settings, monkeypatch):
    def configured(overrides=None):
        return load_settings(overrides=_deep_merge(settings.model_dump(), overrides or {}))
    monkeypatch.setattr(api, "load_settings", configured)
    monkeypatch.setattr(api, "_ensure_worker", lambda: None)
    with TestClient(api.app) as c:
        yield c


def test_invalid_email_does_not_block_next_request(client, settings):
    assert client.post("/api/runs", json={"to": ["invalid"]}).status_code == 422
    response = client.post("/api/runs", json={"exclude_penny": False, "exclude_low_volume": False})
    assert response.status_code == 202
    record = JobStore(settings.output_root).get(response.json()["run_id"])
    assert record["status"] == "queued"
    assert json.loads(record["request"])["exclude_penny"] is False
    assert json.loads(record["request"])["effective"]["analysis"]["days"] == settings.analysis.days
    assert "smtp" not in json.loads(record["request"])["effective"]["email"]


def test_filters_default_on_and_can_be_disabled(client):
    assert api.RunRequest().exclude_penny and api.RunRequest().exclude_low_volume
    defaults = api._settings(api.RunRequest()).screener.default_filters
    full = api._settings(api.RunRequest(exclude_penny=False, exclude_low_volume=False)).screener.default_filters
    assert (defaults.min_price, defaults.min_volume) == (1, 100000)
    assert (full.min_price, full.min_volume) == (0, 0)


def test_run_inherits_live_configuration_and_default_recipient(client, settings):
    from pydantic import SecretStr
    settings.email.dry_run = False
    settings.email.smtp.user = 'sender@example.com'
    settings.email.smtp.password = SecretStr('offline-password')
    config = client.get('/api/config').json()
    assert config['email_dry_run_default'] is False
    assert config['email_default_to'] == ['test@example.com']
    response = client.post('/api/runs', json={})
    assert response.status_code == 202 and response.json()['dry_run'] is False
    job = JobStore(settings.output_root).get(response.json()['run_id'])
    effective = json.loads(job['request'])['effective']['email']
    assert effective['dry_run'] is False and effective['to'] == ['test@example.com']


def test_explicit_preview_overrides_live_default_even_with_recipients(client, settings, monkeypatch):
    settings.email.dry_run = False
    settings.email.smtp.password = None  # preview needs no SMTP credentials
    monkeypatch.setenv('SMTP_PASSWORD', '')
    response = client.post('/api/runs', json={'send': False, 'to': ['preview@example.com']})
    assert response.status_code == 202 and response.json()['dry_run'] is True


def test_preview_configuration_remains_preview_when_send_is_omitted(client):
    response = client.post('/api/runs', json={})
    assert response.status_code == 202 and response.json()['dry_run'] is True


def test_live_without_smtp_is_rejected_before_queueing(client, settings, monkeypatch):
    settings.email.smtp.password = None
    monkeypatch.setenv('SMTP_PASSWORD', '')
    response = client.post('/api/runs', json={'send': True})
    assert response.status_code == 422
    assert not JobStore(settings.output_root).active()


def test_authenticated_api_rejects_missing_token(client, settings, monkeypatch):
    from pydantic import SecretStr
    secured = settings.model_copy(update={"api_token": SecretStr("test-secret")})
    monkeypatch.setattr(api, "_settings", lambda req=None: secured)
    assert client.get("/api/config").status_code == 401
    assert client.get("/api/config", headers={"Authorization": "Bearer test-secret"}).status_code == 200


def test_submission_quota_is_bounded(client):
    for _ in range(10):
        assert client.post("/api/runs", json={}).status_code == 202
    assert client.post("/api/runs", json={}).status_code == 429


def test_sse_replays_after_restart_and_honours_cursor(client, settings):
    root = settings.output_root / "run_replay1"
    root.mkdir(parents=True)
    (root / "summary.json").write_text('{}')
    (root / "events.jsonl").write_text('{"type":"first"}\n{"type":"second"}\n')
    response = client.get("/api/runs/replay1/events", headers={"Last-Event-ID": "1"})
    assert response.status_code == 200
    assert "second" in response.text and "first" not in response.text and "id: 2" in response.text


def test_queue_claim_is_atomic_and_crashed_job_is_reclaimable(tmp_path):
    store = JobStore(tmp_path)
    store.enqueue("job1", {"days": 5})
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda owner: store.claim(owner), ["one", "two"]))
    assert sum(r is not None for r in results) == 1
    with store.connect() as conn:
        conn.execute("UPDATE jobs SET lease_until=0")
    recovered = JobStore(tmp_path).claim("restarted")
    assert recovered["resumed"] and recovered["run_id"] == "job1"


def test_worker_failure_is_persisted(tmp_path):
    store = JobStore(tmp_path)
    store.enqueue("job1", {})
    worker = JobWorker(store, lambda job: (_ for _ in ()).throw(ValueError("broken")))
    assert worker.run_once()
    assert store.get("job1")["status"] == "failed"
    assert "broken" in store.get("job1")["error"]


def test_context_does_not_leak_between_concurrent_runs(settings):
    barrier = threading.Barrier(2)
    def task(run_id):
        start_run(settings, run_id)
        barrier.wait(timeout=5)
        return get_ctx().run_id
    with ThreadPoolExecutor(2) as pool:
        assert list(pool.map(task, ["first", "second"])) == ["first", "second"]


def message():
    msg = EmailMessage()
    msg["Message-ID"] = "<test@example.com>"
    msg["To"] = "test@example.com"
    msg.set_content("synthetic test")
    return msg


def test_concurrent_delivery_claim_sends_once(settings, monkeypatch):
    cfg = settings.email.model_copy(update={"dry_run": False})
    entered, release = threading.Event(), threading.Event()
    sent = []
    def fake_send(msg, config):
        sent.append(msg)
        entered.set()
        assert release.wait(5)
    monkeypatch.setattr(mailer, "send", fake_send)
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(delivery.deliver_message, message(), cfg, settings.output_root, "same-key", {})
        assert entered.wait(5)
        second = delivery.deliver_message(message(), cfg, settings.output_root, "same-key", {})
        assert second["status"] == "delivery_unknown"
        release.set()
        assert first.result()["status"] == "sent"
    assert len(sent) == 1


def test_crash_after_smtp_submission_never_resends(settings, monkeypatch):
    cfg = settings.email.model_copy(update={"dry_run": False})
    sent = []
    def crash(msg, config):
        sent.append(msg)
        raise KeyboardInterrupt("process died before recording acknowledgement")
    monkeypatch.setattr(mailer, "send", crash)
    with pytest.raises(KeyboardInterrupt):
        delivery.deliver_message(message(), cfg, settings.output_root, "crash-key", {})
    assert delivery.deliver_message(message(), cfg, settings.output_root, "crash-key", {})["status"] == "delivery_unknown"
    assert len(sent) == 1


def test_failure_notification_does_not_dedupe_a_later_complete_report():
    args = ("same_run", ["test@example.com"])
    assert delivery.operation_key(*args, source="agent", complete=False) != delivery.operation_key(*args, source="agent", complete=True)


def test_forced_delivery_is_still_idempotent_within_one_run(settings, monkeypatch):
    cfg = settings.email.model_copy(update={"dry_run": False})
    sent = []
    monkeypatch.setattr(mailer, "send", lambda msg, cfg: sent.append(msg))
    info = {"run_id": "same_run"}
    assert delivery.deliver_message(message(), cfg, settings.output_root, "force-key", info, force=True)["status"] == "sent"
    assert delivery.deliver_message(message(), cfg, settings.output_root, "force-key", info, force=True)["status"] == "skipped_duplicate"
    assert len(sent) == 1


def test_old_date_stock_recipient_log_does_not_block_new_sends(settings, monkeypatch):
    from nasdaq_agent.infra import sent_log
    sent_log.record(settings.output_root / "sent_log.json", "2026-10-23", "X", ["test@example.com"],
                    {"message_id": "<legacy@example.com>"})
    sent = []
    monkeypatch.setattr(mailer, "send", lambda msg, cfg: sent.append(msg))
    result = delivery.deliver_message(message(), settings.email.model_copy(update={"dry_run": False}),
        settings.output_root, "new-key", {"trade_date":"2026-10-23", "symbol":"X", "recipients":["test@example.com"]})
    assert result["status"] == "sent" and len(sent) == 1


@pytest.fixture
def saved_email(settings):
    root = settings.output_root / "run_manual1"
    root.mkdir(parents=True)
    msg = message()
    msg["Subject"] = "Saved analysis"
    msg["From"] = "sender@example.com"
    msg["Cc"] = "old-cc@example.com"
    msg["Bcc"] = "old-bcc@example.com"
    (root / "email.eml").write_bytes(bytes(msg))
    (root / "summary.json").write_text(json.dumps({"trade_date": "2026-10-23", "selected": {"symbol": "X"}}))
    return root


def test_manual_send_allows_repeats_and_multiple_recipients(client, saved_email, monkeypatch):
    sent = []
    monkeypatch.setattr(mailer, "send", lambda msg, cfg: sent.append(msg))
    recipients = ["one@example.com", "two@example.com", "three@example.com", "four@example.com", "five@example.com"]
    first = client.post("/api/runs/manual1/send", json={"to": recipients})
    second = client.post("/api/runs/manual1/send", json={"to": recipients})
    assert first.status_code == second.status_code == 200
    assert first.json()["status"] == second.json()["status"] == "sent"
    assert first.json()['archived'] is True
    assert first.json()['report_renderer'] == {'version': 'archived-unknown', 'fingerprint': None}
    assert first.json()["request_id"] != second.json()["request_id"]
    assert first.json()["message_id"] != second.json()["message_id"]
    assert len(sent) == 2
    assert sent[0].get_all("To") == [", ".join(recipients)]
    assert not sent[0]["Cc"] and not sent[0]["Bcc"]
    assert sent[0]["Date"]


def test_manual_request_retry_does_not_send_twice(client, saved_email, monkeypatch):
    from uuid import uuid4
    sent = []
    monkeypatch.setattr(mailer, "send", lambda msg, cfg: sent.append(msg))
    request = {"to": ["one@example.com"], "request_id": str(uuid4())}
    first = client.post("/api/runs/manual1/send", json=request)
    second = client.post("/api/runs/manual1/send", json=request)
    assert first.json()["status"] == "sent"
    assert second.json()["status"] == "skipped_duplicate"
    assert first.json()["message_id"] == second.json()["message_id"]
    assert len(sent) == 1


def test_unknown_manual_delivery_blocks_only_the_same_action(client, saved_email, monkeypatch):
    from uuid import uuid4
    sent = []
    def uncertain(msg, cfg):
        sent.append(msg)
        raise mailer.DeliveryUnknown("SMTP acknowledgement lost")
    monkeypatch.setattr(mailer, "send", uncertain)
    request = {"to": ["one@example.com"], "request_id": str(uuid4())}
    assert client.post("/api/runs/manual1/send", json=request).status_code == 409
    assert client.post("/api/runs/manual1/send", json=request).status_code == 409
    assert len(sent) == 1
    monkeypatch.setattr(mailer, "send", lambda msg, cfg: sent.append(msg))
    response = client.post("/api/runs/manual1/send", json={**request, "request_id": str(uuid4())})
    assert response.json()["status"] == "sent" and len(sent) == 2


@pytest.mark.parametrize("body", [
    {"to": []}, {"to": ["invalid"]}, {"to": ["one@example.com\r\nBcc: other@example.com"]},
    {"to": [f"user{i}@example.com" for i in range(6)]},
    {"to": ["one@example.com"], "request_id": "not-a-uuid"},
])
def test_manual_invalid_recipients_or_action_never_send(client, saved_email, monkeypatch, body):
    sent = []
    monkeypatch.setattr(mailer, "send", lambda msg, cfg: sent.append(msg))
    assert client.post("/api/runs/manual1/send", json=body).status_code == 422
    assert not sent


def test_recipient_addresses_are_unique_within_one_message(client, saved_email, monkeypatch):
    sent = []
    monkeypatch.setattr(mailer, "send", lambda msg, cfg: sent.append(msg))
    response = client.post("/api/runs/manual1/send", json={"to": ["same@example.com", " SAME@example.com ", "other@example.com"]})
    assert response.status_code == 200
    assert response.json()["recipients"] == ["SAME@example.com", "other@example.com"]
    assert str(sent[0]["To"]) == "SAME@example.com, other@example.com"


def test_file_io_bypass_and_chart_path_are_rejected(tmp_path):
    assert policy.check("import pandas as pd\nprint(pd.io.common.get_handle('x', 'r').handle.read())")
    result = runner.run("import matplotlib.pyplot as plt\nsave_chart(plt.figure(), '../escape')", {},
                        tmp_path / "sandbox", timeout_s=15, max_stdout=1000, backend="trusted_local")
    assert not result.ok and "without a path" in result.stderr_tail
    assert not (tmp_path / "escape.png").exists()


def test_output_is_stopped_while_running(tmp_path):
    result = runner.run("while True:\n print('x' * 4096, flush=True)", {}, tmp_path / "sandbox",
                        timeout_s=15, max_stdout=1000, backend="trusted_local")
    assert result.output_limited and not result.timed_out and len(result.stdout) <= 1000


def test_missing_docker_does_not_run_locally(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.shutil, "which", lambda name: None)
    result = runner.run("print('should not execute')", {}, tmp_path / "sandbox", timeout_s=10, max_stdout=1000)
    assert not result.ok and result.stdout == "" and "SandboxUnavailable" in result.stderr_tail


def test_stale_history_tries_backup_inside_provider_chain(ctx, offline_market, monkeypatch):
    from nasdaq_agent.providers import yahoo, nasdaq
    from nasdaq_agent.tools.history import _stock_bars
    calls = []
    stale = bars(STOCK)
    stale["date"] = stale["date"] - pd.Timedelta(days=1)
    monkeypatch.setattr(yahoo, "history", lambda *a: (stale, {}, {}))
    monkeypatch.setattr(nasdaq, "history", lambda *a: calls.append(a) or bars(STOCK))
    _, _, source, warnings = _stock_bars("AIOK", SESSION.previous_date, SESSION.trade_date)
    assert source == "nasdaq" and calls and warnings


def test_benchmark_outage_does_not_discard_stock_analysis(ctx, offline_market, monkeypatch):
    from nasdaq_agent.providers import yahoo, nasdaq
    from nasdaq_agent.tools.history import get_price_history
    original = yahoo.history
    monkeypatch.setattr(yahoo, "history", lambda symbol, *a: original(symbol, *a) if symbol == "AIOK" else (_ for _ in ()).throw(RuntimeError("benchmark offline")))
    monkeypatch.setattr(nasdaq, "history", lambda *a: (_ for _ in ()).throw(RuntimeError("offline")))
    command = get_price_history.func(symbol="AIOK", state={"candidates": SCREENER, "session": SESSION.as_dict()}, tool_call_id="t")
    assert command.update["selected"]["symbol"] == "AIOK"
    assert set(command.update["datasets"]) == {"ds_AIOK"}


def test_missing_session_is_not_mistaken_for_a_daily_return():
    frame = prices.normalise(bars(STOCK))
    frame = frame.drop(index=len(frame) - 3)
    with pytest.raises(Exception, match="consecutive sessions"):
        prices.validate_window(frame, symbol="X", days=5, trade_date=SESSION.trade_date,
                               is_intraday=False, outlier_move=.8)


def test_split_data_survives_a_price_fallback(ctx, offline_market, monkeypatch):
    from nasdaq_agent.providers import yahoo, nasdaq
    from nasdaq_agent.tools.history import _stock_bars
    # Yahoo returns the split but too few bars (as for GCTK on 26 Oct 2026); Nasdaq supplies the prices.
    monkeypatch.setattr(yahoo, "history", lambda *a: (bars(STOCK).tail(1), {SESSION.trade_date: 0.5}, {}))
    monkeypatch.setattr(nasdaq, "history", lambda *a, **k: bars(STOCK))
    _, actions, source, _ = _stock_bars("AIOK", SESSION.previous_date, SESSION.trade_date)
    assert source == "nasdaq" and [(a.kind, a.value) for a in actions] == [("reverse_split", 0.5)]
    monkeypatch.setattr(yahoo, "history", lambda *a: (_ for _ in ()).throw(RuntimeError("offline")))
    _, actions, source, _ = _stock_bars("AIOK", SESSION.previous_date, SESSION.trade_date)
    assert source == "nasdaq" and actions is None          # unknown, not "none"


def test_unchecked_corporate_actions_are_flagged(ctx, offline_market, monkeypatch):
    from nasdaq_agent.providers import yahoo
    from nasdaq_agent.tools.history import get_price_history
    original = yahoo.history
    monkeypatch.setattr(yahoo, "history", lambda symbol, *a: (_ for _ in ()).throw(RuntimeError("offline")) if symbol == "AIOK" else original(symbol, *a))
    command = get_price_history.func(symbol="AIOK", state={"candidates": SCREENER, "session": SESSION.as_dict()}, tool_call_id="t")
    assert "corporate_actions_unchecked" in command.update["selected"]["flags"]
    assert command.update["checks"]["splits"] == [] and command.update["checks"]["dividends"] == []


def _split_on_trade_date(monkeypatch, *, open_, close):
    """AIOK bars that are not split-adjusted, with a 1-for-2 reverse split dated on the trade date."""
    from nasdaq_agent.providers import yahoo, nasdaq
    stock = bars(STOCK[:-1] + [close])
    last = stock.index[-1]
    stock.loc[last, ["Open", "High", "Low"]] = [open_, max(open_, close) * 1.01, min(open_, close) * 0.99]
    yahoo_history, nasdaq_history = yahoo.history, nasdaq.history
    monkeypatch.setattr(yahoo, "history", lambda symbol, *a: (stock.copy(), {SESSION.trade_date: 0.5}, {})
                        if symbol == "AIOK" else yahoo_history(symbol, *a))
    monkeypatch.setattr(nasdaq, "history", lambda symbol, *a, **k: stock.copy() if symbol == "AIOK" else nasdaq_history(symbol, *a, **k))


def _split_state(*candidates):
    from nasdaq_agent.data import ranking
    aiok = {**SCREENER[1], "pct_change": 102.35, "price": 8.6, "flags": []}
    rows = [aiok, *({**SCREENER[2], "flags": []} for _ in candidates[:1])]
    # A higher row ruled out only because its actual move (80%) is below AIOK's screener gain.
    status = ranking.assess({"symbol": "AIOK", "pct_change": 102.35},
                            [{"symbol": "ZZZ", "reason": "quote not verified", "actual_move": 80.0}], is_intraday=False)
    return {"candidates": rows, "session": SESSION.as_dict(), "ranking": status,
            "selection": {"primary_symbol": "AIOK", "filters": {}, "label": "Top NASDAQ common-stock gainer"}}


def test_a_gain_that_is_only_a_split_hands_the_ranking_to_the_next_candidate(ctx, offline_market, monkeypatch):
    from nasdaq_agent.tools.history import get_price_history
    _split_on_trade_date(monkeypatch, open_=8.5, close=8.6)      # opens at twice the 4.25 previous close
    command = get_price_history.func(symbol="AIOK", state=_split_state("CLRO"), tool_call_id="t")
    update = command.update
    assert update["errors"][0]["type"] == "split_artifact" and "CLRO" in update["errors"][0]["hint"]
    assert [c["symbol"] for c in update["candidates"]] == ["CLRO"]
    assert update["selection"]["primary_symbol"] == "CLRO"
    [excluded] = update["selection"]["split_excluded"]
    assert (excluded["symbol"], excluded["screener_pct"], round(excluded["adjusted_pct"], 2)) == ("AIOK", 102.35, 1.18)
    # ZZZ's 80% was below AIOK's screener figure but is above CLRO's 69%: the ranking is reassessed.
    assert update["ranking"]["status"] == "uncertain" and "AIOK" in update["ranking"]["ruled_out"]
    assert update["selected"]["symbol"] == "CLRO" and "ranking_unverified" in update["selected"]["flags"]
    assert "datasets" in update and update["datasets"] == {}


def test_a_split_adjusted_stock_that_still_leads_keeps_its_place(ctx, offline_market, monkeypatch):
    from nasdaq_agent.tools.history import get_price_history
    _split_on_trade_date(monkeypatch, open_=8.5, close=8.6)
    command = get_price_history.func(symbol="AIOK", state=_split_state(), tool_call_id="t")
    update = command.update
    assert update["selected"]["symbol"] == "AIOK"
    assert round(update["checks"]["split_move"]["adjusted_pct"], 2) == 1.18
    assert update["checks"]["splits"][0]["status"] == "adjusted"
    assert not {"data_conflict", "possible_split_artifact", "source_conflict"} & set(update["selected"]["flags"])
    saved = pd.read_csv(update["datasets"]["ds_AIOK"]["path"])
    assert list(saved["close"].tail(2)) == [8.5, 8.6] and saved["volume"].iloc[-2] == 500_000


def test_a_split_date_without_a_price_gap_leaves_the_gain_as_traded(ctx, offline_market, monkeypatch):
    from nasdaq_agent.tools.history import get_price_history
    _split_on_trade_date(monkeypatch, open_=4.3, close=8.6)      # GCTK pattern: opened at the old level, rose in trading
    command = get_price_history.func(symbol="AIOK", state=_split_state("CLRO"), tool_call_id="t")
    update = command.update
    assert update["selected"]["symbol"] == "AIOK" and update["checks"]["split_move"] is None
    assert update["checks"]["splits"] == [{"date": str(SESSION.trade_date), "ratio": 0.5, "status": "no_gap",
                                           "open": 4.3, "previous_close": 4.25}]
    assert "possible_split_artifact" not in update["selected"]["flags"]
