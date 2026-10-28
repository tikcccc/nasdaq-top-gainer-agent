import json

import pytest

from nasdaq_agent.agent import graph as graph_module
from nasdaq_agent.agent.evidence import analysis_problems
from nasdaq_agent.agent.runner import run
from nasdaq_agent.infra import mailer
from nasdaq_agent.tools.history import get_price_history
from nasdaq_agent.tools.code_exec import execute_python
from nasdaq_agent.tools.sentiment import analyze_sentiment
from tests.conftest import SCREENER, SESSION, ANALYSIS_CODE, ScriptedLLM, GOOD_NARRATIVE
from tests.test_graph import HAPPY_PATH, scripted, no_llm_sentiment


def test_real_interruption_resume_and_second_resume_do_not_resend(settings, offline_market, scripted, monkeypatch):
    live = settings.model_copy(update={"email": settings.email.model_copy(update={"dry_run": False})})
    sent = []
    monkeypatch.setattr(mailer, "send", lambda msg, cfg: sent.append(msg) or msg["Message-ID"])
    scripted(HAPPY_PATH)
    class ProcessCrash(BaseException):
        pass
    run_id = "crash_resume"
    def interrupt(event):
        if event["type"] == "tool_result" and event["tool"] == "execute_python" and event["ok"]:
            raise ProcessCrash()
    with pytest.raises(ProcessCrash):
        run(live, run_id=run_id, on_event=interrupt)
    resumed = run(live, resume_id=run_id)
    assert resumed["analysis"] and resumed["email"]["status"] == "sent", resumed
    assert len(sent) == 1
    again = run(live, resume_id=run_id)
    assert again["email"]["status"] == "sent" and len(sent) == 1


def test_send_in_a_batch_is_rejected_before_any_side_effect(settings, offline_market, scripted, monkeypatch):
    sent = []
    monkeypatch.setattr(mailer, "send", lambda msg, cfg: sent.append(msg))
    good = ("send_email", {"selected_news_ids": ["n1"], "include_data_limitations": True})
    scripted(HAPPY_PATH[:7] + [[good, good], [good]])
    result = run(settings)
    assert result["guard_rejections"] == 1 and result["status"] == "ok"
    assert sent == []  # dry run


def snapshot(ctx):
    state = {"candidates": SCREENER, "session": SESSION.as_dict()}
    history = get_price_history.func(symbol="AIOK", state=state, tool_call_id="h")
    state.update({k: v for k, v in history.update.items() if k != "messages"})
    return state


def test_generated_code_cannot_override_identity(ctx, offline_market):
    state = snapshot(ctx)
    code = ANALYSIS_CODE.replace('save_result({', 'save_result({"symbol": "WRONG", ')
    command = execute_python.func(code=code, state=state, tool_call_id="e")
    payload = json.loads(command.update["messages"][0].content)
    assert payload["error"]["type"] == "invalid_result" and "override" in payload["error"]["message"]


def test_report_snapshot_detects_dataset_tampering(ctx, offline_market):
    state = snapshot(ctx)
    result = execute_python.func(code=ANALYSIS_CODE, state=state, tool_call_id="e")
    state.update({k: v for k, v in result.update.items() if k != "messages"})
    assert analysis_problems(state) == []
    path = ctx.data_dir / "ds_AIOK.csv"
    path.write_text(path.read_text().replace("12.02", "99.99"))
    assert any("changed after validation" in p for p in analysis_problems(state))


def test_old_stock_headlines_cannot_score_new_stock(ctx):
    state = {"selected": {"symbol": "CLRO"}, "session": SESSION.as_dict(),
             "headlines": {"n1": {"title": "Old stock news", "symbol": "AIOK", "trade_date": SESSION.trade_date.isoformat()}}}
    result = analyze_sentiment.func(symbol="CLRO", headline_ids=["n1"], state=state, tool_call_id="s")
    assert json.loads(result.update["messages"][0].content)["error"]["type"] == "stale_evidence"


def test_agent_cannot_silently_change_user_filters(ctx, offline_market):
    from nasdaq_agent.tools.gainers import get_top_nasdaq_gainers
    result = get_top_nasdaq_gainers.func(min_price=0, state={}, tool_call_id="scope")
    assert json.loads(result.update["messages"][0].content)["error"]["type"] == "selection_scope_locked"


def test_closed_report_succeeds_without_semantic_service(settings, offline_market, scripted):
    strict = settings.model_copy(update={"jev": settings.jev.model_copy(update={"allow_rule_fallback": False})})
    scripted(HAPPY_PATH)
    result = run(strict)
    assert result["status"] == "ok" and not result["email"].get("fallback")
    assert result["email"]["report_mode"] == "evidence-v2"
    assert not result["stop_reason"]
    assert result["agent_completed"] and not result["task_completed"]  # dry-run is not SMTP delivery


def test_agent_cannot_shorten_the_requested_window(ctx, offline_market):
    result = get_price_history.func(symbol="AIOK", days=2,
                                    state={"candidates": SCREENER, "session": SESSION.as_dict()}, tool_call_id="window")
    assert json.loads(result.update["messages"][0].content)["error"]["type"] == "window_scope_locked"


def test_known_ipo_gap_routes_to_partial_without_model_retry_loop(settings, offline_market, scripted, monkeypatch):
    from nasdaq_agent.providers import yahoo, nasdaq
    from tests.conftest import bars
    monkeypatch.setattr(yahoo, "history", lambda *a: (bars([1,2,3]), {}, {}))
    monkeypatch.setattr(nasdaq, "history", lambda *a: bars([1,2,3]))
    llm = scripted([[("get_top_nasdaq_gainers", {})], [("get_price_history", {"symbol":"AIOK"})]])
    result = run(settings)
    assert llm.calls == 2 and result["status"] == "degraded"
    assert result["selected"]["symbol"] == "AIOK" and not result["analysis"]
    assert not result["agent_completed"] and "insufficient history" in result["stop_reason"]


def test_api_queue_worker_completes_real_graph(settings, offline_market, scripted, monkeypatch):
    import threading
    from fastapi.testclient import TestClient
    from nasdaq_agent import api
    finished = threading.Event()
    actual_run = api.run
    def execute(*args, **kwargs):
        try:
            return actual_run(*args, **kwargs)
        finally:
            finished.set()
    monkeypatch.setattr(api, "_settings", lambda req=None: settings)
    monkeypatch.setattr(api, "run", execute)
    scripted(HAPPY_PATH)
    with TestClient(api.app) as client:
        started = client.post("/api/runs", json={})
        assert started.status_code == 202
        assert finished.wait(20), "queue worker did not finish"
        detail = client.get(f"/api/runs/{started.json()['run_id']}").json()
        assert detail["summary"]["status"] == "ok", detail
        assert detail["summary"]["email"]["status"] == "dry_run"
