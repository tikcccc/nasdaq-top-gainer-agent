"""End-to-end graph runs with a scripted LLM and offline data.

These exercise the real graph, tools, sandbox, guard, fallback and email rendering;
only the LLM and the network are replaced.
"""

import json
import sqlite3

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from nasdaq_agent.agent import graph as graph_module
from nasdaq_agent.agent.runner import run
from nasdaq_agent.infra.schemas import SentimentResult
from nasdaq_agent.tools import sentiment as sentiment_module
from tests.conftest import ANALYSIS_CODE, GOOD_NARRATIVE, ScriptedLLM

HAPPY_PATH = [
    [("record_plan", {"steps": ["screen", "history", "analyse", "news", "email"]})],
    [("get_top_nasdaq_gainers", {"limit": 5})],
    [("get_top_nasdaq_gainers", {"limit": 5, "exclude_derivatives": True, "min_price": 1, "min_volume": 100000})],
    [("get_price_history", {"symbol": "AIOK"}), ("get_news", {"symbol": "AIOK"})],
    [("execute_python", {"code": "import os\nprint(os.listdir())"})],          # policy violation -> fix
    [("execute_python", {"code": ANALYSIS_CODE, "purpose": "5-day analysis"})],
    [("analyze_sentiment", {"symbol": "AIOK", "headline_ids": ["n1", "n2"]})],
    [("send_email", {"narrative": "{{symbol}} jumped 180% today, see {{metrics.cumulative_return}}", "citations": []})],
    [("send_email", {"selected_news_ids": ["n1"], "include_data_limitations": True,
                     "key_findings": [{"kind": "final_session_share"}, {"kind": "news", "ref": "n1"}, {"kind": "vs_sp500"}]})],
    AIMessage(content="Report sent."),
]


@pytest.fixture
def scripted(monkeypatch):
    def install(steps):
        llm = ScriptedLLM(steps)
        monkeypatch.setattr(graph_module, "_llm", lambda run_id: llm)
        return llm
    return install


@pytest.fixture(autouse=True)
def no_llm_sentiment(monkeypatch):
    monkeypatch.setattr(sentiment_module, "_llm", lambda state, settings: SentimentResult(
        method="llm", score=3, label="Positive", confidence=0.8, catalyst="corporate_action",
        catalyst_confidence=0.8, dilution_risk=0.05))


def _trace(settings, run_id):
    path = settings.output_root / f"run_{run_id}" / "trace.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_happy_path_with_self_correction(settings, offline_market, scripted):
    llm = scripted(HAPPY_PATH)
    events = []
    summary = run(settings, on_event=events.append)

    assert summary["status"] == "ok", summary["errors"]
    assert summary["selected"]["symbol"] == "AIOK"
    assert summary["raw_top"]["symbol"] == "ADNWW"
    assert summary["email"]["status"] == "dry_run"
    assert summary["email"]["key_findings_source"] == "agent"
    assert [f["kind"] for f in summary["email"]["key_findings"]] == ["final_session_share", "news", "prior_shock"]
    assert 'rose 23.87% on Oct 20' in summary['email']['key_findings'][-1]['text']
    assert summary["analysis"]["trend"] == "up"
    assert summary["guard_rejections"] == 1
    assert summary["tool_failures"]["execute_python"] == 1   # the os import was blocked
    assert llm.calls == len(HAPPY_PATH) - 1                 # the run ends at delivery: no closing LLM call

    run_dir = settings.output_root / f"run_{summary['run_id']}"
    eml = (run_dir / "email.eml").read_text(encoding="utf-8", errors="replace")
    assert "AIOK" in eml and "Content-ID: <chart>" in eml and "Content-Type: image/png" in eml
    html = (run_dir / "email.html").read_text(encoding="utf-8")
    assert "+182.8%" in html and "ADNWW" not in html
    assert "<svg" in html and "chart-png" in html and "How this report was checked" not in html
    assert "Note:" not in html and "Flags:" not in html and "Volatility (ann.)" not in html   # diagnostics stay in the log
    assert "Volume / prior 5-session avg" in html and "notes" in summary["email"]
    assert "excess vs S&amp;P 500" in html and " pp" in html
    assert '>Method<' in html and 'Screen:' in html and 'Data:' in html
    assert 'Snapshot saved:' not in html
    assert summary['email']['report_facts']['captured_at']
    assert summary['email']['report_facts']['raw_top']['symbol'] == 'ADNWW'
    assert summary['email']['report_facts']['price'] == 12.02
    assert summary['email']['report_facts']['volume'] == 1_000_000
    assert "Key findings." in html and "1. Almost all of the 5-session gain came on Oct 23" in html
    assert "In brief." in html and "AIOK jumped 182.8% on Oct 23; over the last 5 sessions" in html
    assert "How it traded." not in html
    assert html.count('class="hz"') == 6 and html.count("<title>") >= 6   # hover zone + tooltip per day
    assert html.count('class="day-row"') == 5 and "S&amp;P 500" in html
    assert "finance.yahoo.com/quote/AIOK/history/" in html and ">Sources<" in html and "Jev" not in html.split(">Sources<")[-1]
    kinds = [e["type"] for e in events]
    assert "plan" in kinds and "guard" in kinds and kinds[-1] == "run_finished"
    assert any(e["event"] == "guard" and not e["passed"] for e in _trace(settings, summary["run_id"]))


def test_guard_keeps_rejecting_triggers_fallback(settings, offline_market, scripted):
    bad = [("send_email", {"narrative": "{{symbol}} rose 180% and is a strong buy now, {{metrics.trend}}"})]
    scripted(HAPPY_PATH[:7] + [bad, bad, bad, bad])
    summary = run(settings)
    assert summary["status"] == "degraded"
    assert summary["email"]["status"] == "dry_run" and summary["email"]["fallback"]
    assert "guard rejected" in summary["stop_reason"]


def test_llm_outage_still_sends_failure_report(settings, offline_market, scripted):
    scripted([RuntimeError("provider down")])
    summary = run(settings)
    assert summary["status"] == "degraded"
    assert summary["email"]["status"] == "dry_run"
    assert "LLM unavailable" in summary["stop_reason"]
    html = (settings.output_root / f"run_{summary['run_id']}" / "email.html").read_text(encoding="utf-8")
    assert "could not be completed" in html
    assert "provider down" not in html and "Recent errors" not in html and "LLM unavailable" not in html


def test_agent_stopping_early_gets_reference_analysis(settings, offline_market, scripted):
    scripted(HAPPY_PATH[:4] + [AIMessage(content="I give up.")])
    summary = run(settings)
    assert summary["status"] == "degraded"
    assert summary["analysis"]["notes"].startswith("computed by the reference")
    assert "reference implementation" in " ".join(summary["degraded"])


def test_unknown_symbol_is_rejected_with_hint(settings, offline_market, scripted):
    scripted([
        [("get_top_nasdaq_gainers", {"limit": 3})],
        [("get_price_history", {"symbol": "FAKE"})],
        AIMessage(content="stop"),
    ])
    events = []
    run(settings, on_event=events.append)
    err = next(e for e in events if e["type"] == "tool_result" and e["tool"] == "get_price_history")["error"]
    assert err["type"] == "unknown_symbol" and "AIOK" in err["hint"]


def test_new_runs_can_send_the_same_report_again(settings, offline_market, scripted, monkeypatch):
    from nasdaq_agent.infra import mailer
    sent = []
    monkeypatch.setattr(mailer, "send", lambda msg, cfg: sent.append(msg) or msg["Message-ID"])
    live = settings.model_copy(update={"email": settings.email.model_copy(update={"dry_run": False})})
    scripted(HAPPY_PATH)
    assert run(live)["email"]["status"] == "sent"
    scripted(HAPPY_PATH)
    assert run(live)["email"]["status"] == "sent"
    assert len(sent) == 2
    assert sent[0]["Message-ID"] != sent[1]["Message-ID"]


def test_checkpoint_persists_state(settings, offline_market, scripted):
    """Every step is checkpointed to SQLite, which is what --resume continues from."""
    scripted(HAPPY_PATH[:3] + [RuntimeError("crash")])
    first = run(settings)
    assert first["status"] == "degraded"
    conn = sqlite3.connect(settings.output_root / "checkpoints.sqlite")
    saver = SqliteSaver(conn)
    state = graph_module.build_graph(saver).get_state({"configurable": {"thread_id": first["run_id"]}})
    assert state.values["candidates"] and isinstance(state.values["messages"][0], HumanMessage)


def test_screener_is_filtered_by_default(settings, offline_market, scripted):
    """One call is enough: candidates exclude warrants/penny/illiquid rows, raw_top keeps the unfiltered #1."""
    scripted([[("get_top_nasdaq_gainers", {"limit": 5})], AIMessage(content="stop")])
    events = []
    run(settings, on_event=events.append)
    data = next(e for e in events if e["type"] == "tool_result" and e["tool"] == "get_top_nasdaq_gainers")["data"]
    assert data["candidates"][0]["symbol"] == "AIOK"
    assert data["raw_top"]["symbol"] == "ADNWW"
    assert all("derivative" not in c["flags"] for c in data["candidates"])


def test_primary_model_fails_over_fast(settings):
    from nasdaq_agent.llm import factory
    primary = factory._model(settings, "mimo", primary=True)
    fallback = factory._model(settings, "deepseek")
    assert primary.max_retries == 0 and primary.request_timeout == settings.llm.primary_timeout_s
    # Explicit outer retries account for every request instead of hiding them in the SDK.
    assert fallback.max_retries == 0 and fallback.request_timeout == settings.llm.timeout_s


def test_sandbox_outage_goes_straight_to_the_reference_report(settings, offline_market, scripted, monkeypatch):
    """A missing Docker sandbox is not a code bug: no rewrite attempts, no guard deadlock."""
    from nasdaq_agent.sandbox import runner as sandbox_runner
    monkeypatch.setattr(sandbox_runner.shutil, "which", lambda name: None)
    docker = settings.model_copy(update={"sandbox": settings.sandbox.model_copy(update={"backend": "docker"})})
    llm = scripted(HAPPY_PATH[:4] + HAPPY_PATH[5:])

    summary = run(docker)

    assert llm.calls == 5                                    # the run stopped at the first execute_python
    assert summary["status"] == "degraded" and summary["email"]["status"] == "dry_run"
    assert "sandbox unavailable (the Docker CLI is not installed" in summary["stop_reason"]
    assert summary["analysis"]["notes"].startswith("computed by the reference")
    error = next(e for e in _trace(docker, summary["run_id"]) if e["event"] == "tool_result" and e["tool"] == "execute_python")["error"]
    assert error["type"] == "sandbox_unavailable" and error["retryable"] is False
    run_dir = docker.output_root / f"run_{summary['run_id']}"
    for filename in ("email.html", "email.eml"):
        body = (run_dir / filename).read_text(encoding="utf-8")
        assert "sandbox unavailable" not in body and "Partial report:" not in body


def test_last_failed_analysis_attempt_hands_over_to_fallback(settings, offline_market, scripted):
    broken = [("execute_python", {"code": "x = load('ds_AIOK')\nraise ValueError('bad maths')"})]
    llm = scripted(HAPPY_PATH[:4] + [broken] * settings.sandbox.max_attempts + [[("send_email", {"selected_news_ids": []})]])

    summary = run(settings)

    assert llm.calls == 4 + settings.sandbox.max_attempts     # no extra turn arguing with the guard
    assert summary["status"] == "degraded"
    assert f"failed {settings.sandbox.max_attempts} times" in summary["stop_reason"]
    assert summary["analysis"]["notes"].startswith("computed by the reference")


def test_execution_mode_stays_in_run_metadata_not_the_email(settings, offline_market, scripted):
    summary = run(settings) if scripted(HAPPY_PATH) else None
    html = (settings.output_root / f"run_{summary['run_id']}" / "email.html").read_text(encoding="utf-8")
    assert summary["sandbox"] == "trusted_local"
    assert "without container isolation" not in html
