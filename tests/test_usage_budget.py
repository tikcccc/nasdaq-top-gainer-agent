"""Budget invariants, actual SDK requests, recovery and cost completeness; no network."""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, SecretStr

from nasdaq_agent.config import BudgetConfig, ModelPrice
from nasdaq_agent.context import start_run
from nasdaq_agent.infra.monitoring import snapshot
from nasdaq_agent.infra.usage import BudgetExceeded, UsageLedger
from nasdaq_agent.llm import factory, jev, tracked


def configure(settings, **budget):
    return settings.model_copy(update={"budget": BudgetConfig(**budget)})


def reserve(ctx, settings, **kwargs):
    return ctx.ledger.reserve(ctx.run_id, settings, provider="mimo", model="offline", stage="agent",
                              attempt=1, input_estimate=100, output_estimate=50, **kwargs)


def response(inp=100, out=20, cached=0):
    return AIMessage(content="ok", usage_metadata={"input_tokens": inp, "output_tokens": out,
        "total_tokens": inp + out, "input_token_details": {"cache_read": cached}})


def wire_reply(tool_calls=None):
    return {"id": "chat-1", "object": "chat.completion", "created": 1, "model": "offline",
            "choices": [{"index": 0, "finish_reason": "tool_calls" if tool_calls else "stop",
                         "message": {"role": "assistant", "content": "" if tool_calls else "ok", **({"tool_calls": tool_calls} if tool_calls else {})}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}}


def transport_models(monkeypatch, handler):
    original = factory.ChatOpenAI
    monkeypatch.setattr(factory, "ChatOpenAI", lambda **kw: original(
        **kw, http_client=httpx.Client(transport=httpx.MockTransport(handler))))


@pytest.mark.parametrize("budget", [
    {"max_model_requests": 1}, {"max_total_tokens": 200},
    {"max_estimated_cost_usd": .0008, "prices": {"mimo:offline":
        ModelPrice(version="fixture", input_per_million=2, output_per_million=8)}},
])
def test_concurrent_requests_cannot_spend_the_same_slot(settings, budget):
    settings = configure(settings, **budget)
    ctx = start_run(settings, "concurrent")
    def compete():
        # Separate ledger instances/connections exercise cross-worker atomicity.
        ledger = UsageLedger(settings.output_root)
        try:
            ledger.reserve(ctx.run_id, settings, provider="mimo", model="offline", stage="agent",
                           attempt=1, input_estimate=100, output_estimate=50)
            return "admitted"
        except BudgetExceeded:
            return "blocked"
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: compete(), range(8)))
    assert results.count("admitted") == 1 and results.count("blocked") == 7


def test_resume_retains_deadline_and_pending_reservation(settings):
    settings = configure(settings, max_model_requests=1)
    ctx = start_run(settings, "resume_budget")
    reserve(ctx, settings)
    with ctx.ledger.connect() as conn:
        conn.execute("UPDATE usage_runs SET started=? WHERE run_id=?", (time.time() - 30, ctx.run_id))
    resumed = start_run(settings, ctx.run_id)
    assert resumed.seconds_left() < settings.run.deadline_s - 29
    assert resumed.ledger.summary(ctx.run_id)["pending"] == 1
    with pytest.raises(BudgetExceeded, match="model_requests"):
        reserve(resumed, settings)


def test_legacy_run_never_claims_zero_historical_cost(settings):
    folder = settings.output_root / "run_legacy"
    folder.mkdir(parents=True)
    started = time.time() - 30
    (folder / "trace.jsonl").write_text(json.dumps({"event": "run_started", "ts": started}) + "\n", encoding="utf-8")
    ctx = start_run(settings, "legacy")
    assert ctx.started_at == started and ctx.ledger.summary("legacy")["legacy_untracked"]
    assert ctx.ledger.summary("legacy")["estimated_cost_usd"] is None
    assert snapshot(ctx.ledger)["estimated_cost_usd"] is None
    with pytest.raises(BudgetExceeded, match="untracked_legacy_usage"):
        reserve(ctx, configure(settings, max_model_requests=10))


def test_unknown_usage_is_not_free_and_retains_token_reservation(settings):
    settings = configure(settings, max_total_tokens=200)
    ctx = start_run(settings, "unknown_usage")
    call_id = reserve(ctx, settings)
    ctx.ledger.finish(call_id, latency_ms=1, response=AIMessage(content="missing usage"))
    totals = ctx.ledger.summary(ctx.run_id)
    assert totals["usage_coverage"] == 0 and totals["estimated_cost_usd"] is None
    assert totals["reserved_unknown_tokens"] == 150
    with pytest.raises(BudgetExceeded, match="total_tokens"):
        reserve(ctx, settings)


def test_cost_is_versioned_and_cache_discount_is_settled(settings):
    price = ModelPrice(version="fixture-v1", input_per_million=2, output_per_million=8, cached_input_per_million=.5)
    settings = configure(settings, prices={"mimo:offline": price}, max_estimated_cost_usd=.001)
    ctx = start_run(settings, "priced")
    call_id = reserve(ctx, settings)
    ctx.ledger.finish(call_id, latency_ms=1, response=response(cached=50))
    assert ctx.ledger.summary(ctx.run_id)["estimated_cost_usd"] == pytest.approx(.000285)
    row = ctx.ledger.records(ctx.run_id)[0]
    assert json.loads(row["pricing"])["version"] == "fixture-v1"
    changed = configure(settings, max_estimated_cost_usd=.001,
                        prices={"mimo:offline": price.model_copy(update={"input_per_million": 999})})
    with pytest.raises(BudgetExceeded, match="estimated_cost_usd"):
        reserve(ctx, changed)
    assert ctx.ledger.summary(ctx.run_id)["estimated_cost_usd"] == pytest.approx(.000285)


def test_unpriced_model_blocks_dispatch_only_when_cost_limit_enabled(settings, monkeypatch):
    settings = configure(settings, max_estimated_cost_usd=1)
    ctx = start_run(settings, "unpriced")
    sent = []
    transport_models(monkeypatch, lambda req: sent.append(req) or httpx.Response(200, json=wire_reply()))
    with pytest.raises(BudgetExceeded, match="unpriced_model"):
        factory.build_chat(settings).invoke([HumanMessage("hello")])
    assert sent == [] and ctx.ledger.summary(ctx.run_id)["requests"] == 0


def test_actual_sdk_retry_fallback_usage_and_remaining_timeout(settings, monkeypatch):
    ctx = start_run(settings, "sdk_attempts")
    ctx.started_at = time.time() - settings.run.deadline_s + 4
    requests = []
    def handler(req):
        requests.append(req)
        if req.url.host == "mimo.invalid":
            return httpx.Response(503, json={"error": {"message": "down"}})
        if len(requests) == 2:
            return httpx.Response(429, headers={"Retry-After": "0"}, json={"error": {"message": "rate limited"}})
        return httpx.Response(200, json=wire_reply())
    transport_models(monkeypatch, handler)
    result = factory.build_chat(settings).invoke([HumanMessage("hello")])
    assert result.content == "ok" and len(requests) == 3
    assert all(0 < req.extensions["timeout"]["read"] <= 4 for req in requests)
    for req in requests:
        parameter = "max_tokens" if req.url.host == "deepseek.invalid" else "max_completion_tokens"
        assert json.loads(req.content)[parameter] == settings.llm.max_output_tokens
        assert ("max_completion_tokens" if parameter == "max_tokens" else "max_tokens") not in json.loads(req.content)
    rows = ctx.ledger.records(ctx.run_id)
    assert [(r["provider"], r["attempt"]) for r in rows] == [("mimo", 1), ("deepseek", 1), ("deepseek", 2)]
    assert [r["usage_known"] for r in rows] == [0, 0, 1]
    totals = ctx.ledger.summary(ctx.run_id)
    assert totals["input_tokens"] == 100 and totals["usage_coverage"] == pytest.approx(1 / 3)


def test_request_budget_cannot_fail_over_or_retry(settings, monkeypatch):
    settings = configure(settings, max_model_requests=1)
    ctx = start_run(settings, "no_bypass")
    sent = []
    transport_models(monkeypatch, lambda req: sent.append(req) or httpx.Response(503, json={"error": {"message": "down"}}))
    with pytest.raises(BudgetExceeded, match="model_requests"):
        factory.build_chat(settings).invoke([HumanMessage("hello")])
    assert len(sent) == 1 and ctx.ledger.summary(ctx.run_id)["requests"] == 1


def test_structured_sentiment_keeps_usage_and_deadline_timeout(settings, monkeypatch):
    class Score(BaseModel):
        value: int
    ctx = start_run(settings, "structured")
    ctx.started_at = time.time() - settings.run.deadline_s + 3
    sent = []
    def handler(req):
        sent.append(req)
        return httpx.Response(200, json=wire_reply([{"id": "t1", "type": "function",
            "function": {"name": "Score", "arguments": '{"value": 3}'}}]))
    transport_models(monkeypatch, handler)
    assert factory.build_structured(settings, Score).invoke([HumanMessage("score")]).value == 3
    assert sent[0].extensions["timeout"]["read"] <= 3
    totals = ctx.ledger.summary(ctx.run_id)
    assert totals["total_tokens"] == 120 and totals["by_model_stage"][0]["stage"] == "sentiment"


def test_jev_shares_budget_and_extracts_usage(settings, monkeypatch):
    settings = configure(settings, max_model_requests=1)
    settings.jev.enabled = True
    settings.jev.api_key = SecretStr("fixture-key")
    ctx = start_run(settings, "jev_budget")
    calls = []
    def system_one(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(usage=SimpleNamespace(model_dump=lambda: {"input_tokens": 50, "output_tokens": 8}))
    monkeypatch.setattr(jev, "_client", lambda *args: SimpleNamespace(system_one=system_one))
    jev.ask(settings, {}, {"q": jev.Noul(instructions="test")}, stage="sentiment")
    with pytest.raises(BudgetExceeded, match="model_requests"):
        jev.ask(settings, {}, {"q": jev.Noul(instructions="test")})
    assert len(calls) == 1 and ctx.ledger.summary(ctx.run_id)["total_tokens"] == 58


def test_actual_jev_sdk_has_no_hidden_retries(settings, monkeypatch):
    import httpx2
    from typesafe_sdk import TypeSafeClient, RetryPolicy
    settings = configure(settings, max_model_requests=2)
    settings.jev.enabled, settings.jev.api_key = True, SecretStr("fixture")
    ctx = start_run(settings, "jev_transport")
    sent = []
    def handler(req):
        sent.append(req)
        if len(sent) == 1:
            return httpx2.Response(429, headers={"Retry-After": "0"}, json={"error": "rate limited"})
        return httpx2.Response(200, json={"model": "fixture", "usage": {"input_tokens": 42, "output_tokens": 5},
            "answers": {"q": {"type": "noul", "noul": .8}}})
    with TypeSafeClient(api_key="fixture", base_url="https://jev.invalid", model="fixture", compression=False,
                        retry=RetryPolicy(max_retries=0),
                        http_client=httpx2.Client(transport=httpx2.MockTransport(handler))) as client:
        monkeypatch.setattr(jev, "_client", lambda *args: client)
        assert jev.ask(settings, {}, {"q": jev.Noul(instructions="test")}).nouls["q"].noul == .8
    rows = ctx.ledger.records(ctx.run_id)
    assert len(sent) == 2 and [r["attempt"] for r in rows] == [1, 2]
    assert rows[0]["http_status"] == 429 and rows[1]["input_tokens"] == 42


def test_budget_exhaustion_in_sentiment_does_not_call_fallback_model(ctx, monkeypatch):
    from nasdaq_agent.tools import sentiment
    monkeypatch.setattr(jev, "available", lambda settings: True)
    def exhausted(*args):
        raise BudgetExceeded("model_requests")
    monkeypatch.setattr(sentiment, "_jev", exhausted)
    monkeypatch.setattr(sentiment, "_llm", lambda *args: pytest.fail("budget must not fall through to LLM sentiment"))
    result = sentiment.analyze_sentiment.func(symbol="AIOK", headline_ids=["n1"], tool_call_id="s", state={
        "selected": {"symbol": "AIOK"}, "session": {"trade_date": "2026-10-23"},
        "headlines": {"n1": {"symbol": "AIOK", "trade_date": "2026-10-23", "title": "fixture"}}})
    assert result.update["stop_reason"] == "budget_exceeded: model_requests"


def test_monitoring_cost_per_qualified_report_includes_failed_runs(settings):
    settings = configure(settings, prices={"mimo:offline": ModelPrice(
        version="fixture", input_per_million=2, output_per_million=8)})
    for run_id, ok in (("passed", True), ("failed", False)):
        ctx = start_run(settings, run_id)
        call_id = reserve(ctx, settings)
        ctx.ledger.finish(call_id, latency_ms=2, response=response())
        ctx.ledger.finish_run({"run_id": run_id, "status": "ok" if ok else "failed",
            "agent_completed": ok, "task_completed": False, "elapsed_s": 3})
    result = snapshot(ctx.ledger)
    assert result["cost_per_qualified_report_usd"] == pytest.approx(.00072)
    assert result["failed_rate"] == .5 and result["normal_completion_rate"] == .5


def test_retry_after_cannot_wait_past_deadline(ctx):
    class Unavailable(Exception):
        headers = {"Retry-After": str(ctx.settings.run.deadline_s + 1)}
    with pytest.raises(BudgetExceeded, match="deadline"):
        tracked.wait_before_retry(Unavailable(), 1)


def test_process_interruption_retains_unknown_pending_usage(ctx):
    class Crash(BaseException):
        pass
    def crash(timeout):
        raise Crash()
    with pytest.raises(Crash):
        tracked.request(ctx.settings, provider="mimo", model="offline", stage="agent", attempt=1,
                        input_estimate=100, output_estimate=50, timeout_s=10, invoke=crash)
    assert ctx.ledger.summary(ctx.run_id)["pending"] == 1


def test_production_graph_degrades_on_budget_without_another_model(settings, monkeypatch):
    from nasdaq_agent.agent.runner import run
    settings = configure(settings, max_model_requests=1)
    sent = []
    transport_models(monkeypatch, lambda req: sent.append(req) or httpx.Response(503, json={"error": {"message": "down"}}))
    summary = run(settings)
    assert len(sent) == 1 and summary["email"]["fallback"]
    assert summary["stop_reason"] == "budget_exceeded: model_requests"
    assert summary["model_usage"]["requests"] == 1
    metrics = snapshot(start_run(settings, summary["run_id"]).ledger)
    assert metrics["budget_stops"] == 1 and metrics["degraded_rate"] == 1


def test_usage_and_monitoring_api(settings, monkeypatch):
    from fastapi.testclient import TestClient
    from nasdaq_agent import api
    ctx = start_run(settings, "api_usage")
    call_id = reserve(ctx, settings)
    ctx.ledger.finish(call_id, latency_ms=2, response=response())
    monkeypatch.setattr(api, "_settings", lambda req=None: settings)
    with TestClient(api.app) as client:
        data = client.get("/api/runs/api_usage/usage?limit=1").json()
        assert data["summary"]["total_tokens"] == 120 and len(data["requests"]) == 1
        assert client.get("/api/runs/api_usage/usage?offset=1").json()["requests"] == []
        metrics = client.get("/api/metrics").json()
        assert metrics["requests"] == 1 and metrics["queue"]["queued"] == 0
        assert client.get("/api/runs/missing_run/usage").status_code == 404
