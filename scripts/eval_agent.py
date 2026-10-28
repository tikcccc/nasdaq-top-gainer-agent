"""Independent fixture/agent evaluation. Default mode is offline and never sends mail.

    uv run python scripts/eval_agent.py --mode contract --split test
    uv run python scripts/eval_agent.py --mode live --split test --limit 3

Live mode uses the configured real models and Docker but replays fixed synthetic
provider data. It measures decisions separately from deterministic data contracts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import tempfile
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from nasdaq_agent.agent.runner import run
from nasdaq_agent.config import load_settings
from nasdaq_agent.context import start_run
from nasdaq_agent.data import metrics
from nasdaq_agent.data.calendar import MarketSession, _cal
from nasdaq_agent.infra import breaker
from nasdaq_agent.infra.monitoring import aggregate_usage
from nasdaq_agent.providers import yahoo, nasdaq, google_news
from nasdaq_agent.tools import gainers, history
from nasdaq_agent.tools.news import get_news

CASES = Path(__file__).resolve().parent.parent / "tests/fixtures/market_cases.json"


def fixture(case):
    trade = date.fromisoformat(case["trade_date"])
    cal = _cal()
    session = MarketSession(trade, False, cal.previous_session(pd.Timestamp(trade)).date())
    closes = case["closes"]
    main = {"symbol": case["symbol"], "name": f"Example {case['symbol']} Common Stock",
            "price": closes[-1], "pct_change": (closes[-1] / closes[-2] - 1) * 100,
            "net_change": closes[-1] - closes[-2], "volume": 1000000, "exchange": "NMS", "quote_type": "EQUITY",
            "market_time": case["trade_date"]}
    rows = [
        {"symbol": "WARRW", "name": "Example Warrant", "price": .1, "pct_change": 900., "volume": 500,
         "market_time": case["trade_date"]},
        {"symbol": case["penny_symbol"], "name": "Example Penny Common Stock", "price": .5,
         "pct_change": 400., "net_change": .4, "volume": 1000, "market_time": case["trade_date"]}, main]

    def bars(values, end=trade):
        dates = [d.date() for d in cal.sessions_window(pd.Timestamp(end), -len(values))]
        return pd.DataFrame({"date": dates, "open": values, "high": [v*1.02 for v in values],
                             "low": [v*.98 for v in values], "close": values, "volume": 1000000})

    def prices(symbol, start, end, *args, primary=False):
        values = ([.1]*5 + [.5]) if symbol == case["penny_symbol"] else closes if symbol == case["symbol"] else [100,101,102,103,104,105]
        effective_end = session.previous_date if primary and case["regime"] == "stale_primary" else trade
        return bars(values, effective_end)

    news = [{"title": title, "publisher": "Synthetic fixture", "published": case["trade_date"] + "T14:00:00+00:00",
             "link": f"https://example.invalid/{case['id']}/{i}"} for i, title in enumerate(case["news"])]
    return session, rows, prices, news


def evaluate(case, settings, mode):
    session, rows, prices, news = fixture(case)
    started = time.monotonic()
    with (patch.object(gainers, "market_session", lambda: session),
          patch.object(history, "market_session", lambda: session),
          patch.object(yahoo, "screener_gainers", lambda *a, **k: rows),
          patch.object(nasdaq, "screener_gainers", lambda: rows),
          patch.object(yahoo, "history", lambda symbol, start, end: (prices(symbol, start, end, primary=True), {}, {})),
          patch.object(nasdaq, "history", prices),
          patch.object(yahoo, "news", lambda *a, **k: news),
          patch.object(google_news, "news", lambda *a, **k: news)):
        if mode == "live":
            summary = run(settings)
            analysis, selected = summary.get("analysis") or {}, summary.get("selected") or {}
            full_window = case["gold"]["full_window_available"]
            reference = metrics.compute(prices(case["symbol"],None,None),
                {key:prices("^"+key,None,None) for key in ("GSPC","IXIC")},5,case["symbol"]) if full_window else None
            numeric_issues = metrics.compare(analysis,reference) if analysis and reference else ["full analysis unavailable"]
            row = {"id": case["id"], "regime": case["regime"], "mode": mode,
                    "selection_correct": selected.get("symbol") == case["gold"]["filtered_symbol"],
                    "numeric_correct": not numeric_issues and analysis.get("trend") == case["gold"]["trend"] and
                        math.isclose(analysis.get("cumulative_return", math.inf), case["gold"]["cumulative_return"], abs_tol=1e-6),
                    "numeric_issues":numeric_issues,
                    "agent_completed": summary.get("agent_completed", False), "status": summary["status"],
                    "fallback": bool(summary.get("email", {}).get("fallback")), "usage": summary.get("usage"),
                    "model_usage": summary.get("model_usage"),
                    "llm_calls": summary.get("llm_calls"), "guard_rejections": summary.get("guard_rejections"),
                    "models": summary.get("models"), "elapsed_s": round(time.monotonic()-started,2),
                    "run_id": summary["run_id"], "benchmark_correct": bool(analysis) and set(analysis.get("benchmark", {})) == {"GSPC", "IXIC"} and
                        all(math.isclose(v, .05, abs_tol=1e-6) for v in analysis.get("benchmark", {}).values())}
            row["expected_incomplete_handled"] = not case["gold"]["full_window_available"] and row["selection_correct"] and not analysis and row["fallback"]
            row["scenario_pass"] = (row["selection_correct"] and row["numeric_correct"] and row["agent_completed"] and row["benchmark_correct"]) if case["gold"]["full_window_available"] else row["expected_incomplete_handled"]
            trace = settings.output_root / f"run_{summary['run_id']}" / "trace.jsonl"
            row["actual_models"] = sorted({e["provider"] for line in trace.read_text(encoding="utf-8").splitlines()
                if (e := json.loads(line)).get("event") == "llm" and e.get("provider")})
            return row
        start_run(settings)
        breaker.reset()
        screened = gainers.get_top_nasdaq_gainers.func(state={}, tool_call_id="screen").update
        state = {k:v for k,v in screened.items() if k != "messages"}
        selection_ok = state.get("selected", {}).get("symbol") == case["gold"]["filtered_symbol"]
        result = history.get_price_history.func(symbol=case["symbol"], state=state, tool_call_id="history")
        payload = json.loads(result.update["messages"][0].content)
        if not payload["ok"]:
            return {"id":case["id"], "regime":case["regime"], "mode":mode, "selection_correct":selection_ok,
                    "contract_pass":not case["gold"]["full_window_available"], "expected_incomplete":not case["gold"]["full_window_available"],
                    "error":payload["error"]["type"]}
        state.update({k:v for k,v in result.update.items() if k != "messages"})
        stock = pd.read_csv(state["datasets"][state["selected"]["dataset_id"]]["path"])
        actual = metrics.compute(stock, {}, 5, case["symbol"])
        correct = math.isclose(actual["cumulative_return"], case["gold"]["cumulative_return"], abs_tol=1e-6) and actual["trend"] == case["gold"]["trend"]
        news_result = get_news.func(symbol=case["symbol"], state=state, tool_call_id="news")
        news_payload = json.loads(news_result.update["messages"][0].content)
        news_ok = news_payload["ok"] and news_payload["data"]["count"] == len(case["news"])
        news_ok &= all(h["symbol"] == case["symbol"] and h["trade_date"] == case["trade_date"]
                       for h in news_result.update.get("headlines", {}).values())
        full_filters = settings.screener.default_filters.model_copy(update={"min_price":0,"min_volume":0})
        full_settings = settings.model_copy(update={"screener":settings.screener.model_copy(update={"default_filters":full_filters})})
        start_run(full_settings)
        unfiltered = gainers.get_top_nasdaq_gainers.func(state={}, tool_call_id="full").update
        full_ok = unfiltered["selected"]["symbol"] == case["gold"]["unfiltered_common_symbol"]
        return {"id":case["id"], "regime":case["regime"], "mode":mode, "selection_correct":selection_ok,
                "unfiltered_selection_correct":full_ok, "numeric_correct":correct,
                "news_contract_correct":news_ok, "contract_pass":selection_ok and correct and full_ok and news_ok}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["contract", "live"], default="contract")
    p.add_argument("--split", choices=["calibration", "test", "all"], default="test")
    p.add_argument("--limit", type=int)
    p.add_argument("--case-ids", help="comma-separated fixture IDs for a reproducible smoke selection")
    p.add_argument("--provider", choices=["mimo", "deepseek", "openai"])
    p.add_argument("--allow-rule-guard", action="store_true", help="explicit exploratory mode; not a semantic validation result")
    p.add_argument("--output", type=Path, default=Path("outputs/eval/agent.json"))
    p.add_argument("--resume", action="store_true", help="continue this exact dataset/code/provider evaluation; preserve failures")
    args = p.parse_args()
    data = json.loads(CASES.read_text(encoding="utf-8"))
    cases = [c for c in data["cases"] if args.split == "all" or c["split"] == args.split]
    if args.case_ids:
        wanted = set(args.case_ids.split(","))
        cases = [c for c in cases if c["id"] in wanted]
        if {c["id"] for c in cases} != wanted:
            p.error("some --case-ids are absent from the selected split")
    if args.limit is not None:
        cases = cases[:max(0,args.limit)]
    configured=load_settings(overrides={"llm":{"provider":args.provider,"fallback":[]},
        "jev":{"enabled":not args.allow_rule_guard,"allow_rule_fallback":args.allow_rule_guard},
        "email":{"dry_run":True,"to":["eval@example.invalid"]}})
    public_config={key:getattr(configured,key).model_dump(mode="json") for key in
                   ("analysis","screener","data_checks","llm","sandbox","budget")}
    public_config["jev"]=configured.jev.model_dump(mode="json",exclude={"api_key"})
    public_config["run"]=configured.run.model_dump(mode="json",exclude={"output_dir"})
    public_config["models"]={k:{"model":v.model,"base_url":v.base_url,
        "output_token_parameter":v.output_token_parameter} for k,v in configured.providers.items()}
    identity = {"dataset_sha256":hashlib.sha256(CASES.read_bytes()).hexdigest(),
                "guard_sha256":hashlib.sha256(Path(__file__).resolve().parents[1].joinpath("nasdaq_agent/agent/guard.py").read_bytes()).hexdigest(),
                "evaluator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "configuration_sha256":hashlib.sha256(json.dumps(public_config,sort_keys=True).encode()).hexdigest(),
                "prompt_sha256":hashlib.sha256(Path(__file__).resolve().parents[1].joinpath("nasdaq_agent/agent/prompts.py").read_bytes()).hexdigest(),
                "model_runtime_sha256":hashlib.sha256(b"".join(
                    Path(__file__).resolve().parents[1].joinpath(name).read_bytes() for name in
                    ("nasdaq_agent/llm/factory.py", "nasdaq_agent/llm/jev.py", "nasdaq_agent/llm/tracked.py",
                     "nasdaq_agent/infra/usage.py", "nasdaq_agent/infra/monitoring.py", "nasdaq_agent/context.py",
                     "nasdaq_agent/agent/graph.py", "nasdaq_agent/agent/runner.py"))).hexdigest(),
                "mode":args.mode,"provider":args.provider,"split":args.split,
                "case_ids":[c["id"] for c in cases],"allow_rule_guard":args.allow_rule_guard}
    results=[]
    if args.resume and args.output.exists():
        previous=json.loads(args.output.read_text(encoding="utf-8"))
        if previous.get("evaluation_identity") != identity:
            p.error("resume identity changed; use a new output path for a new experiment")
        results=previous["results"]
    def save_report():
        report=build_report(data,args,results,identity)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        pending=args.output.with_suffix(".pending.json")
        pending.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        pending.replace(args.output)
    save_report()
    for case in cases:
        if any(r["id"] == case["id"] for r in results):
            continue
        # Isolate cache and artefacts per case. Live reports stay under outputs for inspection.
        root = Path("outputs/eval/live") / (args.provider or "configured") / case["id"] if args.mode == "live" else Path(tempfile.mkdtemp(prefix="nasdaq_eval_"))
        settings = load_settings(overrides={"run":{"output_dir":str(root.resolve())},
            "email":{"dry_run":True,"to":["eval@example.invalid"]},
            "llm":{"provider":args.provider,"fallback":[]},
            "jev":{"enabled":not args.allow_rule_guard,"allow_rule_fallback":args.allow_rule_guard}})
        case_started=time.monotonic()
        try:
            result=evaluate(case, settings, args.mode)
        except Exception as exc:
            result={"id":case["id"],"regime":case["regime"],"mode":args.mode,"error":f"{type(exc).__name__}: {exc}",
                    "contract_pass":False,"scenario_pass":False,"agent_completed":False,"elapsed_s":round(time.monotonic()-case_started,2)}
        finally:
            if args.mode == "contract":
                import shutil
                if root.resolve().parent != Path(tempfile.gettempdir()).resolve() or not root.name.startswith("nasdaq_eval_"):
                    raise ValueError("refusing to clean an unexpected evaluation directory")
                shutil.rmtree(root)
        results.append(result)
        save_report()
        print(json.dumps(result,ensure_ascii=False),flush=True)
    print(f"Report: {args.output}")
    return 0 if results and all(r.get("contract_pass",False) if args.mode=="contract" else r.get("scenario_pass",False) for r in results) else 1


def build_report(data, args, results, identity):
    report={"dataset_version":data["version"],"dataset_sha256":hashlib.sha256(CASES.read_bytes()).hexdigest(),
            "evaluation_identity":identity,"complete":len(results)==len(identity["case_ids"]),
            "mode":args.mode,"split":args.split,"count":len(results),"limitations":data["limitations"],
            "narrative_checker":"evidence-v1: closed selections, verified prices, attributed headlines; no model-written prose",
            "thresholds":{"numeric_abs_tolerance":1e-6},"usd_cost":None,"results":results}
    report["token_usage_scope"] = "legacy usage/total_tokens: agent chat only; model_usage: all model attempts including sentiment, Jev and retries; unavailable usage remains unknown"
    if args.mode == "contract":
        report["contract_passed"]=sum(r.get("contract_pass",False) for r in results)
    else:
        report["agent_completed"]=sum(r.get("agent_completed",False) for r in results)
        report["scenarios_passed"]=sum(r.get("scenario_pass",False) for r in results)
        report["fallback_count"]=sum(r.get("fallback",False) for r in results)
        report["total_tokens"]=sum((r.get("usage") or {}).get("total_tokens",0) for r in results)
        report["model_usage"] = aggregate_usage([r.get("model_usage") for r in results])
        report["usd_cost"] = report["model_usage"]["estimated_cost_usd"]
        completed = sum(r.get("agent_completed", False) for r in results)
        report["cost_per_qualified_report_usd"] = report["usd_cost"] / completed if completed and report["usd_cost"] is not None else None
        durations=sorted(r["elapsed_s"] for r in results if "elapsed_s" in r)
        report["latency_s"]={"p50":statistics.median(durations),"p95":durations[max(0,math.ceil(.95*len(durations))-1)],
                             "max":max(durations)} if durations else None
        report["errors"]=sum("error" in r for r in results)
        report["guard_rejections"]=sum(r.get("guard_rejections") or 0 for r in results)
        report["by_regime"]={regime:{"count":sum(r.get("regime")==regime for r in results),
            "scenario_passed":sum(r.get("regime")==regime and r.get("scenario_pass",False) for r in results),
            "agent_completed":sum(r.get("regime")==regime and r.get("agent_completed",False) for r in results)}
            for regime in sorted({r["regime"] for r in results if "regime" in r})}
    return report


if __name__ == "__main__":
    raise SystemExit(main())
