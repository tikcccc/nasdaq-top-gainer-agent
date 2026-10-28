"""Replay recorded historical windows through real tools and, optionally, the post-selection agent.

This does not reconstruct a historical screener. Ranking accuracy is explicitly N/A.
All provider calls are intercepted; all email is forced to dry-run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import tempfile
import time
from contextlib import closing
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from nasdaq_agent.agent import runner
from nasdaq_agent.agent.graph import build_graph
from nasdaq_agent.agent.prompts import system_prompt
from nasdaq_agent.config import load_settings
from nasdaq_agent.context import start_run
from nasdaq_agent.data import metrics
from nasdaq_agent.data.calendar import MarketSession, _cal
from nasdaq_agent.infra import breaker, trace
from nasdaq_agent.providers import yahoo, nasdaq, google_news
from nasdaq_agent.tools import history
from nasdaq_agent.tools.news import get_news
from scripts.record_historical import decode

MANIFEST=ROOT/"tests/fixtures/historical/manifest.json"


def verify_recording(case, manifest=MANIFEST):
    for series in case["series"].values():
        raw=(manifest.parent/series["raw_path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=series["sha256"]:
            raise ValueError("historical response checksum mismatch")
        bars,events=decode(json.loads(raw))
        if bars!=series["bars"] or events!=series["events"]:
            raise ValueError("manifest does not match raw provider response")


def evaluate(case, settings, mode):
    verify_recording(case)
    started=time.monotonic()
    symbol=case["symbol"]
    bars=case["series"][symbol]["bars"]
    trade=date.fromisoformat(case["trade_date"])
    session=MarketSession(trade,False,_cal().previous_session(pd.Timestamp(trade)).date())
    candidate=dict(symbol=symbol,name=symbol,price=bars[-1]["close"],volume=bars[-1]["volume"],
                   pct_change=(bars[-1]["close"]/bars[-2]["close"]-1)*100,flags=[])
    initial=dict(session=session.as_dict(),candidates=[candidate],selected={**candidate,"days":5},
                 selection={"primary_symbol":symbol,"label":"Historical symbol window (ranking not evaluated)","filters":{},"filtered":False})

    def recorded_prices(ticker,start,end):
        series=case["series"][ticker]
        events=series["events"]
        def dated_events(kind,value):
            return {datetime.fromtimestamp(e["date"],timezone.utc).date().isoformat():value(e)
                    for e in events.get(kind,{}).values()}
        return (pd.DataFrame(series["bars"]),dated_events("splits",lambda e:e["numerator"]/e["denominator"]),
                dated_events("dividends",lambda e:e["amount"]))

    def unavailable(*args,**kwargs):
        raise RuntimeError("No independent secondary source recorded; network fallback disabled for replay")

    with (patch.object(yahoo,"history",recorded_prices),patch.object(nasdaq,"history",unavailable),
          patch.object(yahoo,"screener_gainers",unavailable),patch.object(nasdaq,"screener_gainers",unavailable),
          patch.object(yahoo,"news",lambda *a,**k:case["news"]),patch.object(google_news,"news",lambda *a,**k:case["news"])):
        ctx=start_run(settings)
        breaker.reset()
        if mode=="live":
            # Reuse the production graph and runner, but seed the historical candidate as an
            # explicit post-selection task. Never fabricate a winner from this small symbol set.
            ctx.prompt=system_prompt(settings)+("\nHISTORICAL REPLAY TASK: This run begins AFTER selection. "
                "The supplied historical symbol, date and candidates are already in state. Override the GOAL "
                "and first-turn screener instruction: do not call get_top_nasdaq_gainers. Start with "
                "get_price_history and get_news for "+symbol+". Ranking has NOT been evaluated; do not call this "
                "stock a top gainer or compare it to other screened stocks. Describe the historical symbol "
                "window using {{selection_label}}. The final session may be down. All other analysis, "
                "citation, report and safety contracts still apply.")
            initial["messages"]=[HumanMessage(f"Analyse the supplied {symbol} historical window ending {case['trade_date']}; ranking is outside this evaluation.")]
            with ctx.ledger.session(),trace.run_logging(ctx.run_dir),closing(sqlite3.connect(settings.output_root/"checkpoints.sqlite",check_same_thread=False)) as conn:
                graph=build_graph(SqliteSaver(conn))
                graph.update_state({"configurable":{"thread_id":ctx.run_id}},initial,as_node="tools")
                summary=runner._run_current(settings,checkpoint_conn=conn,resume_id=ctx.run_id)
            actual=summary.get("analysis") or {}
            issues=metrics.compare(actual,case["gold"]) if actual else ["analysis missing"]
            row=dict(id=case["id"],symbol=symbol,trade_date=case["trade_date"],regime=case["regime"],
                     ranking_accuracy=None,post_selection_completed=summary["agent_completed"],
                     fallback=bool(summary.get("email",{}).get("fallback")),numeric_correct=not issues,numeric_issues=issues,
                     guard_rejections=summary.get("guard_rejections"),usage=summary.get("usage"),
                     model_usage=summary.get("model_usage"),llm_calls=summary.get("llm_calls"),
                     run_id=ctx.run_id,status=summary["status"],stop_reason=summary.get("stop_reason"),
                     elapsed_s=round(time.monotonic()-started,2))
            row["scenario_pass"]=row["post_selection_completed"] and row["numeric_correct"]
            row["actual_models"]=sorted({e["provider"] for line in (ctx.run_dir/"trace.jsonl").read_text(encoding="utf-8").splitlines()
                if (e:=json.loads(line)).get("event")=="llm" and e.get("provider")})
            return row
        update=history.get_price_history.func(symbol=symbol,state=initial,tool_call_id="history").update
        payload=json.loads(update["messages"][0].content)
        if not payload["ok"]:
            return dict(id=case["id"],scenario_pass=False,error=payload["error"])
        state={**initial,**{k:v for k,v in update.items() if k!="messages"}}
        stock=pd.read_csv(state["datasets"][history.dataset_id(symbol)]["path"])
        benchmarks={key:pd.read_csv(state["datasets"][history.dataset_id('^'+key)]["path"]) for key in ("GSPC","IXIC")}
        actual=metrics.compute(stock,benchmarks,5,symbol)
        issues=metrics.compare(actual,case["gold"])
        news_update=get_news.func(symbol=symbol,state=state,tool_call_id="news").update
        news_payload=json.loads(news_update["messages"][0].content)
        dates_correct=list(stock["date"])==[b["date"] for b in bars[-6:]]
        news_correct=news_payload["ok"] and news_payload["data"]["count"]==len(case["news"])
        return dict(id=case["id"],symbol=symbol,trade_date=case["trade_date"],ranking_accuracy=None,
                    numeric_correct=not issues,numeric_issues=issues,dates_correct=dates_correct,news_correct=news_correct,
                    corporate_actions=state["checks"]["corporate_actions"],calendar_close_utc=case["calendar_close_utc"],
                    scenario_pass=not issues and dates_correct and news_correct)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=["contract","live"],default="contract")
    p.add_argument("--provider",choices=["mimo","deepseek"],default="mimo")
    p.add_argument("--case-ids")
    p.add_argument("--output",type=Path,default=ROOT/"outputs/eval/historical.json")
    args=p.parse_args()
    data=json.loads(MANIFEST.read_text(encoding="utf-8"))
    cases=[c for c in data["cases"] if not args.case_ids or c["id"] in args.case_ids.split(',')]
    results=[]
    for case in cases:
        temporary=tempfile.TemporaryDirectory(prefix="nasdaq_historical_") if args.mode=="contract" else None
        root=Path(temporary.name) if temporary else ROOT/"outputs/eval/historical"/args.provider/case["id"]
        settings=load_settings(overrides={"run":{"output_dir":str(root)},"llm":{"provider":args.provider,"fallback":[]},
            "email":{"dry_run":True,"to":["eval@example.invalid"]},"jev":{"enabled":True,"allow_rule_fallback":False}})
        try:
            row=evaluate(case,settings,args.mode)
        except Exception as exc:
            row={"id":case["id"],"scenario_pass":False,"error":f"{type(exc).__name__}: {exc}"}
        finally:
            if temporary:temporary.cleanup()
        results.append(row)
        report=dict(mode=args.mode,provider=args.provider,count=len(results),complete=len(results)==len(cases),
                    manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
                    guard_sha256=hashlib.sha256((ROOT/"nasdaq_agent/agent/guard.py").read_bytes()).hexdigest(),
                    limitations=data["limitations"],ranking_accuracy=None,
                    scenarios_passed=sum(r["scenario_pass"] for r in results),results=results)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        pending=args.output.with_suffix('.pending.json')
        pending.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
        pending.replace(args.output)
        print(json.dumps(row),flush=True)
    return 0 if results and all(r["scenario_pass"] for r in results) else 1


if __name__=="__main__":
    raise SystemExit(main())
