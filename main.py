"""CLI: run the NASDAQ top-gainer agent and watch its decisions live.

    python main.py                 # dry run: email saved to outputs/run_*/email.eml
    python main.py --send          # really send via Gmail SMTP
    python main.py --days 10 --provider deepseek --to someone@example.com --send
    python main.py --resume <run_id>
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from nasdaq_agent.agent.runner import run
from nasdaq_agent.config import load_settings

ICONS = {True: "✔", False: "✖", None: "•"}


def _fmt(value: Any, limit: int = 140) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _result_brief(event: dict) -> str:
    data = event.get("data") or {}
    tool = event["tool"]
    if tool == "get_top_nasdaq_gainers":
        top = (data.get("candidates") or [{}])[0]
        return (f"trade date {data.get('trade_date')} via {data.get('source')}; raw #1 {data['raw_top']['symbol']} "
                f"{data['raw_top']['pct_change']:+.2f}% {data['raw_top']['flags']}; top candidate {top.get('symbol')} "
                f"{top.get('pct_change', 0):+.2f}% {top.get('flags')}")
    if tool == "get_price_history":
        return f"datasets {[d['dataset_id'] for d in data.get('datasets', [])]}; {data.get('checks', {}).get('screener_vs_history')}; {data.get('checks', {}).get('price_confirmation')}; flags {data.get('flags')}"
    if tool == "execute_python":
        a = data.get("analysis") or {}
        return (f"cumulative {a.get('cumulative_return', 0):+.2%}, trend {a.get('trend')}, "
                f"vol(ann) {a.get('volatility_annualized', 0):.0%}, excess vs S&P {a.get('excess_return', 0):+.2%} - {data.get('verified')}")
    if tool == "analyze_sentiment":
        return f"{data.get('label')} (score {data.get('score')}, conf {data.get('confidence')}), catalyst {data.get('catalyst')} via {data.get('method')}"
    if tool == "get_news":
        return f"{data.get('count')} headlines: " + "; ".join(h["title"][:50] for h in data.get("untrusted_headlines", [])[:3])
    if tool == "send_email":
        return f"{data.get('status')} -> {data.get('recipients')} ({data.get('eml_path')})"
    if tool == "record_plan":
        return f"v{data.get('plan_version')}: " + " | ".join(data.get("steps", []))
    return _fmt(data)


def print_event(event: dict) -> None:
    kind = event["type"]
    if kind == "run_started":
        mode = "DRY RUN (no email sent)" if event["dry_run"] else f"LIVE -> {event['recipients']}"
        print(f"\n▶ run {event['run_id']}  models {event['models']}  {mode}\n")
    elif kind == "thought":
        print(f"  💭 {_fmt(event['text'], 300)}")
    elif kind == "tool_call":
        args = dict(event["args"])
        if "code" in args:
            args["code"] = f"<{len(args['code'].splitlines())} lines of generated Python>"
        if "narrative" in args:
            args["narrative"] = _fmt(args["narrative"], 90)
        print(f"  → {event['tool']}({_fmt(args, 220)})")
    elif kind == "tool_result":
        if event["ok"]:
            print(f"    {ICONS[True]} {event['tool']}: {_result_brief(event)}  [{event.get('elapsed_ms')} ms]")
        else:
            err = event.get("error") or {}
            print(f"    {ICONS[False]} {event['tool']}: {err.get('type')}: {_fmt(err.get('message'), 200)}")
            if err.get("hint"):
                print(f"      hint: {_fmt(err['hint'], 200)}")
        for w in event.get("warnings") or []:
            print(f"      ⚠ {_fmt(w, 200)}")
    elif kind == "guard":
        verdict = "passed" if event.get("passed") else f"REJECTED: {event.get('problems')}"
        print(f"  🛡 guard ({event.get('method')}): {verdict}")
    elif kind == "stop":
        print(f"  ⏹ {event['reason']}")
    elif kind == "fallback":
        print(f"  ↯ fallback email: {event['email'].get('status')} ({event.get('reason')})")
    elif kind == "error":
        print(f"  ✖ {event['message']}")
    elif kind == "run_finished":
        s = event["summary"]
        sel = s.get("selected") or {}
        print(f"\n■ {s['status'].upper()} in {s['elapsed_s']}s · {sel.get('symbol')} {sel.get('pct_change')}% on {s.get('trade_date')} "
              f"· email {s['email'].get('status')} · LLM calls {s.get('llm_calls')} · outputs/run_{s['run_id']}/\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Autonomous NASDAQ top-gainer analysis agent")
    p.add_argument("--days", type=int, help="trading days to analyse (default 5)")
    p.add_argument("--to", help="comma-separated recipients (overrides EMAIL_TO)")
    p.add_argument("--provider", choices=["mimo", "deepseek", "openai"], help="primary LLM provider")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--send", action="store_true", help="really send the email")
    mode.add_argument("--dry-run", action="store_true", help="only write the .eml (default)")
    p.add_argument("--no-jev", action="store_true", help="disable Jev sentiment; use the LLM assessment, with the same evidence-only report")
    p.add_argument("--sandbox", choices=["docker", "local"],
                   help="where generated code runs: docker (default, isolated). local is for development and "
                        "debugging only: no OS isolation, so it does not demonstrate the sandbox")
    p.add_argument("--resume", metavar="RUN_ID", help="resume an interrupted run from its checkpoint")
    p.add_argument("--force", action="store_true", help="compatibility flag; new runs can always send again, resume remains idempotent")
    p.add_argument("--quiet", action="store_true", help="print only the final summary")
    return p.parse_args(argv)


def overrides_from(args: argparse.Namespace) -> dict:
    o: dict = {"analysis": {}, "email": {}, "llm": {}, "jev": {}, "sandbox": {}}
    if args.days:
        o["analysis"]["days"] = args.days
    if args.to:
        o["email"]["to"] = [t.strip() for t in args.to.split(",") if t.strip()]
    if args.send:
        o["email"]["dry_run"] = False
    if args.dry_run:
        o["email"]["dry_run"] = True
    if args.provider:
        o["llm"]["provider"] = args.provider
    if args.no_jev:
        o["jev"]["enabled"] = False
    if args.sandbox:
        o["sandbox"]["backend"] = "trusted_local" if args.sandbox == "local" else "docker"
    return o


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)
    settings = load_settings(overrides=overrides_from(args))
    if settings.sandbox.backend == "trusted_local":
        print("WARNING: --sandbox local is a development/debugging option. Model-written code runs with your "
              "user permissions and no container isolation; the AST check is not a substitute for Docker.",
              file=sys.stderr)
    summary = run(settings, resume_id=args.resume, force_send=args.force,
                  on_event=None if args.quiet else print_event)
    if args.quiet:
        print(json.dumps({k: summary[k] for k in ("run_id", "status", "selected", "email")}, default=str, indent=2))
    return summary["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
