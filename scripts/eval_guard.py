"""Evaluate the pre-send guard on a finished run.

Original drafts require explicit adjudicated labels; mutated negatives are scored
separately. No positive is silently dropped for failing a structural check.

    uv run python scripts/eval_guard.py <run_id> [labels.json]
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402

from nasdaq_agent.agent import guard  # noqa: E402
from scripts.legacy import narrative_guard  # noqa: E402
from nasdaq_agent.agent.graph import build_graph  # noqa: E402
from nasdaq_agent.config import load_settings  # noqa: E402
from nasdaq_agent.context import start_run  # noqa: E402

CORRUPTIONS = {
    "wrong_trend": [(r"\{\{metrics\.trend\}\} trend", "steady downtrend")],
    "wrong_benchmark": [(r"(The S&P 500 returned|versus) \{\{metrics\.gspc_return\}\}", r"\1 a strong double-digit rally")],
    "invented_news": [(r"special (cash )?dividend( declaration| per share)?", "dilutive share offering")],
    "advice": [(r"$", " Investors should buy the stock before the rally continues.")],
}


def main(run_id: str, labels_path: str | None = None) -> int:
    settings = load_settings()
    start_run(settings, run_id)
    conn = sqlite3.connect(settings.output_root / "checkpoints.sqlite", check_same_thread=False)
    graph = build_graph(SqliteSaver(conn))
    state = next(s.values for s in graph.get_state_history({"configurable": {"thread_id": run_id}})
                 if s.values.get("analysis") and s.values.get("sentiment"))

    drafts = []
    for line in (settings.output_root / f"run_{run_id}" / "trace.jsonl").read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event["event"] == "llm":
            drafts += [c["args"] for c in event["tool_calls"] if c["name"] == "send_email"]
    cases = []
    labels = json.loads(Path(labels_path).read_text(encoding="utf-8")) if labels_path else {}
    for i, args in enumerate(drafts, 1):
        # A model draft is not ground truth. Supply explicit adjudicated labels.
        cases.append((f"draft_{i}", args, labels.get(f"draft_{i}")))
        for name, subs in CORRUPTIONS.items():
            text = args["narrative"]
            for pattern, repl in subs:
                text = re.sub(pattern, repl, text, count=1, flags=re.I)
            if text != args["narrative"]:
                cases.append((f"draft_{i}+{name}", {**args, "narrative": text}, False))

    correct = scored = 0
    for name, args, should_pass in cases:
        started = time.time()
        problems, details = (narrative_guard.check_legacy_narrative if "narrative" in args else guard.check)(state, args, settings)
        passed = not problems
        structural = problems and details.get("method") is None
        if should_pass is None:
            print(f"UNLABELLED {name:27} verdict={'pass' if passed else 'reject'}; excluded from accuracy, supply a labels JSON")
            continue
        scored += 1
        ok = passed == should_pass
        correct += ok
        tag = "ok " if ok else "MISS"
        print(f"{tag} {name:28} expect={'pass' if should_pass else 'reject':6} got={'pass' if passed else 'reject':6} "
              f"[{details.get('method') or 'structural'}] {time.time() - started:.1f}s")
        if problems and (not ok or not structural):
            for p in problems:
                print(f"       - {p[:150]}")
    print(f"\n{correct}/{scored} correct on explicitly labelled cases; same-run mutations do not measure generalization")
    return 0 if scored and correct == scored else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
