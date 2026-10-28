"""Calibrate the guard's unsupported-causal-claim check on labelled sentences.

    uv run python scripts/eval_speculation.py <run_id> [threshold]

Scores every sentence in scripts/speculation_cases.json with Jev against the
evidence of the given run, prints the score distribution per label and the
accuracy at the threshold (default: config jev.speculation_threshold).
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402

from scripts.legacy import narrative_guard as guard  # noqa: E402
from nasdaq_agent.agent.graph import build_graph  # noqa: E402
from nasdaq_agent.config import load_settings  # noqa: E402
from nasdaq_agent.context import start_run  # noqa: E402
from nasdaq_agent.reporting import placeholders  # noqa: E402

CASES = Path(__file__).with_name("speculation_cases.json")


def main(run_id: str, threshold: str | None = None) -> int:
    settings = load_settings()
    start_run(settings, run_id)
    conn = sqlite3.connect(settings.output_root / "checkpoints.sqlite", check_same_thread=False)
    state = build_graph(SqliteSaver(conn)).get_state({"configurable": {"thread_id": run_id}}).values
    limit = float(threshold) if threshold else settings.jev.speculation_threshold

    cases = json.loads(CASES.read_text(encoding="utf-8"))["cases"]
    facts = placeholders.flatten(placeholders.build_context(state))
    evidence = guard.evidence(state, facts, {"n1", "n2"})
    scores = guard.speculation_scores(settings, evidence, [c["text"] for c in cases])

    correct = scored = 0
    for case, p in sorted(zip(cases, scores), key=lambda x: -x[1]):
        predicted = "reject" if p > limit else "pass"
        if case["label"] == "out_of_scope":
            print(f"n/a  p={p:.2f} out of scope [{case['origin']}] {case['text'][:70]} ({case.get('note', '')[:60]})")
            continue
        scored += 1
        ok = predicted == case["label"]
        correct += ok
        print(f"{'ok ' if ok else 'MISS'} p={p:.2f} expect={case['label']:6} [{case['origin']}] {case['text'][:80]}")
    rejects = [p for c, p in zip(cases, scores) if c["label"] == "reject"]
    passes = [p for c, p in zip(cases, scores) if c["label"] == "pass"]
    print(f"\nreject scores {min(rejects):.2f}-{max(rejects):.2f} | pass scores {min(passes):.2f}-{max(passes):.2f}")
    print(f"{correct}/{scored} correct at threshold {limit}")
    return 0 if correct == scored else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
