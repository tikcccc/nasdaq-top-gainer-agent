"""On-demand monitoring over a bounded set of runs; no background polling."""

import math
from collections import Counter, defaultdict


def percentiles(values):
    values = sorted(v for v in values if v is not None)
    return {key: values[max(0, math.ceil(q * len(values)) - 1)] if values else None
            for key, q in (("p50", .5), ("p95", .95), ("max", 1))}


def aggregate_usage(summaries):
    """Also used by evaluation reports; missing ledger data is never zero cost."""
    known = [s for s in summaries if s is not None]
    requests = sum(s["requests"] for s in known)
    known_requests = sum(s["usage_known_requests"] for s in known)
    complete = len(known) == len(summaries) and all(s["estimated_cost_usd"] is not None for s in known)
    known_cost = sum(s["known_estimated_cost_usd"] for s in known)
    return {"requests": requests, "input_tokens": sum(s["input_tokens"] for s in known),
            "output_tokens": sum(s["output_tokens"] for s in known),
            "usage_known_requests": known_requests, "usage_coverage": known_requests / requests if requests else None,
            "missing_run_ledgers": len(summaries) - len(known),
            "known_estimated_cost_usd": known_cost,
            "estimated_cost_usd": known_cost if complete else None}


def snapshot(ledger, *, limit=100):
    with ledger.connect() as conn:
        runs = [dict(r) for r in conn.execute("SELECT * FROM usage_runs ORDER BY started DESC LIMIT ?", (limit,))]
        calls = [dict(r) for r in conn.execute("""SELECT * FROM model_requests WHERE run_id IN
            (SELECT run_id FROM usage_runs ORDER BY started DESC LIMIT ?)""", (limit,))]
    finished = [r for r in runs if r["finished"] is not None]
    states = Counter(r["status"] for r in finished)
    by_run = defaultdict(list)
    groups = defaultdict(list)
    for call in calls:
        by_run[call["run_id"]].append(call)
        groups[(call["provider"], call["model"], call["stage"])].append(call)
    costs = []
    for r in finished:
        rows = by_run[r["run_id"]]
        costs.append(sum(c["estimated_cost_usd"] for c in rows) if not r["legacy_untracked"] and all(c["estimated_cost_usd"] is not None for c in rows) else None)
    known_cost = sum(c["estimated_cost_usd"] or 0 for c in calls)
    all_cost_known = not any(r["legacy_untracked"] for r in runs) and all(c["estimated_cost_usd"] is not None for c in calls)
    qualified = sum(r["agent_completed"] for r in finished)
    # Only finished runs contribute to cost-per-qualified-report.
    finished_cost = sum(v for v in costs if v is not None)
    complete_finished_cost = all(v is not None for v in costs)
    return {
        "scope": "latest ledger runs in this output directory; legacy runs appear only after adoption",
        "legacy_untracked_runs": sum(r["legacy_untracked"] for r in runs),
        "limit": limit, "runs": len(runs), "finished_runs": len(finished), "unfinished_runs": len(runs) - len(finished),
        "outcomes": dict(states),
        "normal_completion_rate": states["ok"] / len(finished) if finished else None,
        "degraded_rate": states["degraded"] / len(finished) if finished else None,
        "failed_rate": states["failed"] / len(finished) if finished else None,
        "agent_completed": qualified, "task_completed": sum(r["task_completed"] for r in finished),
        "latency_s": percentiles(r["elapsed_s"] for r in finished),
        "guard_rejections": sum(r["guard_rejections"] or 0 for r in finished),
        "budget_stops": sum((r["stop_reason"] or "").startswith("budget_exceeded:") for r in finished),
        "requests": len(calls),
        "usage_coverage": sum(c["usage_known"] for c in calls) / len(calls) if calls else None,
        "known_estimated_cost_usd": known_cost,
        "estimated_cost_usd": known_cost if all_cost_known else None,
        "cost_per_qualified_report_usd": finished_cost / qualified if qualified and complete_finished_cost else None,
        "known_run_cost_usd": percentiles(v for v in costs if v is not None),
        "unknown_cost_runs": sum(v is None for v in costs),
        "models": [{"provider": p, "model": m, "stage": s, "requests": len(rows),
                    "errors": sum(c["outcome"] == "error" for c in rows),
                    "timeouts": sum("Timeout" in (c["error_type"] or "") for c in rows),
                    "rate_limited": sum(c["http_status"] == 429 for c in rows),
                    "retries": sum(c["attempt"] > 1 for c in rows),
                    "fallback_requests": sum(c["fallback"] for c in rows),
                    "latency_ms": percentiles(c["latency_ms"] for c in rows)}
                   for (p, m, s), rows in sorted(groups.items())],
    }
