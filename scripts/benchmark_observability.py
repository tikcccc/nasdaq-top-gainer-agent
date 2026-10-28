"""Offline A/B benchmark of model accounting overhead; never calls a real API.

    uv run python scripts/benchmark_observability.py --calls 50 --delay-ms 20
"""

import argparse
import hashlib
import json
import platform
import sqlite3
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
from langchain_core.messages import HumanMessage

from nasdaq_agent.config import Settings
from nasdaq_agent.context import start_run
from nasdaq_agent.infra.monitoring import percentiles
from nasdaq_agent.llm import factory


def benchmark(calls=50, delay_ms=20):
    def handler(req):
        time.sleep(delay_ms / 1000)
        return httpx.Response(200, json={
            "id": "bench", "object": "chat.completion", "created": 1, "model": "fixture",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}})

    with tempfile.TemporaryDirectory(prefix="nasdaq_observability_") as folder, httpx.Client(transport=httpx.MockTransport(handler)) as client:
        settings = Settings.model_validate({
            "run": {"output_dir": folder, "deadline_s": max(420, calls * delay_ms / 1000 * 4)},
            "llm": {"provider": "fixture", "fallback": [], "max_retries": 0},
            "providers": {"fixture": {"name": "fixture", "api_key": "offline", "model": "fixture", "base_url": "https://fixture.invalid/v1"}}})
        started = time.perf_counter()
        ctx = start_run(settings, "benchmark")
        initialization_ms = (time.perf_counter() - started) * 1000
        original = factory.ChatOpenAI
        with patch.object(factory, "ChatOpenAI", lambda **kwargs: original(**kwargs, http_client=client)):
            baseline = factory._model(settings, "fixture")
            instrumented = factory.build_chat(settings)
        messages = [HumanMessage("Fixed benchmark request")]
        timings = {"baseline": [], "instrumented": []}
        with ctx.ledger.session():
            with ctx.ledger.connect() as conn:
                durability = {"journal_mode": conn.execute("PRAGMA journal_mode").fetchone()[0],
                              "synchronous": conn.execute("PRAGMA synchronous").fetchone()[0]}
            baseline.invoke(messages)
            instrumented.invoke(messages)
            # Alternate order to reduce drift; constructors and warm-up are excluded.
            for i in range(calls):
                order = [("baseline", baseline), ("instrumented", instrumented)]
                for name, model in order if i % 2 == 0 else reversed(order):
                    started = time.perf_counter()
                    model.invoke(messages)
                    timings[name].append((time.perf_counter() - started) * 1000)
        averages = {key: statistics.mean(vals) for key, vals in timings.items()}
        return {"scope": "offline SDK requests with fixed responses; not live-model or full-graph latency",
                "measured_at_utc": datetime.now(timezone.utc).isoformat(),
                "python_version": platform.python_version(), "sqlite_version": sqlite3.sqlite_version,
                "platform": sys.platform, "sqlite_durability": durability,
                "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "calls_per_variant": calls, "simulated_request_delay_ms": delay_ms,
                "one_time_ledger_initialization_ms": initialization_ms,
                "latency_ms": {name: {**percentiles(vals), "mean": averages[name]} for name, vals in timings.items()},
                "mean_overhead_ms_per_request": averages["instrumented"] - averages["baseline"],
                "mean_overhead_percent": (averages["instrumented"] / averages["baseline"] - 1) * 100,
                "accounted_requests_including_warmup": ctx.ledger.summary(ctx.run_id)["requests"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls", type=int, default=50)
    parser.add_argument("--delay-ms", type=float, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.calls < 1 or args.delay_ms < 0:
        parser.error("calls must be positive and delay-ms nonnegative")
    result = benchmark(args.calls, args.delay_ms)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)
