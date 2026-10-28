"""Per-run context shared by tools, providers and the graph.

Context-local: LangChain propagates context to sync tool threads. Explicit
ThreadPoolExecutor calls use copy_context() so concurrent runs stay isolated.
"""

from __future__ import annotations

import secrets
import re
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from nasdaq_agent.config import Settings


@dataclass
class RunContext:
    settings: Settings
    run_id: str
    run_dir: Path
    started_at: float = field(default_factory=time.time)
    force_send: bool = False        # legacy CLI compatibility; new runs can always send again
    model: object = None
    prompt: str | None = None
    ledger: object = None
    deadline_monotonic: float | None = None

    @property
    def data_dir(self) -> Path:
        return self.run_dir / "data"

    @property
    def sandbox_dir(self) -> Path:
        return self.run_dir / "sandbox"

    @property
    def deadline_ts(self) -> float:
        return self.started_at + self.settings.run.deadline_s

    def seconds_left(self) -> float:
        remaining = self.deadline_ts - time.time()
        return min(remaining, self.deadline_monotonic - time.monotonic()) if self.deadline_monotonic is not None else remaining


_current: ContextVar[RunContext | None] = ContextVar("nasdaq_run", default=None)


def new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_") + secrets.token_hex(4)


def start_run(settings: Settings, run_id: str | None = None) -> RunContext:
    """Create (or re-open, when resuming) the run directory and make it current."""
    run_id = run_id or new_run_id()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", run_id):
        raise ValueError("invalid run id")
    run_dir = settings.output_root / f"run_{run_id}"
    for sub in ("", "data", "sandbox"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    ctx = RunContext(settings=settings, run_id=run_id, run_dir=run_dir)
    from nasdaq_agent.infra.usage import UsageLedger
    ctx.ledger = UsageLedger(settings.output_root)
    # Keep the original deadline when resuming runs created before the ledger existed.
    prior_trace = run_dir / "trace.jsonl"
    if prior_trace.exists():
        import json
        with prior_trace.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    event = json.loads(line)
                    if event.get("event") == "run_started":
                        ctx.started_at = float(event["ts"])
                        break
                except (ValueError, KeyError, TypeError):
                    continue
    ctx.started_at = ctx.ledger.start_run(run_id, ctx.started_at, legacy_untracked=prior_trace.exists())
    ctx.deadline_monotonic = time.monotonic() + ctx.deadline_ts - time.time()
    _current.set(ctx)
    return ctx


def get_ctx() -> RunContext:
    ctx = _current.get()
    if ctx is None:
        raise RuntimeError("No active run. Call nasdaq_agent.context.start_run() first.")
    return ctx
