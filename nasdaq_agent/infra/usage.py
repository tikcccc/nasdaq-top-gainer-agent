"""Durable model request ledger and atomic per-run budget reservations.

No prompts or secrets are stored here. Unknown/unfinished requests retain their
reservation after a crash. Token/cost admission uses estimates, not a billing cap.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path

from nasdaq_agent.config import Settings


class BudgetExceeded(Exception):
    def __init__(self, dimension: str):
        self.dimension = dimension
        super().__init__(f"budget_exceeded: {dimension}")


def estimate_tokens(payload) -> int:
    """Cheap, deliberately padded heuristic; provider tokenization can differ."""
    def encode(value):
        return value.model_dump() if hasattr(value, "model_dump") else str(value)
    return max(1, math.ceil(len(json.dumps(payload, default=encode, ensure_ascii=False)) / 3) + 64)


def normalize_usage(response) -> dict | None:
    raw = response.get("raw") if isinstance(response, dict) and "raw" in response else response
    usage = getattr(raw, "usage_metadata", None) or getattr(raw, "usage", None)
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    if not isinstance(usage, dict):
        return None
    inp, out = usage.get("input_tokens"), usage.get("output_tokens")
    if not all(isinstance(n, int) and not isinstance(n, bool) and n >= 0 for n in (inp, out)):
        return None
    cached = (usage.get("input_token_details") or {}).get("cache_read", 0) or 0
    cached = min(inp, max(0, cached)) if isinstance(cached, int) else 0
    return {"input_tokens": inp, "output_tokens": out, "cached_input_tokens": cached}


def cost(price: dict | None, inp: int, out: int, cached: int = 0) -> float | None:
    if price is None:
        return None
    cached_rate = price.get("cached_input_per_million")
    if cached_rate is None:
        cached_rate = price["input_per_million"]
    return (price["per_request"] + ((inp - cached) * price["input_per_million"]
            + cached * cached_rate + out * price["output_per_million"]) / 1_000_000)


class UsageLedger:
    def __init__(self, root: Path):
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / "usage.sqlite"
        self._session_conn = None
        self._lock = threading.RLock()
        with self.connect() as conn:
            # Concurrent first runs can collide while enabling WAL, before the
            # usual transaction busy timeout takes effect. Retry only locking
            # errors, with a bounded startup window for separate workers too.
            wal_deadline = time.monotonic() + 5
            while True:
                try:
                    conn.execute("PRAGMA journal_mode=WAL")
                    break
                except sqlite3.OperationalError as exc:
                    if "locked" not in str(exc).lower() or time.monotonic() >= wal_deadline:
                        raise
                    time.sleep(0.025)
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS usage_runs (
                    run_id TEXT PRIMARY KEY, started REAL NOT NULL, finished REAL,
                    status TEXT, agent_completed INTEGER, task_completed INTEGER,
                    elapsed_s REAL, guard_rejections INTEGER, stop_reason TEXT,
                    legacy_untracked INTEGER NOT NULL DEFAULT 0);
                CREATE INDEX IF NOT EXISTS usage_runs_started ON usage_runs(started);
                CREATE TABLE IF NOT EXISTS model_requests (
                    call_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, provider TEXT NOT NULL,
                    model TEXT NOT NULL, stage TEXT NOT NULL, attempt INTEGER NOT NULL,
                    fallback INTEGER NOT NULL, started REAL NOT NULL, latency_ms REAL,
                    outcome TEXT NOT NULL, error_type TEXT, http_status INTEGER, input_tokens INTEGER,
                    output_tokens INTEGER, cached_input_tokens INTEGER,
                    reserved_tokens INTEGER NOT NULL, reserved_cost REAL,
                    estimated_cost_usd REAL, pricing TEXT, usage_known INTEGER NOT NULL DEFAULT 0);
                CREATE INDEX IF NOT EXISTS model_requests_run ON model_requests(run_id, started);
            """)
            conn.execute("BEGIN IMMEDIATE")
            if "http_status" not in {r[1] for r in conn.execute("PRAGMA table_info(model_requests)")}:
                conn.execute("ALTER TABLE model_requests ADD COLUMN http_status INTEGER")
            if "legacy_untracked" not in {r[1] for r in conn.execute("PRAGMA table_info(usage_runs)")}:
                conn.execute("ALTER TABLE usage_runs ADD COLUMN legacy_untracked INTEGER NOT NULL DEFAULT 0")

    @contextmanager
    def connect(self):
        with self._lock:
            if self._session_conn is not None:
                with self._session_conn:
                    yield self._session_conn
                return
        # Short-lived readers outside a run require no explicit cleanup by callers.
        conn = sqlite3.connect(self.path, timeout=5)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    @contextmanager
    def session(self):
        """Reuse one connection per running context, serialized across tool threads.

        Keep FULL durability; avoiding WAL teardown per request is faster without
        relaxing fsync. SQLite still arbitrates transactions across other workers.
        """
        with self._lock:
            if self._session_conn is not None:
                raise RuntimeError("ledger session is already open")
            conn = sqlite3.connect(self.path, timeout=5, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._session_conn = conn
        try:
            yield self
        finally:
            with self._lock:
                self._session_conn = None
                conn.close()

    def start_run(self, run_id: str, started: float, *, legacy_untracked=False) -> float:
        with self.connect() as conn:
            conn.execute("INSERT OR IGNORE INTO usage_runs(run_id,started,legacy_untracked) VALUES (?,?,?)",
                         (run_id, started, int(legacy_untracked)))
            return conn.execute("SELECT started FROM usage_runs WHERE run_id=?", (run_id,)).fetchone()[0]

    def reserve(self, run_id, settings: Settings, *, provider, model, stage, attempt,
                fallback=False, input_estimate, output_estimate) -> str:
        price_obj = settings.budget.prices.get(f"{provider}:{model}")
        price = price_obj.model_dump() if price_obj else None
        # Admission assumes no cache discount, including unusual cache price tables.
        reserved_cost = cost(price, input_estimate, output_estimate)
        if price and price.get("cached_input_per_million") is not None:
            reserved_cost = max(reserved_cost, cost(price, input_estimate, output_estimate, input_estimate))
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            run = conn.execute("SELECT started,legacy_untracked FROM usage_runs WHERE run_id=?", (run_id,)).fetchone()
            if time.time() >= run["started"] + settings.run.deadline_s:
                raise BudgetExceeded("deadline")
            budget = settings.budget
            if run["legacy_untracked"] and any(value is not None for value in (
                budget.max_model_requests, budget.max_total_tokens, budget.max_estimated_cost_usd)):
                raise BudgetExceeded("untracked_legacy_usage")
            totals = conn.execute("""SELECT count(*) AS requests,
                coalesce(sum(CASE WHEN usage_known=1 THEN input_tokens+output_tokens ELSE reserved_tokens END),0) AS tokens,
                coalesce(sum(coalesce(estimated_cost_usd,reserved_cost)),0) AS cost,
                sum(CASE WHEN reserved_cost IS NULL THEN 1 ELSE 0 END) AS unpriced
                FROM model_requests WHERE run_id=?""", (run_id,)).fetchone()
            if budget.max_model_requests is not None and totals["requests"] >= budget.max_model_requests:
                raise BudgetExceeded("model_requests")
            if budget.max_total_tokens is not None and totals["tokens"] + input_estimate + output_estimate > budget.max_total_tokens:
                raise BudgetExceeded("total_tokens")
            if budget.max_estimated_cost_usd is not None:
                if reserved_cost is None or totals["unpriced"]:
                    raise BudgetExceeded("unpriced_model")
                if totals["cost"] + reserved_cost > budget.max_estimated_cost_usd:
                    raise BudgetExceeded("estimated_cost_usd")
            call_id = uuid.uuid4().hex
            conn.execute("""INSERT INTO model_requests
                (call_id,run_id,provider,model,stage,attempt,fallback,started,outcome,reserved_tokens,reserved_cost,pricing)
                VALUES (?,?,?,?,?,?,?,?,'pending',?,?,?)""",
                (call_id, run_id, provider, model, stage, attempt, int(fallback), time.time(),
                 input_estimate + output_estimate, reserved_cost, json.dumps(price) if price else None))
            return call_id

    def finish(self, call_id, *, latency_ms, response=None, error=None):
        usage = normalize_usage(response)
        with self.connect() as conn:
            row = conn.execute("SELECT pricing FROM model_requests WHERE call_id=?", (call_id,)).fetchone()
            price = json.loads(row[0]) if row[0] else None
            estimated_cost = cost(price, **{
                "inp": usage["input_tokens"], "out": usage["output_tokens"], "cached": usage["cached_input_tokens"]
            }) if usage else None
            conn.execute("""UPDATE model_requests SET latency_ms=?,outcome=?,error_type=?,http_status=?,input_tokens=?,
                output_tokens=?,cached_input_tokens=?,estimated_cost_usd=?,usage_known=?
                WHERE call_id=? AND outcome='pending'""",
                (latency_ms, "error" if error else "success", type(error).__name__ if error else None,
                 getattr(error, "status_code", None) or getattr(error, "status", None),
                 usage["input_tokens"] if usage else None, usage["output_tokens"] if usage else None,
                 usage["cached_input_tokens"] if usage else None, estimated_cost, int(usage is not None), call_id))

    def finish_run(self, summary):
        with self.connect() as conn:
            conn.execute("""UPDATE usage_runs SET finished=?,status=?,agent_completed=?,task_completed=?,
                elapsed_s=?,guard_rejections=?,stop_reason=? WHERE run_id=?""",
                (time.time(), summary["status"], int(summary["agent_completed"]), int(summary["task_completed"]),
                 summary.get("total_elapsed_s", summary["elapsed_s"]), summary.get("guard_rejections") or 0,
                 summary.get("stop_reason"), summary["run_id"]))

    def last_request(self, provider: str) -> dict | None:
        """The most recent finished request to a provider, for status displays."""
        with self.connect() as conn:
            row = conn.execute("""SELECT outcome, error_type, http_status, started FROM model_requests
                WHERE provider=? AND outcome != 'pending' ORDER BY started DESC LIMIT 1""", (provider,)).fetchone()
        return dict(row) if row else None

    def records(self, run_id, limit=100, offset=0):
        with self.connect() as conn:
            return [dict(r) for r in conn.execute("SELECT * FROM model_requests WHERE run_id=? ORDER BY started,call_id LIMIT ? OFFSET ?",
                                                  (run_id, limit, offset))]

    def summary(self, run_id):
        with self.connect() as conn:
            run = conn.execute("SELECT legacy_untracked FROM usage_runs WHERE run_id=?", (run_id,)).fetchone()
            rows = conn.execute("""SELECT provider,model,stage,count(*) AS requests,
                sum(usage_known) AS usage_known_requests,
                sum(outcome='error') AS errors,sum(outcome='pending') AS pending,
                sum(attempt>1) AS retries,sum(fallback) AS fallback_requests,
                coalesce(sum(input_tokens),0) AS input_tokens,coalesce(sum(output_tokens),0) AS output_tokens,
                coalesce(sum(estimated_cost_usd),0) AS known_estimated_cost_usd,
                sum(estimated_cost_usd IS NULL) AS unknown_cost_requests,
                coalesce(sum(CASE WHEN usage_known=0 THEN reserved_tokens ELSE 0 END),0) AS reserved_unknown_tokens,
                coalesce(sum(CASE WHEN estimated_cost_usd IS NULL THEN reserved_cost ELSE 0 END),0) AS reserved_unknown_cost_usd
                FROM model_requests WHERE run_id=? GROUP BY provider,model,stage""", (run_id,)).fetchall()
        groups = [dict(r) for r in rows]
        keys = [k for k in groups[0] if k not in {"provider", "model", "stage"}] if groups else [
            "requests", "usage_known_requests", "errors", "pending", "retries", "fallback_requests",
            "input_tokens", "output_tokens", "known_estimated_cost_usd", "unknown_cost_requests",
            "reserved_unknown_tokens", "reserved_unknown_cost_usd"]
        result = {k: sum(g[k] for g in groups) for k in keys}
        result["usage_coverage"] = result["usage_known_requests"] / result["requests"] if result["requests"] else None
        result["total_tokens"] = result["input_tokens"] + result["output_tokens"]
        result["legacy_untracked"] = bool(run and run[0])
        result["estimated_cost_usd"] = result["known_estimated_cost_usd"] if not result["unknown_cost_requests"] and not result["legacy_untracked"] else None
        result["by_model_stage"] = groups
        return result
