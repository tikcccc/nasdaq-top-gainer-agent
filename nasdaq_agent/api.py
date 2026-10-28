"""HTTP API for the web UI. Same runner as the CLI; events stream over Server-Sent Events.

    uv run --extra ui uvicorn nasdaq_agent.api:app --port 8000

Endpoints
    GET  /api/config                    defaults and whether a run is in progress
    POST /api/runs                      start a run  {days?, provider?, send?, force?, to?}
    GET  /api/runs                      past runs (summaries)
    GET  /api/runs/{id}                 summary + trace of one run
    GET  /api/runs/{id}/series          cleaned OHLCV of the stock and benchmarks (for charts)
    POST /api/runs/{id}/send            send this run's report to {to: [...]}
    GET  /api/runs/{id}/events          live events (SSE); replays from the start
    GET  /api/runs/{id}/files/{name}    chart.png, email.html, code_v1.py ...
"""

from __future__ import annotations

import asyncio
import csv
import email
import json
import re
import sqlite3
import threading
import time
import hmac
from contextlib import asynccontextmanager
from email.policy import default as default_policy
from email.utils import formatdate, make_msgid
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from nasdaq_agent.agent.runner import run
from nasdaq_agent.config import load_settings
from nasdaq_agent.context import new_run_id
from nasdaq_agent.infra import mailer, delivery
from nasdaq_agent.infra.jobs import JobStore, JobWorker, QueueFull
from nasdaq_agent.infra.usage import UsageLedger
from nasdaq_agent.llm import jev
from nasdaq_agent.reporting.version import renderer_identity

@asynccontextmanager
async def lifespan(app):
    _ensure_worker()
    yield
    for worker in list(_workers.values()):
        worker.stop.set()
        worker.thread.join(timeout=2)


app = FastAPI(title="NASDAQ Top-Gainer Agent", version="0.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

STALE_AFTER_S = 600  # an unfinished run whose trace is older than this is treated as abandoned
RUN_ID = re.compile(r"^[\w-]{6,40}$")
EMAIL = re.compile(r"^[^@\s,;<>]+@[^@\s,;<>]+\.[A-Za-z]{2,}$")
SERVABLE = re.compile(r"^(chart[\w-]*\.(png|gif)|email(_preview)?\.(html|eml)|code_v\d+\.py|summary\.json)$")


_workers: dict[str, JobWorker] = {}
_worker_lock = threading.Lock()


class RunRequest(BaseModel):
    days: int | None = Field(default=None, ge=2, le=30)
    provider: Literal["mimo", "deepseek", "openai"] | None = None
    send: bool | None = None  # omitted: inherit configured mode; explicit false: preview
    force: bool = False
    to: list[str] | None = Field(default=None, max_length=5, description="recipients; implies send unless send=false")
    exclude_penny: bool = True
    exclude_low_volume: bool = True


class SendRequest(BaseModel):
    to: list[str] = Field(min_length=1, max_length=5)
    request_id: UUID = Field(default_factory=uuid4, description="Reuse only when retrying the same send action")


def _recipients(to: list[str]) -> list[str]:
    cleaned = list({t.strip().lower(): t.strip() for t in to if t.strip()}.values())
    bad = [t for t in cleaned if not EMAIL.match(t)]
    if bad or not cleaned:
        raise HTTPException(422, f"invalid email address: {', '.join(bad) or '(empty)'}")
    return cleaned


def _settings(req: RunRequest | None = None):
    overrides: dict = {}
    if req:
        email_cfg: dict = {}
        if req.send is not None:
            email_cfg["dry_run"] = not req.send
        elif req.to:
            email_cfg["dry_run"] = False
        if req.to:
            email_cfg["to"] = _recipients(req.to)
        overrides = {
            "analysis": {"days": req.days} if req.days else {},
            "llm": {"provider": req.provider} if req.provider else {},
            "email": email_cfg,
            "screener": {"default_filters": {"min_price": 1 if req.exclude_penny else 0,
                                               "min_volume": 100000 if req.exclude_low_volume else 0}},
        }
    settings = load_settings(overrides=overrides)
    if req and not settings.email.dry_run:
        settings.email.to = _recipients(settings.email.to)
        if not (settings.email.smtp.user and settings.email.smtp.password):
            raise HTTPException(422, "SMTP credentials are required for live email delivery")
    return settings


def _store():
    return JobStore(_settings().output_root)


def _execute_job(job):
    effective = job["request"].get("effective")
    settings = load_settings(overrides=effective) if effective else _settings(RunRequest.model_validate(job["request"]))
    run_id = job["run_id"]
    if (settings.output_root / f"run_{run_id}" / "summary.json").exists():
        return  # worker died after writing its result but before acknowledging the queue
    resume = False
    checkpoint = settings.output_root / "checkpoints.sqlite"
    if job["resumed"] and checkpoint.exists():
        import sqlite3
        from langgraph.checkpoint.sqlite import SqliteSaver
        conn = sqlite3.connect(checkpoint)
        try:
            resume = SqliteSaver(conn).get_tuple({"configurable": {"thread_id": run_id}}) is not None
        finally:
            conn.close()
    run(settings, resume_id=run_id if resume else None, run_id=run_id, force_send=job["request"].get("force", False))


def _ensure_worker():
    store = _store()
    key = str(store.path)
    with _worker_lock:
        worker = _workers.get(key)
        if worker is None or not worker.thread.is_alive():
            worker = JobWorker(store, _execute_job)
            worker.thread.start()
            _workers[key] = worker


@app.middleware("http")
async def access_control(request: Request, call_next):
    token = _settings().api_token
    if request.method != "OPTIONS" and token:
        supplied = request.headers.get("Authorization", "").removeprefix("Bearer ")
        if not hmac.compare_digest(supplied, token.get_secret_value()):
            return JSONResponse({"detail": "authentication required"}, status_code=401)
    if request.method == "POST":
        try:
            _store().rate_limit(request.client.host if request.client else "unknown")
        except QueueFull as exc:
            return JSONResponse({"detail": str(exc)}, status_code=429, headers={"Retry-After": "60"})
    return await call_next(request)


def _active(run_id: str, path: Path) -> bool:
    """Running here, or (e.g. started from the CLI) no summary yet and the trace was written recently."""
    job = _store().get(run_id)
    if job:
        return job["status"] in {"queued", "running"}
    if (path / "summary.json").exists():
        return False
    trace = path / "trace.jsonl"
    return trace.exists() and time.time() - trace.stat().st_mtime < STALE_AFTER_S


def _active_run() -> str | None:
    active = _store().active()
    if active:
        return active[0]["run_id"]
    for path in sorted(_settings().output_root.glob("run_*"), reverse=True)[:5]:
        if _active(path.name[4:], path):
            return path.name[4:]
    return None


def _run_dir(run_id: str) -> Path:
    if not RUN_ID.match(run_id):
        raise HTTPException(400, "bad run id")
    path = _settings().output_root / f"run_{run_id}"
    if not path.is_dir():
        raise HTTPException(404, "run not found")
    return path


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
        if not line.endswith("\n"):
            continue
        try:
            records.append(json.loads(line))
        except ValueError:
            continue  # an interrupted append must not break report polling
    return records


def _last_jev_request(settings) -> dict | None:
    try:
        return UsageLedger(settings.output_root).last_request("jev")
    except sqlite3.Error:
        return None


@app.get("/api/config")
def config() -> dict:
    s = _settings()
    return {
        "days": s.analysis.days,
        "report_renderer": renderer_identity(),
        "email_dry_run_default": s.email.dry_run,
        "email_enabled": bool(s.email.smtp.user and s.email.smtp.password),
        "email_default_to": s.email.to,
        "busy": bool(_store().active()),
        "filters": {"exclude_penny": s.screener.default_filters.min_price > 0,
                    "exclude_low_volume": s.screener.default_filters.min_volume > 0},
        "active_run": _active_run(),
        "jev": jev.status(s, _last_jev_request(s)),
    }


@app.post("/api/runs", status_code=202)
def start(req: RunRequest) -> dict:
    settings = _settings(req)  # validate before reserving the run slot
    run_id = new_run_id()
    store = JobStore(settings.output_root)
    try:
        effective = {"analysis": settings.analysis.model_dump(), "screener": settings.screener.model_dump(),
                     "sandbox": settings.sandbox.model_dump(), "data_checks": settings.data_checks.model_dump(),
                     "jev": settings.jev.model_dump(exclude={"api_key"}),
                     "llm": settings.llm.model_dump(), "budget": settings.budget.model_dump(),
                     "run": {**settings.run.model_dump(mode="json"), "output_dir": str(settings.output_root)},
                     "email": {"to": settings.email.to, "dry_run": settings.email.dry_run, "sender": settings.email.sender}}
        store.enqueue(run_id, {**req.model_dump(), "effective": effective})
    except QueueFull as exc:
        raise HTTPException(429, str(exc)) from exc
    (settings.output_root / f"run_{run_id}").mkdir(exist_ok=True)
    # The persisted queued job remains recoverable if worker startup fails.
    try:
        _ensure_worker()
    except Exception:
        import logging
        logging.getLogger(__name__).exception("worker startup failed; job remains queued")
    return {"run_id": run_id, "dry_run": settings.email.dry_run}


@app.get("/api/runs")
def list_runs(limit: int = 20) -> list[dict]:
    limit = max(1, min(limit, 100))
    root = _settings().output_root
    out = []
    for path in sorted(root.glob("run_*"), reverse=True):
        run_id = path.name[4:]
        summary_file = path / "summary.json"
        if summary_file.exists():
            s = json.loads(summary_file.read_text(encoding="utf-8"))
            sel = s.get("selected") or {}
            out.append({"run_id": run_id, "status": s.get("status"), "symbol": sel.get("symbol"),
                        "pct_change": sel.get("pct_change"), "trade_date": s.get("trade_date"),
                        "email": (s.get("email") or {}).get("status"), "elapsed_s": s.get("elapsed_s")})
        elif _active(run_id, path):
            out.append({"run_id": run_id, "status": "running"})
        if len(out) >= limit:
            break
    return out


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict:
    path = _run_dir(run_id)
    summary = path / "summary.json"
    trace = path / "trace.jsonl"
    job = _store().get(run_id)
    return {
        "run_id": run_id,
        "running": _active(run_id, path),
        "summary": json.loads(summary.read_text(encoding="utf-8")) if summary.exists() else None,
        "trace": _read_jsonl(trace),
        "files": sorted(p.name for p in path.iterdir() if SERVABLE.match(p.name)),
        "job_status": job["status"] if job else None,
        "error": job["error"] if job else None,
    }


@app.get("/api/runs/{run_id}/usage")
def get_usage(run_id: str, limit: int = 100, offset: int = 0) -> dict:
    _run_dir(run_id)
    ledger = UsageLedger(_settings().output_root)
    return {"run_id": run_id, "summary": ledger.summary(run_id),
            "requests": ledger.records(run_id, limit=max(1, min(limit, 500)), offset=max(0, offset))}


@app.get("/api/metrics")
def get_metrics(limit: int = 100) -> dict:
    from nasdaq_agent.infra.monitoring import snapshot
    settings = _settings()
    result = snapshot(UsageLedger(settings.output_root), limit=max(1, min(limit, 1000)))
    active = JobStore(settings.output_root).active()
    queued = [j for j in active if j["status"] == "queued"]
    result["queue"] = {"queued": len(queued), "running": len(active) - len(queued),
                       "oldest_wait_s": max((time.time() - j["created"] for j in queued), default=0)}
    return result


@app.get("/api/runs/{run_id}/series")
def get_series(run_id: str) -> dict:
    """Cleaned daily OHLCV the agent analysed: {"AIOK": [...], "GSPC": [...], "IXIC": [...]}."""
    out: dict[str, list[dict]] = {}
    for path in sorted((_run_dir(run_id) / "data").glob("ds_*.csv")):
        with path.open(encoding="utf-8", newline="") as fh:
            out[path.stem[3:]] = [
                {"date": r["date"], **{k: float(r[k]) for k in ("open", "high", "low", "close", "volume") if r.get(k)}}
                for r in csv.DictReader(fh)
            ]
    return out


@app.post("/api/runs/{run_id}/send")
def send_report(run_id: str, req: SendRequest) -> dict:
    """Send a saved report; each new request_id allows an intentional repeat."""
    recipients = _recipients(req.to)
    eml = _run_dir(run_id) / "email.eml"
    if not eml.exists():
        raise HTTPException(404, "this run has no report to send")
    s = _settings()
    msg = email.message_from_bytes(eml.read_bytes(), policy=default_policy)
    for header in ("To", "Cc", "Bcc", "Date", "Message-ID"):
        del msg[header]
    msg["To"] = ", ".join(recipients)
    msg["Message-ID"] = make_msgid(domain="nasdaq-agent.local")
    msg["Date"] = formatdate(localtime=False)
    summary_file = eml.parent / "summary.json"
    summary = json.loads(summary_file.read_text(encoding="utf-8")) if summary_file.exists() else {}
    selected = summary.get("selected") or {}
    key = delivery.operation_key(f"{run_id}:{req.request_id}", recipients, source="manual")
    cfg = s.email.model_copy(update={"dry_run": False})
    outcome = delivery.deliver_message(msg, cfg, s.output_root, key, {"run_id": run_id, "request_id": str(req.request_id), "recipients": recipients,
                                       "trade_date": summary.get("trade_date", "unknown"), "symbol": selected.get("symbol", "NONE")})
    if outcome["status"] not in {"sent", "skipped_duplicate"}:
        raise HTTPException(409 if outcome["status"] == "delivery_unknown" else 502, outcome)
    return {**outcome, "request_id": str(req.request_id), "recipients": recipients, "subject": msg["Subject"],
            "report_renderer": {'version': msg.get('X-Report-Renderer', 'archived-unknown'),
                                'fingerprint': msg.get('X-Report-Fingerprint')}, 'archived': True}


@app.get("/api/runs/{run_id}/events")
async def events(run_id: str, last_event_id: str | None = Header(default=None)) -> StreamingResponse:
    path = _run_dir(run_id)
    try:
        cursor = max(0, int(last_event_id or 0))
    except ValueError:
        raise HTTPException(400, "invalid Last-Event-ID")

    async def stream():
        sent = cursor
        while True:
            event_file = path / "events.jsonl"
            lines = event_file.read_text(encoding="utf-8").splitlines(keepends=True) if event_file.exists() else []
            complete = [line for line in lines if line.endswith("\n")]
            for i, line in enumerate(complete[sent:], sent + 1):
                yield f"id: {i}\ndata: {line.strip()}\n\n"
                sent = i
            if not _active(run_id, path):
                yield "event: end\ndata: {}\n\n"
                return
            yield ": keep-alive\n\n"
            await asyncio.sleep(1)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.get("/api/runs/{run_id}/files/{name}")
def get_file(run_id: str, name: str) -> FileResponse:
    if not SERVABLE.match(name):
        raise HTTPException(400, "file not available")
    path = _run_dir(run_id) / name
    if not path.exists():
        raise HTTPException(404, "file not found")
    media = {"html": "text/html", "png": "image/png", "gif": "image/gif", "py": "text/plain", "json": "application/json", "eml": "message/rfc822"}
    return FileResponse(path, media_type=media[name.rsplit(".", 1)[-1]])
