"""SQLite job queue with atomic claims, leases and bounded submission rate."""

import json
import sqlite3
import time
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path


class QueueFull(ValueError):
    pass


class JobStore:
    def __init__(self, root: Path):
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / "jobs.sqlite"
        with self.connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    run_id TEXT PRIMARY KEY, request TEXT NOT NULL, status TEXT NOT NULL,
                    created REAL NOT NULL, updated REAL NOT NULL, owner TEXT,
                    lease_until REAL, error TEXT);
                CREATE TABLE IF NOT EXISTS requests (client TEXT NOT NULL, created REAL NOT NULL);
                CREATE INDEX IF NOT EXISTS requests_client ON requests(client, created);
            """)

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def enqueue(self, run_id: str, request: dict, *, max_pending=10):
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if conn.execute("SELECT count(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0] >= max_pending:
                raise QueueFull("report queue is full; try again after a report finishes")
            now = time.time()
            conn.execute("INSERT INTO jobs VALUES (?, ?, 'queued', ?, ?, NULL, NULL, NULL)",
                         (run_id, json.dumps(request), now, now))

    def rate_limit(self, client: str, *, limit=10, window=60):
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            now = time.time()
            conn.execute("DELETE FROM requests WHERE created<?", (now - window,))
            count = conn.execute("SELECT count(*) FROM requests WHERE client=?", (client,)).fetchone()[0]
            if count >= limit:
                raise QueueFull("request quota exceeded; try again in a minute")
            conn.execute("INSERT INTO requests VALUES (?, ?)", (client, now))

    def claim(self, owner: str, *, lease_s=120):
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            now = time.time()
            row = conn.execute("""SELECT * FROM jobs WHERE status='queued'
                OR (status='running' AND lease_until<?) ORDER BY created LIMIT 1""", (now,)).fetchone()
            if row is None:
                return None
            conn.execute("UPDATE jobs SET status='running', owner=?, lease_until=?, updated=? WHERE run_id=?",
                         (owner, now + lease_s, now, row["run_id"]))
            return {**dict(row), "request": json.loads(row["request"]), "resumed": row["status"] == "running"}

    def renew(self, run_id: str, owner: str, *, lease_s=120):
        with self.connect() as conn:
            return conn.execute("UPDATE jobs SET lease_until=?, updated=? WHERE run_id=? AND owner=? AND status='running'",
                                (time.time() + lease_s, time.time(), run_id, owner)).rowcount == 1

    def finish(self, run_id: str, owner: str, *, error=None):
        with self.connect() as conn:
            conn.execute("UPDATE jobs SET status=?, error=?, updated=?, lease_until=NULL WHERE run_id=? AND owner=?",
                         ("failed" if error else "finished", error, time.time(), run_id, owner))

    def get(self, run_id: str):
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE run_id=?", (run_id,)).fetchone()
            return dict(row) if row else None

    def active(self):
        with self.connect() as conn:
            return [dict(r) for r in conn.execute("SELECT * FROM jobs WHERE status IN ('queued','running') ORDER BY created")]


class JobWorker:
    """One worker per API process; SQLite claims allow several API processes safely."""
    def __init__(self, store: JobStore, execute):
        self.store, self.execute = store, execute
        self.owner = uuid.uuid4().hex
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True, name="report-worker")

    def run_once(self):
        job = self.store.claim(self.owner)
        if job is None:
            return False
        finished = threading.Event()
        def heartbeat():
            while not finished.wait(30):
                self.store.renew(job["run_id"], self.owner)
        beat = threading.Thread(target=heartbeat, daemon=True, name="report-lease")
        beat.start()
        try:
            self.execute(job)
        except Exception as exc:
            self.store.finish(job["run_id"], self.owner, error=f"{type(exc).__name__}: {exc}")
        else:
            self.store.finish(job["run_id"], self.owner)
        finally:
            finished.set()
            beat.join(timeout=2)
        return True

    def _loop(self):
        while not self.stop.is_set():
            try:
                worked = self.run_once()
            except Exception:
                import logging
                logging.getLogger(__name__).exception("job worker failed; retrying queue polling")
                worked = False
            if not worked:
                self.stop.wait(1)
