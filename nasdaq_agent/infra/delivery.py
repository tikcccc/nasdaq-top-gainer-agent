"""Durable, cross-process email claims. Ambiguous SMTP outcomes are never auto-retried."""

import hashlib
import json
import smtplib
import sqlite3
import time
import uuid
from pathlib import Path

from nasdaq_agent.infra import mailer


def report_key(trade_date: str, symbol: str, recipients: list[str], *, days=None, selection=None, complete=True) -> str:
    """Historical ledger identifier; new deliveries use operation_key instead."""
    payload = [trade_date, symbol.upper(), sorted(set(r.strip().lower() for r in recipients)), days,
               selection or {}, "analysis" if complete else "failure_notification"]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def operation_key(operation_id: str, recipients: list[str], *, source: str, complete=True) -> str:
    """One user action or agent run, never a date/stock-wide sending limit."""
    payload = ["delivery-v2", source, operation_id,
               sorted(set(r.strip().lower() for r in recipients)),
               "analysis" if complete else "failure_notification"]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _connect(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(root / "delivery.sqlite", timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("""CREATE TABLE IF NOT EXISTS deliveries (
        key TEXT PRIMARY KEY, status TEXT NOT NULL, message_id TEXT NOT NULL,
        info TEXT NOT NULL, updated REAL NOT NULL, error TEXT)""")
    conn.commit()
    return conn


def deliver_message(msg, config, root: Path, key: str, info: dict, *, force=False) -> dict:
    if config.dry_run:
        return {"status": "dry_run", "message_id": msg["Message-ID"]}
    if force:
        key = f"{key}:forced:{info.get('run_id') or uuid.uuid4().hex}"
    conn = _connect(root)
    try:
        conn.execute("BEGIN IMMEDIATE")
        previous = conn.execute("SELECT * FROM deliveries WHERE key=?", (key,)).fetchone()
        if previous and previous["status"] in {"sent", "sending", "unknown"}:
            conn.commit()
            return {"status": "skipped_duplicate" if previous["status"] == "sent" else "delivery_unknown",
                    "message_id": previous["message_id"], "delivery_key": key,
                    "error": previous["error"] if previous["status"] != "sent" else None}
        if previous:
            del msg["Message-ID"]
            msg["Message-ID"] = previous["message_id"]
        conn.execute("INSERT OR REPLACE INTO deliveries VALUES (?, 'sending', ?, ?, ?, NULL)",
                     (key, msg["Message-ID"], json.dumps(info), time.time()))
        conn.commit()  # durable before calling SMTP; a crash after this point requires reconciliation
        try:
            mailer.send(msg, config)
        except (smtplib.SMTPAuthenticationError, smtplib.SMTPRecipientsRefused,
                smtplib.SMTPResponseException, mailer.PreDeliveryError) as exc:
            status, error = "failed", f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            status, error = "unknown", f"{type(exc).__name__}: {exc}"
        else:
            status, error = "sent", None
        conn.execute("UPDATE deliveries SET status=?, error=?, updated=? WHERE key=?",
                     (status, error, time.time(), key))
        conn.commit()
        return {"status": "delivery_unknown" if status == "unknown" else status,
                "message_id": msg["Message-ID"], "delivery_key": key, "error": error}
    finally:
        conn.close()
