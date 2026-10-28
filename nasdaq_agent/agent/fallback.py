"""Deterministic last resort: always send something useful, without the LLM.

Reached when the run hits its deadline / LLM budget, the LLM is unreachable, the
guard keeps rejecting, or the agent stops without sending. Uses whatever validated
facts are already in state; missing analysis is computed by the reference code.
"""

from __future__ import annotations

import logging
import hashlib
from pathlib import Path

import pandas as pd

from nasdaq_agent.data import metrics
from nasdaq_agent.reporting import placeholders
from nasdaq_agent.reporting.render import render_narrative
from nasdaq_agent.reporting import evidence_report
from nasdaq_agent.context import get_ctx
from nasdaq_agent.tools.emailer import DELIVERED, deliver

log = logging.getLogger(__name__)

FULL = """{{symbol}} ({{company}}): {{selection_label}} for {{trade_date}}, session move {{pct_change}}.

Over the last {{days}} trading days it returned {{metrics.cumulative_return}} with a {{metrics.trend}} trend, versus {{metrics.gspc_return}} for the S&P 500 and {{metrics.ixic_return}} for the NASDAQ Composite."""

PARTIAL = """{{symbol}} ({{company}}): {{selection_label}} for {{trade_date}}, session move {{pct_change}}. The multi-day analysis could not be completed."""


def _reference_analysis(state: dict) -> dict | None:
    selected = state.get("selected") or {}
    datasets = state.get("datasets") or {}
    meta = datasets.get(selected.get("dataset_id", ""))
    if not meta:
        return None
    try:
        if selected.get("analysis_key") and any(hashlib.sha256(Path(m["path"]).read_bytes()).hexdigest() != m["sha256"] for m in datasets.values()):
            return None
        stock = pd.read_csv(meta["path"])
        benches = {ds[3:]: pd.read_csv(m["path"]) for ds, m in datasets.items() if m.get("proxy_for") or m["symbol"].startswith("^")}
        return {**metrics.compute(stock, benches, selected["days"], selected["symbol"]),
                "analysis_key": selected.get("analysis_key"), "origin": "fallback"}
    except Exception as exc:  # noqa: BLE001
        log.warning("reference analysis failed: %s", exc)
        return None


def fallback_node(state: dict) -> dict:
    reason = state.get("stop_reason") or "the agent finished without sending the email"
    if (state.get("email") or {}).get("status") in DELIVERED:
        return {}
    update: dict = {"degraded": [reason], "stop_reason": reason}
    working = dict(state)
    if working.get("analysis") and (working.get("selected") or {}).get("analysis_key"):
        from nasdaq_agent.agent.evidence import analysis_problems
        if analysis_problems(working):
            working["analysis"] = {}
            update["analysis"] = {}
    if working.get("selected") and not working.get("analysis"):
        reference = _reference_analysis(working)
        if reference:
            working["analysis"] = reference
            update["analysis"] = reference
            update["degraded"].append("analysis computed by the reference implementation")

    report_mode = "partial-v1"
    if working.get("analysis") and (working.get("selected") or {}).get("analysis_key"):
        working["sentiment"] = {}
        update["sentiment"] = {}
        try:
            report = evidence_report.build(working, {}, get_ctx().settings)
            narrative = report.narrative
            report_mode = report.version
        except evidence_report.ReportInputError:
            narrative = render_narrative(FULL, placeholders.build_context(working))
    elif working.get("selected"):
        template = FULL if working.get("analysis") else PARTIAL
        narrative = render_narrative(template, placeholders.build_context(working))
    else:
        narrative = "Today's top NASDAQ gainer report could not be completed. Please try again later."
    # Technical failure details remain in stop_reason, degraded and the run trace.

    try:
        status = deliver(working, narrative, [], fallback_reason=reason)
    except Exception as exc:  # noqa: BLE001
        log.exception("fallback delivery failed")
        status = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    update["email"] = {**status, "narrative": narrative, "fallback": True, "report_mode": report_mode}
    return update
