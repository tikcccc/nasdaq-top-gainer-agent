"""Pre-send guard: the email is the only outward action, so it is checked before it runs.

The default report contract accepts only evidence selections; the guard
builds the report from verified state and rejects anything that does not validate.
The earlier free-prose checker lives in scripts/legacy/narrative_guard.py for historical
evaluations only.
"""

from __future__ import annotations

import json

from nasdaq_agent.config import Settings
from nasdaq_agent.reporting.version import REPORT_VERSION


def check(state: dict, args: dict, settings: Settings) -> tuple[list[str], dict]:
    """Production send guard: closed evidence selections, no model-written prose."""
    from nasdaq_agent.reporting.evidence_report import build, ReportInputError
    try:
        report = build(state, args, settings)
    except ReportInputError as exc:
        return [str(exc)], {"method": REPORT_VERSION}
    return [], {"method": report.version, "word_count": report.word_count,
                "selection": report.selection, "quoted_ids": report.quoted_ids,
                "omitted_ids": report.omitted_ids, "rendered_preview": report.narrative[:400]}


def rejection_payload(problems: list[str]) -> str:
    return json.dumps({
        "ok": False,
        "error": {
            "type": "guard_rejected",
            "message": " | ".join(problems),
            "retryable": True,
            "hint": "Select valid current-stock evidence IDs and refresh stale analysis if requested. send_email accepts no narrative or risk text.",
        },
    })
