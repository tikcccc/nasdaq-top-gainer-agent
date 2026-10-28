"""send_email: the agent's only side effect. Numbers are filled by code; delivery is idempotent."""

from typing import Annotated

from langchain_core.tools import InjectedToolCallId, tool
from pydantic import Field
from langgraph.prebuilt import InjectedState
from nasdaq_agent.context import get_ctx
from nasdaq_agent.infra import mailer, delivery
from nasdaq_agent.infra.errors import ToolError
from nasdaq_agent.llm.factory import describe
from nasdaq_agent.reporting.evidence_report import EvidenceSelection, KeyFinding, NewsId, ReportInputError, build
from nasdaq_agent.reporting.render import build_report
from nasdaq_agent.tools.base import ToolResult, agent_tool

DELIVERED = {"sent", "dry_run", "skipped_duplicate"}


def deliver(state: dict, narrative: str, citations: list[str], *, fallback_reason: str | None = None) -> dict:
    """Render, save and send once within this run. A new run can send again."""
    ctx = get_ctx()
    cfg = ctx.settings.email
    recipients = cfg.to
    if not recipients and not cfg.dry_run:
        raise ToolError("no_recipient", "no email recipient configured", hint="Set EMAIL_TO in .env or email.to in config.yaml.")
    sender = cfg.sender or cfg.smtp.user or "nasdaq-agent@localhost"
    run_info = {"run_id": ctx.run_id, "models": describe(ctx.settings)}
    report = build_report(state, narrative, run_dir=ctx.run_dir, run_info=run_info, citations=citations,
                          fallback_reason=fallback_reason)

    selected = state.get("selected") or {}
    trade_date = (state.get("session") or {}).get("trade_date", "unknown")
    symbol = selected.get("symbol", "NONE")
    # Data-quality notes go to the run record and log only; reader-relevant limits are in the report text.
    info = {"subject": report.subject, "recipients": recipients, "fallback": bool(fallback_reason),
            "notes": report.notes, "report_facts": report.facts, "report_renderer": report.renderer}

    html_preview = report.html.replace('src="cid:chart"', f'src="{report.images["chart"].name}"') if "chart" in report.images else report.html
    (ctx.run_dir / "email.html").write_text(html_preview, encoding="utf-8")
    msg = mailer.build_message(report, sender, recipients)
    eml = mailer.save_eml(msg, ctx.run_dir / "email.eml")
    info.update(eml_path=str(eml), html_path=str(ctx.run_dir / "email.html"), message_id=msg["Message-ID"])

    key = delivery.operation_key(ctx.run_id, recipients, source="agent", complete=bool(state.get("analysis")))
    outcome = delivery.deliver_message(msg, cfg, ctx.settings.output_root, key,
                                       {"run_id": ctx.run_id, "trade_date": trade_date, "symbol": symbol, **info})
    return {**info, **outcome}


class SendEmailInput(EvidenceSelection):
    # Same contract, but the tool also accepts the shapes models send (a JSON string,
    # "kind" or "news:n1" items); build() normalises them before strict validation.
    key_findings: list[KeyFinding | str] | str | None = Field(None,
        description="1-3 findings, most important first, e.g. [{\"kind\": \"vs_sp500\"}, {\"kind\": \"news\", \"ref\": \"n1\"}]. "
                    "Kinds only; code writes the sentences.")
    state: Annotated[dict, InjectedState] = None
    tool_call_id: Annotated[str, InjectedToolCallId] = ""


@tool("send_email", args_schema=SendEmailInput)
@agent_tool
def send_email(
    selected_news_ids: list[NewsId] = None,
    selected_risk_evidence_ids: list[NewsId] = None,
    include_data_limitations: bool = True,
    key_findings: list | None = None,
    state: Annotated[dict, InjectedState] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
) -> ToolResult:
    """Email the final report to the configured recipients (you cannot choose recipients).

    Select up to three current-stock news IDs and up to two source disclosure IDs.
    Start with the single most relevant headline. Choose 1-3 key_findings, most
    important first, from the kinds in the schema; they lead the report. Include a
    data_limitation finding whenever the ranking, eligibility or price is unconfirmed
    or the data is intraday. No narrative, risk explanation, numbers or citations text
    is accepted. Code writes a report of at most 250 words, with no minimum, from verified
    prices, explicitly attributed source headlines and data limits.
    Titles that do not fit remain in the separate source list. Material limitations
    are always included. Call once after execute_python succeeds; call alone."""
    state = state or {}
    if not state.get("analysis"):
        raise ToolError("missing_prerequisite", "no verified analysis yet",
                        hint="Run execute_python successfully before sending the email.")
    if (state.get("email") or {}).get("status") in DELIVERED:
        raise ToolError("already_sent", "the report was already delivered in this run", retryable=False,
                        hint="Do not call send_email again; reply with a one-line final summary.")
    try:
        args = {"selected_news_ids": selected_news_ids or [],
                "selected_risk_evidence_ids": selected_risk_evidence_ids or [],
                "include_data_limitations": include_data_limitations}
        if key_findings:
            args["key_findings"] = key_findings   # build() normalises the accepted shapes
        report = build(state, args, get_ctx().settings)
    except ReportInputError as exc:
        raise ToolError("report_selection_invalid", str(exc), retryable=True,
                        hint="Use only current-stock/current-date headline IDs; no free prose. Refresh stale analysis or sentiment first.") from exc

    # Revalidation here prevents direct tool use from bypassing the graph guard.
    citations = list(dict.fromkeys(report.citations + report.omitted_ids))
    status = deliver(state, report.narrative, citations)
    warnings = []
    if report.omitted_ids:
        warnings.append("Headline titles omitted from the short body (length/format); retained in the source list: " + ", ".join(report.omitted_ids))
    if status["status"] == "failed":
        warnings.append(f"SMTP delivery failed ({status['error']}); the email was saved to {status['eml_path']}")
    if status["status"] == "delivery_unknown":
        warnings.append("SMTP acceptance is unknown; automatic re-send is blocked pending reconciliation")
    if status["status"] == "skipped_duplicate":
        warnings.append("this run's delivery already completed; retry did not send an extra email")
    return ToolResult(
        data={k: v for k, v in status.items() if k not in {"html_path"}},
        warnings=warnings,
        source="gmail_smtp" if status["status"] == "sent" else None,
        update={"email": {**status, "narrative": report.narrative, "citations": citations,
                          "report_mode": report.version, "evidence_selection": report.selection,
                          "quoted_ids": report.quoted_ids, "omitted_ids": report.omitted_ids,
                          "word_count": report.word_count, "key_findings": report.key_findings,
                          "key_findings_source": report.findings_source}},
    )
