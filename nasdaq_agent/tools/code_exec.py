"""execute_python: run analysis code that the agent wrote, then verify what it produced."""

import re
import secrets
import shutil
from typing import Annotated

import pandas as pd
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from pydantic import Field, ValidationError

from nasdaq_agent.context import get_ctx
from nasdaq_agent.data import metrics
from nasdaq_agent.infra.errors import ToolError
from nasdaq_agent.infra.schemas import AnalysisResult
from nasdaq_agent.reporting.version import REPORT_VERSION
from nasdaq_agent.sandbox import policy, runner
from nasdaq_agent.tools.base import ToolResult, agent_tool

HELPERS = "load(dataset_id) -> DataFrame[date, open, high, low, close, volume]; save_result(dict); save_chart(fig, name='chart')"


def _hint(stderr: str, datasets: dict) -> str:
    if "KeyError" in stderr and "dataset_id" not in stderr:
        return "A column or key does not exist. DataFrames from load() have columns: date, open, high, low, close, volume."
    if "unknown dataset_id" in stderr:
        return f"Use one of the dataset ids: {sorted(datasets)}."
    if "ModuleNotFoundError" in stderr or "ImportError" in stderr:
        return f"Only these imports are available: {sorted(policy.ALLOWED_IMPORTS)}."
    if "TimeoutError" in stderr:
        return "The code ran too long. Avoid loops over large ranges; the data has only a handful of rows."
    return "Read the traceback, fix the code and call execute_python again with the full corrected script."


def _validation_summary(exc: ValidationError) -> str:
    return "; ".join(f"{'.'.join(map(str, e['loc'])) or 'result'}: {e['msg']}" for e in exc.errors()[:6])


@tool("execute_python")
@agent_tool
def execute_python(
    code: Annotated[str, Field(min_length=10, description="A complete Python script. Use load(), save_result() and save_chart(); no file or network access.")],
    purpose: Annotated[str, Field(max_length=200, description="One line on what this code computes.")] = "",
    state: Annotated[dict, InjectedState] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
) -> ToolResult:
    """Execute Python analysis code you wrote, in an isolated sandbox, and verify the result.

    Available inside the script: load(dataset_id) -> DataFrame[date, open, high, low, close,
    volume]; save_result(dict); save_chart(fig, name='chart'). Allowed imports: pandas, numpy,
    matplotlib, math, statistics, json, datetime, collections, itertools. The script must
    call save_result() with the fields listed in the analysis task (returns as fractions,
    0.05 = +5%) and should call save_chart() once. On failure you get the traceback and a
    hint: fix the script and call again (limited attempts). The result is checked against
    an independent calculation; mismatches are returned as errors."""
    ctx = get_ctx()
    settings = ctx.settings.sandbox
    state = state or {}
    selected = state.get("selected")
    datasets = state.get("datasets") or {}
    if not selected or not datasets:
        raise ToolError("missing_prerequisite", "no datasets yet", hint="Call get_price_history for the chosen stock first.")
    failures = (state.get("failures") or {}).get("execute_python", 0)
    if failures >= settings.max_attempts:
        raise ToolError(
            "attempt_limit", f"execute_python already failed {failures} times", retryable=False,
            hint="No attempts are left. The backend sends a report built from the reference calculation.",
            stop_run=f"execute_python failed {failures} times; the report uses the reference calculation",
        )
    try:
        return _run_verified(code, purpose, ctx, state, selected, datasets)
    except ToolError as exc:
        if exc.stop_run is None and failures + 1 >= settings.max_attempts:
            exc.stop_run = f"execute_python failed {failures + 1} times; the report uses the reference calculation"
            exc.hint = "No attempts are left. The backend sends a report built from the reference calculation."
        raise


def _run_verified(code: str, purpose: str, ctx, state: dict, selected: dict, datasets: dict) -> ToolResult:
    settings = ctx.settings.sandbox

    violations = policy.check(code)
    if violations:
        raise ToolError(
            "policy_violation", "; ".join(violations[:8]), retryable=True,
            hint=f"Remove these constructs. Use only the helpers ({HELPERS}) for data and charts.",
        )

    version = (state.get("attempts") or {}).get("execute_python", 0) + 1
    work = ctx.sandbox_dir / f"v{version}_{secrets.token_hex(4)}"
    (ctx.run_dir / f"code_v{version}.py").write_text(f"# purpose: {purpose}\n{code}", encoding="utf-8")
    paths = {ds: meta["path"] for ds, meta in datasets.items()}
    result = runner.run(code, paths, work, timeout_s=min(settings.timeout_s, max(0.1, ctx.seconds_left())),
                        max_stdout=settings.max_stdout_chars, backend=settings.backend, image=settings.image,
                        memory_mb=settings.memory_mb, cpus=settings.cpus)

    if result.unavailable:
        raise ToolError(
            "sandbox_unavailable", f"{result.unavailable}: {result.stderr_tail}", retryable=False,
            hint="The code was not run, so do not rewrite it. The backend sends a report from the reference calculation.",
            stop_run=f"analysis sandbox unavailable ({runner.UNAVAILABLE_LABELS[result.unavailable]}); "
                     "the report uses the reference calculation",
        )
    if not result.ok:
        raise ToolError("code_error", result.stderr_tail or "process failed without output", retryable=True,
                        hint=_hint(result.stderr_tail, datasets))
    if result.result is None:
        raise ToolError("no_result", "the script finished but never called save_result()", retryable=True,
                        hint="Call save_result({...}) with all required fields at the end of the script.")

    symbol, days = selected["symbol"], selected["days"]
    for key, expected in (("symbol", symbol), ("days", days)):
        if key in result.result and result.result[key] != expected:
            raise ToolError("invalid_result", f"generated result cannot override {key}={expected!r}", retryable=True)
    payload = {**result.result, "symbol": symbol, "days": days}
    try:
        analysis = AnalysisResult.model_validate(payload)
    except ValidationError as exc:
        raise ToolError("invalid_result", _validation_summary(exc), retryable=True,
                        hint="Fix the listed fields; returns must be fractions and daily_returns must have exactly "
                             f"{days} values (the last {days} sessions).") from exc
    stock = pd.read_csv(datasets[selected["dataset_id"]]["path"])
    benches = {ds[3:]: pd.read_csv(meta["path"]) for ds, meta in datasets.items() if meta.get("proxy_for") or meta["symbol"].startswith("^")}
    reference = metrics.compute(stock, benches, days, symbol)
    problems = metrics.compare(analysis.model_dump(), reference)
    if problems:
        raise ToolError(
            "result_mismatch", "; ".join(problems), retryable=True,
            hint=f"Check the window (last {days} daily returns = last {days + 1} closes), use 'close', and report fractions.",
        )

    charts = []
    for i, png in enumerate(result.charts):
        dest = ctx.run_dir / ("chart.png" if i == 0 else png.name)
        shutil.copyfile(png, dest)
        charts.append(str(dest))
    rounded = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in analysis.model_dump().items()}
    return ToolResult(
        data={
            "analysis": rounded,
            "report_contract": {"mode": REPORT_VERSION, "next": "select current headline IDs for send_email; backend writes the report, no narrative argument"},
            "verified": "matches an independent calculation within tolerance",
            "charts_saved": len(charts),
            "stdout": re.sub(r"\s+\n", "\n", result.stdout)[-1500:],
            "duration_s": round(result.duration_s, 2),
        },
        warnings=[] if charts else ["no chart saved; call save_chart(fig) if you want a chart in the email"],
        update={"analysis": {**analysis.model_dump(), "analysis_key": selected.get("analysis_key"),
                             "origin": "agent", "sandbox": settings.backend}, "charts": charts},
    )
