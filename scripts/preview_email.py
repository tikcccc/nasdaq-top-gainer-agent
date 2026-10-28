"""Re-render a finished run's email from its checkpoint (for iterating on the template).

    uv run python scripts/preview_email.py <run_id> [narrative.txt]

Writes outputs/run_<id>/email_preview.html (chart referenced as a sibling file).
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402

from nasdaq_agent.agent.graph import build_graph  # noqa: E402
from nasdaq_agent.config import load_settings  # noqa: E402
from nasdaq_agent.context import start_run  # noqa: E402
from nasdaq_agent.llm.factory import describe  # noqa: E402
from nasdaq_agent.reporting import placeholders  # noqa: E402
from nasdaq_agent.reporting.render import build_report, render_narrative  # noqa: E402


def main(run_id: str, narrative_file: str | None = None) -> None:
    settings = load_settings()
    ctx = start_run(settings, run_id)
    conn = sqlite3.connect(settings.output_root / "checkpoints.sqlite", check_same_thread=False)
    state = build_graph(SqliteSaver(conn)).get_state({"configurable": {"thread_id": run_id}}).values
    email = state.get("email") or {}
    if narrative_file:
        narrative = render_narrative(Path(narrative_file).read_text(encoding="utf-8"), placeholders.build_context(state))
    else:
        narrative = email.get("narrative", "")
    report = build_report(state, narrative, run_dir=ctx.run_dir, citations=email.get("citations") or [],
                          run_info={"run_id": run_id, "models": describe(settings)})
    html = report.html
    if "chart" in report.images:
        html = html.replace('src="cid:chart"', f'src="{report.images["chart"].name}"')
    out = ctx.run_dir / "email_preview.html"
    out.write_text(html, encoding="utf-8")
    print(report.subject)
    print(out)


if __name__ == "__main__":
    main(*sys.argv[1:])
