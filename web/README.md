# Web dashboard

Next.js 16 + Tailwind UI for the agent. It talks to the FastAPI backend in `nasdaq_agent/api.py`
(default `http://127.0.0.1:8000`, override with `NEXT_PUBLIC_API_URL`).

```bash
npm install
npm run dev
```

Pages: `/` overview, `/analysis`, `/news`, `/reports` (paginated history). `?report=<run_id>` selects a report and is shareable.

- `src/lib/store.tsx`: shared state (selected report, recipients, the report being generated). Runs are followed by polling
  their trace, so a run started in another tab or from the CLI also shows progress
- `src/components/AppShell.tsx`: top bar, page tabs, "Generate report" button / progress status, toasts
- `src/components/GenerateDialog.tsx`, `ProgressScreen.tsx`: start a report (optional recipients) and follow its progress
- `src/components/views.tsx`: the four pages; `ReportSections.tsx`: the cards they are built from
- `src/components/EChart.tsx`, `charts.ts`: ECharts wrapper and chart options
- `src/lib/report.ts`: builds the report from tool results; progress stages; email validation
