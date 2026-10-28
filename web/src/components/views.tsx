"use client";

import { useMemo, type ReactNode } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useReport } from "@/lib/store";
import { RunButton } from "./AppShell";
import EChart from "./EChart";
import { performanceOption, priceOption, returnsOption, stockSymbol } from "./charts";
import { Candidates, Commentary, Hero, History, KpiRow, Methodology, Sentiment } from "./ReportSections";
import { Button, Card, Empty, Icon, Spinner, fmtRunTime } from "./ui";

function PageHeader({ eyebrow, title, description, actions }: { eyebrow?: ReactNode; title: string; description?: string; actions?: ReactNode }) {
  return (
    <div className="page-heading">
      <div className="eyebrow mb-3">NASDAQ / MARKET INTELLIGENCE</div>
      <div className="page-title-row">
        <h1 className="page-title">{title}<span>.</span></h1>
        {actions}
      </div>
      {description && <p className="page-description">{description}</p>}
      {eyebrow && <div className="report-context">{eyebrow}</div>}
    </div>
  );
}

/** "Report generated … · Change" line shown above report pages. */
function ReportContext() {
  const { runId, loading, view } = useReport();
  if (!runId || !view.selected?.symbol) return null;
  return (
    <span className="inline-flex items-center gap-2">
      Report generated {fmtRunTime(runId)}
      <span className="text-line-strong">·</span>
      <Link href="/reports" className="font-medium text-accent hover:underline">Change report</Link>
      {loading && <Spinner className="h-3.5 w-3.5 text-faint" />}
    </span>
  );
}

function EmailButton() {
  const { setDialog, runId, view, job, config } = useReport();
  return (
    <Button variant="primary" className="overview-email" onClick={() => setDialog("email")}
      disabled={!runId || !view.email || job?.status === "running" || (config ? !config.email_enabled : false)}>
      <Icon name="mail" /> Email report
    </Button>
  );
}

function useCharts() {
  const { series, view } = useReport();
  const symbol = stockSymbol(series, view.selected?.symbol);
  const bars = symbol ? series[symbol] : undefined;
  return useMemo(() => ({
    symbol,
    bars,
    price: bars?.length ? priceOption(bars) : null,
    performance: symbol && bars?.length ? performanceOption(series, symbol) : null,
    returns: view.analysis && bars?.length ? returnsOption(view.analysis, bars) : null,
  }), [series, symbol, bars, view.analysis]);
}

export function OverviewView() {
  const { view } = useReport();
  const charts = useCharts();
  return (
    <>
      <PageHeader eyebrow={<ReportContext />} title="Market overview" description="The session’s standout mover. The evidence behind the momentum." actions={
        <div className="overview-action-buttons" role="group" aria-label="Report actions">
          <RunButton className="overview-run" />
          <EmailButton />
        </div>
      } />
      <div className="space-y-6">
        <Hero v={view} bars={charts.bars} />
        <KpiRow v={view} />
        <Card title="Price & volume" subtitle={charts.symbol ? `${charts.symbol} · Daily price action and trading activity` : "Daily price action and trading activity"} action={<span className="chart-period">DAILY / USD</span>}>
          {charts.price ? <EChart option={charts.price} height={360} /> : <Empty>No price data.</Empty>}
        </Card>
      </div>
    </>
  );
}

export function AnalysisView() {
  const { view } = useReport();
  const charts = useCharts();
  return (
    <>
      <PageHeader eyebrow={<ReportContext />} title="Market analysis" description="Performance, benchmarks and the quality of the underlying data." />
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Card title="Relative performance" subtitle="Change since the start of the window">
          {charts.performance ? <EChart option={charts.performance} height={300} /> : <Empty>No benchmark data.</Empty>}
        </Card>
        <Card title="Daily returns" subtitle="Close-to-close change">
          {charts.returns ? <EChart option={charts.returns} height={300} /> : <Empty>No analysis yet.</Empty>}
        </Card>
        <div className="lg:col-span-2"><Candidates v={view} /></div>
        <div className="lg:col-span-2"><Methodology v={view} /></div>
      </div>
    </>
  );
}

export function NewsView() {
  const { view } = useReport();
  return (
    <>
      <PageHeader eyebrow={<ReportContext />} title="News & commentary" description="A considered perspective on the stories moving the market." />
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,1fr)_340px]">
        <Commentary v={view} />
        <Sentiment v={view} />
      </div>
    </>
  );
}

export function ReportsView() {
  const { runs, runId, openRun } = useReport();
  const router = useRouter();
  const open = async (id: string) => {
    await openRun(id);
    router.push(`/?report=${id}`);
  };
  return (
    <>
      <PageHeader title="Research archive" description="A record of your daily market reports, ready to revisit and share." />
      <div className="archive-label"><span>ALL REPORTS</span><span className="num">{runs.length} report{runs.length === 1 ? "" : "s"}</span></div>
      <History runs={runs} current={runId} onOpen={open} />
    </>
  );
}
