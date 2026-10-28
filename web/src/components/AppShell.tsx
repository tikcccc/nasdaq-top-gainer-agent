"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import type { ReactNode } from "react";
import { STAGES } from "@/lib/report";
import { useReport } from "@/lib/store";
import EmailDialog from "./EmailDialog";
import GenerateDialog from "./GenerateDialog";
import ProgressScreen from "./ProgressScreen";
import { Button, Icon, Spinner, fmtPctUnits, fmtRunTime } from "./ui";

const NAV = [
  { href: "/", label: "Overview", detail: "Daily market pulse", icon: "home" },
  { href: "/analysis", label: "Analysis", detail: "Performance & evidence", icon: "chart" },
  { href: "/news", label: "News & commentary", detail: "Context behind the move", icon: "news" },
  { href: "/reports", label: "Reports", detail: "Your research archive", icon: "folder" },
] as const;

/** Always present: starts an AI analysis, or shows the one in progress. */
export function RunButton({ className = "", size = "md" }: { className?: string; size?: "md" | "lg" }) {
  const { job, setShowProgress, setDialog, offline } = useReport();
  const h = size === "lg" ? "h-11 px-5 text-[15px]" : "";
  if (job?.status === "running") {
    const pct = Math.round((job.done / STAGES.length) * 100);
    return (
      <Button variant="secondary" className={`${h} text-accent ring-accent-line ${className}`} onClick={() => setShowProgress(true)}>
        <Spinner className="h-4 w-4" /> Analysing… <span className="num">{pct}%</span>
      </Button>
    );
  }
  return (
    <Button variant="ai" className={`${h} ${className}`} onClick={() => setDialog("run")} disabled={offline}>
      <Icon name="sparkles" className="h-4 w-4" /> Run AI analysis
    </Button>
  );
}

function AiCard() {
  const { job, runs } = useReport();
  const running = job?.status === "running";
  const last = runs.find((r) => r.status === "ok" || r.status === "degraded");
  return (
    <div className="analyst-card">
      <div className="flex items-center gap-2">
        <span className="analyst-symbol"><Icon name="sparkles" className="h-4 w-4" /></span>
        <span className="text-sm font-semibold">AI Analyst</span>
        <span className={`ml-auto h-2 w-2 rounded-full ${running ? "animate-pulse bg-accent" : "bg-gain"}`} />
      </div>
      <p className="analyst-description">
        {running ? `${STAGES[Math.min(job!.done, STAGES.length - 1)].label}…` : last ? `Last analysis ${fmtRunTime(last.run_id)}.` : "Ready to analyse the latest session."}
      </p>
      {running && (
        <div className="mt-2.5 h-1.5 overflow-hidden rounded-full bg-white">
          <div className="ai-gradient h-full rounded-full transition-[width] duration-700" style={{ width: `${Math.max(6, (job!.done / STAGES.length) * 100)}%` }} />
        </div>
      )}
      <RunButton className="sidebar-run mt-4 w-full" />
    </div>
  );
}

function ReadyBanner() {
  const { job, dismissJob, openRun, showProgress } = useReport();
  const router = useRouter();
  if (!job || job.status === "running" || showProgress) return null;
  const ok = job.status === "ready";
  const sel = job.summary?.selected;
  const view = async () => {
    await openRun(job.runId);
    dismissJob();
    router.push(`/?report=${job.runId}`);
  };
  return (
    <div className={`mb-6 flex flex-wrap items-center gap-3 rounded-2xl border px-4 py-3 text-sm ${ok ? "border-accent-line bg-accent-soft" : "border-loss/30 bg-loss-soft"}`}>
      <span className={`flex h-7 w-7 items-center justify-center rounded-full text-white ${ok ? "ai-gradient" : "bg-loss"}`}>
        {ok ? <Icon name="check" className="h-4 w-4" /> : "!"}
      </span>
      <span className="text-ink">
        {ok ? <>New AI report ready{sel?.symbol && <>: <b className="font-semibold">{sel.symbol}</b> <span className="text-gain">{fmtPctUnits(sel.pct_change)}</span></>}</> : "The latest AI analysis could not be completed."}
      </span>
      <div className="ml-auto flex items-center gap-1">
        {ok && <Button variant="primary" className="h-8 rounded-lg px-3 text-[13px]" onClick={view}>View report</Button>}
        <button type="button" onClick={dismissJob} className="flex h-8 w-8 items-center justify-center rounded-lg text-muted hover:bg-white" aria-label="Dismiss"><Icon name="x" /></button>
      </div>
    </div>
  );
}

export default function AppShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const { job, showProgress, setShowProgress, toast, notify, offline, runId, dialog, setDialog } = useReport();
  const progressOpen = !!job && showProgress;
  const q = runId ? `?report=${runId}` : "";
  const isActive = (href: string) => (href === "/" ? pathname === "/" : pathname.startsWith(href));
  const pageName = progressOpen ? "Research in progress" : NAV.find((n) => isActive(n.href))?.label ?? "Overview";

  return (
    <div className="workspace-shell">
      <a href="#main-content" className="skip-link">Skip to main content</a>
      {/* Sidebar (desktop) */}
      <aside className="workspace-sidebar">
        <Link href={`/${q}`} className="workspace-brand" aria-label="Top Gainer Insights home">
          <span className="brand-monogram">t<span>g</span></span>
          <span className="brand-wordmark">TOP GAINER<small>INSIGHTS</small></span>
        </Link>
        <nav className="workspace-navigation" aria-label="Sections">
          <div className="nav-eyebrow">NASDAQ RESEARCH WORKSPACE</div>
          {NAV.map((n) => {
            const active = isActive(n.href) && !progressOpen;
            return (
              <Link key={n.href} href={`${n.href}${n.href === "/reports" ? "" : q}`} onClick={() => setShowProgress(false)}
                aria-current={active ? "page" : undefined} className={`workspace-nav-link ${active ? "is-active" : ""}`}>
                <Icon name={n.icon} className="h-[18px] w-[18px]" />
                <span>{n.label}<small>{n.detail}</small></span>
              </Link>
            );
          })}
        </nav>
        <div className="sidebar-bottom">
          <div className="sidebar-note"><span /><p>A clearer view.<br />An informed perspective.</p><small>Research grounded in evidence.</small></div>
          <AiCard />
          <div className="sidebar-footer"><span className={`status-dot ${offline ? "is-offline" : ""}`} />{offline ? "Service unavailable" : "NASDAQ market research"}<small>US EQUITIES / DAILY INTELLIGENCE</small></div>
        </div>
      </aside>

      <header className="desktop-topbar">
        <div className="workspace-breadcrumb">Research workspace <span>/</span><strong>{pageName}</strong></div>
        <div className="topbar-meta"><span>US equities</span><span className="topbar-status"><span className={`status-dot ${offline ? "is-offline" : ""}`} />{offline ? "Service unavailable" : "Research workspace"}</span></div>
      </header>

      {/* Top bar (mobile / tablet) */}
      <header className="mobile-topbar sticky top-0 z-20 lg:hidden">
        <div className="flex h-14 items-center justify-between gap-3 px-4">
          <Link href={`/${q}`} className="flex items-center gap-2">
            <span className="mobile-monogram">tg</span>
            <span className="text-[13px] font-semibold tracking-tight">Top Gainer Insights</span>
          </Link>
        </div>
        <nav className="grid grid-cols-4 px-2" aria-label="Sections">
          {NAV.map((n) => {
            const active = isActive(n.href) && !progressOpen;
            return (
              <Link key={n.href} href={`${n.href}${n.href === "/reports" ? "" : q}`} onClick={() => setShowProgress(false)}
                aria-label={n.label} aria-current={active ? "page" : undefined} className={`relative px-2 py-3 text-center text-[12px] whitespace-nowrap ${active ? "font-semibold text-ink" : "text-muted"}`}>
                {n.href === "/news" ? "News" : n.label}
                {active && <span className="absolute inset-x-3 -bottom-px h-0.5 rounded-full bg-accent" />}
              </Link>
            );
          })}
        </nav>
      </header>

      <main id="main-content" tabIndex={-1} className="workspace-main">
        {offline && (
          <div className="mb-6 rounded-2xl border border-loss/30 bg-loss-soft px-4 py-3 text-sm text-loss">
            We can&apos;t reach the reporting service right now. Please try again shortly.
          </div>
        )}
        <ReadyBanner />
        {progressOpen ? <ProgressScreen /> : children}
        <footer className="workspace-footer"><span>TOP GAINER INSIGHTS</span><span>Daily research. Considered decisions.</span></footer>
      </main>

      {dialog === "run" && <GenerateDialog onClose={() => setDialog(null)} />}
      {dialog === "email" && <EmailDialog onClose={() => setDialog(null)} />}

      {toast && (
        <div role="status" className="fixed right-5 bottom-5 z-50 flex max-w-sm items-start gap-3 rounded-2xl border border-line bg-white px-4 py-3 text-sm shadow-[0_12px_32px_-8px_rgba(20,21,28,0.18)]">
          <span className={`mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-xs text-white ${toast.tone === "ok" ? "bg-gain" : "bg-loss"}`}>{toast.tone === "ok" ? "✓" : "!"}</span>
          <span className="text-ink">{toast.text}</span>
          <button type="button" onClick={() => notify(null)} className="ml-1 text-faint hover:text-ink" aria-label="Dismiss">×</button>
        </div>
      )}
    </div>
  );
}
