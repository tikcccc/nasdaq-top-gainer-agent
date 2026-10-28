"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { STAGES } from "@/lib/report";
import { useReport } from "@/lib/store";
import { Button, Icon, Spinner, fmtPctUnits } from "./ui";

const HINT: Record<string, string> = {
  screen: "Ranking NASDAQ common stocks using your selected price and volume filters.",
  prices: "Loading and validating daily prices for the stock and its benchmarks.",
  analysis: "Computing returns, volatility, drawdown and the benchmark comparison.",
  news: "Reading the latest headlines and scoring their sentiment.",
  review: "Checking every figure and claim in the report against the data.",
  deliver: "Preparing the email and delivering the report.",
};

function useElapsed(since: number, running: boolean) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!running) return;
    const t = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(t);
  }, [running]);
  const s = Math.max(0, Math.round((now - since) / 1000));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

export default function ProgressScreen() {
  const { job, setShowProgress, dismissJob, openRun } = useReport();
  const router = useRouter();
  const running = job?.status === "running";
  const elapsed = useElapsed(job?.startedAt ?? 0, running);
  if (!job) return null;

  const pct = Math.round((job.done / STAGES.length) * 100);
  const current = STAGES[Math.min(job.done, STAGES.length - 1)];
  const sel = job.summary?.selected;
  const emailStatus = job.summary?.email?.status;

  const viewReport = async () => {
    await openRun(job.runId);
    dismissJob();
    router.push(`/?report=${job.runId}`);
  };

  return (
    <div className="mx-auto max-w-[620px] py-4 sm:py-8">
      <div className="overflow-hidden rounded-3xl border border-line bg-white">
        <div className="hero-wash px-7 pt-8 pb-7 sm:px-9">
          {running ? (
            <>
              <div className="flex items-center gap-2 text-[12px] font-semibold tracking-wider text-accent uppercase">
                <Icon name="sparkles" className="h-3.5 w-3.5" /> AI analysis in progress
              </div>
              <h1 className="mt-3 text-[26px] leading-tight font-semibold tracking-tight">{current.label}…</h1>
              <p className="mt-1.5 text-sm text-muted">{HINT[current.id]}</p>
              <div className="mt-6 flex items-center justify-between text-[13px]">
                <span className="text-muted">
                  {job.recipients.length ? <>Will be emailed to <b className="font-medium text-ink">{job.recipients.join(", ")}</b></> : "Usually takes about a minute"}
                </span>
                <span className="num font-medium text-ink">{pct}% · {elapsed}</span>
              </div>
              <div className="mt-2.5 h-2.5 overflow-hidden rounded-full bg-accent-soft">
                <div className="ai-gradient shimmer h-full rounded-full transition-[width] duration-700 ease-out" style={{ width: `${Math.max(pct, 5)}%` }} />
              </div>
            </>
          ) : job.status === "ready" ? (
            <div className="text-center">
              <span className="ai-gradient mx-auto flex h-14 w-14 items-center justify-center rounded-2xl text-white"><Icon name="check" className="h-7 w-7" /></span>
              <h1 className="mt-5 text-[24px] font-semibold tracking-tight">Your report is ready</h1>
              <p className="mt-1.5 text-sm text-muted">
                {sel?.symbol && <>Today&apos;s top gainer is <b className="font-semibold text-ink">{sel.symbol}</b> <span className="font-medium text-gain">{fmtPctUnits(sel.pct_change)}</span>. </>}
                {emailStatus === "sent" ? `Emailed to ${(job.recipients.length ? job.recipients : job.summary?.email?.recipients ?? []).join(", ")}.`
                  : emailStatus === "skipped_duplicate" ? "It was already delivered to these recipients today."
                  : emailStatus === "failed" ? "The email could not be delivered; you can send it from the report." : ""}
              </p>
              <div className="mt-6 flex justify-center gap-2">
                <Button variant="secondary" onClick={dismissJob}>Close</Button>
                <Button variant="primary" onClick={viewReport}>View report <Icon name="arrow" /></Button>
              </div>
            </div>
          ) : (
            <div className="text-center">
              <span className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-loss-soft text-2xl font-semibold text-loss">!</span>
              <h1 className="mt-5 text-[24px] font-semibold tracking-tight">The analysis could not be completed</h1>
              <p className="mt-1.5 text-sm text-muted">Something went wrong while preparing the report. Please try again in a few minutes.</p>
              <div className="mt-6 flex justify-center"><Button variant="secondary" onClick={dismissJob}>Close</Button></div>
            </div>
          )}
        </div>

        <ol className="grid gap-1 border-t border-line px-5 py-5 sm:grid-cols-2 sm:px-7">
          {STAGES.map((s, i) => {
            const state = i < job.done ? "done" : i === job.done && running ? "active" : "todo";
            return (
              <li key={s.id} className={`flex items-center gap-3 rounded-xl px-3 py-2.5 ${state === "active" ? "bg-accent-soft" : ""}`}>
                <span className={`flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-xs font-semibold ${
                  state === "done" ? "bg-ink text-white" : state === "active" ? "bg-white text-accent ring-2 ring-accent" : "bg-canvas text-faint"}`}>
                  {state === "done" ? <Icon name="check" className="h-3.5 w-3.5" /> : state === "active" ? <Spinner className="h-3.5 w-3.5" /> : i + 1}
                </span>
                <span className={`text-sm ${state === "todo" ? "text-faint" : state === "active" ? "font-semibold text-ink" : "text-muted"}`}>{s.label}</span>
              </li>
            );
          })}
        </ol>

        {running && (
          <div className="flex flex-wrap items-center justify-between gap-3 border-t border-line bg-canvas/60 px-7 py-4 sm:px-9">
            <span className="text-[13px] text-muted">Keep browsing — progress stays in the sidebar.</span>
            <Button variant="secondary" className="h-9" onClick={() => setShowProgress(false)}>Continue browsing</Button>
          </div>
        )}
      </div>
    </div>
  );
}
