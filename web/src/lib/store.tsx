"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { api, type AppConfig, type RunListItem, type Series, type StockFilters, type Summary } from "./api";
import { STAGES, stagesDone, viewFromSteps, type ReportView } from "./report";
import { fromTrace, type Step } from "./timeline";

export type Toast = { tone: "ok" | "error"; text: string };

/** A report being generated: started here, in another tab, or from the CLI. */
export interface Job {
  runId: string;
  recipients: string[];
  startedAt: number;
  done: number;
  status: "running" | "ready" | "failed";
  summary?: Summary | null;
}

interface Store {
  config: AppConfig | null;
  offline: boolean;
  runs: RunListItem[];
  runId?: string;
  view: ReportView;
  series: Series;
  loading: boolean;
  openRun: (id: string) => Promise<void>;
  email: string;
  setEmail: (v: string) => void;
  job: Job | null;
  showProgress: boolean;
  setShowProgress: (v: boolean) => void;
  dismissJob: () => void;
  generate: (to: string[], filters?: StockFilters, send?: boolean) => Promise<boolean>;
  sending: boolean;
  send: (to: string[]) => Promise<boolean>;
  toast: Toast | null;
  notify: (t: Toast | null) => void;
  dialog: Dialog;
  setDialog: (d: Dialog) => void;
}

export type Dialog = "run" | "email" | null;

const Ctx = createContext<Store | null>(null);

export const useReport = () => {
  const s = useContext(Ctx);
  if (!s) throw new Error("useReport outside ReportProvider");
  return s;
};

const EMAIL_KEY = "tgi.recipients";
const POLL_MS = 1500;

export function ReportProvider({ children }: { children: ReactNode }) {
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [offline, setOffline] = useState(false);
  const [runs, setRuns] = useState<RunListItem[]>([]);
  const [runId, setRunId] = useState<string>();
  const [steps, setSteps] = useState<Step[]>([]);
  const [summary, setSummary] = useState<Summary | null>(null);
  const [series, setSeries] = useState<Series>({});
  const [loading, setLoading] = useState(true);
  const [email, setEmailState] = useState("");
  const [job, setJob] = useState<Job | null>(null);
  const [showProgress, setShowProgress] = useState(false);
  const [sending, setSending] = useState(false);
  const sendInFlight = useRef(false);
  const [toast, setToast] = useState<Toast | null>(null);
  const [dialog, setDialog] = useState<Dialog>(null);
  const tracking = useRef<string | null>(null);

  const notify = useCallback((t: Toast | null) => {
    setToast(t);
    if (t) window.setTimeout(() => setToast((cur) => (cur === t ? null : cur)), 6000);
  }, []);

  const setEmail = (v: string) => {
    setEmailState(v);
    try { localStorage.setItem(EMAIL_KEY, v); } catch { /* storage unavailable */ }
  };

  const refreshRuns = useCallback(() => api.runs().then((r) => { setRuns(r); return r; }), []);

  const openRun = useCallback(async (id: string) => {
    setLoading(true);
    try {
      const [detail, bars] = await Promise.all([api.run(id), api.series(id).catch(() => ({}))]);
      setRunId(id);
      setSteps(fromTrace(detail.trace));
      setSummary(detail.summary);
      setSeries(bars);
      const url = new URL(window.location.href);
      url.searchParams.set("report", id);
      window.history.replaceState(window.history.state, "", url);
    } catch (e) {
      notify({ tone: "error", text: (e as Error).message });
    } finally {
      setLoading(false);
    }
  }, [notify]);

  /** Follow a run by polling its trace until it has a summary (works for runs started anywhere). */
  const track = useCallback((id: string, recipients: string[], showScreen: boolean) => {
    if (tracking.current === id) return;
    tracking.current = id;
    setJob({ runId: id, recipients, startedAt: Date.now(), done: 0, status: "running" });
    setShowProgress(showScreen);
    const tick = async () => {
      if (tracking.current !== id) return;
      try {
        const detail = await api.run(id);
        const trace = fromTrace(detail.trace);
        const firstTs = (detail.trace[0] as { ts?: number } | undefined)?.ts;
        if (detail.running) {
          setJob((j) => (j && j.runId === id ? { ...j, done: stagesDone(trace), startedAt: firstTs ? firstTs * 1000 : j.startedAt } : j));
        } else {
          tracking.current = null;
          const ok = !!detail.summary && detail.summary.status !== "failed";
          setJob((j) => (j && j.runId === id ? { ...j, done: ok ? STAGES.length : stagesDone(trace), status: ok ? "ready" : "failed", summary: detail.summary } : j));
          refreshRuns();
          return;
        }
      } catch { /* the run directory may not exist yet; retry */ }
      window.setTimeout(tick, POLL_MS);
    };
    tick();
  }, [refreshRuns]);

  useEffect(() => {
    let alive = true;
    const loadConfig = () => api.config()
      .then((c) => {
        if (!alive) return c;
        setConfig(c);
        setOffline(false);
        if (c.active_run && !tracking.current) track(c.active_run, [], false);
        return c;
      });

    loadConfig()
      .then((c) => {
        let saved = "";
        try { saved = localStorage.getItem(EMAIL_KEY) ?? ""; } catch { /* storage unavailable */ }
        setEmailState(saved.trim() ? saved : (c.email_default_to ?? []).join(", "));
      })
      .catch(() => setOffline(true));
    refreshRuns()
      .then((r) => {
        const wanted = new URLSearchParams(window.location.search).get("report");
        const pick = r.find((x) => x.run_id === wanted) ?? r.find((x) => x.status === "ok" || x.status === "degraded");
        if (pick) return openRun(pick.run_id);
        setLoading(false);
      })
      .catch(() => { setOffline(true); setLoading(false); });

    // Notice runs started elsewhere (another tab, the CLI).
    const timer = window.setInterval(() => { if (!tracking.current) loadConfig().catch(() => setOffline(true)); }, 5000);
    return () => { alive = false; window.clearInterval(timer); };
  }, [refreshRuns, openRun, track]);

  const generate = async (to: string[], filters?: StockFilters, send?: boolean) => {
    try {
      const live = send ?? !(config?.email_dry_run_default ?? true);
      const { run_id } = await api.start({ ...(to.length ? { to } : {}), send: live, ...filters });
      track(run_id, live ? to : [], true);
      refreshRuns();
      return true;
    } catch (e) {
      const msg = (e as Error).message;
      notify({ tone: "error", text: msg.includes("in progress") ? "A report is already being generated. Please wait for it to finish." : msg });
      return false;
    }
  };

  const send = async (to: string[]) => {
    if (!runId || sendInFlight.current) return false;
    sendInFlight.current = true;
    setSending(true);
    try {
      const r = await api.send(runId, to);
      notify({ tone: "ok", text: r.status === "skipped_duplicate"
        ? "This send request already completed."
        : `Report sent to ${r.recipients.join(", ")}.` });
      return true;
    } catch (e) {
      notify({ tone: "error", text: (e as Error).message });
      return false;
    } finally {
      sendInFlight.current = false;
      setSending(false);
    }
  };

  const dismissJob = () => {
    if (job?.status !== "running") setJob(null);
    setShowProgress(false);
  };

  const view = useMemo(() => {
    const fromSteps = viewFromSteps(steps);
    return summary ? { ...fromSteps, ...summary, headlines: summary.headlines ?? fromSteps.headlines } : fromSteps;
  }, [steps, summary]);

  const store: Store = {
    config, offline, runs, runId, view, series, loading, openRun, email, setEmail,
    job, showProgress, setShowProgress, dismissJob, generate, sending, send, toast, notify, dialog, setDialog,
  };
  return <Ctx.Provider value={store}>{children}</Ctx.Provider>;
}
