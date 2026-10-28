import type { Json, Payload } from "./api";

/** One row in the run timeline, built from live SSE events or a finished run's trace. */
export interface Step {
  key: string;
  kind: "plan" | "thought" | "call" | "result" | "guard" | "stop" | "fallback" | "error" | "start" | "finish";
  tool?: string;
  ok?: boolean | null;
  title: string;
  detail?: string;
  args?: Json;
  data?: Json;
  error?: { type?: string; message?: string; hint?: string; retryable?: boolean };
  warnings?: string[];
  elapsedMs?: number;
}

const TOOL_LABEL: Record<string, string> = {
  record_plan: "Plan",
  get_top_nasdaq_gainers: "Screen top gainers",
  get_price_history: "Fetch & validate prices",
  execute_python: "Run generated analysis code",
  get_news: "Fetch news",
  analyze_sentiment: "Score sentiment (Jev)",
  send_email: "Send email",
};

export const toolLabel = (tool?: string) => (tool ? TOOL_LABEL[tool] ?? tool : "");

const pct = (v: unknown, digits = 2) =>
  typeof v === "number" ? `${v >= 0 ? "+" : ""}${v.toFixed(digits)}%` : "–";
const frac = (v: unknown) => (typeof v === "number" ? pct(v * 100) : "–");

/** Short human summary of a successful tool result. */
export function resultSummary(tool: string | undefined, data: Json | undefined): string {
  if (!data) return "";
  const d = data as Payload;
  switch (tool) {
    case "record_plan":
      return `Plan v${d.plan_version} · ${(d.steps ?? []).length} steps`;
    case "get_top_nasdaq_gainers": {
      const top = d.candidates?.[0];
      return `${d.trade_date} via ${d.source} · raw #1 ${d.raw_top?.symbol} ${pct(d.raw_top?.pct_change)} · top candidate ${top?.symbol} ${pct(top?.pct_change)}`;
    }
    case "get_price_history":
      return `${(d.datasets ?? []).map((x: Json) => x.dataset_id).join(", ")} · ${d.checks?.price_confirmation ?? d.checks?.cross_source ?? ""}`;
    case "execute_python": {
      const a = d.analysis ?? {};
      return `cumulative ${frac(a.cumulative_return)} · trend ${a.trend} · excess vs S&P ${frac(a.excess_return)} · verified`;
    }
    case "get_news":
      return `${d.count} headlines`;
    case "analyze_sentiment":
      return `${d.label} (${typeof d.score === "number" ? d.score.toFixed(1) : "–"}/4) · catalyst ${d.catalyst ?? "–"} · via ${d.method}`;
    case "send_email":
      return `${d.status} → ${(d.recipients ?? []).join(", ")}`;
    default:
      return "";
  }
}

let seq = 0;
const k = () => `s${++seq}`;

/** Live event from the API's SSE stream -> timeline steps. */
export function fromEvent(e: Json): Step[] {
  const ev = e as Payload;
  switch (ev.type) {
    case "run_started":
      return [{ key: k(), kind: "start", title: `Run started · ${ev.models}`, detail: ev.dry_run ? "Dry run: email saved, not sent" : `Live: email to ${ev.recipients?.join(", ")}` }];
    case "thought":
      return [{ key: k(), kind: "thought", title: ev.text }];
    case "tool_call":
      return [{ key: ev.id ?? k(), kind: "call", tool: ev.tool, title: toolLabel(ev.tool), args: ev.args }];
    case "tool_result":
      return [{
        key: `${ev.id ?? k()}-r`, kind: "result", tool: ev.tool, ok: ev.ok, title: toolLabel(ev.tool),
        detail: ev.ok ? resultSummary(ev.tool, ev.data) : undefined, data: ev.data, error: ev.error,
        warnings: ev.warnings, elapsedMs: ev.elapsed_ms,
      }];
    case "plan":
      return [{ key: k(), kind: "plan", title: `Plan v${ev.plan.version}${ev.plan.reason ? ` · revised: ${ev.plan.reason}` : ""}`, data: ev.plan }];
    case "guard":
      return [{ key: k(), kind: "guard", ok: ev.passed, title: ev.passed ? "Pre-send guard passed" : "Pre-send guard rejected the email", detail: ev.passed ? `checked with ${ev.method}` : (ev.problems ?? []).join(" | ") }];
    case "stop":
      return [{ key: k(), kind: "stop", title: ev.reason }];
    case "fallback":
      return [{ key: k(), kind: "fallback", title: `Fallback email: ${ev.email?.status}`, detail: ev.reason }];
    case "error":
      return [{ key: k(), kind: "error", ok: false, title: ev.message }];
    case "run_finished":
      return [{ key: k(), kind: "finish", ok: ev.summary?.status !== "failed", title: `Finished · ${ev.summary?.status} in ${ev.summary?.elapsed_s}s` }];
    default:
      return [];
  }
}

/** Finished run's trace.jsonl entries -> timeline steps. */
export function fromTrace(entries: Json[]): Step[] {
  const steps: Step[] = [];
  for (const raw of entries) {
    const t = raw as Payload;
    if (t.event === "run_started") steps.push({ key: k(), kind: "start", title: `Run started · ${t.models}` });
    else if (t.event === "llm") {
      if (t.content) steps.push({ key: k(), kind: "thought", title: t.content });
      for (const c of t.tool_calls ?? []) steps.push({ key: k(), kind: "call", tool: c.name, title: toolLabel(c.name), args: c.args });
    } else if (t.event === "tool_result") {
      steps.push({
        key: k(), kind: "result", tool: t.tool, ok: t.ok, title: toolLabel(t.tool), error: t.error ?? undefined,
        warnings: t.warnings ?? [], elapsedMs: t.elapsed_ms, data: t.data ?? undefined,
        detail: t.ok ? resultSummary(t.tool, t.data) || "completed" : undefined,
      });
    } else if (t.event === "guard") {
      steps.push({ key: k(), kind: "guard", ok: t.passed, title: t.passed ? "Pre-send guard passed" : "Pre-send guard rejected the email", detail: t.passed ? `checked with ${t.method}` : (t.problems ?? []).join(" | ") });
    } else if (t.event === "run_finished") {
      steps.push({ key: k(), kind: "finish", ok: t.status !== "failed", title: `Finished · ${t.status} in ${t.elapsed_s}s` });
    }
  }
  return steps;
}
