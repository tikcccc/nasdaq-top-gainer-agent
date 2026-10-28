import type { Headline, Payload, Summary } from "./api";
import type { Step } from "./timeline";

export type Candidate = { rank: number; symbol: string; name: string; price: number; pct_change: number; volume?: number | null; market_cap?: number | null; flags: string[] };

export interface ReportView extends Partial<Summary> {
  candidates?: Candidate[];
}

/** Build the report progressively from tool results (live events or a recorded trace). */
export function viewFromSteps(steps: Step[]): ReportView {
  const view: ReportView = {};
  for (const s of steps) {
    if (s.kind !== "result" || !s.ok || !s.data) continue;
    const d = s.data as Payload;
    if (s.tool === "get_top_nasdaq_gainers") {
      view.trade_date = d.trade_date;
      view.raw_top = d.raw_top;
      view.candidates = d.candidates;
      view.selection = d.selection;
      view.ranking = d.ranking;
    } else if (s.tool === "get_price_history") {
      const c = view.candidates?.find((x) => x.symbol === d.selected?.symbol);
      view.selected = { ...d.selected, name: c?.name, price: c?.price, volume: c?.volume ?? undefined, flags: d.flags };
    } else if (s.tool === "execute_python") {
      view.analysis = d.analysis;
    } else if (s.tool === "get_news") {
      view.headlines = Object.fromEntries((d.untrusted_headlines ?? []).map((h: Headline) => [h.id, h]));
    } else if (s.tool === "analyze_sentiment") {
      view.sentiment = d as Summary["sentiment"];
    } else if (s.tool === "send_email") {
      view.email = d;
    }
  }
  return view;
}

export const STAGES = [
  { id: "screen", label: "Screening the market", tools: ["get_top_nasdaq_gainers"] },
  { id: "prices", label: "Retrieving price history", tools: ["get_price_history"] },
  { id: "analysis", label: "Quantitative analysis", tools: ["execute_python"] },
  { id: "news", label: "News & sentiment", tools: ["analyze_sentiment"] },
  { id: "review", label: "Quality review", tools: [] },
  { id: "deliver", label: "Delivering report", tools: ["send_email"] },
] as const;

/** Number of completed stages, from the steps seen so far. */
export function stagesDone(steps: Step[]): number {
  const done = new Set<string>();
  for (const s of steps) {
    if (s.kind === "result" && s.ok) {
      for (const st of STAGES) if ((st.tools as readonly string[]).includes(s.tool ?? "")) done.add(st.id);
    }
    if (s.kind === "guard" && s.ok) done.add("review");
    if (s.kind === "fallback" || s.kind === "finish") STAGES.forEach((st) => done.add(st.id));
  }
  let n = 0;
  while (n < STAGES.length && done.has(STAGES[n].id)) n++;
  return n;
}

const FLAG_LABEL: Record<string, string> = {
  derivative: "Warrant / right / unit",
  penny: "Sub-$1",
  low_volume: "Low volume",
  micro_cap: "Micro-cap",
  data_conflict: "Data conflict",
  source_conflict: "Vendors disagree",
  eligibility_unconfirmed: "Eligibility unconfirmed",
  unverified_quote: "Quote unverified",
  ranking_unverified: "Ranking unconfirmed",
  near_threshold: "Near filter threshold",
};

export const flagLabel = (f: string) => FLAG_LABEL[f] ?? f.replaceAll("_", " ");

const SOURCE_LABEL: Record<string, string> = { yahoo: "Yahoo Finance", nasdaq: "Nasdaq.com", google: "Google News", jev: "TypeSafe Jev" };
export const sourceLabel = (s: string) => SOURCE_LABEL[s] ?? s;

const EMAIL = /^[^@\s,;<>]+@[^@\s,;<>]+\.[A-Za-z]{2,}$/;

export const parseEmails = (raw: string) => [...new Map(raw.split(/[\s,;]+/).map((s) => s.trim()).filter(Boolean).map((s) => [s.toLowerCase(), s])).values()];

/** Validation message for a recipient list, or null when it is fine. */
export function emailError(recipients: string[], required = true): string | null {
  if (!recipients.length) return required ? "Enter at least one email address." : null;
  const invalid = recipients.filter((r) => !EMAIL.test(r));
  if (invalid.length) return `Invalid address: ${invalid.join(", ")}`;
  return recipients.length > 5 ? "Up to 5 recipients." : null;
}
