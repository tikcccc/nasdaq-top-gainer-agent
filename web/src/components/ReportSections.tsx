"use client";

import { useState, type ReactNode } from "react";
import type { Bar, Payload, RunListItem } from "@/lib/api";
import { flagLabel, sourceLabel, type ReportView } from "@/lib/report";
import EChart from "./EChart";
import { sentimentOption, sparkOption } from "./charts";
import { Badge, Card, Empty, Icon, Kpi, deliveryStatus, fmtCompact, fmtDate, fmtPct, fmtPctUnits, fmtRunTime, runStatus } from "./ui";

const stripCitations = (s: string) => s.replace(/\s*\[[a-z]\d+\]/g, "");

/** First sentence of the commentary, without its bold lead-in. */
export function takeaway(text?: string): string | null {
  if (!text) return null;
  const first = text.split(/\n+/).map((l) => l.trim()).find((l) => l && !l.startsWith("- "));
  if (!first) return null;
  const body = stripCitations(first.replace(/^\*\*[^*]+\*\*\s*/, ""));
  return body.match(/^.+?[.!?](?=\s+[A-Z]|$)/)?.[0] ?? body;
}

// Mirrors UNVERIFIED_FLAGS in nasdaq_agent/infra/schemas.py.
const UNVERIFIED = ["data_conflict", "possible_split_artifact", "stale", "source_conflict", "ranking_unverified", "eligibility_unconfirmed", "unverified_quote"];

/** Short, user-facing statements of what this report could not confirm. */
export function reportStatus(v: ReportView): { unverified: boolean; issues: string[]; degraded: string | null } {
  const flags = v.selected?.flags ?? [];
  const checks = (v.checks ?? {}) as Payload;
  const issues: string[] = [];
  if (v.ranking?.status === "uncertain" || flags.includes("ranking_unverified")) issues.push("Ranking not confirmed");
  if (v.ranking?.status === "provisional_intraday") issues.push("Intraday, provisional");
  if (flags.includes("eligibility_unconfirmed")) issues.push("Filter eligibility unconfirmed");
  if (flags.includes("unverified_quote")) issues.push("Quote not verified");
  if (checks.price_confirmation?.status === "single_source") issues.push("Price from one vendor");
  if (checks.price_confirmation?.status === "conflict" || flags.includes("source_conflict")) issues.push("Vendors disagree on price");
  if (flags.includes("data_conflict")) issues.push("Quote and history disagree");
  if (flags.includes("possible_split_artifact")) issues.push("Possible split artefact");
  if (flags.includes("stale")) issues.push("Stale quote");
  const degraded = (v.status === "degraded" || v.email?.fallback) && !v.analysis
    ? "Multi-day analysis is unavailable for this report." : null;
  return { unverified: flags.some((f) => UNVERIFIED.includes(f)) || v.ranking?.status === "uncertain", issues, degraded };
}

export function Hero({ v, bars }: { v: ReportView; bars?: Bar[] }) {
  const sel = v.selected;
  if (!sel?.symbol) return <Card><Empty>No report yet. Run an AI analysis to create one.</Empty></Card>;
  const summary = takeaway(v.email?.narrative);
  const status = reportStatus(v);
  return (
    <section className="stock-hero overflow-hidden rounded-xl border border-line">
      <div className="hero-header"><span className="eyebrow"><span className="status-dot" /> SESSION SPOTLIGHT</span><span>NASDAQ · {fmtDate(v.trade_date, { month: "short", day: "numeric", year: "numeric" })}</span></div>
      <div className="grid gap-7 p-5 sm:p-7 xl:grid-cols-[minmax(0,1fr)_280px]">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <Badge tone={status.unverified ? "amber" : "blue"} dot>{v.selection?.primary_symbol && v.selection.primary_symbol !== sel.symbol
              ? "Supplementary analysis" : status.unverified ? "Unverified NASDAQ gainer candidate" : v.selection?.label ?? "Top NASDAQ gainer"}</Badge>
            <span className="text-[11px] text-muted">COMMON STOCK</span>
          </div>
          {(status.issues.length > 0 || status.degraded) && (
            <div className="mt-3 flex flex-wrap gap-1.5" aria-label="Report status">
              {status.degraded && <Badge tone="amber">Limited analysis</Badge>}
              {status.issues.map((i) => <Badge key={i} tone="amber">{i}</Badge>)}
            </div>
          )}
          {status.degraded && <p className="mt-2 text-xs leading-relaxed text-muted">{status.degraded}</p>}
          <div className="mt-4 flex flex-wrap items-end gap-x-4 gap-y-1">
            <h2 className="stock-symbol">{sel.symbol}</h2>
            <span className="pb-1 text-[13px] text-muted">{sel.name}</span>
          </div>
          {v.selection && <p className="mt-3 text-xs text-muted">Selection: common stocks
            {!!v.selection.filters.min_price && ` · price ≥ $${v.selection.filters.min_price}`}
            {!!v.selection.filters.min_volume && ` · daily volume ≥ ${v.selection.filters.min_volume.toLocaleString()} shares`}
          </p>}
          <div className="mt-4 flex flex-wrap items-center gap-3">
            <span className="num text-[28px] leading-none font-medium tracking-tight">{typeof sel.price === "number" ? `$${sel.price.toFixed(2)}` : "–"}</span>
            <span className="num inline-flex items-center gap-1 rounded-md bg-gain-soft px-2.5 py-1 text-[13px] font-semibold text-gain">↗ {fmtPctUnits(sel.pct_change)}</span>
            <span className="text-[13px] text-faint">Volume {fmtCompact(sel.volume)}</span>
          </div>
          {summary && (
            <div className="research-takeaway">
              <span className="mt-0.5 shrink-0 text-accent"><Icon name="sparkles" className="h-4 w-4" /></span>
              <div>
                <div className="text-[10px] font-semibold tracking-[0.12em] text-accent uppercase">Research takeaway</div>
                <p className="mt-1.5 text-[13px] leading-[1.8] text-ink">{summary}</p>
                {!!v.email?.key_findings?.length && (
                  <>
                    <div className="mt-4 text-[10px] font-semibold tracking-[0.12em] text-accent uppercase">
                      Key findings
                    </div>
                    <ol className="mt-1.5 list-decimal space-y-1.5 pl-5 text-[13px] leading-[1.7] text-ink">
                      {v.email.key_findings.map((f) => <li key={`${f.kind}-${f.ref ?? ""}`}>{stripCitations(f.text)}</li>)}
                    </ol>
                  </>
                )}
              </div>
            </div>
          )}
        </div>
        {bars && bars.length > 1 && (
          <div className="spark-panel">
            <div className="flex items-baseline justify-between">
              <span className="text-[12px] font-medium text-muted">{bars.length - 1}-day price</span>
              <span className={`num text-sm font-semibold ${v.analysis && v.analysis.cumulative_return < 0 ? "text-loss" : "text-gain"}`}>{fmtPct(v.analysis?.cumulative_return)}</span>
            </div>
            <EChart option={sparkOption(bars)} height={130} />
            <div className="flex justify-between text-[11px] text-faint">
              <span>{fmtDate(bars[0].date, { month: "short", day: "numeric" })}</span>
              <span>{fmtDate(bars[bars.length - 1].date, { month: "short", day: "numeric" })}</span>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}

export function KpiRow({ v }: { v: ReportView }) {
  const a = v.analysis;
  return (
    <div className="metrics-grid">
      <Kpi label={`${a?.days ?? 5}-day return`} value={fmtPct(a?.cumulative_return)} tone={a ? (a.cumulative_return >= 0 ? "up" : "down") : undefined} hint={a ? `Trend ${a.trend}` : undefined} />
      <Kpi label="vs S&P 500" value={fmtPct(a?.excess_return)} tone={a?.excess_return != null ? (a.excess_return >= 0 ? "up" : "down") : undefined} hint={a ? `Index ${fmtPct(a.benchmark?.GSPC)}` : undefined} />
      <Kpi label="Volatility (ann.)" value={a ? fmtPct(a.volatility_annualized, 0).replace("+", "") : "–"} hint={a ? `Daily ${fmtPct(a.volatility_daily).replace("+", "")}` : undefined} />
      <Kpi label="Max drawdown" value={fmtPct(a?.max_drawdown)} tone={a ? "down" : undefined} hint="Peak to trough" />
    </div>
  );
}

/** **bold**, "- " bullets and paragraphs; source markers like [d1][n2] are dropped. */
function renderNarrative(text: string): ReactNode[] {
  const inline = (s: string) =>
    stripCitations(s).split(/(\*\*[^*]+\*\*)/g).map((part, i) =>
      part.startsWith("**") ? <strong key={i} className="font-semibold text-ink">{part.slice(2, -2)}</strong> : part);
  const out: ReactNode[] = [];
  let bullets: string[] = [];
  const flush = () => {
    if (bullets.length) out.push(<ul key={`u${out.length}`} className="space-y-2">{bullets.map((b, i) => (
      <li key={i} className="flex gap-3"><span className="mt-2.5 h-1.5 w-1.5 shrink-0 rounded-full bg-accent" /><span>{inline(b)}</span></li>
    ))}</ul>);
    bullets = [];
  };
  for (const line of text.split(/\n+/)) {
    const t = line.trim();
    if (!t) continue;
    if (t.startsWith("- ")) bullets.push(t.slice(2));
    else { flush(); out.push(<p key={`p${out.length}`}>{inline(t)}</p>); }
  }
  flush();
  return out;
}

export function Commentary({ v }: { v: ReportView }) {
  const text = v.email?.narrative;
  return (
    <Card title="Analyst commentary" subtitle="AI research with figures checked against the source data." className="h-full">
      {text ? <div className="space-y-4 text-[14px] leading-[1.9] text-muted">{renderNarrative(text)}</div> : <Empty>Commentary will appear once the report is complete.</Empty>}
    </Card>
  );
}

const CATALYST: Record<string, string> = { corporate_action: "Corporate action", earnings: "Earnings", clinical: "Clinical / regulatory", m_and_a: "M&A", offering: "Offering", none: "None identified" };

function scoredBy(method: string | undefined): string | null {
  return method === "jev" || method === "llm" ? "Model estimate based on the available headlines" : null;
}

export function Sentiment({ v }: { v: ReportView }) {
  const s = v.sentiment;
  const headlines = Object.values(v.headlines ?? {});
  const scorer = scoredBy(s?.method);
  return (
    <div className="space-y-5">
      <Card title="News sentiment" subtitle={headlines.length ? `Based on ${headlines.length} headline${headlines.length > 1 ? "s" : ""}` : undefined}>
        {typeof s?.score === "number" ? (
          <>
            <EChart option={sentimentOption(s.score)} height={150} />
            <div className="-mt-1 text-center">
              <Badge tone={s.uncertain ? "amber" : s.score >= 2.5 ? "green" : s.score < 1.5 ? "red" : "slate"}>{s.label}</Badge>
              {typeof s.confidence === "number" && <span className="ml-2 text-xs text-muted">{(s.confidence * 100).toFixed(0)}% confidence</span>}
              {scorer && <p className="mt-1.5 text-xs text-muted">{scorer}</p>}
            </div>
            <dl className="mt-5 grid grid-cols-2 gap-3 text-[13px]">
              <div className="rounded-xl bg-canvas px-3 py-2.5"><dt className="text-faint">Catalyst</dt><dd className="mt-0.5 font-medium">{s.catalyst ? CATALYST[s.catalyst] ?? s.catalyst.replaceAll("_", " ") : "–"}</dd></div>
              <div className="rounded-xl bg-canvas px-3 py-2.5"><dt className="text-faint">Dilution risk</dt><dd className="num mt-0.5 font-medium">{typeof s.dilution_risk === "number" ? `${(s.dilution_risk * 100).toFixed(0)}%` : "–"}</dd></div>
            </dl>
          </>
        ) : s ? <Badge tone="amber">{s.label}</Badge> : <Empty>No sentiment yet.</Empty>}
      </Card>
      <Card title="Headlines">
        {headlines.length ? (
          <ul className="space-y-4">
            {headlines.map((h) => (
              <li key={h.id}>
                <div className="text-[12px] text-faint">{h.publisher}{h.published && <> · {fmtDate(h.published, { month: "short", day: "numeric" })}</>}</div>
                {h.link ? <a href={h.link} target="_blank" rel="noreferrer" className="mt-0.5 block text-sm leading-snug font-medium text-ink hover:text-accent">{h.title}</a>
                  : <span className="mt-0.5 block text-sm leading-snug font-medium">{h.title}</span>}
              </li>
            ))}
          </ul>
        ) : <Empty>No news yet.</Empty>}
      </Card>
    </div>
  );
}

export function Candidates({ v }: { v: ReportView }) {
  const rows = (v.candidates ?? []).slice(0, 8);
  const excluded = v.raw_top?.symbol && v.raw_top.symbol !== v.selected?.symbol ? v.raw_top : null;
  return (
    <Card title="Screener ranking" subtitle={v.selection?.label ?? "Top NASDAQ common-stock gainers for the session"} bodyClassName="pb-2">
      {excluded && (
        <p className="mx-6 mb-3 rounded-xl bg-canvas px-4 py-3 text-[13px] text-muted">
          The highest raw mover, <b className="font-semibold text-ink">{excluded.symbol}</b> ({fmtPctUnits(excluded.pct_change, 0)}), differs from the stock analysed under the selected filters
          {(excluded.flags ?? []).length > 0 && <> ({excluded.flags!.map(flagLabel).join(", ").toLowerCase()})</>}.
        </p>
      )}
      {rows.length ? (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[600px] text-sm">
            <thead>
              <tr className="text-left text-[12px] font-medium text-faint">
                <th className="py-2.5 pr-2 pl-6 font-medium">#</th>
                <th className="px-2 font-medium">Company</th>
                <th className="px-2 text-right font-medium">Change</th>
                <th className="px-2 text-right font-medium">Price</th>
                <th className="px-2 text-right font-medium">Volume</th>
                <th className="px-2 pr-6 text-right font-medium">Market cap</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((c) => {
                const picked = c.symbol === v.selected?.symbol;
                return (
                  <tr key={c.symbol} className={`border-t border-line ${picked ? "bg-accent-soft/60" : ""}`}>
                    <td className="num py-3 pr-2 pl-6 text-faint">{c.rank}</td>
                    <td className="px-2">
                      <div className="flex items-center gap-2">
                        <span className="font-semibold">{c.symbol}</span>
                        {picked && <Badge tone="blue">Selected</Badge>}
                        {c.flags.map((f) => <Badge key={f} tone="amber">{flagLabel(f)}</Badge>)}
                      </div>
                      <div className="max-w-[260px] truncate text-xs text-faint">{c.name}</div>
                    </td>
                    <td className="num px-2 text-right font-semibold text-gain">{fmtPctUnits(c.pct_change)}</td>
                    <td className="num px-2 text-right">${c.price.toFixed(2)}</td>
                    <td className="num px-2 text-right text-muted">{fmtCompact(c.volume)}</td>
                    <td className="num px-2 pr-6 text-right text-muted">{c.market_cap ? `$${fmtCompact(c.market_cap)}` : "–"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : <div className="px-6 pb-4"><Empty>No screener data.</Empty></div>}
    </Card>
  );
}

const PAGE_SIZE = 8;

export function History({ runs, current, onOpen }: { runs: RunListItem[]; current?: string; onOpen: (id: string) => void }) {
  const [page, setPage] = useState(0);
  const pages = Math.max(1, Math.ceil(runs.length / PAGE_SIZE));
  const p = Math.min(page, pages - 1);
  const rows = runs.slice(p * PAGE_SIZE, (p + 1) * PAGE_SIZE);
  if (!runs.length) return <Card><Empty>No reports yet.</Empty></Card>;
  return (
    <div>
      <div className="grid gap-3">
        {rows.map((r) => {
          const st = runStatus(r.status);
          const em = deliveryStatus(r.email);
          const active = r.run_id === current;
          const openable = r.status !== "running" && !!r.symbol;
          return (
            <button key={r.run_id} type="button" disabled={!openable} onClick={() => onOpen(r.run_id)}
              className={`report-row grid grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-4 rounded-lg border bg-white px-5 py-4 text-left transition sm:grid-cols-[auto_minmax(0,1fr)_auto_auto_auto] ${
                active ? "border-accent bg-accent-soft/30" : "border-line enabled:hover:border-line-strong"}`}>
              <span className={`flex h-11 w-11 items-center justify-center rounded-xl text-[12px] font-bold ${r.symbol ? "bg-ink text-white" : "bg-canvas text-faint"}`}>
                {r.status === "running" ? <span className="h-2 w-2 animate-pulse rounded-full bg-accent" /> : (r.symbol ?? "–").slice(0, 4)}
              </span>
              <span className="min-w-0">
                <span className="flex items-center gap-2">
                  <span className="font-semibold">{r.symbol ?? "AI analysis"}</span>
                  {r.pct_change !== undefined && <span className="num text-sm font-semibold text-gain">{fmtPctUnits(r.pct_change)}</span>}
                </span>
                <span className="mt-0.5 block truncate text-[13px] text-muted">
                  {r.trade_date ? `Session ${fmtDate(r.trade_date)} · ` : ""}Generated {fmtRunTime(r.run_id)}
                </span>
              </span>
              <span className="hidden sm:block"><Badge tone={st.tone} dot>{st.label}</Badge></span>
              <span className="hidden sm:block">{r.status !== "running" && <Badge tone={em.tone}>{em.label}</Badge>}</span>
              <span className="text-[13px] font-medium text-accent">{active ? <span className="text-faint">Viewing</span> : openable ? <Icon name="arrow" /> : null}</span>
            </button>
          );
        })}
      </div>
      <div className="mt-5 flex flex-wrap items-center justify-between gap-3 text-[13px] text-muted">
        <span className="num">Showing {p * PAGE_SIZE + 1}–{p * PAGE_SIZE + rows.length} of {runs.length}</span>
        <div className="flex items-center gap-1.5">
          <button type="button" onClick={() => setPage(p - 1)} disabled={p === 0}
            className="h-9 rounded-xl bg-white px-3 font-medium text-ink ring-1 ring-line ring-inset hover:bg-canvas disabled:opacity-40">← Prev</button>
          {Array.from({ length: pages }, (_, i) => (
            <button key={i} type="button" onClick={() => setPage(i)} aria-current={i === p ? "page" : undefined}
              className={`num h-9 min-w-9 rounded-xl px-2 font-medium ${i === p ? "bg-ink text-white" : "text-muted hover:bg-white"}`}>{i + 1}</button>
          ))}
          <button type="button" onClick={() => setPage(p + 1)} disabled={p >= pages - 1}
            className="h-9 rounded-xl bg-white px-3 font-medium text-ink ring-1 ring-line ring-inset hover:bg-canvas disabled:opacity-40">Next →</button>
        </div>
      </div>
    </div>
  );
}

const CONFIRMATION_LABEL: Record<string, string> = { consistent: "Consistent", single_source: "Single source", conflict: "Conflict" };
const ELIGIBILITY_LABEL: Record<string, string> = { confirmed: "Confirmed by all sources", unconfirmed: "Unconfirmed", not_applied: "No price/volume filter" };
const RANKING_LABEL: Record<string, string> = { verified: "Verified", uncertain: "Not confirmed", provisional_intraday: "Intraday, provisional" };

export function Methodology({ v }: { v: ReportView }) {
  const checks = (v.checks ?? {}) as Payload;
  const screener = checks.screener_vs_history ?? checks.cross_source;
  const confirmation = checks.price_confirmation;
  const eligibility = checks.filter_eligibility;
  const ranking = v.ranking;
  const sources =[...new Set(Object.entries(v.sources ?? {}).filter(([k]) => k !== "sentiment").map(([, src]) => sourceLabel(src)))];
  const items = [
    { k: "Data sources", v: sources.length ? sources.join(", ") : "–" },
    { k: "Screener vs history", v: screener?.message ? `${screener.status === "ok" ? "Passed" : "Warning"} · ${screener.message}` : "–" },
    { k: "Ranking", v: ranking?.message ? `${RANKING_LABEL[ranking.status] ?? ranking.status} · ${ranking.message}` : "–" },
    { k: "Price, two vendors", v: confirmation?.message ? `${CONFIRMATION_LABEL[confirmation.status] ?? confirmation.status} · ${confirmation.message}` : "–" },
    { k: "Filter eligibility", v: eligibility?.status ? `${ELIGIBILITY_LABEL[eligibility.status] ?? eligibility.status}${eligibility.failing?.length ? ` · below the filter in ${eligibility.failing.join(", ")}` : ""}` : "–" },
    { k: "Corporate actions", v: Array.isArray(checks.corporate_actions) ? (checks.corporate_actions.join("; ") || "None in window") : "–" },
    { k: "Report figures", v: checks.guard?.passed ? "Recomputed from the verified data" : v.analysis ? "Calculated from the available price data" : "Unavailable" },
  ];
  return (
    <Card title="Data quality">
      <dl className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {items.map((i) => (
          <div key={i.k} className="rounded-xl bg-canvas px-4 py-3">
            <dt className="text-[12px] text-faint">{i.k}</dt>
            <dd className="mt-1 text-[13px] font-medium text-ink">{i.v}</dd>
          </div>
        ))}
      </dl>
      <p className="mt-4 text-xs text-faint">For information only; not investment advice. Prices are split-adjusted daily closes; returns exclude dividends.</p>
    </Card>
  );
}
