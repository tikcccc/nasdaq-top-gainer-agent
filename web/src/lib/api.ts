export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export type Json = Record<string, unknown>;
export interface StockFilters { exclude_penny: boolean; exclude_low_volume: boolean }
export interface Selection { label: string; filtered: boolean; primary_symbol: string; filters: { min_price?: number; min_volume?: number; min_market_cap?: number; exclude_derivatives?: boolean } }

export interface AppConfig {
  days: number;
  email_dry_run_default: boolean;
  email_enabled: boolean;
  email_default_to: string[];
  busy: boolean;
  active_run: string | null;
  jev?: {
    state: "configured" | "auth_failed" | "disabled" | "sdk_missing" | "key_missing";
    sdk_installed: boolean; api_key_configured: boolean; model: string;
    last_request?: { outcome: string; error_type?: string | null; http_status?: number | null; started: number } | null;
  };
}

export interface Bar {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

/** Symbol (without ^) -> daily bars, e.g. {AIOK: [...], GSPC: [...], IXIC: [...]}. */
export type Series = Record<string, Bar[]>;

export interface RunListItem {
  run_id: string;
  status: string;
  symbol?: string;
  pct_change?: number;
  trade_date?: string;
  email?: string;
  elapsed_s?: number;
}

export interface Analysis {
  days: number;
  daily_returns: number[];
  avg_daily_return: number;
  cumulative_return: number;
  trend: string;
  volatility_daily: number;
  volatility_annualized: number;
  max_drawdown: number;
  benchmark: Record<string, number>;
  excess_return: number | null;
  notes?: string | null;
}

export interface Sentiment {
  method: string;
  n_headlines: number;
  score?: number | null;
  label: string;
  confidence?: number | null;
  catalyst?: string | null;
  dilution_risk?: number | null;
  uncertain: boolean;
  headline_ids: string[];
}

export interface Headline {
  id: string;
  title: string;
  publisher: string;
  published?: string | null;
  link?: string | null;
}

export interface Summary {
  run_id: string;
  status: "ok" | "degraded" | "failed";
  elapsed_s: number;
  models: string;
  trade_date?: string;
  raw_top?: { symbol?: string; pct_change?: number; flags?: string[] };
  selected?: { symbol?: string; name?: string; pct_change?: number; price?: number; volume?: number; flags?: string[]; days?: number };
  analysis?: Analysis | null;
  sentiment?: Sentiment | null;
  headlines?: Record<string, Headline> | null;
  email?: {
    status?: string; subject?: string; recipients?: string[]; narrative?: string; fallback?: boolean; error?: string;
    key_findings?: { kind: string; ref?: string | null; text: string }[]; key_findings_source?: "agent" | "default";
  };
  plan?: { version: number; steps: string[]; history?: { version: number; steps: string[]; replaced_because?: string }[] } | null;
  checks?: Json | null;
  sources?: Record<string, string> | null;
  llm_calls?: number;
  tool_attempts?: Record<string, number>;
  tool_failures?: Record<string, number>;
  guard_rejections?: number;
  warnings?: string[];
  errors?: Json[];
  degraded?: string[];
  stop_reason?: string | null;
  charts?: string[];
  selection?: Selection;
  ranking?: { status: "verified" | "uncertain" | "provisional_intraday"; message: string; unresolved?: { symbol: string; reason: string }[] } | null;
}

export interface RunDetail {
  run_id: string;
  running: boolean;
  summary: Summary | null;
  trace: Json[];
  files: string[];
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, { cache: "no-store", ...init });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    const detail = (body as { detail?: unknown }).detail;
    const delivery = detail as { status?: string; error?: string } | undefined;
    throw new Error(typeof detail === "string" ? detail
      : delivery?.status === "delivery_unknown" ? "Delivery could not be confirmed. Check your inbox before sending again."
      : delivery?.error ?? `${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  config: () => request<AppConfig>("/api/config"),
  runs: () => request<RunListItem[]>("/api/runs"),
  run: (id: string) => request<RunDetail>(`/api/runs/${id}`),
  series: (id: string) => request<Series>(`/api/runs/${id}/series`),
  start: (body: { to?: string[]; send?: boolean } & Partial<StockFilters>) =>
    request<{ run_id: string; dry_run: boolean }>("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  send: (id: string, to: string[], requestId = crypto.randomUUID()) =>
    request<{ status: string; recipients: string[]; subject: string }>(`/api/runs/${id}/send`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ to, request_id: requestId }),
    }),
  fileUrl: (id: string, name: string) => `${API_URL}/api/runs/${id}/files/${name}`,
  eventsUrl: (id: string) => `${API_URL}/api/runs/${id}/events`,
};

/** Tool payloads differ per tool and evolve with the backend; they are read defensively. */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type Payload = Record<string, any>;
