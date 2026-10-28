import type { ButtonHTMLAttributes, ReactNode } from "react";

export function Card({ title, subtitle, action, children, className = "", bodyClassName = "px-6 pb-6" }: {
  title?: ReactNode; subtitle?: ReactNode; action?: ReactNode; children: ReactNode; className?: string; bodyClassName?: string;
}) {
  return (
    <section className={`research-panel rounded-xl border border-line bg-white ${className}`}>
      {title ? (
        <header className="flex items-start justify-between gap-3 px-6 pt-5 pb-4">
          <div>
            <h2 className="panel-title text-[15px] font-semibold tracking-tight text-ink">{title}</h2>
            {subtitle && <p className="mt-0.5 text-[13px] text-muted">{subtitle}</p>}
          </div>
          {action}
        </header>
      ) : <div className="pt-6" />}
      <div className={bodyClassName}>{children}</div>
    </section>
  );
}

const TONES = {
  slate: "bg-slate-100 text-slate-600",
  green: "bg-gain-soft text-gain",
  red: "bg-loss-soft text-loss",
  amber: "bg-amber-50 text-amber-700",
  blue: "bg-accent-soft text-accent",
} as const;

export type Tone = keyof typeof TONES;

export function Badge({ tone = "slate", dot, children }: { tone?: Tone; dot?: boolean; children: ReactNode }) {
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium whitespace-nowrap ${TONES[tone]}`}>
      {dot && <span className="h-1.5 w-1.5 rounded-full bg-current" />}
      {children}
    </span>
  );
}

export function Button({ variant = "primary", className = "", ...props }: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "primary" | "ai" | "secondary" | "ghost" }) {
  const styles = {
    primary: "bg-ink text-white hover:bg-black focus-visible:outline-ink",
    ai: "ai-gradient text-white hover:brightness-110 focus-visible:outline-accent",
    secondary: "bg-white text-ink ring-1 ring-inset ring-line-strong hover:bg-slate-50 focus-visible:outline-slate-400",
    ghost: "text-accent hover:bg-accent-soft focus-visible:outline-accent",
  }[variant];
  return (
    <button
      type="button"
      {...props}
      className={`inline-flex h-10 items-center justify-center gap-2 rounded-lg px-4 text-[13px] font-medium whitespace-nowrap transition focus-visible:outline-2 focus-visible:outline-offset-2 disabled:cursor-not-allowed disabled:opacity-50 ${styles} ${className}`}
    />
  );
}

export function Spinner({ className = "h-4 w-4" }: { className?: string }) {
  return (
    <svg className={`animate-spin ${className}`} viewBox="0 0 24 24" fill="none" aria-hidden>
      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeOpacity="0.25" strokeWidth="3" />
      <path d="M22 12a10 10 0 0 0-10-10" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
    </svg>
  );
}

/** Small inline icon set (20px grid, stroke). */
const PATHS = {
  sparkles: "M10 2.5l1.6 4.4 4.4 1.6-4.4 1.6L10 14.5l-1.6-4.4L4 8.5l4.4-1.6L10 2.5Zm5.5 10l.7 1.8 1.8.7-1.8.7-.7 1.8-.7-1.8-1.8-.7 1.8-.7.7-1.8Z",
  home: "M3.5 9 10 3.5 16.5 9v7.5a1 1 0 0 1-1 1H12v-5H8v5H4.5a1 1 0 0 1-1-1V9Z",
  chart: "M3.5 16.5h13M6 13.5V9m4 4.5V5.5m4 8V8",
  news: "M4.5 4h9a1 1 0 0 1 1 1v10.5a1.5 1.5 0 0 1-3 0V7H4.5m0-3v11a1.5 1.5 0 0 0 1.5 1.5h9m-8-9h4m-4 3h4",
  folder: "M3 6.5A1.5 1.5 0 0 1 4.5 5H8l1.5 1.5h6A1.5 1.5 0 0 1 17 8v6.5a1.5 1.5 0 0 1-1.5 1.5h-11A1.5 1.5 0 0 1 3 14.5v-8Z",
  mail: "M3 5.5h14v9H3v-9Zm0 0 7 5 7-5",
  check: "M4.5 10.5 8 14l7.5-8",
  arrow: "M5 10h10m-4-4 4 4-4 4",
  eye: "M2.5 10S5 4.5 10 4.5 17.5 10 17.5 10 15 15.5 10 15.5 2.5 10 2.5 10Zm7.5 2.25a2.25 2.25 0 1 0 0-4.5 2.25 2.25 0 0 0 0 4.5Z",
  x: "M5.5 5.5l9 9m0-9-9 9",
} as const;

export function Icon({ name, className = "h-4 w-4" }: { name: keyof typeof PATHS; className?: string }) {
  return (
    <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth={name === "sparkles" ? 0 : 1.6} strokeLinecap="round" strokeLinejoin="round" className={className} aria-hidden>
      <path d={PATHS[name]} fill={name === "sparkles" ? "currentColor" : "none"} />
    </svg>
  );
}

export function Kpi({ label, value, hint, tone }: { label: string; value: ReactNode; hint?: ReactNode; tone?: "up" | "down" }) {
  const color = tone === "up" ? "text-gain" : tone === "down" ? "text-loss" : "text-ink";
  return (
    <div className="metric-card">
      <div className="text-[10px] font-medium tracking-[0.1em] text-muted uppercase">{label}</div>
      <div className={`num mt-4 text-[29px] leading-none font-medium tracking-tight ${color}`}>{value}</div>
      {hint && <div className="mt-2 text-[12px] text-faint">{hint}</div>}
    </div>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="empty-state"><Icon name="chart" className="h-6 w-6" /><p>{children}</p></div>;
}

export const fmtPct = (fraction?: number | null, digits = 2) =>
  typeof fraction === "number" ? `${fraction >= 0 ? "+" : ""}${(fraction * 100).toFixed(digits)}%` : "–";

export const fmtPctUnits = (percent?: number | null, digits = 2) =>
  typeof percent === "number" ? `${percent >= 0 ? "+" : ""}${percent.toFixed(digits)}%` : "–";

export const fmtCompact = (n?: number | null) =>
  typeof n === "number" ? new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 }).format(n) : "–";

export const fmtDate = (iso?: string, opts: Intl.DateTimeFormatOptions = { month: "short", day: "numeric", year: "numeric" }) =>
  iso ? new Date(`${iso.slice(0, 10)}T12:00:00`).toLocaleDateString("en-US", opts) : "–";

/** "20261025_014315_3c5c" -> "Oct 25, 01:43" */
export const fmtRunTime = (runId?: string) => {
  const m = runId && /^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})/.exec(runId);
  return m ? `${fmtDate(`${m[1]}-${m[2]}-${m[3]}`, { month: "short", day: "numeric" })}, ${m[4]}:${m[5]}` : runId ?? "–";
};

/** Run status as a business user reads it. */
export const runStatus = (status?: string): { label: string; tone: Tone } =>
  status === "ok" ? { label: "Completed", tone: "green" }
    : status === "degraded" ? { label: "Completed with notes", tone: "amber" }
    : status === "running" ? { label: "In progress", tone: "blue" }
    : status ? { label: "Failed", tone: "red" } : { label: "–", tone: "slate" };

export const deliveryStatus = (status?: string): { label: string; tone: Tone } =>
  status === "sent" ? { label: "Delivered", tone: "green" }
    : status === "skipped_duplicate" ? { label: "Already delivered", tone: "slate" }
    : status === "dry_run" ? { label: "Not sent", tone: "slate" }
    : status === "failed" ? { label: "Delivery failed", tone: "red" }
    : { label: "–", tone: "slate" };
