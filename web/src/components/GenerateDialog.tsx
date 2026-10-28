"use client";

import { useEffect, useRef, useState } from "react";
import { emailError, parseEmails } from "@/lib/report";
import { useReport } from "@/lib/store";
import EmailField from "./EmailField";
import Modal from "./Modal";
import { Button, Icon, Spinner } from "./ui";

const JEV_STATUS: Record<string, string> = {
  configured: "Headline sentiment: TypeSafe Jev (falls back to the chat model if a request fails)",
  auth_failed: "Headline sentiment: chat model fallback (Jev rejected the API key on its last request; check TYPESAFE_API_KEY)",
  disabled: "Headline sentiment: chat model (Jev disabled in settings)",
  sdk_missing: "Headline sentiment: chat model (install the jev extra to use TypeSafe Jev)",
  key_missing: "Headline sentiment: chat model (set TYPESAFE_API_KEY to use TypeSafe Jev)",
};

const STEPS = ["Screen today's NASDAQ gainers", "Analyse 5 days of prices vs the market", "Read the news and score sentiment", "Write and fact-check the report"];

export default function GenerateDialog({ onClose }: { onClose: () => void }) {
  const { email, setEmail, generate, config } = useReport();
  const [previewOnly, setPreviewOnly] = useState(config?.email_dry_run_default ?? false);
  const [touched, setTouched] = useState(false);
  const [starting, setStarting] = useState(false);
  const [excludePenny, setExcludePenny] = useState(true);
  const [excludeLowVolume, setExcludeLowVolume] = useState(true);
  const input = useRef<HTMLInputElement>(null);
  const recipients = parseEmails(email);
  const error = emailError(recipients, !previewOnly);

  useEffect(() => { input.current?.focus(); }, []);

  const start = async () => {
    setTouched(true);
    if (error) return;
    setStarting(true);
    const ok = await generate(recipients, { exclude_penny: excludePenny, exclude_low_volume: excludeLowVolume }, !previewOnly);
    setStarting(false);
    if (ok) onClose();
  };

  return (
    <Modal
      title="Run AI analysis"
      description="Find the latest session's top NASDAQ common-stock gainer within your selected filters."
      icon={<span className="ai-gradient flex h-11 w-11 items-center justify-center rounded-2xl text-white"><Icon name="sparkles" className="h-5 w-5" /></span>}
      onClose={onClose}
      footer={<>
        <Button variant="secondary" onClick={onClose}>Cancel</Button>
        <Button variant="ai" onClick={start} disabled={starting || (!previewOnly && config?.email_enabled === false)}>{starting ? <><Spinner /> Starting…</> : <><Icon name="sparkles" /> {previewOnly ? "Generate preview" : "Generate & email"}</>}</Button>
      </>}
    >
      <ul className="mb-6 grid gap-2 rounded-2xl bg-canvas px-4 py-3.5 text-[13px] text-muted">
        {STEPS.map((s) => (
          <li key={s} className="flex items-center gap-2.5"><Icon name="check" className="h-4 w-4 text-accent" />{s}</li>
        ))}
        {config?.jev && <li className="pl-6.5 text-xs text-faint">{JEV_STATUS[config.jev.state] ?? config.jev.state}</li>}
      </ul>
      <fieldset className="mb-6 space-y-3 rounded-2xl border border-line p-4" disabled={starting}>
        <legend className="px-1 text-sm font-medium">Stock filters</legend>
        <label className="flex items-center gap-3 text-sm">
          <input type="checkbox" checked={excludePenny} onChange={(e) => setExcludePenny(e.target.checked)} className="h-4 w-4 accent-blue-600" />
          Exclude stocks priced below $1
        </label>
        <label className="flex items-center gap-3 text-sm">
          <input type="checkbox" checked={excludeLowVolume} onChange={(e) => setExcludeLowVolume(e.target.checked)} className="h-4 w-4 accent-blue-600" />
          Exclude stocks with daily volume below 100,000 shares
        </label>
        <p className="text-xs leading-relaxed text-muted">{excludePenny || excludeLowVolume
          ? "The report identifies the top gainer within these filters."
          : "The report ranks all NASDAQ common stocks, including low-price and low-volume stocks."}</p>
      </fieldset>
      <label className="mb-4 flex items-center gap-3 text-sm">
        <input type="checkbox" checked={previewOnly} onChange={(e) => setPreviewOnly(e.target.checked)} disabled={starting} className="h-4 w-4 accent-blue-600" />
        Preview only — do not send email
      </label>
      <EmailField ref={input} id="gen-email" label={previewOnly ? "Recipients (optional for preview)" : "Email recipients"} value={email} onChange={setEmail}
        error={touched ? error : null} disabled={starting} />
    </Modal>
  );
}
