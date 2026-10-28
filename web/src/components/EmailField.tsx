"use client";

import { forwardRef, useEffect, useRef } from "react";
import { Icon } from "./ui";

const EmailField = forwardRef<HTMLInputElement, {
  id: string; label: string; value: string; onChange: (v: string) => void; error?: string | null; hint?: string; disabled?: boolean;
}>(
  function EmailField({ id, label, value, onChange, error, hint, disabled }, ref) {
    const rows = value.split(/[,;\n]/).map((address) => address.trim());
    const inputs = useRef<(HTMLInputElement | null)[]>([]);
    const focusAdded = useRef(false);

    useEffect(() => {
      if (focusAdded.current) {
        inputs.current[rows.length - 1]?.focus();
        focusAdded.current = false;
      }
    }, [value, rows.length]);

    const update = (index: number, address: string) => {
      onChange(rows.map((row, i) => i === index ? address : row).join(","));
    };

    const add = () => {
      if (rows.length >= 5 || !rows[rows.length - 1]) return;
      focusAdded.current = true;
      onChange([...rows, ""].join(","));
    };

    return (
      <fieldset disabled={disabled} className="min-w-0">
        <div className="mb-2.5 flex items-center justify-between">
          <label htmlFor={id} className="text-[13px] font-medium text-ink">{label}</label>
          <span className="text-xs tabular-nums text-faint">{rows.filter(Boolean).length}/5</span>
        </div>
        <div className="space-y-2">
          {rows.map((address, index) => (
            <div key={index} className="flex items-center gap-2">
              <div className="relative min-w-0 flex-1">
                <Icon name="mail" className="pointer-events-none absolute top-1/2 left-3.5 h-4 w-4 -translate-y-1/2 text-faint" />
                <input
                  ref={(node) => {
                    inputs.current[index] = node;
                    if (index === 0) {
                      if (typeof ref === "function") ref(node);
                      else if (ref) ref.current = node;
                    }
                  }}
                  id={index === 0 ? id : `${id}-${index}`}
                  aria-label={index === 0 ? label : `Recipient ${index + 1}`}
                  type="text"
                  inputMode="email"
                  autoComplete="email"
                  placeholder="name@company.com"
                  value={address}
                  onChange={(e) => update(index, e.target.value)}
                  onPaste={(e) => {
                    const pasted = e.clipboardData.getData("text").split(/[\s,;]+/).filter(Boolean);
                    if (pasted.length > 1) {
                      e.preventDefault();
                      onChange([...rows.slice(0, index), ...pasted, ...rows.slice(index + 1)].join(","));
                    }
                  }}
                  onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); add(); } }}
                  aria-invalid={!!error}
                  aria-describedby={error || hint ? `${id}-feedback` : undefined}
                  className={`h-11 w-full rounded-xl border bg-canvas/60 pr-3 pl-10 text-sm outline-none transition placeholder:text-faint focus:bg-white focus:ring-4 ${
                    error ? "border-loss/50 focus:ring-loss-soft" : "border-line focus:border-accent focus:ring-accent-soft"}`}
                />
              </div>
              {rows.length > 1 && (
                <button type="button" onClick={() => onChange(rows.filter((_, i) => i !== index).join(","))}
                  aria-label={`Remove recipient ${index + 1}`}
                  className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg text-faint transition hover:bg-loss-soft hover:text-loss">
                  <Icon name="x" className="h-4 w-4" />
                </button>
              )}
            </div>
          ))}
        </div>
        {rows.length < 5 && (
          <button type="button" onClick={add} disabled={!rows[rows.length - 1]}
            className="mt-3 inline-flex items-center gap-1.5 text-[13px] font-medium text-accent hover:underline disabled:cursor-default disabled:text-faint disabled:no-underline">
            <span aria-hidden="true" className="text-base leading-none">+</span> Add recipient
          </button>
        )}
        {(error || hint) && <p id={`${id}-feedback`} role={error ? "alert" : undefined} className={`mt-2 text-xs ${error ? "text-loss" : "text-faint"}`}>{error ?? hint}</p>}
      </fieldset>
    );
  },
);

export default EmailField;
