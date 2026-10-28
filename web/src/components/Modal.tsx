"use client";

import { useEffect, type ReactNode } from "react";
import { Icon } from "./ui";

export default function Modal({ title, description, icon, onClose, footer, children }: {
  title: string; description?: ReactNode; icon?: ReactNode; onClose: () => void; footer: ReactNode; children?: ReactNode;
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="modal-overlay fixed inset-0 z-40 flex items-start justify-center overflow-y-auto bg-ink/30 px-4 py-[8vh] backdrop-blur-sm" onMouseDown={onClose}>
      <div role="dialog" aria-modal="true" aria-labelledby="modal-title" onMouseDown={(e) => e.stopPropagation()}
        className="research-modal w-full max-w-[460px] rounded-xl border border-line bg-white shadow-[0_24px_64px_-12px_rgba(20,40,38,0.2)]">
        <div className="relative px-7 pt-7">
          <button type="button" onClick={onClose} className="absolute top-5 right-5 flex h-8 w-8 items-center justify-center rounded-lg text-faint hover:bg-canvas hover:text-ink" aria-label="Close">
            <Icon name="x" />
          </button>
          {icon}
          <h2 id="modal-title" className={`${icon ? "mt-4 " : ""}text-[19px] font-semibold tracking-tight`}>{title}</h2>
          {description && <p className="mt-1.5 text-sm leading-relaxed text-muted">{description}</p>}
          {children && <div className="mt-6">{children}</div>}
        </div>
        <div className="mt-7 flex justify-end gap-2 px-7 pb-7">{footer}</div>
      </div>
    </div>
  );
}
