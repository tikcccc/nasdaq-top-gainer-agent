"use client";

import { useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";
import { emailError, parseEmails } from "@/lib/report";
import { useReport } from "@/lib/store";
import EmailField from "./EmailField";
import Modal from "./Modal";
import { Button, Icon, Spinner, fmtDate } from "./ui";

export default function EmailDialog({ onClose }: { onClose: () => void }) {
  const { email, setEmail, send, sending, runId, view } = useReport();
  const [touched, setTouched] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const recipients = parseEmails(email);
  const error = emailError(recipients);

  useEffect(() => { input.current?.focus(); }, []);

  const submit = async () => {
    setTouched(true);
    if (!error && await send(recipients)) onClose();
  };

  return (
    <Modal
      title="Send report"
      description={<span className="inline-flex items-center gap-2"><b className="rounded-md bg-accent-soft px-2 py-0.5 text-xs font-semibold text-accent">{view.selected?.symbol}</b><span className="text-xs">{fmtDate(view.trade_date)}</span></span>}
      onClose={() => { if (!sending) onClose(); }}
      footer={<>
        {runId && (
          <a href={api.fileUrl(runId, "email.html")} target="_blank" rel="noreferrer" className="mr-auto inline-flex h-10 items-center gap-1.5 text-sm font-medium text-accent hover:underline">
            <Icon name="eye" /> Preview
          </a>
        )}
        <Button variant="primary" onClick={submit} disabled={sending}>{sending ? <><Spinner /> Sending…</> : <><Icon name="mail" /> Send{recipients.length > 1 ? ` to ${recipients.length}` : ""}</>}</Button>
      </>}
    >
      <EmailField ref={input} id="send-email" label="To" value={email} onChange={setEmail} error={touched ? error : null} disabled={sending} />
    </Modal>
  );
}
