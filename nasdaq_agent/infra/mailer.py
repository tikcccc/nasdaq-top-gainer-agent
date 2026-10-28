"""Build the MIME message and deliver it over SMTP (Gmail), always keeping an .eml copy."""

from __future__ import annotations

import logging
import smtplib
import socket
from email.message import EmailMessage
from email.utils import make_msgid
from pathlib import Path

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from nasdaq_agent.config import EmailConfig
from nasdaq_agent.reporting.render import Report

log = logging.getLogger(__name__)


class DeliveryUnknown(RuntimeError):
    """The server may have accepted DATA; retrying could send a duplicate."""


class PreDeliveryError(RuntimeError):
    """Connection/setup failed before a message could be accepted."""


def build_message(report: Report, sender: str, recipients: list[str]) -> EmailMessage:
    msg = EmailMessage()
    msg["Subject"] = report.subject
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg["Message-ID"] = make_msgid(domain="nasdaq-agent.local")
    if report.renderer:
        msg['X-Report-Renderer'] = report.renderer['version']
        msg['X-Report-Fingerprint'] = report.renderer['fingerprint']
    msg.set_content(report.text)
    msg.add_alternative(report.html, subtype="html")
    html_part = msg.get_payload()[-1]
    for cid, path in report.images.items():
        subtype = {".gif": "gif", ".jpg": "jpeg", ".jpeg": "jpeg"}.get(path.suffix.lower(), "png")
        html_part.add_related(path.read_bytes(), maintype="image", subtype=subtype, cid=f"<{cid}>", filename=path.name)
    return msg


def save_eml(msg: EmailMessage, path: Path) -> Path:
    path.write_bytes(bytes(msg))
    return path


def _transient(exc: BaseException) -> bool:
    if isinstance(exc, PreDeliveryError):
        return _transient(exc.__cause__) if exc.__cause__ else False
    if isinstance(exc, smtplib.SMTPAuthenticationError):
        return False
    if isinstance(exc, smtplib.SMTPResponseException):
        return 400 <= exc.smtp_code < 500
    return isinstance(exc, (smtplib.SMTPServerDisconnected, smtplib.SMTPConnectError, socket.timeout, OSError))


@retry(retry=retry_if_exception(_transient), stop=stop_after_attempt(3), wait=wait_exponential_jitter(initial=2, max=20), reraise=True)
def send(msg: EmailMessage, config: EmailConfig) -> str:
    smtp = config.smtp
    if not (smtp.user and smtp.password):
        raise smtplib.SMTPAuthenticationError(535, b"SMTP_USER / SMTP_PASSWORD not configured")
    server = None
    try:
        server = smtplib.SMTP(smtp.host, smtp.port, timeout=30)
        server.starttls()
        server.login(smtp.user, smtp.password.get_secret_value())
    except (smtplib.SMTPServerDisconnected, socket.timeout, OSError) as exc:
        if server:
            server.close()
        raise PreDeliveryError("SMTP setup failed before message submission") from exc
    except Exception:
        if server:
            server.close()
        raise
    try:
        refused = server.send_message(msg)
        if refused:
            raise DeliveryUnknown(f"partial recipient acceptance; refused recipients: {sorted(refused)}")
    except (smtplib.SMTPServerDisconnected, socket.timeout, OSError) as exc:
        raise DeliveryUnknown("connection lost during SMTP DATA; acceptance is unknown") from exc
    finally:
        # QUIT failure must not turn an already accepted message into a retry.
        server.close()
    log.info("email sent to %s", msg["To"])
    return msg["Message-ID"]
