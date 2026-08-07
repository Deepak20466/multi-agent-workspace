"""Email-sending tool via SMTP, wrapped with retry resilience.

Defaults to a dry-run mode (logs the message instead of sending) unless
SMTP_* environment variables are fully configured, so the tool is safe
to wire into agent flows during local development.
"""

from __future__ import annotations

import logging
import os
import smtplib
from email.message import EmailMessage

from src.utils.retry_handler import with_retry, RetryableError

logger = logging.getLogger("send_email")

SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASS")


class EmailSendError(RuntimeError):
    pass


def _is_configured() -> bool:
    return all([SMTP_HOST, SMTP_USER, SMTP_PASSWORD])


@with_retry(exceptions=(RetryableError, smtplib.SMTPException, ConnectionError, TimeoutError))
def _send_via_smtp(message: EmailMessage) -> None:
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(message)
    except smtplib.SMTPServerDisconnected as exc:
        raise RetryableError(str(exc)) from exc


def send_email(to: str, subject: str, body: str, from_addr: str | None = None) -> str:
    """Send an email, or log a dry-run if SMTP is not configured.

    Returns a human-readable status string.
    """

    if not _is_configured():
        logger.info("DRY RUN email to=%s subject=%s body=%s", to, subject, body)
        return f"[dry-run] email to {to} logged, not sent (SMTP not configured)"

    message = EmailMessage()
    message["To"] = to
    message["From"] = from_addr or SMTP_USER
    message["Subject"] = subject
    message.set_content(body)

    try:
        _send_via_smtp(message)
    except Exception as exc:
        raise EmailSendError(f"failed to send email to {to}: {exc}") from exc

    return f"email sent to {to}"


TOOL_SPEC = {
    "name": "send_email",
    "description": "Send an email notification. Falls back to a dry-run log if SMTP isn't configured.",
    "parameters": {
        "type": "object",
        "properties": {
            "to": {"type": "string"},
            "subject": {"type": "string"},
            "body": {"type": "string"},
        },
        "required": ["to", "subject", "body"],
    },
}
