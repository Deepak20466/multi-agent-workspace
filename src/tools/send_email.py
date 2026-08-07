"""Email-sending tool via SMTP, wrapped with retry resilience.

Validates the recipient address with a regex, screens the subject/body
for PII with Presidio's `AnalyzerEngine` and blocks the send if more
than 3 entities are found, and logs every attempt (sent or blocked) to
telemetry. Defaults to a dry-run mode (logs the message instead of
sending) unless SMTP_* environment variables are fully configured, so
the tool is safe to wire into agent flows during local development.
"""

from __future__ import annotations

import logging
import os
import re
import smtplib
from email.message import EmailMessage

from langchain_core.tools import tool

from src.telemetry import log_event
from src.utils.retry_handler import with_retry, RetryableError

logger = logging.getLogger("send_email")

SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASS")

_EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_MAX_PII_ENTITIES = 3

_analyzer = None


def _get_analyzer():
    # Imported lazily: presidio_analyzer's import alone costs several
    # seconds, and eagerly importing it here would drag that cost into
    # every module that imports `src.tools` (e.g. python_repl's
    # subprocess spawn, which re-imports the whole `src.tools` package
    # and has its own tight timeout budget).
    global _analyzer
    if _analyzer is None:
        from presidio_analyzer import AnalyzerEngine

        _analyzer = AnalyzerEngine()
    return _analyzer


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


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email notification after validating the recipient address and screening the content for PII. Falls back to a dry-run log if SMTP isn't configured."""
    if not _EMAIL_PATTERN.match(to):
        raise EmailSendError(f"invalid email address: {to!r}")

    findings = _get_analyzer().analyze(text=f"{subject}\n{body}", language="en")
    if len(findings) > _MAX_PII_ENTITIES:
        log_event(
            "send_email_blocked",
            to=to,
            pii_entity_count=len(findings),
            pii_entity_types=sorted({f.entity_type for f in findings}),
        )
        raise EmailSendError(
            f"blocked: message contains {len(findings)} PII entities (limit {_MAX_PII_ENTITIES})"
        )

    if not _is_configured():
        logger.info("DRY RUN email to=%s subject=%s body=%s", to, subject, body)
        log_event("send_email_dry_run", to=to, subject=subject)
        return f"[dry-run] email to {to} logged, not sent (SMTP not configured)"

    message = EmailMessage()
    message["To"] = to
    message["From"] = SMTP_USER
    message["Subject"] = subject
    message.set_content(body)

    try:
        _send_via_smtp(message)
    except Exception as exc:
        log_event("send_email_failed", to=to, error=str(exc))
        raise EmailSendError(f"failed to send email to {to}: {exc}") from exc

    log_event("send_email_sent", to=to, subject=subject)
    return f"email sent to {to}"
