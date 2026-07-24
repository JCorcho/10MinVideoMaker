"""Gmail SMTP/IMAP integration with attachment-first job extraction."""

from __future__ import annotations

from dataclasses import dataclass
from email import policy
from email.message import EmailMessage, Message
from email.parser import BytesParser
from email.utils import formataddr, make_msgid, parseaddr
import imaplib
import json
import os
import smtplib
import ssl
import time
from typing import Any, Iterable, Mapping

from .contracts import ContractValidationError, JobPayload, parse_job_payload
from .oauth import OAuthError, refresh_access_token
from .state_store import PipelineState, PipelineStateStore

PIPELINE_SUBJECT = "Run the LTX video pipeline"


class MailConfigurationError(RuntimeError):
    """Raised when Gmail credentials or recipient information is incomplete."""


class MailTransportError(RuntimeError):
    """Raised when Gmail returns an unsuccessful transport response."""


@dataclass(frozen=True)
class GmailSettings:
    username: str
    recipient: str
    allowed_senders: frozenset[str]
    auth_mode: str
    secret: str
    oauth_client_id: str = ""
    oauth_client_secret: str = ""
    oauth_refresh_token: str = ""
    imap_host: str = "imap.gmail.com"
    smtp_host: str = "smtp.gmail.com"
    imap_port: int = 993
    smtp_port: int = 465

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] | None = None) -> "GmailSettings":
        values = os.environ if environment is None else environment
        username = values.get("TENMIN_GMAIL_USERNAME", "").strip()
        recipient = values.get("TENMIN_GMAIL_RECIPIENT", username).strip()
        auth_mode = values.get("TENMIN_GMAIL_AUTH_MODE", "app_password").strip().lower()
        if auth_mode not in {"app_password", "oauth2"}:
            raise MailConfigurationError("TENMIN_GMAIL_AUTH_MODE must be app_password or oauth2.")
        secret = values.get(
            "TENMIN_GMAIL_APP_PASSWORD" if auth_mode == "app_password" else "TENMIN_GMAIL_OAUTH2_TOKEN",
            "",
        ).strip()
        oauth_client_id = values.get("TENMIN_GMAIL_OAUTH_CLIENT_ID", "").strip()
        oauth_client_secret = values.get("TENMIN_GMAIL_OAUTH_CLIENT_SECRET", "").strip()
        oauth_refresh_token = values.get("TENMIN_GMAIL_OAUTH_REFRESH_TOKEN", "").strip()
        senders = values.get("TENMIN_GMAIL_ALLOWED_SENDERS", username)
        allowed_senders = frozenset(sender.strip().casefold() for sender in senders.split(",") if sender.strip())
        oauth_ready = bool(
            secret or (oauth_client_id and oauth_client_secret and oauth_refresh_token)
        )
        selected_secret_ready = bool(secret) if auth_mode == "app_password" else oauth_ready
        if not username or not recipient or not selected_secret_ready or not allowed_senders:
            raise MailConfigurationError(
                "Set TENMIN_GMAIL_USERNAME, TENMIN_GMAIL_RECIPIENT, and the selected Gmail auth secret."
            )
        return cls(
            username,
            recipient,
            allowed_senders,
            auth_mode,
            secret,
            oauth_client_id,
            oauth_client_secret,
            oauth_refresh_token,
        )


@dataclass(frozen=True)
class MailboxMessage:
    uid: str
    message_key: str
    sender: str
    subject: str
    message: Message


def build_pipeline_request(
    settings: GmailSettings,
    *,
    previous_job_id: str | None = None,
    succeeded: bool | None = None,
) -> EmailMessage:
    """Build the stable initial-trigger email without sending it."""
    message = EmailMessage()
    message["From"] = formataddr(("10MinVideoMaker", settings.username))
    message["To"] = settings.recipient
    message["Subject"] = PIPELINE_SUBJECT
    message["Message-ID"] = make_msgid(domain=settings.username.split("@")[-1])
    body = ["Please return one valid 10MinVideoMaker JSON job as a .json attachment or plain-text body."]
    if previous_job_id:
        result = "succeeded" if succeeded else "did not complete"
        body.append(f"Previous job {previous_job_id} {result}.")
    message.set_content("\n".join(body))
    return message


def _decode_part(part: Message) -> str:
    payload = part.get_payload(decode=True)
    if payload is None:
        raw_payload = part.get_payload()
        return raw_payload if isinstance(raw_payload, str) else ""
    charset = part.get_content_charset() or "utf-8"
    return payload.decode(charset, errors="replace")


def _json_documents(text: str) -> Iterable[Any]:
    decoder = json.JSONDecoder()
    stripped = text.strip()
    if not stripped:
        return
    try:
        yield json.loads(stripped)
        return
    except json.JSONDecodeError:
        pass
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            document, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        yield document


def _parse_document(text: str, source: str) -> JobPayload:
    for document in _json_documents(text):
        if isinstance(document, Mapping) and "job_id" in document and "scenes" in document:
            return parse_job_payload(document)
    raise ContractValidationError(f"No valid job JSON with job_id and scenes was found in {source}.")


def extract_job_payload(message: Message) -> JobPayload:
    """Prefer a .json attachment; otherwise parse the plain-text email body."""
    json_attachments: list[tuple[str, str]] = []
    for part in message.walk():
        filename = part.get_filename() or ""
        content_type = part.get_content_type().casefold()
        is_json_attachment = part.get_content_disposition() == "attachment" and (
            filename.casefold().endswith(".json") or content_type == "application/json"
        )
        if is_json_attachment:
            json_attachments.append((filename or "JSON attachment", _decode_part(part)))
    if json_attachments:
        filename, content = json_attachments[0]
        return _parse_document(content, filename)

    plain_parts = [
        _decode_part(part)
        for part in message.walk()
        if part.get_content_type().casefold() == "text/plain" and part.get_content_disposition() != "attachment"
    ]
    return _parse_document("\n".join(plain_parts), "message body")


class GmailClient:
    """Small standard-library Gmail client; credentials are read only from environment at runtime."""

    def __init__(self, settings: GmailSettings):
        self.settings = settings
        self._cached_access_token = ""
        self._access_token_expires_at = 0.0

    def send_request(self, *, previous_job_id: str | None = None, succeeded: bool | None = None) -> str:
        message = build_pipeline_request(self.settings, previous_job_id=previous_job_id, succeeded=succeeded)
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(self.settings.smtp_host, self.settings.smtp_port, context=context, timeout=30) as smtp:
            self._authenticate_smtp(smtp)
            smtp.send_message(message)
        return str(message["Message-ID"])

    def unread_pipeline_messages(self) -> list[MailboxMessage]:
        context = ssl.create_default_context()
        with imaplib.IMAP4_SSL(self.settings.imap_host, self.settings.imap_port, ssl_context=context, timeout=30) as client:
            self._authenticate_imap(client)
            status, _ = client.select("INBOX")
            if status != "OK":
                raise MailTransportError("Could not select Gmail INBOX.")
            status, data = client.uid("SEARCH", None, "UNSEEN", "SUBJECT", f'"{PIPELINE_SUBJECT}"')
            if status != "OK":
                raise MailTransportError("Could not search Gmail for pipeline responses.")
            messages: list[MailboxMessage] = []
            for raw_uid in data[0].split() if data and data[0] else []:
                uid = raw_uid.decode("ascii")
                status, fetched = client.uid("FETCH", uid, "(RFC822)")
                if status != "OK" or not fetched or not isinstance(fetched[0], tuple):
                    continue
                message = BytesParser(policy=policy.default).parsebytes(fetched[0][1])
                sender = parseaddr(message.get("From", ""))[1].casefold()
                message_key = str(message.get("Message-ID") or f"imap-uid:{uid}").strip()
                messages.append(MailboxMessage(uid, message_key, sender, str(message.get("Subject", "")), message))
            return messages

    def mark_seen(self, uid: str) -> None:
        context = ssl.create_default_context()
        with imaplib.IMAP4_SSL(self.settings.imap_host, self.settings.imap_port, ssl_context=context, timeout=30) as client:
            self._authenticate_imap(client)
            status, _ = client.select("INBOX")
            if status != "OK":
                raise MailTransportError("Could not select Gmail INBOX.")
            status, _ = client.uid("STORE", uid, "+FLAGS", "(\\Seen)")
            if status != "OK":
                raise MailTransportError(f"Could not mark Gmail UID {uid} as read.")

    def validate_credentials(self) -> None:
        """Authenticate to both transports without reading messages or sending email."""
        context = ssl.create_default_context()
        try:
            with smtplib.SMTP_SSL(
                self.settings.smtp_host,
                self.settings.smtp_port,
                context=context,
                timeout=30,
            ) as smtp:
                self._authenticate_smtp(smtp)
                status, response = smtp.mail(self.settings.username)
                if status != 250:
                    raise MailTransportError(
                        f"Gmail rejected the authenticated sender envelope: {response!r}"
                    )
                smtp.rset()
            with imaplib.IMAP4_SSL(
                self.settings.imap_host,
                self.settings.imap_port,
                ssl_context=context,
                timeout=30,
            ) as client:
                self._authenticate_imap(client)
        except (OSError, smtplib.SMTPException, imaplib.IMAP4.error, OAuthError) as error:
            raise MailTransportError(f"Gmail credential validation failed: {error}") from error

    def _authenticate_smtp(self, smtp: smtplib.SMTP_SSL) -> None:
        if self.settings.auth_mode == "app_password":
            smtp.login(self.settings.username, self.settings.secret)
            return
        smtp.ehlo_or_helo_if_needed()
        token = f"user={self.settings.username}\x01auth=Bearer {self._oauth_access_token()}\x01\x01"
        smtp.auth("XOAUTH2", lambda _challenge=None: token)

    def _authenticate_imap(self, client: imaplib.IMAP4_SSL) -> None:
        if self.settings.auth_mode == "app_password":
            status, _ = client.login(self.settings.username, self.settings.secret)
        else:
            token = f"user={self.settings.username}\x01auth=Bearer {self._oauth_access_token()}\x01\x01"
            status, _ = client.authenticate(
                "XOAUTH2",
                lambda _challenge=None: token.encode("utf-8"),
            )
        if status != "OK":
            raise MailTransportError("Gmail authentication failed.")

    def _oauth_access_token(self) -> str:
        if self.settings.oauth_refresh_token:
            if self._cached_access_token and time.monotonic() < self._access_token_expires_at:
                return self._cached_access_token
            token, expires_in = refresh_access_token(
                client_id=self.settings.oauth_client_id,
                client_secret=self.settings.oauth_client_secret,
                refresh_token=self.settings.oauth_refresh_token,
            )
            self._cached_access_token = token
            self._access_token_expires_at = time.monotonic() + max(expires_in - 60, 30)
            return token
        if self.settings.secret:
            return self.settings.secret
        raise MailConfigurationError("OAuth2 is configured without an access or refresh token.")


class GmailPollingService:
    """Coordinates mailbox acceptance with the durable state-machine gate."""

    def __init__(self, store: PipelineStateStore, client: GmailClient):
        self.store = store
        self.client = client

    def poll_once(self) -> JobPayload | None:
        if self.store.snapshot().state not in {PipelineState.IDLE, PipelineState.WAITING_FOR_GROK}:
            return None
        for mailbox_message in self.client.unread_pipeline_messages():
            if mailbox_message.sender not in self.client.settings.allowed_senders:
                continue
            try:
                payload = extract_job_payload(mailbox_message.message)
            except ContractValidationError:
                # A malformed response is terminal for that email but does not poison the running job state.
                if self.store.claim_message(mailbox_message.message_key):
                    self.client.mark_seen(mailbox_message.uid)
                continue
            if self.store.claim_inbound_job(mailbox_message.message_key, payload):
                self.client.mark_seen(mailbox_message.uid)
                return payload
        return None
