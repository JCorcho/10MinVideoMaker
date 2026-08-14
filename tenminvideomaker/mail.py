"""Gmail integration with attachment, body, and Google Drive job extraction."""

from __future__ import annotations

from dataclasses import dataclass
from email import policy
from email.message import EmailMessage, Message
from email.parser import BytesParser
from email.utils import formataddr, make_msgid, parseaddr
import hashlib
import imaplib
import json
import logging
import os
import smtplib
import ssl
import time
from typing import Any, Callable, Iterable, Mapping

from .contracts import ContractValidationError, JobPayload, parse_job_payload
from .drive import (
    DriveDownloadError,
    download_google_drive_json,
    google_drive_file_urls,
    validate_google_drive_access,
)
from .oauth import OAuthError, refresh_access_token
from .state_store import PipelineState, PipelineStateStore

PIPELINE_REQUEST_SUBJECT = "Run the LTX video pipeline"
PIPELINE_RESPONSE_SUBJECT = "LTX_JOB_COMPLETE"

# Backward-compatible public name for callers that build the outbound request.
PIPELINE_SUBJECT = PIPELINE_REQUEST_SUBJECT
LOGGER = logging.getLogger("10MinVideoMaker.mail")


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
    request_id: str | None = None,
) -> EmailMessage:
    """Build the stable initial-trigger email without sending it."""
    message = EmailMessage()
    message["From"] = formataddr(("10MinVideoMaker", settings.username))
    message["To"] = settings.recipient
    message["Subject"] = PIPELINE_REQUEST_SUBJECT
    if request_id is None:
        message["Message-ID"] = make_msgid(domain=settings.username.split("@")[-1])
    else:
        if not request_id or len(request_id) > 256:
            raise ValueError("request_id must be a non-empty bounded string.")
        digest = hashlib.sha256(request_id.encode("utf-8")).hexdigest()
        domain = settings.username.split("@")[-1]
        message["Message-ID"] = f"<tenmin-qc-{digest}@{domain}>"
    body = [
        f"Send the completed handoff as a new email with the exact subject {PIPELINE_RESPONSE_SUBJECT}. "
        "Do not reply to this request email.",
        "Please return one valid 10MinVideoMaker JSON job as a .json attachment, plain-text body, "
        "or a Google Drive file link.",
        f"For a Drive link, share the file with {settings.username} or allow anyone with the link to view it.",
    ]
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
    normalized, normalized_count = _normalize_seed_integer_literals(text)
    if normalized_count:
        try:
            document = json.loads(normalized.strip())
            LOGGER.warning(
                "Normalized %d leading-zero seed integer value(s) in inbound JSON.",
                normalized_count,
            )
            yield document
            return
        except json.JSONDecodeError:
            pass
    for index, character in enumerate(normalized):
        if character != "{":
            continue
        try:
            document, _ = decoder.raw_decode(normalized[index:])
        except json.JSONDecodeError:
            continue
        if normalized_count:
            LOGGER.warning(
                "Normalized %d leading-zero seed integer value(s) in inbound JSON.",
                normalized_count,
            )
        yield document


def _normalize_seed_integer_literals(text: str) -> tuple[str, int]:
    """Repair only JSON integer values for seed keys that contain redundant leading zeroes."""
    result: list[str] = []
    index = 0
    normalized_count = 0
    length = len(text)
    while index < length:
        if text[index] != '"':
            result.append(text[index])
            index += 1
            continue

        string_start = index
        index += 1
        escaped = False
        while index < length:
            character = text[index]
            index += 1
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                break
        raw_string = text[string_start:index]
        result.append(raw_string)
        try:
            key = json.loads(raw_string)
        except json.JSONDecodeError:
            continue
        if key not in {"seed", "original_seed"}:
            continue

        separator_end = index
        while separator_end < length and text[separator_end].isspace():
            separator_end += 1
        if separator_end >= length or text[separator_end] != ":":
            continue
        value_start = separator_end + 1
        while value_start < length and text[value_start].isspace():
            value_start += 1
        digit_start = value_start
        if digit_start < length and text[digit_start] == "-":
            digit_start += 1
        digit_end = digit_start
        while digit_end < length and text[digit_end].isdigit():
            digit_end += 1
        digits = text[digit_start:digit_end]
        if (
            len(digits) <= 1
            or not digits.startswith("0")
            or digit_end >= length
            or text[digit_end] not in " \t\r\n,}]"
        ):
            continue

        normalized_digits = digits.lstrip("0") or "0"
        result.append(text[index:digit_start])
        result.append(normalized_digits)
        index = digit_end
        normalized_count += 1
    return "".join(result), normalized_count


def _parse_document(text: str, source: str) -> JobPayload:
    for document in _json_documents(text):
        if isinstance(document, Mapping) and "job_id" in document and "scenes" in document:
            return parse_job_payload(document)
    raise ContractValidationError(f"No valid job JSON with job_id and scenes was found in {source}.")


def extract_job_payload(
    message: Message,
    *,
    drive_loader: Callable[[str], str] | None = None,
) -> JobPayload:
    """Prefer an attachment, then body JSON, then a supported Google Drive file link."""
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
    plain_text = "\n".join(plain_parts)
    try:
        return _parse_document(plain_text, "message body")
    except ContractValidationError as body_error:
        link_parts = [
            _decode_part(part)
            for part in message.walk()
            if part.get_content_type().casefold() in {"text/plain", "text/html"}
            and part.get_content_disposition() != "attachment"
        ]
        drive_links = google_drive_file_urls("\n".join(link_parts))
        if not drive_links:
            raise body_error
        if drive_loader is None:
            raise ContractValidationError(
                "A Google Drive job link was found, but no Drive downloader is configured."
            ) from body_error
        invalid_reasons: list[str] = []
        for drive_link in drive_links:
            content = drive_loader(drive_link)
            try:
                return _parse_document(content, "Google Drive file")
            except ContractValidationError as parse_error:
                # Keep the contract detail (e.g. bad version_id) instead of a
                # generic "no JSON" message that hides why a real Drive file failed.
                invalid_reasons.append(str(parse_error))
        if len(invalid_reasons) == 1:
            raise ContractValidationError(
                f"Google Drive job file failed validation: {invalid_reasons[0]}"
            ) from body_error
        raise ContractValidationError(
            "No valid job JSON was found in "
            f"{len(invalid_reasons)} Google Drive file(s): "
            + "; ".join(invalid_reasons)
        ) from body_error


class GmailClient:
    """Small standard-library Gmail client; credentials are read only from environment at runtime."""

    def __init__(self, settings: GmailSettings):
        self.settings = settings
        self._cached_access_token = ""
        self._access_token_expires_at = 0.0

    def send_request(
        self,
        *,
        previous_job_id: str | None = None,
        succeeded: bool | None = None,
        request_id: str | None = None,
    ) -> str:
        message = build_pipeline_request(
            self.settings,
            previous_job_id=previous_job_id,
            succeeded=succeeded,
            request_id=request_id,
        )
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(self.settings.smtp_host, self.settings.smtp_port, context=context, timeout=30) as smtp:
            self._authenticate_smtp(smtp)
            smtp.send_message(message)
        return str(message["Message-ID"])

    def request_message_id(self, request_id: str) -> str:
        return str(
            build_pipeline_request(self.settings, request_id=request_id)["Message-ID"]
        )

    def request_was_sent(self, request_id: str) -> bool:
        """Reconcile one deterministic QC request through Gmail's All Mail view."""
        message_id = self.request_message_id(request_id)
        context = ssl.create_default_context()
        with imaplib.IMAP4_SSL(
            self.settings.imap_host,
            self.settings.imap_port,
            ssl_context=context,
            timeout=30,
        ) as client:
            self._authenticate_imap(client)
            status, _ = client.select("[Gmail]/All Mail", readonly=True)
            if status != "OK":
                raise MailTransportError(
                    "Could not select Gmail All Mail to reconcile the QC request."
                )
            status, data = client.uid(
                "SEARCH",
                None,
                "HEADER",
                "Message-ID",
                f'"{message_id}"',
            )
            if status != "OK":
                raise MailTransportError(
                    "Could not reconcile the deterministic QC request Message-ID."
                )
            return bool(data and data[0] and data[0].split())

    def unread_pipeline_messages(self) -> list[MailboxMessage]:
        context = ssl.create_default_context()
        with imaplib.IMAP4_SSL(self.settings.imap_host, self.settings.imap_port, ssl_context=context, timeout=30) as client:
            self._authenticate_imap(client)
            status, _ = client.select("INBOX")
            if status != "OK":
                raise MailTransportError("Could not select Gmail INBOX.")
            status, data = client.uid(
                "SEARCH",
                None,
                "UNSEEN",
                "SUBJECT",
                f'"{PIPELINE_RESPONSE_SUBJECT}"',
            )
            if status != "OK":
                raise MailTransportError("Could not search Gmail for pipeline responses.")
            messages: list[MailboxMessage] = []
            for raw_uid in data[0].split() if data and data[0] else []:
                uid = raw_uid.decode("ascii")
                # BODY.PEEK[] is essential here: RFC822/BODY[] may set \Seen merely
                # because candidates were inspected. Only mark the one message that
                # was accepted (or deliberately rejected) after parsing.
                status, fetched = client.uid("FETCH", uid, "(BODY.PEEK[])")
                if status != "OK" or not fetched or not isinstance(fetched[0], tuple):
                    continue
                message = BytesParser(policy=policy.default).parsebytes(fetched[0][1])
                sender = parseaddr(message.get("From", ""))[1].casefold()
                message_key = str(message.get("Message-ID") or f"imap-uid:{uid}").strip()
                subject = str(message.get("Subject", "")).strip()
                # IMAP SUBJECT matching is substring-based. Enforce the exact handoff
                # subject here so replies, forwards, and similarly named mail cannot
                # enter the pipeline.
                if subject != PIPELINE_RESPONSE_SUBJECT:
                    continue
                messages.append(MailboxMessage(uid, message_key, sender, subject, message))
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
        """Authenticate to mail and, for OAuth, validate read-only Drive access."""
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
            if self.settings.auth_mode == "oauth2":
                validate_google_drive_access(self._oauth_access_token())
        except (
            OSError,
            smtplib.SMTPException,
            imaplib.IMAP4.error,
            OAuthError,
            DriveDownloadError,
        ) as error:
            raise MailTransportError(f"Gmail credential validation failed: {error}") from error

    def download_drive_json(self, share_url: str) -> str:
        access_token = self._oauth_access_token() if self.settings.auth_mode == "oauth2" else ""
        try:
            return download_google_drive_json(share_url, access_token=access_token)
        except DriveDownloadError as error:
            raise MailTransportError(f"Google Drive job download failed: {error}") from error

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

    def poll_once(self, *, review_required: bool = False) -> JobPayload | None:
        if self.store.snapshot().state not in {PipelineState.IDLE, PipelineState.WAITING_FOR_GROK}:
            return None
        mailbox_messages = self.client.unread_pipeline_messages()
        LOGGER.info(
            "Gmail returned %d unread exact-subject candidate message(s).",
            len(mailbox_messages),
        )
        for mailbox_message in mailbox_messages:
            # Defense in depth for alternate/test Gmail clients: only the dedicated
            # completion subject may be treated as a job handoff.
            if mailbox_message.subject.strip() != PIPELINE_RESPONSE_SUBJECT:
                continue
            if mailbox_message.sender not in self.client.settings.allowed_senders:
                continue
            try:
                payload = extract_job_payload(
                    mailbox_message.message,
                    drive_loader=self.client.download_drive_json,
                )
            except ContractValidationError as error:
                # A malformed response is terminal for that email but does not poison the running job state.
                LOGGER.warning("Rejected one Gmail job candidate: %s", error)
                if self.store.claim_message(mailbox_message.message_key):
                    self.client.mark_seen(mailbox_message.uid)
                continue
            claim = self.store.claim_inbound_job(
                mailbox_message.message_key,
                payload,
                review_required=review_required,
            )
            # Lightweight test clients from earlier releases return a bool;
            # the durable store returns a richer collision-aware result.
            accepted = claim if isinstance(claim, bool) else claim.accepted
            accepted_payload = payload if isinstance(claim, bool) else claim.payload
            duplicate_content = False if isinstance(claim, bool) else claim.duplicate_content
            if accepted:
                self.client.mark_seen(mailbox_message.uid)
                return accepted_payload
            if duplicate_content:
                self.client.mark_seen(mailbox_message.uid)
                LOGGER.info(
                    "Skipped parsed job %s because its content matches accepted job %s.",
                    payload.job_id,
                    claim.source_job_id,
                )
                continue
            LOGGER.info(
                "Skipped parsed job %s because its Gmail message was already accepted or the pipeline became busy.",
                payload.job_id,
            )
        return None
