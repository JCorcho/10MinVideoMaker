from __future__ import annotations

from email.message import EmailMessage
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tenminvideomaker.contracts import ContractValidationError
from tenminvideomaker.mail import (
    GmailClient,
    GmailPollingService,
    GmailSettings,
    MailboxMessage,
    MailTransportError,
    PIPELINE_REQUEST_SUBJECT,
    PIPELINE_RESPONSE_SUBJECT,
    PIPELINE_SUBJECT,
    build_pipeline_request,
    extract_job_payload,
)
from tenminvideomaker.state_store import PipelineState

from test_contracts import payload


class MailTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = GmailSettings(
            username="owner@example.com",
            recipient="owner@example.com",
            allowed_senders=frozenset({"owner@example.com"}),
            auth_mode="app_password",
            secret="not-a-real-secret",
        )

    def test_attachment_is_preferred_over_body(self) -> None:
        message = EmailMessage()
        body_payload = payload()
        body_payload["job_id"] = "body-job"
        message.set_content(json.dumps(body_payload))
        attachment_payload = payload()
        attachment_payload["job_id"] = "attachment-job"
        message.add_attachment(json.dumps(attachment_payload), subtype="json", filename="job.json")
        self.assertEqual(extract_job_payload(message).job_id, "attachment-job")

    def test_plain_text_body_can_contain_a_json_document(self) -> None:
        message = EmailMessage()
        message.set_content(f"Grok response follows:\n{json.dumps(payload())}\nThank you.")
        self.assertEqual(extract_job_payload(message).job_id, "20260724-1610")

    def test_google_drive_link_is_downloaded_when_body_has_no_json(self) -> None:
        message = EmailMessage()
        drive_url = (
            "https://drive.google.com/file/d/"
            "1AbCdEfGhIjKlMnOpQrStUvWxYz/view?usp=sharing"
        )
        message.set_content(f"Job complete. Download scenes.json here: {drive_url}")
        loaded = []

        def drive_loader(url: str) -> str:
            loaded.append(url)
            return json.dumps(payload())

        self.assertEqual(
            extract_job_payload(message, drive_loader=drive_loader).job_id,
            "20260724-1610",
        )
        self.assertEqual(loaded, [drive_url])

    def test_drive_metadata_envelope_falls_back_to_its_file_link(self) -> None:
        message = EmailMessage()
        drive_url = (
            "https://drive.google.com/file/d/"
            "1AykG5rvgsNw2mz83op8z6h8tQI1V_iwY/view?usp=drivesdk"
        )
        message.set_content(
            "-----BEGIN_SCENES_JSON-----\n"
            + json.dumps(
                {
                    "job_id": "20260725-0035",
                    "created_at": "2026-07-25T00:35:00Z",
                    "drive_web_view_link": drive_url,
                    "drive_file_id": "1AykG5rvgsNw2mz83op8z6h8tQI1V_iwY",
                    "local_artifact": "video_job_20260725_0035.json",
                    "scenes_count": 20,
                }
            )
            + "\n-----END_SCENES_JSON-----"
        )
        loaded = []

        def drive_loader(url: str) -> str:
            loaded.append(url)
            return json.dumps(payload())

        self.assertEqual(
            extract_job_payload(message, drive_loader=drive_loader).job_id,
            "20260724-1610",
        )
        self.assertEqual(loaded, [drive_url])

    def test_body_json_is_preferred_over_drive_link(self) -> None:
        message = EmailMessage()
        drive_url = (
            "https://drive.google.com/file/d/"
            "1AbCdEfGhIjKlMnOpQrStUvWxYz/view?usp=sharing"
        )
        message.set_content(f"{json.dumps(payload())}\nBackup: {drive_url}")
        self.assertEqual(
            extract_job_payload(
                message,
                drive_loader=lambda _url: self.fail("Drive loader should not run"),
            ).job_id,
            "20260724-1610",
        )

    def test_google_drive_link_can_be_found_in_html_only_body(self) -> None:
        message = EmailMessage()
        drive_url = (
            "https://drive.google.com/file/d/"
            "1AbCdEfGhIjKlMnOpQrStUvWxYz/view?usp=sharing&amp;resourcekey=AbCdEfGhIjKl"
        )
        message.set_content("HTML version contains the job link.")
        message.add_alternative(
            f'<html><body><a href="{drive_url}">scenes.json</a></body></html>',
            subtype="html",
        )
        self.assertEqual(
            extract_job_payload(
                message,
                drive_loader=lambda _url: json.dumps(payload()),
            ).job_id,
            "20260724-1610",
        )

    def test_invalid_json_attachment_does_not_fall_back_to_body(self) -> None:
        message = EmailMessage()
        message.set_content(json.dumps(payload()))
        message.add_attachment("{}", subtype="json", filename="invalid.json")
        with self.assertRaises(ContractValidationError):
            extract_job_payload(message)

    def test_request_subject_is_exact_and_can_report_prior_job(self) -> None:
        message = build_pipeline_request(self.settings, previous_job_id="20260724-1610", succeeded=True)
        self.assertEqual(message["Subject"], PIPELINE_REQUEST_SUBJECT)
        self.assertEqual(PIPELINE_SUBJECT, PIPELINE_REQUEST_SUBJECT)
        self.assertNotEqual(PIPELINE_REQUEST_SUBJECT, PIPELINE_RESPONSE_SUBJECT)
        self.assertIn("20260724-1610 succeeded", message.get_content())
        self.assertIn("Google Drive file link", message.get_content())
        self.assertIn(f"exact subject {PIPELINE_RESPONSE_SUBJECT}", message.get_content())
        self.assertIn("Do not reply", message.get_content())

    def test_imap_searches_for_unread_exact_completion_subject(self) -> None:
        exact = EmailMessage()
        exact["From"] = "owner@example.com"
        exact["Subject"] = PIPELINE_RESPONSE_SUBJECT
        exact["Message-ID"] = "<exact@example.com>"
        exact.set_content(json.dumps(payload()))

        reply = EmailMessage()
        reply["From"] = "owner@example.com"
        reply["Subject"] = f"Re: {PIPELINE_RESPONSE_SUBJECT}"
        reply["Message-ID"] = "<reply@example.com>"
        reply.set_content(json.dumps(payload()))

        class FakeImap:
            search_arguments = None

            def __init__(self, *_args, **_kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def login(self, _username, _secret):
                return "OK", [b"authenticated"]

            def select(self, mailbox):
                self.selected_mailbox = mailbox
                return "OK", [b"2"]

            def uid(self, command, *arguments):
                if command == "SEARCH":
                    FakeImap.search_arguments = arguments
                    return "OK", [b"41 42"]
                if command == "FETCH" and arguments[0] == "41":
                    return "OK", [(b"41 (RFC822)", exact.as_bytes())]
                if command == "FETCH" and arguments[0] == "42":
                    return "OK", [(b"42 (RFC822)", reply.as_bytes())]
                raise AssertionError((command, arguments))

        with patch("tenminvideomaker.mail.imaplib.IMAP4_SSL", FakeImap):
            messages = GmailClient(self.settings).unread_pipeline_messages()

        self.assertEqual(
            FakeImap.search_arguments,
            (None, "UNSEEN", "SUBJECT", f'"{PIPELINE_RESPONSE_SUBJECT}"'),
        )
        self.assertEqual([message.uid for message in messages], ["41"])
        self.assertEqual(messages[0].subject, PIPELINE_RESPONSE_SUBJECT)

    def test_poller_rejects_outbound_request_subject(self) -> None:
        message = EmailMessage()
        message.set_content(json.dumps(payload()))

        class Store:
            def snapshot(self):
                return SimpleNamespace(state=PipelineState.WAITING_FOR_GROK)

            def claim_inbound_job(self, _message_key, _payload):
                raise AssertionError("Outbound request mail must never be claimed")

        class Client:
            settings = self.settings

            def unread_pipeline_messages(self):
                return [
                    MailboxMessage(
                        "42",
                        "request-message",
                        "owner@example.com",
                        PIPELINE_REQUEST_SUBJECT,
                        message,
                    )
                ]

        self.assertIsNone(GmailPollingService(Store(), Client()).poll_once())

    def test_drive_transport_error_leaves_message_unseen_for_retry(self) -> None:
        message = EmailMessage()
        message.set_content(
            "https://drive.google.com/file/d/"
            "1AbCdEfGhIjKlMnOpQrStUvWxYz/view?usp=sharing"
        )

        class Store:
            def snapshot(self):
                return SimpleNamespace(state=PipelineState.WAITING_FOR_GROK)

            def claim_message(self, _message_key):
                raise AssertionError("Transport failures must not claim the message")

        class Client:
            settings = self.settings
            marked_seen = []

            def unread_pipeline_messages(self):
                return [
                    MailboxMessage(
                        "42",
                        "message-key",
                        "owner@example.com",
                        PIPELINE_RESPONSE_SUBJECT,
                        message,
                    )
                ]

            def download_drive_json(self, _url):
                raise MailTransportError("temporary Drive outage")

            def mark_seen(self, uid):
                self.marked_seen.append(uid)

        client = Client()
        with self.assertRaisesRegex(MailTransportError, "temporary Drive outage"):
            GmailPollingService(Store(), client).poll_once()
        self.assertEqual(client.marked_seen, [])

    def test_environment_supports_oauth2_without_persisting_token(self) -> None:
        settings = GmailSettings.from_environment(
            {
                "TENMIN_GMAIL_USERNAME": "owner@example.com",
                "TENMIN_GMAIL_RECIPIENT": "owner@example.com",
                "TENMIN_GMAIL_AUTH_MODE": "oauth2",
                "TENMIN_GMAIL_OAUTH2_TOKEN": "ephemeral-token",
            }
        )
        self.assertEqual(settings.auth_mode, "oauth2")
        self.assertEqual(settings.secret, "ephemeral-token")

    def test_environment_supports_persistent_oauth_refresh_credentials(self) -> None:
        settings = GmailSettings.from_environment(
            {
                "TENMIN_GMAIL_USERNAME": "owner@example.com",
                "TENMIN_GMAIL_RECIPIENT": "owner@example.com",
                "TENMIN_GMAIL_AUTH_MODE": "oauth2",
                "TENMIN_GMAIL_OAUTH_CLIENT_ID": "client-id",
                "TENMIN_GMAIL_OAUTH_CLIENT_SECRET": "client-secret",
                "TENMIN_GMAIL_OAUTH_REFRESH_TOKEN": "refresh-token",
            }
        )
        self.assertEqual(settings.oauth_client_id, "client-id")
        self.assertEqual(settings.oauth_refresh_token, "refresh-token")
        self.assertEqual(settings.secret, "")

    def test_oauth_access_token_is_refreshed_once_then_cached(self) -> None:
        settings = GmailSettings(
            username="owner@example.com",
            recipient="owner@example.com",
            allowed_senders=frozenset({"owner@example.com"}),
            auth_mode="oauth2",
            secret="",
            oauth_client_id="client-id",
            oauth_client_secret="client-secret",
            oauth_refresh_token="refresh-token",
        )
        client = GmailClient(settings)
        with patch(
            "tenminvideomaker.mail.refresh_access_token",
            return_value=("access-token", 3600),
        ) as refresh:
            self.assertEqual(client._oauth_access_token(), "access-token")
            self.assertEqual(client._oauth_access_token(), "access-token")
        refresh.assert_called_once_with(
            client_id="client-id",
            client_secret="client-secret",
            refresh_token="refresh-token",
        )

    def test_smtp_oauth_callback_accepts_initial_zero_argument_call(self) -> None:
        settings = GmailSettings(
            username="owner@example.com",
            recipient="owner@example.com",
            allowed_senders=frozenset({"owner@example.com"}),
            auth_mode="oauth2",
            secret="access-token",
        )

        class FakeSmtp:
            mechanism = ""
            initial_response = ""
            events = []

            def ehlo_or_helo_if_needed(self):
                self.events.append("ehlo")

            def auth(self, mechanism, authobject):
                self.events.append("auth")
                self.mechanism = mechanism
                self.initial_response = authobject()

        smtp = FakeSmtp()
        GmailClient(settings)._authenticate_smtp(smtp)
        self.assertEqual(smtp.mechanism, "XOAUTH2")
        self.assertEqual(smtp.events, ["ehlo", "auth"])
        self.assertIn("user=owner@example.com", smtp.initial_response)
        self.assertIn("auth=Bearer access-token", smtp.initial_response)


if __name__ == "__main__":
    unittest.main()
