from __future__ import annotations

from email.message import EmailMessage
import json
import unittest
from unittest.mock import patch

from tenminvideomaker.contracts import ContractValidationError
from tenminvideomaker.mail import (
    GmailClient,
    GmailSettings,
    PIPELINE_SUBJECT,
    build_pipeline_request,
    extract_job_payload,
)

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

    def test_invalid_json_attachment_does_not_fall_back_to_body(self) -> None:
        message = EmailMessage()
        message.set_content(json.dumps(payload()))
        message.add_attachment("{}", subtype="json", filename="invalid.json")
        with self.assertRaises(ContractValidationError):
            extract_job_payload(message)

    def test_request_subject_is_exact_and_can_report_prior_job(self) -> None:
        message = build_pipeline_request(self.settings, previous_job_id="20260724-1610", succeeded=True)
        self.assertEqual(message["Subject"], PIPELINE_SUBJECT)
        self.assertIn("20260724-1610 succeeded", message.get_content())

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


if __name__ == "__main__":
    unittest.main()
