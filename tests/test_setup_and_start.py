from __future__ import annotations

import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

import scripts.setup_and_start as setup_module
from scripts.setup_and_start import (
    _local_comfy_url,
    configure_civitai,
    configure_gmail,
    edit_optional_settings,
    oauth_drive_scopes_ready,
    offer_saved_job_retry,
    required_gmail_ready,
)
from tenminvideomaker.oauth import GOOGLE_OAUTH_SCOPE_VALUE
from tenminvideomaker.contracts import parse_job_payload
from tenminvideomaker.state_store import PipelineState, PipelineStateStore, SceneState

from test_contracts import payload


def _answers(*values: str):
    iterator = iter(values)
    return lambda _prompt: next(iterator)


class SetupAndStartTests(unittest.TestCase):
    def test_required_detection_accepts_app_password_or_refresh_token(self) -> None:
        self.assertTrue(
            required_gmail_ready(
                {
                    "TENMIN_GMAIL_USERNAME": "owner@example.com",
                    "TENMIN_GMAIL_AUTH_MODE": "app_password",
                    "TENMIN_GMAIL_APP_PASSWORD": "sixteencharacters",
                }
            )
        )
        self.assertTrue(
            required_gmail_ready(
                {
                    "TENMIN_GMAIL_USERNAME": "owner@example.com",
                    "TENMIN_GMAIL_AUTH_MODE": "oauth2",
                    "TENMIN_GMAIL_OAUTH_CLIENT_ID": "client-id",
                    "TENMIN_GMAIL_OAUTH_CLIENT_SECRET": "client-secret",
                    "TENMIN_GMAIL_OAUTH_REFRESH_TOKEN": "refresh-token",
                    "TENMIN_GMAIL_OAUTH_SCOPES": GOOGLE_OAUTH_SCOPE_VALUE,
                }
            )
        )
        self.assertFalse(
            required_gmail_ready(
                {
                    "TENMIN_GMAIL_USERNAME": "owner@example.com",
                    "TENMIN_GMAIL_AUTH_MODE": "oauth2",
                    "TENMIN_GMAIL_OAUTH_CLIENT_ID": "client-id",
                    "TENMIN_GMAIL_OAUTH_CLIENT_SECRET": "client-secret",
                    "TENMIN_GMAIL_OAUTH_REFRESH_TOKEN": "mail-only-refresh-token",
                }
            )
        )
        self.assertFalse(required_gmail_ready({"TENMIN_GMAIL_USERNAME": "not-an-email"}))

    def test_app_password_setup_collects_required_values(self) -> None:
        environment = {}
        opened = []
        configure_gmail(
            environment,
            input_func=_answers("owner@example.com", "1", "n"),
            secret_input=lambda _prompt: "abcd efgh ijkl mnop",
            open_url=opened.append,
        )
        self.assertEqual(environment["TENMIN_GMAIL_AUTH_MODE"], "app_password")
        self.assertEqual(environment["TENMIN_GMAIL_APP_PASSWORD"], "abcdefghijklmnop")
        self.assertEqual(environment["TENMIN_GMAIL_RECIPIENT"], "owner@example.com")
        self.assertEqual(opened, [])

    def test_oauth_setup_runs_browser_authorization_and_saves_refresh_details(self) -> None:
        environment = {}
        authorization = {}

        def authorize(**kwargs):
            authorization.update(kwargs)
            return "refresh-token"

        configure_gmail(
            environment,
            input_func=_answers("owner@example.com", "2", "n", "client-id", "n"),
            secret_input=lambda _prompt: "client-secret",
            open_url=lambda _url: None,
            oauth_authorize=authorize,
        )
        self.assertEqual(environment["TENMIN_GMAIL_AUTH_MODE"], "oauth2")
        self.assertEqual(environment["TENMIN_GMAIL_OAUTH_CLIENT_ID"], "client-id")
        self.assertEqual(environment["TENMIN_GMAIL_OAUTH_REFRESH_TOKEN"], "refresh-token")
        self.assertTrue(oauth_drive_scopes_ready(environment))
        self.assertEqual(authorization["login_hint"], "owner@example.com")

    def test_existing_mail_only_oauth_is_reauthorized_for_drive(self) -> None:
        environment = {
            "TENMIN_GMAIL_USERNAME": "owner@example.com",
            "TENMIN_GMAIL_RECIPIENT": "owner@example.com",
            "TENMIN_GMAIL_ALLOWED_SENDERS": "owner@example.com",
            "TENMIN_GMAIL_AUTH_MODE": "oauth2",
            "TENMIN_GMAIL_OAUTH_CLIENT_ID": "client-id",
            "TENMIN_GMAIL_OAUTH_CLIENT_SECRET": "client-secret",
            "TENMIN_GMAIL_OAUTH_REFRESH_TOKEN": "old-refresh-token",
        }
        authorization = {}

        def authorize(**kwargs):
            authorization.update(kwargs)
            return "drive-refresh-token"

        configure_gmail(
            environment,
            input_func=_answers("n"),
            open_url=lambda _url: None,
            oauth_authorize=authorize,
        )
        self.assertEqual(
            environment["TENMIN_GMAIL_OAUTH_REFRESH_TOKEN"],
            "drive-refresh-token",
        )
        self.assertTrue(oauth_drive_scopes_ready(environment))
        self.assertEqual(authorization["client_id"], "client-id")

    def test_optional_editor_lists_and_changes_selected_value(self) -> None:
        environment = {"TENMIN_GMAIL_USERNAME": "owner@example.com"}
        edit_optional_settings(
            environment,
            input_func=_answers("4", "60", "0"),
        )
        self.assertEqual(environment["TENMIN_POLL_SECONDS"], "60")

    def test_civitai_setup_opens_account_page_and_collects_secret(self) -> None:
        environment = {}
        opened = []
        configure_civitai(
            environment,
            input_func=_answers("y"),
            secret_input=lambda _prompt: "civitai-api-token",
            open_url=opened.append,
        )
        self.assertEqual(environment["TENMIN_CIVITAI_TOKEN"], "civitai-api-token")
        self.assertEqual(opened, ["https://civitai.com/user/account"])

    def test_comfy_url_is_restricted_to_the_authorized_local_server(self) -> None:
        self.assertTrue(_local_comfy_url("http://127.0.0.1:8188"))
        self.assertTrue(_local_comfy_url("http://localhost:8188"))
        self.assertFalse(_local_comfy_url("https://example.com:8188"))

    def test_launcher_offers_to_retry_saved_failed_job(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = PipelineStateStore(root / "runtime" / "pipeline.sqlite3")
            job = parse_job_payload(payload())
            store.claim_job(job)
            store.set_scene_state(
                job.job_id,
                1,
                SceneState.FAILED,
                error="token required",
            )
            store.transition(
                PipelineState.ERROR,
                job_id=job.job_id,
                error="asset preparation failed",
            )
            with patch.object(setup_module, "PROJECT_ROOT", root):
                retried = offer_saved_job_retry(input_func=_answers(""))
            self.assertEqual(retried, job.job_id)
            self.assertEqual(store.snapshot().state, PipelineState.DOWNLOADING_ASSETS)
            self.assertEqual(
                store.scene_states(job.job_id),
                {1: SceneState.PENDING},
            )


if __name__ == "__main__":
    unittest.main()
