from __future__ import annotations

import unittest
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch

import scripts.setup_and_start as setup_module
from scripts.setup_and_start import (
    _local_comfy_url,
    configure_civitai,
    configure_discord,
    configure_lan_access,
    configure_gmail,
    edit_optional_settings,
    ensure_comfyui,
    _yes_no,
    oauth_drive_scopes_ready,
    offer_saved_job_retry,
    required_gmail_ready,
)
from tenminvideomaker.oauth import GOOGLE_OAUTH_SCOPE_VALUE
from tenminvideomaker.contracts import parse_job_payload
from tenminvideomaker.state_store import PipelineState, PipelineStateStore, SceneState

from test_contracts import payload

FAKE_DISCORD_WEBHOOK = (
    "https://discord.com" + "/api/webhooks/123456789/token-value"
)


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

    def test_discord_setup_collects_only_a_valid_webhook_secret(self) -> None:
        environment = {}
        configure_discord(
            environment,
            secret_input=lambda _prompt: FAKE_DISCORD_WEBHOOK,
        )
        self.assertEqual(
            environment["TENMIN_DISCORD_WEBHOOK_URL"],
            FAKE_DISCORD_WEBHOOK,
        )
        with self.assertRaisesRegex(Exception, "valid Discord"):
            configure_discord(
                {},
                secret_input=lambda _prompt: "https://example.com/not-discord",
            )

    def test_lan_setup_requires_a_password_and_can_disable_access(self) -> None:
        environment = {}
        configure_lan_access(
            environment,
            input_func=_answers("y"),
            secret_input=lambda _prompt: "mobile-password",
        )
        self.assertEqual(environment["TENMIN_GUI_LAN_ENABLED"], "true")
        self.assertEqual(environment["TENMIN_GUI_LAN_PASSWORD"], "mobile-password")
        configure_lan_access(environment, input_func=_answers("n"))
        self.assertEqual(environment["TENMIN_GUI_LAN_ENABLED"], "false")
        self.assertNotIn("TENMIN_GUI_LAN_PASSWORD", environment)

    def test_comfy_url_is_restricted_to_the_authorized_local_server(self) -> None:
        self.assertTrue(_local_comfy_url("http://127.0.0.1:8188"))
        self.assertTrue(_local_comfy_url("http://localhost:8188"))
        self.assertFalse(_local_comfy_url("https://example.com:8188"))

    def test_optional_settings_timeout_uses_no_default(self) -> None:
        with patch.object(setup_module, "_console_input_with_timeout", return_value=None) as reader:
            self.assertFalse(
                _yes_no(
                    "Change optional environment settings before starting?",
                    default=False,
                    timeout_seconds=10,
                )
            )
        reader.assert_called_once_with(
            "Change optional environment settings before starting? [y/N] ",
            10,
        )

    def test_gui_shared_comfy_guard_starts_the_verified_local_launcher(self) -> None:
        client = Mock()
        client.alive.side_effect = [False, True]
        completed = SimpleNamespace(returncode=0, stdout="", stderr="")
        storage = SimpleNamespace(state_root=Path(r"D:\\LTX_Supervisor_Storage"))
        with (
            patch.object(setup_module, "ComfyHttpClient", return_value=client),
            patch.object(setup_module.StorageLayout, "configured", return_value=storage),
            patch.object(setup_module.subprocess, "run", return_value=completed) as run,
        ):
            ensure_comfyui({"TENMIN_COMFY_URL": "http://127.0.0.1:8188"})

        command = run.call_args.args[0]
        self.assertIn("restart_comfyui.ps1", " ".join(command))
        self.assertIn("-EasyInstallRoot", command)
        self.assertIn("-ProjectRuntimeRoot", command)
        self.assertEqual(client.alive.call_count, 2)

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
                retried = offer_saved_job_retry(
                    input_func=_answers(""),
                    database_path=store.database_path,
                )
            self.assertEqual(retried, job.job_id)
            self.assertEqual(store.snapshot().state, PipelineState.DOWNLOADING_ASSETS)
            self.assertEqual(
                store.scene_states(job.job_id),
                {1: SceneState.PENDING},
            )

    def test_launcher_abandons_saved_job_when_retry_is_declined(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = PipelineStateStore(root / "runtime" / "pipeline.sqlite3")
            job = parse_job_payload(payload())
            store.claim_job(job)
            store.set_scene_state(
                job.job_id,
                1,
                SceneState.FAILED,
                error="unsafe or malformed payload",
            )
            store.transition(
                PipelineState.ERROR,
                job_id=job.job_id,
                error="asset preparation failed",
            )

            with patch.object(setup_module, "PROJECT_ROOT", root):
                retried = offer_saved_job_retry(
                    input_func=_answers("n"),
                    database_path=store.database_path,
                )

            self.assertIsNone(retried)
            snapshot = store.snapshot()
            record = store.scene_records(job.job_id)[0]
            self.assertEqual(snapshot.state, PipelineState.IDLE)
            self.assertIsNone(snapshot.job_id)
            self.assertEqual(record.state, SceneState.CANCELLED)
            self.assertIn("Abandoned by the user", record.error)

    def test_launcher_offers_resume_for_interrupted_running_job(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = PipelineStateStore(root / "runtime" / "pipeline.sqlite3")
            job = parse_job_payload(payload())
            store.claim_job(job)
            store.begin_scene_stage(
                job.job_id,
                1,
                PipelineState.RUNNING_I2V,
                prompt_id="active-prompt",
            )

            with patch.object(setup_module, "PROJECT_ROOT", root):
                resumed = offer_saved_job_retry(
                    input_func=_answers(""),
                    database_path=store.database_path,
                )

            self.assertEqual(resumed, job.job_id)
            self.assertEqual(store.snapshot().state, PipelineState.DOWNLOADING_ASSETS)
            self.assertEqual(store.scene_states(job.job_id), {1: SceneState.PENDING})
            self.assertIsNone(store.scene_records(job.job_id)[0].prompt_id)

    def test_declining_running_job_cancels_project_prompt_before_abandon(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = PipelineStateStore(root / "runtime" / "pipeline.sqlite3")
            job = parse_job_payload(payload())
            store.claim_job(job)
            store.begin_scene_stage(
                job.job_id,
                1,
                PipelineState.RUNNING_T2I,
                prompt_id="active-prompt",
            )
            comfy_client = Mock()
            comfy_client.cancel_project_prompts.return_value = ("active-prompt",)

            with patch.object(setup_module, "PROJECT_ROOT", root):
                resumed = offer_saved_job_retry(
                    input_func=_answers("n"),
                    comfy_client=comfy_client,
                    database_path=store.database_path,
                )

            self.assertIsNone(resumed)
            comfy_client.cancel_project_prompts.assert_called_once_with()
            self.assertEqual(store.snapshot().state, PipelineState.IDLE)
            self.assertEqual(
                store.scene_records(job.job_id)[0].state,
                SceneState.CANCELLED,
            )

    def test_launcher_can_resume_final_assembly_with_no_unfinished_scenes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = PipelineStateStore(root / "runtime" / "pipeline.sqlite3")
            job = parse_job_payload(payload())
            store.claim_job(job)
            store.set_scene_state(job.job_id, 1, SceneState.SUCCEEDED)
            store.transition(PipelineState.STITCHING, job_id=job.job_id)

            with patch.object(setup_module, "PROJECT_ROOT", root):
                resumed = offer_saved_job_retry(
                    input_func=_answers(""),
                    database_path=store.database_path,
                )

            self.assertEqual(resumed, job.job_id)
            self.assertEqual(store.snapshot().state, PipelineState.DOWNLOADING_ASSETS)

    def test_completed_waiting_job_does_not_prompt_on_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = PipelineStateStore(root / "runtime" / "pipeline.sqlite3")
            job = parse_job_payload(payload())
            store.claim_job(job)
            store.set_scene_state(job.job_id, 1, SceneState.SUCCEEDED)
            store.transition(PipelineState.WAITING_FOR_GROK, job_id=job.job_id)

            with patch.object(setup_module, "PROJECT_ROOT", root):
                resumed = offer_saved_job_retry(
                    input_func=lambda _prompt: self.fail("restart must not prompt"),
                    database_path=store.database_path,
                )

            self.assertIsNone(resumed)
            self.assertEqual(store.snapshot().state, PipelineState.WAITING_FOR_GROK)


if __name__ == "__main__":
    unittest.main()
