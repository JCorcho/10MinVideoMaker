from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

from tenminvideomaker.configuration import (
    SecretStore,
    load_project_environment,
    read_env_file,
    save_project_environment,
    write_env_file,
)
from tenminvideomaker.storage import StorageLayout


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_TEMP_ROOT = PROJECT_ROOT / "runtime" / "test-temp"
FAKE_DISCORD_WEBHOOK = "https://discord.com" + "/api/webhooks/123/token-secret"


class ConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)

    def test_secret_store_does_not_write_plaintext(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary:
            path = Path(temporary) / "secrets.json"
            store = SecretStore(
                path,
                protect=lambda value: b"protected:" + value[::-1],
                unprotect=lambda value: value.removeprefix(b"protected:")[::-1],
            )
            store.save(
                {
                    "TENMIN_GMAIL_APP_PASSWORD": "sixteencharacters",
                    "TENMIN_CIVITAI_TOKEN": "civitai-secret",
                    "TENMIN_DISCORD_WEBHOOK_URL": FAKE_DISCORD_WEBHOOK,
                    "IGNORED": "not-allowed",
                }
            )
            serialized = path.read_text(encoding="utf-8")
            self.assertNotIn("sixteencharacters", serialized)
            self.assertNotIn("civitai-secret", serialized)
            self.assertNotIn("token-secret", serialized)
            self.assertNotIn("not-allowed", serialized)
            self.assertEqual(
                store.load(),
                {
                    "TENMIN_CIVITAI_TOKEN": "civitai-secret",
                    "TENMIN_DISCORD_WEBHOOK_URL": FAKE_DISCORD_WEBHOOK,
                    "TENMIN_GMAIL_APP_PASSWORD": "sixteencharacters",
                },
            )

    def test_env_file_round_trip_and_os_values_take_precedence(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary:
            root = Path(temporary)
            storage = StorageLayout(root / "storage")
            write_env_file(
                storage.settings_path,
                {
                    "TENMIN_GMAIL_USERNAME": "local@example.com",
                    "TENMIN_GMAIL_OAUTH_SCOPES": (
                        "https://mail.google.com/ "
                        "https://www.googleapis.com/auth/drive.readonly"
                    ),
                    "TENMIN_POLL_SECONDS": "300",
                    "TENMIN_STATUS_INTERVAL_SECONDS": "15",
                    "NOT_ALLOWED": "ignored",
                },
            )
            self.assertEqual(
                read_env_file(storage.settings_path),
                {
                    "TENMIN_GMAIL_USERNAME": "local@example.com",
                    "TENMIN_GMAIL_OAUTH_SCOPES": (
                        "https://mail.google.com/ "
                        "https://www.googleapis.com/auth/drive.readonly"
                    ),
                    "TENMIN_POLL_SECONDS": "300",
                    "TENMIN_STATUS_INTERVAL_SECONDS": "15",
                },
            )
            merged = load_project_environment(
                root,
                base_environment={
                    "TENMIN_GMAIL_USERNAME": "process@example.com",
                    "UNRELATED": "preserved",
                },
                storage_layout=storage,
            )
            self.assertEqual(merged["TENMIN_GMAIL_USERNAME"], "process@example.com")
            self.assertEqual(merged["TENMIN_POLL_SECONDS"], "300")
            self.assertEqual(merged["TENMIN_STATUS_INTERVAL_SECONDS"], "15")
            self.assertEqual(merged["UNRELATED"], "preserved")

    def test_non_project_callers_cannot_write_the_production_storage_root(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary:
            root = Path(temporary)
            save_project_environment(
                root,
                {"TENMIN_GMAIL_USERNAME": "isolated@example.com"},
            )

            isolated = root / "runtime-storage" / "config" / "settings.env"
            self.assertTrue(isolated.is_file())
            self.assertEqual(
                read_env_file(isolated)["TENMIN_GMAIL_USERNAME"],
                "isolated@example.com",
            )

    @unittest.skipUnless(os.name == "nt", "Windows DPAPI is required")
    def test_project_save_uses_dpapi_for_secrets(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary:
            root = Path(temporary)
            values = {
                "TENMIN_GMAIL_USERNAME": "owner@example.com",
                "TENMIN_GMAIL_AUTH_MODE": "oauth2",
                "TENMIN_GMAIL_OAUTH_CLIENT_ID": "client-id",
                "TENMIN_GMAIL_OAUTH_CLIENT_SECRET": "client-secret-value",
                "TENMIN_GMAIL_OAUTH_REFRESH_TOKEN": "refresh-token-value",
                "TENMIN_CIVITAI_TOKEN": "civitai-token-value",
                "TENMIN_DISCORD_WEBHOOK_URL": (
                    "https://discord.com" + "/api/webhooks/123/discord-token-value"
                ),
            }
            storage = StorageLayout(root / "storage")
            save_project_environment(root, values, storage_layout=storage)
            env_text = storage.settings_path.read_text(encoding="utf-8")
            secret_text = storage.secrets_path.read_text(encoding="utf-8")
            json.loads(secret_text)
            self.assertNotIn("client-secret-value", env_text + secret_text)
            self.assertNotIn("refresh-token-value", env_text + secret_text)
            self.assertNotIn("civitai-token-value", env_text + secret_text)
            self.assertNotIn("discord-token-value", env_text + secret_text)
            loaded = load_project_environment(
                root,
                base_environment={},
                storage_layout=storage,
            )
            self.assertEqual(loaded["TENMIN_GMAIL_OAUTH_CLIENT_SECRET"], "client-secret-value")
            self.assertEqual(loaded["TENMIN_GMAIL_OAUTH_REFRESH_TOKEN"], "refresh-token-value")
            self.assertEqual(loaded["TENMIN_CIVITAI_TOKEN"], "civitai-token-value")
            self.assertEqual(
                loaded["TENMIN_DISCORD_WEBHOOK_URL"],
                "https://discord.com" + "/api/webhooks/123/discord-token-value",
            )


if __name__ == "__main__":
    unittest.main()
