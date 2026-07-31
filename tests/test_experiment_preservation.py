from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "ltx23-style-conversion"
MANIFEST = EXPERIMENT_ROOT / "manifest.json"


class StyleConversionPreservationTests(unittest.TestCase):
    def test_manifest_binds_all_frozen_workflows_to_exact_sha256(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(
            manifest["source_run_id"],
            "continuation-acceptance-20260731-065935",
        )
        self.assertEqual(len(manifest["workflows"]), 4)
        for entry in manifest["workflows"]:
            path = EXPERIMENT_ROOT / entry["relative_path"]
            self.assertTrue(path.is_file(), path)
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                entry["sha256"],
            )

    def test_frozen_workflows_contain_no_secret_markers(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        text = "\n".join(
            (EXPERIMENT_ROOT / entry["relative_path"]).read_text(encoding="utf-8")
            for entry in manifest["workflows"]
        ).casefold()
        for marker in (
            "discord.com/api/webhooks",
            "api_key",
            "password",
            "client_secret",
            "refresh_token",
            "github_token",
        ):
            self.assertNotIn(marker, text)

    def test_manifest_keeps_large_raw_media_on_project_storage(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

        self.assertEqual(
            {entry["case_name"] for entry in manifest["raw_media"]},
            {"decoded_17_frame", "latent_overlap"},
        )
        for entry in manifest["raw_media"]:
            self.assertTrue(
                entry["audit_path"].startswith("D:\\LTX_Supervisor_Storage\\"),
                entry,
            )
            self.assertRegex(entry["sha256"], r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
