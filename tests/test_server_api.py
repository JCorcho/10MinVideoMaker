from __future__ import annotations

from pathlib import Path
import unittest

from tenminvideomaker.assets import AssetResolution
from tenminvideomaker.server_api import resolve_asset_request


class FakeManager:
    def __init__(self) -> None:
        self.dynamic = None
        self.required = None

    def resolve_or_download(self, lora):
        self.dynamic = lora
        return AssetResolution(
            lora.name,
            Path(r"C:\models\loras\canonical.safetensors"),
            downloaded=False,
            local_filename="canonical.safetensors",
        )

    def require_local(self, requirement):
        self.required = requirement
        return AssetResolution(
            requirement.filename,
            Path(r"C:\models\loras") / requirement.filename,
            downloaded=False,
            local_filename=requirement.filename,
        )


class ServerApiTests(unittest.TestCase):
    def test_dynamic_request_normalizes_civitai_version_and_filename(self) -> None:
        manager = FakeManager()
        result = resolve_asset_request(
            {
                "kind": "dynamic",
                "lora": {
                    "name": "Display name",
                    "download_url": "https://civitai.com/api/download/models/3184055",
                    "weight": 0.85,
                    "model_id": 2847192,
                },
            },
            manager,
        )
        self.assertEqual(manager.dynamic.version_id, 3184055)
        self.assertEqual(manager.dynamic.model_id, 2847192)
        self.assertEqual(result["local_filename"], "canonical.safetensors")
        self.assertTrue(result["succeeded"])

    def test_required_request_never_becomes_a_download(self) -> None:
        manager = FakeManager()
        result = resolve_asset_request(
            {
                "kind": "required",
                "filename": "LTX2.3_DMD_reshaped_r256.safetensors",
                "weight": 1.0,
            },
            manager,
        )
        self.assertEqual(
            manager.required.filename,
            "LTX2.3_DMD_reshaped_r256.safetensors",
        )
        self.assertFalse(result["downloaded"])

    def test_request_rejects_version_url_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not match"):
            resolve_asset_request(
                {
                    "kind": "dynamic",
                    "lora": {
                        "name": "Mismatch",
                        "download_url": "https://civitai.com/api/download/models/123",
                        "weight": 1.0,
                        "version_id": 456,
                    },
                },
                FakeManager(),
            )


if __name__ == "__main__":
    unittest.main()
