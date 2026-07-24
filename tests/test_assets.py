from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from tenminvideomaker.assets import LocalLoraRequirement, LoraAssetManager, predictable_lora_filename
from tenminvideomaker.contracts import LoraSpec


class AssetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.lora_directory = self.root / "loras"
        self.manifest_path = self.root / "runtime" / "assets.json"
        self.calls: list[tuple[str, Path]] = []

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def manager(self, downloader=None) -> LoraAssetManager:
        return LoraAssetManager(
            [self.lora_directory],
            self.manifest_path,
            retries=2,
            retry_delay_seconds=0,
            downloader=downloader,
        )

    def test_predictable_filename_is_safe(self) -> None:
        self.assertEqual(predictable_lora_filename("Snow / Atmosphere"), "Snow_Atmosphere.safetensors")

    def test_missing_lora_downloads_once_and_records_manifest(self) -> None:
        def downloader(url: str, destination: Path) -> None:
            self.calls.append((url, destination))
            destination.write_bytes(b"weights")

        lora = LoraSpec("Snow Atmosphere", "https://example.com/snow", 0.7)
        result = self.manager(downloader).resolve_or_download(lora)
        self.assertTrue(result.succeeded)
        self.assertTrue(result.downloaded)
        self.assertEqual(result.path.name, "Snow_Atmosphere.safetensors")
        self.assertEqual(len(self.calls), 1)
        second = self.manager(downloader).resolve_or_download(lora)
        self.assertTrue(second.succeeded)
        self.assertFalse(second.downloaded)
        self.assertEqual(len(self.calls), 1)

    def test_failed_lora_returns_per_asset_error_after_retries(self) -> None:
        def downloader(_url: str, _destination: Path) -> None:
            raise OSError("offline")

        result = self.manager(downloader).resolve_or_download(LoraSpec("Broken", "https://example.com/broken", 1.0))
        self.assertFalse(result.succeeded)
        self.assertIn("after 2 attempts", result.error)

    def test_required_hardcoded_lora_is_not_downloaded(self) -> None:
        result = self.manager().require_local(LocalLoraRequirement("DMD.safetensors", 1.0))
        self.assertFalse(result.succeeded)
        self.assertIn("Required local LoRA is missing", result.error)
