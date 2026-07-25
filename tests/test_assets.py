from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from tenminvideomaker.assets import (
    AssetAuthenticationRequired,
    CivitaiLoraMetadata,
    LocalLoraRequirement,
    LoraAssetManager,
    predictable_lora_filename,
)
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

    def manager(self, downloader=None, **kwargs) -> LoraAssetManager:
        return LoraAssetManager(
            [self.lora_directory],
            self.manifest_path,
            retries=2,
            retry_delay_seconds=0,
            downloader=downloader,
            **kwargs,
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

    def test_civitai_metadata_finds_existing_canonical_filename(self) -> None:
        canonical = self.lora_directory / "publisher-original-name.safetensors"
        canonical.parent.mkdir(parents=True)
        canonical.write_bytes(b"weights")
        metadata = CivitaiLoraMetadata(
            version_id=3184055,
            model_id=2847192,
            filename=canonical.name,
            download_url="https://civitai.com/api/download/models/3184055",
            sha256=None,
            size_bytes=7,
        )
        lora = LoraSpec(
            "Elsa Frozen Anima",
            metadata.download_url,
            0.85,
            model_id=metadata.model_id,
            version_id=metadata.version_id,
        )
        result = self.manager(
            metadata_fetcher=lambda _lora: metadata,
            visible_lora_names=(canonical.name,),
        ).resolve_or_download(lora)
        self.assertTrue(result.succeeded)
        self.assertFalse(result.downloaded)
        self.assertEqual(result.path, canonical)
        self.assertEqual(result.local_filename, canonical.name)

    def test_existing_anima_lora_is_blocked_from_ltx_resolution(self) -> None:
        installed = self.lora_directory / "Character_Anima.safetensors"
        installed.parent.mkdir(parents=True)
        installed.write_bytes(b"weights")
        metadata = CivitaiLoraMetadata(
            version_id=123,
            model_id=456,
            filename=installed.name,
            download_url="https://civitai.com/api/download/models/123",
            sha256=None,
            size_bytes=7,
            base_model="Anima",
        )
        result = self.manager(
            metadata_fetcher=lambda _lora: metadata,
        ).resolve_or_download(
            LoraSpec("Character Anima", metadata.download_url, 0.8, version_id=123),
            expected_base_model="LTXV 2.3",
        )
        self.assertFalse(result.succeeded)
        self.assertIn("Anima LoRA, not LTXV 2.3", result.error)

    def test_existing_ltxv_23_lora_passes_family_validation(self) -> None:
        installed = self.lora_directory / "Motion.safetensors"
        installed.parent.mkdir(parents=True)
        installed.write_bytes(b"weights")
        metadata = CivitaiLoraMetadata(
            version_id=123,
            model_id=456,
            filename=installed.name,
            download_url="https://civitai.com/api/download/models/123",
            sha256=None,
            size_bytes=7,
            base_model="LTXV 2.3",
        )
        result = self.manager(
            metadata_fetcher=lambda _lora: metadata,
        ).resolve_or_download(
            LoraSpec("Motion", metadata.download_url, 0.8, version_id=123),
            expected_base_model="LTXV 2.3",
        )
        self.assertTrue(result.succeeded)
        self.assertEqual(result.base_model, "LTXV 2.3")

    def test_authentication_failure_is_not_retried(self) -> None:
        calls = []

        def downloader(_url: str, _destination: Path) -> None:
            calls.append("attempt")
            raise AssetAuthenticationRequired("token required")

        result = self.manager(downloader).resolve_or_download(
            LoraSpec("Private", "https://example.com/private", 1.0)
        )
        self.assertFalse(result.succeeded)
        self.assertEqual(calls, ["attempt"])
        self.assertEqual(result.error, "token required")

    def test_hash_mismatch_removes_download_and_reports_failure(self) -> None:
        def downloader(_url: str, destination: Path) -> None:
            destination.write_bytes(b"not expected")

        metadata = CivitaiLoraMetadata(
            version_id=123,
            model_id=456,
            filename="source.safetensors",
            download_url="https://civitai.com/api/download/models/123",
            sha256="0" * 64,
            size_bytes=None,
        )
        result = self.manager(
            downloader,
            metadata_fetcher=lambda _lora: metadata,
        ).resolve_or_download(
            LoraSpec("Hash Checked", metadata.download_url, 1.0, version_id=123)
        )
        self.assertFalse(result.succeeded)
        self.assertIn("SHA-256 mismatch", result.error)
        self.assertFalse((self.lora_directory / "Hash_Checked.safetensors").exists())

    def test_civitai_token_is_added_only_to_civitai_downloads(self) -> None:
        manager = self.manager(civitai_token="secret token")
        authenticated = manager._authenticated_download_url(
            "https://civitai.com/api/download/models/123?type=Model&token=old"
        )
        self.assertIn("token=secret+token", authenticated)
        self.assertNotIn("token=old", authenticated)
        other = "https://example.com/model.safetensors"
        self.assertEqual(manager._authenticated_download_url(other), other)

    def test_metadata_filename_cannot_escape_active_lora_root(self) -> None:
        outside = self.root / "outside.safetensors"
        outside.write_bytes(b"untrusted")

        def downloader(_url: str, destination: Path) -> None:
            destination.write_bytes(b"downloaded")

        metadata = CivitaiLoraMetadata(
            version_id=123,
            model_id=456,
            filename="../outside.safetensors",
            download_url="https://civitai.com/api/download/models/123",
            sha256=None,
            size_bytes=None,
        )
        result = self.manager(
            downloader,
            metadata_fetcher=lambda _lora: metadata,
        ).resolve_or_download(
            LoraSpec("Safe Destination", metadata.download_url, 1.0, version_id=123)
        )
        self.assertTrue(result.succeeded)
        self.assertTrue(result.downloaded)
        self.assertEqual(result.path, self.lora_directory / "Safe_Destination.safetensors")
