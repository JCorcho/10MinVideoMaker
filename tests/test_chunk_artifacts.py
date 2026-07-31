from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from tenminvideomaker.chunk_artifacts import (
    ChunkArtifactError,
    latent_checkpoint_is_valid,
    load_latent_checkpoint,
    save_latent_checkpoint,
    sha256_file,
)
from tenminvideomaker.storage import StorageError, StorageLayout


class ChunkArtifactTests(unittest.TestCase):
    coordinates = {
        "job_id": "job-20260729",
        "scene_id": 2,
        "revision": 3,
        "chunk_index": 4,
        "attempt_number": 1,
        "artifact_kind": "stage1_handoff",
    }

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.layout = StorageLayout(Path(self.temporary_directory.name))

    @staticmethod
    def latent() -> dict[str, object]:
        samples = torch.arange(
            1 * 128 * 3 * 2 * 2,
            dtype=torch.float32,
        ).reshape(1, 128, 3, 2, 2)
        return {
            "samples": samples,
            "noise_mask": torch.full(
                (1, 1, 3, 2, 2),
                0.25,
                dtype=torch.float16,
            ),
            "batch_index": torch.tensor([7], dtype=torch.int64),
            "downscale_ratio_spacial": 0.5,
        }

    def checkpoint_path(self) -> Path:
        return self.layout.chunk_checkpoint_path(**self.coordinates)

    def manifest_path(self) -> Path:
        return self.layout.chunk_checkpoint_manifest_path(**self.coordinates)

    def test_round_trip_preserves_samples_and_allowed_metadata(self) -> None:
        original = self.latent()

        checkpoint, manifest = save_latent_checkpoint(
            self.layout,
            original,
            **self.coordinates,
        )
        loaded, loaded_manifest = load_latent_checkpoint(
            self.layout,
            **self.coordinates,
        )

        self.assertEqual(checkpoint, self.checkpoint_path())
        self.assertTrue(checkpoint.is_relative_to(self.layout.root))
        self.assertEqual(loaded_manifest, manifest)
        self.assertEqual(set(loaded), set(original))
        for name in ("samples", "noise_mask", "batch_index"):
            self.assertTrue(torch.equal(loaded[name], original[name]))
            self.assertEqual(loaded[name].device.type, "cpu")
        self.assertEqual(loaded["downscale_ratio_spacial"], 0.5)
        self.assertEqual(manifest["sha256"], sha256_file(checkpoint))
        self.assertEqual(manifest["byte_size"], checkpoint.stat().st_size)
        self.assertEqual(
            manifest["tensors"]["samples"],
            {"shape": [1, 128, 3, 2, 2], "dtype": "torch.float32"},
        )
        self.assertEqual(
            manifest["scalar_metadata"],
            {"downscale_ratio_spacial": 0.5},
        )

    def test_same_size_checkpoint_tamper_is_rejected_by_hash(self) -> None:
        save_latent_checkpoint(
            self.layout,
            self.latent(),
            **self.coordinates,
        )
        checkpoint = self.checkpoint_path()
        content = bytearray(checkpoint.read_bytes())
        content[-1] ^= 0x01
        checkpoint.write_bytes(content)

        with self.assertRaisesRegex(
            ChunkArtifactError,
            "SHA-256 verification failed",
        ):
            load_latent_checkpoint(self.layout, **self.coordinates)
        self.assertFalse(
            latent_checkpoint_is_valid(self.layout, **self.coordinates)
        )

    def test_expected_temporal_token_count_is_enforced(self) -> None:
        save_latent_checkpoint(
            self.layout,
            self.latent(),
            **self.coordinates,
        )
        loaded, _manifest = load_latent_checkpoint(
            self.layout,
            **self.coordinates,
            expected_temporal_tokens=3,
        )
        self.assertEqual(int(loaded["samples"].shape[2]), 3)
        with self.assertRaisesRegex(
            ChunkArtifactError,
            "temporal shape",
        ):
            load_latent_checkpoint(
                self.layout,
                **self.coordinates,
                expected_temporal_tokens=4,
            )
        self.assertFalse(
            latent_checkpoint_is_valid(
                self.layout,
                **self.coordinates,
                expected_temporal_tokens=4,
            )
        )

    def test_audio_latent_round_trip_uses_descriptor_and_hash_validation(self) -> None:
        coordinates = {
            **self.coordinates,
            "artifact_kind": "stage2_audio",
        }
        audio = {
            "samples": torch.arange(
                1 * 8 * 12 * 16,
                dtype=torch.float32,
            ).reshape(1, 8, 12, 16)
        }

        checkpoint, manifest = save_latent_checkpoint(
            self.layout,
            audio,
            **coordinates,
        )
        loaded, loaded_manifest = load_latent_checkpoint(
            self.layout,
            **coordinates,
            # This field is video-only and must not be applied to audio shape.
            expected_temporal_tokens=999,
        )

        self.assertTrue(torch.equal(loaded["samples"], audio["samples"]))
        self.assertEqual(loaded_manifest, manifest)
        self.assertEqual(manifest["sha256"], sha256_file(checkpoint))
        self.assertTrue(
            latent_checkpoint_is_valid(
                self.layout,
                **coordinates,
                expected_temporal_tokens=1,
            )
        )

    def test_missing_checkpoint_or_manifest_is_never_finalized(self) -> None:
        save_latent_checkpoint(
            self.layout,
            self.latent(),
            **self.coordinates,
        )
        self.manifest_path().unlink()
        with self.assertRaisesRegex(
            ChunkArtifactError,
            "incomplete or missing",
        ):
            load_latent_checkpoint(self.layout, **self.coordinates)

        save_latent_checkpoint(
            self.layout,
            self.latent(),
            **self.coordinates,
        )
        self.checkpoint_path().unlink()
        with self.assertRaisesRegex(
            ChunkArtifactError,
            "incomplete or missing",
        ):
            load_latent_checkpoint(self.layout, **self.coordinates)

    def test_unreadable_and_identity_mismatched_manifests_are_rejected(self) -> None:
        save_latent_checkpoint(
            self.layout,
            self.latent(),
            **self.coordinates,
        )
        self.manifest_path().write_text("{not-json", encoding="utf-8")
        with self.assertRaisesRegex(ChunkArtifactError, "manifest is unreadable"):
            load_latent_checkpoint(self.layout, **self.coordinates)

        _, manifest = save_latent_checkpoint(
            self.layout,
            self.latent(),
            **self.coordinates,
        )
        mismatched = dict(manifest)
        mismatched["chunk_index"] = self.coordinates["chunk_index"] + 1
        self.manifest_path().write_text(
            json.dumps(mismatched),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            ChunkArtifactError,
            "chunk_index does not match",
        ):
            load_latent_checkpoint(self.layout, **self.coordinates)

    def test_manifest_descriptors_and_scalar_metadata_are_verified(self) -> None:
        _, manifest = save_latent_checkpoint(
            self.layout,
            self.latent(),
            **self.coordinates,
        )
        descriptor_tamper = dict(manifest)
        descriptor_tamper["tensors"] = {
            **manifest["tensors"],
            "samples": {
                "shape": [1, 128, 99, 2, 2],
                "dtype": "torch.float32",
            },
        }
        self.manifest_path().write_text(
            json.dumps(descriptor_tamper),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            ChunkArtifactError,
            "tensor descriptors do not match",
        ):
            load_latent_checkpoint(self.layout, **self.coordinates)

        invalid_scalars = dict(manifest)
        invalid_scalars["scalar_metadata"] = []
        self.manifest_path().write_text(
            json.dumps(invalid_scalars),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            ChunkArtifactError,
            "scalar metadata is invalid",
        ):
            load_latent_checkpoint(self.layout, **self.coordinates)

    def test_checkpoint_without_successfully_written_manifest_is_not_valid(self) -> None:
        self.assertFalse(
            latent_checkpoint_is_valid(self.layout, **self.coordinates)
        )
        with patch(
            "tenminvideomaker.chunk_artifacts.write_json_atomic",
            side_effect=OSError("simulated manifest write failure"),
        ):
            with self.assertRaisesRegex(OSError, "manifest write failure"):
                save_latent_checkpoint(
                    self.layout,
                    self.latent(),
                    **self.coordinates,
                )

        self.assertTrue(self.checkpoint_path().is_file())
        self.assertFalse(self.manifest_path().exists())
        self.assertFalse(
            latent_checkpoint_is_valid(self.layout, **self.coordinates)
        )

        save_latent_checkpoint(
            self.layout,
            self.latent(),
            **self.coordinates,
        )
        self.assertTrue(
            latent_checkpoint_is_valid(self.layout, **self.coordinates)
        )
        self.assertEqual(
            list(self.checkpoint_path().parent.glob("*.tmp")),
            [],
        )

    def test_unknown_latent_keys_are_rejected(self) -> None:
        invalid = self.latent()
        invalid["audio"] = torch.zeros(1)

        with self.assertRaisesRegex(
            ChunkArtifactError,
            "unsupported keys: audio",
        ):
            save_latent_checkpoint(
                self.layout,
                invalid,
                **self.coordinates,
            )

    def test_invalid_sample_shapes_and_dtypes_are_rejected(self) -> None:
        invalid_samples = (
            torch.zeros((1, 128, 3, 2)),
            torch.zeros((2, 128, 3, 2, 2)),
            torch.zeros((1, 64, 3, 2, 2)),
            torch.zeros((1, 128, 0, 2, 2)),
        )
        for samples in invalid_samples:
            with self.subTest(shape=tuple(samples.shape)):
                with self.assertRaises(ChunkArtifactError):
                    save_latent_checkpoint(
                        self.layout,
                        {"samples": samples},
                        **self.coordinates,
                    )

        with self.assertRaisesRegex(
            ChunkArtifactError,
            "floating-point dtype",
        ):
            save_latent_checkpoint(
                self.layout,
                {
                    "samples": torch.zeros(
                        (1, 128, 3, 2, 2),
                        dtype=torch.int64,
                    )
                },
                **self.coordinates,
            )

    def test_invalid_optional_metadata_is_rejected(self) -> None:
        invalid_noise_masks = (
            "not-a-tensor",
            torch.zeros((1, 1, 3, 2)),
            torch.zeros((2, 1, 3, 2, 2)),
            torch.zeros((1, 2, 3, 2, 2)),
            torch.zeros((1, 1, 4, 2, 2)),
        )
        for noise_mask in invalid_noise_masks:
            invalid = self.latent()
            invalid["noise_mask"] = noise_mask
            with self.subTest(noise_mask=getattr(noise_mask, "shape", noise_mask)):
                with self.assertRaises(ChunkArtifactError):
                    save_latent_checkpoint(
                        self.layout,
                        invalid,
                        **self.coordinates,
                    )

        invalid_batch_indices = (
            "not-a-tensor",
            torch.zeros((1, 1), dtype=torch.int64),
        )
        for batch_index in invalid_batch_indices:
            invalid = self.latent()
            invalid["batch_index"] = batch_index
            with self.subTest(batch_index=getattr(batch_index, "shape", batch_index)):
                with self.assertRaisesRegex(ChunkArtifactError, "batch_index"):
                    save_latent_checkpoint(
                        self.layout,
                        invalid,
                        **self.coordinates,
                    )

        for downscale in (
            True,
            0,
            -0.5,
            "0.5",
            float("nan"),
            float("inf"),
            float("-inf"),
        ):
            invalid = self.latent()
            invalid["downscale_ratio_spacial"] = downscale
            with self.subTest(downscale=downscale):
                with self.assertRaisesRegex(
                    ChunkArtifactError,
                    "positive number",
                ):
                    save_latent_checkpoint(
                        self.layout,
                        invalid,
                        **self.coordinates,
                    )

    def test_invalid_path_coordinates_cannot_escape_storage_layout(self) -> None:
        invalid_coordinate_sets = (
            {"job_id": "../outside"},
            {"job_id": r"job\outside"},
            {"scene_id": 0},
            {"scene_id": True},
            {"revision": 0},
            {"revision": True},
            {"chunk_index": -1},
            {"chunk_index": True},
            {"attempt_number": 0},
            {"attempt_number": True},
            {"artifact_kind": "../checkpoint"},
        )
        for override in invalid_coordinate_sets:
            coordinates = {**self.coordinates, **override}
            with self.subTest(override=override):
                with self.assertRaises(StorageError):
                    save_latent_checkpoint(
                        self.layout,
                        self.latent(),
                        **coordinates,
                    )


if __name__ == "__main__":
    unittest.main()
