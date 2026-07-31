from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tenminvideomaker.constants import (
    CONTINUATION_VIDEO_UPSCALER,
    I2V_SPATIAL_UPSCALER,
    MANDATORY_I2V_LORAS,
)
from tenminvideomaker.continuation import CONTINUATION_STRATEGY
from tenminvideomaker.continuation_validation import (
    ContinuationRolloutError,
    require_auto_rollout_approval,
    validate_auto_rollout_manifest,
    validation_manifest_path,
)
from tenminvideomaker.storage import StorageLayout
from tenminvideomaker.workflow_builder import LTX_CHECKPOINT, LTX_TEXT_ENCODER


IMPLEMENTATION_HASH = "1" * 64
CONTRACT_HASH = "2" * 64


def approved_document() -> dict[str, object]:
    filenames = (
        LTX_CHECKPOINT,
        LTX_TEXT_ENCODER,
        I2V_SPATIAL_UPSCALER,
        CONTINUATION_VIDEO_UPSCALER,
        *(filename for filename, _weight in MANDATORY_I2V_LORAS),
    )
    return {
        "schema_version": 2,
        "strategy": CONTINUATION_STRATEGY,
        "status": "approved",
        "implementation_sha256": IMPLEMENTATION_HASH,
        "node_contracts_sha256": CONTRACT_HASH,
        "reviewer": "owner",
        "completed_at": "2026-07-30T00:00:00Z",
        "external_assets": {
            filename: {
                "sha256": "3" * 64,
                "source": "local verified source",
                "license": "owner verified",
            }
            for filename in filenames
        },
        "generations": {
            "common_base": {"completed": True},
            "single_frame": {"completed": True},
            "decoded_17_frame": {"completed": True},
            "latent_overlap": {
                "completed": True,
                "peak_vram_bytes": 15_000_000_000,
            },
        },
        "decision": {
            "no_oom": True,
            "lcm_guider_validated": True,
            "production_seam_motion_continuous": True,
            "style_identity_preserved": True,
            "anatomy_stable": True,
            "audio_video_profile_validated": True,
            "runtime_acceptable": True,
        },
    }


class ContinuationValidationTests(unittest.TestCase):
    def test_approved_manifest_matches_current_runtime_identity(self) -> None:
        validate_auto_rollout_manifest(
            approved_document(),
            implementation_sha256=IMPLEMENTATION_HASH,
            node_contracts_sha256=CONTRACT_HASH,
        )

    def test_changed_implementation_fails_closed(self) -> None:
        with self.assertRaisesRegex(
            ContinuationRolloutError,
            "implementation changed",
        ):
            validate_auto_rollout_manifest(
                approved_document(),
                implementation_sha256="4" * 64,
                node_contracts_sha256=CONTRACT_HASH,
            )

    def test_missing_bounded_decision_fails_closed(self) -> None:
        document = approved_document()
        document["decision"]["style_identity_preserved"] = False
        with self.assertRaisesRegex(
            ContinuationRolloutError,
            "style_identity_preserved",
        ):
            validate_auto_rollout_manifest(
                document,
                implementation_sha256=IMPLEMENTATION_HASH,
                node_contracts_sha256=CONTRACT_HASH,
            )

    def test_missing_style_stable_upscaler_hash_fails_closed(self) -> None:
        document = approved_document()
        del document["external_assets"][CONTINUATION_VIDEO_UPSCALER]
        with self.assertRaisesRegex(
            ContinuationRolloutError,
            "missing external asset RealESRGAN_x2.pth",
        ):
            validate_auto_rollout_manifest(
                document,
                implementation_sha256=IMPLEMENTATION_HASH,
                node_contracts_sha256=CONTRACT_HASH,
            )

    def test_auto_approval_reads_only_durable_storage_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            storage = StorageLayout(Path(temporary))
            storage.ensure()
            path = validation_manifest_path(storage)
            path.write_text(json.dumps(approved_document()), encoding="utf-8")
            self.assertEqual(
                require_auto_rollout_approval(
                    storage,
                    implementation_sha256=IMPLEMENTATION_HASH,
                    node_contracts_sha256=CONTRACT_HASH,
                ),
                path,
            )

    def test_missing_manifest_blocks_auto_rollout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            storage = StorageLayout(Path(temporary))
            storage.ensure()
            with self.assertRaisesRegex(
                ContinuationRolloutError,
                "locked until bounded validation",
            ):
                require_auto_rollout_approval(
                    storage,
                    implementation_sha256=IMPLEMENTATION_HASH,
                    node_contracts_sha256=CONTRACT_HASH,
                )

    def test_validation_manifest_uses_current_schema_filename(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            storage = StorageLayout(Path(temporary))
            self.assertEqual(
                validation_manifest_path(storage).name,
                "continuation-validation-v2.json",
            )


if __name__ == "__main__":
    unittest.main()
