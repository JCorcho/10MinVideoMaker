from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tenminvideomaker.constants import (
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
        *(filename for filename, _weight in MANDATORY_I2V_LORAS),
    )
    return {
        "schema_version": 4,
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
            "exact_frame_handoff": {
                "completed": True,
                "peak_vram_bytes": 15_000_000_000,
                "chunk_count": 2,
                "stage2_video_spatial_tokens": [[42, 24], [42, 24]],
                "assembled_profile": {
                    "width": 768,
                    "height": 1344,
                    "fps": "24",
                    "decoded_video_frames": 240,
                },
                "handoff": {
                    "source_frame_index": 120,
                    "continuation_dropped_frame": 0,
                    "pixel_exact": True,
                    "rgb_mae": 0.0,
                },
                "spatial_detail": {
                    "base_boundary": {"laplacian_variance": 120.0},
                    "continuation_first_new": {"laplacian_variance": 96.0},
                    "detail_retention_ratio": 0.8,
                },
                "visual_review": "Sharp, continuous safe fixture with stable identity.",
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
            "native_full_resolution_video": True,
            "exact_final_frame_handoff": True,
            "realism_detail_preserved": True,
            "no_unusable_blur": True,
            "seam_continuity_approved": True,
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

    def test_real_esrgan_is_not_a_required_continuation_asset(self) -> None:
        document = approved_document()
        self.assertNotIn("RealESRGAN_x2.pth", document["external_assets"])
        validate_auto_rollout_manifest(
            document,
            implementation_sha256=IMPLEMENTATION_HASH,
            node_contracts_sha256=CONTRACT_HASH,
        )

    def test_half_resolution_stage2_generation_fails_closed(self) -> None:
        document = approved_document()
        document["generations"]["exact_frame_handoff"][
            "stage2_video_spatial_tokens"
        ] = [[42, 24], [21, 12]]
        with self.assertRaisesRegex(
            ContinuationRolloutError,
            "native 42x24",
        ):
            validate_auto_rollout_manifest(
                document,
                implementation_sha256=IMPLEMENTATION_HASH,
                node_contracts_sha256=CONTRACT_HASH,
            )

    def test_missing_detail_metrics_fails_closed(self) -> None:
        document = approved_document()
        del document["generations"]["exact_frame_handoff"]["spatial_detail"]
        with self.assertRaisesRegex(ContinuationRolloutError, "spatial detail"):
            validate_auto_rollout_manifest(
                document,
                implementation_sha256=IMPLEMENTATION_HASH,
                node_contracts_sha256=CONTRACT_HASH,
            )

    def test_non_exact_handoff_fails_closed(self) -> None:
        document = approved_document()
        document["generations"]["exact_frame_handoff"]["handoff"][
            "rgb_mae"
        ] = 0.01
        with self.assertRaisesRegex(ContinuationRolloutError, "pixel-exact"):
            validate_auto_rollout_manifest(
                document,
                implementation_sha256=IMPLEMENTATION_HASH,
                node_contracts_sha256=CONTRACT_HASH,
            )

    def test_low_detail_retention_fails_closed(self) -> None:
        document = approved_document()
        detail = document["generations"]["exact_frame_handoff"]["spatial_detail"]
        detail["continuation_first_new"]["laplacian_variance"] = 60.0
        detail["detail_retention_ratio"] = 0.5
        with self.assertRaisesRegex(ContinuationRolloutError, "70%"):
            validate_auto_rollout_manifest(
                document,
                implementation_sha256=IMPLEMENTATION_HASH,
                node_contracts_sha256=CONTRACT_HASH,
            )

    def test_wrong_assembled_profile_fails_closed(self) -> None:
        document = approved_document()
        document["generations"]["exact_frame_handoff"]["assembled_profile"][
            "width"
        ] = 704
        with self.assertRaisesRegex(ContinuationRolloutError, "768x1344"):
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
                "continuation-validation-v4.json",
            )


if __name__ == "__main__":
    unittest.main()
