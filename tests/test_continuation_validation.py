from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tenminvideomaker.constants import I2V_SPATIAL_UPSCALER, MANDATORY_I2V_LORAS
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
        "schema_version": 1,
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
            "decoded_25_frame": {"completed": True},
            "latent_overlap": {
                "completed": True,
                "peak_vram_bytes": 15_000_000_000,
            },
        },
        "decision": {
            "no_oom": True,
            "lcm_guider_validated": True,
            "lower_flow_discontinuity_than_single_frame": True,
            "anatomy_not_worse_than_25_frame": True,
            "second_pass_seam_not_worse": True,
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
        document["decision"]["anatomy_not_worse_than_25_frame"] = False
        with self.assertRaisesRegex(
            ContinuationRolloutError,
            "anatomy_not_worse",
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


if __name__ == "__main__":
    unittest.main()
