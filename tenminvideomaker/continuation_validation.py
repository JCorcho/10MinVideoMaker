"""Fail-closed approval gate for automatic LTX continuation rollout."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

from .constants import (
    I2V_SPATIAL_UPSCALER,
    MANDATORY_I2V_LORAS,
)
from .continuation import CONTINUATION_STRATEGY
from .storage import StorageLayout
from .workflow_builder import LTX_CHECKPOINT, LTX_TEXT_ENCODER


VALIDATION_SCHEMA_VERSION = 4
VALIDATION_FILENAME = "continuation-validation-v4.json"
MINIMUM_DETAIL_RETENTION_RATIO = 0.70
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_DECISIONS = (
    "no_oom",
    "lcm_guider_validated",
    "production_seam_motion_continuous",
    "style_identity_preserved",
    "anatomy_stable",
    "audio_video_profile_validated",
    "runtime_acceptable",
    "native_full_resolution_video",
    "exact_final_frame_handoff",
    "realism_detail_preserved",
    "no_unusable_blur",
    "seam_continuity_approved",
)
_REQUIRED_GENERATIONS = ("exact_frame_handoff",)
_REQUIRED_EXTERNAL_ASSETS = (
    LTX_CHECKPOINT,
    LTX_TEXT_ENCODER,
    I2V_SPATIAL_UPSCALER,
    *(filename for filename, _weight in MANDATORY_I2V_LORAS),
)


class ContinuationRolloutError(RuntimeError):
    """Raised when automatic rollout lacks bounded human-approved evidence."""


def validation_manifest_path(storage: StorageLayout) -> Path:
    """Return project-owned D-drive approval path."""
    return storage.state_root / VALIDATION_FILENAME


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value.strip().casefold()) is None:
        raise ContinuationRolloutError(f"{field} must be a lowercase SHA-256 value.")
    return value.strip().casefold()


def _positive_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and value > 0
    )


def validate_auto_rollout_manifest(
    document: Mapping[str, Any],
    *,
    implementation_sha256: str,
    node_contracts_sha256: str,
) -> None:
    """Validate bounded GPU evidence and bind it to current implementation."""
    if document.get("schema_version") != VALIDATION_SCHEMA_VERSION:
        raise ContinuationRolloutError("Continuation validation schema version is not current.")
    if document.get("strategy") != CONTINUATION_STRATEGY:
        raise ContinuationRolloutError("Continuation validation strategy does not match runtime.")
    if document.get("status") != "approved":
        raise ContinuationRolloutError("Continuation validation is not approved.")
    if _require_sha256(
        document.get("implementation_sha256"),
        "implementation_sha256",
    ) != implementation_sha256:
        raise ContinuationRolloutError(
            "Continuation implementation changed after bounded validation."
        )
    if _require_sha256(
        document.get("node_contracts_sha256"),
        "node_contracts_sha256",
    ) != node_contracts_sha256:
        raise ContinuationRolloutError(
            "Live ComfyUI node contracts changed after bounded validation."
        )

    reviewer = document.get("reviewer")
    completed_at = document.get("completed_at")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ContinuationRolloutError("Continuation validation needs a reviewer.")
    if not isinstance(completed_at, str) or not completed_at.strip():
        raise ContinuationRolloutError("Continuation validation needs a completion timestamp.")

    external_assets = document.get("external_assets")
    if not isinstance(external_assets, Mapping):
        raise ContinuationRolloutError("Continuation validation needs external asset hashes.")
    for filename in _REQUIRED_EXTERNAL_ASSETS:
        entry = external_assets.get(filename)
        if not isinstance(entry, Mapping):
            raise ContinuationRolloutError(
                f"Continuation validation is missing external asset {filename}."
            )
        _require_sha256(entry.get("sha256"), f"external_assets.{filename}.sha256")
        for field in ("source", "license"):
            value = entry.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ContinuationRolloutError(
                    f"external_assets.{filename}.{field} is required."
                )

    generations = document.get("generations")
    if not isinstance(generations, Mapping):
        raise ContinuationRolloutError("Continuation validation needs generation results.")
    for name in _REQUIRED_GENERATIONS:
        result = generations.get(name)
        if not isinstance(result, Mapping) or result.get("completed") is not True:
            raise ContinuationRolloutError(
                f"Continuation validation generation {name} is incomplete."
            )
    exact = generations["exact_frame_handoff"]
    chunk_count = exact.get("chunk_count")
    if isinstance(chunk_count, bool) or not isinstance(chunk_count, int) or chunk_count < 2:
        raise ContinuationRolloutError(
            "Exact-frame validation needs at least two completed chunks."
        )
    stage2_tokens = exact.get("stage2_video_spatial_tokens")
    if (
        not isinstance(stage2_tokens, list)
        or len(stage2_tokens) != chunk_count
        or any(tokens != [42, 24] for tokens in stage2_tokens)
    ):
        raise ContinuationRolloutError(
            "Exact-frame validation must use native 42x24 stage-two video tokens "
            "for every chunk."
        )

    peak_vram = exact.get("peak_vram_bytes")
    if isinstance(peak_vram, bool) or not isinstance(peak_vram, int) or peak_vram < 1:
        raise ContinuationRolloutError(
            "Exact-frame validation needs positive peak_vram_bytes."
        )

    profile = exact.get("assembled_profile")
    if not isinstance(profile, Mapping):
        raise ContinuationRolloutError(
            "Exact-frame validation needs an assembled 768x1344 profile."
        )
    if profile.get("width") != 768 or profile.get("height") != 1344:
        raise ContinuationRolloutError(
            "Exact-frame validation output must be 768x1344."
        )
    if str(profile.get("fps")) not in {"24", "24/1"}:
        raise ContinuationRolloutError("Exact-frame validation output must be 24 fps.")
    expected_frames = chunk_count * 120
    if profile.get("decoded_video_frames") != expected_frames:
        raise ContinuationRolloutError(
            "Exact-frame validation must prove 120 assembled output frames per "
            "bounded validation chunk."
        )

    handoff = exact.get("handoff")
    rgb_mae = handoff.get("rgb_mae") if isinstance(handoff, Mapping) else None
    if (
        not isinstance(handoff, Mapping)
        or handoff.get("source_frame_index") != 120
        or handoff.get("continuation_dropped_frame") != 0
        or handoff.get("pixel_exact") is not True
        or isinstance(rgb_mae, bool)
        or not isinstance(rgb_mae, (int, float))
        or rgb_mae != 0
    ):
        raise ContinuationRolloutError(
            "Exact-frame validation needs a pixel-exact frame-120 handoff and "
            "must drop duplicate continuation frame zero."
        )

    spatial_detail = exact.get("spatial_detail")
    if not isinstance(spatial_detail, Mapping):
        raise ContinuationRolloutError(
            "Exact-frame validation needs spatial detail measurements."
        )
    measured: dict[str, float] = {}
    for frame_name in ("base_boundary", "continuation_first_new"):
        frame_detail = spatial_detail.get(frame_name)
        laplacian = (
            frame_detail.get("laplacian_variance")
            if isinstance(frame_detail, Mapping)
            else None
        )
        if (
            not _positive_number(laplacian)
        ):
            raise ContinuationRolloutError(
                "Exact-frame validation needs positive spatial detail "
                f"measurement for {frame_name}."
            )
        measured[frame_name] = float(laplacian)
    retention = measured["continuation_first_new"] / measured["base_boundary"]
    reported_retention = spatial_detail.get("detail_retention_ratio")
    if (
        not _positive_number(reported_retention)
        or abs(float(reported_retention) - retention) > 0.01
        or retention < MINIMUM_DETAIL_RETENTION_RATIO
    ):
        raise ContinuationRolloutError(
            "Exact-frame continuation must retain at least 70% of boundary "
            "Laplacian detail."
        )
    visual_review = exact.get("visual_review")
    if not isinstance(visual_review, str) or not visual_review.strip():
        raise ContinuationRolloutError(
            "Exact-frame validation needs a non-empty safe-fixture visual review."
        )

    decision = document.get("decision")
    if not isinstance(decision, Mapping):
        raise ContinuationRolloutError("Continuation validation needs decision results.")
    for name in _REQUIRED_DECISIONS:
        if decision.get(name) is not True:
            raise ContinuationRolloutError(
                f"Continuation validation decision {name} must be accepted."
            )


def require_auto_rollout_approval(
    storage: StorageLayout,
    *,
    implementation_sha256: str,
    node_contracts_sha256: str,
) -> Path:
    """Load and validate auto-rollout evidence; explicit mode bypasses this."""
    path = validation_manifest_path(storage)
    try:
        import json

        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ContinuationRolloutError(
            f"Automatic continuation is locked until bounded validation is approved at {path}."
        ) from error
    except (OSError, ValueError) as error:
        raise ContinuationRolloutError(
            f"Continuation validation manifest is unreadable: {path}."
        ) from error
    if not isinstance(document, Mapping):
        raise ContinuationRolloutError("Continuation validation manifest must be an object.")
    validate_auto_rollout_manifest(
        document,
        implementation_sha256=implementation_sha256,
        node_contracts_sha256=node_contracts_sha256,
    )
    return path
