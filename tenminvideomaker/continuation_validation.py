"""Fail-closed approval gate for automatic LTX continuation rollout."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

from .constants import I2V_SPATIAL_UPSCALER, MANDATORY_I2V_LORAS
from .continuation import CONTINUATION_STRATEGY
from .storage import StorageLayout
from .workflow_builder import LTX_CHECKPOINT, LTX_TEXT_ENCODER


VALIDATION_SCHEMA_VERSION = 1
VALIDATION_FILENAME = "continuation-validation-v1.json"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_DECISIONS = (
    "no_oom",
    "lcm_guider_validated",
    "lower_flow_discontinuity_than_single_frame",
    "anatomy_not_worse_than_25_frame",
    "second_pass_seam_not_worse",
    "runtime_acceptable",
)
_REQUIRED_GENERATIONS = (
    "common_base",
    "single_frame",
    "decoded_17_frame",
    "latent_overlap",
)
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
        raise ContinuationRolloutError("Continuation validation needs four generation results.")
    for name in _REQUIRED_GENERATIONS:
        result = generations.get(name)
        if not isinstance(result, Mapping) or result.get("completed") is not True:
            raise ContinuationRolloutError(
                f"Continuation validation generation {name} is incomplete."
            )
    latent_overlap = generations["latent_overlap"]
    peak_vram = latent_overlap.get("peak_vram_bytes")
    if isinstance(peak_vram, bool) or not isinstance(peak_vram, int) or peak_vram < 1:
        raise ContinuationRolloutError(
            "Latent-overlap validation needs positive peak_vram_bytes."
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
