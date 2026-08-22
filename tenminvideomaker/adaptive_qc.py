"""Validated, monotonic adaptive-QC interventions.

The model proposes intent.  This module owns authority, mutations, seeds and
recipe identity so model output can never edit workflow JSON or durable state.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

from .qc_contracts import QcTier, canonical_json


class AdaptivePlanError(ValueError):
    """A repair plan is stale, unsupported, unsafe, duplicate, or a no-op."""


@dataclass(frozen=True)
class AdaptiveRepairPlan:
    authority_tier: str
    strategy: str
    hypothesis: str
    preserve_start_frame: bool
    changes: tuple[Mapping[str, Any], ...]
    failure_addressed: tuple[str, ...]
    difference_from_prior_attempts: str
    expected_effect: str


# These map only to controls already consumed by workflow_builder.py.
TIER_A_PRESETS: Mapping[str, Mapping[str, Any]] = {
    "balanced_default": {},
    "reduced_motion_pressure": {
        "i2v.first_pass.reference_strength": 0.85,
        "i2v.first_pass.image_strength": 0.85,
    },
    "stronger_start_conditioning": {
        "i2v.first_pass.reference_strength": 0.95,
        "i2v.first_pass.image_strength": 0.95,
    },
    "lower_distillation_influence": {
        # TenStrip's published higher-quality hybrid-DMD schedule.
        "i2v.first_pass.sigmas": [
            1.0, 0.968, 0.926, 0.875, 0.812, 0.741,
            0.661, 0.574, 0.482, 0.241, 0.121, 0.0,
        ],
    },
    "detail_stability": {
        "i2v.first_pass.reference_strength": 0.9,
        "i2v.first_pass.image_compression": 25,
    },
    "lower_guidance": {"i2v.first_pass.cfg": 0.8},
    "higher_guidance_within_safe_range": {"i2v.first_pass.cfg": 1.2},
}

_PLAN_KEYS = {
    "schema_version", "authority_tier", "strategy", "hypothesis",
    "preserve_start_frame", "changes", "failure_addressed",
    "difference_from_prior_attempts", "expected_effect",
}
_CHANGE_KEYS = {"path", "operation", "value"}
_ALLOWED_PATHS = {
    "A": {"i2v.first_pass.preset"},
    "B": {"i2v.prompt", "i2v.negative", "t2i.prompt"},
    "C": {"i2v.prompt", "scene_context.camera", "scene_context.staging"},
    "D": {"i2v.prompt"},
}
_AUTHORITY_ORDER = {"A": 0, "B": 1, "C": 2, "D": 3}
_START_FRAME_FAMILIES = {"anatomy", "topology", "face", "eyes", "hands", "limbs", "identity"}


def _strict_object(raw: str | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(raw, Mapping):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        raise AdaptivePlanError("Planner output must be one JSON object.")
    decoder = json.JSONDecoder()
    try:
        value, end = decoder.raw_decode(raw.lstrip())
    except json.JSONDecodeError as error:
        raise AdaptivePlanError("Planner output is malformed JSON.") from error
    if raw.lstrip()[end:].strip() or not isinstance(value, Mapping):
        raise AdaptivePlanError("Planner output must contain only one JSON object.")
    return value


def validate_repair_plan(
    raw: str | Mapping[str, Any],
    *,
    minimum_authority: str,
    failed_strategies: Sequence[str] = (),
) -> AdaptiveRepairPlan:
    value = _strict_object(raw)
    if set(value) != _PLAN_KEYS:
        raise AdaptivePlanError("Repair plan keys do not match adaptive_repair_plan_v1.")
    if value["schema_version"] != "adaptive_repair_plan_v1":
        raise AdaptivePlanError("Unsupported repair-plan schema version.")
    authority = value["authority_tier"]
    if authority not in _AUTHORITY_ORDER or minimum_authority not in _AUTHORITY_ORDER:
        raise AdaptivePlanError("authority_tier must be A, B, C, or D.")
    if _AUTHORITY_ORDER[authority] < _AUTHORITY_ORDER[minimum_authority]:
        raise AdaptivePlanError("Repair authority may not move backward.")
    strategy = value["strategy"]
    for field in ("strategy", "hypothesis", "difference_from_prior_attempts", "expected_effect"):
        if not isinstance(value[field], str) or not value[field].strip():
            raise AdaptivePlanError(f"{field} must contain text.")
    if strategy in failed_strategies:
        raise AdaptivePlanError("Planner repeated a failed strategy.")
    if not isinstance(value["preserve_start_frame"], bool):
        raise AdaptivePlanError("preserve_start_frame must be boolean.")
    failures = value["failure_addressed"]
    if not isinstance(failures, list) or not failures or any(
        not isinstance(item, str) or not item.strip() for item in failures
    ):
        raise AdaptivePlanError("failure_addressed must be a non-empty text array.")
    changes = value["changes"]
    if not isinstance(changes, list) or not changes:
        raise AdaptivePlanError("A repair plan may not be a no-op.")
    normalized: list[Mapping[str, Any]] = []
    for change in changes:
        if not isinstance(change, Mapping) or set(change) != _CHANGE_KEYS:
            raise AdaptivePlanError("Every repair change requires path, operation, and value.")
        path = change["path"]
        operation = change["operation"]
        if path not in _ALLOWED_PATHS[authority]:
            raise AdaptivePlanError(f"Unsupported Tier-{authority} path: {path}.")
        if operation not in {"replace", "append"}:
            raise AdaptivePlanError("Repair operation must be replace or append.")
        if path == "i2v.first_pass.preset":
            if operation != "replace" or change["value"] not in TIER_A_PRESETS:
                raise AdaptivePlanError("Tier-A preset is not approved.")
        elif not isinstance(change["value"], str) or not change["value"].strip():
            raise AdaptivePlanError(f"{path} requires non-empty text.")
        normalized.append(dict(change))
    if not value["preserve_start_frame"] and authority == "A":
        raise AdaptivePlanError("Tier A must preserve the starting frame.")
    return AdaptiveRepairPlan(
        authority, strategy.strip(), value["hypothesis"].strip(),
        value["preserve_start_frame"], tuple(normalized),
        tuple(item.strip().casefold() for item in failures),
        value["difference_from_prior_attempts"].strip(),
        value["expected_effect"].strip(),
    )


def _set_path(document: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    target = document
    for part in parts[:-1]:
        child = target.get(part)
        if not isinstance(child, dict):
            raise AdaptivePlanError(f"Repair path does not exist: {path}.")
        target = child
    if parts[-1] not in target:
        raise AdaptivePlanError(f"Repair path does not exist: {path}.")
    target[parts[-1]] = value


def apply_repair_plan(
    source_document: Mapping[str, Any],
    plan: AdaptiveRepairPlan,
    *,
    controller_seed: int,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Apply validated intent and a controller-owned unique seed."""
    if isinstance(controller_seed, bool) or not isinstance(controller_seed, int) or not 0 <= controller_seed < 2**64:
        raise AdaptivePlanError("controller_seed must be an unsigned 64-bit integer.")
    document = deepcopy(dict(source_document))
    before = canonical_json(document)
    changed: dict[str, Mapping[str, Any]] = {}
    i2v = document.get("i2v")
    if not isinstance(i2v, dict) or "seed" not in i2v:
        raise AdaptivePlanError("Source document lacks i2v.seed.")
    old_seed = i2v["seed"]
    i2v["seed"] = str(controller_seed) if isinstance(old_seed, str) else controller_seed
    changed["i2v.seed"] = {"before": old_seed, "after": i2v["seed"]}
    for change in plan.changes:
        path, operation, value = change["path"], change["operation"], change["value"]
        if path == "i2v.first_pass.preset":
            for actual_path, actual_value in TIER_A_PRESETS[value].items():
                cursor: Any = document
                for token in actual_path.split("."):
                    cursor = cursor[token]
                old = deepcopy(cursor)
                _set_path(document, actual_path, deepcopy(actual_value))
                changed[actual_path] = {"before": old, "after": actual_value}
            continue
        cursor: Any = document
        for token in path.split("."):
            cursor = cursor[token]
        old = deepcopy(cursor)
        new = value.strip() if operation == "replace" else f"{old.rstrip()} {value.strip()}"
        _set_path(document, path, new)
        changed[path] = {"before": old, "after": new}
    if canonical_json(document) == before:
        raise AdaptivePlanError("Repair normalized to a no-op.")
    return document, changed


def normalized_defect_family(evidence: Sequence[Mapping[str, Any]]) -> str:
    categories: list[str] = []
    for item in evidence:
        category = str(item.get("category", "other")).strip().casefold().replace("-", "_")
        categories.append(category or "other")
    return sorted(categories)[0] if categories else "other"


def start_frame_defect(evidence: Sequence[Mapping[str, Any]]) -> bool:
    for item in evidence:
        try:
            start = float(item.get("start_time_seconds", 99.0))
        except (TypeError, ValueError):
            continue
        family = str(item.get("category", "")).strip().casefold().replace("-", "_")
        if start <= 0.5 and family in _START_FRAME_FAMILIES:
            return True
    return False


def next_active_strategy(
    *,
    attempted: Sequence[str],
    evidence: Sequence[Mapping[str, Any]],
) -> tuple[QcTier | None, str, str]:
    """Select an untried monotonic strategy; None means defer this scene."""
    tried = set(attempted)
    if start_frame_defect(evidence) and "regenerate_start_frame" not in tried:
        return QcTier.B2, "regenerate_start_frame", "B"
    sequence = (
        (QcTier.A1, "new_seed", "A"),
        (QcTier.A2, "reduced_motion_pressure", "A"),
        (QcTier.B1, "constrained_prompt_repair", "B"),
        (QcTier.B2, "regenerate_start_frame", "B"),
        (QcTier.C, "current_scene_shot_redesign", "C"),
        (QcTier.D, "current_scene_semantic_replan", "D"),
    )
    for item in sequence:
        if item[1] not in tried:
            return item
    return None, "deferred_untried_strategy_required", "D"


def deterministic_adaptive_seed(
    *, job_id: str, scene_id: int, parent_candidate_id: str, strategy: str, attempt_number: int
) -> int:
    token = f"{job_id}|{scene_id}|{parent_candidate_id}|{strategy}|{attempt_number}"
    return int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:8], "big")
