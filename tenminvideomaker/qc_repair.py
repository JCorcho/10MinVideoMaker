"""Bounded deterministic A1 retry construction.

This tranche intentionally contains no B1 prompt-planning code.  A1 reuses the
normal VIDEO_ONLY revision path and changes exactly one semantic leaf: i2v.seed.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

from .contracts import JobPayload
from .qc_contracts import QcCandidateState, QcTier, canonical_json, derive_retry_seed
from .review import ReviewValidationError, validate_scene_edit
from .state_store import (
    PipelineStateStore,
    QcCandidateRecord,
    StateTransitionError,
)
from .storage import StorageLayout


@dataclass(frozen=True)
class A1Retry:
    candidate: QcCandidateRecord
    document: Mapping[str, Any]
    seed: int


def _uint64(value: object, field: str) -> int:
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value.strip())
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 2**64 - 1:
        raise StateTransitionError(f"{field} must be an unsigned 64-bit integer.")
    return value


def build_a1_document(
    source_document: Mapping[str, Any], *, seed: int
) -> Mapping[str, Any]:
    """Deep-copy one canonical revision document and alter only ``i2v.seed``."""
    seed = _uint64(seed, "seed")
    if not isinstance(source_document, Mapping):
        raise StateTransitionError("A1 source document must be an object.")
    document = deepcopy(dict(source_document))
    i2v = document.get("i2v")
    if not isinstance(i2v, dict):
        raise StateTransitionError("A1 source document must contain an i2v object.")
    if "seed" not in i2v or not isinstance(i2v.get("prompt"), str):
        raise StateTransitionError("A1 source document lacks the locked I2V fields.")
    i2v["seed"] = str(seed) if isinstance(i2v["seed"], str) else seed
    return document


def _without_i2v_seed(document: Mapping[str, Any]) -> dict[str, Any]:
    result = json.loads(canonical_json(document))
    if not isinstance(result.get("i2v"), dict):
        raise StateTransitionError("A1 document lacks the I2V contract.")
    result["i2v"].pop("seed", None)
    return result


def _stable_candidate_id(source: QcCandidateRecord, seed: int) -> str:
    identity = canonical_json(
        {
            "job_id": source.job_id,
            "scene_id": source.scene_id,
            "source_revision": source.revision,
            "source_candidate_id": source.candidate_id,
            "tier": QcTier.A1.value,
            "seed": str(seed),
        }
    )
    return f"candidate-a1-{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:32]}"


def schedule_a1_retry(
    store: PipelineStateStore,
    layout: StorageLayout,
    *,
    original_job: JobPayload,
    source_candidate_id: str,
    source_document: Mapping[str, Any],
) -> A1Retry:
    """Idempotently persist the one A1 candidate and normal VIDEO_ONLY revision."""
    source = store.qc_candidate(source_candidate_id)
    if source.tier != QcTier.ORIGINAL:
        raise StateTransitionError("A1 can descend only from the ORIGINAL candidate.")
    if source.job_id != original_job.job_id:
        raise StateTransitionError("A1 source job identity does not match the payload.")
    document_job_id = source_document.get("job_id")
    document_scene_id = source_document.get("scene_id")
    if document_job_id != source.job_id or document_scene_id != source.scene_id:
        raise StateTransitionError("A1 source document identity is stale.")
    i2v = source_document.get("i2v")
    if not isinstance(i2v, Mapping):
        raise StateTransitionError("A1 source document lacks the I2V contract.")
    prompt = i2v.get("prompt")
    negative = i2v.get("negative")
    source_seed = _uint64(i2v.get("seed"), "i2v.seed")
    if (
        prompt != source.current_prompt
        or negative != source.negative_prompt
        or source_seed != source.current_seed
    ):
        raise StateTransitionError(
            "A1 source prompt/negative/seed does not match immutable candidate evidence."
        )

    existing = tuple(
        item
        for item in store.qc_candidates(source.job_id, source.scene_id)
        if item.tier == QcTier.A1
    )
    if existing:
        candidate = existing[0]
        if candidate.parent_candidate_id != source.candidate_id:
            raise StateTransitionError("The existing A1 belongs to another lineage.")
        revision = next(
            (
                item
                for item in store.scene_revisions(source.job_id, source.scene_id)
                if item.revision == candidate.revision
            ),
            None,
        )
        if revision is None:
            raise StateTransitionError("The existing A1 candidate lost its revision.")
        expected = build_a1_document(source_document, seed=candidate.current_seed)
        if canonical_json(revision.parameters) != canonical_json(expected):
            raise StateTransitionError("The existing A1 document is inconsistent.")
        return A1Retry(candidate, revision.parameters, candidate.current_seed)

    used_seeds = tuple(
        value
        for candidate in store.qc_candidates(source.job_id, source.scene_id)
        for value in (candidate.original_seed, candidate.current_seed)
    )
    seed = derive_retry_seed(
        job_id=source.job_id,
        scene_id=source.scene_id,
        source_revision=source.revision,
        original_seed=source.original_seed,
        tier=QcTier.A1,
        used_seeds=used_seeds,
    )
    candidate_document = build_a1_document(source_document, seed=seed)
    try:
        validated = validate_scene_edit(
            original_job, source.scene_id, candidate_document
        )
    except ReviewValidationError as error:
        raise StateTransitionError(f"A1 document failed scene validation: {error}") from error
    if _without_i2v_seed(validated.document) != _without_i2v_seed(source_document):
        raise StateTransitionError("A1 validation changed a locked generation field.")
    if validated.document["i2v"]["prompt"] != source_document["i2v"]["prompt"]:
        raise StateTransitionError("A1 validation changed the I2V prompt.")

    revisions = store.scene_revisions(source.job_id, source.scene_id)
    source_revision = next(
        (item for item in revisions if item.revision == source.revision), None
    )
    if source_revision is None or not source_revision.frame_path:
        raise StateTransitionError("A1 requires the original accepted T2I frame.")
    expected_revision = max(item.revision for item in revisions) + 1
    candidate_id = _stable_candidate_id(source, seed)
    candidate = store.ensure_a1_candidate_revision(
        candidate_id=candidate_id,
        parent_candidate_id=source.candidate_id,
        job_id=source.job_id,
        scene_id=source.scene_id,
        expected_revision=expected_revision,
        parameters=validated.document,
        frame_path=source_revision.frame_path,
        source_video_path=str(
            layout.scene_clip_path(source.job_id, source.scene_id, expected_revision)
        ),
        original_prompt=source.original_prompt,
        current_prompt=source.current_prompt,
        original_seed=source.original_seed,
        current_seed=seed,
        negative_prompt=source.negative_prompt,
        negative_prompt_sha256=source.negative_prompt_sha256,
    )
    if candidate.state != QcCandidateState.PENDING_GENERATION:
        raise StateTransitionError("New A1 candidate did not enter PENDING_GENERATION.")
    return A1Retry(candidate, validated.document, seed)
