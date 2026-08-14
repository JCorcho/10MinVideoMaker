"""Bounded A1 seed retry and constrained B1 prompt repair construction."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Mapping, Sequence

from .contracts import JobPayload
from .qc_contracts import QcCandidateState, QcTier, canonical_json, derive_retry_seed
from .review import ReviewValidationError, validate_scene_edit
from .state_store import (
    PipelineStateStore,
    QcCandidateRecord,
    QcRepairRecord,
    StateTransitionError,
)
from .storage import StorageLayout, write_immutable_json


class B1PatchValidationError(ValueError):
    """Raised when a planner response exceeds the Phase-1 mutation contract."""


class RepairGenerationError(RuntimeError):
    """Classify a repair-render failure before its durable budget is charged."""

    def __init__(self, candidate_id: str, reason: str, *, retryable: bool) -> None:
        super().__init__(reason)
        self.candidate_id = candidate_id
        self.reason = reason
        self.retryable = retryable


@dataclass(frozen=True)
class ValidatedRepairPatch:
    prompt: str
    seed: int
    summary: str
    source_candidate_id: str
    source_candidate_sha256: str
    source_revision: int
    source_document_sha256: str
    evaluation_id: str
    repair_input_sha256: str
    raw_patch: Mapping[str, Any]


@dataclass(frozen=True)
class A1Retry:
    candidate: QcCandidateRecord
    document: Mapping[str, Any]
    seed: int


@dataclass(frozen=True)
class B1Retry:
    repair: QcRepairRecord
    candidate: QcCandidateRecord | None
    document: Mapping[str, Any] | None
    seed: int
    rejection_reason: str | None = None


_SHA256 = re.compile(r"[0-9a-f]{64}")


def _strict_json_object(raw_text: str) -> dict[str, Any]:
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise B1PatchValidationError("Planner output must be one JSON object.")
    decoder = json.JSONDecoder()
    try:
        value, end = decoder.raw_decode(raw_text.lstrip())
    except json.JSONDecodeError as error:
        raise B1PatchValidationError("Planner output is malformed JSON.") from error
    if raw_text.lstrip()[end:].strip() or not isinstance(value, dict):
        raise B1PatchValidationError(
            "Planner output must contain only one JSON object without prose."
        )
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    actual = set(value)
    if actual != expected:
        unknown = sorted(actual - expected)
        missing = sorted(expected - actual)
        detail = []
        if unknown:
            detail.append("unknown=" + ",".join(unknown))
        if missing:
            detail.append("missing=" + ",".join(missing))
        raise B1PatchValidationError(
            f"{field} has invalid keys ({'; '.join(detail)})."
        )


def _parsed_patch_evidence(raw_text: str) -> Mapping[str, Any]:
    try:
        value = _strict_json_object(raw_text)
    except B1PatchValidationError:
        return {}
    patch = value.get("patch")
    if not isinstance(patch, Mapping):
        return {}
    return json.loads(canonical_json(patch))


def parse_and_validate_b1_patch(
    raw_text: str,
    *,
    source_document: Mapping[str, Any],
    required_seed: int,
    repair_input_hash: str,
    current_candidate_hash: str,
    current_candidate_id: str,
    evaluation_id: str,
    source_revision: int,
    source_document_sha256: str,
    prior_attempts: Sequence[tuple[str, int, str]] = (),
    generation_config_hash: str = "",
    required_prompt_fragments: Sequence[str] = (),
) -> ValidatedRepairPatch:
    """Validate strict model JSON; seed/config remain controller-owned."""
    required_seed = _uint64(required_seed, "required_seed")
    for name, value in (
        ("repair_input_hash", repair_input_hash),
        ("current_candidate_hash", current_candidate_hash),
        ("source_document_sha256", source_document_sha256),
    ):
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise B1PatchValidationError(f"{name} must be lowercase SHA-256.")
    value = _strict_json_object(raw_text)
    _exact_keys(value, {"schema_version", "source", "patch", "summary"}, "response")
    if value["schema_version"] != 1:
        raise B1PatchValidationError("Unsupported B1 patch schema version.")
    source = value["source"]
    patch = value["patch"]
    if not isinstance(source, Mapping) or not isinstance(patch, Mapping):
        raise B1PatchValidationError("source and patch must be objects.")
    _exact_keys(
        source,
        {
            "candidate_id",
            "candidate_sha256",
            "evaluation_id",
            "repair_input_sha256",
            "source_revision",
            "source_document_sha256",
        },
        "source",
    )
    expected_source = {
        "candidate_id": current_candidate_id,
        "candidate_sha256": current_candidate_hash,
        "evaluation_id": evaluation_id,
        "repair_input_sha256": repair_input_hash,
        "source_revision": source_revision,
        "source_document_sha256": source_document_sha256,
    }
    if dict(source) != expected_source:
        raise B1PatchValidationError(
            "Repair proposal source identity/hash does not match current durable state."
        )
    _exact_keys(patch, {"i2v"}, "patch")
    i2v_patch = patch["i2v"]
    if not isinstance(i2v_patch, Mapping):
        raise B1PatchValidationError("patch.i2v must be an object.")
    _exact_keys(i2v_patch, {"prompt"}, "patch.i2v")
    prompt = i2v_patch["prompt"]
    summary = value["summary"]
    if not isinstance(prompt, str) or not prompt.strip():
        raise B1PatchValidationError("B1 i2v.prompt must contain text.")
    if not isinstance(summary, str) or not summary.strip():
        raise B1PatchValidationError("B1 summary must contain text.")
    prompt = prompt.strip()
    current_i2v = source_document.get("i2v")
    if not isinstance(current_i2v, Mapping) or not isinstance(
        current_i2v.get("prompt"), str
    ):
        raise B1PatchValidationError("Current source document lacks i2v.prompt.")
    segments = current_i2v.get("segments", [])
    if isinstance(segments, Sequence) and not isinstance(segments, (str, bytes)) and segments:
        raise B1PatchValidationError(
            "B1 is inapplicable because explicit segment prompts override i2v.prompt."
        )
    if prompt == current_i2v["prompt"].strip():
        raise B1PatchValidationError("B1 prompt must differ from the current prompt.")
    missing_fragments = [
        fragment
        for fragment in required_prompt_fragments
        if not isinstance(fragment, str)
        or not fragment.strip()
        or fragment.strip().casefold() not in prompt.casefold()
    ]
    if missing_fragments:
        raise B1PatchValidationError(
            "B1 prompt lost required fixed/safety prompt content."
        )
    attempt_identity = (prompt, required_seed, generation_config_hash)
    if attempt_identity in tuple(prior_attempts):
        raise B1PatchValidationError(
            "B1 prompt+seed+generation configuration duplicates a prior attempt."
        )
    return ValidatedRepairPatch(
        prompt=prompt,
        seed=required_seed,
        summary=summary.strip(),
        source_candidate_id=current_candidate_id,
        source_candidate_sha256=current_candidate_hash,
        source_revision=source_revision,
        source_document_sha256=source_document_sha256,
        evaluation_id=evaluation_id,
        repair_input_sha256=repair_input_hash,
        raw_patch={"i2v": {"prompt": prompt}},
    )


def _without_b1_mutable_fields(document: Mapping[str, Any]) -> dict[str, Any]:
    result = json.loads(canonical_json(document))
    i2v = result.get("i2v")
    if not isinstance(i2v, dict):
        raise B1PatchValidationError("B1 document lacks the I2V contract.")
    i2v.pop("prompt", None)
    i2v.pop("seed", None)
    return result


def apply_b1_patch(
    original_job: JobPayload,
    scene_id: int,
    source_document: Mapping[str, Any],
    patch: ValidatedRepairPatch,
):
    """Apply only prompt+controller seed, then re-run the full scene validator."""
    document = deepcopy(dict(source_document))
    i2v = document.get("i2v")
    if not isinstance(i2v, dict) or "seed" not in i2v:
        raise B1PatchValidationError("B1 source document lacks locked I2V fields.")
    i2v["prompt"] = patch.prompt
    i2v["seed"] = str(patch.seed) if isinstance(i2v["seed"], str) else patch.seed
    try:
        validated = validate_scene_edit(original_job, scene_id, document)
    except ReviewValidationError as error:
        raise B1PatchValidationError(
            f"B1 document failed scene validation: {error}"
        ) from error
    if _without_b1_mutable_fields(validated.document) != _without_b1_mutable_fields(
        source_document
    ):
        raise B1PatchValidationError("B1 validation changed a locked generation field.")
    if validated.document["i2v"]["prompt"] != patch.prompt:
        raise B1PatchValidationError("B1 validation changed the proposed prompt.")
    if int(validated.document["i2v"]["seed"]) != patch.seed:
        raise B1PatchValidationError("B1 validation changed the controller seed.")
    return validated


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


def _stable_b1_candidate_id(source: QcCandidateRecord, seed: int) -> str:
    identity = canonical_json(
        {
            "job_id": source.job_id,
            "scene_id": source.scene_id,
            "source_revision": source.revision,
            "source_candidate_id": source.candidate_id,
            "tier": QcTier.B1.value,
            "seed": str(seed),
        }
    )
    return f"candidate-b1-{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:32]}"


def _stable_repair_id(source: QcCandidateRecord, repair_input_hash: str) -> str:
    digest = hashlib.sha256(
        f"{source.candidate_id}|{repair_input_hash}".encode("utf-8")
    ).hexdigest()
    return f"repair-b1-{digest[:32]}"


def schedule_b1_retry(
    store: PipelineStateStore,
    layout: StorageLayout,
    *,
    original_job: JobPayload,
    source_candidate_id: str,
    evaluation_id: str,
    source_document: Mapping[str, Any],
    raw_output: str,
    planner_identity: Mapping[str, Any],
    repair_input_hash: str,
    prior_repair_summaries: Sequence[Any] = (),
    required_prompt_fragments: Sequence[str] = (),
    planner_failure_reason: str | None = None,
) -> B1Retry:
    """Persist one B1 plan before idempotently allocating its render revision."""
    source = store.qc_candidate(source_candidate_id)
    if source.tier != QcTier.A1:
        raise StateTransitionError("B1 can descend only from the A1 candidate.")
    if source.job_id != original_job.job_id or source.source_video_sha256 is None:
        raise StateTransitionError("B1 source candidate identity/video is incomplete.")
    evaluations = {
        item.evaluation_id: item for item in store.qc_evaluations(source.candidate_id)
    }
    evaluation = evaluations.get(evaluation_id)
    if (
        evaluation is None
        or evaluation.state != "COMPLETE"
        or evaluation.normalized_decision is None
        or evaluation.normalized_decision.value != "FAIL"
    ):
        raise StateTransitionError("B1 requires the completed FAIL evaluation for A1.")
    if source_document.get("job_id") != source.job_id or source_document.get(
        "scene_id"
    ) != source.scene_id:
        raise StateTransitionError("B1 source document identity is stale.")
    current_i2v = source_document.get("i2v")
    if not isinstance(current_i2v, Mapping):
        raise StateTransitionError("B1 source document lacks the I2V contract.")
    if (
        current_i2v.get("prompt") != source.current_prompt
        or current_i2v.get("negative") != source.negative_prompt
        or _uint64(current_i2v.get("seed"), "i2v.seed") != source.current_seed
    ):
        raise StateTransitionError("B1 source document does not match candidate evidence.")

    existing_repairs = store.qc_repairs(source.candidate_id)
    if existing_repairs:
        repair = existing_repairs[0]
        if repair.repair_input_sha256 != repair_input_hash:
            raise StateTransitionError("The one B1 plan has different immutable input.")
        if repair.status != "ACCEPTED":
            store.set_qc_candidate_state(
                source.candidate_id,
                QcCandidateState.HOLD_FOR_REVIEW,
                next_action="hold_for_review",
            )
            return B1Retry(repair, None, None, 0, repair.reason)
        saved = repair.proposed_patch
        seed = _uint64(saved.get("derived_seed"), "derived_seed")
        prompt = saved.get("i2v", {}).get("prompt") if isinstance(saved.get("i2v"), Mapping) else None
        if not isinstance(prompt, str):
            raise StateTransitionError("Persisted B1 proposal lost its prompt.")
        validated_patch = ValidatedRepairPatch(
            prompt=prompt,
            seed=seed,
            summary=str(saved.get("summary", "")),
            source_candidate_id=source.candidate_id,
            source_candidate_sha256=source.source_video_sha256,
            source_revision=source.revision,
            source_document_sha256=hashlib.sha256(
                canonical_json(source_document).encode("utf-8")
            ).hexdigest(),
            evaluation_id=evaluation_id,
            repair_input_sha256=repair_input_hash,
            raw_patch={"i2v": {"prompt": prompt}},
        )
        validated = apply_b1_patch(
            original_job, source.scene_id, source_document, validated_patch
        )
        result_candidate_id = str(saved.get("result_candidate_id"))
        result_revision = int(saved.get("result_revision"))
    else:
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
            tier=QcTier.B1,
            used_seeds=used_seeds,
        )
        generation_config_hash = hashlib.sha256(
            canonical_json(_without_b1_mutable_fields(source_document)).encode("utf-8")
        ).hexdigest()
        prior_attempts = tuple(
            (candidate.current_prompt, candidate.current_seed, generation_config_hash)
            for candidate in store.qc_candidates(source.job_id, source.scene_id)
        )
        repair_id = _stable_repair_id(source, repair_input_hash)
        manifest_path = layout.qc_repair_manifest_path(
            source.job_id, source.scene_id, source.revision, repair_id
        )
        parsed_model_patch = _parsed_patch_evidence(raw_output)
        try:
            if planner_failure_reason is not None:
                raise B1PatchValidationError(planner_failure_reason)
            validated_patch = parse_and_validate_b1_patch(
                raw_output,
                source_document=source_document,
                required_seed=seed,
                repair_input_hash=repair_input_hash,
                current_candidate_hash=source.source_video_sha256,
                current_candidate_id=source.candidate_id,
                evaluation_id=evaluation_id,
                source_revision=source.revision,
                source_document_sha256=hashlib.sha256(
                    canonical_json(source_document).encode("utf-8")
                ).hexdigest(),
                prior_attempts=prior_attempts,
                generation_config_hash=generation_config_hash,
                required_prompt_fragments=required_prompt_fragments,
            )
            validated = apply_b1_patch(
                original_job, source.scene_id, source_document, validated_patch
            )
            result_revision = max(
                item.revision
                for item in store.scene_revisions(source.job_id, source.scene_id)
            ) + 1
            result_candidate_id = _stable_b1_candidate_id(source, seed)
            proposed_patch = {
                "i2v": {"prompt": validated_patch.prompt},
                "summary": validated_patch.summary,
                "derived_seed": str(seed),
                "result_candidate_id": result_candidate_id,
                "result_revision": result_revision,
            }
            status = "ACCEPTED"
            reason = None
        except B1PatchValidationError as error:
            validated = None
            result_revision = 0
            result_candidate_id = ""
            proposed_patch = parsed_model_patch
            status = "REJECTED"
            reason = str(error)
        manifest = {
            "schema_version": 1,
            "repair_id": repair_id,
            "planner_identity": dict(planner_identity),
            "repair_input_sha256": repair_input_hash,
            "source_candidate_id": source.candidate_id,
            "source_candidate_sha256": source.source_video_sha256,
            "evaluation_id": evaluation_id,
            "raw_response": raw_output,
            "parsed_patch": parsed_model_patch,
            "validated_patch": proposed_patch if status == "ACCEPTED" else None,
            "validation_status": status,
            "rejection_reason": reason,
            "derived_seed": str(seed),
            "result_candidate_id": result_candidate_id or None,
            "result_revision": result_revision or None,
            "prior_repair_summaries": list(prior_repair_summaries),
        }
        write_immutable_json(manifest_path, manifest)
        manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        repair = store.record_qc_repair(
            repair_id=repair_id,
            candidate_id=source.candidate_id,
            evaluation_id=evaluation_id,
            planner_identity=planner_identity,
            repair_input_sha256=repair_input_hash,
            raw_output=raw_output,
            proposed_patch=proposed_patch,
            status=status,
            reason=reason,
            prior_repair_summaries=prior_repair_summaries,
            evidence_manifest_path=str(manifest_path),
            evidence_manifest_sha256=manifest_sha,
        )
        if status != "ACCEPTED" or validated is None:
            store.set_qc_candidate_state(
                source.candidate_id,
                QcCandidateState.HOLD_FOR_REVIEW,
                next_action="hold_for_review",
            )
            return B1Retry(repair, None, None, seed, reason)

    source_revision = next(
        (
            item
            for item in store.scene_revisions(source.job_id, source.scene_id)
            if item.revision == source.revision
        ),
        None,
    )
    if source_revision is None or not source_revision.frame_path:
        raise StateTransitionError("B1 requires the lineage's accepted T2I frame.")
    candidate = store.ensure_b1_candidate_revision(
        candidate_id=result_candidate_id,
        parent_candidate_id=source.candidate_id,
        job_id=source.job_id,
        scene_id=source.scene_id,
        expected_revision=result_revision,
        parameters=validated.document,
        frame_path=source_revision.frame_path,
        source_video_path=str(
            layout.scene_clip_path(source.job_id, source.scene_id, result_revision)
        ),
        original_prompt=source.original_prompt,
        current_prompt=validated_patch.prompt,
        original_seed=source.original_seed,
        current_seed=seed,
        negative_prompt=source.negative_prompt,
        negative_prompt_sha256=source.negative_prompt_sha256,
    )
    return B1Retry(repair, candidate, validated.document, seed)


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

    candidate_seeds = tuple(
        value
        for candidate in store.qc_candidates(source.job_id, source.scene_id)
        for value in (candidate.original_seed, candidate.current_seed)
    )
    revision_seeds: list[int] = []
    for revision in store.scene_revisions(source.job_id, source.scene_id):
        revision_i2v = revision.parameters.get("i2v")
        if isinstance(revision_i2v, Mapping) and "seed" in revision_i2v:
            revision_seeds.append(
                _uint64(revision_i2v["seed"], "revision i2v.seed")
            )
    used_seeds = (*candidate_seeds, *revision_seeds)
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
