"""Serialized, restart-safe Phase-1 generation/QC epoch controller."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import logging
from pathlib import Path
import shutil
from datetime import datetime, timezone
import socket
from tempfile import TemporaryDirectory
from typing import Any, Callable, Mapping, Sequence

from .adaptive_qc import (
    apply_repair_plan,
    deterministic_adaptive_seed,
    next_active_strategy,
    normalized_defect_family,
    validate_repair_plan,
)

from .contracts import JobPayload
from .qc_backend import (
    BackendIdentity,
    HeadlessVideoEvaluator,
    QcBackendError,
    RepairPlannerRequest,
    load_production_rubric,
    load_repair_planner_prompt,
)
from .qc_config import QualityControlSettings
from .qc_contracts import (
    QcArtifactStage,
    QcCandidateState,
    QcDecision,
    QcError,
    QcEvidencePolicy,
    QcHumanDecision,
    QcTier,
    canonical_json,
    evaluation_idempotency_key,
)
from .qc_repair import RepairGenerationError, schedule_a1_retry, schedule_b1_retry
from .qc_video import sample_video_frames
from .review import scene_review_document, validate_scene_edit
from .state_store import (
    canonical_draft_recipe_sha256,
    IncompleteLegacySelectionError,
    ManualFinalSceneSelection,
    PipelineState,
    PipelineStateStore,
    QcCandidateRecord,
    QcEvaluationRecord,
    SceneState,
    StateTransitionError,
)
from .storage import (
    StorageLayout,
    write_immutable_json,
    write_immutable_text,
)


LOGGER = logging.getLogger("10MinVideoMaker.qc_controller")


class QcControllerError(RuntimeError):
    """Raised when an epoch cannot preserve the Phase-1 trust boundary."""


@dataclass(frozen=True)
class QcEpochResult:
    ready_for_finalization: bool
    selection: tuple[ManualFinalSceneSelection, ...]
    waiting_for_human: bool
    generated_count: int = 0
    evaluated_count: int = 0


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity_mapping(identity: BackendIdentity) -> dict[str, Any]:
    return asdict(identity)


_CANONICAL_DEFECT_CATEGORIES = frozenset(
    {
        "anatomy",
        "topology",
        "face",
        "eyes",
        "mouth",
        "hands",
        "limbs",
        "skin",
        "hair",
        "clothing",
        "identity",
        "ownership",
        "contact",
        "temporal_consistency",
        "object_continuity",
        "morphing",
        "motion_continuity",
        "physics",
        "lighting_color",
        "camera_continuity",
        "blur_artifact",
        "other",
    }
)


def _canonical_defect_category(value: str) -> str:
    token = value.strip().lower().replace("-", "_").replace(" ", "_")
    return token if token in _CANONICAL_DEFECT_CATEGORIES else "other"


def _normalized_planner_windows(
    windows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Strip raw judge prose and retain only schema-validated defect evidence."""
    normalized: list[dict[str, Any]] = []
    try:
        for window in windows:
            response = window.get("response")
            if not isinstance(response, Mapping):
                raise ValueError("QC window response is missing.")
            raw_errors = response.get("errors")
            if not isinstance(raw_errors, list):
                raise ValueError("QC window errors are not an array.")
            parsed_defects = tuple(
                QcError.from_mapping(item)
                if isinstance(item, Mapping)
                else (_ for _ in ()).throw(ValueError("QC error is not an object."))
                for item in raw_errors
            )
            defects = tuple(
                {
                    "category": _canonical_defect_category(item.category),
                    "severity": item.severity,
                    "confidence": item.confidence,
                    "start_time_seconds": item.start_time_seconds,
                    "end_time_seconds": item.end_time_seconds,
                }
                for item in parsed_defects
            )
            raw_decision = response.get("decision")
            decision = (
                QcDecision.UNCERTAIN
                if raw_decision is None
                else QcDecision(raw_decision)
            )
            confidence = response.get("confidence")
            if confidence is not None and (
                isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not 0.0 <= float(confidence) <= 1.0
            ):
                raise ValueError("QC window confidence is invalid.")
            window_number = window.get("window_number")
            if isinstance(window_number, bool) or not isinstance(window_number, int):
                raise ValueError("QC window number is invalid.")
            source_indices = window.get("source_frame_indices")
            timestamps = window.get("timestamps_seconds")
            image_sha256s = window.get("image_sha256s")
            if (
                not isinstance(source_indices, list)
                or not isinstance(timestamps, list)
                or not isinstance(image_sha256s, list)
                or len(source_indices) != len(timestamps)
                or len(source_indices) != len(image_sha256s)
            ):
                raise ValueError("QC window source positions are invalid.")
            if any(
                not isinstance(item, str)
                or len(item) != 64
                or any(character not in "0123456789abcdef" for character in item)
                for item in image_sha256s
            ):
                raise ValueError("QC window image hashes are invalid.")
            normalized.append(
                {
                    "window_number": window_number,
                    "source_frame_indices": [int(item) for item in source_indices],
                    "timestamps_seconds": [float(item) for item in timestamps],
                    "image_sha256s": list(image_sha256s),
                    "decision": decision.value,
                    "confidence": (
                        None if confidence is None else float(confidence)
                    ),
                    "defects": list(defects),
                    "confirmation_of_window": window.get("confirmation_of_window"),
                }
            )
    except (TypeError, ValueError) as error:
        raise QcControllerError(
            "Durable QC evidence failed planner-input normalization."
        ) from error
    return tuple(normalized)


class Phase1QcController:
    """Own whole generation/QC epochs; durable rows determine every next action."""

    def __init__(
        self,
        *,
        store: PipelineStateStore,
        layout: StorageLayout,
        settings: QualityControlSettings,
        backend_factory: Callable[[], Any],
        prompt_root: Path,
        sample_video: Callable[..., Any] = sample_video_frames,
        ffmpeg_command: str = "ffmpeg",
        ffprobe_command: str = "ffprobe",
        qc_port_open: Callable[[], bool] | None = None,
    ):
        self.store = store
        self.layout = layout
        self.settings = settings
        self.backend_factory = backend_factory
        self.prompt_root = Path(prompt_root)
        self.sample_video = sample_video
        self.ffmpeg_command = ffmpeg_command
        self.ffprobe_command = ffprobe_command
        self.qc_port_open = qc_port_open or self._default_qc_port_open
        self._active_backend: Any | None = None

    def _default_qc_port_open(self) -> bool:
        try:
            with socket.create_connection(
                (self.settings.loopback_host, self.settings.loopback_port),
                timeout=0.25,
            ):
                return True
        except OSError:
            return False

    def close(self) -> None:
        self._close_active_backend()

    def _close_active_backend(self) -> None:
        backend = self._active_backend
        if backend is not None:
            backend.close()
            if self._active_backend is backend:
                self._active_backend = None

    @staticmethod
    def _revision_document(store: PipelineStateStore, candidate: QcCandidateRecord) -> Mapping[str, Any]:
        for revision in store.scene_revisions(candidate.job_id, candidate.scene_id):
            if revision.revision == candidate.revision:
                return revision.parameters
        raise StateTransitionError("QC candidate lost its bound scene revision.")

    @staticmethod
    def _deterministic_b2_seed(
        *,
        job_id: str,
        scene_id: int,
        parent_candidate_id: str,
        token: str,
    ) -> int:
        if token not in {"B2-T2I", "B2-I2V"}:
            raise StateTransitionError("Unknown B2 seed token.")
        return int.from_bytes(
            hashlib.sha256(f"{job_id}|{scene_id}|{parent_candidate_id}|{token}".encode("utf-8")).digest()[:8],
            "big",
        )

    @staticmethod
    def _candidate_id_for_b2(
        *,
        job_id: str,
        scene_id: int,
        parent_candidate_id: str,
        b2_t2i_seed: int,
        b2_i2v_seed: int,
    ) -> str:
        identity = canonical_json(
            {
                "job_id": job_id,
                "scene_id": scene_id,
                "parent_candidate_id": parent_candidate_id,
                "tier": QcTier.B2.value,
                "t2i_seed": str(b2_t2i_seed),
                "i2v_seed": str(b2_i2v_seed),
            }
        )
        return "candidate-b2-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]

    @staticmethod
    def _to_u64_seed(value: Any, field: str) -> int:
        try:
            value = int(value)
        except (TypeError, ValueError) as error:
            raise StateTransitionError(f"{field} must be an unsigned 64-bit integer.") from error
        if value < 0 or value > 2**64 - 1:
            raise StateTransitionError(f"{field} must be an unsigned 64-bit integer.")
        return value

    @staticmethod
    def _b2_start_frame_detected(evaluation: QcEvaluationRecord) -> bool:
        if evaluation.normalized_decision != QcDecision.FAIL:
            return False
        for window in evaluation.suspect_windows:
            if not isinstance(window, Mapping):
                continue
            response = window.get("response")
            if not isinstance(response, Mapping):
                continue
            raw_errors = response.get("errors")
            if not isinstance(raw_errors, list):
                continue
            for raw_error in raw_errors:
                if not isinstance(raw_error, Mapping):
                    continue
                category = str(raw_error.get("category", "")).strip().lower()
                if not category:
                    continue
                try:
                    start_time = float(raw_error.get("start_time_seconds", 1.0))
                    severity = int(raw_error.get("severity"))
                except (TypeError, ValueError):
                    continue
                if start_time > 0.5:
                    continue
                if any(
                    token in category
                    for token in (
                        "anatomy", "morph", "topology", "limb", "hand",
                        "finger", "face", "eye", "identity",
                    )
                ):
                    return True
                if severity >= 5 and "temporal" in category:
                    return True
        return False

    def _ensure_qc_b2_revision(
        self,
        *,
        job_id: str,
        scene_id: int,
        revision: int,
        parameters: Mapping[str, Any],
        frame_path: str,
    ) -> None:
        parameters_json = canonical_json(parameters)
        now = datetime.now(timezone.utc).isoformat()
        with self.store._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """
                SELECT parameters_json, frame_path FROM scene_revisions
                WHERE job_id = ? AND scene_id = ? AND revision = ?
                """,
                (job_id, scene_id, revision),
            ).fetchone()
            if existing is not None:
                try:
                    existing_parameters = json.loads(existing["parameters_json"])
                except (TypeError, json.JSONDecodeError) as error:
                    raise StateTransitionError("B2 revision document is invalid.") from error
                if (
                    existing["frame_path"] != frame_path
                    or canonical_json(existing_parameters) != parameters_json
                ):
                    raise StateTransitionError(
                        "An existing B2 revision already exists with different evidence."
                    )
                return
            next_revision = connection.execute(
                "SELECT COALESCE(MAX(revision), 0) + 1 AS revision FROM scene_revisions WHERE job_id = ? AND scene_id = ?",
                (job_id, scene_id),
            ).fetchone()
            if next_revision is None or int(next_revision["revision"]) != revision:
                raise StateTransitionError(
                    "The expected B2 revision is stale; regenerate from current durable state."
                )
            connection.execute(
                """
                INSERT INTO scene_revisions (
                    job_id, scene_id, revision, remake_mode, parameters_json, state,
                    frame_path, video_path, error, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?)
                """,
                (
                    job_id,
                    scene_id,
                    revision,
                    "video_only",
                    parameters_json,
                    SceneState.PENDING.value,
                    frame_path,
                    now,
                    now,
                ),
            )

    def schedule_b2_retry(
        self,
        *,
        original_job: JobPayload,
        source_candidate_id: str,
        source_document: Mapping[str, Any],
    ) -> QcCandidateRecord:
        source = self.store.qc_candidate(source_candidate_id)
        if source.tier != QcTier.B1:
            raise StateTransitionError("B2 can only descend from the failed B1 candidate.")
        if source.job_id != original_job.job_id:
            raise StateTransitionError("B2 source candidate identity is stale.")
        if source_document.get("job_id") != source.job_id or source_document.get("scene_id") != source.scene_id:
            raise StateTransitionError("B2 source document identity is stale.")
        self.store.validate_qc_candidate_source_identity(
            source_candidate_id,
            source_document,
        )
        source_t2i = source_document.get("t2i")
        source_i2v = source_document.get("i2v")
        if not isinstance(source_t2i, Mapping) or not isinstance(source_i2v, Mapping):
            raise StateTransitionError("B2 source document lacks the scene contract.")
        source_t2i_seed = self._to_u64_seed(
            source_t2i.get("seed"),
            "source T2I seed",
        )
        source_i2v_seed = self._to_u64_seed(
            source_i2v.get("seed"),
            "source I2V seed",
        )
        source_i2v_prompt = source_i2v.get("prompt")
        source_negative = source_i2v.get("negative")
        if (
            source_i2v_prompt != source.current_prompt
            or source_negative != source.negative_prompt
            or source_i2v_seed != source.current_seed
        ):
            raise StateTransitionError("B2 source evidence does not match its candidate.")
        b2_t2i_seed = self._deterministic_b2_seed(
            job_id=source.job_id,
            scene_id=source.scene_id,
            parent_candidate_id=source.candidate_id,
            token="B2-T2I",
        )
        b2_i2v_seed = self._deterministic_b2_seed(
            job_id=source.job_id,
            scene_id=source.scene_id,
            parent_candidate_id=source.candidate_id,
            token="B2-I2V",
        )
        if b2_t2i_seed == source_t2i_seed:
            raise StateTransitionError("B2 requires a new T2I seed.")
        if b2_i2v_seed == source_i2v_seed:
            raise StateTransitionError("B2 requires a new I2V seed.")
        revision = source.revision + 1
        b2_document = json.loads(canonical_json(source_document))
        b2_t2i = dict(b2_document.get("t2i") or {})
        b2_i2v = dict(b2_document.get("i2v") or {})
        b2_t2i["seed"] = b2_t2i_seed
        b2_i2v["prompt"] = source_i2v_prompt
        b2_i2v["negative"] = source_negative
        b2_i2v["seed"] = b2_i2v_seed
        b2_document["t2i"] = b2_t2i
        b2_document["i2v"] = b2_i2v
        validated = validate_scene_edit(original_job, source.scene_id, b2_document)
        b2_document = validated.document

        frame_path = str(self.layout.scene_frame_path(source.job_id, source.scene_id, revision))
        source_video_path = str(
            self.layout.scene_draft_path(source.job_id, source.scene_id, revision)
            if source.artifact_stage == QcArtifactStage.DRAFT
            else self.layout.scene_clip_path(source.job_id, source.scene_id, revision)
        )
        source_revision = next(
            item for item in self.store.scene_revisions(source.job_id, source.scene_id)
            if item.revision == source.revision
        )
        if not source_revision.frame_path or not Path(source_revision.frame_path).is_file():
            raise StateTransitionError("B2 source lost its immutable starting frame.")
        Path(frame_path).parent.mkdir(parents=True, exist_ok=True)
        if not Path(frame_path).is_file():
            shutil.copyfile(source_revision.frame_path, frame_path)
        self._ensure_qc_b2_revision(
            job_id=source.job_id,
            scene_id=source.scene_id,
            revision=revision,
            parameters=b2_document,
            frame_path=frame_path,
        )

        candidate_id = self._candidate_id_for_b2(
            job_id=source.job_id,
            scene_id=source.scene_id,
            parent_candidate_id=source.candidate_id,
            b2_t2i_seed=b2_t2i_seed,
            b2_i2v_seed=b2_i2v_seed,
        )
        existing = tuple(
            item
            for item in self.store.qc_candidates(source.job_id, source.scene_id)
            if item.tier == QcTier.B2
        )
        if existing:
            candidate = existing[0]
            if candidate.parent_candidate_id != source.candidate_id:
                raise StateTransitionError("Existing B2 candidate belongs to a different lineage.")
            if candidate.revision != revision:
                raise StateTransitionError("Existing B2 revision is stale.")
            if candidate.current_seed != b2_i2v_seed:
                raise StateTransitionError("Existing B2 candidate has a different I2V seed.")
            if canonical_json(
                self._revision_document(self.store, candidate)
            ) != canonical_json(b2_document):
                raise StateTransitionError("Existing B2 candidate has immutable drift.")
            return candidate

        return self.store.ensure_qc_candidate(
            candidate_id=candidate_id,
            job_id=source.job_id,
            scene_id=source.scene_id,
            revision=revision,
            tier=QcTier.B2,
            parent_candidate_id=source.candidate_id,
            source_video_path=source_video_path,
            source_video_sha256=None,
            original_prompt=source.original_prompt,
            current_prompt=source.current_prompt,
            original_seed=source.original_seed,
            current_seed=b2_i2v_seed,
            negative_prompt=source.negative_prompt,
            negative_prompt_sha256=source.negative_prompt_sha256,
            state=QcCandidateState.PENDING_GENERATION,
            next_action="render_b2",
            artifact_stage=(
                QcArtifactStage.DRAFT
                if source.artifact_stage == QcArtifactStage.DRAFT
                else QcArtifactStage.LEGACY_FINAL
            ),
            attempt_number=source.attempt_number + 1,
            authority_tier="B",
            strategy="regenerate_start_frame",
        )

    def register_original_candidates(self, job: JobPayload) -> tuple[QcCandidateRecord, ...]:
        """Atomically bind the exact pre-QC legacy selection to immutable bytes."""
        scene_ids = tuple(scene.scene_id for scene in job.scenes)
        existing = tuple(
            item for item in self.store.qc_candidates(job.job_id)
            if item.tier == QcTier.ORIGINAL
        )
        if existing:
            if tuple(item.scene_id for item in existing) != tuple(sorted(scene_ids)):
                missing = tuple(sorted(set(scene_ids) - {item.scene_id for item in existing}))
                if missing:
                    self.store.hold_incomplete_qc_job(job.job_id, missing)
                    return ()
                raise StateTransitionError("The durable pre-QC baseline snapshot is invalid.")
            for candidate in existing:
                document = self._revision_document(self.store, candidate)
                self.store.validate_qc_candidate_source_identity(
                    candidate.candidate_id, document
                )
            return existing
        try:
            baseline = self.store.legacy_final_selection(job.job_id, scene_ids)
        except IncompleteLegacySelectionError as error:
            self.store.hold_incomplete_qc_job(job.job_id, error.missing_scene_ids)
            return ()
        selected = {item.scene_id: item for item in baseline}
        prepared: list[dict[str, Any]] = []
        for scene in job.scenes:
            choice = selected[scene.scene_id]
            revision = next(
                item for item in self.store.scene_revisions(job.job_id, scene.scene_id)
                if item.revision == choice.revision
            )
            document = revision.parameters
            i2v = document.get("i2v")
            if not isinstance(i2v, Mapping):
                raise StateTransitionError("Original revision lacks its I2V contract.")
            prompt = i2v.get("prompt")
            negative = i2v.get("negative")
            seed = i2v.get("seed")
            if not isinstance(prompt, str) or not isinstance(negative, str):
                raise StateTransitionError("Original revision lacks immutable prompts.")
            try:
                seed = int(seed)
            except (TypeError, ValueError) as error:
                raise StateTransitionError("Original revision has an invalid I2V seed.") from error
            video_hash = _sha256_file(revision.video_path)
            if not revision.frame_path:
                raise StateTransitionError(
                    "Original revision lacks its immutable starting frame."
                )
            frame_hash = _sha256_file(revision.frame_path)
            document_hash = hashlib.sha256(
                canonical_json(document).encode("utf-8")
            ).hexdigest()
            identity = canonical_json(
                {"job_id": job.job_id, "scene_id": scene.scene_id,
                 "revision": revision.revision, "video_sha256": video_hash}
            )
            candidate_id = "candidate-original-" + hashlib.sha256(
                identity.encode("utf-8")
            ).hexdigest()[:32]
            prepared.append(
                {
                    "candidate_id": candidate_id,
                    "job_id": job.job_id,
                    "scene_id": scene.scene_id,
                    "revision": revision.revision,
                    "source_video_path": revision.video_path,
                    "source_video_sha256": video_hash,
                    "original_prompt": prompt,
                    "original_seed": seed,
                    "negative_prompt": negative,
                    "negative_prompt_sha256": hashlib.sha256(
                        negative.encode("utf-8")
                    ).hexdigest(),
                    "revision_document_sha256": document_hash,
                    "source_frame_path": revision.frame_path,
                    "source_frame_sha256": frame_hash,
                    "artifact_stage": (
                        QcArtifactStage.DRAFT.value
                        if Path(revision.video_path).name.casefold() == "draft.mp4"
                        else QcArtifactStage.LEGACY_FINAL.value
                    ),
                    "recipe_sha256": (
                        canonical_draft_recipe_sha256(
                            document,
                            source_frame_sha256=frame_hash,
                        )
                        if Path(revision.video_path).name.casefold() == "draft.mp4"
                        else None
                    ),
                }
            )
        return self.store.ensure_original_qc_candidates(prepared)

    def _completed_evaluation(self, candidate_id: str) -> QcEvaluationRecord:
        completed = [
            item for item in self.store.qc_evaluations(candidate_id)
            if item.state == "COMPLETE"
        ]
        if not completed:
            raise StateTransitionError("QC routing requires completed durable evidence.")
        return completed[-1]

    @staticmethod
    def _evaluation_defects(
        evaluation: QcEvaluationRecord,
    ) -> tuple[Mapping[str, Any], ...]:
        defects: list[Mapping[str, Any]] = []
        for window in _normalized_planner_windows(evaluation.suspect_windows):
            defects.extend(window["defects"])
        return tuple(defects)

    def _adaptive_strategy_history(self, candidate: QcCandidateRecord) -> tuple[str, ...]:
        aliases = {
            QcTier.A1: "new_seed",
            QcTier.A2: "reduced_motion_pressure",
            QcTier.B1: "constrained_prompt_repair",
            QcTier.B2: "regenerate_start_frame",
            QcTier.C: "current_scene_shot_redesign",
            QcTier.D: "current_scene_semantic_replan",
        }
        return tuple(
            item.strategy
            if item.strategy not in {item.tier.value.casefold(), "legacy_original"}
            else aliases.get(item.tier, item.strategy)
            for item in self.store.qc_candidates(candidate.job_id, candidate.scene_id)
            if item.tier != QcTier.ORIGINAL
        )

    def _schedule_server_validated_strategy(
        self,
        job: JobPayload,
        candidate: QcCandidateRecord,
        evaluation: QcEvaluationRecord,
        *,
        tier: QcTier,
        strategy: str,
        authority: str,
    ) -> QcCandidateRecord:
        source_document = self._revision_document(self.store, candidate)
        attempt_number = max(
            (item.attempt_number for item in self.store.qc_candidates(candidate.job_id, candidate.scene_id)),
            default=0,
        ) + 1
        seed = deterministic_adaptive_seed(
            job_id=candidate.job_id,
            scene_id=candidate.scene_id,
            parent_candidate_id=candidate.candidate_id,
            strategy=strategy,
            attempt_number=attempt_number,
        )
        if strategy == "reduced_motion_pressure":
            raw_plan = {
                "schema_version": "adaptive_repair_plan_v1",
                "authority_tier": "A",
                "strategy": strategy,
                "hypothesis": "The defect survived a seed retry and needs stronger source conditioning.",
                "preserve_start_frame": True,
                "changes": [{"path": "i2v.first_pass.preset", "operation": "replace", "value": strategy}],
                "failure_addressed": [normalized_defect_family(self._evaluation_defects(evaluation))],
                "difference_from_prior_attempts": "Changes verified first-pass conditioning controls as well as seed.",
                "expected_effect": "Reduced stochastic motion pressure with stronger start-frame adherence.",
            }
        else:
            if tier == QcTier.C:
                suffix = (
                    "Use one stable camera angle and simplified, sequential subject motion; "
                    "preserve every required identity, action, location, and transition."
                )
            else:
                variant = sum(
                    1 for item in self.store.qc_candidates(candidate.job_id, candidate.scene_id)
                    if item.tier == QcTier.D
                ) + 1
                patterns = (
                    "Use explicit chronological beats with unambiguous subject and body-part ownership.",
                    "Use continuity-preserving static staging and one clearly bounded action per beat.",
                    "Use simplified motion with the required transition shown before any secondary action.",
                )
                suffix = (
                    f"Current-scene semantic replan {variant}: "
                    f"{patterns[(variant - 1) % len(patterns)]} Preserve every identity and location."
                )
            raw_plan = {
                "schema_version": "adaptive_repair_plan_v1",
                "authority_tier": authority,
                "strategy": strategy,
                "hypothesis": "Repeated geometry or continuity evidence requires a higher-authority current-scene design.",
                "preserve_start_frame": True,
                "changes": [{"path": "i2v.prompt", "operation": "append", "value": suffix}],
                "failure_addressed": [normalized_defect_family(self._evaluation_defects(evaluation))],
                "difference_from_prior_attempts": "Escalates from sampling/prompt repair to a constrained current-scene redesign.",
                "expected_effect": "A simpler, explicit shot that retains the original scene contract.",
            }
        plan = validate_repair_plan(
            raw_plan,
            minimum_authority=candidate.authority_tier,
            failed_strategies=self._adaptive_strategy_history(candidate),
        )
        document, changed = apply_repair_plan(source_document, plan, controller_seed=seed)
        validated = validate_scene_edit(job, candidate.scene_id, document)
        revision = next(
            item for item in self.store.scene_revisions(candidate.job_id, candidate.scene_id)
            if item.revision == candidate.revision
        )
        if not revision.frame_path:
            raise StateTransitionError("Adaptive repair lost its starting frame.")
        next_revision = max(
            item.revision for item in self.store.scene_revisions(candidate.job_id, candidate.scene_id)
        ) + 1
        child = self.store.create_adaptive_candidate_revision(
            parent_candidate_id=candidate.candidate_id,
            tier=tier,
            parameters=validated.document,
            frame_path=revision.frame_path,
            source_video_path=str(
                self.layout.scene_draft_path(candidate.job_id, candidate.scene_id, next_revision)
            ),
            authority_tier=authority,
            strategy=strategy,
            defect_family=normalized_defect_family(self._evaluation_defects(evaluation)),
            intervention={
                "schema_version": "adaptive_attempt_v1",
                "plan": raw_plan,
                "hypothesis": plan.hypothesis,
                "fields_changed": changed,
                "source_evaluation_id": evaluation.evaluation_id,
                "source_recipe_sha256": candidate.recipe_sha256,
                "planner": "controller_validated_strategy",
            },
            human_feedback=(
                None
                if self.store.qc_human_decision(candidate.candidate_id) is None
                else {
                    "decision": self.store.qc_human_decision(candidate.candidate_id).decision.value,
                    "note": self.store.qc_human_decision(candidate.candidate_id).note,
                }
            ),
        )
        self.store.set_qc_candidate_state(
            candidate.candidate_id, QcCandidateState.SUPERSEDED, next_action=None
        )
        return child

    def _schedule_start_frame_regeneration(
        self,
        job: JobPayload,
        candidate: QcCandidateRecord,
        evaluation: QcEvaluationRecord,
    ) -> QcCandidateRecord:
        """Skip seed spinning when frame-zero evidence is already defective."""
        document = json.loads(canonical_json(self._revision_document(self.store, candidate)))
        attempt_number = max(
            (item.attempt_number for item in self.store.qc_candidates(candidate.job_id, candidate.scene_id)),
            default=0,
        ) + 1
        t2i_seed = deterministic_adaptive_seed(
            job_id=candidate.job_id, scene_id=candidate.scene_id,
            parent_candidate_id=candidate.candidate_id,
            strategy="regenerate_start_frame_t2i", attempt_number=attempt_number,
        )
        i2v_seed = deterministic_adaptive_seed(
            job_id=candidate.job_id, scene_id=candidate.scene_id,
            parent_candidate_id=candidate.candidate_id,
            strategy="regenerate_start_frame_i2v", attempt_number=attempt_number,
        )
        t2i = document.get("t2i")
        i2v = document.get("i2v")
        if not isinstance(t2i, dict) or not isinstance(i2v, dict):
            raise StateTransitionError("B2 requires the full T2I/I2V contract.")
        prior_t2i_seed, prior_i2v_seed = t2i.get("seed"), i2v.get("seed")
        t2i["seed"] = str(t2i_seed) if isinstance(prior_t2i_seed, str) else t2i_seed
        i2v["seed"] = str(i2v_seed) if isinstance(prior_i2v_seed, str) else i2v_seed
        validated = validate_scene_edit(job, candidate.scene_id, document)
        source_revision = next(
            item for item in self.store.scene_revisions(candidate.job_id, candidate.scene_id)
            if item.revision == candidate.revision
        )
        if not source_revision.frame_path or not Path(source_revision.frame_path).is_file():
            raise StateTransitionError("B2 source lost its immutable starting frame.")
        next_revision = max(
            item.revision for item in self.store.scene_revisions(candidate.job_id, candidate.scene_id)
        ) + 1
        frame_path = self.layout.scene_frame_path(
            candidate.job_id, candidate.scene_id, next_revision
        )
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        if not frame_path.is_file():
            shutil.copyfile(source_revision.frame_path, frame_path)
        defects = self._evaluation_defects(evaluation)
        child = self.store.create_adaptive_candidate_revision(
            parent_candidate_id=candidate.candidate_id,
            tier=QcTier.B2,
            parameters=validated.document,
            frame_path=str(frame_path),
            source_video_path=str(
                self.layout.scene_draft_path(
                    candidate.job_id, candidate.scene_id, next_revision
                )
            ),
            authority_tier="B",
            strategy="regenerate_start_frame",
            defect_family=normalized_defect_family(defects),
            intervention={
                "schema_version": "adaptive_attempt_v1",
                "strategy": "regenerate_start_frame",
                "hypothesis": "The defect is visible before meaningful motion, so the starting image must change.",
                "preserve_start_frame": False,
                "fields_changed": {
                    "t2i.seed": {"before": prior_t2i_seed, "after": t2i["seed"]},
                    "i2v.seed": {"before": prior_i2v_seed, "after": i2v["seed"]},
                },
                "source_evaluation_id": evaluation.evaluation_id,
            },
        )
        self.store.set_qc_candidate_state(
            candidate.candidate_id, QcCandidateState.SUPERSEDED, next_action=None
        )
        return child

    def route_completed_evaluation(
        self,
        job: JobPayload,
        candidate_id: str,
        *,
        backend: Any | None = None,
        planner_identity: Mapping[str, Any] | None = None,
    ) -> QcCandidateRecord:
        """Route PASS to humans and quality defects through monotonic A-to-D repair."""
        candidate = self.store.qc_candidate(candidate_id)
        evaluation = self._completed_evaluation(candidate_id)
        decision = evaluation.normalized_decision
        if decision is None:
            raise StateTransitionError("Completed QC evidence lacks a normalized decision.")
        defects = self._evaluation_defects(evaluation)
        self.store.update_adaptive_attempt(
            candidate_id,
            state="QC_COMPLETE",
            result={
                "qwen_decision": decision.value,
                "defect_categories": sorted({str(item.get("category", "other")) for item in defects}),
                "suspect_windows": list(_normalized_planner_windows(evaluation.suspect_windows)),
                "maximum_severity": max((int(item.get("severity", 0)) for item in defects), default=0),
                "maximum_confidence": max((float(item.get("confidence", 0.0)) for item in defects), default=0.0),
            },
        )
        human_decision = self.store.qc_human_decision(candidate_id)
        if (
            human_decision is not None
            and human_decision.decision in {QcHumanDecision.REJECT, QcHumanDecision.HOLD}
            and candidate.state == QcCandidateState.PENDING_QC
            and candidate.next_action == "route_existing_evidence"
        ):
            # Human defect evidence outranks a prior Qwen PASS after explicit Resume.
            decision = QcDecision.FAIL

        if decision == QcDecision.PASS:
            state = (
                QcCandidateState.ACCEPTED
                if self.settings.auto_advance_pass
                else QcCandidateState.PASS_PENDING_HUMAN
            )
            routed = self.store.set_qc_candidate_state(
                candidate_id,
                state,
                next_action=None if state == QcCandidateState.ACCEPTED else "await_human",
            )
            if state == QcCandidateState.ACCEPTED:
                self.store.promote_accepted_qc_candidate(candidate_id)
            return routed

        if (
            candidate.tier in {QcTier.ORIGINAL, QcTier.A1, QcTier.A2}
            and self._b2_start_frame_detected(evaluation)
        ):
            return self._schedule_start_frame_regeneration(job, candidate, evaluation)

        if candidate.tier == QcTier.B1:
            if self._b2_start_frame_detected(evaluation):
                candidate_b2 = self.schedule_b2_retry(
                    original_job=job,
                    source_candidate_id=candidate.candidate_id,
                    source_document=self._revision_document(self.store, candidate),
                )
                if candidate_b2.recipe_sha256:
                    self.store.ensure_adaptive_attempt(
                        candidate_b2.candidate_id,
                        defect_family=normalized_defect_family(defects),
                        intervention={
                            "schema_version": "adaptive_attempt_v1",
                            "strategy": "regenerate_start_frame",
                            "source_evaluation_id": evaluation.evaluation_id,
                        },
                    )
                self.store.set_qc_candidate_state(
                    candidate_id,
                    QcCandidateState.SUPERSEDED,
                    next_action=None,
                )
                return candidate_b2
            return self._schedule_server_validated_strategy(
                job, candidate, evaluation, tier=QcTier.C,
                strategy="current_scene_shot_redesign", authority="C",
            )

        if candidate.tier == QcTier.B2:
            return self._schedule_server_validated_strategy(
                job, candidate, evaluation, tier=QcTier.C,
                strategy="current_scene_shot_redesign", authority="C",
            )

        if candidate.tier == QcTier.C:
            return self._schedule_server_validated_strategy(
                job, candidate, evaluation, tier=QcTier.D,
                strategy="current_scene_semantic_replan", authority="D",
            )

        if candidate.tier == QcTier.D:
            if candidate.next_action == "route_existing_evidence":
                variant = sum(
                    1 for item in self.store.qc_candidates(candidate.job_id, candidate.scene_id)
                    if item.tier == QcTier.D
                ) + 1
                return self._schedule_server_validated_strategy(
                    job, candidate, evaluation, tier=QcTier.D,
                    strategy=f"deferred_current_scene_replan_v{variant}", authority="D",
                )
            return self.store.set_qc_candidate_state(
                candidate_id,
                QcCandidateState.DEFERRED_AUTOMATED_REPAIR,
                next_action="deferred_untried_strategy_required",
            )

        source_document = self._revision_document(self.store, candidate)
        try:
            self.store.validate_qc_candidate_source_identity(
                candidate.candidate_id,
                source_document,
            )
        except StateTransitionError as error:
            return self.store.set_qc_candidate_state(
                candidate_id,
                QcCandidateState.HOLD_FOR_REVIEW,
                next_action="source_identity_failure:" + str(error),
            )
        if candidate.tier == QcTier.ORIGINAL:
            retry = schedule_a1_retry(
                self.store,
                self.layout,
                original_job=job,
                source_candidate_id=candidate_id,
                source_document=source_document,
            )
            self.store.ensure_adaptive_attempt(
                retry.candidate.candidate_id,
                defect_family=normalized_defect_family(defects),
                intervention={
                    "schema_version": "adaptive_attempt_v1",
                    "strategy": "new_seed",
                    "source_evaluation_id": evaluation.evaluation_id,
                    "fields_changed": {"i2v.seed": {"before": candidate.current_seed, "after": retry.candidate.current_seed}},
                },
                human_feedback=(
                    None if human_decision is None
                    else {"decision": human_decision.decision.value, "note": human_decision.note}
                ),
            )
            self.store.set_qc_candidate_state(
                candidate_id,
                QcCandidateState.SUPERSEDED,
                next_action=None,
            )
            return retry.candidate

        if candidate.tier == QcTier.A1:
            return self._schedule_server_validated_strategy(
                job, candidate, evaluation, tier=QcTier.A2,
                strategy="reduced_motion_pressure", authority="A",
            )
        if candidate.tier != QcTier.A2:
            raise StateTransitionError("Adaptive repair has no route for this candidate tier.")
        existing = self.store.qc_repairs(candidate_id)
        if existing:
            repair = existing[0]
            if repair.status != "ACCEPTED":
                return self._schedule_server_validated_strategy(
                    job, candidate, evaluation, tier=QcTier.C,
                    strategy="current_scene_shot_redesign", authority="C",
                )
            # The repair evidence is deliberately durable before the child
            # revision is allocated.  Re-enter the idempotent scheduler so a
            # restart in that crash window finishes candidate creation instead
            # of assuming the result row already exists.
            retry = schedule_b1_retry(
                self.store,
                self.layout,
                original_job=job,
                source_candidate_id=candidate_id,
                evaluation_id=evaluation.evaluation_id,
                source_document=source_document,
                raw_output=repair.raw_output,
                planner_identity=repair.planner_identity,
                repair_input_hash=repair.repair_input_sha256,
                prior_repair_summaries=repair.prior_repair_summaries,
            )
            if retry.candidate is None:
                return self.store.qc_candidate(candidate_id)
            self.store.ensure_adaptive_attempt(
                retry.candidate.candidate_id,
                defect_family=normalized_defect_family(defects),
                intervention={
                    "schema_version": "adaptive_attempt_v1",
                    "strategy": "constrained_prompt_repair",
                    "source_evaluation_id": evaluation.evaluation_id,
                    "planner_identity": dict(repair.planner_identity),
                    "planner_prompt_sha256": repair.planner_identity.get("planner_prompt_sha256"),
                    "repair_input_sha256": repair.repair_input_sha256,
                    "prior_history": list(repair.prior_repair_summaries),
                },
            )
            self.store.set_qc_candidate_state(
                candidate_id,
                QcCandidateState.SUPERSEDED,
                next_action=None,
            )
            return retry.candidate
        if backend is None or planner_identity is None:
            raise QcControllerError("A1 FAIL requires the isolated text planner backend.")
        request = self._repair_request(job, candidate, evaluation)
        durable_planner_identity = {
            **dict(planner_identity),
            "planner_prompt_version": request.prompt.version,
            "planner_prompt_sha256": request.prompt.sha256,
            "system_prompt_version": request.prompt.system_prompt_version,
            "system_prompt_sha256": request.prompt.system_prompt_sha256,
            "request_recipe_version": request.prompt.request_recipe_version,
        }
        claim, created = self.store.claim_qc_repair_planner(
            candidate_id=candidate_id,
            evaluation_id=evaluation.evaluation_id,
            repair_input_sha256=request.repair_input_sha256,
            planner_identity=durable_planner_identity,
        )
        if not created:
            # A durable claim with no repair record means the process died
            # after crossing the external-inference boundary.  The remote
            # response is unknowable, so repeating the model call would break
            # exactly-once semantics.  Persist that ambiguity and hold.
            retry = schedule_b1_retry(
                self.store,
                self.layout,
                original_job=job,
                source_candidate_id=candidate_id,
                evaluation_id=evaluation.evaluation_id,
                source_document=source_document,
                raw_output="",
                planner_identity=claim.planner_identity,
                repair_input_hash=request.repair_input_sha256,
                prior_repair_summaries=request.previous_repairs,
                planner_failure_reason=(
                    "planner_invocation_ambiguous_after_restart"
                ),
            )
            self.store.complete_qc_repair_planner_claim(
                candidate_id, state="FAILED_CLOSED"
            )
            return self.store.qc_candidate(candidate_id)
        try:
            response = backend.plan_repair(request)
            retry = schedule_b1_retry(
                self.store,
                self.layout,
                original_job=job,
                source_candidate_id=candidate_id,
                evaluation_id=evaluation.evaluation_id,
                source_document=source_document,
                raw_output=response.raw_text,
                planner_identity=durable_planner_identity,
                repair_input_hash=request.repair_input_sha256,
                prior_repair_summaries=request.previous_repairs,
            )
        except Exception as error:
            LOGGER.exception("B1 planner failed closed for %s.", candidate_id)
            repairs = self.store.qc_repairs(candidate_id)
            if not repairs:
                schedule_b1_retry(
                    self.store,
                    self.layout,
                    original_job=job,
                    source_candidate_id=candidate_id,
                    evaluation_id=evaluation.evaluation_id,
                    source_document=source_document,
                    raw_output="",
                    planner_identity=durable_planner_identity,
                    repair_input_hash=request.repair_input_sha256,
                    prior_repair_summaries=request.previous_repairs,
                    planner_failure_reason=(
                        "planner_infrastructure_failure:" + type(error).__name__
                    ),
                )
                self.store.complete_qc_repair_planner_claim(
                    candidate_id, state="FAILED_CLOSED"
                )
            else:
                self.store.complete_qc_repair_planner_claim(
                    candidate_id, state="COMPLETED"
                )
            return self.store.set_qc_candidate_state(
                candidate_id,
                QcCandidateState.HOLD_FOR_REVIEW,
                next_action="planner_failure:" + type(error).__name__,
            )
        self.store.complete_qc_repair_planner_claim(
            candidate_id, state="COMPLETED"
        )
        if retry.candidate is None:
            return self._schedule_server_validated_strategy(
                job, candidate, evaluation, tier=QcTier.C,
                strategy="current_scene_shot_redesign", authority="C",
            )
        self.store.ensure_adaptive_attempt(
            retry.candidate.candidate_id,
            defect_family=normalized_defect_family(defects),
            intervention={
                "schema_version": "adaptive_attempt_v1",
                "strategy": "constrained_prompt_repair",
                "source_evaluation_id": evaluation.evaluation_id,
                "planner_identity": durable_planner_identity,
                "planner_prompt_sha256": request.prompt.sha256,
                "repair_input_sha256": request.repair_input_sha256,
                "prior_history": list(request.previous_repairs),
            },
        )
        self.store.set_qc_candidate_state(
            candidate_id,
            QcCandidateState.SUPERSEDED,
            next_action=None,
        )
        return retry.candidate

    def _repair_request(
        self,
        job: JobPayload,
        candidate: QcCandidateRecord,
        evaluation: QcEvaluationRecord,
    ) -> RepairPlannerRequest:
        document = self._revision_document(self.store, candidate)
        fixed = json.loads(canonical_json(document))
        fixed_i2v = fixed.get("i2v", {})
        generation_config = {
            key: value for key, value in fixed_i2v.items()
            if key not in {"prompt", "negative"}
        } if isinstance(fixed_i2v, Mapping) else {}
        suspect_windows = _normalized_planner_windows(evaluation.suspect_windows)
        defects = tuple(
            defect
            for window in suspect_windows
            for defect in window["defects"]
        )
        normalized_qc = {
            "decision": evaluation.normalized_decision.value,
            "strong_window_count": evaluation.strong_window_count,
            "frame_accounting": dict(evaluation.frame_accounting),
            "defect_categories": sorted(
                {str(item["category"]) for item in defects}
            ),
            "maximum_defect_severity": max(
                (int(item["severity"]) for item in defects),
                default=None,
            ),
            "evaluation_state": evaluation.state,
        }
        previous_items: list[Mapping[str, Any]] = []
        for lineage_candidate in self.store.qc_candidates(candidate.job_id, candidate.scene_id):
            evaluations = []
            for prior_evaluation in self.store.qc_evaluations(lineage_candidate.candidate_id):
                prior_windows = _normalized_planner_windows(prior_evaluation.suspect_windows)
                evaluations.append(
                    {
                        "evaluation_id": prior_evaluation.evaluation_id,
                        "decision": (
                            None if prior_evaluation.normalized_decision is None
                            else prior_evaluation.normalized_decision.value
                        ),
                        "defect_categories": sorted({
                            str(defect["category"])
                            for window in prior_windows for defect in window["defects"]
                        }),
                        "suspect_windows": list(prior_windows),
                    }
                )
            human = self.store.qc_human_decision(lineage_candidate.candidate_id)
            previous_items.append(
                {
                    "candidate_id": lineage_candidate.candidate_id,
                    "parent_candidate_id": lineage_candidate.parent_candidate_id,
                    "attempt_number": lineage_candidate.attempt_number,
                    "authority_tier": lineage_candidate.authority_tier,
                    "strategy": lineage_candidate.strategy,
                    "recipe_sha256": lineage_candidate.recipe_sha256,
                    "artifact_stage": lineage_candidate.artifact_stage.value,
                    "qwen_evaluations": evaluations,
                    "planner_repairs": [
                        {
                            "status": repair.status,
                            "reason": repair.reason,
                            "repair_input_sha256": repair.repair_input_sha256,
                            "proposed_patch": dict(repair.proposed_patch),
                        }
                        for repair in self.store.qc_repairs(lineage_candidate.candidate_id)
                    ],
                    "human_feedback": (
                        None if human is None else {"decision": human.decision.value, "note": human.note}
                    ),
                }
            )
        for attempt in self.store.adaptive_attempts(candidate.job_id, candidate.scene_id):
            previous_items.append(
                {
                    "adaptive_attempt_id": attempt.attempt_id,
                    "candidate_id": attempt.candidate_id,
                    "attempt_number": attempt.attempt_number,
                    "authority_tier": attempt.authority_tier,
                    "strategy": attempt.strategy,
                    "defect_family": attempt.defect_family,
                    "state": attempt.state,
                    "intervention": dict(attempt.intervention),
                    "human_feedback": attempt.human_feedback,
                }
            )
        previous = tuple(previous_items)
        prompt = load_repair_planner_prompt(
            self.prompt_root / "production_i2v_repair_v1.txt"
        )
        input_document = {
            "job": {"job_id": job.job_id},
            "scene": {"scene_id": candidate.scene_id},
            "source": {
                "candidate_id": candidate.candidate_id,
                "candidate_sha256": candidate.source_video_sha256,
                "evaluation_id": evaluation.evaluation_id,
                "source_revision": candidate.revision,
                "source_document_sha256": hashlib.sha256(
                    canonical_json(fixed).encode("utf-8")
                ).hexdigest(),
            },
            "current_i2v_prompt": candidate.current_prompt,
            "negative_prompt": candidate.negative_prompt,
            "fixed_scene_facts": fixed,
            "generation_config": generation_config,
            "normalized_qc": normalized_qc,
            "suspect_windows": list(suspect_windows),
            "previous_repairs": list(previous),
            "mutable_fields": ["i2v.prompt"],
            "locked_fields": ["*", "!i2v.prompt"],
            "planner_prompt_sha256": prompt.sha256,
        }
        input_hash = hashlib.sha256(
            canonical_json(input_document).encode("utf-8")
        ).hexdigest()
        return RepairPlannerRequest(
            job_identity=input_document["job"],
            scene_identity=input_document["scene"],
            source_identity={**input_document["source"], "repair_input_sha256": input_hash},
            current_i2v_prompt=candidate.current_prompt,
            negative_prompt=candidate.negative_prompt,
            fixed_scene_facts=fixed,
            generation_config=generation_config,
            normalized_qc=normalized_qc,
            suspect_windows=suspect_windows,
            previous_repairs=previous,
            mutable_fields=("i2v.prompt",),
            locked_fields=("all fields except i2v.prompt",),
            repair_input_sha256=input_hash,
            prompt=prompt,
        )

    def _strict_generation_barrier(self, supervisor: Any) -> None:
        running, pending = supervisor.comfy.queue_counts()
        if running or pending:
            raise QcControllerError("ComfyUI queue is not idle at the QC epoch boundary.")
        supervisor.comfy.free_memory()
        running, pending = supervisor.comfy.queue_counts()
        if running or pending:
            raise QcControllerError("ComfyUI work appeared while entering the QC epoch.")

    def _evaluate_candidate(
        self,
        candidate: QcCandidateRecord,
        backend: Any,
        identity: BackendIdentity,
    ) -> QcEvaluationRecord:
        if candidate.artifact_stage == QcArtifactStage.FINAL:
            raise QcControllerError(
                "FINAL artifacts are never sent through the ordinary Qwen draft-QC loop."
            )
        source = Path(candidate.source_video_path)
        if not source.is_file() or _sha256_file(source) != candidate.source_video_sha256:
            raise QcControllerError("QC source video bytes changed after candidate binding.")
        rubric = load_production_rubric(
            self.prompt_root / "production_ltx_video_qc_v1.txt"
        )
        effective_document = self.settings.effective_document()
        effective_hash = self.settings.effective_sha256()
        key = evaluation_idempotency_key(
            source_video_sha256=candidate.source_video_sha256,
            evaluator_id=identity.evaluator_id,
            evaluator_version=identity.evaluator_version,
            backend_version=identity.backend_version,
            executable_sha256=identity.executable_sha256,
            model_sha256=identity.model_sha256,
            projector_sha256=identity.projector_sha256,
            effective_config_sha256=effective_hash,
            prompt_sha256=rubric.sha256,
        )
        evaluation_id = "evaluation-" + key[:32]
        existing = self.store.begin_qc_evaluation(
            evaluation_id=evaluation_id,
            idempotency_key=key,
            candidate_id=candidate.candidate_id,
            source_video_path=candidate.source_video_path,
            source_video_sha256=candidate.source_video_sha256,
            evaluator_identity=_identity_mapping(identity),
            effective_config=effective_document,
            effective_config_sha256=effective_hash,
            prompt_version=rubric.version,
            prompt_sha256=rubric.sha256,
            sampling_config={
                "fps": self.settings.sampling_fps,
                "preprocessing": effective_document["sampling"]["preprocessing"],
            },
            window_config={
                "frames_per_window": self.settings.frames_per_window,
                "minimum_strong_windows": self.settings.minimum_strong_windows,
            },
        )
        if existing.state == "COMPLETE":
            return existing
        evidence_root = (
            Path(candidate.source_video_path).parent
            / "qc"
            / "evaluations"
            / evaluation_id
        )
        manifest_path = evidence_root / "result.json"
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if (
                    manifest.get("evaluation_id") != evaluation_id
                    or manifest.get("candidate_id") != candidate.candidate_id
                    or manifest.get("source_video_sha256") != candidate.source_video_sha256
                ):
                    raise QcControllerError("Saved QC evidence identity is stale.")
                normalized = manifest["normalized"]
                raw_path = Path(manifest["raw_path"]).resolve()
                accounting_path = Path(manifest["frame_accounting_path"]).resolve()
                for evidence_path in (raw_path, accounting_path):
                    evidence_path.relative_to(evidence_root.resolve())
                return self.store.complete_qc_evaluation(
                    evaluation_id,
                    raw_result=raw_path.read_text(encoding="utf-8"),
                    normalized_decision=QcDecision(normalized["decision"]),
                    suspect_windows=manifest.get("windows", []),
                    strong_window_count=int(normalized["strong_window_count"]),
                    frame_accounting=json.loads(accounting_path.read_text(encoding="utf-8")),
                    evidence_manifest_path=str(manifest_path),
                    evidence_manifest_sha256=_sha256_file(manifest_path),
                    next_action="route",
                )
            except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
                raise QcControllerError(
                    "Incomplete QC evaluation has invalid persisted evidence."
                ) from error
        self.store.set_qc_candidate_state(
            candidate.candidate_id, QcCandidateState.QC_RUNNING,
            next_action="evaluate",
        )
        temporary_parent = self.layout.root / "tmp" / "qc"
        temporary_parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=temporary_parent) as temporary:
            sampled = self.sample_video(
                source,
                target_fps=self.settings.sampling_fps,
                ffprobe_command=self.ffprobe_command,
                ffmpeg_command=self.ffmpeg_command,
                temporary_root=Path(temporary),
            )
            if (
                dict(sampled.preprocessing)
                != effective_document["sampling"]["preprocessing"]
            ):
                raise QcControllerError(
                    "Sampled frames do not match the validated preprocessing identity."
                )
            result = HeadlessVideoEvaluator(
                backend,
                rubric,
                policy=QcEvidencePolicy(
                    self.settings.minimum_error_severity,
                    self.settings.minimum_error_confidence,
                    self.settings.minimum_strong_windows,
                ),
                frames_per_window=self.settings.frames_per_window,
            ).evaluate_sampled(sampled)
        raw_path = write_immutable_text(evidence_root / "raw.txt", result.raw_result)
        accounting_path = write_immutable_json(
            evidence_root / "frame-accounting.json", result.frame_accounting
        )
        windows = [
            {
                "window_number": item.window_number,
                "source_frame_indices": list(item.source_frame_indices),
                "timestamps_seconds": list(item.timestamps_seconds),
                "image_sha256s": list(item.image_sha256s),
                "response": item.response.to_dict(),
                "confirmation_of_window": item.confirmation_of_window,
            }
            for item in result.normalized.windows
        ]
        if result.normalized.confirmation is not None:
            item = result.normalized.confirmation
            windows.append(
                {
                    "window_number": item.window_number,
                    "source_frame_indices": list(item.source_frame_indices),
                    "timestamps_seconds": list(item.timestamps_seconds),
                    "image_sha256s": list(item.image_sha256s),
                    "response": item.response.to_dict(),
                    "confirmation_of_window": item.confirmation_of_window,
                }
            )
        manifest = {
            "schema_version": 1,
            "evaluation_id": evaluation_id,
            "candidate_id": candidate.candidate_id,
            "source_video_path": candidate.source_video_path,
            "source_video_sha256": candidate.source_video_sha256,
            "evaluator_identity": _identity_mapping(identity),
            "effective_config_sha256": effective_hash,
            "prompt_version": rubric.version,
            "prompt_sha256": rubric.sha256,
            "normalized": result.normalized.to_dict(),
            "windows": windows,
            "raw_path": str(raw_path),
            "frame_accounting_path": str(accounting_path),
        }
        manifest_path = write_immutable_json(evidence_root / "result.json", manifest)
        manifest_hash = _sha256_file(manifest_path)
        return self.store.complete_qc_evaluation(
            evaluation_id,
            raw_result=result.raw_result,
            normalized_decision=result.normalized.decision,
            suspect_windows=windows,
            strong_window_count=result.normalized.strong_window_count,
            frame_accounting=result.frame_accounting,
            evidence_manifest_path=str(manifest_path),
            evidence_manifest_sha256=manifest_hash,
            next_action="route",
        )

    def run_epoch(self, job: JobPayload, supervisor: Any) -> QcEpochResult:
        """Run bounded whole-model epochs; never load/unload once per scene."""
        # A prior shutdown/port verification failure retains the exact backend
        # owner.  Prove it is gone before any Comfy repair generation can run.
        self._close_active_backend()
        scene_ids = tuple(scene.scene_id for scene in job.scenes)
        if not self.settings.quality_control_enabled:
            return QcEpochResult(
                True,
                self.store.original_final_selection(job.job_id, scene_ids),
                False,
            )
        originals = self.register_original_candidates(job)
        if not originals:
            return QcEpochResult(False, (), True)
        generated = 0
        evaluated = 0
        seen_work_states: set[tuple[tuple[Any, ...], ...]] = set()
        while True:
            candidates = self.store.qc_candidates(job.job_id)
            deferred = [
                item
                for item in candidates
                if item.state == QcCandidateState.DEFERRED_AUTOMATED_REPAIR
                and item.next_action == "resume_adaptive"
            ]
            if deferred:
                for item in deferred:
                    self.store.set_qc_candidate_state(
                        item.candidate_id,
                        QcCandidateState.PENDING_QC,
                        next_action="route_existing_evidence",
                    )
                continue
            finals = [
                item
                for item in candidates
                if item.state
                in {QcCandidateState.FINAL_PENDING, QcCandidateState.FINAL_RENDERING}
            ]
            generation = [
                item for item in candidates
                if item.state in {QcCandidateState.PENDING_GENERATION, QcCandidateState.GENERATING}
            ]
            pending_before_generation = [
                item for item in candidates
                if item.state in {
                    QcCandidateState.PENDING_QC,
                    QcCandidateState.QC_RUNNING,
                }
            ]
            if not generation and not pending_before_generation and not finals:
                break
            work_state = tuple(
                (
                    item.candidate_id,
                    item.tier.value,
                    item.state.value,
                    item.infrastructure_failure_count,
                    item.next_action,
                    item.source_video_sha256,
                    item.generation_prompt_id,
                )
                for item in candidates
            )
            if work_state in seen_work_states:
                raise QcControllerError(
                    "QC durable work state repeated without progress; refusing a retry loop."
                )
            seen_work_states.add(work_state)
            if finals:
                if self.qc_port_open():
                    raise QcControllerError(
                        "Qwen must be unloaded before approved FINAL rendering."
                    )
                supervisor.render_qc_finals(job, finals)
                generated += len(finals)
                supervisor.release_memory()
                continue
            if generation:
                if self.qc_port_open():
                    raise QcControllerError(
                        "The dedicated QC port is owned by an unverified process; "
                        "repair generation is blocked until it exits."
                    )
                retry_scheduled = False
                for candidate in generation:
                    try:
                        document = self._revision_document(self.store, candidate)
                        supervisor.render_qc_candidates(
                            job,
                            ((candidate, document),),
                        )
                        generated += 1
                    except RepairGenerationError as error:
                        persisted = self.store.record_qc_generation_failure(
                            candidate.candidate_id,
                            {
                                "kind": type(error).__name__,
                                "reason": error.reason,
                                "message": str(error),
                            },
                            retryable=error.retryable,
                        )
                        retry_scheduled = retry_scheduled or (
                            persisted.state == QcCandidateState.PENDING_GENERATION
                        )
                    except StateTransitionError as error:
                        self.store.record_qc_generation_failure(
                            candidate.candidate_id,
                            {
                                "kind": type(error).__name__,
                                "message": str(error),
                            },
                            retryable=False,
                        )
                    except Exception as error:
                        LOGGER.exception(
                            "Transient QC repair generation failed for %s.",
                            candidate.candidate_id,
                        )
                        persisted = self.store.record_qc_generation_failure(
                            candidate.candidate_id,
                            {
                                "kind": type(error).__name__,
                                "message": str(error),
                            },
                            retryable=True,
                        )
                        retry_scheduled = retry_scheduled or (
                            persisted.state == QcCandidateState.PENDING_GENERATION
                        )
                supervisor.release_memory()
                if retry_scheduled:
                    return QcEpochResult(False, (), False, generated, evaluated)

            pending = [
                item for item in self.store.qc_candidates(job.job_id)
                if item.state in {QcCandidateState.PENDING_QC, QcCandidateState.QC_RUNNING}
            ]
            if pending:
                self._strict_generation_barrier(supervisor)
                backend: Any | None = None
                identity: BackendIdentity | None = None
                try:
                    try:
                        backend = self.backend_factory()
                        self._active_backend = backend
                        identity = backend.start()
                    except Exception as error:
                        LOGGER.exception("QC worker failed before readiness.")
                        for candidate in pending:
                            self.store.record_qc_infrastructure_failure(
                                candidate.candidate_id,
                                {
                                    "kind": type(error).__name__,
                                    "message": str(error),
                                    "phase": "startup",
                                },
                            )
                        continue
                    abort_backend = False
                    for candidate in pending:
                        try:
                            self._evaluate_candidate(candidate, backend, identity)
                            evaluated += 1
                            self.route_completed_evaluation(
                                job,
                                candidate.candidate_id,
                                backend=backend,
                                planner_identity=_identity_mapping(identity),
                            )
                        except QcBackendError as error:
                            LOGGER.exception(
                                "QC backend trust boundary failed for %s; "
                                "aborting this worker epoch.",
                                candidate.candidate_id,
                            )
                            self.store.record_qc_infrastructure_failure(
                                candidate.candidate_id,
                                {"kind": type(error).__name__, "message": str(error)},
                            )
                            abort_backend = True
                            break
                        except Exception as error:
                            LOGGER.exception("QC failed closed for %s.", candidate.candidate_id)
                            self.store.record_qc_infrastructure_failure(
                                candidate.candidate_id,
                                {"kind": type(error).__name__, "message": str(error)},
                            )
                    if abort_backend:
                        continue
                finally:
                    if backend is not None:
                        backend.close()
                        if self._active_backend is backend:
                            self._active_backend = None
                continue

        candidates = self.store.qc_candidates(job.job_id)
        try:
            selection = self.store.qc_final_selection(job.job_id, scene_ids)
        except StateTransitionError:
            selection = ()
        if selection:
            return QcEpochResult(True, selection, False, generated, evaluated)
        waiting = any(
            item.state in {
                QcCandidateState.PASS_PENDING_HUMAN,
                QcCandidateState.HOLD_FOR_REVIEW,
                QcCandidateState.PAUSED,
                QcCandidateState.NEEDS_ADJACENT_SCENE_APPROVAL,
            }
            for item in candidates
        )
        self.store.transition(
            PipelineState.AWAITING_QC_REVIEW,
            job_id=job.job_id,
        )
        return QcEpochResult(False, (), waiting, generated, evaluated)
