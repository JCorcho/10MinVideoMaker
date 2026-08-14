"""Crash-safe state and resumable per-scene progress for the pipeline."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterator, Mapping, Sequence
from uuid import uuid4

from .contracts import JobPayload, job_content_fingerprint, parse_job_payload
from .qc_contracts import (
    QcCandidateState,
    QcDecision,
    QcHumanDecision,
    QcTier,
    evaluation_idempotency_key,
)


class PipelineState(StrEnum):
    IDLE = "idle"
    WAITING_FOR_GROK = "waiting_for_grok"
    AWAITING_REVIEW = "awaiting_review"
    DOWNLOADING_ASSETS = "downloading_assets"
    RUNNING_T2I = "running_t2i"
    RUNNING_I2V = "running_i2v"
    RUNNING_QC = "running_qc"
    AWAITING_QC_REVIEW = "awaiting_qc_review"
    QC_BLOCKED = "qc_blocked"
    STITCHING = "stitching"
    ERROR = "error"


class SceneState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ChunkState(StrEnum):
    """Durable lifecycle for one temporal continuation chunk."""

    PLANNED = "planned"
    BLOCKED_UPSTREAM = "blocked_upstream"
    READY = "ready"
    GENERATING_STAGE1 = "generating_stage1"
    STAGE1_PERSISTING = "stage1_persisting"
    STAGE1_COMPLETE = "stage1_complete"
    GENERATING_STAGE2 = "generating_stage2"
    STAGE2_PERSISTING = "stage2_persisting"
    DECODED = "decoded"
    VALIDATING = "validating"
    COMPLETE = "complete"
    FAILED_RETRYABLE = "failed_retryable"
    FAILED_TERMINAL = "failed_terminal"
    STALE_UPSTREAM = "stale_upstream"
    INVALIDATED = "invalidated"
    CANCELLED = "cancelled"


class JobState(StrEnum):
    AWAITING_REVIEW = "awaiting_review"
    QUEUED = "queued"
    RUNNING = "running"
    PARTIAL = "partial"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RemakeMode(StrEnum):
    VIDEO_ONLY = "video_only"
    IMAGE_AND_VIDEO = "image_and_video"


class RemakeBatchState(StrEnum):
    DRAFT = "draft"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ManualFinalState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class StateTransitionError(RuntimeError):
    """Raised when an operation would violate the single-job state machine."""


class IncompleteLegacySelectionError(StateTransitionError):
    """Raised with exact scene identities when pre-QC selection is incomplete."""

    def __init__(self, missing_scene_ids: Sequence[int], message: str):
        super().__init__(message)
        self.missing_scene_ids = tuple(sorted(set(int(item) for item in missing_scene_ids)))


@dataclass(frozen=True)
class PipelineSnapshot:
    state: PipelineState
    job_id: str | None
    active_scene_id: int | None
    error: str | None


@dataclass(frozen=True)
class QcJobHoldRecord:
    job_id: str
    kind: str
    missing_scene_ids: tuple[int, ...]
    evidence: Mapping[str, Any]
    evidence_sha256: str
    created_at: str


@dataclass(frozen=True)
class InboundJobClaim:
    """Result of one atomic Gmail-message/job claim attempt."""

    accepted: bool
    payload: JobPayload | None
    duplicate_content: bool = False
    source_job_id: str | None = None


@dataclass(frozen=True)
class SceneRecord:
    job_id: str
    scene_id: int
    state: SceneState
    frame_path: str | None
    video_path: str | None
    error: str | None
    t2i_attempts: int
    i2v_attempts: int
    prompt_id: str | None
    prompt_stage: str | None
    include_in_manual_final: bool


@dataclass(frozen=True)
class JobRecord:
    job_id: str
    status: JobState
    created_at: str
    updated_at: str
    final_path: str | None
    scene_count: int
    succeeded_count: int
    failed_count: int


@dataclass(frozen=True)
class SceneRevision:
    job_id: str
    scene_id: int
    revision: int
    remake_mode: RemakeMode
    parameters: Mapping[str, Any]
    state: SceneState
    frame_path: str | None
    video_path: str | None
    error: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class RemakeBatchRecord:
    batch_id: str
    state: RemakeBatchState
    collision_policy: str | None
    created_at: str
    updated_at: str
    item_count: int
    completed_count: int


@dataclass(frozen=True)
class RemakeItemRecord:
    batch_id: str
    position: int
    job_id: str
    scene_id: int
    revision: int
    state: SceneState
    error: str | None
    prompt_id: str | None
    prompt_stage: str | None


@dataclass(frozen=True)
class ManualFinalRecord:
    request_id: str
    job_id: str
    state: ManualFinalState
    output_path: str | None
    error: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class ManualFinalSceneSelection:
    scene_id: int
    revision: int
    video_path: str


@dataclass(frozen=True)
class ContinuationPlanRecord:
    job_id: str
    scene_id: int
    revision: int
    plan_hash: str
    plan: Mapping[str, Any]
    created_at: str


@dataclass(frozen=True)
class SceneChunkRecord:
    job_id: str
    scene_id: int
    revision: int
    chunk_index: int
    plan_hash: str
    chunk: Mapping[str, Any]
    state: ChunkState
    accepted_attempt_number: int | None
    accepted_artifact_hash: str | None
    error: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class ChunkAttemptRecord:
    job_id: str
    scene_id: int
    revision: int
    chunk_index: int
    attempt_number: int
    variation_index: int
    state: ChunkState
    seed: int
    parameters: Mapping[str, Any]
    upstream_artifact_hash: str | None
    artifact_manifest_path: str | None
    artifact_hash: str | None
    video_path: str | None
    result: Mapping[str, Any]
    error: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class ChunkProgress:
    total_count: int
    complete_count: int
    ready_count: int
    active_count: int
    blocked_count: int
    failed_count: int
    invalidated_count: int
    cancelled_count: int
    next_chunk_index: int | None


@dataclass(frozen=True)
class QcCandidateRecord:
    candidate_id: str
    job_id: str
    scene_id: int
    revision: int
    tier: QcTier
    parent_candidate_id: str | None
    source_video_path: str
    source_video_sha256: str | None
    original_prompt: str
    current_prompt: str
    original_seed: int
    current_seed: int
    negative_prompt: str
    negative_prompt_sha256: str
    state: QcCandidateState
    next_action: str | None
    infrastructure_failure_count: int
    last_failure: Mapping[str, Any] | None
    generation_prompt_id: str | None
    generation_prompt_stage: str | None
    generation_route: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class QcEvaluationRecord:
    evaluation_id: str
    idempotency_key: str
    candidate_id: str
    source_video_path: str
    source_video_sha256: str
    evaluator_identity: Mapping[str, Any]
    effective_config: Mapping[str, Any]
    effective_config_sha256: str
    prompt_version: str
    prompt_sha256: str
    sampling_config: Mapping[str, Any]
    window_config: Mapping[str, Any]
    raw_result: str | None
    normalized_decision: QcDecision | None
    suspect_windows: Sequence[Mapping[str, Any]]
    strong_window_count: int | None
    frame_accounting: Mapping[str, Any]
    evidence_manifest_path: str | None
    evidence_manifest_sha256: str | None
    next_action: str | None
    state: str
    started_at: str
    completed_at: str | None


@dataclass(frozen=True)
class QcRepairRecord:
    repair_id: str
    candidate_id: str
    evaluation_id: str | None
    planner_identity: Mapping[str, Any]
    repair_input_sha256: str
    raw_output: str
    proposed_patch: Mapping[str, Any]
    status: str
    reason: str | None
    prior_repair_summaries: Sequence[Any]
    evidence_manifest_path: str
    evidence_manifest_sha256: str
    created_at: str


@dataclass(frozen=True)
class QcRepairClaimRecord:
    claim_id: str
    candidate_id: str
    evaluation_id: str
    repair_input_sha256: str
    planner_identity: Mapping[str, Any]
    state: str
    created_at: str
    completed_at: str | None


@dataclass(frozen=True)
class QcHumanDecisionRecord:
    decision_id: str
    candidate_id: str
    decision: QcHumanDecision
    note: str | None
    actor: str
    result_sha256: str
    evidence_sha256: str
    created_at: str


@dataclass(frozen=True)
class QcHumanDecisionResult:
    decision: QcHumanDecisionRecord
    candidate: QcCandidateRecord
    replayed: bool


@dataclass(frozen=True)
class QcFinalizationPlanRecord:
    job_id: str
    version: int
    selection: tuple[Mapping[str, Any], ...]
    plan_sha256: str
    state: str
    final_path: str
    final_sha256: str | None
    next_request_id: str | None
    next_request_receipt: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class QcFinalizationStepRecord:
    job_id: str
    step_key: str
    kind: str
    state: str
    evidence: Mapping[str, Any]
    evidence_sha256: str
    receipt: Mapping[str, Any] | None
    created_at: str
    updated_at: str


_ACTIVE_CHUNK_STATES = frozenset(
    {
        ChunkState.GENERATING_STAGE1,
        ChunkState.STAGE1_PERSISTING,
        ChunkState.STAGE1_COMPLETE,
        ChunkState.GENERATING_STAGE2,
        ChunkState.STAGE2_PERSISTING,
        ChunkState.DECODED,
        ChunkState.VALIDATING,
    }
)
_FINAL_ATTEMPT_STATES = frozenset(
    {
        ChunkState.COMPLETE,
        ChunkState.FAILED_RETRYABLE,
        ChunkState.FAILED_TERMINAL,
        ChunkState.STALE_UPSTREAM,
        ChunkState.INVALIDATED,
        ChunkState.CANCELLED,
    }
)
_UINT64_MAX = (1 << 64) - 1


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise StateTransitionError(f"Evidence must be JSON serializable: {error}") from error


def _positive_revision(revision: int) -> int:
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        raise StateTransitionError("revision must be a positive integer.")
    return revision


def _nonnegative_chunk_index(chunk_index: int) -> int:
    if isinstance(chunk_index, bool) or not isinstance(chunk_index, int) or chunk_index < 0:
        raise StateTransitionError("chunk_index must be a non-negative integer.")
    return chunk_index


def _positive_attempt_number(attempt_number: int) -> int:
    if (
        isinstance(attempt_number, bool)
        or not isinstance(attempt_number, int)
        or attempt_number < 1
    ):
        raise StateTransitionError("attempt_number must be a positive integer.")
    return attempt_number


def _uint64_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= _UINT64_MAX:
        raise StateTransitionError("seed must be an unsigned 64-bit integer.")
    return seed


def _evidence_id(value: str, field: str) -> str:
    import re

    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value
    ):
        raise StateTransitionError(f"{field} contains unsafe identity characters.")
    return value


def _sha256(value: str, field: str) -> str:
    import re

    normalized = str(value).strip().casefold()
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise StateTransitionError(f"{field} must be a SHA-256 digest.")
    return normalized


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise StateTransitionError(f"{field} must be non-empty text.")
    return value


def _require_path_below(path: str, root: Path, field: str) -> str:
    value = _required_text(path, field)
    try:
        Path(value).expanduser().resolve().relative_to(root.expanduser().resolve())
    except (OSError, ValueError) as error:
        raise StateTransitionError(
            f"{field} must remain below its candidate revision evidence root."
        ) from error
    return value


def _json_load(value: object, default: Any) -> Any:
    if value is None:
        return default
    return json.loads(str(value))


class PipelineStateStore:
    """A small SQLite store with transactional claiming and scene-level recovery."""

    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _classify_old_i2v_owner(
        connection: sqlite3.Connection,
        job_id: str,
        scene_id: int,
        revision: int,
        prompt_id: object,
    ) -> str:
        """Split the old generic I2V stage without trusting a stale plan."""
        if not isinstance(prompt_id, str) or not prompt_id:
            return "i2v_legacy"
        rows = connection.execute(
            """
            SELECT result_json
            FROM chunk_attempts
            WHERE job_id = ? AND scene_id = ? AND revision = ?
            """,
            (job_id, scene_id, revision),
        ).fetchall()
        for row in rows:
            try:
                result = json.loads(row["result_json"])
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(result, Mapping):
                continue
            if any(
                result.get(key) == prompt_id
                for key in ("stage1_prompt_id", "stage2_prompt_id")
            ):
                return "i2v_continuation"
        return "i2v_legacy"

    def initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS pipeline_state (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    state TEXT NOT NULL,
                    job_id TEXT,
                    active_scene_id INTEGER,
                    error TEXT,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'queued',
                    final_path TEXT,
                    updated_at TEXT
                );
                CREATE TABLE IF NOT EXISTS scenes (
                    job_id TEXT NOT NULL,
                    scene_id INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    frame_path TEXT,
                    video_path TEXT,
                    error TEXT,
                    t2i_attempts INTEGER NOT NULL DEFAULT 0,
                    i2v_attempts INTEGER NOT NULL DEFAULT 0,
                    prompt_id TEXT,
                    prompt_stage TEXT,
                    include_in_manual_final INTEGER NOT NULL DEFAULT 1,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (job_id, scene_id),
                    FOREIGN KEY (job_id) REFERENCES jobs(job_id)
                );
                CREATE TABLE IF NOT EXISTS mail_messages (
                    message_key TEXT PRIMARY KEY,
                    job_id TEXT,
                    processed_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS scene_revisions (
                    job_id TEXT NOT NULL,
                    scene_id INTEGER NOT NULL,
                    revision INTEGER NOT NULL,
                    remake_mode TEXT NOT NULL,
                    parameters_json TEXT NOT NULL,
                    state TEXT NOT NULL,
                    frame_path TEXT,
                    video_path TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (job_id, scene_id, revision),
                    FOREIGN KEY (job_id, scene_id) REFERENCES scenes(job_id, scene_id)
                );
                CREATE TABLE IF NOT EXISTS remake_batches (
                    batch_id TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    collision_policy TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS remake_items (
                    batch_id TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    job_id TEXT NOT NULL,
                    scene_id INTEGER NOT NULL,
                    revision INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    error TEXT,
                    prompt_id TEXT,
                    prompt_stage TEXT,
                    PRIMARY KEY (batch_id, position),
                    UNIQUE (batch_id, job_id, scene_id, revision),
                    FOREIGN KEY (batch_id) REFERENCES remake_batches(batch_id),
                    FOREIGN KEY (job_id, scene_id, revision)
                        REFERENCES scene_revisions(job_id, scene_id, revision)
                );
                CREATE TABLE IF NOT EXISTS manual_final_requests (
                    request_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    selection_json TEXT NOT NULL,
                    output_path TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (job_id) REFERENCES jobs(job_id)
                );
                CREATE TABLE IF NOT EXISTS continuation_plans (
                    job_id TEXT NOT NULL,
                    scene_id INTEGER NOT NULL,
                    revision INTEGER NOT NULL CHECK (revision >= 1),
                    plan_hash TEXT NOT NULL,
                    plan_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (job_id, scene_id, revision),
                    FOREIGN KEY (job_id, scene_id) REFERENCES scenes(job_id, scene_id)
                );
                CREATE TABLE IF NOT EXISTS scene_chunks (
                    job_id TEXT NOT NULL,
                    scene_id INTEGER NOT NULL,
                    revision INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
                    plan_hash TEXT NOT NULL,
                    chunk_json TEXT NOT NULL,
                    state TEXT NOT NULL,
                    accepted_attempt_number INTEGER,
                    accepted_artifact_hash TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (job_id, scene_id, revision, chunk_index),
                    FOREIGN KEY (job_id, scene_id, revision)
                        REFERENCES continuation_plans(job_id, scene_id, revision)
                );
                CREATE TABLE IF NOT EXISTS chunk_attempts (
                    job_id TEXT NOT NULL,
                    scene_id INTEGER NOT NULL,
                    revision INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    attempt_number INTEGER NOT NULL CHECK (attempt_number >= 1),
                    variation_index INTEGER NOT NULL DEFAULT 0 CHECK (variation_index >= 0),
                    state TEXT NOT NULL,
                    seed TEXT NOT NULL,
                    parameters_json TEXT NOT NULL,
                    upstream_artifact_hash TEXT,
                    artifact_manifest_path TEXT,
                    artifact_hash TEXT,
                    video_path TEXT,
                    result_json TEXT NOT NULL DEFAULT '{}',
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (
                        job_id, scene_id, revision, chunk_index, attempt_number
                    ),
                    FOREIGN KEY (job_id, scene_id, revision, chunk_index)
                        REFERENCES scene_chunks(job_id, scene_id, revision, chunk_index)
                );
                CREATE TABLE IF NOT EXISTS qc_candidates (
                    candidate_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    scene_id INTEGER NOT NULL CHECK (scene_id >= 1),
                    revision INTEGER NOT NULL CHECK (revision >= 1),
                    tier TEXT NOT NULL,
                    parent_candidate_id TEXT,
                    source_video_path TEXT NOT NULL,
                    source_video_sha256 TEXT,
                    original_prompt TEXT NOT NULL,
                    current_prompt TEXT NOT NULL,
                    original_seed TEXT NOT NULL,
                    current_seed TEXT NOT NULL,
                    negative_prompt TEXT NOT NULL,
                    negative_prompt_sha256 TEXT NOT NULL,
                    state TEXT NOT NULL,
                    next_action TEXT,
                    infrastructure_failure_count INTEGER NOT NULL DEFAULT 0
                        CHECK (infrastructure_failure_count >= 0),
                    last_failure_json TEXT,
                    generation_prompt_id TEXT,
                    generation_prompt_stage TEXT,
                    generation_route TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE (job_id, scene_id, tier),
                    FOREIGN KEY (job_id, scene_id, revision)
                        REFERENCES scene_revisions(job_id, scene_id, revision),
                    FOREIGN KEY (parent_candidate_id) REFERENCES qc_candidates(candidate_id)
                );
                CREATE TABLE IF NOT EXISTS qc_evaluations (
                    evaluation_id TEXT PRIMARY KEY,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    candidate_id TEXT NOT NULL,
                    source_video_path TEXT NOT NULL,
                    source_video_sha256 TEXT NOT NULL,
                    evaluator_id TEXT NOT NULL,
                    evaluator_version TEXT NOT NULL,
                    backend_family TEXT NOT NULL,
                    backend_version TEXT NOT NULL,
                    executable_path TEXT NOT NULL,
                    executable_sha256 TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    model_path TEXT NOT NULL,
                    model_sha256 TEXT NOT NULL,
                    quantization TEXT NOT NULL,
                    projector_path TEXT NOT NULL,
                    projector_sha256 TEXT NOT NULL,
                    projector_precision TEXT NOT NULL,
                    gpu_uuid TEXT NOT NULL,
                    gpu_name TEXT NOT NULL,
                    effective_config_json TEXT NOT NULL,
                    effective_config_sha256 TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    prompt_sha256 TEXT NOT NULL,
                    sampling_config_json TEXT NOT NULL,
                    window_config_json TEXT NOT NULL,
                    raw_result TEXT,
                    normalized_decision TEXT,
                    suspect_windows_json TEXT NOT NULL DEFAULT '[]',
                    strong_window_count INTEGER,
                    frame_accounting_json TEXT NOT NULL DEFAULT '{}',
                    evidence_manifest_path TEXT,
                    evidence_manifest_sha256 TEXT,
                    next_action TEXT,
                    state TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    FOREIGN KEY (candidate_id) REFERENCES qc_candidates(candidate_id)
                );
                CREATE TABLE IF NOT EXISTS qc_repairs (
                    repair_id TEXT PRIMARY KEY,
                    candidate_id TEXT NOT NULL,
                    evaluation_id TEXT,
                    planner_identity_json TEXT NOT NULL,
                    repair_input_sha256 TEXT NOT NULL,
                    raw_output TEXT NOT NULL,
                    proposed_patch_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    reason TEXT,
                    prior_repair_summaries_json TEXT NOT NULL,
                    evidence_manifest_path TEXT NOT NULL,
                    evidence_manifest_sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (candidate_id) REFERENCES qc_candidates(candidate_id),
                    FOREIGN KEY (evaluation_id) REFERENCES qc_evaluations(evaluation_id)
                );
                CREATE TABLE IF NOT EXISTS qc_repair_claims (
                    claim_id TEXT PRIMARY KEY,
                    candidate_id TEXT NOT NULL UNIQUE,
                    evaluation_id TEXT NOT NULL,
                    repair_input_sha256 TEXT NOT NULL,
                    planner_identity_json TEXT NOT NULL,
                    state TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    FOREIGN KEY (candidate_id) REFERENCES qc_candidates(candidate_id),
                    FOREIGN KEY (evaluation_id) REFERENCES qc_evaluations(evaluation_id)
                );
                CREATE TABLE IF NOT EXISTS qc_human_decisions (
                    decision_id TEXT PRIMARY KEY,
                    candidate_id TEXT NOT NULL UNIQUE,
                    decision TEXT NOT NULL,
                    note TEXT,
                    actor TEXT NOT NULL,
                    result_sha256 TEXT NOT NULL,
                    evidence_sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (candidate_id) REFERENCES qc_candidates(candidate_id)
                );
                CREATE TABLE IF NOT EXISTS qc_job_holds (
                    job_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    missing_scene_ids_json TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    evidence_sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (job_id) REFERENCES jobs(job_id)
                );
                CREATE TABLE IF NOT EXISTS qc_finalization_plans (
                    job_id TEXT PRIMARY KEY,
                    version INTEGER NOT NULL,
                    selection_json TEXT NOT NULL,
                    plan_sha256 TEXT NOT NULL UNIQUE,
                    state TEXT NOT NULL,
                    final_path TEXT NOT NULL,
                    final_sha256 TEXT,
                    next_request_id TEXT,
                    next_request_receipt TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (job_id) REFERENCES jobs(job_id)
                );
                CREATE TABLE IF NOT EXISTS qc_finalization_steps (
                    job_id TEXT NOT NULL,
                    step_key TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    state TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    evidence_sha256 TEXT NOT NULL,
                    receipt_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (job_id, step_key),
                    FOREIGN KEY (job_id) REFERENCES qc_finalization_plans(job_id)
                );
                CREATE INDEX IF NOT EXISTS idx_scene_chunks_state
                    ON scene_chunks(job_id, scene_id, revision, state, chunk_index);
                CREATE INDEX IF NOT EXISTS idx_chunk_attempts_state
                    ON chunk_attempts(job_id, scene_id, revision, chunk_index, state);
                CREATE INDEX IF NOT EXISTS idx_qc_candidates_state
                    ON qc_candidates(job_id, scene_id, state, tier);
                CREATE INDEX IF NOT EXISTS idx_qc_evaluations_candidate
                    ON qc_evaluations(candidate_id, started_at);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_qc_repairs_one_per_candidate
                    ON qc_repairs(candidate_id);
                """
            )
            qc_candidate_columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(qc_candidates)"
                ).fetchall()
            }
            for column in (
                "generation_prompt_id",
                "generation_prompt_stage",
                "generation_route",
            ):
                if column not in qc_candidate_columns:
                    connection.execute(
                        f"ALTER TABLE qc_candidates ADD COLUMN {column} TEXT"
                    )
            scene_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(scenes)").fetchall()
            }
            prompt_stage_added = "prompt_stage" not in scene_columns
            for column, declaration in (
                ("t2i_attempts", "INTEGER NOT NULL DEFAULT 0"),
                ("i2v_attempts", "INTEGER NOT NULL DEFAULT 0"),
                ("prompt_id", "TEXT"),
                ("prompt_stage", "TEXT"),
                ("include_in_manual_final", "INTEGER NOT NULL DEFAULT 1"),
            ):
                if column not in scene_columns:
                    connection.execute(f"ALTER TABLE scenes ADD COLUMN {column} {declaration}")
            if prompt_stage_added:
                # Older builds persisted a prompt ID without its owning graph.
                # Reclaim only the one active RUNNING scene whose stage can be
                # proven from the durable pipeline state. Clear every ambiguous
                # ID so it can never be routed into the wrong workflow.
                pipeline = connection.execute(
                    """
                    SELECT state, job_id, active_scene_id
                    FROM pipeline_state WHERE singleton = 1
                    """
                ).fetchone()
                prompt_rows = connection.execute(
                    """
                    SELECT job_id, scene_id, state, prompt_id
                    FROM scenes WHERE prompt_id IS NOT NULL
                    """
                ).fetchall()
                for prompt_row in prompt_rows:
                    stage = None
                    is_active = bool(
                        pipeline is not None
                        and pipeline["job_id"] == prompt_row["job_id"]
                        and pipeline["active_scene_id"] == prompt_row["scene_id"]
                        and SceneState(prompt_row["state"]) == SceneState.RUNNING
                    )
                    if is_active and PipelineState(pipeline["state"]) == PipelineState.RUNNING_T2I:
                        stage = "t2i"
                    elif (
                        is_active
                        and PipelineState(pipeline["state"]) == PipelineState.RUNNING_I2V
                    ):
                        stage = self._classify_old_i2v_owner(
                            connection,
                            prompt_row["job_id"],
                            int(prompt_row["scene_id"]),
                            1,
                            prompt_row["prompt_id"],
                        )
                    connection.execute(
                        """
                        UPDATE scenes
                        SET prompt_id = CASE WHEN ? IS NULL THEN NULL ELSE prompt_id END,
                            prompt_stage = ?
                        WHERE job_id = ? AND scene_id = ?
                        """,
                        (
                            stage,
                            stage,
                            prompt_row["job_id"],
                            prompt_row["scene_id"],
                        ),
                    )
            old_scene_owners = connection.execute(
                """
                SELECT job_id, scene_id, prompt_id
                FROM scenes WHERE prompt_stage = 'i2v'
                """
            ).fetchall()
            for owner in old_scene_owners:
                connection.execute(
                    """
                    UPDATE scenes SET prompt_stage = ?
                    WHERE job_id = ? AND scene_id = ? AND prompt_stage = 'i2v'
                    """,
                    (
                        self._classify_old_i2v_owner(
                            connection,
                            owner["job_id"],
                            int(owner["scene_id"]),
                            1,
                            owner["prompt_id"],
                        ),
                        owner["job_id"],
                        owner["scene_id"],
                    ),
                )
            remake_item_columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(remake_items)"
                ).fetchall()
            }
            if "prompt_id" not in remake_item_columns:
                connection.execute(
                    "ALTER TABLE remake_items ADD COLUMN prompt_id TEXT"
                )
            if "prompt_stage" not in remake_item_columns:
                connection.execute(
                    "ALTER TABLE remake_items ADD COLUMN prompt_stage TEXT"
                )
            old_remake_owners = connection.execute(
                """
                SELECT batch_id, position, job_id, scene_id, revision, prompt_id
                FROM remake_items WHERE prompt_stage = 'i2v'
                """
            ).fetchall()
            for owner in old_remake_owners:
                connection.execute(
                    """
                    UPDATE remake_items SET prompt_stage = ?
                    WHERE batch_id = ? AND position = ? AND prompt_stage = 'i2v'
                    """,
                    (
                        self._classify_old_i2v_owner(
                            connection,
                            owner["job_id"],
                            int(owner["scene_id"]),
                            int(owner["revision"]),
                            owner["prompt_id"],
                        ),
                        owner["batch_id"],
                        owner["position"],
                    ),
                )
            manual_final_columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(manual_final_requests)"
                ).fetchall()
            }
            if "selection_json" not in manual_final_columns:
                connection.execute(
                    "ALTER TABLE manual_final_requests "
                    "ADD COLUMN selection_json TEXT NOT NULL DEFAULT '[]'"
                )
            job_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(jobs)").fetchall()
            }
            for column, declaration in (
                ("status", "TEXT NOT NULL DEFAULT 'queued'"),
                ("final_path", "TEXT"),
                ("updated_at", "TEXT"),
            ):
                if column not in job_columns:
                    connection.execute(f"ALTER TABLE jobs ADD COLUMN {column} {declaration}")
            connection.execute(
                "UPDATE jobs SET updated_at = created_at WHERE updated_at IS NULL"
            )
            connection.execute(
                """
                UPDATE jobs
                SET status = CASE
                    WHEN NOT EXISTS (
                        SELECT 1 FROM scenes s
                        WHERE s.job_id = jobs.job_id AND s.state != ?
                    ) THEN ?
                    WHEN EXISTS (
                        SELECT 1 FROM scenes s
                        WHERE s.job_id = jobs.job_id AND s.state = ?
                    ) AND NOT EXISTS (
                        SELECT 1 FROM scenes s
                        WHERE s.job_id = jobs.job_id AND s.state IN (?, ?)
                    ) THEN ?
                    WHEN NOT EXISTS (
                        SELECT 1 FROM scenes s
                        WHERE s.job_id = jobs.job_id AND s.state IN (?, ?)
                    ) THEN ?
                    ELSE status
                END
                WHERE status = ?
                """,
                (
                    SceneState.SUCCEEDED,
                    JobState.SUCCEEDED,
                    SceneState.SUCCEEDED,
                    SceneState.PENDING,
                    SceneState.RUNNING,
                    JobState.PARTIAL,
                    SceneState.PENDING,
                    SceneState.RUNNING,
                    JobState.FAILED,
                    JobState.QUEUED,
                ),
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO pipeline_state (singleton, state, job_id, active_scene_id, error, updated_at)
                VALUES (1, ?, NULL, NULL, NULL, ?)
                """,
                (PipelineState.IDLE, _utc_now()),
            )

    def snapshot(self) -> PipelineSnapshot:
        self.initialize()
        with self._connection() as connection:
            row = connection.execute("SELECT state, job_id, active_scene_id, error FROM pipeline_state WHERE singleton = 1").fetchone()
        return PipelineSnapshot(
            state=PipelineState(row["state"]),
            job_id=row["job_id"],
            active_scene_id=row["active_scene_id"],
            error=row["error"],
        )

    @staticmethod
    def _assert_active_automatic_job_connection(
        connection: sqlite3.Connection,
        job_id: str,
        *,
        scene_id: int | None = None,
    ) -> None:
        """Reject late automatic work after cancellation or ownership transfer."""
        pipeline = connection.execute(
            "SELECT job_id FROM pipeline_state WHERE singleton = 1"
        ).fetchone()
        job = connection.execute(
            "SELECT status FROM jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        if job is None:
            raise StateTransitionError(f"Unknown job {job_id}.")
        if pipeline is None or pipeline["job_id"] != job_id:
            active_job = pipeline["job_id"] if pipeline is not None else None
            raise StateTransitionError(
                f"Job {job_id} no longer owns the automatic pipeline "
                f"(active job: {active_job or 'none'})."
            )
        if JobState(job["status"]) == JobState.CANCELLED:
            raise StateTransitionError(
                f"Cancelled job {job_id} cannot commit late automatic work."
            )
        if scene_id is None:
            return
        scene = connection.execute(
            "SELECT state FROM scenes WHERE job_id = ? AND scene_id = ?",
            (job_id, scene_id),
        ).fetchone()
        if scene is None:
            raise StateTransitionError(f"Unknown scene {scene_id} for job {job_id}.")
        if SceneState(scene["state"]) == SceneState.CANCELLED:
            raise StateTransitionError(
                f"Cancelled scene {scene_id} cannot commit late automatic work."
            )

    def transition(
        self,
        state: PipelineState,
        *,
        job_id: str | None = None,
        active_scene_id: int | None = None,
        error: str | None = None,
    ) -> None:
        self.initialize()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute("SELECT job_id FROM pipeline_state WHERE singleton = 1").fetchone()
            if job_id is not None:
                self._assert_active_automatic_job_connection(
                    connection,
                    job_id,
                    scene_id=active_scene_id,
                )
            resolved_job_id = job_id if job_id is not None else current["job_id"]
            connection.execute(
                """
                UPDATE pipeline_state
                SET state = ?, job_id = ?, active_scene_id = ?, error = ?, updated_at = ?
                WHERE singleton = 1
                """,
                (state, resolved_job_id, active_scene_id, error, _utc_now()),
            )

    def claim_message(self, message_key: str, job_id: str | None = None) -> bool:
        """Return True only once for a stable IMAP UID/message-id pair."""
        self.initialize()
        with self._connection() as connection:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO mail_messages (message_key, job_id, processed_at) VALUES (?, ?, ?)",
                (message_key, job_id, _utc_now()),
            )
            return cursor.rowcount == 1

    def claim_job(self, payload: JobPayload, *, review_required: bool = False) -> None:
        """Atomically accept one new job only while the pipeline is available to poll."""
        self.initialize()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._claim_job(connection, payload, review_required=review_required)

    def claim_inbound_job(
        self,
        message_key: str,
        payload: JobPayload,
        *,
        review_required: bool = False,
    ) -> InboundJobClaim:
        """Atomically accept a handoff, remapping only genuine reused-ID collisions."""
        self.initialize()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute("SELECT state FROM pipeline_state WHERE singleton = 1").fetchone()
            if PipelineState(current["state"]) not in {PipelineState.IDLE, PipelineState.WAITING_FOR_GROK}:
                return InboundJobClaim(False, None)
            message_record = connection.execute(
                "SELECT job_id FROM mail_messages WHERE message_key = ?", (message_key,)
            ).fetchone()
            if message_record is not None and message_record["job_id"] is not None:
                return InboundJobClaim(False, None)

            source_job_id = payload.job_id
            existing = connection.execute(
                "SELECT payload_json FROM jobs WHERE job_id = ?", (source_job_id,)
            ).fetchone()
            if existing is not None:
                accepted_payload = parse_job_payload(json.loads(existing["payload_json"]))
                if job_content_fingerprint(accepted_payload) == job_content_fingerprint(payload):
                    if message_record is None:
                        connection.execute(
                            "INSERT INTO mail_messages (message_key, job_id, processed_at) VALUES (?, ?, ?)",
                            (message_key, source_job_id, _utc_now()),
                        )
                    else:
                        connection.execute(
                            "UPDATE mail_messages SET job_id = ?, processed_at = ? WHERE message_key = ?",
                            (source_job_id, _utc_now(), message_key),
                        )
                    return InboundJobClaim(
                        False,
                        None,
                        duplicate_content=True,
                        source_job_id=source_job_id,
                    )
                payload = self._remap_colliding_job_id(connection, payload)
            if message_record is None:
                connection.execute(
                    "INSERT INTO mail_messages (message_key, job_id, processed_at) VALUES (?, ?, ?)",
                    (message_key, payload.job_id, _utc_now()),
                )
            else:
                # A message can become parseable after its Drive file is corrected
                # or a narrowly compatible parser repair is added. Upgrade the
                # earlier terminal record instead of permanently discarding the job.
                connection.execute(
                    "UPDATE mail_messages SET job_id = ?, processed_at = ? WHERE message_key = ?",
                    (payload.job_id, _utc_now(), message_key),
                )
            self._claim_job(connection, payload, review_required=review_required)
            return InboundJobClaim(True, payload, source_job_id=source_job_id)

    @staticmethod
    def _remap_colliding_job_id(connection: sqlite3.Connection, payload: JobPayload) -> JobPayload:
        """Allocate a project-local suffix while preserving the original Grok ID."""
        suffix = 2
        while True:
            suffix_text = f"-local-{suffix}"
            local_job_id = f"{payload.job_id[:128 - len(suffix_text)]}{suffix_text}"
            if connection.execute(
                "SELECT 1 FROM jobs WHERE job_id = ?", (local_job_id,)
            ).fetchone() is None:
                break
            suffix += 1
        raw = json.loads(json.dumps(payload.raw))
        raw["job_id"] = local_job_id
        raw["source_job_id"] = payload.job_id
        return parse_job_payload(raw)

    @staticmethod
    def _claim_job(
        connection: sqlite3.Connection,
        payload: JobPayload,
        *,
        review_required: bool = False,
    ) -> None:
        current = connection.execute("SELECT state, job_id FROM pipeline_state WHERE singleton = 1").fetchone()
        if PipelineState(current["state"]) not in {PipelineState.IDLE, PipelineState.WAITING_FOR_GROK}:
            raise StateTransitionError(f"Cannot accept {payload.job_id}; pipeline is {current['state']}.")
        existing = connection.execute("SELECT 1 FROM jobs WHERE job_id = ?", (payload.job_id,)).fetchone()
        if existing:
            raise StateTransitionError(f"Job {payload.job_id} has already been accepted.")

        now = _utc_now()
        job_state = JobState.AWAITING_REVIEW if review_required else JobState.QUEUED
        connection.execute(
            """
            INSERT INTO jobs (job_id, payload_json, created_at, status, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (payload.job_id, json.dumps(payload.raw, sort_keys=True), now, job_state, now),
        )
        connection.executemany(
            """
            INSERT INTO scenes (job_id, scene_id, state, frame_path, video_path, error, updated_at)
            VALUES (?, ?, ?, NULL, NULL, NULL, ?)
            """,
            [(payload.job_id, scene.scene_id, SceneState.PENDING, now) for scene in payload.scenes],
        )
        pipeline_state = (
            PipelineState.AWAITING_REVIEW
            if review_required
            else PipelineState.DOWNLOADING_ASSETS
        )
        connection.execute(
            """
            UPDATE pipeline_state
            SET state = ?, job_id = ?, active_scene_id = NULL, error = NULL, updated_at = ?
            WHERE singleton = 1
            """,
            (pipeline_state, payload.job_id, now),
        )

    def approve_job(self, job_id: str) -> None:
        """Release one reviewed inbound job to the normal render pipeline."""
        self.initialize()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT state, job_id FROM pipeline_state WHERE singleton = 1"
            ).fetchone()
            if (
                current["job_id"] != job_id
                or PipelineState(current["state"]) != PipelineState.AWAITING_REVIEW
            ):
                raise StateTransitionError(f"Job {job_id} is not awaiting review.")
            now = _utc_now()
            connection.execute(
                "UPDATE jobs SET status = ?, updated_at = ? WHERE job_id = ?",
                (JobState.QUEUED, now, job_id),
            )
            connection.execute(
                """
                UPDATE pipeline_state
                SET state = ?, active_scene_id = NULL, error = NULL, updated_at = ?
                WHERE singleton = 1
                """,
                (PipelineState.DOWNLOADING_ASSETS, now),
            )

    def set_scene_state(
        self,
        job_id: str,
        scene_id: int,
        state: SceneState,
        *,
        frame_path: str | None = None,
        video_path: str | None = None,
        error: str | None = None,
    ) -> None:
        self.initialize()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            exists = connection.execute(
                """
                SELECT s.state, j.status
                FROM scenes s
                JOIN jobs j ON j.job_id = s.job_id
                WHERE s.job_id = ? AND s.scene_id = ?
                """,
                (job_id, scene_id),
            ).fetchone()
            if not exists:
                raise StateTransitionError(f"Unknown scene {scene_id} for job {job_id}.")
            if (
                state != SceneState.CANCELLED
                and (
                    SceneState(exists["state"]) == SceneState.CANCELLED
                    or JobState(exists["status"]) == JobState.CANCELLED
                )
            ):
                raise StateTransitionError(
                    "A cancelled automatic scene cannot be resurrected by late work."
                )
            if state != SceneState.CANCELLED:
                self._assert_active_automatic_job_connection(
                    connection,
                    job_id,
                    scene_id=scene_id,
                )
            connection.execute(
                """
                UPDATE scenes
                SET state = ?,
                    frame_path = COALESCE(?, frame_path),
                    video_path = COALESCE(?, video_path),
                    error = ?,
                    updated_at = ?
                WHERE job_id = ? AND scene_id = ?
                """,
                (state, frame_path, video_path, error, _utc_now(), job_id, scene_id),
            )

    def begin_scene_stage(
        self,
        job_id: str,
        scene_id: int,
        pipeline_state: PipelineState,
        *,
        prompt_id: str | None = None,
        prompt_stage: str | None = None,
        resume: bool = False,
    ) -> int:
        if pipeline_state not in {PipelineState.RUNNING_T2I, PipelineState.RUNNING_I2V}:
            raise StateTransitionError("Scene stages can only begin in running_t2i or running_i2v.")
        attempt_column = "t2i_attempts" if pipeline_state == PipelineState.RUNNING_T2I else "i2v_attempts"
        self.initialize()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._assert_active_automatic_job_connection(
                connection,
                job_id,
                scene_id=scene_id,
            )
            row = connection.execute(
                f"""
                SELECT {attempt_column}, prompt_id, prompt_stage
                FROM scenes WHERE job_id = ? AND scene_id = ?
                """,
                (job_id, scene_id),
            ).fetchone()
            if row is None:
                raise StateTransitionError(f"Unknown scene {scene_id} for job {job_id}.")
            existing_attempts = int(row[attempt_column])
            if resume:
                if existing_attempts < 1:
                    raise StateTransitionError(
                        f"Scene {scene_id} has no {pipeline_state.value} attempt to resume."
                    )
                attempt = existing_attempts
            else:
                attempt = existing_attempts + 1
            if pipeline_state == PipelineState.RUNNING_T2I:
                stage = prompt_stage or "t2i"
                if stage != "t2i":
                    raise StateTransitionError("T2I scene ownership must use stage t2i.")
            else:
                stage = prompt_stage or "i2v_legacy"
                if stage not in {"i2v_legacy", "i2v_continuation"}:
                    raise StateTransitionError(
                        "I2V scene ownership must use i2v_legacy or "
                        "i2v_continuation."
                    )
            resolved_prompt_id = prompt_id
            if (
                resolved_prompt_id is None
                and resume
                and row["prompt_stage"] == stage
            ):
                resolved_prompt_id = row["prompt_id"]
            connection.execute(
                f"""
                UPDATE scenes
                SET state = ?, {attempt_column} = ?, prompt_id = ?,
                    prompt_stage = ?, error = NULL, updated_at = ?
                WHERE job_id = ? AND scene_id = ?
                """,
                (
                    SceneState.RUNNING,
                    attempt,
                    resolved_prompt_id,
                    stage,
                    _utc_now(),
                    job_id,
                    scene_id,
                ),
            )
            connection.execute(
                """
                UPDATE pipeline_state
                SET state = ?, job_id = ?, active_scene_id = ?, error = NULL, updated_at = ?
                WHERE singleton = 1
                """,
                (pipeline_state, job_id, scene_id, _utc_now()),
            )
        return attempt

    def set_scene_prompt_id(
        self,
        job_id: str,
        scene_id: int,
        prompt_id: str,
        *,
        stage: str,
    ) -> None:
        if stage not in {
            "t2i",
            "i2v_legacy",
            "i2v_continuation",
            "delivery",
        }:
            raise StateTransitionError(
                "Scene prompt stage must be t2i, i2v_legacy, "
                "i2v_continuation, or delivery."
            )
        self.initialize()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._assert_active_automatic_job_connection(
                connection,
                job_id,
                scene_id=scene_id,
            )
            cursor = connection.execute(
                """
                UPDATE scenes
                SET prompt_id = ?, prompt_stage = ?, updated_at = ?
                WHERE job_id = ? AND scene_id = ?
                """,
                (prompt_id, stage, _utc_now(), job_id, scene_id),
            )
            if cursor.rowcount != 1:
                raise StateTransitionError(f"Unknown scene {scene_id} for job {job_id}.")

    def clear_scene_prompt_id(self, job_id: str, scene_id: int) -> None:
        """Clear failed prompt ownership while the automatic job still owns the scene."""
        self.initialize()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._assert_active_automatic_job_connection(
                connection,
                job_id,
                scene_id=scene_id,
            )
            cursor = connection.execute(
                """
                UPDATE scenes
                SET prompt_id = NULL, updated_at = ?
                WHERE job_id = ? AND scene_id = ?
                """,
                (_utc_now(), job_id, scene_id),
            )
            if cursor.rowcount != 1:
                raise StateTransitionError(f"Unknown scene {scene_id} for job {job_id}.")

    def load_job(self, job_id: str) -> JobPayload:
        self.initialize()
        with self._connection() as connection:
            row = connection.execute(
                "SELECT payload_json FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        if row is None:
            raise StateTransitionError(f"Unknown job {job_id}.")
        from .contracts import parse_job_payload

        return parse_job_payload(json.loads(row["payload_json"]))

    def raw_job(self, job_id: str) -> Mapping[str, Any]:
        return self.load_job(job_id).raw

    def list_jobs(self) -> tuple[JobRecord, ...]:
        self.initialize()
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT j.job_id, j.status, j.created_at, j.updated_at, j.final_path,
                       COUNT(s.scene_id) AS scene_count,
                       SUM(CASE WHEN s.state = ? THEN 1 ELSE 0 END) AS succeeded_count,
                       SUM(CASE WHEN s.state = ? THEN 1 ELSE 0 END) AS failed_count
                FROM jobs j
                LEFT JOIN scenes s ON s.job_id = j.job_id
                GROUP BY j.job_id
                ORDER BY j.created_at DESC, j.job_id DESC
                """,
                (SceneState.SUCCEEDED, SceneState.FAILED),
            ).fetchall()
        return tuple(
            JobRecord(
                job_id=row["job_id"],
                status=JobState(row["status"]),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                final_path=row["final_path"],
                scene_count=int(row["scene_count"] or 0),
                succeeded_count=int(row["succeeded_count"] or 0),
                failed_count=int(row["failed_count"] or 0),
            )
            for row in rows
        )

    def set_job_status(
        self,
        job_id: str,
        status: JobState,
        *,
        final_path: str | None = None,
    ) -> None:
        self.initialize()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if status != JobState.CANCELLED:
                self._assert_active_automatic_job_connection(connection, job_id)
            cursor = connection.execute(
                """
                UPDATE jobs
                SET status = ?, final_path = COALESCE(?, final_path), updated_at = ?
                WHERE job_id = ?
                """,
                (status, final_path, _utc_now(), job_id),
            )
            if cursor.rowcount != 1:
                raise StateTransitionError(f"Unknown job {job_id}.")

    def create_scene_revision(
        self,
        job_id: str,
        scene_id: int,
        *,
        remake_mode: RemakeMode,
        parameters: Mapping[str, Any],
        state: SceneState = SceneState.PENDING,
        frame_path: str | None = None,
        video_path: str | None = None,
    ) -> int:
        self.initialize()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            exists = connection.execute(
                "SELECT 1 FROM scenes WHERE job_id = ? AND scene_id = ?",
                (job_id, scene_id),
            ).fetchone()
            if not exists:
                raise StateTransitionError(f"Unknown scene {scene_id} for job {job_id}.")
            row = connection.execute(
                """
                SELECT COALESCE(MAX(revision), 0) + 1 AS revision
                FROM scene_revisions WHERE job_id = ? AND scene_id = ?
                """,
                (job_id, scene_id),
            ).fetchone()
            revision = int(row["revision"])
            now = _utc_now()
            connection.execute(
                """
                INSERT INTO scene_revisions (
                    job_id, scene_id, revision, remake_mode, parameters_json, state,
                    frame_path, video_path, error, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)
                """,
                (
                    job_id,
                    scene_id,
                    revision,
                    remake_mode,
                    json.dumps(parameters, ensure_ascii=False, sort_keys=True),
                    state,
                    frame_path,
                    video_path,
                    now,
                    now,
                ),
            )
        return revision

    def ensure_original_scene_revision(
        self,
        job_id: str,
        scene_id: int,
        *,
        parameters: Mapping[str, Any],
        frame_path: str | None = None,
        video_path: str | None = None,
    ) -> None:
        self.initialize()
        with self._connection() as connection:
            exists = connection.execute(
                "SELECT state FROM scenes WHERE job_id = ? AND scene_id = ?",
                (job_id, scene_id),
            ).fetchone()
            if exists is None:
                raise StateTransitionError(f"Unknown scene {scene_id} for job {job_id}.")
            now = _utc_now()
            connection.execute(
                """
                INSERT OR IGNORE INTO scene_revisions (
                    job_id, scene_id, revision, remake_mode, parameters_json, state,
                    frame_path, video_path, error, created_at, updated_at
                ) VALUES (?, ?, 1, ?, ?, ?, ?, ?, NULL, ?, ?)
                """,
                (
                    job_id,
                    scene_id,
                    RemakeMode.IMAGE_AND_VIDEO,
                    json.dumps(parameters, ensure_ascii=False, sort_keys=True),
                    SceneState(exists["state"]),
                    frame_path,
                    video_path,
                    now,
                    now,
                ),
            )

    def update_scene_revision(
        self,
        job_id: str,
        scene_id: int,
        revision: int,
        *,
        state: SceneState,
        frame_path: str | None = None,
        video_path: str | None = None,
        error: str | None = None,
    ) -> None:
        self.initialize()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """
                SELECT state FROM scene_revisions
                WHERE job_id = ? AND scene_id = ? AND revision = ?
                """,
                (job_id, scene_id, revision),
            ).fetchone()
            if existing is None:
                raise StateTransitionError(
                    f"Unknown revision {revision} for scene {scene_id} of job {job_id}."
                )
            if revision == 1 and state != SceneState.CANCELLED:
                self._assert_active_automatic_job_connection(
                    connection,
                    job_id,
                    scene_id=scene_id,
                )
            if (
                SceneState(existing["state"]) == SceneState.CANCELLED
                and state != SceneState.CANCELLED
            ):
                raise StateTransitionError(
                    "A cancelled scene revision cannot be resurrected by late work."
                )
            cursor = connection.execute(
                """
                UPDATE scene_revisions
                SET state = ?,
                    frame_path = COALESCE(?, frame_path),
                    video_path = COALESCE(?, video_path),
                    error = ?,
                    updated_at = ?
                WHERE job_id = ? AND scene_id = ? AND revision = ?
                """,
                (
                    state,
                    frame_path,
                    video_path,
                    error,
                    _utc_now(),
                    job_id,
                    scene_id,
                    revision,
                ),
            )
            if cursor.rowcount != 1:  # pragma: no cover - guarded by the read above.
                raise StateTransitionError("Scene revision update was not applied.")

    @staticmethod
    def _continuation_work_is_active_connection(
        connection: sqlite3.Connection,
        job_id: str,
        scene_id: int,
        revision: int,
    ) -> bool:
        if revision == 1:
            row = connection.execute(
                """
                SELECT j.status AS job_status, s.state AS scene_state,
                       p.job_id AS active_job_id
                FROM jobs j
                JOIN scenes s ON s.job_id = j.job_id
                JOIN pipeline_state p ON p.singleton = 1
                WHERE j.job_id = ? AND s.scene_id = ?
                """,
                (job_id, scene_id),
            ).fetchone()
            return bool(
                row is not None
                and row["active_job_id"] == job_id
                and JobState(row["job_status"]) != JobState.CANCELLED
                and SceneState(row["scene_state"]) != SceneState.CANCELLED
            )
        row = connection.execute(
            """
            SELECT state FROM scene_revisions
            WHERE job_id = ? AND scene_id = ? AND revision = ?
            """,
            (job_id, scene_id, revision),
        ).fetchone()
        return bool(
            row is not None
            and SceneState(row["state"]) != SceneState.CANCELLED
        )

    def continuation_work_is_active(
        self,
        job_id: str,
        scene_id: int,
        revision: int,
    ) -> bool:
        """Check cancellation ownership for automatic work or a remake revision."""
        _positive_revision(revision)
        self.initialize()
        with self._connection() as connection:
            return self._continuation_work_is_active_connection(
                connection,
                job_id,
                scene_id,
                revision,
            )

    def scene_revisions(self, job_id: str, scene_id: int) -> tuple[SceneRevision, ...]:
        self.initialize()
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT job_id, scene_id, revision, remake_mode, parameters_json,
                       state, frame_path, video_path, error, created_at, updated_at
                FROM scene_revisions
                WHERE job_id = ? AND scene_id = ?
                ORDER BY revision DESC
                """,
                (job_id, scene_id),
            ).fetchall()
        return tuple(
            SceneRevision(
                job_id=row["job_id"],
                scene_id=int(row["scene_id"]),
                revision=int(row["revision"]),
                remake_mode=RemakeMode(row["remake_mode"]),
                parameters=json.loads(row["parameters_json"]),
                state=SceneState(row["state"]),
                frame_path=row["frame_path"],
                video_path=row["video_path"],
                error=row["error"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            for row in rows
        )

    @staticmethod
    def _qc_candidate_record(row: sqlite3.Row) -> QcCandidateRecord:
        return QcCandidateRecord(
            candidate_id=row["candidate_id"],
            job_id=row["job_id"],
            scene_id=int(row["scene_id"]),
            revision=int(row["revision"]),
            tier=QcTier(row["tier"]),
            parent_candidate_id=row["parent_candidate_id"],
            source_video_path=row["source_video_path"],
            source_video_sha256=row["source_video_sha256"],
            original_prompt=row["original_prompt"],
            current_prompt=row["current_prompt"],
            original_seed=int(row["original_seed"]),
            current_seed=int(row["current_seed"]),
            negative_prompt=row["negative_prompt"],
            negative_prompt_sha256=row["negative_prompt_sha256"],
            state=QcCandidateState(row["state"]),
            next_action=row["next_action"],
            infrastructure_failure_count=int(row["infrastructure_failure_count"]),
            last_failure=_json_load(row["last_failure_json"], None),
            generation_prompt_id=row["generation_prompt_id"],
            generation_prompt_stage=row["generation_prompt_stage"],
            generation_route=row["generation_route"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def ensure_qc_candidate(
        self,
        *,
        candidate_id: str,
        job_id: str,
        scene_id: int,
        revision: int,
        tier: QcTier,
        parent_candidate_id: str | None,
        source_video_path: str,
        source_video_sha256: str | None,
        original_prompt: str,
        current_prompt: str,
        original_seed: int,
        current_seed: int,
        negative_prompt: str,
        negative_prompt_sha256: str,
        state: QcCandidateState,
        next_action: str | None,
    ) -> QcCandidateRecord:
        candidate_id = _evidence_id(candidate_id, "candidate_id")
        _positive_revision(revision)
        source_video_path = _required_text(source_video_path, "source_video_path")
        if source_video_sha256 is None:
            if state not in {
                QcCandidateState.PENDING_GENERATION,
                QcCandidateState.GENERATING,
            }:
                raise StateTransitionError(
                    "Only a not-yet-generated candidate may omit its video hash."
                )
        else:
            source_video_sha256 = _sha256(
                source_video_sha256, "source_video_sha256"
            )
        original_prompt = _required_text(original_prompt, "original_prompt")
        current_prompt = _required_text(current_prompt, "current_prompt")
        original_seed = _uint64_seed(original_seed)
        current_seed = _uint64_seed(current_seed)
        if not isinstance(negative_prompt, str):
            raise StateTransitionError("negative_prompt must be text.")
        negative_prompt_sha256 = _sha256(
            negative_prompt_sha256, "negative_prompt_sha256"
        )
        if tier == QcTier.ORIGINAL:
            if parent_candidate_id is not None:
                raise StateTransitionError(
                    "The ORIGINAL candidate must bind a pre-QC baseline without a parent."
                )
        else:
            if revision <= 1 or parent_candidate_id is None:
                raise StateTransitionError(
                    "A repair candidate must bind a later revision and parent candidate."
                )
            parent_candidate_id = _evidence_id(
                parent_candidate_id, "parent_candidate_id"
            )
        immutable = (
            candidate_id,
            job_id,
            scene_id,
            revision,
            tier.value,
            parent_candidate_id,
            source_video_path,
            source_video_sha256,
            original_prompt,
            current_prompt,
            str(original_seed),
            str(current_seed),
            negative_prompt,
            negative_prompt_sha256,
        )
        self.initialize()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            revision_row = connection.execute(
                """
                SELECT video_path FROM scene_revisions
                WHERE job_id = ? AND scene_id = ? AND revision = ?
                """,
                (job_id, scene_id, revision),
            ).fetchone()
            if revision_row is None:
                raise StateTransitionError(
                    f"QC candidate references unknown scene revision {revision}."
                )
            revision_video_path = revision_row["video_path"]
            if (
                revision_video_path is not None
                and revision_video_path != source_video_path
            ) or (source_video_sha256 is not None and revision_video_path is None):
                raise StateTransitionError(
                    "QC candidate video path is not bound to its scene revision."
                )
            if parent_candidate_id is not None:
                parent = connection.execute(
                    """
                    SELECT job_id, scene_id FROM qc_candidates
                    WHERE candidate_id = ?
                    """,
                    (parent_candidate_id,),
                ).fetchone()
                if (
                    parent is None
                    or parent["job_id"] != job_id
                    or int(parent["scene_id"]) != scene_id
                ):
                    raise StateTransitionError(
                        "A QC parent candidate must belong to the same job and scene."
                    )
            existing = connection.execute(
                """
                SELECT * FROM qc_candidates
                WHERE candidate_id = ? OR (job_id = ? AND scene_id = ? AND tier = ?)
                """,
                (candidate_id, job_id, scene_id, tier.value),
            ).fetchone()
            if existing is not None:
                stored = (
                    existing["candidate_id"],
                    existing["job_id"],
                    int(existing["scene_id"]),
                    int(existing["revision"]),
                    existing["tier"],
                    existing["parent_candidate_id"],
                    existing["source_video_path"],
                    existing["source_video_sha256"],
                    existing["original_prompt"],
                    existing["current_prompt"],
                    existing["original_seed"],
                    existing["current_seed"],
                    existing["negative_prompt"],
                    existing["negative_prompt_sha256"],
                )
                if stored != immutable:
                    raise StateTransitionError(
                        "Existing QC candidate identity/evidence is immutable."
                    )
                return self._qc_candidate_record(existing)
            now = _utc_now()
            connection.execute(
                """
                INSERT INTO qc_candidates (
                    candidate_id, job_id, scene_id, revision, tier,
                    parent_candidate_id, source_video_path, source_video_sha256,
                    original_prompt, current_prompt, original_seed, current_seed,
                    negative_prompt, negative_prompt_sha256, state, next_action,
                    infrastructure_failure_count, last_failure_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, ?, ?)
                """,
                (*immutable, state.value, next_action, now, now),
            )
            row = connection.execute(
                "SELECT * FROM qc_candidates WHERE candidate_id = ?",
                (candidate_id,),
            ).fetchone()
        return self._qc_candidate_record(row)

    def ensure_original_qc_candidates(
        self,
        candidates: Sequence[Mapping[str, Any]],
    ) -> tuple[QcCandidateRecord, ...]:
        """Atomically snapshot every scene's exact pre-QC legacy selection."""
        if not candidates:
            raise StateTransitionError("A QC baseline snapshot cannot be empty.")
        prepared: list[tuple[Any, ...]] = []
        job_ids: set[str] = set()
        scene_ids: set[int] = set()
        for item in candidates:
            candidate_id = _evidence_id(item.get("candidate_id"), "candidate_id")
            job_id = _required_text(item.get("job_id"), "job_id")
            scene_id = int(item.get("scene_id"))
            revision = _positive_revision(int(item.get("revision")))
            source_video_path = _required_text(
                item.get("source_video_path"), "source_video_path"
            )
            source_video_sha256 = _sha256(
                item.get("source_video_sha256"), "source_video_sha256"
            )
            original_prompt = _required_text(
                item.get("original_prompt"), "original_prompt"
            )
            original_seed = _uint64_seed(item.get("original_seed"))
            negative_prompt = item.get("negative_prompt")
            if not isinstance(negative_prompt, str):
                raise StateTransitionError("negative_prompt must be text.")
            negative_prompt_sha256 = _sha256(
                item.get("negative_prompt_sha256"), "negative_prompt_sha256"
            )
            job_ids.add(job_id)
            if scene_id in scene_ids:
                raise StateTransitionError(
                    "A QC baseline snapshot must contain one candidate per scene."
                )
            scene_ids.add(scene_id)
            prepared.append(
                (
                    candidate_id,
                    job_id,
                    scene_id,
                    revision,
                    QcTier.ORIGINAL.value,
                    None,
                    source_video_path,
                    source_video_sha256,
                    original_prompt,
                    original_prompt,
                    str(original_seed),
                    str(original_seed),
                    negative_prompt,
                    negative_prompt_sha256,
                )
            )
        if len(job_ids) != 1:
            raise StateTransitionError(
                "A QC baseline snapshot must belong to exactly one job."
            )
        job_id = next(iter(job_ids))
        self.initialize()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """
                SELECT * FROM qc_candidates
                WHERE job_id = ? AND tier = ? ORDER BY scene_id
                """,
                (job_id, QcTier.ORIGINAL.value),
            ).fetchall()
            if existing:
                stored = [
                    (
                        row["candidate_id"], row["job_id"], int(row["scene_id"]),
                        int(row["revision"]), row["tier"], row["parent_candidate_id"],
                        row["source_video_path"], row["source_video_sha256"],
                        row["original_prompt"], row["current_prompt"],
                        row["original_seed"], row["current_seed"],
                        row["negative_prompt"], row["negative_prompt_sha256"],
                    )
                    for row in existing
                ]
                if stored != prepared:
                    raise StateTransitionError(
                        "The durable pre-QC baseline snapshot is immutable."
                    )
                return tuple(self._qc_candidate_record(row) for row in existing)

            active_manual = connection.execute(
                """
                SELECT selection_json FROM manual_final_requests
                WHERE job_id = ? AND state IN (?, ?)
                ORDER BY created_at DESC LIMIT 1
                """,
                (job_id, ManualFinalState.QUEUED.value, ManualFinalState.RUNNING.value),
            ).fetchone()
            manual_by_scene: dict[int, Mapping[str, Any]] | None = None
            if active_manual is not None:
                try:
                    manual_items = json.loads(active_manual["selection_json"])
                    manual_by_scene = {
                        int(item["scene_id"]): item for item in manual_items
                    }
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                    raise StateTransitionError(
                        "The active legacy manual-final snapshot is invalid."
                    ) from error
            for values in prepared:
                if manual_by_scene is not None:
                    selected = manual_by_scene.get(values[2])
                    matches_legacy = bool(
                        selected is not None
                        and int(selected["revision"]) == values[3]
                        and str(selected["video_path"]) == values[6]
                    )
                else:
                    latest = connection.execute(
                        """
                        SELECT revision, video_path FROM scene_revisions
                        WHERE job_id = ? AND scene_id = ? AND state = ?
                            AND video_path IS NOT NULL
                        ORDER BY revision DESC LIMIT 1
                        """,
                        (values[1], values[2], SceneState.SUCCEEDED.value),
                    ).fetchone()
                    matches_legacy = bool(
                        latest is not None
                        and int(latest["revision"]) == values[3]
                        and latest["video_path"] == values[6]
                    )
                revision_row = connection.execute(
                    """
                    SELECT state, video_path FROM scene_revisions
                    WHERE job_id = ? AND scene_id = ? AND revision = ?
                    """,
                    (values[1], values[2], values[3]),
                ).fetchone()
                if not matches_legacy or (
                    revision_row is None
                    or revision_row["state"] != SceneState.SUCCEEDED.value
                    or revision_row["video_path"] != values[6]
                ):
                    raise StateTransitionError(
                        "The pre-QC legacy selection changed before its baseline snapshot committed."
                    )
            now = _utc_now()
            connection.executemany(
                """
                INSERT INTO qc_candidates (
                    candidate_id, job_id, scene_id, revision, tier,
                    parent_candidate_id, source_video_path, source_video_sha256,
                    original_prompt, current_prompt, original_seed, current_seed,
                    negative_prompt, negative_prompt_sha256, state, next_action,
                    infrastructure_failure_count, last_failure_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, ?, ?)
                """,
                [
                    (*values, QcCandidateState.PENDING_QC.value, "evaluate", now, now)
                    for values in prepared
                ],
            )
            rows = connection.execute(
                """
                SELECT * FROM qc_candidates
                WHERE job_id = ? AND tier = ? ORDER BY scene_id
                """,
                (job_id, QcTier.ORIGINAL.value),
            ).fetchall()
        return tuple(self._qc_candidate_record(row) for row in rows)

    def qc_candidates(
        self, job_id: str, scene_id: int | None = None
    ) -> tuple[QcCandidateRecord, ...]:
        self.initialize()
        sql = "SELECT * FROM qc_candidates WHERE job_id = ?"
        parameters: list[object] = [job_id]
        if scene_id is not None:
            sql += " AND scene_id = ?"
            parameters.append(scene_id)
        sql += " ORDER BY scene_id, revision, created_at"
        with self._connection() as connection:
            rows = connection.execute(sql, parameters).fetchall()
        return tuple(self._qc_candidate_record(row) for row in rows)

    def qc_candidate(self, candidate_id: str) -> QcCandidateRecord:
        self.initialize()
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM qc_candidates WHERE candidate_id = ?", (candidate_id,)
            ).fetchone()
        if row is None:
            raise StateTransitionError(f"Unknown QC candidate {candidate_id}.")
        return self._qc_candidate_record(row)

    def ensure_a1_candidate_revision(
        self,
        *,
        candidate_id: str,
        parent_candidate_id: str,
        job_id: str,
        scene_id: int,
        expected_revision: int,
        parameters: Mapping[str, Any],
        frame_path: str,
        source_video_path: str,
        original_prompt: str,
        current_prompt: str,
        original_seed: int,
        current_seed: int,
        negative_prompt: str,
        negative_prompt_sha256: str,
    ) -> QcCandidateRecord:
        """Atomically allocate the one A1 VIDEO_ONLY revision and candidate."""
        candidate_id = _evidence_id(candidate_id, "candidate_id")
        parent_candidate_id = _evidence_id(
            parent_candidate_id, "parent_candidate_id"
        )
        expected_revision = _positive_revision(expected_revision)
        if expected_revision <= 1:
            raise StateTransitionError("A1 must use a later scene revision.")
        frame_path = _required_text(frame_path, "frame_path")
        source_video_path = _required_text(source_video_path, "source_video_path")
        parameters_json = _canonical_json(parameters)
        original_seed = _uint64_seed(original_seed)
        current_seed = _uint64_seed(current_seed)
        if current_seed == original_seed:
            raise StateTransitionError("A1 must change the seed.")
        if current_prompt != original_prompt:
            raise StateTransitionError("A1 must preserve the I2V prompt exactly.")
        negative_prompt_sha256 = _sha256(
            negative_prompt_sha256, "negative_prompt_sha256"
        )
        self.initialize()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """
                SELECT * FROM qc_candidates
                WHERE job_id = ? AND scene_id = ? AND tier = ?
                """,
                (job_id, scene_id, QcTier.A1.value),
            ).fetchone()
            if existing is not None:
                if (
                    existing["candidate_id"] != candidate_id
                    or existing["parent_candidate_id"] != parent_candidate_id
                    or int(existing["revision"]) != expected_revision
                    or existing["source_video_path"] != source_video_path
                    or int(existing["current_seed"]) != current_seed
                    or existing["current_prompt"] != current_prompt
                ):
                    raise StateTransitionError(
                        "The one A1 retry already exists with different immutable evidence."
                    )
                revision_row = connection.execute(
                    """
                    SELECT parameters_json FROM scene_revisions
                    WHERE job_id = ? AND scene_id = ? AND revision = ?
                    """,
                    (job_id, scene_id, expected_revision),
                ).fetchone()
                if (
                    revision_row is None
                    or _canonical_json(json.loads(revision_row["parameters_json"]))
                    != parameters_json
                ):
                    raise StateTransitionError("The persisted A1 revision is inconsistent.")
                return self._qc_candidate_record(existing)
            parent = connection.execute(
                """
                SELECT * FROM qc_candidates WHERE candidate_id = ?
                """,
                (parent_candidate_id,),
            ).fetchone()
            if (
                parent is None
                or parent["job_id"] != job_id
                or int(parent["scene_id"]) != scene_id
                or QcTier(parent["tier"]) != QcTier.ORIGINAL
            ):
                raise StateTransitionError(
                    "A1 must descend from the ORIGINAL candidate for this scene."
                )
            next_row = connection.execute(
                """
                SELECT COALESCE(MAX(revision), 0) + 1 AS revision
                FROM scene_revisions WHERE job_id = ? AND scene_id = ?
                """,
                (job_id, scene_id),
            ).fetchone()
            if int(next_row["revision"]) != expected_revision:
                raise StateTransitionError(
                    "The expected A1 revision is stale; recompute from durable state."
                )
            used_seed = connection.execute(
                """
                SELECT 1 FROM qc_candidates
                WHERE job_id = ? AND scene_id = ?
                    AND (original_seed = ? OR current_seed = ?)
                """,
                (job_id, scene_id, str(current_seed), str(current_seed)),
            ).fetchone()
            if used_seed is not None:
                raise StateTransitionError("A1 seed was already used in this scene lineage.")
            revision_seed_rows = connection.execute(
                """
                SELECT parameters_json FROM scene_revisions
                WHERE job_id = ? AND scene_id = ?
                """,
                (job_id, scene_id),
            ).fetchall()
            for revision_seed_row in revision_seed_rows:
                revision_parameters = json.loads(revision_seed_row["parameters_json"])
                revision_i2v = revision_parameters.get("i2v")
                if not isinstance(revision_i2v, Mapping) or "seed" not in revision_i2v:
                    continue
                revision_seed = revision_i2v["seed"]
                if isinstance(revision_seed, str) and revision_seed.strip().isdigit():
                    revision_seed = int(revision_seed.strip())
                if revision_seed == current_seed:
                    raise StateTransitionError(
                        "A1 seed was already used in this scene revision lineage."
                    )
            now = _utc_now()
            connection.execute(
                """
                INSERT INTO scene_revisions (
                    job_id, scene_id, revision, remake_mode, parameters_json,
                    state, frame_path, video_path, error, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?)
                """,
                (
                    job_id,
                    scene_id,
                    expected_revision,
                    RemakeMode.VIDEO_ONLY.value,
                    parameters_json,
                    SceneState.PENDING.value,
                    frame_path,
                    now,
                    now,
                ),
            )
            connection.execute(
                """
                INSERT INTO qc_candidates (
                    candidate_id, job_id, scene_id, revision, tier,
                    parent_candidate_id, source_video_path, source_video_sha256,
                    original_prompt, current_prompt, original_seed, current_seed,
                    negative_prompt, negative_prompt_sha256, state, next_action,
                    infrastructure_failure_count, last_failure_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, ?, ?)
                """,
                (
                    candidate_id,
                    job_id,
                    scene_id,
                    expected_revision,
                    QcTier.A1.value,
                    parent_candidate_id,
                    source_video_path,
                    original_prompt,
                    current_prompt,
                    str(original_seed),
                    str(current_seed),
                    negative_prompt,
                    negative_prompt_sha256,
                    QcCandidateState.PENDING_GENERATION.value,
                    "render_a1",
                    now,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM qc_candidates WHERE candidate_id = ?", (candidate_id,)
            ).fetchone()
        return self._qc_candidate_record(row)

    def ensure_b1_candidate_revision(
        self,
        *,
        candidate_id: str,
        parent_candidate_id: str,
        job_id: str,
        scene_id: int,
        expected_revision: int,
        parameters: Mapping[str, Any],
        frame_path: str,
        source_video_path: str,
        original_prompt: str,
        current_prompt: str,
        original_seed: int,
        current_seed: int,
        negative_prompt: str,
        negative_prompt_sha256: str,
    ) -> QcCandidateRecord:
        """Atomically allocate the one B1 VIDEO_ONLY revision and candidate."""
        candidate_id = _evidence_id(candidate_id, "candidate_id")
        parent_candidate_id = _evidence_id(parent_candidate_id, "parent_candidate_id")
        expected_revision = _positive_revision(expected_revision)
        frame_path = _required_text(frame_path, "frame_path")
        source_video_path = _required_text(source_video_path, "source_video_path")
        parameters_json = _canonical_json(parameters)
        original_seed = _uint64_seed(original_seed)
        current_seed = _uint64_seed(current_seed)
        if current_prompt == original_prompt:
            raise StateTransitionError("B1 must change the I2V prompt.")
        if current_seed == original_seed:
            raise StateTransitionError("B1 must use a distinct controller-derived seed.")
        negative_prompt_sha256 = _sha256(
            negative_prompt_sha256, "negative_prompt_sha256"
        )
        self.initialize()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM qc_candidates WHERE job_id = ? AND scene_id = ? AND tier = ?",
                (job_id, scene_id, QcTier.B1.value),
            ).fetchone()
            if existing is not None:
                if (
                    existing["candidate_id"] != candidate_id
                    or existing["parent_candidate_id"] != parent_candidate_id
                    or int(existing["revision"]) != expected_revision
                    or existing["source_video_path"] != source_video_path
                    or int(existing["current_seed"]) != current_seed
                    or existing["current_prompt"] != current_prompt
                ):
                    raise StateTransitionError(
                        "The one B1 retry already exists with different immutable evidence."
                    )
                revision_row = connection.execute(
                    "SELECT parameters_json FROM scene_revisions WHERE job_id = ? AND scene_id = ? AND revision = ?",
                    (job_id, scene_id, expected_revision),
                ).fetchone()
                if revision_row is None or _canonical_json(
                    json.loads(revision_row["parameters_json"])
                ) != parameters_json:
                    raise StateTransitionError("The persisted B1 revision is inconsistent.")
                return self._qc_candidate_record(existing)
            parent = connection.execute(
                "SELECT * FROM qc_candidates WHERE candidate_id = ?",
                (parent_candidate_id,),
            ).fetchone()
            if (
                parent is None
                or parent["job_id"] != job_id
                or int(parent["scene_id"]) != scene_id
                or QcTier(parent["tier"]) != QcTier.A1
            ):
                raise StateTransitionError("B1 must descend from this scene's A1 candidate.")
            next_row = connection.execute(
                "SELECT COALESCE(MAX(revision), 0) + 1 AS revision FROM scene_revisions WHERE job_id = ? AND scene_id = ?",
                (job_id, scene_id),
            ).fetchone()
            if int(next_row["revision"]) != expected_revision:
                raise StateTransitionError(
                    "The expected B1 revision is stale; recompute from durable state."
                )
            for row in connection.execute(
                "SELECT original_seed, current_seed FROM qc_candidates WHERE job_id = ? AND scene_id = ?",
                (job_id, scene_id),
            ).fetchall():
                if current_seed in {int(row["original_seed"]), int(row["current_seed"])}:
                    raise StateTransitionError("B1 seed was already used in this scene lineage.")
            now = _utc_now()
            connection.execute(
                """
                INSERT INTO scene_revisions (
                    job_id, scene_id, revision, remake_mode, parameters_json,
                    state, frame_path, video_path, error, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?)
                """,
                (
                    job_id, scene_id, expected_revision, RemakeMode.VIDEO_ONLY.value,
                    parameters_json, SceneState.PENDING.value, frame_path, now, now,
                ),
            )
            connection.execute(
                """
                INSERT INTO qc_candidates (
                    candidate_id, job_id, scene_id, revision, tier,
                    parent_candidate_id, source_video_path, source_video_sha256,
                    original_prompt, current_prompt, original_seed, current_seed,
                    negative_prompt, negative_prompt_sha256, state, next_action,
                    infrastructure_failure_count, last_failure_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, ?, ?)
                """,
                (
                    candidate_id, job_id, scene_id, expected_revision, QcTier.B1.value,
                    parent_candidate_id, source_video_path, original_prompt,
                    current_prompt, str(original_seed), str(current_seed), negative_prompt,
                    negative_prompt_sha256, QcCandidateState.PENDING_GENERATION.value,
                    "render_b1", now, now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM qc_candidates WHERE candidate_id = ?", (candidate_id,)
            ).fetchone()
        return self._qc_candidate_record(row)

    def complete_qc_candidate_generation(
        self,
        candidate_id: str,
        *,
        source_video_path: str,
        source_video_sha256: str,
    ) -> QcCandidateRecord:
        source_video_sha256 = _sha256(
            source_video_sha256, "source_video_sha256"
        )
        artifact = Path(source_video_path)
        if not artifact.is_file():
            raise StateTransitionError("Generated QC candidate video does not exist.")
        digest = hashlib.sha256()
        with artifact.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != source_video_sha256:
            raise StateTransitionError(
                "Generated QC candidate hash does not match the persisted artifact."
            )
        self.initialize()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM qc_candidates WHERE candidate_id = ?", (candidate_id,)
            ).fetchone()
            if existing is None:
                raise StateTransitionError(f"Unknown QC candidate {candidate_id}.")
            if existing["source_video_path"] != source_video_path:
                raise StateTransitionError("Candidate generation path is immutable.")
            if existing["source_video_sha256"] is not None:
                if existing["source_video_sha256"] != source_video_sha256:
                    raise StateTransitionError("Candidate video evidence is immutable.")
                return self._qc_candidate_record(existing)
            if QcCandidateState(existing["state"]) not in {
                QcCandidateState.PENDING_GENERATION,
                QcCandidateState.GENERATING,
            }:
                raise StateTransitionError(
                    "Only a generating candidate may record its video hash."
                )
            now = _utc_now()
            connection.execute(
                """
                UPDATE scene_revisions SET state = ?, video_path = ?, error = NULL,
                    updated_at = ?
                WHERE job_id = ? AND scene_id = ? AND revision = ?
                """,
                (
                    SceneState.SUCCEEDED.value,
                    source_video_path,
                    now,
                    existing["job_id"],
                    existing["scene_id"],
                    existing["revision"],
                ),
            )
            connection.execute(
                """
                UPDATE qc_candidates SET source_video_sha256 = ?, state = ?,
                    next_action = ?, updated_at = ? WHERE candidate_id = ?
                """,
                (
                    source_video_sha256,
                    QcCandidateState.PENDING_QC.value,
                    "evaluate",
                    now,
                    candidate_id,
                ),
            )
            row = connection.execute(
                "SELECT * FROM qc_candidates WHERE candidate_id = ?", (candidate_id,)
            ).fetchone()
        return self._qc_candidate_record(row)

    def set_qc_candidate_state(
        self,
        candidate_id: str,
        state: QcCandidateState,
        *,
        next_action: str | None,
    ) -> QcCandidateRecord:
        self.initialize()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE qc_candidates SET state = ?, next_action = ?, updated_at = ?
                WHERE candidate_id = ?
                """,
                (state.value, next_action, _utc_now(), candidate_id),
            )
            if cursor.rowcount != 1:
                raise StateTransitionError(f"Unknown QC candidate {candidate_id}.")
            row = connection.execute(
                "SELECT * FROM qc_candidates WHERE candidate_id = ?", (candidate_id,)
            ).fetchone()
        return self._qc_candidate_record(row)

    def set_qc_generation_owner(
        self,
        candidate_id: str,
        *,
        prompt_id: str | None,
        prompt_stage: str | None,
        route: str,
    ) -> QcCandidateRecord:
        if route not in {"legacy", "continuation"}:
            raise StateTransitionError("QC generation route must be legacy or continuation.")
        if prompt_id is not None:
            prompt_id = _required_text(prompt_id, "prompt_id")
        if prompt_stage is not None:
            prompt_stage = _required_text(prompt_stage, "prompt_stage")
        self.initialize()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM qc_candidates WHERE candidate_id = ?", (candidate_id,)
            ).fetchone()
            if row is None:
                raise StateTransitionError(f"Unknown QC candidate {candidate_id}.")
            if row["generation_route"] not in {None, route}:
                raise StateTransitionError("QC generation route is immutable after launch.")
            connection.execute(
                """
                UPDATE qc_candidates SET generation_prompt_id = ?,
                    generation_prompt_stage = ?, generation_route = ?, state = ?,
                    updated_at = ? WHERE candidate_id = ?
                """,
                (
                    prompt_id, prompt_stage, route,
                    QcCandidateState.GENERATING.value, _utc_now(), candidate_id,
                ),
            )
            updated = connection.execute(
                "SELECT * FROM qc_candidates WHERE candidate_id = ?", (candidate_id,)
            ).fetchone()
        return self._qc_candidate_record(updated)

    def record_qc_infrastructure_failure(
        self,
        candidate_id: str,
        failure: Mapping[str, Any],
        *,
        maximum_failures: int = 2,
    ) -> QcCandidateRecord:
        if maximum_failures != 2:
            raise StateTransitionError("Phase 1 permits exactly two infrastructure failures.")
        failure_json = _canonical_json(failure)
        self.initialize()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT infrastructure_failure_count FROM qc_candidates WHERE candidate_id = ?",
                (candidate_id,),
            ).fetchone()
            if row is None:
                raise StateTransitionError(f"Unknown QC candidate {candidate_id}.")
            prior_count = int(row["infrastructure_failure_count"])
            if prior_count >= maximum_failures:
                existing = connection.execute(
                    "SELECT * FROM qc_candidates WHERE candidate_id = ?",
                    (candidate_id,),
                ).fetchone()
                return self._qc_candidate_record(existing)
            count = prior_count + 1
            state = (
                QcCandidateState.HOLD_FOR_REVIEW
                if count >= maximum_failures
                else QcCandidateState.PENDING_QC
            )
            next_action = "hold_for_review" if count >= maximum_failures else "retry_evaluation"
            connection.execute(
                """
                UPDATE qc_candidates
                SET infrastructure_failure_count = ?, last_failure_json = ?,
                    state = ?, next_action = ?, updated_at = ?
                WHERE candidate_id = ?
                """,
                (count, failure_json, state.value, next_action, _utc_now(), candidate_id),
            )
            updated = connection.execute(
                "SELECT * FROM qc_candidates WHERE candidate_id = ?", (candidate_id,)
            ).fetchone()
        return self._qc_candidate_record(updated)

    def record_qc_generation_failure(
        self,
        candidate_id: str,
        failure: Mapping[str, Any],
        *,
        retryable: bool,
        maximum_failures: int = 2,
    ) -> QcCandidateRecord:
        """Charge the durable repair-render budget and fail closed when exhausted."""
        if maximum_failures != 2:
            raise StateTransitionError(
                "Phase 1 permits one repair-generation retry (two attempts total)."
            )
        failure_document = dict(failure)
        failure_document["phase"] = "repair_generation"
        failure_document["retryable"] = bool(retryable)
        self.initialize()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM qc_candidates WHERE candidate_id = ?",
                (candidate_id,),
            ).fetchone()
            if row is None:
                raise StateTransitionError(f"Unknown QC candidate {candidate_id}.")
            current_state = QcCandidateState(row["state"])
            if current_state == QcCandidateState.HOLD_FOR_REVIEW:
                return self._qc_candidate_record(row)
            if current_state not in {
                QcCandidateState.PENDING_GENERATION,
                QcCandidateState.GENERATING,
            }:
                raise StateTransitionError(
                    "Only an unfinished repair candidate may record a generation failure."
                )
            prior_count = int(row["infrastructure_failure_count"])
            count = min(prior_count + 1, maximum_failures)
            should_hold = not retryable or count >= maximum_failures
            state = (
                QcCandidateState.HOLD_FOR_REVIEW
                if should_hold
                else QcCandidateState.PENDING_GENERATION
            )
            next_action = (
                "hold_for_review" if should_hold else "retry_repair_generation"
            )
            failure_document["attempt_count"] = count
            failure_json = _canonical_json(failure_document)
            now = _utc_now()
            connection.execute(
                """
                UPDATE qc_candidates
                SET infrastructure_failure_count = ?, last_failure_json = ?,
                    state = ?, next_action = ?, updated_at = ?
                WHERE candidate_id = ?
                """,
                (count, failure_json, state.value, next_action, now, candidate_id),
            )
            connection.execute(
                """
                UPDATE scene_revisions
                SET state = ?, error = ?, updated_at = ?
                WHERE job_id = ? AND scene_id = ? AND revision = ?
                """,
                (
                    SceneState.FAILED.value,
                    failure_document.get("message") or failure_document.get("reason"),
                    now,
                    row["job_id"],
                    row["scene_id"],
                    row["revision"],
                ),
            )
            updated = connection.execute(
                "SELECT * FROM qc_candidates WHERE candidate_id = ?",
                (candidate_id,),
            ).fetchone()
        return self._qc_candidate_record(updated)

    @staticmethod
    def _qc_evaluation_record(row: sqlite3.Row) -> QcEvaluationRecord:
        identity_keys = (
            "evaluator_id",
            "evaluator_version",
            "backend_family",
            "backend_version",
            "executable_path",
            "executable_sha256",
            "model_id",
            "model_path",
            "model_sha256",
            "quantization",
            "projector_path",
            "projector_sha256",
            "projector_precision",
            "gpu_uuid",
            "gpu_name",
        )
        return QcEvaluationRecord(
            evaluation_id=row["evaluation_id"],
            idempotency_key=row["idempotency_key"],
            candidate_id=row["candidate_id"],
            source_video_path=row["source_video_path"],
            source_video_sha256=row["source_video_sha256"],
            evaluator_identity={key: row[key] for key in identity_keys},
            effective_config=_json_load(row["effective_config_json"], {}),
            effective_config_sha256=row["effective_config_sha256"],
            prompt_version=row["prompt_version"],
            prompt_sha256=row["prompt_sha256"],
            sampling_config=_json_load(row["sampling_config_json"], {}),
            window_config=_json_load(row["window_config_json"], {}),
            raw_result=row["raw_result"],
            normalized_decision=(
                QcDecision(row["normalized_decision"])
                if row["normalized_decision"] is not None
                else None
            ),
            suspect_windows=_json_load(row["suspect_windows_json"], []),
            strong_window_count=(
                int(row["strong_window_count"])
                if row["strong_window_count"] is not None
                else None
            ),
            frame_accounting=_json_load(row["frame_accounting_json"], {}),
            evidence_manifest_path=row["evidence_manifest_path"],
            evidence_manifest_sha256=row["evidence_manifest_sha256"],
            next_action=row["next_action"],
            state=row["state"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
        )

    def begin_qc_evaluation(
        self,
        *,
        evaluation_id: str,
        idempotency_key: str,
        candidate_id: str,
        source_video_path: str,
        source_video_sha256: str,
        evaluator_identity: Mapping[str, Any],
        effective_config: Mapping[str, Any],
        effective_config_sha256: str,
        prompt_version: str,
        prompt_sha256: str,
        sampling_config: Mapping[str, Any],
        window_config: Mapping[str, Any],
    ) -> QcEvaluationRecord:
        evaluation_id = _evidence_id(evaluation_id, "evaluation_id")
        candidate_id = _evidence_id(candidate_id, "candidate_id")
        idempotency_key = _sha256(idempotency_key, "idempotency_key")
        source_video_sha256 = _sha256(source_video_sha256, "source_video_sha256")
        effective_config_sha256 = _sha256(
            effective_config_sha256, "effective_config_sha256"
        )
        prompt_sha256 = _sha256(prompt_sha256, "prompt_sha256")
        identity_keys = (
            "evaluator_id",
            "evaluator_version",
            "backend_family",
            "backend_version",
            "executable_path",
            "executable_sha256",
            "model_id",
            "model_path",
            "model_sha256",
            "quantization",
            "projector_path",
            "projector_sha256",
            "projector_precision",
            "gpu_uuid",
            "gpu_name",
        )
        missing = [key for key in identity_keys if key not in evaluator_identity]
        if missing:
            raise StateTransitionError(
                f"Evaluator identity is missing: {', '.join(missing)}."
            )
        identity_values: list[str] = []
        for key in identity_keys:
            value = _required_text(evaluator_identity[key], f"evaluator_identity.{key}")
            if key.endswith("sha256"):
                value = _sha256(value, f"evaluator_identity.{key}")
            identity_values.append(value)
        expected_idempotency_key = evaluation_idempotency_key(
            source_video_sha256=source_video_sha256,
            evaluator_id=identity_values[0],
            evaluator_version=identity_values[1],
            backend_version=identity_values[3],
            executable_sha256=identity_values[5],
            model_sha256=identity_values[8],
            projector_sha256=identity_values[11],
            effective_config_sha256=effective_config_sha256,
            prompt_sha256=prompt_sha256,
        )
        if idempotency_key != expected_idempotency_key:
            raise StateTransitionError(
                "Evaluation idempotency key does not match immutable evidence identities."
            )
        immutable = (
            evaluation_id,
            idempotency_key,
            candidate_id,
            _required_text(source_video_path, "source_video_path"),
            source_video_sha256,
            *identity_values,
            _canonical_json(effective_config),
            effective_config_sha256,
            _required_text(prompt_version, "prompt_version"),
            prompt_sha256,
            _canonical_json(sampling_config),
            _canonical_json(window_config),
        )
        self.initialize()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            candidate = connection.execute(
                "SELECT source_video_path, source_video_sha256 FROM qc_candidates WHERE candidate_id = ?",
                (candidate_id,),
            ).fetchone()
            if candidate is None:
                raise StateTransitionError(f"Unknown QC candidate {candidate_id}.")
            if (
                candidate["source_video_path"] != source_video_path
                or candidate["source_video_sha256"] != source_video_sha256
            ):
                raise StateTransitionError(
                    "Evaluation source does not match immutable candidate video evidence."
                )
            existing = connection.execute(
                """
                SELECT * FROM qc_evaluations
                WHERE evaluation_id = ? OR idempotency_key = ?
                """,
                (evaluation_id, idempotency_key),
            ).fetchone()
            if existing is not None:
                stored = tuple(existing[key] for key in (
                    "evaluation_id", "idempotency_key", "candidate_id",
                    "source_video_path", "source_video_sha256", *identity_keys,
                    "effective_config_json", "effective_config_sha256",
                    "prompt_version", "prompt_sha256", "sampling_config_json",
                    "window_config_json",
                ))
                if stored != immutable:
                    raise StateTransitionError(
                        "Existing evaluation idempotency evidence is immutable."
                    )
                return self._qc_evaluation_record(existing)
            now = _utc_now()
            placeholders = ", ".join("?" for _ in range(len(immutable) + 2))
            connection.execute(
                f"""
                INSERT INTO qc_evaluations (
                    evaluation_id, idempotency_key, candidate_id,
                    source_video_path, source_video_sha256,
                    {', '.join(identity_keys)}, effective_config_json,
                    effective_config_sha256, prompt_version, prompt_sha256,
                    sampling_config_json, window_config_json, state, started_at
                ) VALUES ({placeholders})
                """,
                (*immutable, "RUNNING", now),
            )
            row = connection.execute(
                "SELECT * FROM qc_evaluations WHERE evaluation_id = ?",
                (evaluation_id,),
            ).fetchone()
        return self._qc_evaluation_record(row)

    def complete_qc_evaluation(
        self,
        evaluation_id: str,
        *,
        raw_result: str,
        normalized_decision: QcDecision,
        suspect_windows: Sequence[Mapping[str, Any]],
        strong_window_count: int,
        frame_accounting: Mapping[str, Any],
        evidence_manifest_path: str,
        evidence_manifest_sha256: str,
        next_action: str,
    ) -> QcEvaluationRecord:
        if isinstance(strong_window_count, bool) or strong_window_count < 0:
            raise StateTransitionError("strong_window_count must be non-negative.")
        self.initialize()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM qc_evaluations WHERE evaluation_id = ?",
                (evaluation_id,),
            ).fetchone()
            if existing is None:
                raise StateTransitionError(f"Unknown QC evaluation {evaluation_id}.")
            manifest_path = _require_path_below(
                evidence_manifest_path,
                Path(existing["source_video_path"]).parent
                / "qc"
                / "evaluations"
                / evaluation_id,
                "evidence_manifest_path",
            )
            completion = (
                raw_result,
                normalized_decision.value,
                _canonical_json(list(suspect_windows)),
                strong_window_count,
                _canonical_json(frame_accounting),
                manifest_path,
                _sha256(evidence_manifest_sha256, "evidence_manifest_sha256"),
                _required_text(next_action, "next_action"),
            )
            if existing["state"] == "COMPLETE":
                stored = tuple(existing[key] for key in (
                    "raw_result", "normalized_decision", "suspect_windows_json",
                    "strong_window_count", "frame_accounting_json",
                    "evidence_manifest_path", "evidence_manifest_sha256", "next_action",
                ))
                if stored != completion:
                    raise StateTransitionError(
                        "Completed QC evaluation evidence is immutable."
                    )
                return self._qc_evaluation_record(existing)
            completed_at = _utc_now()
            connection.execute(
                """
                UPDATE qc_evaluations
                SET raw_result = ?, normalized_decision = ?, suspect_windows_json = ?,
                    strong_window_count = ?, frame_accounting_json = ?,
                    evidence_manifest_path = ?, evidence_manifest_sha256 = ?,
                    next_action = ?, state = 'COMPLETE', completed_at = ?
                WHERE evaluation_id = ? AND state = 'RUNNING'
                """,
                (*completion, completed_at, evaluation_id),
            )
            row = connection.execute(
                "SELECT * FROM qc_evaluations WHERE evaluation_id = ?",
                (evaluation_id,),
            ).fetchone()
        return self._qc_evaluation_record(row)

    def qc_evaluations(self, candidate_id: str) -> tuple[QcEvaluationRecord, ...]:
        self.initialize()
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM qc_evaluations
                WHERE candidate_id = ? ORDER BY started_at, evaluation_id
                """,
                (candidate_id,),
            ).fetchall()
        return tuple(self._qc_evaluation_record(row) for row in rows)

    @staticmethod
    def _qc_repair_record(row: sqlite3.Row) -> QcRepairRecord:
        return QcRepairRecord(
            repair_id=row["repair_id"],
            candidate_id=row["candidate_id"],
            evaluation_id=row["evaluation_id"],
            planner_identity=_json_load(row["planner_identity_json"], {}),
            repair_input_sha256=row["repair_input_sha256"],
            raw_output=row["raw_output"],
            proposed_patch=_json_load(row["proposed_patch_json"], {}),
            status=row["status"],
            reason=row["reason"],
            prior_repair_summaries=_json_load(row["prior_repair_summaries_json"], []),
            evidence_manifest_path=row["evidence_manifest_path"],
            evidence_manifest_sha256=row["evidence_manifest_sha256"],
            created_at=row["created_at"],
        )

    def record_qc_repair(
        self,
        *,
        repair_id: str,
        candidate_id: str,
        evaluation_id: str | None,
        planner_identity: Mapping[str, Any],
        repair_input_sha256: str,
        raw_output: str,
        proposed_patch: Mapping[str, Any],
        status: str,
        reason: str | None,
        prior_repair_summaries: Sequence[Any],
        evidence_manifest_path: str,
        evidence_manifest_sha256: str,
    ) -> QcRepairRecord:
        repair_id = _evidence_id(repair_id, "repair_id")
        candidate_id = _evidence_id(candidate_id, "candidate_id")
        if evaluation_id is not None:
            evaluation_id = _evidence_id(evaluation_id, "evaluation_id")
        self.initialize()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            candidate = connection.execute(
                "SELECT source_video_path FROM qc_candidates WHERE candidate_id = ?",
                (candidate_id,),
            ).fetchone()
            if candidate is None:
                raise StateTransitionError(f"Unknown QC candidate {candidate_id}.")
            manifest_path = _require_path_below(
                evidence_manifest_path,
                Path(candidate["source_video_path"]).parent
                / "qc"
                / "repairs"
                / repair_id,
                "evidence_manifest_path",
            )
            values = (
                repair_id,
                candidate_id,
                evaluation_id,
                _canonical_json(planner_identity),
                _sha256(repair_input_sha256, "repair_input_sha256"),
                raw_output,
                _canonical_json(proposed_patch),
                _required_text(status, "status"),
                reason,
                _canonical_json(list(prior_repair_summaries)),
                manifest_path,
                _sha256(evidence_manifest_sha256, "evidence_manifest_sha256"),
            )
            existing = connection.execute(
                "SELECT * FROM qc_repairs WHERE repair_id = ?", (repair_id,)
            ).fetchone()
            if existing is not None:
                stored = tuple(existing[key] for key in (
                    "repair_id", "candidate_id", "evaluation_id",
                    "planner_identity_json", "repair_input_sha256", "raw_output",
                    "proposed_patch_json", "status", "reason",
                    "prior_repair_summaries_json", "evidence_manifest_path",
                    "evidence_manifest_sha256",
                ))
                if stored != values:
                    raise StateTransitionError("Recorded QC repair evidence is immutable.")
                return self._qc_repair_record(existing)
            connection.execute(
                """
                INSERT INTO qc_repairs (
                    repair_id, candidate_id, evaluation_id, planner_identity_json,
                    repair_input_sha256, raw_output, proposed_patch_json, status,
                    reason, prior_repair_summaries_json, evidence_manifest_path,
                    evidence_manifest_sha256, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (*values, _utc_now()),
            )
            row = connection.execute(
                "SELECT * FROM qc_repairs WHERE repair_id = ?", (repair_id,)
            ).fetchone()
        return self._qc_repair_record(row)

    def qc_repairs(self, candidate_id: str) -> tuple[QcRepairRecord, ...]:
        self.initialize()
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM qc_repairs WHERE candidate_id = ? ORDER BY created_at, repair_id",
                (candidate_id,),
            ).fetchall()
        return tuple(self._qc_repair_record(row) for row in rows)

    @staticmethod
    def _qc_repair_claim_record(row: sqlite3.Row) -> QcRepairClaimRecord:
        return QcRepairClaimRecord(
            claim_id=row["claim_id"],
            candidate_id=row["candidate_id"],
            evaluation_id=row["evaluation_id"],
            repair_input_sha256=row["repair_input_sha256"],
            planner_identity=_json_load(row["planner_identity_json"], {}),
            state=row["state"],
            created_at=row["created_at"],
            completed_at=row["completed_at"],
        )

    def claim_qc_repair_planner(
        self,
        *,
        candidate_id: str,
        evaluation_id: str,
        repair_input_sha256: str,
        planner_identity: Mapping[str, Any],
    ) -> tuple[QcRepairClaimRecord, bool]:
        """Durably consume the one permitted B1 planner invocation."""
        candidate_id = _evidence_id(candidate_id, "candidate_id")
        evaluation_id = _evidence_id(evaluation_id, "evaluation_id")
        repair_input_sha256 = _sha256(
            repair_input_sha256, "repair_input_sha256"
        )
        identity_json = _canonical_json(planner_identity)
        claim_id = "repair-claim-" + hashlib.sha256(
            f"{candidate_id}|{repair_input_sha256}".encode("utf-8")
        ).hexdigest()[:32]
        self.initialize()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM qc_repair_claims WHERE candidate_id = ?",
                (candidate_id,),
            ).fetchone()
            if existing is not None:
                immutable = (
                    existing["evaluation_id"],
                    existing["repair_input_sha256"],
                )
                if immutable != (
                    evaluation_id,
                    repair_input_sha256,
                ):
                    raise StateTransitionError(
                        "The one B1 planner claim has different immutable input."
                    )
                return self._qc_repair_claim_record(existing), False
            connection.execute(
                """
                INSERT INTO qc_repair_claims (
                    claim_id, candidate_id, evaluation_id,
                    repair_input_sha256, planner_identity_json, state,
                    created_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, 'CLAIMED', ?, NULL)
                """,
                (
                    claim_id,
                    candidate_id,
                    evaluation_id,
                    repair_input_sha256,
                    identity_json,
                    _utc_now(),
                ),
            )
            row = connection.execute(
                "SELECT * FROM qc_repair_claims WHERE claim_id = ?", (claim_id,)
            ).fetchone()
        return self._qc_repair_claim_record(row), True

    def complete_qc_repair_planner_claim(
        self, candidate_id: str, *, state: str
    ) -> QcRepairClaimRecord:
        candidate_id = _evidence_id(candidate_id, "candidate_id")
        if state not in {"COMPLETED", "FAILED_CLOSED"}:
            raise ValueError("Invalid QC repair planner claim terminal state.")
        self.initialize()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM qc_repair_claims WHERE candidate_id = ?",
                (candidate_id,),
            ).fetchone()
            if row is None:
                raise StateTransitionError("No durable B1 planner claim exists.")
            if row["state"] == "CLAIMED":
                connection.execute(
                    """
                    UPDATE qc_repair_claims
                    SET state = ?, completed_at = ?
                    WHERE candidate_id = ? AND state = 'CLAIMED'
                    """,
                    (state, _utc_now(), candidate_id),
                )
            elif row["state"] != state:
                raise StateTransitionError(
                    "The B1 planner claim already has a conflicting outcome."
                )
            final = connection.execute(
                "SELECT * FROM qc_repair_claims WHERE candidate_id = ?",
                (candidate_id,),
            ).fetchone()
        return self._qc_repair_claim_record(final)

    def qc_repair_planner_claim(
        self, candidate_id: str
    ) -> QcRepairClaimRecord | None:
        candidate_id = _evidence_id(candidate_id, "candidate_id")
        self.initialize()
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM qc_repair_claims WHERE candidate_id = ?",
                (candidate_id,),
            ).fetchone()
        return None if row is None else self._qc_repair_claim_record(row)

    @staticmethod
    def _qc_human_decision_record(row: sqlite3.Row) -> QcHumanDecisionRecord:
        return QcHumanDecisionRecord(
            decision_id=row["decision_id"],
            candidate_id=row["candidate_id"],
            decision=QcHumanDecision(row["decision"]),
            note=row["note"],
            actor=row["actor"],
            result_sha256=row["result_sha256"],
            evidence_sha256=row["evidence_sha256"],
            created_at=row["created_at"],
        )

    def record_qc_human_decision(
        self,
        *,
        decision_id: str,
        candidate_id: str,
        decision: QcHumanDecision,
        note: str | None,
        actor: str,
        result_sha256: str,
        evidence_sha256: str,
    ) -> QcHumanDecisionRecord:
        values = (
            _evidence_id(decision_id, "decision_id"),
            _evidence_id(candidate_id, "candidate_id"),
            decision.value,
            note,
            _required_text(actor, "actor"),
            _sha256(result_sha256, "result_sha256"),
            _sha256(evidence_sha256, "evidence_sha256"),
        )
        self.initialize()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """
                SELECT * FROM qc_human_decisions
                WHERE decision_id = ? OR candidate_id = ?
                """,
                (values[0], values[1]),
            ).fetchone()
            if existing is not None:
                stored = tuple(existing[key] for key in (
                    "decision_id", "candidate_id", "decision", "note", "actor",
                    "result_sha256", "evidence_sha256",
                ))
                if stored != values:
                    raise StateTransitionError(
                        "A terminal human QC decision cannot be overwritten."
                    )
                return self._qc_human_decision_record(existing)
            connection.execute(
                """
                INSERT INTO qc_human_decisions (
                    decision_id, candidate_id, decision, note, actor,
                    result_sha256, evidence_sha256, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (*values, _utc_now()),
            )
            row = connection.execute(
                "SELECT * FROM qc_human_decisions WHERE decision_id = ?", (values[0],)
            ).fetchone()
        return self._qc_human_decision_record(row)

    def qc_human_decision(self, candidate_id: str) -> QcHumanDecisionRecord | None:
        self.initialize()
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM qc_human_decisions WHERE candidate_id = ?",
                (candidate_id,),
            ).fetchone()
        return self._qc_human_decision_record(row) if row is not None else None

    def decide_qc_candidate(
        self,
        *,
        job_id: str,
        scene_id: int,
        candidate_id: str,
        decision: QcHumanDecision,
        note: str | None,
        actor: str = "local-operator",
    ) -> QcHumanDecisionResult:
        """Atomically record one terminal human action and route/promote it."""
        candidate_id = _evidence_id(candidate_id, "candidate_id")
        actor = _required_text(actor, "actor")
        if note is not None and not isinstance(note, str):
            raise StateTransitionError("Human decision note must be text or null.")
        decision_id = "decision-qc-" + hashlib.sha256(
            candidate_id.encode("utf-8")
        ).hexdigest()[:32]
        self.initialize()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            candidate = connection.execute(
                "SELECT * FROM qc_candidates WHERE candidate_id = ?",
                (candidate_id,),
            ).fetchone()
            if (
                candidate is None
                or candidate["job_id"] != job_id
                or int(candidate["scene_id"]) != scene_id
            ):
                raise StateTransitionError("QC candidate does not match the route identity.")
            pipeline = connection.execute(
                "SELECT state, job_id FROM pipeline_state WHERE singleton = 1"
            ).fetchone()
            job = connection.execute(
                "SELECT status FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if (
                pipeline is None
                or pipeline["job_id"] != job_id
                or PipelineState(pipeline["state"]) != PipelineState.AWAITING_QC_REVIEW
                or job is None
                or JobState(job["status"]) != JobState.RUNNING
            ):
                raise StateTransitionError(
                    "Human QC decisions require the active QC review job to remain RUNNING."
                )
            existing = connection.execute(
                "SELECT * FROM qc_human_decisions WHERE candidate_id = ?",
                (candidate_id,),
            ).fetchone()
            if existing is not None:
                if (
                    existing["decision"] != decision.value
                    or existing["note"] != note
                    or existing["actor"] != actor
                ):
                    raise StateTransitionError(
                        "A terminal human QC decision cannot be overwritten."
                    )
                return QcHumanDecisionResult(
                    self._qc_human_decision_record(existing),
                    self._qc_candidate_record(candidate),
                    True,
                )
            if QcCandidateState(candidate["state"]) != QcCandidateState.PASS_PENDING_HUMAN:
                raise StateTransitionError(
                    "Only the current PASS_PENDING_HUMAN candidate may be decided."
                )
            evaluation = connection.execute(
                """
                SELECT * FROM qc_evaluations
                WHERE candidate_id = ? AND state = 'COMPLETE'
                ORDER BY completed_at DESC, evaluation_id DESC LIMIT 1
                """,
                (candidate_id,),
            ).fetchone()
            if (
                evaluation is None
                or evaluation["normalized_decision"] != QcDecision.PASS.value
                or not evaluation["evidence_manifest_sha256"]
                or not candidate["source_video_sha256"]
            ):
                raise StateTransitionError(
                    "Human approval requires durable completed PASS evidence."
                )
            if (
                evaluation["source_video_path"] != candidate["source_video_path"]
                or evaluation["source_video_sha256"] != candidate["source_video_sha256"]
            ):
                raise StateTransitionError(
                    "Human QC evidence no longer matches the current candidate identity."
                )
            revision = connection.execute(
                """
                SELECT frame_path, video_path, state FROM scene_revisions
                WHERE job_id = ? AND scene_id = ? AND revision = ?
                """,
                (job_id, scene_id, candidate["revision"]),
            ).fetchone()
            if (
                revision is None
                or revision["state"] != SceneState.SUCCEEDED.value
                or revision["video_path"] != candidate["source_video_path"]
                or not Path(candidate["source_video_path"]).is_file()
                or _file_sha256(candidate["source_video_path"])
                != candidate["source_video_sha256"]
            ):
                raise StateTransitionError(
                    "QC candidate no longer matches its successful revision hash."
                )
            now = _utc_now()
            connection.execute(
                """
                INSERT INTO qc_human_decisions (
                    decision_id, candidate_id, decision, note, actor,
                    result_sha256, evidence_sha256, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decision_id, candidate_id, decision.value, note, actor,
                    candidate["source_video_sha256"],
                    evaluation["evidence_manifest_sha256"], now,
                ),
            )
            if decision == QcHumanDecision.APPROVE:
                connection.execute(
                    "UPDATE qc_candidates SET state = ?, next_action = NULL, updated_at = ? WHERE candidate_id = ?",
                    (QcCandidateState.ACCEPTED.value, now, candidate_id),
                )
                connection.execute(
                    """
                    UPDATE qc_candidates SET state = ?, next_action = NULL, updated_at = ?
                    WHERE job_id = ? AND scene_id = ? AND candidate_id != ?
                    """,
                    (
                        QcCandidateState.SUPERSEDED.value, now,
                        job_id, scene_id, candidate_id,
                    ),
                )
                connection.execute(
                    """
                    UPDATE scenes SET state = ?, frame_path = COALESCE(?, frame_path),
                        video_path = ?, error = NULL, updated_at = ?
                    WHERE job_id = ? AND scene_id = ?
                    """,
                    (
                        SceneState.SUCCEEDED.value, revision["frame_path"],
                        candidate["source_video_path"], now, job_id, scene_id,
                    ),
                )
            else:
                connection.execute(
                    "UPDATE qc_candidates SET state = ?, next_action = ?, updated_at = ? WHERE candidate_id = ?",
                    (
                        QcCandidateState.HOLD_FOR_REVIEW.value,
                        "hold_for_review", now, candidate_id,
                    ),
                )
            stored_decision = connection.execute(
                "SELECT * FROM qc_human_decisions WHERE decision_id = ?",
                (decision_id,),
            ).fetchone()
            stored_candidate = connection.execute(
                "SELECT * FROM qc_candidates WHERE candidate_id = ?",
                (candidate_id,),
            ).fetchone()
        return QcHumanDecisionResult(
            self._qc_human_decision_record(stored_decision),
            self._qc_candidate_record(stored_candidate),
            False,
        )

    def qc_final_selection(
        self,
        job_id: str,
        scene_ids: Sequence[int],
    ) -> tuple[ManualFinalSceneSelection, ...]:
        """Resolve exactly one durable ACCEPTED candidate for every scene."""
        required = tuple(sorted(set(scene_ids)))
        self.initialize()
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT c.scene_id, c.revision, c.source_video_path,
                       c.source_video_sha256,
                       r.video_path, r.state
                FROM qc_candidates c
                JOIN scene_revisions r ON r.job_id = c.job_id
                    AND r.scene_id = c.scene_id AND r.revision = c.revision
                WHERE c.job_id = ? AND c.state = ?
                ORDER BY c.scene_id
                """,
                (job_id, QcCandidateState.ACCEPTED.value),
            ).fetchall()
        if tuple(int(row["scene_id"]) for row in rows) != required:
            raise StateTransitionError(
                "QC finalization requires one ACCEPTED candidate for every required scene."
            )
        result = []
        for row in rows:
            if (
                row["state"] != SceneState.SUCCEEDED.value
                or row["video_path"] != row["source_video_path"]
                or not Path(row["source_video_path"]).is_file()
                or _file_sha256(row["source_video_path"])
                != row["source_video_sha256"]
            ):
                raise StateTransitionError(
                    "An ACCEPTED candidate is missing its bound successful video hash."
                )
            result.append(
                ManualFinalSceneSelection(
                    scene_id=int(row["scene_id"]),
                    revision=int(row["revision"]),
                    video_path=row["source_video_path"],
                )
            )
        return tuple(result)

    @staticmethod
    def _qc_finalization_plan_record(
        row: sqlite3.Row,
    ) -> QcFinalizationPlanRecord:
        return QcFinalizationPlanRecord(
            job_id=row["job_id"],
            version=int(row["version"]),
            selection=tuple(json.loads(row["selection_json"])),
            plan_sha256=row["plan_sha256"],
            state=row["state"],
            final_path=row["final_path"],
            final_sha256=row["final_sha256"],
            next_request_id=row["next_request_id"],
            next_request_receipt=row["next_request_receipt"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _qc_finalization_step_record(
        row: sqlite3.Row,
    ) -> QcFinalizationStepRecord:
        return QcFinalizationStepRecord(
            job_id=row["job_id"],
            step_key=row["step_key"],
            kind=row["kind"],
            state=row["state"],
            evidence=json.loads(row["evidence_json"]),
            evidence_sha256=row["evidence_sha256"],
            receipt=(
                None if row["receipt_json"] is None else json.loads(row["receipt_json"])
            ),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def ensure_qc_finalization_plan(
        self,
        job_id: str,
        scene_ids: Sequence[int],
        *,
        final_path: str,
    ) -> QcFinalizationPlanRecord:
        """Commit the accepted selection before any QC finalization side effect."""
        required = tuple(sorted(set(int(item) for item in scene_ids)))
        resolved_final = str(Path(final_path).resolve())
        self.initialize()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._assert_active_automatic_job_connection(connection, job_id)
            existing = connection.execute(
                "SELECT * FROM qc_finalization_plans WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if existing is not None:
                record = self._qc_finalization_plan_record(existing)
                if (
                    tuple(int(item["scene_id"]) for item in record.selection) != required
                    or record.final_path != resolved_final
                ):
                    raise StateTransitionError(
                        "The durable QC finalization plan is immutable."
                    )
                for item in record.selection:
                    if (
                        not Path(str(item["artifact_path"])).is_file()
                        or _file_sha256(str(item["artifact_path"]))
                        != item["artifact_sha256"]
                    ):
                        raise StateTransitionError(
                            "A snapshotted QC finalization artifact failed integrity validation."
                        )
                return record

            rows = connection.execute(
                """
                SELECT c.candidate_id, c.scene_id, c.revision,
                       c.source_video_path, c.source_video_sha256,
                       r.state AS revision_state, r.video_path AS revision_video_path
                FROM qc_candidates c
                JOIN scene_revisions r ON r.job_id = c.job_id
                    AND r.scene_id = c.scene_id AND r.revision = c.revision
                WHERE c.job_id = ? AND c.state = ?
                ORDER BY c.scene_id
                """,
                (job_id, QcCandidateState.ACCEPTED.value),
            ).fetchall()
            if tuple(int(row["scene_id"]) for row in rows) != required:
                raise StateTransitionError(
                    "QC finalization plan requires one ACCEPTED candidate for every scene."
                )
            selection: list[dict[str, Any]] = []
            for position, row in enumerate(rows, start=1):
                path = str(row["source_video_path"])
                artifact_hash = row["source_video_sha256"]
                if (
                    row["revision_state"] != SceneState.SUCCEEDED.value
                    or row["revision_video_path"] != path
                    or not artifact_hash
                    or not Path(path).is_file()
                    or _file_sha256(path) != artifact_hash
                ):
                    raise StateTransitionError(
                        "An ACCEPTED candidate cannot be snapshotted because its artifact changed."
                    )
                selection.append(
                    {
                        "position": position,
                        "candidate_id": row["candidate_id"],
                        "scene_id": int(row["scene_id"]),
                        "revision": int(row["revision"]),
                        "artifact_path": path,
                        "artifact_sha256": artifact_hash,
                    }
                )
            document = {
                "schema_version": 1,
                "job_id": job_id,
                "ordered_selection": selection,
                "final_path": resolved_final,
            }
            selection_json = _canonical_json(selection)
            plan_sha256 = hashlib.sha256(
                _canonical_json(document).encode("utf-8")
            ).hexdigest()
            now = _utc_now()
            connection.execute(
                """
                INSERT INTO qc_finalization_plans (
                    job_id, version, selection_json, plan_sha256, state,
                    final_path, final_sha256, next_request_id,
                    next_request_receipt, created_at, updated_at
                ) VALUES (?, 1, ?, ?, 'PLAN_COMMITTED', ?, NULL, NULL, NULL, ?, ?)
                """,
                (job_id, selection_json, plan_sha256, resolved_final, now, now),
            )
            stored = connection.execute(
                "SELECT * FROM qc_finalization_plans WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return self._qc_finalization_plan_record(stored)

    def qc_finalization_plan(self, job_id: str) -> QcFinalizationPlanRecord | None:
        self.initialize()
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM qc_finalization_plans WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return None if row is None else self._qc_finalization_plan_record(row)

    def advance_qc_finalization_plan(
        self,
        job_id: str,
        state: str,
        *,
        final_sha256: str | None = None,
        next_request_id: str | None = None,
        next_request_receipt: str | None = None,
    ) -> QcFinalizationPlanRecord:
        order = {
            "PLAN_COMMITTED": 0,
            "DELIVERING": 1,
            "DELIVERED": 2,
            "STITCHING": 3,
            "STITCHED": 4,
            "JOB_COMMITTED": 5,
            "NEXT_REQUEST_INTENT": 6,
            "COMPLETED": 7,
        }
        if state not in order:
            raise StateTransitionError("Unknown QC finalization plan state.")
        self.initialize()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._assert_active_automatic_job_connection(connection, job_id)
            row = connection.execute(
                "SELECT * FROM qc_finalization_plans WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if row is None:
                raise StateTransitionError("QC finalization plan is not committed.")
            current = self._qc_finalization_plan_record(row)
            if order[state] < order[current.state]:
                return current
            for old, new, label in (
                (current.final_sha256, final_sha256, "final artifact hash"),
                (current.next_request_id, next_request_id, "next request id"),
                (
                    current.next_request_receipt,
                    next_request_receipt,
                    "next request receipt",
                ),
            ):
                if old is not None and new is not None and old != new:
                    raise StateTransitionError(
                        f"The QC finalization {label} is immutable."
                    )
            now = _utc_now()
            connection.execute(
                """
                UPDATE qc_finalization_plans
                SET state = ?,
                    final_sha256 = COALESCE(final_sha256, ?),
                    next_request_id = COALESCE(next_request_id, ?),
                    next_request_receipt = COALESCE(next_request_receipt, ?),
                    updated_at = ?
                WHERE job_id = ?
                """,
                (
                    state,
                    final_sha256,
                    next_request_id,
                    next_request_receipt,
                    now,
                    job_id,
                ),
            )
            stored = connection.execute(
                "SELECT * FROM qc_finalization_plans WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return self._qc_finalization_plan_record(stored)

    def begin_qc_finalization_step(
        self,
        job_id: str,
        step_key: str,
        *,
        kind: str,
        evidence: Mapping[str, Any],
    ) -> QcFinalizationStepRecord:
        _evidence_id(step_key, "finalization step key")
        _evidence_id(kind, "finalization step kind")
        evidence_json = _canonical_json(dict(evidence))
        evidence_sha256 = hashlib.sha256(evidence_json.encode("utf-8")).hexdigest()
        self.initialize()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._assert_active_automatic_job_connection(connection, job_id)
            existing = connection.execute(
                """
                SELECT * FROM qc_finalization_steps
                WHERE job_id = ? AND step_key = ?
                """,
                (job_id, step_key),
            ).fetchone()
            if existing is not None:
                record = self._qc_finalization_step_record(existing)
                if record.kind != kind or record.evidence_sha256 != evidence_sha256:
                    raise StateTransitionError(
                        "Durable QC finalization step evidence is immutable."
                    )
                return record
            now = _utc_now()
            connection.execute(
                """
                INSERT INTO qc_finalization_steps (
                    job_id, step_key, kind, state, evidence_json,
                    evidence_sha256, receipt_json, created_at, updated_at
                ) VALUES (?, ?, ?, 'INTENT', ?, ?, NULL, ?, ?)
                """,
                (job_id, step_key, kind, evidence_json, evidence_sha256, now, now),
            )
            row = connection.execute(
                """
                SELECT * FROM qc_finalization_steps
                WHERE job_id = ? AND step_key = ?
                """,
                (job_id, step_key),
            ).fetchone()
        return self._qc_finalization_step_record(row)

    def complete_qc_finalization_step(
        self,
        job_id: str,
        step_key: str,
        *,
        receipt: Mapping[str, Any],
    ) -> QcFinalizationStepRecord:
        requested_receipt = dict(receipt)
        self.initialize()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._assert_active_automatic_job_connection(connection, job_id)
            existing = connection.execute(
                """
                SELECT * FROM qc_finalization_steps
                WHERE job_id = ? AND step_key = ?
                """,
                (job_id, step_key),
            ).fetchone()
            if existing is None:
                raise StateTransitionError(
                    "QC finalization step must have durable intent before completion."
                )
            record = self._qc_finalization_step_record(existing)
            merged_receipt = dict(record.receipt or {})
            for key, value in requested_receipt.items():
                if key in merged_receipt and merged_receipt[key] != value:
                    raise StateTransitionError(
                        "Durable QC finalization receipt fields are immutable."
                    )
                merged_receipt[key] = value
            receipt_json = _canonical_json(merged_receipt)
            if record.state == "COMPLETED":
                if _canonical_json(record.receipt) != receipt_json:
                    raise StateTransitionError(
                        "Durable QC finalization step receipt is immutable."
                    )
                return record
            now = _utc_now()
            connection.execute(
                """
                UPDATE qc_finalization_steps
                SET state = 'COMPLETED', receipt_json = ?, updated_at = ?
                WHERE job_id = ? AND step_key = ? AND state IN ('INTENT', 'DISPATCHING')
                """,
                (receipt_json, now, job_id, step_key),
            )
            row = connection.execute(
                """
                SELECT * FROM qc_finalization_steps
                WHERE job_id = ? AND step_key = ?
                """,
                (job_id, step_key),
            ).fetchone()
        return self._qc_finalization_step_record(row)

    def mark_qc_finalization_step_dispatching(
        self,
        job_id: str,
        step_key: str,
        *,
        receipt: Mapping[str, Any] | None = None,
    ) -> QcFinalizationStepRecord:
        """Persist the ambiguity boundary immediately before an external send."""
        return self.set_qc_finalization_step_dispatch_state(
            job_id,
            step_key,
            state="DISPATCHING",
            receipt=receipt,
        )

    def set_qc_finalization_step_dispatch_state(
        self,
        job_id: str,
        step_key: str,
        *,
        state: str,
        receipt: Mapping[str, Any] | None = None,
    ) -> QcFinalizationStepRecord:
        """Persist prompt binding or a terminal external-dispatch outcome."""
        if state not in {"DISPATCHING", "AMBIGUOUS", "FAILED"}:
            raise StateTransitionError("Unknown finalization dispatch state.")
        self.initialize()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._assert_active_automatic_job_connection(connection, job_id)
            row = connection.execute(
                """
                SELECT * FROM qc_finalization_steps
                WHERE job_id = ? AND step_key = ?
                """,
                (job_id, step_key),
            ).fetchone()
            if row is None:
                raise StateTransitionError("QC finalization dispatch has no intent.")
            record = self._qc_finalization_step_record(row)
            if record.state == "COMPLETED":
                return record
            if record.state in {"AMBIGUOUS", "FAILED"}:
                if record.state != state:
                    raise StateTransitionError(
                        "Terminal finalization dispatch state is immutable."
                    )
            elif record.state == "INTENT" and state != "DISPATCHING":
                raise StateTransitionError(
                    "Finalization dispatch must enter DISPATCHING before termination."
                )
            elif record.state not in {"INTENT", "DISPATCHING"}:
                raise StateTransitionError("QC finalization dispatch state is invalid.")
            merged_receipt = dict(record.receipt or {})
            for key, value in dict(receipt or {}).items():
                if key in merged_receipt and merged_receipt[key] != value:
                    raise StateTransitionError(
                        "Durable finalization dispatch receipt fields are immutable."
                    )
                merged_receipt[key] = value
            connection.execute(
                """
                UPDATE qc_finalization_steps
                SET state = ?, receipt_json = ?, updated_at = ?
                WHERE job_id = ? AND step_key = ?
                """,
                (
                    state,
                    _canonical_json(merged_receipt) if merged_receipt else None,
                    _utc_now(),
                    job_id,
                    step_key,
                ),
            )
            row = connection.execute(
                """
                SELECT * FROM qc_finalization_steps
                WHERE job_id = ? AND step_key = ?
                """,
                (job_id, step_key),
            ).fetchone()
            return self._qc_finalization_step_record(row)

    def bind_qc_finalization_step_prompt_id(
        self,
        job_id: str,
        step_key: str,
        prompt_id: str,
    ) -> QcFinalizationStepRecord:
        """Append one authoritative ComfyUI prompt ID to a dispatch intent."""
        prompt_id = _required_text(prompt_id, "prompt_id")
        self.initialize()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._assert_active_automatic_job_connection(connection, job_id)
            row = connection.execute(
                "SELECT * FROM qc_finalization_steps WHERE job_id = ? AND step_key = ?",
                (job_id, step_key),
            ).fetchone()
            if row is None:
                raise StateTransitionError("QC finalization dispatch has no intent.")
            record = self._qc_finalization_step_record(row)
            if record.state != "DISPATCHING":
                raise StateTransitionError(
                    "A prompt ID can bind only to a DISPATCHING finalization step."
                )
            receipt = dict(record.receipt or {})
            prompt_ids = list(receipt.get("prompt_ids") or [])
            if prompt_id not in prompt_ids:
                prompt_ids.append(prompt_id)
            receipt["prompt_ids"] = prompt_ids
            connection.execute(
                """
                UPDATE qc_finalization_steps
                SET receipt_json = ?, updated_at = ?
                WHERE job_id = ? AND step_key = ? AND state = 'DISPATCHING'
                """,
                (_canonical_json(receipt), _utc_now(), job_id, step_key),
            )
            updated = connection.execute(
                "SELECT * FROM qc_finalization_steps WHERE job_id = ? AND step_key = ?",
                (job_id, step_key),
            ).fetchone()
        return self._qc_finalization_step_record(updated)

    def qc_finalization_steps(
        self,
        job_id: str,
    ) -> tuple[QcFinalizationStepRecord, ...]:
        self.initialize()
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM qc_finalization_steps
                WHERE job_id = ? ORDER BY created_at, step_key
                """,
                (job_id,),
            ).fetchall()
        return tuple(self._qc_finalization_step_record(row) for row in rows)

    @staticmethod
    def _qc_job_hold_record(row: sqlite3.Row) -> QcJobHoldRecord:
        return QcJobHoldRecord(
            job_id=row["job_id"],
            kind=row["kind"],
            missing_scene_ids=tuple(
                int(item) for item in json.loads(row["missing_scene_ids_json"])
            ),
            evidence=json.loads(row["evidence_json"]),
            evidence_sha256=row["evidence_sha256"],
            created_at=row["created_at"],
        )

    def hold_incomplete_qc_job(
        self,
        job_id: str,
        missing_scene_ids: Sequence[int],
    ) -> QcJobHoldRecord:
        """Persist one immutable missing-scene incident and enter a stable hold."""
        missing = tuple(sorted(set(int(item) for item in missing_scene_ids)))
        if not missing or any(item < 1 for item in missing):
            raise StateTransitionError("A QC incomplete-scene hold needs scene identities.")
        self.initialize()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT state, job_id FROM pipeline_state WHERE singleton = 1"
            ).fetchone()
            job = connection.execute(
                "SELECT status, payload_json FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if (
                current is None
                or current["job_id"] != job_id
                or job is None
                or JobState(job["status"]) != JobState.RUNNING
            ):
                raise StateTransitionError(
                    "Only the active RUNNING job may enter a QC incomplete-scene hold."
                )
            existing = connection.execute(
                "SELECT * FROM qc_job_holds WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if existing is None:
                payload = json.loads(job["payload_json"])
                titles = {
                    int(item.get("id")): str(item.get("title", ""))
                    for item in payload.get("scenes", [])
                    if isinstance(item, Mapping) and isinstance(item.get("id"), int)
                }
                scene_rows = {
                    int(row["scene_id"]): row
                    for row in connection.execute(
                        "SELECT scene_id, state, error FROM scenes WHERE job_id = ?",
                        (job_id,),
                    ).fetchall()
                }
                evidence_scenes: list[dict[str, Any]] = []
                for scene_id in missing:
                    scene = scene_rows.get(scene_id)
                    revisions = connection.execute(
                        """
                        SELECT revision, state, video_path, error
                        FROM scene_revisions
                        WHERE job_id = ? AND scene_id = ?
                        ORDER BY revision
                        """,
                        (job_id, scene_id),
                    ).fetchall()
                    revision_evidence = []
                    for revision in revisions:
                        video_path = revision["video_path"]
                        try:
                            path_exists = bool(video_path and Path(video_path).is_file())
                        except OSError:
                            path_exists = False
                        revision_evidence.append(
                            {
                                "revision": int(revision["revision"]),
                                "state": revision["state"],
                                "video_path": video_path,
                                "video_path_exists": path_exists,
                                "error": revision["error"],
                            }
                        )
                    evidence_scenes.append(
                        {
                            "scene_id": scene_id,
                            "title": titles.get(scene_id, ""),
                            "state": scene["state"] if scene is not None else "missing",
                            "error": scene["error"] if scene is not None else "scene row missing",
                            "revisions": revision_evidence,
                        }
                    )
                evidence = {
                    "schema_version": 1,
                    "kind": "incomplete_pre_qc_scene_set",
                    "job_id": job_id,
                    "missing_scene_ids": list(missing),
                    "missing_scenes": evidence_scenes,
                }
                evidence_json = _canonical_json(evidence)
                now = _utc_now()
                connection.execute(
                    """
                    INSERT INTO qc_job_holds (
                        job_id, kind, missing_scene_ids_json, evidence_json,
                        evidence_sha256, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job_id,
                        "incomplete_pre_qc_scene_set",
                        _canonical_json(list(missing)),
                        evidence_json,
                        hashlib.sha256(evidence_json.encode("utf-8")).hexdigest(),
                        now,
                    ),
                )
                existing = connection.execute(
                    "SELECT * FROM qc_job_holds WHERE job_id = ?",
                    (job_id,),
                ).fetchone()
            held_missing = tuple(
                int(item) for item in json.loads(existing["missing_scene_ids_json"])
            )
            rendered = ", ".join(f"{item:02d}" for item in held_missing)
            connection.execute(
                """
                UPDATE pipeline_state
                SET state = ?, active_scene_id = NULL, error = ?, updated_at = ?
                WHERE singleton = 1 AND state != ?
                """,
                (
                    PipelineState.QC_BLOCKED.value,
                    "QC blocked: pre-QC selection is missing scene(s) " + rendered,
                    _utc_now(),
                    PipelineState.QC_BLOCKED.value,
                ),
            )
        return self._qc_job_hold_record(existing)

    def qc_job_hold(self, job_id: str) -> QcJobHoldRecord | None:
        self.initialize()
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM qc_job_holds WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return None if row is None else self._qc_job_hold_record(row)

    def original_final_selection(
        self, job_id: str, scene_ids: Sequence[int]
    ) -> tuple[ManualFinalSceneSelection, ...]:
        """Resolve the immutable pre-QC baseline, never a repair pointer."""
        required = tuple(sorted(set(scene_ids)))
        self.initialize()
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT c.scene_id, c.revision, c.source_video_path,
                       c.source_video_sha256, r.video_path, r.state
                FROM qc_candidates c
                JOIN scene_revisions r ON r.job_id = c.job_id
                    AND r.scene_id = c.scene_id AND r.revision = c.revision
                WHERE c.job_id = ? AND c.tier = ?
                ORDER BY c.scene_id
                """,
                (job_id, QcTier.ORIGINAL.value),
            ).fetchall()
        if not rows:
            # A deployment that has never enabled QC must remain byte-for-byte
            # equivalent to the legacy selector and must not create QC state.
            return self.legacy_final_selection(job_id, required)
        if tuple(int(row["scene_id"]) for row in rows) != required:
            raise StateTransitionError(
                "Kill-switch finalization requires every snapshotted pre-QC scene."
            )
        selected: list[ManualFinalSceneSelection] = []
        for row in rows:
            if (
                row["state"] != SceneState.SUCCEEDED.value
                or row["video_path"] != row["source_video_path"]
                or not Path(row["source_video_path"]).is_file()
                or _file_sha256(row["source_video_path"])
                != row["source_video_sha256"]
            ):
                raise StateTransitionError(
                    "The snapshotted pre-QC baseline no longer matches its artifact hash."
                )
            selected.append(
                ManualFinalSceneSelection(
                    scene_id=int(row["scene_id"]),
                    revision=int(row["revision"]),
                    video_path=row["source_video_path"],
                )
            )
        return tuple(selected)

    def legacy_final_selection(
        self, job_id: str, scene_ids: Sequence[int]
    ) -> tuple[ManualFinalSceneSelection, ...]:
        """Run the exact pre-QC latest-successful legacy scene selector."""
        required = tuple(sorted(set(scene_ids)))
        self.initialize()
        selected: list[ManualFinalSceneSelection] = []
        unavailable: list[int] = []
        with self._connection() as connection:
            active_manual = connection.execute(
                """
                SELECT selection_json FROM manual_final_requests
                WHERE job_id = ? AND state IN (?, ?)
                ORDER BY created_at DESC LIMIT 1
                """,
                (
                    job_id,
                    ManualFinalState.QUEUED.value,
                    ManualFinalState.RUNNING.value,
                ),
            ).fetchone()
            if active_manual is not None:
                try:
                    saved = {
                        int(item["scene_id"]): ManualFinalSceneSelection(
                            scene_id=int(item["scene_id"]),
                            revision=int(item["revision"]),
                            video_path=str(item["video_path"]),
                        )
                        for item in json.loads(active_manual["selection_json"])
                    }
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                    raise StateTransitionError(
                        "The active legacy manual-final snapshot is invalid."
                    ) from error
                missing_saved = tuple(
                    scene_id
                    for scene_id in required
                    if scene_id not in saved
                    or not Path(saved[scene_id].video_path).is_file()
                )
                if tuple(sorted(saved)) != required or missing_saved:
                    raise IncompleteLegacySelectionError(
                        missing_saved or required,
                        "The active legacy manual-final snapshot is incomplete.",
                    )
                return tuple(saved[scene_id] for scene_id in required)
            known = connection.execute(
                "SELECT scene_id, state, video_path FROM scenes WHERE job_id = ?",
                (job_id,),
            ).fetchall()
            by_scene = {int(row["scene_id"]): row for row in known}
            for scene_id in required:
                scene = by_scene.get(scene_id)
                if scene is None:
                    unavailable.append(scene_id)
                    continue
                revision = connection.execute(
                    """
                    SELECT revision, video_path FROM scene_revisions
                    WHERE job_id = ? AND scene_id = ? AND state = ?
                        AND video_path IS NOT NULL
                    ORDER BY revision DESC LIMIT 1
                    """,
                    (job_id, scene_id, SceneState.SUCCEEDED.value),
                ).fetchone()
                if revision is None and (
                    SceneState(scene["state"]) == SceneState.SUCCEEDED
                    and scene["video_path"]
                ):
                    revision = {"revision": 1, "video_path": scene["video_path"]}
                if revision is None or not Path(revision["video_path"]).is_file():
                    unavailable.append(scene_id)
                    continue
                selected.append(
                    ManualFinalSceneSelection(
                        scene_id=scene_id,
                        revision=int(revision["revision"]),
                        video_path=str(revision["video_path"]),
                    )
                )
        if unavailable:
            rendered = ", ".join(f"{scene_id:02d}" for scene_id in unavailable)
            raise IncompleteLegacySelectionError(
                unavailable,
                "Legacy selection has no successful video revision for scene(s): "
                f"{rendered}.",
            )
        return tuple(selected)

    def promote_accepted_qc_candidate(self, candidate_id: str) -> None:
        self.initialize()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            candidate = connection.execute(
                """
                SELECT * FROM qc_candidates WHERE candidate_id = ?
                """,
                (candidate_id,),
            ).fetchone()
            if candidate is None:
                raise StateTransitionError(f"Unknown QC candidate {candidate_id}.")
            if QcCandidateState(candidate["state"]) != QcCandidateState.ACCEPTED:
                raise StateTransitionError(
                    "Only an ACCEPTED QC candidate may be promoted."
                )
            revision = connection.execute(
                """
                SELECT frame_path, video_path, state FROM scene_revisions
                WHERE job_id = ? AND scene_id = ? AND revision = ?
                """,
                (
                    candidate["job_id"],
                    candidate["scene_id"],
                    candidate["revision"],
                ),
            ).fetchone()
            if (
                revision is None
                or revision["state"] != SceneState.SUCCEEDED.value
                or revision["video_path"] != candidate["source_video_path"]
                or not Path(candidate["source_video_path"]).is_file()
                or _file_sha256(candidate["source_video_path"])
                != candidate["source_video_sha256"]
            ):
                raise StateTransitionError(
                    "Accepted candidate no longer matches its immutable scene revision hash."
                )
            now = _utc_now()
            connection.execute(
                """
                UPDATE scenes SET state = ?, frame_path = COALESCE(?, frame_path),
                    video_path = ?, error = NULL, updated_at = ?
                WHERE job_id = ? AND scene_id = ?
                """,
                (
                    SceneState.SUCCEEDED.value,
                    revision["frame_path"],
                    candidate["source_video_path"],
                    now,
                    candidate["job_id"],
                    candidate["scene_id"],
                ),
            )
            connection.execute(
                """
                UPDATE qc_candidates SET state = ?, next_action = NULL, updated_at = ?
                WHERE job_id = ? AND scene_id = ? AND candidate_id != ?
                    AND state != ?
                """,
                (
                    QcCandidateState.SUPERSEDED.value,
                    now,
                    candidate["job_id"],
                    candidate["scene_id"],
                    candidate_id,
                    QcCandidateState.ACCEPTED.value,
                ),
            )

    def create_remake_batch(
        self,
        items: Sequence[tuple[str, int, RemakeMode, Mapping[str, Any]]],
    ) -> tuple[str, tuple[tuple[str, int, int], ...]]:
        if not items:
            raise StateTransitionError("A remake batch must contain at least one scene.")
        self.initialize()
        batch_id = uuid4().hex
        created: list[tuple[str, int, int]] = []
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            now = _utc_now()
            connection.execute(
                """
                INSERT INTO remake_batches (batch_id, state, collision_policy, created_at, updated_at)
                VALUES (?, ?, NULL, ?, ?)
                """,
                (batch_id, RemakeBatchState.DRAFT, now, now),
            )
            for position, (job_id, scene_id, mode, parameters) in enumerate(items, 1):
                exists = connection.execute(
                    "SELECT frame_path FROM scenes WHERE job_id = ? AND scene_id = ?",
                    (job_id, scene_id),
                ).fetchone()
                if exists is None:
                    raise StateTransitionError(
                        f"Unknown scene {scene_id} for job {job_id}."
                    )
                if mode == RemakeMode.VIDEO_ONLY and (
                    not exists["frame_path"]
                    or not Path(exists["frame_path"]).is_file()
                ):
                    raise StateTransitionError(
                        f"Scene {scene_id} has no cached frame for a video-only remake."
                    )
                row = connection.execute(
                    """
                    SELECT COALESCE(MAX(revision), 0) + 1 AS revision
                    FROM scene_revisions WHERE job_id = ? AND scene_id = ?
                    """,
                    (job_id, scene_id),
                ).fetchone()
                revision = int(row["revision"])
                inherited_frame = (
                    exists["frame_path"] if mode == RemakeMode.VIDEO_ONLY else None
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
                        mode,
                        json.dumps(parameters, ensure_ascii=False, sort_keys=True),
                        SceneState.PENDING,
                        inherited_frame,
                        now,
                        now,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO remake_items (
                        batch_id, position, job_id, scene_id, revision, state, error
                    ) VALUES (?, ?, ?, ?, ?, ?, NULL)
                    """,
                    (
                        batch_id,
                        position,
                        job_id,
                        scene_id,
                        revision,
                        SceneState.PENDING,
                    ),
                )
                created.append((job_id, scene_id, revision))
        return batch_id, tuple(created)

    def queue_remake_batch(self, batch_id: str, collision_policy: str) -> None:
        if collision_policy not in {"after_current", "interrupt_current"}:
            raise StateTransitionError("Unknown remake collision policy.")
        self.initialize()
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE remake_batches
                SET state = ?, collision_policy = ?, updated_at = ?
                WHERE batch_id = ? AND state = ?
                """,
                (
                    RemakeBatchState.QUEUED,
                    collision_policy,
                    _utc_now(),
                    batch_id,
                    RemakeBatchState.DRAFT,
                ),
            )
            if cursor.rowcount != 1:
                raise StateTransitionError(f"Batch {batch_id} is not a draft.")

    def next_queued_remake_batch(self) -> RemakeBatchRecord | None:
        return next(
            (
                batch
                for batch in reversed(self.list_remake_batches())
                if batch.state == RemakeBatchState.QUEUED
            ),
            None,
        )

    def recover_interrupted_remake_batches(self) -> tuple[str, ...]:
        """Requeue process-interrupted batches without discarding valid artifacts."""
        self.initialize()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                """
                SELECT batch_id FROM remake_batches
                WHERE state = ? ORDER BY created_at
                """,
                (RemakeBatchState.RUNNING,),
            ).fetchall()
            batch_ids = tuple(str(row["batch_id"]) for row in rows)
            now = _utc_now()
            for batch_id in batch_ids:
                # Revision success is committed before item success. Preserve
                # that narrow crash window instead of regenerating valid work.
                connection.execute(
                    """
                    UPDATE remake_items
                    SET state = CASE
                            WHEN EXISTS (
                                SELECT 1 FROM scene_revisions r
                                WHERE r.job_id = remake_items.job_id
                                  AND r.scene_id = remake_items.scene_id
                                  AND r.revision = remake_items.revision
                                  AND r.state = ?
                            ) THEN ?
                            ELSE ?
                        END,
                        error = NULL,
                        prompt_id = CASE
                            WHEN EXISTS (
                                SELECT 1 FROM scene_revisions r
                                WHERE r.job_id = remake_items.job_id
                                  AND r.scene_id = remake_items.scene_id
                                  AND r.revision = remake_items.revision
                                  AND r.state = ?
                            ) THEN NULL
                            ELSE prompt_id
                        END,
                        prompt_stage = CASE
                            WHEN EXISTS (
                                SELECT 1 FROM scene_revisions r
                                WHERE r.job_id = remake_items.job_id
                                  AND r.scene_id = remake_items.scene_id
                                  AND r.revision = remake_items.revision
                                  AND r.state = ?
                            ) THEN NULL
                            ELSE prompt_stage
                        END
                    WHERE batch_id = ? AND state = ?
                    """,
                    (
                        SceneState.SUCCEEDED,
                        SceneState.SUCCEEDED,
                        SceneState.PENDING,
                        SceneState.SUCCEEDED,
                        SceneState.SUCCEEDED,
                        batch_id,
                        SceneState.RUNNING,
                    ),
                )
                connection.execute(
                    """
                    UPDATE scene_revisions
                    SET state = ?, error = NULL, updated_at = ?
                    WHERE state = ?
                      AND EXISTS (
                          SELECT 1 FROM remake_items i
                          WHERE i.batch_id = ?
                            AND i.job_id = scene_revisions.job_id
                            AND i.scene_id = scene_revisions.scene_id
                            AND i.revision = scene_revisions.revision
                            AND i.state = ?
                      )
                    """,
                    (
                        SceneState.PENDING,
                        now,
                        SceneState.RUNNING,
                        batch_id,
                        SceneState.PENDING,
                    ),
                )
                connection.execute(
                    """
                    UPDATE remake_batches
                    SET state = ?, updated_at = ?
                    WHERE batch_id = ? AND state = ?
                    """,
                    (
                        RemakeBatchState.QUEUED,
                        now,
                        batch_id,
                        RemakeBatchState.RUNNING,
                    ),
                )
        return batch_ids

    def remake_items(self, batch_id: str) -> tuple[RemakeItemRecord, ...]:
        self.initialize()
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT batch_id, position, job_id, scene_id, revision, state,
                       error, prompt_id, prompt_stage
                FROM remake_items WHERE batch_id = ? ORDER BY position
                """,
                (batch_id,),
            ).fetchall()
        return tuple(
            RemakeItemRecord(
                batch_id=row["batch_id"],
                position=int(row["position"]),
                job_id=row["job_id"],
                scene_id=int(row["scene_id"]),
                revision=int(row["revision"]),
                state=SceneState(row["state"]),
                error=row["error"],
                prompt_id=row["prompt_id"],
                prompt_stage=row["prompt_stage"],
            )
            for row in rows
        )

    def set_remake_batch_state(
        self,
        batch_id: str,
        state: RemakeBatchState,
    ) -> None:
        self.initialize()
        with self._connection() as connection:
            cursor = connection.execute(
                "UPDATE remake_batches SET state = ?, updated_at = ? WHERE batch_id = ?",
                (state, _utc_now(), batch_id),
            )
            if cursor.rowcount != 1:
                raise StateTransitionError(f"Unknown remake batch {batch_id}.")

    def set_remake_item_state(
        self,
        batch_id: str,
        position: int,
        state: SceneState,
        *,
        error: str | None = None,
    ) -> None:
        self.initialize()
        release_prompt = state in {SceneState.SUCCEEDED, SceneState.CANCELLED}
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE remake_items
                SET state = ?, error = ?,
                    prompt_id = CASE WHEN ? THEN NULL ELSE prompt_id END,
                    prompt_stage = CASE WHEN ? THEN NULL ELSE prompt_stage END
                WHERE batch_id = ? AND position = ?
                """,
                (
                    state,
                    error,
                    release_prompt,
                    release_prompt,
                    batch_id,
                    position,
                ),
            )
            if cursor.rowcount != 1:
                raise StateTransitionError(
                    f"Unknown item {position} in remake batch {batch_id}."
                )

    def set_remake_item_prompt_id(
        self,
        batch_id: str,
        position: int,
        prompt_id: str,
        *,
        stage: str,
    ) -> None:
        """Persist exact ComfyUI ownership for restart-safe remake reclaim."""
        if not isinstance(prompt_id, str) or not prompt_id:
            raise StateTransitionError("prompt_id must be non-empty text.")
        if stage not in {"t2i", "i2v_legacy", "i2v_continuation"}:
            raise StateTransitionError(
                "Remake prompt stage must be t2i, i2v_legacy, or "
                "i2v_continuation."
            )
        self.initialize()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT i.state AS item_state, r.state AS revision_state
                FROM remake_items i
                JOIN scene_revisions r
                  ON r.job_id = i.job_id
                 AND r.scene_id = i.scene_id
                 AND r.revision = i.revision
                WHERE i.batch_id = ? AND i.position = ?
                """,
                (batch_id, position),
            ).fetchone()
            if row is None:
                raise StateTransitionError(
                    f"Unknown item {position} in remake batch {batch_id}."
                )
            if (
                SceneState(row["item_state"]) == SceneState.CANCELLED
                or SceneState(row["revision_state"]) == SceneState.CANCELLED
            ):
                raise StateTransitionError(
                    "A cancelled remake cannot claim a ComfyUI prompt."
                )
            connection.execute(
                """
                UPDATE remake_items SET prompt_id = ?, prompt_stage = ?
                WHERE batch_id = ? AND position = ?
                """,
                (prompt_id, stage, batch_id, position),
            )

    def clear_remake_item_prompt(
        self,
        batch_id: str,
        position: int,
        *,
        expected_stage: str | None = None,
        preserve_stage: bool = False,
    ) -> None:
        """Clear one completed/failed remake prompt without touching another stage."""
        self.initialize()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if expected_stage is None:
                cursor = connection.execute(
                    """
                    UPDATE remake_items SET prompt_id = NULL, prompt_stage = NULL
                    WHERE batch_id = ? AND position = ?
                    """,
                    (batch_id, position),
                )
            else:
                stage_assignment = (
                    "prompt_stage = prompt_stage"
                    if preserve_stage
                    else "prompt_stage = NULL"
                )
                cursor = connection.execute(
                    f"""
                    UPDATE remake_items SET prompt_id = NULL, {stage_assignment}
                    WHERE batch_id = ? AND position = ? AND prompt_stage = ?
                    """,
                    (batch_id, position, expected_stage),
                )
            if cursor.rowcount not in {0, 1}:
                raise StateTransitionError(
                    f"Could not clear prompt for item {position} in batch {batch_id}."
                )

    def list_remake_batches(self) -> tuple[RemakeBatchRecord, ...]:
        self.initialize()
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT b.batch_id, b.state, b.collision_policy, b.created_at, b.updated_at,
                       COUNT(i.position) AS item_count,
                       SUM(CASE WHEN i.state = ? THEN 1 ELSE 0 END) AS completed_count
                FROM remake_batches b
                LEFT JOIN remake_items i ON i.batch_id = b.batch_id
                GROUP BY b.batch_id
                ORDER BY b.created_at DESC
                """,
                (SceneState.SUCCEEDED,),
            ).fetchall()
        return tuple(
            RemakeBatchRecord(
                batch_id=row["batch_id"],
                state=RemakeBatchState(row["state"]),
                collision_policy=row["collision_policy"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                item_count=int(row["item_count"] or 0),
                completed_count=int(row["completed_count"] or 0),
            )
            for row in rows
        )

    def set_scene_manual_final_inclusion(
        self,
        job_id: str,
        scene_id: int,
        *,
        included: bool,
    ) -> None:
        """Persist a scene's opt-in state for manually requested final assembly only."""
        self.initialize()
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE scenes
                SET include_in_manual_final = ?, updated_at = ?
                WHERE job_id = ? AND scene_id = ?
                """,
                (int(included), _utc_now(), job_id, scene_id),
            )
            if cursor.rowcount != 1:
                raise StateTransitionError(f"Unknown scene {scene_id} for job {job_id}.")

    def queue_manual_final(
        self,
        job_id: str,
        *,
        quality_control_enabled: bool = False,
    ) -> ManualFinalRecord:
        """Snapshot current manual-final choices without changing automatic job assembly."""
        self.initialize()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            exists = connection.execute(
                "SELECT 1 FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if exists is None:
                raise StateTransitionError(f"Unknown job {job_id}.")
            active = connection.execute(
                """
                SELECT request_id, job_id, state, output_path, error, created_at, updated_at
                FROM manual_final_requests
                WHERE job_id = ? AND state IN (?, ?)
                ORDER BY created_at DESC LIMIT 1
                """,
                (job_id, ManualFinalState.QUEUED, ManualFinalState.RUNNING),
            ).fetchone()
            if active is not None:
                return self._manual_final_record(active)

            selection = self._manual_final_selection(
                connection,
                job_id,
                quality_control_enabled=quality_control_enabled,
            )
            request_id = uuid4().hex
            now = _utc_now()
            connection.execute(
                """
                INSERT INTO manual_final_requests (
                    request_id, job_id, state, selection_json, output_path, error,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, NULL, NULL, ?, ?)
                """,
                (
                    request_id,
                    job_id,
                    ManualFinalState.QUEUED,
                    json.dumps(selection, ensure_ascii=False, sort_keys=True),
                    now,
                    now,
                ),
            )
        return ManualFinalRecord(
            request_id=request_id,
            job_id=job_id,
            state=ManualFinalState.QUEUED,
            output_path=None,
            error=None,
            created_at=now,
            updated_at=now,
        )

    def next_queued_manual_final(self) -> ManualFinalRecord | None:
        self.initialize()
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT request_id, job_id, state, output_path, error, created_at, updated_at
                FROM manual_final_requests
                WHERE state = ?
                ORDER BY created_at ASC LIMIT 1
                """,
                (ManualFinalState.QUEUED,),
            ).fetchone()
        return self._manual_final_record(row) if row is not None else None

    def latest_manual_final(self, job_id: str) -> ManualFinalRecord | None:
        self.initialize()
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT request_id, job_id, state, output_path, error, created_at, updated_at
                FROM manual_final_requests
                WHERE job_id = ?
                ORDER BY created_at DESC LIMIT 1
                """,
                (job_id,),
            ).fetchone()
        return self._manual_final_record(row) if row is not None else None

    def latest_manual_final_any(self) -> ManualFinalRecord | None:
        self.initialize()
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT request_id, job_id, state, output_path, error, created_at, updated_at
                FROM manual_final_requests
                ORDER BY created_at DESC LIMIT 1
                """
            ).fetchone()
        return self._manual_final_record(row) if row is not None else None

    def manual_final_selection(
        self,
        request_id: str,
    ) -> tuple[ManualFinalSceneSelection, ...]:
        self.initialize()
        with self._connection() as connection:
            row = connection.execute(
                "SELECT selection_json FROM manual_final_requests WHERE request_id = ?",
                (request_id,),
            ).fetchone()
        if row is None:
            raise StateTransitionError(f"Unknown manual final request {request_id}.")
        try:
            values = json.loads(row["selection_json"])
            return tuple(
                ManualFinalSceneSelection(
                    scene_id=int(item["scene_id"]),
                    revision=int(item["revision"]),
                    video_path=str(item["video_path"]),
                )
                for item in values
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise StateTransitionError(
                f"Manual final request {request_id} has an invalid saved selection."
            ) from error

    def set_manual_final_state(
        self,
        request_id: str,
        state: ManualFinalState,
        *,
        output_path: str | None = None,
        error: str | None = None,
    ) -> None:
        self.initialize()
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE manual_final_requests
                SET state = ?, output_path = COALESCE(?, output_path), error = ?, updated_at = ?
                WHERE request_id = ?
                """,
                (state, output_path, error, _utc_now(), request_id),
            )
            if cursor.rowcount != 1:
                raise StateTransitionError(f"Unknown manual final request {request_id}.")

    def set_job_final_path(self, job_id: str, final_path: str) -> None:
        self.initialize()
        with self._connection() as connection:
            cursor = connection.execute(
                "UPDATE jobs SET final_path = ?, updated_at = ? WHERE job_id = ?",
                (final_path, _utc_now(), job_id),
            )
            if cursor.rowcount != 1:
                raise StateTransitionError(f"Unknown job {job_id}.")

    @staticmethod
    def _manual_final_record(row: sqlite3.Row) -> ManualFinalRecord:
        return ManualFinalRecord(
            request_id=row["request_id"],
            job_id=row["job_id"],
            state=ManualFinalState(row["state"]),
            output_path=row["output_path"],
            error=row["error"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _manual_final_selection(
        connection: sqlite3.Connection,
        job_id: str,
        *,
        quality_control_enabled: bool = False,
    ) -> list[dict[str, Any]]:
        scenes = connection.execute(
            """
            SELECT scene_id, state, video_path
            FROM scenes
            WHERE job_id = ? AND include_in_manual_final = 1
            ORDER BY scene_id
            """,
            (job_id,),
        ).fetchall()
        if not scenes:
            raise StateTransitionError(
                "At least one scene must be included in the manual final."
            )
        selected: list[dict[str, Any]] = []
        unavailable: list[int] = []
        has_qc = connection.execute(
            "SELECT 1 FROM qc_candidates WHERE job_id = ? LIMIT 1", (job_id,)
        ).fetchone() is not None
        for scene in scenes:
            if quality_control_enabled:
                revision = connection.execute(
                    """
                    SELECT r.revision, r.video_path
                    FROM qc_candidates c
                    JOIN scene_revisions r ON r.job_id = c.job_id
                        AND r.scene_id = c.scene_id AND r.revision = c.revision
                    WHERE c.job_id = ? AND c.scene_id = ? AND c.state = ?
                        AND r.state = ? AND r.video_path = c.source_video_path
                    """,
                    (
                        job_id, scene["scene_id"], QcCandidateState.ACCEPTED.value,
                        SceneState.SUCCEEDED.value,
                    ),
                ).fetchone()
            elif has_qc:
                # Kill switch: never let a newer unapproved retry become the
                # baseline. Restore the immutable pre-QC selector snapshot.
                revision = connection.execute(
                    """
                    SELECT r.revision, r.video_path
                    FROM qc_candidates c
                    JOIN scene_revisions r ON r.job_id = c.job_id
                        AND r.scene_id = c.scene_id AND r.revision = c.revision
                    WHERE c.job_id = ? AND c.scene_id = ? AND c.tier = ?
                        AND r.state = ? AND r.video_path = c.source_video_path
                    """,
                    (
                        job_id, scene["scene_id"], QcTier.ORIGINAL.value,
                        SceneState.SUCCEEDED.value,
                    ),
                ).fetchone()
            else:
                revision = connection.execute(
                    """
                    SELECT revision, video_path
                    FROM scene_revisions
                    WHERE job_id = ? AND scene_id = ? AND state = ? AND video_path IS NOT NULL
                    ORDER BY revision DESC LIMIT 1
                    """,
                    (job_id, scene["scene_id"], SceneState.SUCCEEDED),
                ).fetchone()
            if revision is None and not (quality_control_enabled or has_qc) and (
                SceneState(scene["state"]) == SceneState.SUCCEEDED
                and scene["video_path"]
            ):
                revision = {"revision": 1, "video_path": scene["video_path"]}
            if revision is None or not Path(revision["video_path"]).is_file():
                unavailable.append(int(scene["scene_id"]))
                continue
            selected.append(
                {
                    "scene_id": int(scene["scene_id"]),
                    "revision": int(revision["revision"]),
                    "video_path": str(revision["video_path"]),
                }
            )
        if unavailable:
            rendered = ", ".join(f"{scene_id:02d}" for scene_id in unavailable)
            raise StateTransitionError(
                "Included scene(s) have no successful video revision: "
                f"{rendered}. Exclude them or finish their remake first."
            )
        return selected

    @staticmethod
    def _continuation_plan_record(row: sqlite3.Row) -> ContinuationPlanRecord:
        return ContinuationPlanRecord(
            job_id=row["job_id"],
            scene_id=int(row["scene_id"]),
            revision=int(row["revision"]),
            plan_hash=row["plan_hash"],
            plan=json.loads(row["plan_json"]),
            created_at=row["created_at"],
        )

    @staticmethod
    def _chunk_record(row: sqlite3.Row) -> SceneChunkRecord:
        return SceneChunkRecord(
            job_id=row["job_id"],
            scene_id=int(row["scene_id"]),
            revision=int(row["revision"]),
            chunk_index=int(row["chunk_index"]),
            plan_hash=row["plan_hash"],
            chunk=json.loads(row["chunk_json"]),
            state=ChunkState(row["state"]),
            accepted_attempt_number=(
                int(row["accepted_attempt_number"])
                if row["accepted_attempt_number"] is not None
                else None
            ),
            accepted_artifact_hash=row["accepted_artifact_hash"],
            error=row["error"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _chunk_attempt_record(row: sqlite3.Row) -> ChunkAttemptRecord:
        seed = int(row["seed"])
        _uint64_seed(seed)
        return ChunkAttemptRecord(
            job_id=row["job_id"],
            scene_id=int(row["scene_id"]),
            revision=int(row["revision"]),
            chunk_index=int(row["chunk_index"]),
            attempt_number=int(row["attempt_number"]),
            variation_index=int(row["variation_index"]),
            state=ChunkState(row["state"]),
            seed=seed,
            parameters=json.loads(row["parameters_json"]),
            upstream_artifact_hash=row["upstream_artifact_hash"],
            artifact_manifest_path=row["artifact_manifest_path"],
            artifact_hash=row["artifact_hash"],
            video_path=row["video_path"],
            result=json.loads(row["result_json"]),
            error=row["error"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def ensure_continuation_plan(
        self,
        job_id: str,
        scene_id: int,
        revision: int,
        plan_hash: str,
        plan: Mapping[str, Any],
    ) -> ContinuationPlanRecord:
        """Create one immutable plan, or return the equivalent existing plan."""
        _positive_revision(revision)
        if not isinstance(plan_hash, str) or not plan_hash.strip():
            raise StateTransitionError("plan_hash must be a non-empty string.")
        plan_json = _canonical_json(plan)
        self.initialize()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            scene = connection.execute(
                "SELECT 1 FROM scenes WHERE job_id = ? AND scene_id = ?",
                (job_id, scene_id),
            ).fetchone()
            if scene is None:
                raise StateTransitionError(f"Unknown scene {scene_id} for job {job_id}.")
            if not self._continuation_work_is_active_connection(
                connection,
                job_id,
                scene_id,
                revision,
            ):
                raise StateTransitionError(
                    "A cancelled or detached scene cannot create a continuation plan."
                )
            existing = connection.execute(
                """
                SELECT job_id, scene_id, revision, plan_hash, plan_json, created_at
                FROM continuation_plans
                WHERE job_id = ? AND scene_id = ? AND revision = ?
                """,
                (job_id, scene_id, revision),
            ).fetchone()
            if existing is not None:
                if (
                    existing["plan_hash"] != plan_hash
                    or _canonical_json(json.loads(existing["plan_json"])) != plan_json
                ):
                    raise StateTransitionError(
                        f"Continuation plan for {job_id} scene {scene_id} revision "
                        f"{revision} is immutable."
                    )
                return self._continuation_plan_record(existing)
            now = _utc_now()
            connection.execute(
                """
                INSERT INTO continuation_plans (
                    job_id, scene_id, revision, plan_hash, plan_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (job_id, scene_id, revision, plan_hash, plan_json, now),
            )
            row = connection.execute(
                """
                SELECT job_id, scene_id, revision, plan_hash, plan_json, created_at
                FROM continuation_plans
                WHERE job_id = ? AND scene_id = ? AND revision = ?
                """,
                (job_id, scene_id, revision),
            ).fetchone()
        return self._continuation_plan_record(row)

    def continuation_plan(
        self,
        job_id: str,
        scene_id: int,
        revision: int,
    ) -> ContinuationPlanRecord | None:
        _positive_revision(revision)
        self.initialize()
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT job_id, scene_id, revision, plan_hash, plan_json, created_at
                FROM continuation_plans
                WHERE job_id = ? AND scene_id = ? AND revision = ?
                """,
                (job_id, scene_id, revision),
            ).fetchone()
        return self._continuation_plan_record(row) if row is not None else None

    def continuation_revision_snapshot(
        self,
        job_id: str,
        scene_id: int,
        revision: int,
    ) -> Mapping[str, Any]:
        """Return a JSON-safe audit snapshot before an explicit strategy upgrade."""
        _positive_revision(revision)
        self.initialize()
        with self._connection() as connection:
            plan = connection.execute(
                """
                SELECT job_id, scene_id, revision, plan_hash, plan_json, created_at
                FROM continuation_plans
                WHERE job_id = ? AND scene_id = ? AND revision = ?
                """,
                (job_id, scene_id, revision),
            ).fetchone()
            if plan is None:
                raise StateTransitionError(
                    f"No continuation plan exists for {job_id} scene {scene_id} "
                    f"revision {revision}."
                )
            chunks = connection.execute(
                """
                SELECT * FROM scene_chunks
                WHERE job_id = ? AND scene_id = ? AND revision = ?
                ORDER BY chunk_index
                """,
                (job_id, scene_id, revision),
            ).fetchall()
            attempts = connection.execute(
                """
                SELECT * FROM chunk_attempts
                WHERE job_id = ? AND scene_id = ? AND revision = ?
                ORDER BY chunk_index, attempt_number
                """,
                (job_id, scene_id, revision),
            ).fetchall()
        return {
            "job_id": job_id,
            "scene_id": scene_id,
            "revision": revision,
            "plan": {
                "plan_hash": plan["plan_hash"],
                "document": json.loads(plan["plan_json"]),
                "created_at": plan["created_at"],
            },
            "chunks": [
                {
                    **dict(row),
                    "chunk": json.loads(row["chunk_json"]),
                }
                for row in chunks
            ],
            "attempts": [
                {
                    **dict(row),
                    "parameters": json.loads(row["parameters_json"]),
                    "result": json.loads(row["result_json"]),
                }
                for row in attempts
            ],
        }

    def reset_legacy_original_continuation(
        self,
        job_id: str,
        scene_id: int,
        *,
        expected_plan_hash: str,
        expected_strategy: str,
    ) -> None:
        """Drop archived revision-one chunk rows so a new strategy can re-plan.

        The caller must first persist ``continuation_revision_snapshot`` and verify
        that no ComfyUI prompt is running. Artifact files are intentionally left to
        the caller so they can be moved into D-drive history without copying them.
        """
        if not expected_plan_hash or not expected_strategy:
            raise StateTransitionError("Strategy upgrade identity must be non-empty.")
        self.initialize()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT state, job_id FROM pipeline_state WHERE singleton = 1"
            ).fetchone()
            if current is None or current["job_id"] != job_id:
                raise StateTransitionError(
                    f"Cannot upgrade {job_id}; it is not the active saved job."
                )
            plan = connection.execute(
                """
                SELECT plan_hash, plan_json FROM continuation_plans
                WHERE job_id = ? AND scene_id = ? AND revision = 1
                """,
                (job_id, scene_id),
            ).fetchone()
            if plan is None:
                raise StateTransitionError(
                    f"No legacy continuation exists for {job_id} scene {scene_id}."
                )
            document = json.loads(plan["plan_json"])
            if (
                plan["plan_hash"] != expected_plan_hash
                or document.get("strategy") != expected_strategy
            ):
                raise StateTransitionError(
                    "Continuation plan changed before upgrade; no state was modified."
                )
            connection.execute(
                """
                DELETE FROM chunk_attempts
                WHERE job_id = ? AND scene_id = ? AND revision = 1
                """,
                (job_id, scene_id),
            )
            connection.execute(
                """
                DELETE FROM scene_chunks
                WHERE job_id = ? AND scene_id = ? AND revision = 1
                """,
                (job_id, scene_id),
            )
            connection.execute(
                """
                DELETE FROM continuation_plans
                WHERE job_id = ? AND scene_id = ? AND revision = 1
                """,
                (job_id, scene_id),
            )
            now = _utc_now()
            connection.execute(
                """
                UPDATE scenes
                SET state = ?, video_path = NULL, error = NULL,
                    i2v_attempts = 0, prompt_id = NULL, prompt_stage = NULL,
                    updated_at = ?
                WHERE job_id = ? AND scene_id = ?
                """,
                (SceneState.PENDING, now, job_id, scene_id),
            )
            connection.execute(
                """
                UPDATE scene_revisions
                SET state = ?, video_path = NULL, error = NULL, updated_at = ?
                WHERE job_id = ? AND scene_id = ? AND revision = 1
                """,
                (SceneState.PENDING, now, job_id, scene_id),
            )
            connection.execute(
                "UPDATE jobs SET status = ?, updated_at = ? WHERE job_id = ?",
                (JobState.QUEUED, now, job_id),
            )
            connection.execute(
                """
                UPDATE pipeline_state
                SET state = ?, job_id = ?, active_scene_id = NULL,
                    error = NULL, updated_at = ?
                WHERE singleton = 1
                """,
                (PipelineState.DOWNLOADING_ASSETS, job_id, now),
            )

    def plan_chunks(
        self,
        job_id: str,
        scene_id: int,
        revision: int,
        plan_hash: str,
        chunks: Sequence[Mapping[str, Any]],
    ) -> tuple[SceneChunkRecord, ...]:
        """Atomically materialize an immutable, contiguous chunk plan."""
        _positive_revision(revision)
        if not chunks:
            raise StateTransitionError("A continuation plan must contain at least one chunk.")
        normalized: list[tuple[int, str]] = []
        for position, chunk in enumerate(chunks):
            raw_index = chunk.get("chunk_index", chunk.get("index"))
            chunk_index = _nonnegative_chunk_index(raw_index)
            if chunk_index != position:
                raise StateTransitionError(
                    "Chunk indices must be unique and contiguous starting at zero."
                )
            normalized.append((chunk_index, _canonical_json(chunk)))

        self.initialize()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if not self._continuation_work_is_active_connection(
                connection,
                job_id,
                scene_id,
                revision,
            ):
                raise StateTransitionError(
                    "A cancelled or detached scene cannot materialize continuation chunks."
                )
            plan_row = connection.execute(
                """
                SELECT plan_hash FROM continuation_plans
                WHERE job_id = ? AND scene_id = ? AND revision = ?
                """,
                (job_id, scene_id, revision),
            ).fetchone()
            if plan_row is None:
                raise StateTransitionError(
                    f"No continuation plan exists for {job_id} scene {scene_id} "
                    f"revision {revision}."
                )
            if plan_row["plan_hash"] != plan_hash:
                raise StateTransitionError("Chunk plan hash does not match its continuation plan.")
            existing = connection.execute(
                """
                SELECT * FROM scene_chunks
                WHERE job_id = ? AND scene_id = ? AND revision = ?
                ORDER BY chunk_index
                """,
                (job_id, scene_id, revision),
            ).fetchall()
            if existing:
                expected = [
                    (int(row["chunk_index"]), _canonical_json(json.loads(row["chunk_json"])))
                    for row in existing
                ]
                if expected != normalized or any(
                    row["plan_hash"] != plan_hash for row in existing
                ):
                    raise StateTransitionError(
                        f"Chunks for {job_id} scene {scene_id} revision {revision} "
                        "are immutable."
                    )
                return tuple(self._chunk_record(row) for row in existing)
            now = _utc_now()
            connection.executemany(
                """
                INSERT INTO scene_chunks (
                    job_id, scene_id, revision, chunk_index, plan_hash, chunk_json,
                    state, accepted_attempt_number, accepted_artifact_hash, error,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, ?, ?)
                """,
                [
                    (
                        job_id,
                        scene_id,
                        revision,
                        chunk_index,
                        plan_hash,
                        chunk_json,
                        ChunkState.READY if chunk_index == 0 else ChunkState.BLOCKED_UPSTREAM,
                        now,
                        now,
                    )
                    for chunk_index, chunk_json in normalized
                ],
            )
            rows = connection.execute(
                """
                SELECT * FROM scene_chunks
                WHERE job_id = ? AND scene_id = ? AND revision = ?
                ORDER BY chunk_index
                """,
                (job_id, scene_id, revision),
            ).fetchall()
        return tuple(self._chunk_record(row) for row in rows)

    def chunk_records(
        self,
        job_id: str,
        scene_id: int,
        revision: int,
    ) -> tuple[SceneChunkRecord, ...]:
        _positive_revision(revision)
        self.initialize()
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM scene_chunks
                WHERE job_id = ? AND scene_id = ? AND revision = ?
                ORDER BY chunk_index
                """,
                (job_id, scene_id, revision),
            ).fetchall()
        return tuple(self._chunk_record(row) for row in rows)

    def scene_chunks(
        self,
        job_id: str,
        scene_id: int,
        revision: int,
    ) -> tuple[SceneChunkRecord, ...]:
        """Compatibility alias with the persisted table's domain name."""
        return self.chunk_records(job_id, scene_id, revision)

    def begin_chunk_attempt(
        self,
        job_id: str,
        scene_id: int,
        revision: int,
        chunk_index: int,
        *,
        seed: int,
        variation_index: int = 0,
        parameters: Mapping[str, Any] | None = None,
        upstream_artifact_hash: str | None = None,
        attempt_number: int | None = None,
    ) -> ChunkAttemptRecord:
        """Begin or resume one immutable chunk attempt."""
        _positive_revision(revision)
        _nonnegative_chunk_index(chunk_index)
        _uint64_seed(seed)
        if (
            isinstance(variation_index, bool)
            or not isinstance(variation_index, int)
            or variation_index < 0
        ):
            raise StateTransitionError("variation_index must be a non-negative integer.")
        if attempt_number is not None:
            _positive_attempt_number(attempt_number)
        parameters_json = _canonical_json(parameters or {})

        self.initialize()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            job = connection.execute(
                "SELECT status FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if job is None:
                raise StateTransitionError(f"Unknown job {job_id}.")
            if not self._continuation_work_is_active_connection(
                connection,
                job_id,
                scene_id,
                revision,
            ):
                raise StateTransitionError(
                    "A cancelled job or scene revision cannot begin chunk attempts."
                )
            chunk = connection.execute(
                """
                SELECT * FROM scene_chunks
                WHERE job_id = ? AND scene_id = ? AND revision = ? AND chunk_index = ?
                """,
                (job_id, scene_id, revision, chunk_index),
            ).fetchone()
            if chunk is None:
                raise StateTransitionError(
                    f"Unknown chunk {chunk_index} for {job_id} scene {scene_id} "
                    f"revision {revision}."
                )
            if chunk_index == 0:
                if upstream_artifact_hash is not None:
                    raise StateTransitionError("Chunk zero cannot have an upstream chunk hash.")
                resolved_upstream_hash = None
            else:
                predecessor = connection.execute(
                    """
                    SELECT accepted_artifact_hash FROM scene_chunks
                    WHERE job_id = ? AND scene_id = ? AND revision = ?
                      AND chunk_index = ?
                    """,
                    (job_id, scene_id, revision, chunk_index - 1),
                ).fetchone()
                if predecessor is None or not predecessor["accepted_artifact_hash"]:
                    raise StateTransitionError(
                        f"Chunk {chunk_index} is blocked until chunk {chunk_index - 1} "
                        "has an accepted attempt."
                    )
                resolved_upstream_hash = predecessor["accepted_artifact_hash"]
                if (
                    upstream_artifact_hash is not None
                    and upstream_artifact_hash != resolved_upstream_hash
                ):
                    raise StateTransitionError("The supplied upstream artifact hash is stale.")

            def matching_attempt(row: sqlite3.Row) -> bool:
                return (
                    int(row["seed"]) == seed
                    and int(row["variation_index"]) == variation_index
                    and _canonical_json(json.loads(row["parameters_json"])) == parameters_json
                    and row["upstream_artifact_hash"] == resolved_upstream_hash
                )

            if attempt_number is not None:
                existing = connection.execute(
                    """
                    SELECT * FROM chunk_attempts
                    WHERE job_id = ? AND scene_id = ? AND revision = ?
                      AND chunk_index = ? AND attempt_number = ?
                    """,
                    (job_id, scene_id, revision, chunk_index, attempt_number),
                ).fetchone()
                if existing is not None:
                    if not matching_attempt(existing):
                        raise StateTransitionError(
                            f"Chunk attempt {attempt_number} already exists with "
                            "different immutable inputs."
                        )
                    return self._chunk_attempt_record(existing)
                state_placeholders = ",".join("?" for _ in _ACTIVE_CHUNK_STATES)
                active = connection.execute(
                    f"""
                    SELECT 1 FROM chunk_attempts
                    WHERE job_id = ? AND scene_id = ? AND revision = ?
                      AND chunk_index = ? AND state IN ({state_placeholders})
                    LIMIT 1
                    """,
                    (
                        job_id,
                        scene_id,
                        revision,
                        chunk_index,
                        *(state.value for state in _ACTIVE_CHUNK_STATES),
                    ),
                ).fetchone()
                if active is not None:
                    raise StateTransitionError(
                        f"Chunk {chunk_index} already has an active attempt."
                    )
                maximum = connection.execute(
                    """
                    SELECT COALESCE(MAX(attempt_number), 0) AS maximum
                    FROM chunk_attempts
                    WHERE job_id = ? AND scene_id = ? AND revision = ?
                      AND chunk_index = ?
                    """,
                    (job_id, scene_id, revision, chunk_index),
                ).fetchone()
                expected_attempt = int(maximum["maximum"]) + 1
                if attempt_number != expected_attempt:
                    raise StateTransitionError(
                        f"The next chunk attempt number must be {expected_attempt}."
                    )
            else:
                state_placeholders = ",".join("?" for _ in _ACTIVE_CHUNK_STATES)
                active = connection.execute(
                    f"""
                    SELECT * FROM chunk_attempts
                    WHERE job_id = ? AND scene_id = ? AND revision = ?
                      AND chunk_index = ? AND state IN ({state_placeholders})
                    ORDER BY attempt_number DESC LIMIT 1
                    """,
                    (
                        job_id,
                        scene_id,
                        revision,
                        chunk_index,
                        *(state.value for state in _ACTIVE_CHUNK_STATES),
                    ),
                ).fetchone()
                if active is not None:
                    if not matching_attempt(active):
                        raise StateTransitionError(
                            f"Chunk {chunk_index} already has a different active attempt."
                        )
                    return self._chunk_attempt_record(active)
                maximum = connection.execute(
                    """
                    SELECT COALESCE(MAX(attempt_number), 0) AS maximum
                    FROM chunk_attempts
                    WHERE job_id = ? AND scene_id = ? AND revision = ?
                      AND chunk_index = ?
                    """,
                    (job_id, scene_id, revision, chunk_index),
                ).fetchone()
                attempt_number = int(maximum["maximum"]) + 1

            chunk_state = ChunkState(chunk["state"])
            if chunk_state not in {
                ChunkState.READY,
                ChunkState.FAILED_RETRYABLE,
                ChunkState.INVALIDATED,
                ChunkState.COMPLETE,
            }:
                raise StateTransitionError(
                    f"Chunk {chunk_index} cannot start while it is {chunk_state.value}."
                )
            now = _utc_now()
            connection.execute(
                """
                INSERT INTO chunk_attempts (
                    job_id, scene_id, revision, chunk_index, attempt_number,
                    variation_index, state, seed, parameters_json,
                    upstream_artifact_hash, artifact_manifest_path, artifact_hash,
                    video_path, result_json, error, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, '{}',
                          NULL, ?, ?)
                """,
                (
                    job_id,
                    scene_id,
                    revision,
                    chunk_index,
                    attempt_number,
                    variation_index,
                    ChunkState.GENERATING_STAGE1,
                    str(seed),
                    parameters_json,
                    resolved_upstream_hash,
                    now,
                    now,
                ),
            )
            connection.execute(
                """
                UPDATE scene_chunks
                SET state = ?, error = NULL, updated_at = ?
                WHERE job_id = ? AND scene_id = ? AND revision = ?
                  AND chunk_index = ? AND accepted_attempt_number IS NULL
                """,
                (
                    ChunkState.GENERATING_STAGE1,
                    now,
                    job_id,
                    scene_id,
                    revision,
                    chunk_index,
                ),
            )
            row = connection.execute(
                """
                SELECT * FROM chunk_attempts
                WHERE job_id = ? AND scene_id = ? AND revision = ?
                  AND chunk_index = ? AND attempt_number = ?
                """,
                (job_id, scene_id, revision, chunk_index, attempt_number),
            ).fetchone()
        return self._chunk_attempt_record(row)

    def update_chunk_attempt(
        self,
        job_id: str,
        scene_id: int,
        revision: int,
        chunk_index: int,
        attempt_number: int,
        state: ChunkState,
        *,
        artifact_manifest_path: str | None = None,
        artifact_hash: str | None = None,
        video_path: str | None = None,
        result: Mapping[str, Any] | None = None,
        error: str | None = None,
    ) -> ChunkAttemptRecord:
        """Advance execution status without changing immutable attempt inputs."""
        _positive_revision(revision)
        _nonnegative_chunk_index(chunk_index)
        _positive_attempt_number(attempt_number)
        state = ChunkState(state)
        if state in {ChunkState.PLANNED, ChunkState.BLOCKED_UPSTREAM, ChunkState.READY}:
            raise StateTransitionError(f"{state.value} is not an attempt execution state.")
        result_json = _canonical_json(result) if result is not None else None

        self.initialize()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if (
                state != ChunkState.CANCELLED
                and not self._continuation_work_is_active_connection(
                    connection,
                    job_id,
                    scene_id,
                    revision,
                )
            ):
                raise StateTransitionError(
                    "Cancelled or detached continuation work cannot commit a "
                    "late chunk-attempt update."
                )
            existing = connection.execute(
                """
                SELECT * FROM chunk_attempts
                WHERE job_id = ? AND scene_id = ? AND revision = ?
                  AND chunk_index = ? AND attempt_number = ?
                """,
                (job_id, scene_id, revision, chunk_index, attempt_number),
            ).fetchone()
            if existing is None:
                raise StateTransitionError(
                    f"Unknown chunk attempt {attempt_number} for chunk {chunk_index}."
                )
            selected = connection.execute(
                """
                SELECT accepted_attempt_number FROM scene_chunks
                WHERE job_id = ? AND scene_id = ? AND revision = ?
                  AND chunk_index = ?
                """,
                (job_id, scene_id, revision, chunk_index),
            ).fetchone()
            if selected is None:
                raise StateTransitionError(f"Unknown chunk {chunk_index}.")
            current_state = ChunkState(existing["state"])
            if (
                state == ChunkState.COMPLETE
                and artifact_hash is None
                and not existing["artifact_hash"]
            ):
                raise StateTransitionError(
                    "A complete chunk attempt requires an artifact hash."
                )
            if current_state in _FINAL_ATTEMPT_STATES:
                changed_output = (
                    (
                        artifact_manifest_path is not None
                        and artifact_manifest_path != existing["artifact_manifest_path"]
                    )
                    or (
                        artifact_hash is not None
                        and artifact_hash != existing["artifact_hash"]
                    )
                    or (video_path is not None and video_path != existing["video_path"])
                    or (
                        result_json is not None
                        and result_json
                        != _canonical_json(json.loads(existing["result_json"]))
                    )
                )
                if changed_output:
                    raise StateTransitionError(
                        f"Finalized chunk attempt {attempt_number} has immutable outputs."
                    )
                if current_state == state:
                    return self._chunk_attempt_record(existing)
                if (
                    selected["accepted_attempt_number"] is not None
                    and int(selected["accepted_attempt_number"]) == attempt_number
                ):
                    raise StateTransitionError(
                        "An accepted chunk attempt must be invalidated transactionally."
                    )
                if state not in {
                    ChunkState.STALE_UPSTREAM,
                    ChunkState.INVALIDATED,
                    ChunkState.CANCELLED,
                }:
                    raise StateTransitionError(
                        f"Finalized chunk attempt {attempt_number} cannot return to "
                        f"{state.value}."
                    )
            now = _utc_now()
            connection.execute(
                """
                UPDATE chunk_attempts
                SET state = ?,
                    artifact_manifest_path = COALESCE(?, artifact_manifest_path),
                    artifact_hash = COALESCE(?, artifact_hash),
                    video_path = COALESCE(?, video_path),
                    result_json = COALESCE(?, result_json),
                    error = ?,
                    updated_at = ?
                WHERE job_id = ? AND scene_id = ? AND revision = ?
                  AND chunk_index = ? AND attempt_number = ?
                """,
                (
                    state,
                    artifact_manifest_path,
                    artifact_hash,
                    video_path,
                    result_json,
                    error,
                    now,
                    job_id,
                    scene_id,
                    revision,
                    chunk_index,
                    attempt_number,
                ),
            )
            if (
                (
                    selected["accepted_attempt_number"] is None
                    or int(selected["accepted_attempt_number"]) == attempt_number
                )
            ):
                connection.execute(
                    """
                    UPDATE scene_chunks SET state = ?, error = ?, updated_at = ?
                    WHERE job_id = ? AND scene_id = ? AND revision = ?
                      AND chunk_index = ?
                    """,
                    (state, error, now, job_id, scene_id, revision, chunk_index),
                )
            row = connection.execute(
                """
                SELECT * FROM chunk_attempts
                WHERE job_id = ? AND scene_id = ? AND revision = ?
                  AND chunk_index = ? AND attempt_number = ?
                """,
                (job_id, scene_id, revision, chunk_index, attempt_number),
            ).fetchone()
        return self._chunk_attempt_record(row)

    def chunk_attempts(
        self,
        job_id: str,
        scene_id: int,
        revision: int,
        chunk_index: int,
    ) -> tuple[ChunkAttemptRecord, ...]:
        _positive_revision(revision)
        _nonnegative_chunk_index(chunk_index)
        self.initialize()
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM chunk_attempts
                WHERE job_id = ? AND scene_id = ? AND revision = ?
                  AND chunk_index = ?
                ORDER BY attempt_number
                """,
                (job_id, scene_id, revision, chunk_index),
            ).fetchall()
        return tuple(self._chunk_attempt_record(row) for row in rows)

    @staticmethod
    def _invalidate_chunks_from_connection(
        connection: sqlite3.Connection,
        job_id: str,
        scene_id: int,
        revision: int,
        chunk_index: int,
        reason: str,
    ) -> list[int]:
        rows = connection.execute(
            """
            SELECT chunk_index FROM scene_chunks
            WHERE job_id = ? AND scene_id = ? AND revision = ?
              AND chunk_index >= ?
            ORDER BY chunk_index
            """,
            (job_id, scene_id, revision, chunk_index),
        ).fetchall()
        if not rows:
            raise StateTransitionError(
                f"Unknown chunk {chunk_index} for {job_id} scene {scene_id} "
                f"revision {revision}."
            )
        now = _utc_now()
        indices = [int(row["chunk_index"]) for row in rows]
        predecessor_ready = chunk_index == 0
        if chunk_index > 0:
            predecessor = connection.execute(
                """
                SELECT state, accepted_attempt_number, accepted_artifact_hash
                FROM scene_chunks
                WHERE job_id = ? AND scene_id = ? AND revision = ?
                  AND chunk_index = ?
                """,
                (job_id, scene_id, revision, chunk_index - 1),
            ).fetchone()
            predecessor_ready = bool(
                predecessor is not None
                and ChunkState(predecessor["state"]) == ChunkState.COMPLETE
                and predecessor["accepted_attempt_number"] is not None
                and predecessor["accepted_artifact_hash"]
            )
        for index in indices:
            chunk_state = (
                ChunkState.READY
                if index == chunk_index and predecessor_ready
                else ChunkState.STALE_UPSTREAM
            )
            attempt_state = (
                ChunkState.INVALIDATED
                if index == chunk_index
                else ChunkState.STALE_UPSTREAM
            )
            connection.execute(
                """
                UPDATE scene_chunks
                SET state = ?, accepted_attempt_number = NULL,
                    accepted_artifact_hash = NULL, error = ?, updated_at = ?
                WHERE job_id = ? AND scene_id = ? AND revision = ?
                  AND chunk_index = ?
                """,
                (
                    chunk_state,
                    reason,
                    now,
                    job_id,
                    scene_id,
                    revision,
                    index,
                ),
            )
            connection.execute(
                """
                UPDATE chunk_attempts
                SET state = ?, error = ?, updated_at = ?
                WHERE job_id = ? AND scene_id = ? AND revision = ?
                  AND chunk_index = ? AND state != ?
                """,
                (
                    attempt_state,
                    reason,
                    now,
                    job_id,
                    scene_id,
                    revision,
                    index,
                    ChunkState.CANCELLED,
                ),
            )
        return indices

    def invalidate_chunks_from(
        self,
        job_id: str,
        scene_id: int,
        revision: int,
        chunk_index: int,
        *,
        reason: str = "Invalidated by an upstream continuation change.",
    ) -> list[int]:
        """Invalidate one chunk and every dependent descendant."""
        _positive_revision(revision)
        _nonnegative_chunk_index(chunk_index)
        self.initialize()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            job = connection.execute(
                "SELECT status FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if job is None:
                raise StateTransitionError(f"Unknown job {job_id}.")
            if not self._continuation_work_is_active_connection(
                connection,
                job_id,
                scene_id,
                revision,
            ):
                raise StateTransitionError(
                    "A cancelled job or scene revision cannot invalidate chunks."
                )
            return self._invalidate_chunks_from_connection(
                connection,
                job_id,
                scene_id,
                revision,
                chunk_index,
                reason,
            )

    def select_chunk_attempt(
        self,
        job_id: str,
        scene_id: int,
        revision: int,
        chunk_index: int,
        attempt_number: int,
        *,
        artifact_hash: str | None = None,
    ) -> SceneChunkRecord:
        """Select a completed attempt and release its immediate dependent chunk."""
        _positive_revision(revision)
        _nonnegative_chunk_index(chunk_index)
        _positive_attempt_number(attempt_number)
        self.initialize()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            job = connection.execute(
                "SELECT status FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if job is None:
                raise StateTransitionError(f"Unknown job {job_id}.")
            if not self._continuation_work_is_active_connection(
                connection,
                job_id,
                scene_id,
                revision,
            ):
                raise StateTransitionError(
                    "A cancelled job or scene revision cannot accept chunk attempts."
                )
            attempt = connection.execute(
                """
                SELECT * FROM chunk_attempts
                WHERE job_id = ? AND scene_id = ? AND revision = ?
                  AND chunk_index = ? AND attempt_number = ?
                """,
                (job_id, scene_id, revision, chunk_index, attempt_number),
            ).fetchone()
            if attempt is None:
                raise StateTransitionError(
                    f"Unknown chunk attempt {attempt_number} for chunk {chunk_index}."
                )
            if ChunkState(attempt["state"]) != ChunkState.COMPLETE:
                raise StateTransitionError("Only a complete chunk attempt can be accepted.")
            resolved_hash = attempt["artifact_hash"]
            if not resolved_hash:
                raise StateTransitionError(
                    "A complete chunk attempt needs an artifact hash before selection."
                )
            if artifact_hash is not None and artifact_hash != resolved_hash:
                raise StateTransitionError("The accepted artifact hash does not match the attempt.")
            if chunk_index > 0:
                predecessor = connection.execute(
                    """
                    SELECT accepted_artifact_hash FROM scene_chunks
                    WHERE job_id = ? AND scene_id = ? AND revision = ?
                      AND chunk_index = ?
                    """,
                    (job_id, scene_id, revision, chunk_index - 1),
                ).fetchone()
                if (
                    predecessor is None
                    or predecessor["accepted_artifact_hash"]
                    != attempt["upstream_artifact_hash"]
                ):
                    raise StateTransitionError(
                        "The attempt was generated from a stale upstream artifact."
                    )
            current = connection.execute(
                """
                SELECT state, accepted_attempt_number, accepted_artifact_hash
                FROM scene_chunks
                WHERE job_id = ? AND scene_id = ? AND revision = ?
                  AND chunk_index = ?
                """,
                (job_id, scene_id, revision, chunk_index),
            ).fetchone()
            if current is None:
                raise StateTransitionError(f"Unknown chunk {chunk_index}.")
            if ChunkState(current["state"]) == ChunkState.CANCELLED:
                raise StateTransitionError("A cancelled chunk cannot accept an attempt.")
            if (
                current["accepted_attempt_number"] is not None
                and int(current["accepted_attempt_number"]) == attempt_number
                and current["accepted_artifact_hash"] == resolved_hash
            ):
                row = connection.execute(
                    """
                    SELECT * FROM scene_chunks
                    WHERE job_id = ? AND scene_id = ? AND revision = ?
                      AND chunk_index = ?
                    """,
                    (job_id, scene_id, revision, chunk_index),
                ).fetchone()
                return self._chunk_record(row)
            if current["accepted_attempt_number"] is not None:
                descendant = connection.execute(
                    """
                    SELECT 1 FROM scene_chunks
                    WHERE job_id = ? AND scene_id = ? AND revision = ?
                      AND chunk_index = ?
                    """,
                    (job_id, scene_id, revision, chunk_index + 1),
                ).fetchone()
                if descendant is not None:
                    self._invalidate_chunks_from_connection(
                        connection,
                        job_id,
                        scene_id,
                        revision,
                        chunk_index + 1,
                        "Invalidated because the selected upstream attempt changed.",
                    )
            now = _utc_now()
            connection.execute(
                """
                UPDATE scene_chunks
                SET state = ?, accepted_attempt_number = ?,
                    accepted_artifact_hash = ?, error = NULL, updated_at = ?
                WHERE job_id = ? AND scene_id = ? AND revision = ?
                  AND chunk_index = ?
                """,
                (
                    ChunkState.COMPLETE,
                    attempt_number,
                    resolved_hash,
                    now,
                    job_id,
                    scene_id,
                    revision,
                    chunk_index,
                ),
            )
            connection.execute(
                """
                UPDATE scene_chunks
                SET state = ?, error = NULL, updated_at = ?
                WHERE job_id = ? AND scene_id = ? AND revision = ?
                  AND chunk_index = ? AND accepted_attempt_number IS NULL
                  AND state IN (?, ?, ?)
                """,
                (
                    ChunkState.READY,
                    now,
                    job_id,
                    scene_id,
                    revision,
                    chunk_index + 1,
                    ChunkState.BLOCKED_UPSTREAM,
                    ChunkState.STALE_UPSTREAM,
                    ChunkState.INVALIDATED,
                ),
            )
            row = connection.execute(
                """
                SELECT * FROM scene_chunks
                WHERE job_id = ? AND scene_id = ? AND revision = ?
                  AND chunk_index = ?
                """,
                (job_id, scene_id, revision, chunk_index),
            ).fetchone()
        return self._chunk_record(row)

    def chunk_progress(
        self,
        job_id: str,
        scene_id: int,
        revision: int,
    ) -> ChunkProgress:
        chunks = self.chunk_records(job_id, scene_id, revision)
        complete = sum(
            chunk.state == ChunkState.COMPLETE
            and chunk.accepted_attempt_number is not None
            for chunk in chunks
        )
        ready = sum(
            chunk.state in {ChunkState.READY, ChunkState.FAILED_RETRYABLE}
            for chunk in chunks
        )
        active = sum(chunk.state in _ACTIVE_CHUNK_STATES for chunk in chunks)
        blocked = sum(
            chunk.state in {ChunkState.BLOCKED_UPSTREAM, ChunkState.STALE_UPSTREAM}
            for chunk in chunks
        )
        failed = sum(
            chunk.state in {ChunkState.FAILED_RETRYABLE, ChunkState.FAILED_TERMINAL}
            for chunk in chunks
        )
        invalidated = sum(
            chunk.state in {ChunkState.INVALIDATED, ChunkState.STALE_UPSTREAM}
            for chunk in chunks
        )
        cancelled = sum(chunk.state == ChunkState.CANCELLED for chunk in chunks)
        next_chunk = next(
            (
                chunk.chunk_index
                for chunk in chunks
                if chunk.state in {ChunkState.READY, ChunkState.FAILED_RETRYABLE}
            ),
            None,
        )
        return ChunkProgress(
            total_count=len(chunks),
            complete_count=complete,
            ready_count=ready,
            active_count=active,
            blocked_count=blocked,
            failed_count=failed,
            invalidated_count=invalidated,
            cancelled_count=cancelled,
            next_chunk_index=next_chunk,
        )

    def scene_records(self, job_id: str) -> tuple[SceneRecord, ...]:
        self.initialize()
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT job_id, scene_id, state, frame_path, video_path, error,
                       t2i_attempts, i2v_attempts, prompt_id, prompt_stage,
                       include_in_manual_final
                FROM scenes WHERE job_id = ? ORDER BY scene_id
                """,
                (job_id,),
            ).fetchall()
        return tuple(
            SceneRecord(
                job_id=row["job_id"],
                scene_id=int(row["scene_id"]),
                state=SceneState(row["state"]),
                frame_path=row["frame_path"],
                video_path=row["video_path"],
                error=row["error"],
                t2i_attempts=int(row["t2i_attempts"]),
                i2v_attempts=int(row["i2v_attempts"]),
                prompt_id=row["prompt_id"],
                prompt_stage=row["prompt_stage"],
                include_in_manual_final=bool(row["include_in_manual_final"]),
            )
            for row in rows
        )

    def scene_states(self, job_id: str) -> dict[int, SceneState]:
        self.initialize()
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT scene_id, state FROM scenes WHERE job_id = ? ORDER BY scene_id", (job_id,)
            ).fetchall()
        return {int(row["scene_id"]): SceneState(row["state"]) for row in rows}

    def requeue_unfinished_scenes(self, job_id: str) -> list[int]:
        """Make failed, cancelled, and interrupted scenes runnable without touching successes."""
        self.initialize()
        resumable = (SceneState.PENDING, SceneState.RUNNING, SceneState.FAILED, SceneState.CANCELLED)
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT scene_id FROM scenes WHERE job_id = ? AND state != ? ORDER BY scene_id",
                (job_id, SceneState.SUCCEEDED),
            ).fetchall()
            connection.execute(
                """
                UPDATE scenes
                SET state = ?, error = NULL, updated_at = ?
                WHERE job_id = ? AND state IN (?, ?, ?, ?)
                """,
                (SceneState.PENDING, _utc_now(), job_id, *resumable),
            )
        return [int(row["scene_id"]) for row in rows]

    def retry_job(self, job_id: str) -> list[int]:
        """Atomically resume a saved job without replacing completed scenes."""
        self.initialize()
        resumable = (
            SceneState.PENDING,
            SceneState.RUNNING,
            SceneState.FAILED,
            SceneState.CANCELLED,
        )
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            exists = connection.execute(
                "SELECT status FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if not exists:
                raise StateTransitionError(f"Unknown job {job_id}.")
            pipeline = connection.execute(
                """
                SELECT state, job_id, active_scene_id
                FROM pipeline_state WHERE singleton = 1
                """
            ).fetchone()
            if pipeline["job_id"] not in {None, job_id}:
                raise StateTransitionError(
                    f"Cannot retry {job_id}; active job {pipeline['job_id']} still "
                    "owns the pipeline."
                )
            rows = connection.execute(
                """
                SELECT scene_id, state, frame_path, prompt_id, prompt_stage,
                       t2i_attempts, i2v_attempts
                FROM scenes
                WHERE job_id = ? AND state != ?
                ORDER BY scene_id
                """,
                (job_id, SceneState.SUCCEEDED),
            ).fetchall()
            if not rows:
                raise StateTransitionError(f"Job {job_id} has no unfinished scenes.")
            for row in rows:
                expected_pipeline_state = {
                    "t2i": PipelineState.RUNNING_T2I,
                    "i2v_legacy": PipelineState.RUNNING_I2V,
                    "i2v_continuation": PipelineState.RUNNING_I2V,
                }.get(row["prompt_stage"])
                genuinely_interrupted = bool(
                    SceneState(row["state"]) == SceneState.RUNNING
                    and row["prompt_id"]
                    and expected_pipeline_state is not None
                    and pipeline["job_id"] == job_id
                    and pipeline["active_scene_id"] == row["scene_id"]
                    and PipelineState(pipeline["state"])
                    in {expected_pipeline_state, PipelineState.ERROR}
                )
                if genuinely_interrupted:
                    continue

                stage = row["prompt_stage"]
                frame_path = row["frame_path"]
                try:
                    frame_exists = bool(
                        isinstance(frame_path, str)
                        and frame_path
                        and Path(frame_path).is_file()
                    )
                except OSError:
                    frame_exists = False
                if not frame_exists:
                    stage = "t2i"
                elif stage not in {"t2i", "i2v_legacy", "i2v_continuation"}:
                    if int(row["t2i_attempts"]) > 0 and not row["frame_path"]:
                        stage = "t2i"
                    elif int(row["i2v_attempts"]) > 0:
                        has_plan = connection.execute(
                            """
                            SELECT 1 FROM continuation_plans
                            WHERE job_id = ? AND scene_id = ? AND revision = 1
                            """,
                            (job_id, row["scene_id"]),
                        ).fetchone()
                        stage = (
                            "i2v_continuation"
                            if has_plan is not None
                            else "i2v_legacy"
                        )
                t2i_attempts = int(row["t2i_attempts"])
                i2v_attempts = int(row["i2v_attempts"])
                if not frame_exists:
                    t2i_attempts = 0
                    i2v_attempts = 0
                elif stage == "t2i":
                    t2i_attempts = 0
                elif stage == "i2v_legacy":
                    i2v_attempts = 0
                connection.execute(
                    """
                    UPDATE scenes
                    SET t2i_attempts = ?, i2v_attempts = ?,
                        prompt_id = NULL, prompt_stage = ?
                    WHERE job_id = ? AND scene_id = ?
                    """,
                    (
                        t2i_attempts,
                        i2v_attempts,
                        stage,
                        job_id,
                        row["scene_id"],
                    ),
                )
            connection.execute(
                """
                UPDATE scenes
                SET state = ?, error = NULL, updated_at = ?
                WHERE job_id = ? AND state IN (?, ?, ?, ?)
                """,
                (SceneState.PENDING, _utc_now(), job_id, *resumable),
            )
            now = _utc_now()
            unfinished_scene_ids = {int(row["scene_id"]) for row in rows}
            if unfinished_scene_ids:
                placeholders = ",".join("?" for _ in unfinished_scene_ids)
                # An explicit retry starts a new bounded retry epoch for failed
                # chunk attempts. Cancelled and invalidated attempts remain as
                # immutable audit history and never consume the new epoch.
                connection.execute(
                    f"""
                    UPDATE chunk_attempts
                    SET state = ?, error = CASE
                        WHEN error IS NULL OR TRIM(error) = '' THEN ?
                        ELSE error || ' | ' || ?
                    END, updated_at = ?
                    WHERE job_id = ? AND revision = 1
                      AND scene_id IN ({placeholders})
                      AND state IN (?, ?)
                    """,
                    (
                        ChunkState.INVALIDATED,
                        "Superseded by an explicit saved-job retry.",
                        "Superseded by an explicit saved-job retry.",
                        now,
                        job_id,
                        *sorted(unfinished_scene_ids),
                        ChunkState.FAILED_RETRYABLE,
                        ChunkState.FAILED_TERMINAL,
                    ),
                )
                for scene_id in sorted(unfinished_scene_ids):
                    chunks = connection.execute(
                        """
                        SELECT chunk_index, state, accepted_attempt_number
                        FROM scene_chunks
                        WHERE job_id = ? AND scene_id = ? AND revision = 1
                        ORDER BY chunk_index
                        """,
                        (job_id, scene_id),
                    ).fetchall()
                    first_incomplete = next(
                        (
                            int(chunk["chunk_index"])
                            for chunk in chunks
                            if not (
                                ChunkState(chunk["state"]) == ChunkState.COMPLETE
                                and chunk["accepted_attempt_number"] is not None
                            )
                        ),
                        None,
                    )
                    if first_incomplete is None:
                        continue
                    for chunk in chunks:
                        chunk_index = int(chunk["chunk_index"])
                        if (
                            ChunkState(chunk["state"]) == ChunkState.COMPLETE
                            and chunk["accepted_attempt_number"] is not None
                        ):
                            continue
                        active_placeholders = ",".join(
                            "?" for _ in _ACTIVE_CHUNK_STATES
                        )
                        active = connection.execute(
                            f"""
                            SELECT state FROM chunk_attempts
                            WHERE job_id = ? AND scene_id = ? AND revision = 1
                              AND chunk_index = ?
                              AND state IN ({active_placeholders})
                            ORDER BY attempt_number DESC LIMIT 1
                            """,
                            (
                                job_id,
                                scene_id,
                                chunk_index,
                                *(state.value for state in _ACTIVE_CHUNK_STATES),
                            ),
                        ).fetchone()
                        if active is not None:
                            restored_state = ChunkState(active["state"])
                        elif chunk_index == first_incomplete:
                            restored_state = ChunkState.READY
                        else:
                            restored_state = ChunkState.BLOCKED_UPSTREAM
                        connection.execute(
                            """
                            UPDATE scene_chunks
                            SET state = ?, error = NULL, updated_at = ?
                            WHERE job_id = ? AND scene_id = ? AND revision = 1
                              AND chunk_index = ?
                            """,
                            (
                                restored_state,
                                now,
                                job_id,
                                scene_id,
                                chunk_index,
                            ),
                        )
            connection.execute(
                "UPDATE jobs SET status = ?, updated_at = ? WHERE job_id = ?",
                (JobState.QUEUED, now, job_id),
            )
            connection.execute(
                """
                UPDATE pipeline_state
                SET state = ?, job_id = ?, active_scene_id = NULL, error = NULL, updated_at = ?
                WHERE singleton = 1
                """,
                (PipelineState.DOWNLOADING_ASSETS, job_id, now),
            )
        return [int(row["scene_id"]) for row in rows]

    @classmethod
    def _invalidate_continuation_revision_for_rerun(
        cls,
        connection: sqlite3.Connection,
        job_id: str,
        scene_id: int,
        revision: int,
        reason: str,
    ) -> None:
        first_chunk = connection.execute(
            """
            SELECT MIN(chunk_index) AS first_chunk
            FROM scene_chunks
            WHERE job_id = ? AND scene_id = ? AND revision = ?
            """,
            (job_id, scene_id, revision),
        ).fetchone()
        if first_chunk is None or first_chunk["first_chunk"] is None:
            return
        cls._invalidate_chunks_from_connection(
            connection,
            job_id,
            scene_id,
            revision,
            int(first_chunk["first_chunk"]),
            reason,
        )

    def requeue_i2v_for_job(self, job_id: str) -> list[int]:
        """Invalidate every scene clip while preserving deterministic T2I frames."""
        self.initialize()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            exists = connection.execute(
                "SELECT 1 FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if not exists:
                raise StateTransitionError(f"Unknown job {job_id}.")
            rows = connection.execute(
                "SELECT scene_id FROM scenes WHERE job_id = ? ORDER BY scene_id",
                (job_id,),
            ).fetchall()
            now = _utc_now()
            for row in rows:
                self._invalidate_continuation_revision_for_rerun(
                    connection,
                    job_id,
                    int(row["scene_id"]),
                    1,
                    "Invalidated by an explicit full-job I2V rerun.",
                )
            connection.execute(
                """
                UPDATE scenes
                SET state = ?, video_path = NULL, error = NULL,
                    i2v_attempts = 0, prompt_id = NULL,
                    prompt_stage = NULL, updated_at = ?
                WHERE job_id = ?
                """,
                (SceneState.PENDING, now, job_id),
            )
            connection.execute(
                """
                UPDATE scene_revisions
                SET state = ?, video_path = NULL, error = NULL, updated_at = ?
                WHERE job_id = ? AND revision = 1
                """,
                (SceneState.PENDING, now, job_id),
            )
            connection.execute(
                """
                UPDATE pipeline_state
                SET state = ?, job_id = ?, active_scene_id = NULL,
                    error = NULL, updated_at = ?
                WHERE singleton = 1
                """,
                (PipelineState.DOWNLOADING_ASSETS, job_id, now),
            )
        return [int(row["scene_id"]) for row in rows]

    def requeue_scene_i2v(self, job_id: str, scene_id: int) -> None:
        """Reset one interrupted I2V scene without changing other scene artifacts."""
        self.initialize()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            exists = connection.execute(
                "SELECT 1 FROM scenes WHERE job_id = ? AND scene_id = ?",
                (job_id, scene_id),
            ).fetchone()
            if not exists:
                raise StateTransitionError(
                    f"Unknown scene {scene_id} for job {job_id}."
                )
            now = _utc_now()
            self._invalidate_continuation_revision_for_rerun(
                connection,
                job_id,
                scene_id,
                1,
                "Invalidated by an explicit scene I2V rerun.",
            )
            connection.execute(
                """
                UPDATE scenes
                SET state = ?, video_path = NULL, error = NULL,
                    i2v_attempts = 0, prompt_id = NULL,
                    prompt_stage = NULL, updated_at = ?
                WHERE job_id = ? AND scene_id = ?
                """,
                (SceneState.PENDING, now, job_id, scene_id),
            )
            connection.execute(
                """
                UPDATE scene_revisions
                SET state = ?, video_path = NULL, error = NULL, updated_at = ?
                WHERE job_id = ? AND scene_id = ? AND revision = 1
                """,
                (SceneState.PENDING, now, job_id, scene_id),
            )
            connection.execute(
                """
                UPDATE pipeline_state
                SET state = ?, job_id = ?, active_scene_id = NULL,
                    error = NULL, updated_at = ?
                WHERE singleton = 1
                """,
                (PipelineState.DOWNLOADING_ASSETS, job_id, now),
            )

    def abandon_job(
        self,
        job_id: str,
        *,
        reason: str = "Abandoned by the user from the one-click launcher.",
    ) -> list[int]:
        """Atomically release a saved job while preserving its audit history."""
        self.initialize()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT state, job_id FROM pipeline_state WHERE singleton = 1"
            ).fetchone()
            if current["job_id"] != job_id:
                raise StateTransitionError(
                    f"Cannot abandon job {job_id}; the active job is {current['job_id'] or 'none'}."
                )
            rows = connection.execute(
                "SELECT scene_id FROM scenes WHERE job_id = ? AND state != ? ORDER BY scene_id",
                (job_id, SceneState.SUCCEEDED),
            ).fetchall()
            now = _utc_now()
            nonreviewable_states = (
                QcCandidateState.PENDING_GENERATION,
                QcCandidateState.GENERATING,
                QcCandidateState.PENDING_QC,
                QcCandidateState.QC_RUNNING,
                QcCandidateState.PASS_PENDING_HUMAN,
                QcCandidateState.HOLD_FOR_REVIEW,
            )
            candidate_placeholders = ",".join("?" for _ in nonreviewable_states)
            connection.execute(
                f"""
                UPDATE qc_candidates
                SET state = ?, next_action = ?, updated_at = ?
                WHERE job_id = ? AND state IN ({candidate_placeholders})
                """,
                (
                    QcCandidateState.SUPERSEDED.value,
                    "job_cancelled",
                    now,
                    job_id,
                    *(state.value for state in nonreviewable_states),
                ),
            )
            active_placeholders = ",".join("?" for _ in _ACTIVE_CHUNK_STATES)
            connection.execute(
                f"""
                UPDATE chunk_attempts
                SET state = ?,
                    error = CASE
                        WHEN error IS NULL OR TRIM(error) = '' THEN ?
                        ELSE error || ' | ' || ?
                    END,
                    updated_at = ?
                WHERE job_id = ? AND state IN ({active_placeholders})
                """,
                (
                    ChunkState.CANCELLED,
                    reason,
                    reason,
                    now,
                    job_id,
                    *(state.value for state in _ACTIVE_CHUNK_STATES),
                ),
            )
            connection.execute(
                """
                UPDATE scene_chunks
                SET state = ?,
                    error = CASE
                        WHEN error IS NULL OR TRIM(error) = '' THEN ?
                        ELSE error || ' | ' || ?
                    END,
                    updated_at = ?
                WHERE job_id = ?
                  AND NOT (state = ? AND accepted_attempt_number IS NOT NULL)
                """,
                (
                    ChunkState.CANCELLED,
                    reason,
                    reason,
                    now,
                    job_id,
                    ChunkState.COMPLETE,
                ),
            )
            connection.execute(
                """
                UPDATE scenes
                SET state = ?,
                    error = CASE
                        WHEN error IS NULL OR TRIM(error) = '' THEN ?
                        ELSE error || ' | ' || ?
                    END,
                    prompt_id = NULL,
                    prompt_stage = NULL,
                    updated_at = ?
                WHERE job_id = ? AND state != ?
                """,
                (
                    SceneState.CANCELLED,
                    reason,
                    reason,
                    now,
                    job_id,
                    SceneState.SUCCEEDED,
                ),
            )
            connection.execute(
                """
                UPDATE pipeline_state
                SET state = ?, job_id = NULL, active_scene_id = NULL, error = NULL, updated_at = ?
                WHERE singleton = 1
                """,
                (PipelineState.IDLE, now),
            )
            connection.execute(
                "UPDATE jobs SET status = ?, updated_at = ? WHERE job_id = ?",
                (JobState.CANCELLED, now, job_id),
            )
        return [int(row["scene_id"]) for row in rows]
