"""Crash-safe state and resumable per-scene progress for the pipeline."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterator, Mapping, Sequence
from uuid import uuid4

from .contracts import JobPayload, job_content_fingerprint, parse_job_payload


class PipelineState(StrEnum):
    IDLE = "idle"
    WAITING_FOR_GROK = "waiting_for_grok"
    AWAITING_REVIEW = "awaiting_review"
    DOWNLOADING_ASSETS = "downloading_assets"
    RUNNING_T2I = "running_t2i"
    RUNNING_I2V = "running_i2v"
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


@dataclass(frozen=True)
class PipelineSnapshot:
    state: PipelineState
    job_id: str | None
    active_scene_id: int | None
    error: str | None


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


def _canonical_json(value: Mapping[str, Any]) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise StateTransitionError(f"Chunk metadata must be JSON serializable: {error}") from error


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
                CREATE INDEX IF NOT EXISTS idx_scene_chunks_state
                    ON scene_chunks(job_id, scene_id, revision, state, chunk_index);
                CREATE INDEX IF NOT EXISTS idx_chunk_attempts_state
                    ON chunk_attempts(job_id, scene_id, revision, chunk_index, state);
                """
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

    def queue_manual_final(self, job_id: str) -> ManualFinalRecord:
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

            selection = self._manual_final_selection(connection, job_id)
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
        for scene in scenes:
            revision = connection.execute(
                """
                SELECT revision, video_path
                FROM scene_revisions
                WHERE job_id = ? AND scene_id = ? AND state = ? AND video_path IS NOT NULL
                ORDER BY revision DESC LIMIT 1
                """,
                (job_id, scene["scene_id"], SceneState.SUCCEEDED),
            ).fetchone()
            if revision is None and (
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
