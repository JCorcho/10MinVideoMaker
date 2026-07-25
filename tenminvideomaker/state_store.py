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

from .contracts import JobPayload


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


class StateTransitionError(RuntimeError):
    """Raised when an operation would violate the single-job state machine."""


@dataclass(frozen=True)
class PipelineSnapshot:
    state: PipelineState
    job_id: str | None
    active_scene_id: int | None
    error: str | None


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


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


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
                    PRIMARY KEY (batch_id, position),
                    UNIQUE (batch_id, job_id, scene_id, revision),
                    FOREIGN KEY (batch_id) REFERENCES remake_batches(batch_id),
                    FOREIGN KEY (job_id, scene_id, revision)
                        REFERENCES scene_revisions(job_id, scene_id, revision)
                );
                """
            )
            scene_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(scenes)").fetchall()
            }
            for column, declaration in (
                ("t2i_attempts", "INTEGER NOT NULL DEFAULT 0"),
                ("i2v_attempts", "INTEGER NOT NULL DEFAULT 0"),
                ("prompt_id", "TEXT"),
            ):
                if column not in scene_columns:
                    connection.execute(f"ALTER TABLE scenes ADD COLUMN {column} {declaration}")
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
            current = connection.execute("SELECT job_id FROM pipeline_state WHERE singleton = 1").fetchone()
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
    ) -> bool:
        """Atomically de-duplicate an IMAP message and accept its job exactly once."""
        self.initialize()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute("SELECT state FROM pipeline_state WHERE singleton = 1").fetchone()
            if PipelineState(current["state"]) not in {PipelineState.IDLE, PipelineState.WAITING_FOR_GROK}:
                return False
            message_record = connection.execute(
                "SELECT job_id FROM mail_messages WHERE message_key = ?", (message_key,)
            ).fetchone()
            already_accepted = connection.execute("SELECT 1 FROM jobs WHERE job_id = ?", (payload.job_id,)).fetchone()
            if already_accepted or (
                message_record is not None and message_record["job_id"] is not None
            ):
                return False
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
            return True

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
            exists = connection.execute(
                "SELECT 1 FROM scenes WHERE job_id = ? AND scene_id = ?", (job_id, scene_id)
            ).fetchone()
            if not exists:
                raise StateTransitionError(f"Unknown scene {scene_id} for job {job_id}.")
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
    ) -> int:
        if pipeline_state not in {PipelineState.RUNNING_T2I, PipelineState.RUNNING_I2V}:
            raise StateTransitionError("Scene stages can only begin in running_t2i or running_i2v.")
        attempt_column = "t2i_attempts" if pipeline_state == PipelineState.RUNNING_T2I else "i2v_attempts"
        self.initialize()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                f"SELECT {attempt_column} FROM scenes WHERE job_id = ? AND scene_id = ?",
                (job_id, scene_id),
            ).fetchone()
            if row is None:
                raise StateTransitionError(f"Unknown scene {scene_id} for job {job_id}.")
            attempt = int(row[attempt_column]) + 1
            connection.execute(
                f"""
                UPDATE scenes
                SET state = ?, {attempt_column} = ?, prompt_id = ?, error = NULL, updated_at = ?
                WHERE job_id = ? AND scene_id = ?
                """,
                (SceneState.RUNNING, attempt, prompt_id, _utc_now(), job_id, scene_id),
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

    def set_scene_prompt_id(self, job_id: str, scene_id: int, prompt_id: str) -> None:
        self.initialize()
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE scenes SET prompt_id = ?, updated_at = ?
                WHERE job_id = ? AND scene_id = ?
                """,
                (prompt_id, _utc_now(), job_id, scene_id),
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
            if cursor.rowcount != 1:
                raise StateTransitionError(
                    f"Unknown revision {revision} for scene {scene_id} of job {job_id}."
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

    def remake_items(self, batch_id: str) -> tuple[RemakeItemRecord, ...]:
        self.initialize()
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT batch_id, position, job_id, scene_id, revision, state, error
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
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE remake_items SET state = ?, error = ?
                WHERE batch_id = ? AND position = ?
                """,
                (state, error, batch_id, position),
            )
            if cursor.rowcount != 1:
                raise StateTransitionError(
                    f"Unknown item {position} in remake batch {batch_id}."
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

    def scene_records(self, job_id: str) -> tuple[SceneRecord, ...]:
        self.initialize()
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT job_id, scene_id, state, frame_path, video_path, error,
                       t2i_attempts, i2v_attempts, prompt_id
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
                "SELECT 1 FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if not exists:
                raise StateTransitionError(f"Unknown job {job_id}.")
            rows = connection.execute(
                "SELECT scene_id FROM scenes WHERE job_id = ? AND state != ? ORDER BY scene_id",
                (job_id, SceneState.SUCCEEDED),
            ).fetchall()
            if not rows:
                raise StateTransitionError(f"Job {job_id} has no unfinished scenes.")
            connection.execute(
                """
                UPDATE scenes
                SET state = ?, error = NULL, prompt_id = NULL, updated_at = ?
                WHERE job_id = ? AND state IN (?, ?, ?, ?)
                """,
                (SceneState.PENDING, _utc_now(), job_id, *resumable),
            )
            connection.execute(
                """
                UPDATE pipeline_state
                SET state = ?, job_id = ?, active_scene_id = NULL, error = NULL, updated_at = ?
                WHERE singleton = 1
                """,
                (PipelineState.DOWNLOADING_ASSETS, job_id, _utc_now()),
            )
        return [int(row["scene_id"]) for row in rows]

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
            connection.execute(
                """
                UPDATE scenes
                SET state = ?, video_path = NULL, error = NULL,
                    i2v_attempts = 0, prompt_id = NULL, updated_at = ?
                WHERE job_id = ?
                """,
                (SceneState.PENDING, _utc_now(), job_id),
            )
            connection.execute(
                """
                UPDATE pipeline_state
                SET state = ?, job_id = ?, active_scene_id = NULL,
                    error = NULL, updated_at = ?
                WHERE singleton = 1
                """,
                (PipelineState.DOWNLOADING_ASSETS, job_id, _utc_now()),
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
            connection.execute(
                """
                UPDATE scenes
                SET state = ?, video_path = NULL, error = NULL,
                    i2v_attempts = 0, prompt_id = NULL, updated_at = ?
                WHERE job_id = ? AND scene_id = ?
                """,
                (SceneState.PENDING, _utc_now(), job_id, scene_id),
            )
            connection.execute(
                """
                UPDATE pipeline_state
                SET state = ?, job_id = ?, active_scene_id = NULL,
                    error = NULL, updated_at = ?
                WHERE singleton = 1
                """,
                (PipelineState.DOWNLOADING_ASSETS, job_id, _utc_now()),
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
            connection.execute(
                """
                UPDATE scenes
                SET state = ?,
                    error = CASE
                        WHEN error IS NULL OR TRIM(error) = '' THEN ?
                        ELSE error || ' | ' || ?
                    END,
                    prompt_id = NULL,
                    updated_at = ?
                WHERE job_id = ? AND state != ?
                """,
                (
                    SceneState.CANCELLED,
                    reason,
                    reason,
                    _utc_now(),
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
                (PipelineState.IDLE, _utc_now()),
            )
            connection.execute(
                "UPDATE jobs SET status = ?, updated_at = ? WHERE job_id = ?",
                (JobState.CANCELLED, _utc_now(), job_id),
            )
        return [int(row["scene_id"]) for row in rows]
