"""Crash-safe state and resumable per-scene progress for the pipeline."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
import json
from pathlib import Path
import sqlite3
from typing import Iterator

from .contracts import JobPayload


class PipelineState(StrEnum):
    IDLE = "idle"
    WAITING_FOR_GROK = "waiting_for_grok"
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
                    created_at TEXT NOT NULL
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

    def claim_job(self, payload: JobPayload) -> None:
        """Atomically accept one new job only while the pipeline is available to poll."""
        self.initialize()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._claim_job(connection, payload)

    def claim_inbound_job(self, message_key: str, payload: JobPayload) -> bool:
        """Atomically de-duplicate an IMAP message and accept its job exactly once."""
        self.initialize()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute("SELECT state FROM pipeline_state WHERE singleton = 1").fetchone()
            if PipelineState(current["state"]) not in {PipelineState.IDLE, PipelineState.WAITING_FOR_GROK}:
                return False
            already_seen = connection.execute(
                "SELECT 1 FROM mail_messages WHERE message_key = ?", (message_key,)
            ).fetchone()
            already_accepted = connection.execute("SELECT 1 FROM jobs WHERE job_id = ?", (payload.job_id,)).fetchone()
            if already_seen or already_accepted:
                return False
            connection.execute(
                "INSERT INTO mail_messages (message_key, job_id, processed_at) VALUES (?, ?, ?)",
                (message_key, payload.job_id, _utc_now()),
            )
            self._claim_job(connection, payload)
            return True

    @staticmethod
    def _claim_job(connection: sqlite3.Connection, payload: JobPayload) -> None:
        current = connection.execute("SELECT state, job_id FROM pipeline_state WHERE singleton = 1").fetchone()
        if PipelineState(current["state"]) not in {PipelineState.IDLE, PipelineState.WAITING_FOR_GROK}:
            raise StateTransitionError(f"Cannot accept {payload.job_id}; pipeline is {current['state']}.")
        existing = connection.execute("SELECT 1 FROM jobs WHERE job_id = ?", (payload.job_id,)).fetchone()
        if existing:
            raise StateTransitionError(f"Job {payload.job_id} has already been accepted.")

        connection.execute(
            "INSERT INTO jobs (job_id, payload_json, created_at) VALUES (?, ?, ?)",
            (payload.job_id, json.dumps(payload.raw, sort_keys=True), _utc_now()),
        )
        connection.executemany(
            """
            INSERT INTO scenes (job_id, scene_id, state, frame_path, video_path, error, updated_at)
            VALUES (?, ?, ?, NULL, NULL, NULL, ?)
            """,
            [(payload.job_id, scene.scene_id, SceneState.PENDING, _utc_now()) for scene in payload.scenes],
        )
        connection.execute(
            """
            UPDATE pipeline_state
            SET state = ?, job_id = ?, active_scene_id = NULL, error = NULL, updated_at = ?
            WHERE singleton = 1
            """,
            (PipelineState.DOWNLOADING_ASSETS, payload.job_id, _utc_now()),
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
