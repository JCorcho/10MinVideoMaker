"""D-drive storage layout and non-destructive legacy migration."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import sqlite3
from typing import Mapping


DEFAULT_STORAGE_ROOT = Path(r"D:\LTX_Supervisor_Storage")
LEGACY_OUTPUT_ROOT = Path(r"D:\output\10minfinals")
STORAGE_ENVIRONMENT_KEY = "TENMIN_STORAGE_ROOT"


class StorageError(RuntimeError):
    """Raised when project storage cannot be resolved or migrated safely."""


def write_json_atomic(path: str | Path, value: object) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return destination


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def configured_storage_root(
    environment: Mapping[str, str] | None = None,
) -> Path:
    values = os.environ if environment is None else environment
    configured = values.get(STORAGE_ENVIRONMENT_KEY, "").strip()
    root = _resolved(configured or DEFAULT_STORAGE_ROOT)
    if root.drive.casefold() != "d:":
        raise StorageError("10MinVideoMaker persistent storage must remain on the D: drive.")
    return root


@dataclass(frozen=True)
class StorageLayout:
    """Every persistent runtime path owned by the standalone supervisor."""

    root: Path

    @classmethod
    def configured(
        cls,
        environment: Mapping[str, str] | None = None,
    ) -> "StorageLayout":
        return cls(configured_storage_root(environment))

    @property
    def config_root(self) -> Path:
        return self.root / "config"

    @property
    def settings_path(self) -> Path:
        return self.config_root / "settings.env"

    @property
    def secrets_path(self) -> Path:
        return self.config_root / "secrets.json"

    @property
    def state_root(self) -> Path:
        return self.root / "state"

    @property
    def database_path(self) -> Path:
        return self.state_root / "pipeline.sqlite3"

    @property
    def asset_manifest_path(self) -> Path:
        return self.state_root / "assets.json"

    @property
    def jobs_root(self) -> Path:
        return self.root / "jobs"

    @property
    def finals_root(self) -> Path:
        return self.root / "finals"

    @property
    def logs_root(self) -> Path:
        return self.root / "logs"

    @property
    def temp_root(self) -> Path:
        return self.root / "temp"

    @property
    def migration_marker(self) -> Path:
        return self.state_root / "legacy-migration-v1.json"

    @property
    def instance_lock_path(self) -> Path:
        return self.state_root / "supervisor.lock"

    def ensure(self) -> None:
        for directory in (
            self.config_root,
            self.state_root,
            self.jobs_root,
            self.finals_root,
            self.logs_root,
            self.temp_root,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def job_root(self, job_id: str) -> Path:
        _validate_job_id(job_id)
        return self.jobs_root / job_id

    def source_payload_path(self, job_id: str) -> Path:
        return self.job_root(job_id) / "source" / "grok-job.json"

    def scene_root(self, job_id: str, scene_id: int) -> Path:
        _validate_scene_id(scene_id)
        return self.job_root(job_id) / "scenes" / f"scene_{scene_id:04d}"

    def revision_root(self, job_id: str, scene_id: int, revision: int) -> Path:
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            raise StorageError("revision must be a positive integer.")
        return self.scene_root(job_id, scene_id) / "revisions" / f"{revision:04d}"

    def scene_frame_path(self, job_id: str, scene_id: int, revision: int = 1) -> Path:
        return self.revision_root(job_id, scene_id, revision) / "frame.png"

    def scene_clip_path(self, job_id: str, scene_id: int, revision: int = 1) -> Path:
        return self.revision_root(job_id, scene_id, revision) / "video.mp4"

    def generation_manifest_path(
        self,
        job_id: str,
        scene_id: int,
        revision: int = 1,
    ) -> Path:
        return self.revision_root(job_id, scene_id, revision) / "generation-manifest.json"

    def final_path(self, job_id: str) -> Path:
        _validate_job_id(job_id)
        return self.finals_root / f"{job_id}_final.mp4"


def _validate_job_id(job_id: str) -> None:
    import re

    if not isinstance(job_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}",
        job_id,
    ):
        raise StorageError("job_id contains unsafe path characters.")


def _validate_scene_id(scene_id: int) -> None:
    if isinstance(scene_id, bool) or not isinstance(scene_id, int) or scene_id < 1:
        raise StorageError("scene_id must be a positive integer.")


def _copy_if_missing(source: Path, destination: Path) -> bool:
    if not source.is_file() or destination.exists():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def _backup_sqlite(source: Path, destination: Path) -> bool:
    if not source.is_file() or destination.exists():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".sqlite3.migrating")
    if temporary.exists():
        temporary.unlink()
    source_connection = sqlite3.connect(source)
    destination_connection = sqlite3.connect(temporary)
    try:
        source_connection.backup(destination_connection)
    finally:
        destination_connection.close()
        source_connection.close()
    os.replace(temporary, destination)
    return True


def _materialize_payloads(layout: StorageLayout) -> int:
    if not layout.database_path.is_file():
        return 0
    written = 0
    connection = sqlite3.connect(layout.database_path)
    try:
        rows = connection.execute(
            "SELECT job_id, payload_json FROM jobs ORDER BY created_at, job_id"
        ).fetchall()
    finally:
        connection.close()
    for job_id, payload_json in rows:
        destination = layout.source_payload_path(str(job_id))
        if destination.exists():
            continue
        try:
            payload = json.loads(payload_json)
        except (TypeError, json.JSONDecodeError) as error:
            raise StorageError(f"Saved job {job_id} contains invalid JSON: {error}") from error
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
        written += 1
    return written


def _copy_legacy_media(
    layout: StorageLayout,
    legacy_output_root: Path,
) -> tuple[int, int, int]:
    """Copy only paths recorded by this project's database plus named final files."""
    if not layout.database_path.is_file():
        return (0, 0, 0)
    frames = clips = finals = 0
    connection = sqlite3.connect(layout.database_path)
    try:
        scene_rows = connection.execute(
            "SELECT job_id, scene_id, frame_path, video_path FROM scenes"
        ).fetchall()
        job_rows = connection.execute("SELECT job_id FROM jobs").fetchall()
    finally:
        connection.close()
    for job_id, scene_id, frame_path, video_path in scene_rows:
        expected_frame = legacy_output_root / ".work" / str(job_id) / "frames" / (
            f"scene_{int(scene_id):04d}.png"
        )
        expected_clip = legacy_output_root / ".work" / str(job_id) / "clips" / (
            f"scene_{int(scene_id):04d}.mp4"
        )
        source_frame = Path(frame_path) if frame_path else expected_frame
        source_clip = Path(video_path) if video_path else expected_clip
        if _copy_if_missing(
            source_frame,
            layout.scene_frame_path(str(job_id), int(scene_id)),
        ):
            frames += 1
        if _copy_if_missing(
            source_clip,
            layout.scene_clip_path(str(job_id), int(scene_id)),
        ):
            clips += 1
    for (job_id,) in job_rows:
        if _copy_if_missing(
            legacy_output_root / f"{job_id}_final.mp4",
            layout.final_path(str(job_id)),
        ):
            finals += 1
    return frames, clips, finals


def _rewrite_migrated_paths(layout: StorageLayout) -> None:
    if not layout.database_path.is_file():
        return
    connection = sqlite3.connect(layout.database_path)
    try:
        scene_rows = connection.execute(
            "SELECT job_id, scene_id FROM scenes"
        ).fetchall()
        for job_id, scene_id in scene_rows:
            frame = layout.scene_frame_path(str(job_id), int(scene_id))
            clip = layout.scene_clip_path(str(job_id), int(scene_id))
            connection.execute(
                """
                UPDATE scenes
                SET frame_path = CASE WHEN ? THEN ? ELSE frame_path END,
                    video_path = CASE WHEN ? THEN ? ELSE video_path END
                WHERE job_id = ? AND scene_id = ?
                """,
                (
                    frame.is_file(),
                    str(frame),
                    clip.is_file(),
                    str(clip),
                    job_id,
                    scene_id,
                ),
            )
        job_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(jobs)").fetchall()
        }
        if "final_path" in job_columns:
            for (job_id,) in connection.execute("SELECT job_id FROM jobs").fetchall():
                final = layout.final_path(str(job_id))
                if final.is_file():
                    connection.execute(
                        "UPDATE jobs SET final_path = ? WHERE job_id = ?",
                        (str(final), job_id),
                    )
        connection.commit()
    finally:
        connection.close()


def migrate_legacy_storage(
    project_root: str | Path,
    *,
    layout: StorageLayout | None = None,
    legacy_output_root: str | Path = LEGACY_OUTPUT_ROOT,
) -> dict[str, int | bool | str]:
    """Copy legacy project state into D: without deleting or modifying the source."""
    project_root = _resolved(project_root)
    layout = layout or StorageLayout.configured()
    layout.ensure()
    if layout.migration_marker.exists():
        return {
            "already_migrated": True,
            "database": False,
            "settings": 0,
            "secrets": 0,
            "payloads": 0,
            "frames": 0,
            "clips": 0,
            "finals": 0,
            "storage_root": str(layout.root),
        }

    database_copied = _backup_sqlite(
        project_root / "runtime" / "pipeline.sqlite3",
        layout.database_path,
    )
    settings_copied = int(
        _copy_if_missing(project_root / ".env", layout.settings_path)
    )
    secrets_copied = int(
        _copy_if_missing(project_root / "runtime" / "secrets.json", layout.secrets_path)
    )
    _copy_if_missing(
        project_root / "runtime" / "assets.json",
        layout.asset_manifest_path,
    )
    payload_count = _materialize_payloads(layout)
    frame_count, clip_count, final_count = _copy_legacy_media(
        layout,
        _resolved(legacy_output_root),
    )
    _rewrite_migrated_paths(layout)
    result: dict[str, int | bool | str] = {
        "already_migrated": False,
        "database": database_copied,
        "settings": settings_copied,
        "secrets": secrets_copied,
        "payloads": payload_count,
        "frames": frame_count,
        "clips": clip_count,
        "finals": final_count,
        "storage_root": str(layout.root),
    }
    temporary = layout.migration_marker.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, layout.migration_marker)
    return result
