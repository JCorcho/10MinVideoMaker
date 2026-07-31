"""Safe, browser-friendly media review for bounded continuation acceptance runs."""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import threading
from typing import Any, Mapping

from .storage import StorageLayout


class AcceptanceReviewError(RuntimeError):
    """Raised when an acceptance review artifact cannot be safely served."""


_RUN_ID = re.compile(r"continuation-acceptance-\d{8}-\d{6}\Z")
_BASE_ROLE = "base"
_CASE_ORDER = (
    "single_frame",
    "decoded_17_frame",
    "latent_overlap",
)
_REQUIRED_CASES = ("common_base",) + _CASE_ORDER
_CASE_DETAILS: dict[str, dict[str, Any]] = {
    "single_frame": {
        "title": "Single final frame",
        "summary": "Chunk 2 starts from only the final frame of the base window.",
        "boundary": {"left": [119, 120], "right": [0, 1]},
        "stills": (
            ("base_0119.png", "Base frame 119", "base"),
            ("base_0120.png", "Base final frame 120", "base"),
            ("case_0000.png", "Continuation first frame 0", "continuation"),
            ("case_0001.png", "Continuation frame 1", "continuation"),
        ),
    },
    "decoded_17_frame": {
        "title": "Decoded 17-frame guide",
        "summary": "Chunk 2 receives a 17-frame decoded guide from the base window.",
        "boundary": {"left": [111, 112], "right": [16, 17]},
        "stills": (
            ("base_0096.png", "Base guide start 96", "base"),
            ("base_0111.png", "Base guide penultimate 111", "base"),
            ("base_0112.png", "Base guide final 112", "base"),
            ("case_0000.png", "Continuation guide start 0", "continuation"),
            ("case_0016.png", "Continuation guide final 16", "continuation"),
            ("case_0017.png", "First new continuation frame 17", "continuation"),
        ),
    },
    "latent_overlap": {
        "title": "Latent 25-frame overlap",
        "summary": "Chunk 2 receives the final 25-frame latent overlap from the base window.",
        "boundary": {"left": [119, 120], "right": [24, 25]},
        "stills": (
            ("base_0096.png", "Base overlap start 96", "base"),
            ("base_0119.png", "Base penultimate frame 119", "base"),
            ("base_0120.png", "Base final frame 120", "base"),
            ("case_0000.png", "Continuation overlap start 0", "continuation"),
            ("case_0024.png", "Continuation overlap final 24", "continuation"),
            ("case_0025.png", "First new continuation frame 25", "continuation"),
        ),
    },
}


class AcceptanceReviewService:
    """Read one completed acceptance matrix without changing pipeline state."""

    def __init__(self, storage: StorageLayout, *, ffmpeg_command: str = "ffmpeg"):
        self.storage = storage
        self.ffmpeg_command = ffmpeg_command
        self._locks: dict[Path, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def list_runs(self) -> list[dict[str, object]]:
        root = self.storage.root / "acceptance"
        if not root.is_dir():
            return []
        runs: list[dict[str, object]] = []
        for candidate in root.iterdir():
            if not candidate.is_dir() or not _RUN_ID.fullmatch(candidate.name):
                continue
            try:
                document, _ = self._load_run(candidate.name)
            except AcceptanceReviewError:
                continue
            runs.append(
                {
                    "run_id": candidate.name,
                    "source_job_id": document.get("source_job_id"),
                    "source_scene_id": document.get("source_scene_id"),
                }
            )
        return sorted(runs, key=lambda item: str(item["run_id"]), reverse=True)

    def review_document(self, run_id: str) -> dict[str, object]:
        document, _ = self._load_run(run_id)
        for case_name in _REQUIRED_CASES:
            self._raw_video_path(document, run_id, case_name)
        cases: dict[str, object] = {}
        for case_name in _CASE_ORDER:
            details = _CASE_DETAILS[case_name]
            cases[case_name] = {
                "role": case_name,
                "title": details["title"],
                "summary": details["summary"],
                "boundary": details["boundary"],
                "video_url": self._media_url(run_id, case_name),
                "stills": [
                    {
                        "name": filename,
                        "label": label,
                        "side": side,
                        "url": self._still_url(run_id, case_name, filename),
                    }
                    for filename, label, side in details["stills"]
                ],
            }
        return {
            "run_id": run_id,
            "source_job_id": document.get("source_job_id"),
            "source_scene_id": document.get("source_scene_id"),
            "fps": 24,
            "base": {
                "role": _BASE_ROLE,
                "title": "Shared base window",
                "summary": "First 121-frame window used as the reference for every case.",
                "video_url": self._media_url(run_id, _BASE_ROLE),
            },
            "case_order": list(_CASE_ORDER),
            "cases": cases,
        }

    def review_proxy_path(self, run_id: str, role: str) -> Path:
        document, run_root = self._load_run(run_id)
        source = self._raw_video_path(document, run_id, self._case_for_role(role))
        proxy = self._safe_path(run_root / "review" / f"{role}.mp4")
        with self._proxy_lock(proxy):
            if proxy.is_file() and proxy.stat().st_size > 0:
                return proxy
            proxy.parent.mkdir(parents=True, exist_ok=True)
            temporary = proxy.with_name(f"{proxy.stem}.partial{proxy.suffix}")
            temporary.unlink(missing_ok=True)
            command = [
                self.ffmpeg_command,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source),
                "-map",
                "0:v:0",
                "-map",
                "0:a?",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "160k",
                "-movflags",
                "+faststart",
                str(temporary),
            ]
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0 or not temporary.is_file() or temporary.stat().st_size == 0:
                temporary.unlink(missing_ok=True)
                detail = (completed.stderr or "FFmpeg did not produce review media.").strip()
                raise AcceptanceReviewError(f"FFmpeg review proxy failed: {detail}")
            temporary.replace(proxy)
        return proxy

    def still_path(self, run_id: str, case_name: str, asset_name: str) -> Path:
        _, run_root = self._load_run(run_id)
        details = _CASE_DETAILS.get(case_name)
        if details is None:
            raise AcceptanceReviewError("Unknown review case.")
        filenames = {entry[0] for entry in details["stills"]}
        if asset_name not in filenames:
            raise AcceptanceReviewError("Unknown review still.")
        path = self._safe_path(run_root / "metrics" / case_name / asset_name)
        if not path.is_file():
            raise AcceptanceReviewError("Review still is missing.")
        return path

    def _load_run(self, run_id: str) -> tuple[Mapping[str, Any], Path]:
        if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
            raise AcceptanceReviewError("Acceptance run ID is invalid.")
        run_root = self._safe_path(self.storage.root / "acceptance" / run_id)
        run_path = self._safe_path(run_root / "run.json")
        if not run_path.is_file():
            raise AcceptanceReviewError("Acceptance run is missing.")
        try:
            document = json.loads(run_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise AcceptanceReviewError("Acceptance run document is invalid.") from error
        if not isinstance(document, Mapping):
            raise AcceptanceReviewError("Acceptance run document is invalid.")
        if document.get("state") != "awaiting_human_review":
            raise AcceptanceReviewError("Acceptance run is not ready for review.")
        if document.get("run_id") != run_id:
            raise AcceptanceReviewError("Acceptance run ID does not match its document.")
        return document, run_root

    def _raw_video_path(
        self,
        document: Mapping[str, Any],
        run_id: str,
        case_name: str,
    ) -> Path:
        cases = document.get("cases")
        case = cases.get(case_name) if isinstance(cases, Mapping) else None
        stage2 = case.get("stage2") if isinstance(case, Mapping) else None
        raw_path = stage2.get("raw_video_path") if isinstance(stage2, Mapping) else None
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise AcceptanceReviewError("Acceptance raw video path is invalid.")
        path = self._safe_path(raw_path)
        expected_root = self._safe_path(self.storage.job_root(run_id))
        try:
            path.relative_to(expected_root)
        except ValueError as error:
            raise AcceptanceReviewError("Acceptance raw video is outside project storage.") from error
        if not path.is_file():
            raise AcceptanceReviewError("Acceptance raw video is missing.")
        return path

    def _safe_path(self, value: str | Path) -> Path:
        path = Path(value).resolve()
        try:
            path.relative_to(self.storage.root.resolve())
        except ValueError as error:
            raise AcceptanceReviewError("Review artifact is outside project storage.") from error
        return path

    @staticmethod
    def _case_for_role(role: str) -> str:
        if role == _BASE_ROLE:
            return "common_base"
        if role in _CASE_ORDER:
            return role
        raise AcceptanceReviewError("Unknown review video role.")

    @staticmethod
    def _media_url(run_id: str, role: str) -> str:
        return f"/api/acceptance-runs/{run_id}/media/{role}"

    @staticmethod
    def _still_url(run_id: str, case_name: str, asset_name: str) -> str:
        return f"/api/acceptance-runs/{run_id}/stills/{case_name}/{asset_name}"

    def _proxy_lock(self, path: Path) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(path, threading.Lock())
