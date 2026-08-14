"""Deterministic FFmpeg-backed frame sampling for production video QC.

The selection and four-frame slicing semantics are ported from the validated
standalone LTX23-VLM-Video-QC-Lab at commit f634ca2.  Production owns this copy
so the research checkout is never a mutable runtime dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import statistics
import subprocess
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4


class QcVideoError(RuntimeError):
    """Raised when video metadata or deterministic frame extraction fails."""


@dataclass(frozen=True)
class VideoMetadata:
    source_fps: float
    source_frame_count: int
    duration_seconds: float


@dataclass(frozen=True)
class FrameSelection:
    indices: tuple[int, ...]
    timestamps_seconds: tuple[float, ...]
    effective_fps: float
    source_duration_seconds: float


@dataclass(frozen=True)
class SampledFrame:
    source_index: int
    timestamp_seconds: float
    image_path: Path
    image_bytes: bytes | None = None

    def bytes(self) -> bytes:
        if self.image_bytes is not None:
            return self.image_bytes
        return self.image_path.read_bytes()


@dataclass(frozen=True)
class SampledVideo:
    metadata: VideoMetadata
    target_fps: float
    frames: tuple[SampledFrame, ...]


@dataclass(frozen=True)
class QcWindow:
    window_number: int
    frames: tuple[SampledFrame, ...]
    confirmation_of_window: int | None = None

    @property
    def source_frame_indices(self) -> tuple[int, ...]:
        return tuple(item.source_index for item in self.frames)

    @property
    def timestamps_seconds(self) -> tuple[float, ...]:
        return tuple(item.timestamp_seconds for item in self.frames)


def _ratio(value: object) -> float:
    text = str(value or "0")
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        if float(denominator) == 0:
            return 0.0
        return float(numerator) / float(denominator)
    return float(text)


def select_frame_indices(
    total_frames: int,
    source_fps: float,
    duration: float,
    target_fps: float = 2.0,
    *,
    actual_timestamps: Sequence[float] | None = None,
) -> FrameSelection:
    """Preserve the lab's chronological target-FPS rounding exactly."""
    if isinstance(total_frames, bool) or total_frames < 1:
        raise ValueError("Video has no decodable frames.")
    if target_fps <= 0:
        raise ValueError("Target FPS must be greater than zero.")
    if source_fps <= 0:
        source_fps = 1.0
    if duration <= 0:
        duration = total_frames / source_fps
    count = max(1, math.ceil(max(duration, total_frames / source_fps) * target_fps))
    selected: list[int] = []
    for position in range(count):
        index = int(round(position * source_fps / target_fps))
        if index >= total_frames:
            break
        if not selected or selected[-1] != index:
            selected.append(index)
    indices = tuple(selected or [0])
    if actual_timestamps is not None:
        if len(actual_timestamps) < total_frames:
            raise ValueError("Actual timestamp data does not cover every source frame.")
        timestamps = tuple(float(actual_timestamps[index]) for index in indices)
    else:
        timestamps = tuple(index / source_fps for index in indices)
    effective_fps = len(indices) / duration
    deltas = tuple(
        right - left
        for left, right in zip(timestamps, timestamps[1:])
        if right > left
    )
    if deltas:
        effective_fps = 1.0 / statistics.median(deltas)
    return FrameSelection(indices, timestamps, effective_fps, duration)


def chronological_windows(
    sampled: SampledVideo, *, frame_count: int = 4
) -> tuple[QcWindow, ...]:
    if isinstance(frame_count, bool) or frame_count < 1:
        raise ValueError("frame_count must be positive.")
    if not sampled.frames:
        raise ValueError("A sampled video must contain at least one frame.")
    return tuple(
        QcWindow(
            window_number=offset // frame_count + 1,
            frames=sampled.frames[offset : offset + frame_count],
        )
        for offset in range(0, len(sampled.frames), frame_count)
    )


def shifted_confirmation_window(
    sampled: SampledVideo,
    suspect: QcWindow,
    *,
    frame_count: int = 4,
) -> QcWindow | None:
    """Build the lab's shifted overlap, refusing an identical-frame replay."""
    windows = chronological_windows(sampled, frame_count=frame_count)
    if suspect.window_number > len(windows):
        raise ValueError("The suspect window is outside the sampled video.")
    suspect_offset = (suspect.window_number - 1) * frame_count
    confirmation_start = max(
        0,
        min(
            suspect_offset - frame_count // 2,
            len(sampled.frames) - frame_count,
        ),
    )
    if confirmation_start == suspect_offset and len(sampled.frames) > frame_count:
        confirmation_start = min(
            len(sampled.frames) - frame_count,
            suspect_offset + max(1, frame_count // 2),
        )
    confirmation_end = min(len(sampled.frames), confirmation_start + frame_count)
    frames = sampled.frames[confirmation_start:confirmation_end]
    if tuple(item.source_index for item in frames) == suspect.source_frame_indices:
        return None
    return QcWindow(
        window_number=len(windows) + 1,
        frames=frames,
        confirmation_of_window=suspect.window_number,
    )


def build_frame_accounting(
    sampled: SampledVideo,
    *,
    planned_windows: Sequence[QcWindow],
    processed_windows: Sequence[QcWindow],
    confirmation: QcWindow | None,
    early_exit: bool,
    early_exit_reason: str | None,
) -> dict[str, object]:
    inspected = tuple(frame for window in processed_windows for frame in window.frames)
    confirmation_frames = () if confirmation is None else confirmation.frames
    return {
        "source_fps": sampled.metadata.source_fps,
        "source_frame_count": sampled.metadata.source_frame_count,
        "source_duration_seconds": sampled.metadata.duration_seconds,
        "requested_sampling_fps": sampled.target_fps,
        "initial_selected_frame_count": len(sampled.frames),
        "initial_selected_source_frame_indices": [
            item.source_index for item in sampled.frames
        ],
        "initial_selected_source_timestamps": [
            item.timestamp_seconds for item in sampled.frames
        ],
        "selected_frame_count": len(inspected),
        "selected_source_frame_indices": [item.source_index for item in inspected],
        "selected_source_timestamps": [item.timestamp_seconds for item in inspected],
        "unique_selected_frames_inspected": len({item.source_index for item in inspected}),
        "confirmation_frame_exposures": len(confirmation_frames),
        "frame_count_represented_in_model_input": len(inspected)
        + len(confirmation_frames),
        "frames_per_window": max(len(item.frames) for item in planned_windows),
        "planned_window_count": len(planned_windows),
        "processed_window_count": len(processed_windows),
        "windows": [
            {
                "window_number": item.window_number,
                "source_frame_indices": list(item.source_frame_indices),
                "timestamps_seconds": list(item.timestamps_seconds),
            }
            for item in processed_windows
        ],
        "confirmation": None
        if confirmation is None
        else {
            "window_number": confirmation.window_number,
            "confirmation_of_window": confirmation.confirmation_of_window,
            "source_frame_indices": list(confirmation.source_frame_indices),
            "timestamps_seconds": list(confirmation.timestamps_seconds),
        },
        "early_exit_applied": bool(early_exit),
        "early_exit_reason": early_exit_reason,
    }


def _default_runner(command: Sequence[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, **kwargs)  # type: ignore[arg-type]


def _metadata_from_probe(payload: Mapping[str, Any]) -> tuple[VideoMetadata, tuple[float, ...]]:
    streams = payload.get("streams")
    if not isinstance(streams, list) or not streams or not isinstance(streams[0], Mapping):
        raise QcVideoError("FFprobe returned no primary video stream.")
    stream = streams[0]
    source_fps = _ratio(stream.get("avg_frame_rate")) or _ratio(stream.get("r_frame_rate"))
    if source_fps <= 0:
        raise QcVideoError("FFprobe returned an invalid source frame rate.")
    raw_frames = payload.get("frames")
    actual_timestamps: list[float] = []
    if isinstance(raw_frames, list):
        for item in raw_frames:
            if not isinstance(item, Mapping):
                continue
            value = item.get("best_effort_timestamp_time", item.get("pts_time"))
            try:
                actual_timestamps.append(float(value))
            except (TypeError, ValueError):
                actual_timestamps.append(len(actual_timestamps) / source_fps)
    try:
        declared_count = int(stream.get("nb_frames") or 0)
    except (TypeError, ValueError):
        declared_count = 0
    format_data = payload.get("format") if isinstance(payload.get("format"), Mapping) else {}
    try:
        duration = float(stream.get("duration") or format_data.get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    total_frames = len(actual_timestamps) or declared_count or int(round(duration * source_fps))
    if total_frames < 1:
        raise QcVideoError("FFprobe could not determine a decodable frame count.")
    if duration <= 0:
        duration = total_frames / source_fps
    if not actual_timestamps:
        actual_timestamps = [index / source_fps for index in range(total_frames)]
    return VideoMetadata(source_fps, total_frames, duration), tuple(actual_timestamps)


def sample_video_frames(
    video_path: Path,
    *,
    target_fps: float,
    ffprobe_command: str,
    ffmpeg_command: str,
    temporary_root: Path,
    run_command: Callable[..., Any] = _default_runner,
) -> SampledVideo:
    """Probe all source timestamps, then extract only exact selected indices."""
    source = Path(video_path).resolve()
    if not source.is_file():
        raise QcVideoError(f"Candidate video does not exist: {source}")
    probe = run_command(
        [
            ffprobe_command,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_streams",
            "-show_format",
            "-show_frames",
            "-show_entries",
            "stream=avg_frame_rate,r_frame_rate,nb_frames,duration:"
            "format=duration:frame=best_effort_timestamp_time,pts_time",
            "-of",
            "json",
            str(source),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if probe.returncode:
        raise QcVideoError(probe.stderr.strip() or "FFprobe failed.")
    try:
        metadata, actual_timestamps = _metadata_from_probe(json.loads(probe.stdout))
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise QcVideoError(f"FFprobe returned invalid JSON: {error}") from error
    selection = select_frame_indices(
        metadata.source_frame_count,
        metadata.source_fps,
        metadata.duration_seconds,
        target_fps,
        actual_timestamps=actual_timestamps,
    )
    output_root = Path(temporary_root).resolve() / f"qc-sample-{uuid4().hex}"
    output_root.mkdir(parents=True, exist_ok=False)
    expression = "+".join(f"eq(n\\,{index})" for index in selection.indices)
    extract = run_command(
        [
            ffmpeg_command,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-vf",
            f"select='{expression}'",
            "-fps_mode",
            "vfr",
            "-q:v",
            "2",
            str(output_root / "frame-%06d.jpg"),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if extract.returncode:
        raise QcVideoError(extract.stderr.strip() or "FFmpeg frame extraction failed.")
    paths = tuple(sorted(output_root.glob("frame-*.jpg")))
    if len(paths) != len(selection.indices):
        raise QcVideoError(
            "FFmpeg extracted a different number of frames than the deterministic selection."
        )
    frames = tuple(
        SampledFrame(index, timestamp, path)
        for index, timestamp, path in zip(
            selection.indices, selection.timestamps_seconds, paths
        )
    )
    return SampledVideo(metadata, target_fps, frames)
