"""Deterministic FFmpeg-backed frame sampling for production video QC.

The selection and four-frame slicing semantics are ported from the validated
standalone LTX23-VLM-Video-QC-Lab at commit f634ca2.  Production owns this copy
so the research checkout is never a mutable runtime dependency.

The lab decodes through PyAV while production extracts selected lossless RGB24
PNGs through FFmpeg. Both use libav decoding and the subsequent OpenCV resize
and JPEG-88 steps are identical, but exact decoded/JPEG bytes are not promised
across different libav/OpenCV builds. Per-frame dimensions and hashes are
persisted so a deployed runtime remains semantically and byte-level auditable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4


class QcVideoError(RuntimeError):
    """Raised when video metadata or deterministic frame extraction fails."""


PREPROCESSING_VERSION = "vlm-qc-lab-f634ca2-image-v1"
VALIDATED_LAB_COMMIT = "f634ca2ab7ca95ddd9abde7fe840031eba0696f4"
IMAGE_MAX_SHORT_EDGE = 512
IMAGE_MAX_PIXELS = 458_752
IMAGE_DIMENSION_MULTIPLE = 16
IMAGE_JPEG_QUALITY = 88


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
    width: int | None = None
    height: int | None = None
    image_sha256: str | None = None

    def bytes(self) -> bytes:
        if self.image_bytes is not None:
            return self.image_bytes
        return self.image_path.read_bytes()

    @property
    def content_sha256(self) -> str:
        return self.image_sha256 or hashlib.sha256(self.bytes()).hexdigest()


@dataclass(frozen=True)
class SampledVideo:
    metadata: VideoMetadata
    target_fps: float
    frames: tuple[SampledFrame, ...]
    preprocessing: Mapping[str, object] = field(default_factory=dict)


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

    @property
    def image_sha256s(self) -> tuple[str, ...]:
        return tuple(item.content_sha256 for item in self.frames)


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
    if not set(item.content_sha256 for item in frames).difference(
        suspect.image_sha256s
    ):
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
    suspect_frames = next(
        (
            item.frames
            for item in processed_windows
            if confirmation is not None
            and item.window_number == confirmation.confirmation_of_window
        ),
        None,
    )
    confirmation_independent = (
        None
        if confirmation is None
        else suspect_frames is not None
        and tuple(item.source_index for item in confirmation.frames)
        != tuple(item.source_index for item in suspect_frames)
        and bool(
            set(item.content_sha256 for item in confirmation.frames).difference(
                item.content_sha256 for item in suspect_frames
            )
        )
    )
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
        "confirmation_independence_rule": (
            "shifted_indices_and_at_least_one_new_image_sha256_required"
        ),
        "confirmation_is_independent": confirmation_independent,
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
                "image_sha256s": list(item.image_sha256s),
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
            "image_sha256s": list(confirmation.image_sha256s),
        },
        "early_exit_applied": bool(early_exit),
        "early_exit_reason": early_exit_reason,
        "preprocessing": dict(sampled.preprocessing),
        "sampled_frame_artifacts": [
            {
                "source_frame_index": item.source_index,
                "timestamp_seconds": item.timestamp_seconds,
                "width": item.width,
                "height": item.height,
                "jpeg_sha256": item.image_sha256,
            }
            for item in sampled.frames
        ],
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


def _preprocessing_provenance() -> dict[str, object]:
    return {
        "version": PREPROCESSING_VERSION,
        "validated_lab_commit": VALIDATED_LAB_COMMIT,
        "decoder": "ffmpeg_selected_png_rgb24",
        "orientation": "encoded_pixels_no_autorotate",
        "color_pipeline": "rgb24_to_opencv_bgr_to_jpeg",
        "resize_interpolation": "opencv_inter_area",
        "max_short_edge": IMAGE_MAX_SHORT_EDGE,
        "max_pixels": IMAGE_MAX_PIXELS,
        "dimension_multiple": IMAGE_DIMENSION_MULTIPLE,
        "jpeg_quality": IMAGE_JPEG_QUALITY,
    }


def _benchmark_resize(image: Any) -> Any:
    """Apply the reviewed lab's exact scale, floor, and INTER_AREA semantics."""
    try:
        import cv2
    except ImportError as error:
        raise QcVideoError(
            "Production QC preprocessing requires OpenCV, matching the validated lab."
        ) from error
    height, width = image.shape[:2]
    scale = min(
        1.0,
        IMAGE_MAX_SHORT_EDGE / max(1, min(height, width)),
        math.sqrt(IMAGE_MAX_PIXELS / max(1, height * width)),
    )
    if scale < 1.0:
        new_width = max(
            IMAGE_DIMENSION_MULTIPLE,
            int(width * scale) // IMAGE_DIMENSION_MULTIPLE * IMAGE_DIMENSION_MULTIPLE,
        )
        new_height = max(
            IMAGE_DIMENSION_MULTIPLE,
            int(height * scale) // IMAGE_DIMENSION_MULTIPLE * IMAGE_DIMENSION_MULTIPLE,
        )
        image = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA,
        )
    return image


def _encode_benchmark_jpeg(
    source_path: Path,
    destination: Path,
) -> tuple[bytes, int, int, str]:
    try:
        import cv2
    except ImportError as error:
        raise QcVideoError(
            "Production QC preprocessing requires OpenCV, matching the validated lab."
        ) from error
    image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if image is None:
        raise QcVideoError(f"OpenCV could not decode extracted frame {source_path}.")
    image = _benchmark_resize(image)
    ok, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, IMAGE_JPEG_QUALITY],
    )
    if not ok:
        raise QcVideoError(f"OpenCV could not encode sampled frame {source_path}.")
    payload = encoded.tobytes()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return (
        payload,
        int(image.shape[1]),
        int(image.shape[0]),
        hashlib.sha256(payload).hexdigest(),
    )


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
    decoded_root = output_root / "decoded"
    decoded_root.mkdir(parents=True, exist_ok=False)
    expression = "+".join(f"eq(n\\,{index})" for index in selection.indices)
    extract = run_command(
        [
            ffmpeg_command,
            "-hide_banner",
            "-loglevel",
            "error",
            "-noautorotate",
            "-i",
            str(source),
            "-vf",
            f"select='{expression}'",
            "-fps_mode",
            "vfr",
            "-c:v",
            "png",
            "-pix_fmt",
            "rgb24",
            str(decoded_root / "source-%06d.png"),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if extract.returncode:
        raise QcVideoError(extract.stderr.strip() or "FFmpeg frame extraction failed.")
    decoded_paths = tuple(sorted(decoded_root.glob("source-*.png")))
    if len(decoded_paths) != len(selection.indices):
        raise QcVideoError(
            "FFmpeg extracted a different number of frames than the deterministic selection."
        )
    frames: list[SampledFrame] = []
    for position, (index, timestamp, decoded_path) in enumerate(
        zip(selection.indices, selection.timestamps_seconds, decoded_paths),
        1,
    ):
        image_path = output_root / "frames" / f"frame-{position:06d}.jpg"
        payload, width, height, image_sha256 = _encode_benchmark_jpeg(
            decoded_path,
            image_path,
        )
        frames.append(
            SampledFrame(
                source_index=index,
                timestamp_seconds=timestamp,
                image_path=image_path,
                image_bytes=payload,
                width=width,
                height=height,
                image_sha256=image_sha256,
            )
        )
    return SampledVideo(
        metadata,
        target_fps,
        tuple(frames),
        _preprocessing_provenance(),
    )
