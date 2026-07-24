"""FFmpeg preflight and deterministic scene concatenation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import subprocess
from typing import Callable, Iterable, Sequence

from .constants import PRODUCTION_FPS, PRODUCTION_HEIGHT, PRODUCTION_WIDTH


class AssemblyError(RuntimeError):
    """Raised when clips do not meet the fixed production profile or FFmpeg fails."""


@dataclass(frozen=True)
class VideoStreamInfo:
    path: Path
    width: int
    height: int
    frame_rate: Fraction


def validate_video_profile(streams: Iterable[VideoStreamInfo]) -> tuple[VideoStreamInfo, ...]:
    checked = tuple(streams)
    if not checked:
        raise AssemblyError("At least one successful scene clip is required for stitching.")
    expected_rate = Fraction(PRODUCTION_FPS, 1)
    for stream in checked:
        if stream.width != PRODUCTION_WIDTH or stream.height != PRODUCTION_HEIGHT:
            raise AssemblyError(
                f"{stream.path} is {stream.width}x{stream.height}; expected {PRODUCTION_WIDTH}x{PRODUCTION_HEIGHT}."
            )
        if stream.frame_rate != expected_rate:
            raise AssemblyError(f"{stream.path} is {stream.frame_rate} fps; expected {PRODUCTION_FPS} fps.")
    return checked


def concat_list_text(clips: Sequence[str | Path]) -> str:
    if not clips:
        raise AssemblyError("Cannot create an FFmpeg concat list with no clips.")
    lines = []
    for clip in clips:
        path = Path(clip).resolve()
        if not path.is_file():
            raise AssemblyError(f"Scene clip is missing: {path}")
        escaped = path.as_posix().replace("'", r"'\''")
        lines.append(f"file '{escaped}'")
    return "\n".join(lines) + "\n"


class FfmpegAssembler:
    """Uses copy-mode concat after a separate fixed-profile preflight."""

    def __init__(
        self,
        output_root: str | Path = r"D:\output\10minfinals",
        *,
        ffmpeg_executable: str = "ffmpeg",
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ):
        self.output_root = Path(output_root)
        self.ffmpeg_executable = ffmpeg_executable
        self._runner = runner

    def final_path(self, job_id: str) -> Path:
        if not job_id or any(character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-" for character in job_id):
            raise AssemblyError("Job id is not safe for an output filename.")
        return self.output_root / f"{job_id}_final.mp4"

    def stitch(self, job_id: str, clips: Sequence[str | Path], concat_directory: str | Path) -> Path:
        output_path = self.final_path(job_id)
        concat_directory = Path(concat_directory)
        concat_directory.mkdir(parents=True, exist_ok=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        concat_path = concat_directory / f"{job_id}_concat.txt"
        concat_path.write_text(concat_list_text(clips), encoding="utf-8", newline="\n")
        command = [
            self.ffmpeg_executable,
            "-hide_banner",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_path),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        completed = self._runner(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise AssemblyError(f"FFmpeg stitching failed: {completed.stderr.strip()}")
        if not output_path.is_file():
            raise AssemblyError("FFmpeg reported success but did not create the final video.")
        return output_path


def probe_video(
    path: str | Path,
    *,
    ffprobe_executable: str = "ffprobe",
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> VideoStreamInfo:
    """Read only the primary video stream metadata needed for concat safety."""
    video_path = Path(path)
    if not video_path.is_file():
        raise AssemblyError(f"Scene clip is missing: {video_path}")
    command = [
        ffprobe_executable,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate",
        "-of",
        "json",
        str(video_path),
    ]
    completed = runner(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise AssemblyError(f"FFprobe failed for {video_path}: {completed.stderr.strip()}")
    try:
        streams = json.loads(completed.stdout).get("streams", [])
        stream = streams[0]
        return VideoStreamInfo(
            path=video_path,
            width=int(stream["width"]),
            height=int(stream["height"]),
            frame_rate=Fraction(str(stream["r_frame_rate"])),
        )
    except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise AssemblyError(f"FFprobe did not return a usable video stream for {video_path}.") from error
