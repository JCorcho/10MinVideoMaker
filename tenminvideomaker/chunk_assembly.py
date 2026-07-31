"""Controlled FFmpeg assembly for one chunked-continuation scene.

Raw continuation chunks are never stream-copied.  Each accepted chunk is
trimmed by exact frame index, timestamp-reset, and joined through one
scene-level encode.  At audio seams this module applies a 100 ms quarter-sine
fade-out to the preceding chunk and fade-in to the following chunk.  The
fades do not overlap samples or change segment duration; they only suppress
clicks while a future seam analyser can choose a content-aware crossfade.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
import os
from pathlib import Path
import subprocess
from typing import Callable, Mapping, Sequence
from uuid import uuid4

from .constants import PRODUCTION_FPS, PRODUCTION_HEIGHT, PRODUCTION_WIDTH
from .continuation import (
    CAUSAL_REFINEMENT_PREROLL_FRAMES,
    SceneFramePlan,
    refinement_raw_frame_count,
)


class SceneChunkAssemblyError(RuntimeError):
    """Raised when a raw chunk or assembled scene fails structural validation."""


@dataclass(frozen=True)
class ChunkSlice:
    """One raw input's independently aligned video and audio ranges."""

    chunk_index: int
    start_frame: int
    end_frame_exclusive: int
    audio_start_frame: int
    audio_end_frame_exclusive: int

    @property
    def frame_count(self) -> int:
        return self.end_frame_exclusive - self.start_frame


@dataclass(frozen=True)
class MediaProbe:
    """The primary video/audio stream facts used for deterministic assembly."""

    path: Path
    width: int
    height: int
    r_frame_rate: Fraction
    avg_frame_rate: Fraction
    decoded_video_frames: int
    video_codec: str
    video_profile: str
    pixel_format: str
    audio_codec: str | None
    audio_sample_rate: int | None
    audio_channels: int | None


def chunk_slices(plan: SceneFramePlan) -> tuple[ChunkSlice, ...]:
    """Map the causal handoff timeline onto style-stable raw chunk indexes.

    A later bounded handoff decodes eight sacrificial causal frames before its
    first clean frame.  That clean frame is global ``window_start + 8``.  The
    preceding chunk therefore owns eight more frames than the old sampled
    refinement route, while the final chunk owns eight fewer.  Stage-two audio
    is conditioned eight frames earlier than the direct video decode, so its
    trim advances by a second eight-frame offset on later chunks.
    """
    slices: list[ChunkSlice] = []
    for position, chunk in enumerate(plan.chunks):
        start = 0 if chunk.is_initial else CAUSAL_REFINEMENT_PREROLL_FRAMES
        if position + 1 < len(plan.chunks):
            next_chunk = plan.chunks[position + 1]
            end = (
                next_chunk.global_window_start_frame
                - chunk.global_window_start_frame
                + CAUSAL_REFINEMENT_PREROLL_FRAMES
            )
        else:
            end = chunk.model_window_frames
        raw_frames = refinement_raw_frame_count(chunk)
        if not 0 <= start < end <= raw_frames:
            raise SceneChunkAssemblyError(
                f"Chunk {chunk.index} visible range {start}:{end} exceeds "
                f"its {raw_frames}-frame raw refinement."
            )
        audio_start = start + (
            0 if chunk.is_initial else CAUSAL_REFINEMENT_PREROLL_FRAMES
        )
        audio_end = audio_start + (end - start)
        if audio_end > raw_frames:
            raise SceneChunkAssemblyError(
                f"Chunk {chunk.index} audio range {audio_start}:{audio_end} "
                f"exceeds its {raw_frames}-frame raw refinement."
            )
        slices.append(
            ChunkSlice(
                chunk_index=chunk.index,
                start_frame=start,
                end_frame_exclusive=end,
                audio_start_frame=audio_start,
                audio_end_frame_exclusive=audio_end,
            )
        )
    if sum(item.frame_count for item in slices) != plan.generation_master_frames:
        raise SceneChunkAssemblyError(
            "Chunk slices do not cover the generation master exactly."
        )
    return tuple(slices)


def _parse_positive_int(value: object, *, field: str, path: Path) -> int:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError) as error:
        raise SceneChunkAssemblyError(
            f"FFprobe returned an invalid {field} for {path}."
        ) from error
    if parsed < 1:
        raise SceneChunkAssemblyError(
            f"FFprobe returned an invalid {field} for {path}."
        )
    return parsed


def _parse_rate(value: object, *, field: str, path: Path) -> Fraction:
    try:
        parsed = Fraction(str(value))
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise SceneChunkAssemblyError(
            f"FFprobe returned an invalid {field} for {path}."
        ) from error
    if parsed <= 0:
        raise SceneChunkAssemblyError(
            f"FFprobe returned an invalid {field} for {path}."
        )
    return parsed


def _stream(
    streams: Sequence[object],
    codec_type: str,
) -> Mapping[str, object] | None:
    for candidate in streams:
        if (
            isinstance(candidate, Mapping)
            and candidate.get("codec_type") == codec_type
        ):
            return candidate
    return None


def _media_probe(
    path: Path,
    *,
    ffprobe_executable: str,
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> MediaProbe:
    command = [
        ffprobe_executable,
        "-v",
        "error",
        "-count_frames",
        "-show_entries",
        (
            "stream=codec_type,codec_name,profile,width,height,pix_fmt,"
            "r_frame_rate,avg_frame_rate,nb_read_frames,sample_rate,channels"
        ),
        "-of",
        "json",
        str(path),
    ]
    completed = runner(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise SceneChunkAssemblyError(
            f"FFprobe failed for {path}: {completed.stderr.strip()}"
        )
    try:
        document = json.loads(completed.stdout)
        streams = document["streams"]
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise SceneChunkAssemblyError(
            f"FFprobe did not return usable stream data for {path}."
        ) from error
    if not isinstance(streams, list):
        raise SceneChunkAssemblyError(
            f"FFprobe did not return usable stream data for {path}."
        )

    video = _stream(streams, "video")
    if video is None:
        raise SceneChunkAssemblyError(f"{path} has no video stream.")
    audio = _stream(streams, "audio")
    audio_rate = (
        _parse_positive_int(
            audio.get("sample_rate"),
            field="audio sample rate",
            path=path,
        )
        if audio is not None
        else None
    )
    audio_channels = (
        _parse_positive_int(
            audio.get("channels"),
            field="audio channel count",
            path=path,
        )
        if audio is not None
        else None
    )
    return MediaProbe(
        path=path,
        width=_parse_positive_int(video.get("width"), field="width", path=path),
        height=_parse_positive_int(video.get("height"), field="height", path=path),
        r_frame_rate=_parse_rate(
            video.get("r_frame_rate"),
            field="r_frame_rate",
            path=path,
        ),
        avg_frame_rate=_parse_rate(
            video.get("avg_frame_rate"),
            field="avg_frame_rate",
            path=path,
        ),
        decoded_video_frames=_parse_positive_int(
            video.get("nb_read_frames"),
            field="decoded frame count",
            path=path,
        ),
        video_codec=str(video.get("codec_name", "")).lower(),
        video_profile=str(video.get("profile", "")),
        pixel_format=str(video.get("pix_fmt", "")).lower(),
        audio_codec=(
            str(audio.get("codec_name", "")).lower() if audio is not None else None
        ),
        audio_sample_rate=audio_rate,
        audio_channels=audio_channels,
    )


def _validate_fixed_video_profile(probe: MediaProbe) -> None:
    expected_rate = Fraction(PRODUCTION_FPS, 1)
    if probe.width != PRODUCTION_WIDTH or probe.height != PRODUCTION_HEIGHT:
        raise SceneChunkAssemblyError(
            f"{probe.path} is {probe.width}x{probe.height}; expected "
            f"{PRODUCTION_WIDTH}x{PRODUCTION_HEIGHT}."
        )
    if (
        probe.r_frame_rate != expected_rate
        or probe.avg_frame_rate != expected_rate
    ):
        raise SceneChunkAssemblyError(
            f"{probe.path} is not exact CFR {PRODUCTION_FPS}/1 "
            f"(r={probe.r_frame_rate}, avg={probe.avg_frame_rate})."
        )


def _seconds(frames: int) -> str:
    return f"{frames / PRODUCTION_FPS:.9f}"


class SceneChunkAssembler:
    """Assemble raw continuation chunks into one exact-duration scene MP4.

    The orchestrator supplies all input and destination paths.  Paths are
    passed to subprocesses as individual arguments, never interpolated into a
    shell command.  The destination is replaced only after the temporary file
    passes a decoded-frame FFprobe validation.
    """

    AUDIO_EDGE_FADE_SECONDS = 0.100

    def __init__(
        self,
        *,
        ffmpeg_executable: str = "ffmpeg",
        ffprobe_executable: str = "ffprobe",
        ffmpeg_runner: Callable[
            ..., subprocess.CompletedProcess[str]
        ] = subprocess.run,
        ffprobe_runner: Callable[
            ..., subprocess.CompletedProcess[str]
        ] = subprocess.run,
    ):
        self.ffmpeg_executable = ffmpeg_executable
        self.ffprobe_executable = ffprobe_executable
        self._ffmpeg_runner = ffmpeg_runner
        self._ffprobe_runner = ffprobe_runner

    def assemble(
        self,
        plan: SceneFramePlan,
        raw_chunks: Sequence[str | Path],
        destination: str | Path,
    ) -> Path:
        if plan.fps != PRODUCTION_FPS:
            raise SceneChunkAssemblyError(
                f"Scene plan is {plan.fps} fps; expected {PRODUCTION_FPS}."
            )
        if len(raw_chunks) != plan.chunk_count:
            raise SceneChunkAssemblyError(
                f"Expected {plan.chunk_count} raw chunks; received "
                f"{len(raw_chunks)}."
            )

        inputs: list[Path] = []
        for raw_path in raw_chunks:
            candidate = Path(raw_path).expanduser()
            try:
                resolved = candidate.resolve(strict=True)
            except OSError as error:
                raise SceneChunkAssemblyError(
                    f"Raw continuation chunk is missing: {candidate}"
                ) from error
            if not resolved.is_file():
                raise SceneChunkAssemblyError(
                    f"Raw continuation chunk is not a file: {resolved}"
                )
            inputs.append(resolved)

        output = Path(destination).expanduser().resolve(strict=False)
        if output.suffix.lower() != ".mp4":
            raise SceneChunkAssemblyError("Scene destination must be an .mp4 file.")
        if output.exists() and not output.is_file():
            raise SceneChunkAssemblyError(
                f"Scene destination is not a file: {output}"
            )
        if output in inputs:
            raise SceneChunkAssemblyError(
                "Scene destination cannot overwrite a raw input chunk."
            )
        output.parent.mkdir(parents=True, exist_ok=True)

        slices = chunk_slices(plan)
        for path, chunk, visible in zip(inputs, plan.chunks, slices, strict=True):
            probe = self.validate_chunk(plan, chunk.index, path)
            if visible.end_frame_exclusive > probe.decoded_video_frames:
                raise SceneChunkAssemblyError(
                    f"{path} is too short for visible range "
                    f"{visible.start_frame}:{visible.end_frame_exclusive}."
                )

        temporary = output.with_name(
            f".{output.stem}.assembling-{uuid4().hex}.mp4"
        )
        command = self._command(plan, inputs, slices, temporary)
        try:
            completed = self._ffmpeg_runner(
                command,
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                raise SceneChunkAssemblyError(
                    f"FFmpeg chunk assembly failed: {completed.stderr.strip()}"
                )
            if not temporary.is_file():
                raise SceneChunkAssemblyError(
                    "FFmpeg reported success but did not create the scene file."
                )
            assembled = _media_probe(
                temporary,
                ffprobe_executable=self.ffprobe_executable,
                runner=self._ffprobe_runner,
            )
            self._validate_output(plan, assembled)
            os.replace(temporary, output)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        return output

    def validate_chunk(
        self,
        plan: SceneFramePlan,
        chunk_index: int,
        raw_chunk: str | Path,
    ) -> MediaProbe:
        """Verify one finalized raw worker chunk before accepting its lineage."""
        if (
            isinstance(chunk_index, bool)
            or not isinstance(chunk_index, int)
            or not 0 <= chunk_index < plan.chunk_count
        ):
            raise SceneChunkAssemblyError("chunk_index is outside the scene plan.")
        path = Path(raw_chunk).expanduser()
        try:
            resolved = path.resolve(strict=True)
        except OSError as error:
            raise SceneChunkAssemblyError(
                f"Raw continuation chunk is missing: {path}"
            ) from error
        if not resolved.is_file():
            raise SceneChunkAssemblyError(
                f"Raw continuation chunk is not a file: {resolved}"
            )
        probe = _media_probe(
            resolved,
            ffprobe_executable=self.ffprobe_executable,
            runner=self._ffprobe_runner,
        )
        _validate_fixed_video_profile(probe)
        if probe.audio_codec is None:
            raise SceneChunkAssemblyError(
                f"{resolved} has no audio stream; chunk assembly requires audio."
            )
        if probe.video_codec != "ffv1" or probe.pixel_format != "yuv444p":
            raise SceneChunkAssemblyError(
                f"{resolved} must be a lossless FFV1/yuv444p continuation "
                f"master; received {probe.video_codec or 'unknown'}/"
                f"{probe.pixel_format or 'unknown'}."
            )
        expected_frames = refinement_raw_frame_count(plan.chunks[chunk_index])
        if probe.decoded_video_frames != expected_frames:
            raise SceneChunkAssemblyError(
                f"{resolved} has {probe.decoded_video_frames} decoded frames; "
                f"expected {expected_frames} for chunk {chunk_index}."
            )
        return probe

    def validate_scene(
        self,
        plan: SceneFramePlan,
        scene_path: str | Path,
    ) -> MediaProbe:
        """Verify an assembled scene before recovery reuses it."""
        path = Path(scene_path).expanduser()
        try:
            resolved = path.resolve(strict=True)
        except OSError as error:
            raise SceneChunkAssemblyError(
                f"Assembled continuation scene is missing: {path}"
            ) from error
        if not resolved.is_file():
            raise SceneChunkAssemblyError(
                f"Assembled continuation scene is not a file: {resolved}"
            )
        probe = _media_probe(
            resolved,
            ffprobe_executable=self.ffprobe_executable,
            runner=self._ffprobe_runner,
        )
        self._validate_output(plan, probe)
        return probe

    def _command(
        self,
        plan: SceneFramePlan,
        inputs: Sequence[Path],
        slices: Sequence[ChunkSlice],
        temporary: Path,
    ) -> list[str]:
        command = [self.ffmpeg_executable, "-hide_banner", "-y"]
        for path in inputs:
            command.extend(("-i", str(path)))

        filters: list[str] = []
        concat_inputs: list[str] = []
        last_index = len(slices) - 1
        for index, visible in enumerate(slices):
            duration = visible.frame_count / PRODUCTION_FPS
            audio_start = visible.audio_start_frame / PRODUCTION_FPS
            audio_end = visible.audio_end_frame_exclusive / PRODUCTION_FPS
            filters.append(
                f"[{index}:v]trim=start_frame={visible.start_frame}:"
                f"end_frame={visible.end_frame_exclusive},"
                f"setpts=PTS-STARTPTS,setsar=1[v{index}]"
            )
            audio_filters = [
                f"[{index}:a]atrim=start={audio_start:.9f}:end={audio_end:.9f}",
                "asetpts=PTS-STARTPTS",
                "aresample=48000",
                (
                    "aformat=sample_fmts=fltp:sample_rates=48000:"
                    "channel_layouts=stereo"
                ),
                f"apad=pad_dur={duration:.9f}",
                f"atrim=duration={duration:.9f}",
            ]
            fade = min(self.AUDIO_EDGE_FADE_SECONDS, duration / 2)
            if index > 0 and fade > 0:
                audio_filters.append(
                    f"afade=t=in:st=0:d={fade:.9f}:curve=qsin"
                )
            if index < last_index and fade > 0:
                audio_filters.append(
                    f"afade=t=out:st={duration - fade:.9f}:"
                    f"d={fade:.9f}:curve=qsin"
                )
            filters.append(",".join(audio_filters) + f"[a{index}]")
            concat_inputs.extend((f"[v{index}]", f"[a{index}]"))

        filters.append(
            "".join(concat_inputs)
            + f"concat=n={len(slices)}:v=1:a=1[vcat][acat]"
        )
        filters.append(
            f"[vcat]trim=start_frame=0:end_frame={plan.timeline_output_frames},"
            "setpts=PTS-STARTPTS[vout]"
        )
        filters.append(
            f"[acat]atrim=duration={_seconds(plan.timeline_output_frames)},"
            "asetpts=PTS-STARTPTS[aout]"
        )
        command.extend(
            (
                "-filter_complex",
                ";".join(filters),
                "-map",
                "[vout]",
                "-map",
                "[aout]",
                "-c:v",
                "libx264",
                "-profile:v",
                "high",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "19",
                "-preset",
                "slow",
                "-r",
                str(PRODUCTION_FPS),
                "-fps_mode",
                "cfr",
                "-g",
                "48",
                "-keyint_min",
                "48",
                "-sc_threshold",
                "0",
                "-x264-params",
                "open-gop=0",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-ar",
                "48000",
                "-ac",
                "2",
                "-movflags",
                "+faststart",
                str(temporary),
            )
        )
        return command

    @staticmethod
    def _validate_output(plan: SceneFramePlan, probe: MediaProbe) -> None:
        _validate_fixed_video_profile(probe)
        if probe.decoded_video_frames != plan.timeline_output_frames:
            raise SceneChunkAssemblyError(
                f"Assembled scene has {probe.decoded_video_frames} decoded "
                f"frames; expected exactly {plan.timeline_output_frames}."
            )
        if probe.video_codec != "h264":
            raise SceneChunkAssemblyError(
                f"Assembled scene codec is {probe.video_codec or 'unknown'}; "
                "expected H.264."
            )
        if probe.video_profile.lower() != "high":
            raise SceneChunkAssemblyError(
                f"Assembled scene profile is {probe.video_profile or 'unknown'}; "
                "expected High."
            )
        if probe.pixel_format != "yuv420p":
            raise SceneChunkAssemblyError(
                f"Assembled scene pixel format is "
                f"{probe.pixel_format or 'unknown'}; expected yuv420p."
            )
        if probe.audio_codec != "aac":
            raise SceneChunkAssemblyError(
                f"Assembled scene audio codec is "
                f"{probe.audio_codec or 'missing'}; expected AAC."
            )
        if probe.audio_sample_rate != 48000:
            raise SceneChunkAssemblyError(
                f"Assembled scene audio rate is "
                f"{probe.audio_sample_rate or 'missing'}; expected 48000 Hz."
            )
        if probe.audio_channels != 2:
            raise SceneChunkAssemblyError(
                f"Assembled scene audio has "
                f"{probe.audio_channels or 'missing'} channels; expected stereo."
            )
