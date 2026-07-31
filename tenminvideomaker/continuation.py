"""Deterministic temporal planning for crash-resumable LTX continuation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Iterable, Mapping, Sequence

from .constants import PRODUCTION_FPS

CONTINUATION_SCHEMA_VERSION = 1
CONTINUATION_FEATURE_FLAG = "ltx_chunked_continuation_v1"
CONTINUATION_STRATEGY = "ltx23_latent_overlap_v1"
SEED_POLICY = "derived_v1"

BASE_WINDOW_TRANSITIONS = 120
OVERLAP_PIXEL_FRAMES = 24
FULL_EXTENSION_NEW_TRANSITIONS = BASE_WINDOW_TRANSITIONS - OVERLAP_PIXEL_FRAMES
LTX_TEMPORAL_STEP = 8
CAUSAL_REFINEMENT_PREROLL_FRAMES = 8


class ContinuationPlanError(ValueError):
    """Raised when a requested timeline cannot produce a valid LTX plan."""


@dataclass(frozen=True)
class TemporalSegment:
    """One optional Grok-authored beat on the scene's global timeline."""

    index: int
    start_frame: int
    end_frame_exclusive: int
    positive_prompt: str
    negative_prompt_additions: tuple[str, ...] = ()
    seed_override: int | None = None
    variation_index: int = 0

    def overlaps(self, start_frame: int, end_frame_exclusive: int) -> bool:
        return self.start_frame < end_frame_exclusive and start_frame < self.end_frame_exclusive


@dataclass(frozen=True)
class ContinuationChunkPlan:
    """Node-native temporal accounting for one first-pass model invocation."""

    index: int
    new_transition_frames: int
    model_window_frames: int
    global_window_start_frame: int
    global_window_end_frame_exclusive: int
    global_new_start_frame: int
    global_new_end_frame_exclusive: int
    cumulative_master_frames: int
    seed: int
    prompt: str
    negative: str
    segment_indices: tuple[int, ...]
    prompt_segmentation_quality: str
    variation_index: int = 0

    @property
    def is_initial(self) -> bool:
        return self.index == 0


@dataclass(frozen=True)
class SceneFramePlan:
    """Exact requested timeline and the LTX-compatible generation master."""

    schema_version: int
    feature_flag: str
    strategy: str
    fps: int
    requested_duration_seconds: float
    timeline_output_frames: int
    generation_transition_frames: int
    generation_master_frames: int
    base_window_transitions: int
    overlap_pixel_frames: int
    chunks: tuple[ContinuationChunkPlan, ...]

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @property
    def requires_continuation(self) -> bool:
        return len(self.chunks) > 1

    def to_document(self) -> dict[str, object]:
        return asdict(self)

    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.to_document(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ChunkAssemblySpan:
    """The exact local frame range committed from one refined chunk window."""

    chunk_index: int
    input_start_frame: int
    input_end_frame_exclusive: int
    master_start_frame: int
    master_end_frame_exclusive: int

    @property
    def frame_count(self) -> int:
        return self.input_end_frame_exclusive - self.input_start_frame


def timeline_frame_count(seconds: float, *, fps: int = PRODUCTION_FPS) -> int:
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        raise ContinuationPlanError("requested duration must be numeric.")
    if not math.isfinite(float(seconds)) or seconds <= 0:
        raise ContinuationPlanError("requested duration must be greater than zero.")
    if isinstance(fps, bool) or not isinstance(fps, int) or fps < 1:
        raise ContinuationPlanError("fps must be a positive integer.")
    return max(1, round(float(seconds) * fps))


def generation_transition_count(timeline_frames: int) -> int:
    if (
        isinstance(timeline_frames, bool)
        or not isinstance(timeline_frames, int)
        or timeline_frames < 1
    ):
        raise ContinuationPlanError("timeline_frames must be a positive integer.")
    return LTX_TEMPORAL_STEP * math.ceil(
        max(timeline_frames - 1, 0) / LTX_TEMPORAL_STEP
    )


def transition_contributions(total_transitions: int) -> tuple[int, ...]:
    """Split an 8-aligned scene into one base window plus continuation increments."""
    if (
        isinstance(total_transitions, bool)
        or not isinstance(total_transitions, int)
        or total_transitions < 0
        or total_transitions % LTX_TEMPORAL_STEP
    ):
        raise ContinuationPlanError(
            "generation transitions must be a non-negative multiple of 8."
        )
    if total_transitions <= BASE_WINDOW_TRANSITIONS:
        return (total_transitions,)

    contributions = [BASE_WINDOW_TRANSITIONS]
    remaining = total_transitions - BASE_WINDOW_TRANSITIONS
    while remaining:
        contribution = min(FULL_EXTENSION_NEW_TRANSITIONS, remaining)
        if contribution % LTX_TEMPORAL_STEP:
            raise ContinuationPlanError(
                "continuation contribution must remain a multiple of 8."
            )
        contributions.append(contribution)
        remaining -= contribution
    return tuple(contributions)


def derived_chunk_seed(
    *,
    job_id: str,
    scene_id: int,
    revision: int,
    base_seed: int,
    chunk_index: int,
    prompt: str,
    variation_index: int = 0,
) -> int:
    """Map stable scene inputs to an unsigned 64-bit per-chunk seed."""
    if not 0 <= base_seed <= 0xFFFFFFFFFFFFFFFF:
        raise ContinuationPlanError("base_seed must fit in an unsigned 64-bit integer.")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (scene_id, revision, chunk_index, variation_index)
    ):
        raise ContinuationPlanError("seed coordinates must be non-negative integers.")
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    material = "\0".join(
        (
            SEED_POLICY,
            job_id,
            str(scene_id),
            str(revision),
            str(base_seed),
            str(chunk_index),
            prompt_hash,
            str(variation_index),
        )
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big", signed=False)


def materialize_segments(
    raw_segments: Sequence[Mapping[str, object]] | None,
    *,
    timeline_frames: int,
    fps: int = PRODUCTION_FPS,
) -> tuple[TemporalSegment, ...]:
    """Convert optional beat durations into deterministic contiguous frame ranges."""
    if not raw_segments:
        return ()

    segments: list[TemporalSegment] = []
    cursor = 0
    for position, raw in enumerate(raw_segments):
        index_value = raw.get("index", position)
        if (
            isinstance(index_value, bool)
            or not isinstance(index_value, int)
            or index_value < 0
        ):
            raise ContinuationPlanError(f"segments[{position}].index must be non-negative.")
        duration_value = raw.get("requested_duration_seconds")
        transition_value = raw.get("new_transition_frames")
        if (duration_value is None) == (transition_value is None):
            raise ContinuationPlanError(
                f"segments[{position}] requires exactly one of "
                "requested_duration_seconds or new_transition_frames."
            )
        if transition_value is not None:
            if (
                isinstance(transition_value, bool)
                or not isinstance(transition_value, int)
                or transition_value < 0
                or transition_value % LTX_TEMPORAL_STEP
            ):
                raise ContinuationPlanError(
                    f"segments[{position}].new_transition_frames must be a "
                    "non-negative multiple of 8."
                )
            frame_length = max(1, transition_value)
        elif duration_value is not None:
            frame_length = timeline_frame_count(float(duration_value), fps=fps)
        prompt = str(raw.get("positive_prompt", "")).strip()
        if not prompt:
            raise ContinuationPlanError(
                f"segments[{position}].positive_prompt must be non-empty."
            )
        additions_value = raw.get("negative_prompt_additions", ())
        if not isinstance(additions_value, (list, tuple)) or not all(
            isinstance(item, str) and item.strip() for item in additions_value
        ):
            raise ContinuationPlanError(
                f"segments[{position}].negative_prompt_additions must be strings."
            )
        seed_override_value = raw.get("seed_override")
        if seed_override_value is not None and (
            isinstance(seed_override_value, bool)
            or not isinstance(seed_override_value, int)
            or not 0 <= seed_override_value <= 0xFFFFFFFFFFFFFFFF
        ):
            raise ContinuationPlanError(
                f"segments[{position}].seed_override must fit unsigned 64-bit."
            )
        variation_value = raw.get("variation_index", 0)
        if (
            isinstance(variation_value, bool)
            or not isinstance(variation_value, int)
            or variation_value < 0
        ):
            raise ContinuationPlanError(
                f"segments[{position}].variation_index must be non-negative."
            )

        end = cursor + frame_length
        segments.append(
            TemporalSegment(
                index=index_value,
                start_frame=cursor,
                end_frame_exclusive=end,
                positive_prompt=prompt,
                negative_prompt_additions=tuple(
                    item.strip() for item in additions_value
                ),
                seed_override=seed_override_value,
                variation_index=variation_value,
            )
        )
        cursor = end

    drift = timeline_frames - cursor
    rounding_allowance = max(1, math.ceil(len(segments) / 2))
    if abs(drift) > rounding_allowance:
        raise ContinuationPlanError(
            "Explicit segments must cover the complete scene timeline; "
            f"coverage differs by {abs(drift)} frame(s), beyond the "
            f"{rounding_allowance}-frame rounding allowance."
        )
    if drift:
        final = segments[-1]
        adjusted_end = final.end_frame_exclusive + drift
        if adjusted_end <= final.start_frame:
            raise ContinuationPlanError(
                "Segment rounding would collapse the final temporal beat."
            )
        segments[-1] = TemporalSegment(
            index=final.index,
            start_frame=final.start_frame,
            end_frame_exclusive=adjusted_end,
            positive_prompt=final.positive_prompt,
            negative_prompt_additions=final.negative_prompt_additions,
            seed_override=final.seed_override,
            variation_index=final.variation_index,
        )
    return tuple(segments)


def _continuity_prefix(continuity: Mapping[str, object] | None) -> str:
    if not continuity:
        return ""
    ordered_fields = (
        "identity_anchors",
        "wardrobe_anchors",
        "environment_anchors",
        "camera_axis",
        "screen_direction",
    )
    values: list[str] = []
    for field in ordered_fields:
        value = continuity.get(field)
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
        elif isinstance(value, (list, tuple)):
            values.extend(
                item.strip()
                for item in value
                if isinstance(item, str) and item.strip()
            )
    return " ".join(values)


def _resolve_chunk_prompt(
    *,
    fallback_prompt: str,
    fallback_negative: str,
    continuity: Mapping[str, object] | None,
    segments: Sequence[TemporalSegment],
    window_start: int,
    window_end_exclusive: int,
    is_initial: bool,
) -> tuple[str, str, tuple[int, ...], str, int | None, int]:
    anchors = _continuity_prefix(continuity)
    matching = tuple(
        segment
        for segment in segments
        if segment.overlaps(window_start, window_end_exclusive)
    )
    if matching:
        beats = " Then ".join(segment.positive_prompt for segment in matching)
        prompt_parts = [anchors, beats]
        quality = "explicit_segments"
        additions = tuple(
            addition
            for segment in matching
            for addition in segment.negative_prompt_additions
        )
        seed_override = matching[0].seed_override
        variation_index = matching[0].variation_index
        indices = tuple(segment.index for segment in matching)
    else:
        prompt_parts = [anchors, fallback_prompt]
        quality = "fallback_reused_prompt"
        additions = ()
        seed_override = None
        variation_index = 0
        indices = ()
    if not is_initial:
        prompt_parts.insert(
            1,
            "Continue seamlessly from the preceding frames without restarting "
            "pose, motion, camera movement, wardrobe, lighting, or environment.",
        )
    prompt = " ".join(part.strip() for part in prompt_parts if part and part.strip())
    negative = ", ".join((fallback_negative, *additions))
    return prompt, negative, indices, quality, seed_override, variation_index


def build_scene_frame_plan(
    *,
    job_id: str,
    scene_id: int,
    revision: int,
    requested_duration_seconds: float,
    base_seed: int,
    fallback_prompt: str,
    fallback_negative: str,
    continuity: Mapping[str, object] | None = None,
    raw_segments: Sequence[Mapping[str, object]] | None = None,
    fps: int = PRODUCTION_FPS,
) -> SceneFramePlan:
    timeline_frames = timeline_frame_count(requested_duration_seconds, fps=fps)
    total_transitions = generation_transition_count(timeline_frames)
    master_frames = total_transitions + 1
    segments = materialize_segments(
        raw_segments,
        timeline_frames=timeline_frames,
        fps=fps,
    )

    chunks: list[ContinuationChunkPlan] = []
    cumulative_transitions = 0
    for index, contribution in enumerate(transition_contributions(total_transitions)):
        previous_transitions = cumulative_transitions
        cumulative_transitions += contribution
        if index == 0:
            window_start = 0
            model_window_frames = contribution + 1
            new_start = 0
        else:
            window_start = max(0, previous_transitions - OVERLAP_PIXEL_FRAMES)
            model_window_frames = OVERLAP_PIXEL_FRAMES + contribution + 1
            new_start = previous_transitions
        window_end = cumulative_transitions + 1
        prompt, negative, segment_indices, quality, seed_override, variation = (
            _resolve_chunk_prompt(
                fallback_prompt=fallback_prompt,
                fallback_negative=fallback_negative,
                continuity=continuity,
                segments=segments,
                window_start=window_start,
                window_end_exclusive=window_end,
                is_initial=index == 0,
            )
        )
        seed = (
            seed_override
            if seed_override is not None
            else derived_chunk_seed(
                job_id=job_id,
                scene_id=scene_id,
                revision=revision,
                base_seed=base_seed,
                chunk_index=index,
                prompt=prompt,
                variation_index=variation,
            )
        )
        chunks.append(
            ContinuationChunkPlan(
                index=index,
                new_transition_frames=contribution,
                model_window_frames=model_window_frames,
                global_window_start_frame=window_start,
                global_window_end_frame_exclusive=window_end,
                global_new_start_frame=new_start,
                global_new_end_frame_exclusive=cumulative_transitions + 1,
                cumulative_master_frames=cumulative_transitions + 1,
                seed=seed,
                prompt=prompt,
                negative=negative,
                segment_indices=segment_indices,
                prompt_segmentation_quality=quality,
                variation_index=variation,
            )
        )

    return SceneFramePlan(
        schema_version=CONTINUATION_SCHEMA_VERSION,
        feature_flag=CONTINUATION_FEATURE_FLAG,
        strategy=CONTINUATION_STRATEGY,
        fps=fps,
        requested_duration_seconds=float(requested_duration_seconds),
        timeline_output_frames=timeline_frames,
        generation_transition_frames=total_transitions,
        generation_master_frames=master_frames,
        base_window_transitions=BASE_WINDOW_TRANSITIONS,
        overlap_pixel_frames=OVERLAP_PIXEL_FRAMES,
        chunks=tuple(chunks),
    )


def chunk_plan_documents(
    chunks: Iterable[ContinuationChunkPlan],
) -> tuple[dict[str, object], ...]:
    return tuple(asdict(chunk) for chunk in chunks)


def assembly_spans(plan: SceneFramePlan) -> tuple[ChunkAssemblySpan, ...]:
    """Commit fused overlaps from the later window without duplicate frames.

    Every non-final window keeps only the frames before the following window
    starts. The final window keeps its entire overlap-plus-new range. This is
    the one-overlap delayed-commit policy: a later refined window owns the
    visible overlap that conditioned it.
    """
    spans: list[ChunkAssemblySpan] = []
    master_cursor = 0
    for position, chunk in enumerate(plan.chunks):
        if position + 1 < len(plan.chunks):
            next_chunk = plan.chunks[position + 1]
            local_end = (
                next_chunk.global_window_start_frame
                - chunk.global_window_start_frame
            )
        else:
            local_end = chunk.model_window_frames
        if not 0 < local_end <= chunk.model_window_frames:
            raise ContinuationPlanError(
                f"Chunk {chunk.index} has an invalid delayed-commit range."
            )
        span = ChunkAssemblySpan(
            chunk_index=chunk.index,
            input_start_frame=0,
            input_end_frame_exclusive=local_end,
            master_start_frame=master_cursor,
            master_end_frame_exclusive=master_cursor + local_end,
        )
        spans.append(span)
        master_cursor = span.master_end_frame_exclusive
    if master_cursor != plan.generation_master_frames:
        raise ContinuationPlanError(
            "Delayed-commit ranges do not cover the generation master exactly."
        )
    return tuple(spans)


def handoff_latent_token_count(chunk: ContinuationChunkPlan) -> int:
    """Exact bounded first-pass tokens persisted for one continuation window."""
    window_tokens = ((chunk.model_window_frames - 1) // LTX_TEMPORAL_STEP) + 1
    return window_tokens + (0 if chunk.is_initial else 1)


def refinement_raw_frame_count(chunk: ContinuationChunkPlan) -> int:
    """Frames emitted before removing the causal slice preroll."""
    return chunk.model_window_frames + (
        0 if chunk.is_initial else CAUSAL_REFINEMENT_PREROLL_FRAMES
    )


def continuation_is_enabled(
    *,
    scene_frame_count: int,
    continuation: Mapping[str, object] | None,
    mode: str,
) -> bool:
    """Resolve the rollout flag without invalidating completed legacy clips."""
    normalized_mode = mode.strip().casefold()
    if normalized_mode not in {"disabled", "explicit", "auto"}:
        raise ContinuationPlanError(
            "continuation mode must be disabled, explicit, or auto."
        )
    if scene_frame_count <= BASE_WINDOW_TRANSITIONS + 1:
        return False
    explicit = None if continuation is None else continuation.get("enabled", True)
    if explicit is not None and not isinstance(explicit, bool):
        raise ContinuationPlanError("continuation.enabled must be a boolean.")
    if normalized_mode == "disabled":
        return False
    if explicit is False:
        return False
    if normalized_mode == "explicit":
        return explicit is True
    return True
