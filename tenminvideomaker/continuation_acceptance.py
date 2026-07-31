"""Deterministic, bounded inputs for the continuation GPU acceptance matrix."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any, Mapping

from .continuation import ContinuationChunkPlan, SceneFramePlan, build_scene_frame_plan
from .contracts import JobPayload, parse_job_payload

ACCEPTANCE_DURATION_SECONDS = 9.0
ACCEPTANCE_FIRST_BEAT_SECONDS = 5.0

_FIRST_BEAT = (
    "A clearly adult, fully clothed person walks steadily toward screen right "
    "through a softly lit interior while keeping one hand on a handled travel bag. "
    "The camera tracks laterally at a slow constant speed; the subject's visible "
    "hands, clothing, and stable directional lighting remain clear."
)
_SECOND_BEAT = (
    "The same clearly adult, fully clothed person keeps walking toward screen right "
    "without pausing, keeps one hand on the handled travel bag, gradually turns the "
    "torso toward camera, and continues through the stable directional light while "
    "the slow lateral camera tracking never stops."
)


class ContinuationAcceptanceError(ValueError):
    """Raised when a supplied production source cannot form the bounded matrix."""


@dataclass(frozen=True)
class AcceptancePlans:
    """Shared base plus comparable continuation-window inputs."""

    full: SceneFramePlan
    diagnostic_plan: SceneFramePlan
    base: ContinuationChunkPlan
    latent_overlap: ContinuationChunkPlan
    diagnostic: ContinuationChunkPlan


def build_acceptance_job(
    source_payload: Mapping[str, Any],
    *,
    source_scene_id: int,
    acceptance_job_id: str,
) -> JobPayload:
    """Copy one source scene, retaining only I2V-compatible production assets."""
    if not isinstance(source_payload, Mapping):
        raise ContinuationAcceptanceError("source_payload must be an object.")
    if isinstance(source_scene_id, bool) or not isinstance(source_scene_id, int):
        raise ContinuationAcceptanceError("source_scene_id must be an integer.")
    raw = deepcopy(dict(source_payload))
    scenes = raw.get("scenes")
    if not isinstance(scenes, list):
        raise ContinuationAcceptanceError("source_payload.scenes must be an array.")
    selected = next(
        (
            deepcopy(scene)
            for scene in scenes
            if isinstance(scene, Mapping) and scene.get("id") == source_scene_id
        ),
        None,
    )
    if selected is None:
        raise ContinuationAcceptanceError(
            f"source scene {source_scene_id} does not exist in the source payload."
        )
    i2v = selected.get("i2v")
    if not isinstance(i2v, dict):
        raise ContinuationAcceptanceError("selected scene must contain an i2v object.")

    raw["job_id"] = acceptance_job_id
    raw["scenes"] = [selected]
    raw["total_estimated_sec"] = ACCEPTANCE_DURATION_SECONDS
    selected["estimated_sec"] = ACCEPTANCE_DURATION_SECONDS
    i2v["prompt"] = _FIRST_BEAT
    i2v["continuity"] = {
        "identity_anchors": ["The same clearly adult person remains consistent."],
        "wardrobe_anchors": ["The same fully clothed wardrobe remains unchanged."],
        "environment_anchors": ["The interior and directional lighting remain stable."],
        "camera_axis": "Preserve the lateral tracking axis.",
        "screen_direction": "Movement continues toward screen right.",
    }
    i2v["segments"] = [
        {
            "index": 0,
            "requested_duration_seconds": ACCEPTANCE_FIRST_BEAT_SECONDS,
            "positive_prompt": _FIRST_BEAT,
            "negative_prompt_additions": ["motion restart", "scene cut"],
            "variation_index": 0,
        },
        {
            "index": 1,
            "requested_duration_seconds": (
                ACCEPTANCE_DURATION_SECONDS - ACCEPTANCE_FIRST_BEAT_SECONDS
            ),
            "positive_prompt": _SECOND_BEAT,
            "negative_prompt_additions": ["motion restart", "pose reset", "scene cut"],
            "variation_index": 0,
        },
    ]
    i2v["continuation"] = {
        "enabled": True,
        "strategy": "ltx23_latent_overlap_v1",
        "fps": 24,
        "base_window_transition_frames": 120,
        "overlap_transition_frames": 24,
        "seed_policy": "derived_v1",
    }
    return parse_job_payload(raw)


def build_acceptance_plans(job: JobPayload, *, revision: int) -> AcceptancePlans:
    """Build a shared 121-frame base and equivalent second-window diagnostics."""
    if len(job.scenes) != 1:
        raise ContinuationAcceptanceError("acceptance job must contain exactly one scene.")
    scene = job.scenes[0]
    full = build_scene_frame_plan(
        job_id=job.job_id,
        scene_id=scene.scene_id,
        revision=revision,
        requested_duration_seconds=scene.estimated_sec,
        base_seed=scene.i2v.seed,
        fallback_prompt=scene.i2v.prompt,
        fallback_negative=scene.i2v.negative,
        continuity=scene.i2v.continuity,
        raw_segments=scene.i2v.segments,
    )
    if full.chunk_count != 2:
        raise ContinuationAcceptanceError(
            "acceptance job must produce exactly one base and one continuation window."
        )
    base, latent_overlap = full.chunks
    if base.model_window_frames != 121 or latent_overlap.model_window_frames != 121:
        raise ContinuationAcceptanceError(
            "acceptance windows must both be exactly 121 frames."
        )

    diagnostic_plan = build_scene_frame_plan(
        job_id=job.job_id,
        scene_id=scene.scene_id,
        revision=revision,
        requested_duration_seconds=ACCEPTANCE_FIRST_BEAT_SECONDS,
        base_seed=scene.i2v.seed,
        fallback_prompt=latent_overlap.prompt,
        fallback_negative=latent_overlap.negative,
    )
    if diagnostic_plan.chunk_count != 1:
        raise ContinuationAcceptanceError("diagnostic continuation must fit one window.")
    diagnostic = replace(
        diagnostic_plan.chunks[0],
        seed=latent_overlap.seed,
        prompt=latent_overlap.prompt,
        negative=latent_overlap.negative,
        segment_indices=latent_overlap.segment_indices,
        prompt_segmentation_quality=latent_overlap.prompt_segmentation_quality,
        variation_index=latent_overlap.variation_index,
    )
    diagnostic_plan = replace(diagnostic_plan, chunks=(diagnostic,))
    return AcceptancePlans(
        full=full,
        diagnostic_plan=diagnostic_plan,
        base=base,
        latent_overlap=latent_overlap,
        diagnostic=diagnostic,
    )
