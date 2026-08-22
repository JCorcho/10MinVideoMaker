"""Human-readable scene parameter documents and strict edit validation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import math
from typing import Any, Mapping

from .constants import (
    I2V_FIRST_PASS_SIGMAS,
    I2V_FIRST_PASS_SAMPLER,
    I2V_UPSCALE_PASS_SAMPLER,
    I2V_SPATIAL_UPSCALER,
    I2V_UPSCALE_PASS_SIGMAS,
    LTX_CHECKPOINT,
    LTX_TEXT_ENCODER,
    MANDATORY_I2V_LORAS,
    PRODUCTION_FPS,
    PRODUCTION_HEIGHT,
    PRODUCTION_WIDTH,
    frame_count_for_seconds,
)
from .contracts import (
    ContractValidationError,
    JobPayload,
    LoraSpec,
    SceneSpec,
    parse_job_payload,
)


class ReviewValidationError(ValueError):
    """Raised when a human-edited scene document is not render-safe."""


@dataclass(frozen=True)
class SceneWorkflowOverrides:
    t2i_passes: tuple[Mapping[str, Any], ...]
    face_detailer: Mapping[str, Any] | None
    i2v_first_pass: Mapping[str, Any]
    i2v_second_pass: Mapping[str, Any]
    chunking: Mapping[str, Any]
    spatial_upscaler: Mapping[str, Any]
    temporal_continuation: Mapping[str, Any] | None
    continuity: Mapping[str, Any] | None
    segments: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class ValidatedSceneEdit:
    job: JobPayload
    scene: SceneSpec
    workflow: SceneWorkflowOverrides
    document: Mapping[str, Any]


def _lora_document(lora: LoraSpec) -> dict[str, Any]:
    result: dict[str, Any] = {
        "name": lora.name,
        "download_url": lora.download_url,
        "weight": lora.weight,
    }
    if lora.model_id is not None:
        result["model_id"] = lora.model_id
    if lora.version_id is not None:
        result["version_id"] = lora.version_id
    return result


def _review_duration_seconds(scene: SceneSpec) -> float:
    """Expose the duration that an existing continuation revision actually used.

    Older schema-v2 payloads could carry a second duration inside
    ``i2v.continuation``.  The browser has one duration control, so fold that
    legacy override into the visible scene value and omit the hidden duplicate
    from all newly edited revisions.
    """
    continuation = scene.i2v.continuation
    if continuation is not None:
        requested = continuation.get("requested_duration_seconds")
        if isinstance(requested, (int, float)) and not isinstance(requested, bool):
            return float(requested)
    return scene.estimated_sec


def _continuation_document(scene: SceneSpec) -> dict[str, Any] | None:
    if scene.i2v.continuation is None:
        return None
    result = deepcopy(dict(scene.i2v.continuation))
    result.pop("requested_duration_seconds", None)
    return result


def _segment_documents(scene: SceneSpec) -> list[dict[str, Any]]:
    result = [deepcopy(dict(item)) for item in scene.i2v.segments]
    for segment in result:
        if segment.get("seed_override") is not None:
            # JSON numbers above 2**53 cannot cross the browser boundary
            # losslessly.  Keep the typed SceneSpec integer for rendering and
            # expose an exact decimal string to every revision/editor surface.
            segment["seed_override"] = str(segment["seed_override"])
    return result


def _continuation_frame_profile(seconds: float) -> tuple[int, int]:
    timeline_frames = max(1, round(float(seconds) * PRODUCTION_FPS))
    generation_master_frames = max(
        9,
        8 * math.ceil(max(timeline_frames - 1, 0) / 8) + 1,
    )
    return timeline_frames, generation_master_frames


def _default_t2i(job: JobPayload, scene: SceneSpec) -> dict[str, Any]:
    if job.character.base_model.casefold() == "anima":
        passes = [
            {
                "name": "Anima pass",
                "sampler": "er_sde",
                "scheduler": "beta57",
                "steps": 30,
                "cfg": 4.5,
                "denoise": 1.0,
            }
        ]
        face_detailer = None
    else:
        passes = [
            {
                "name": "Pony first pass",
                "sampler": "res_3m_ode",
                "scheduler": "karras",
                "steps": 30,
                "cfg": 6.0,
                "start_step": 0,
                "end_step": 30,
                "add_noise": True,
                "return_with_leftover_noise": False,
            },
            {
                "name": "Pony second pass",
                "sampler": "res_5s_ode",
                "scheduler": "karras",
                "steps": 30,
                "cfg": 6.0,
                "start_step": 0,
                "end_step": 30,
                "add_noise": True,
                "return_with_leftover_noise": False,
            },
        ]
        face_detailer = {
            "enabled": True,
            "detector": "bbox/face_yolov8m.pt",
            "guide_size": 512,
            "max_size": 1024,
            "steps": 20,
            "cfg": 5.0,
            "sampler": "dpmpp_2m_sde",
            "scheduler": "karras",
            "denoise": 0.38,
            "feather": 5,
            "bbox_threshold": 0.5,
            "bbox_dilation": 10,
            "bbox_crop_factor": 3.0,
            "drop_size": 120,
            "noise_mask_feather": 20,
        }
    return {
        "prompt": scene.t2i.prompt,
        "negative": scene.t2i.negative,
        "seed": str(scene.t2i.seed),
        "loras": [_lora_document(item) for item in scene.t2i.loras],
        "passes": passes,
        "face_detailer": face_detailer,
    }


def _default_i2v(scene: SceneSpec) -> dict[str, Any]:
    return {
        "prompt": scene.i2v.prompt,
        "negative": scene.i2v.negative,
        "seed": str(scene.i2v.seed),
        "loras": [_lora_document(item) for item in scene.i2v.loras],
        "mandatory_loras": [
            {"name": filename, "weight": weight, "locked": True}
            for filename, weight in MANDATORY_I2V_LORAS
        ],
        "first_pass": {
            "sampler": I2V_FIRST_PASS_SAMPLER,
            "sigmas": list(I2V_FIRST_PASS_SIGMAS),
            "cfg": 1.0,
            "reference_strength": 0.75,
            "image_strength": 0.75,
            "image_compression": 35,
        },
        "second_pass": {
            "sampler": I2V_UPSCALE_PASS_SAMPLER,
            "sigmas": list(I2V_UPSCALE_PASS_SIGMAS),
            "cfg": 1.0,
            "reference_strength": 1.0,
            "image_strength": 1.0,
            "image_compression": 30,
        },
        "chunking": {"chunks": 2, "dimension_threshold": 4096},
        "spatial_upscaler": {
            "model": I2V_SPATIAL_UPSCALER,
            "tile_size": 11,
            "overlap": 6,
            "max_size_without_tiling": 22,
        },
        "temporal_continuation": _continuation_document(scene),
        "continuity": (
            deepcopy(dict(scene.i2v.continuity))
            if scene.i2v.continuity is not None
            else None
        ),
        "segments": _segment_documents(scene),
    }


def scene_review_document(job: JobPayload, scene: SceneSpec) -> dict[str, Any]:
    raw_scene = next(
        item
        for item in job.raw["scenes"]
        if isinstance(item, Mapping) and item.get("id") == scene.scene_id
    )
    excluded_scene = {"id", "title", "estimated_sec", "t2i", "i2v"}
    excluded_job = {"job_id", "character", "ltxv_character_lora", "scenes"}
    estimated_seconds = _review_duration_seconds(scene)
    timeline_frames, generation_master_frames = _continuation_frame_profile(
        estimated_seconds
    )
    return {
        "job_id": job.job_id,
        "scene_id": scene.scene_id,
        "title": scene.title,
        "estimated_seconds": estimated_seconds,
        "character": {
            "name": job.character.name,
            "series": job.character.series,
            "base_model": job.character.base_model,
            "global_lora": _lora_document(job.character.lora),
            "ltx_character_lora": (
                _lora_document(job.ltxv_character_lora)
                if job.ltxv_character_lora
                else None
            ),
        },
        "t2i": _default_t2i(job, scene),
        "i2v": _default_i2v(scene),
        "production_profile": {
            "width": PRODUCTION_WIDTH,
            "height": PRODUCTION_HEIGHT,
            "fps": PRODUCTION_FPS,
            "frame_count": frame_count_for_seconds(estimated_seconds),
            "timeline_output_frames": timeline_frames,
            "generation_master_frames": generation_master_frames,
            "ltx_checkpoint": LTX_CHECKPOINT,
            "ltx_text_encoder": LTX_TEXT_ENCODER,
            "locked": True,
        },
        "scene_context": {
            key: deepcopy(value)
            for key, value in raw_scene.items()
            if key not in excluded_scene
        },
        "job_context": {
            key: deepcopy(value)
            for key, value in job.raw.items()
            if key not in excluded_job
        },
    }


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReviewValidationError(f"{field} must be an object.")
    return value


def _list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise ReviewValidationError(f"{field} must be a list.")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReviewValidationError(f"{field} must contain text.")
    return value.strip()


def _integer(value: Any, field: str, minimum: int, maximum: int) -> int:
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value.strip())
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ReviewValidationError(
            f"{field} must be an integer from {minimum} to {maximum}."
        )
    return value


def _number(value: Any, field: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReviewValidationError(f"{field} must be a number.")
    result = float(value)
    if not minimum <= result <= maximum:
        raise ReviewValidationError(
            f"{field} must be between {minimum:g} and {maximum:g}."
        )
    return result


def _sigmas(value: Any, field: str) -> tuple[float, ...]:
    values = _list(value, field)
    if len(values) < 2:
        raise ReviewValidationError(f"{field} must contain at least two values.")
    result = tuple(_number(item, f"{field}[{index}]", 0.0, 1000.0) for index, item in enumerate(values))
    if any(left < right for left, right in zip(result, result[1:])):
        raise ReviewValidationError(f"{field} must be non-increasing.")
    if result[-1] != 0.0:
        raise ReviewValidationError(f"{field} must end in 0.")
    return result


def _validated_t2i_pass(value: Any, field: str) -> dict[str, Any]:
    data = _mapping(value, field)
    result = dict(data)
    result["sampler"] = _text(data.get("sampler"), f"{field}.sampler")
    result["scheduler"] = _text(data.get("scheduler"), f"{field}.scheduler")
    result["steps"] = _integer(data.get("steps"), f"{field}.steps", 1, 200)
    result["cfg"] = _number(data.get("cfg"), f"{field}.cfg", 0.0, 100.0)
    if "denoise" in data:
        result["denoise"] = _number(data["denoise"], f"{field}.denoise", 0.0, 1.0)
    if "start_step" in data:
        result["start_step"] = _integer(
            data["start_step"], f"{field}.start_step", 0, result["steps"]
        )
    if "end_step" in data:
        result["end_step"] = _integer(
            data["end_step"], f"{field}.end_step", 0, result["steps"]
        )
    return result


def _validated_i2v_pass(value: Any, field: str) -> dict[str, Any]:
    data = _mapping(value, field)
    return {
        **dict(data),
        "sampler": _text(data.get("sampler"), f"{field}.sampler"),
        "sigmas": _sigmas(data.get("sigmas"), f"{field}.sigmas"),
        "cfg": _number(data.get("cfg"), f"{field}.cfg", 0.0, 100.0),
        "reference_strength": _number(
            data.get("reference_strength"), f"{field}.reference_strength", 0.0, 2.0
        ),
        "image_strength": _number(
            data.get("image_strength"), f"{field}.image_strength", 0.0, 2.0
        ),
        "image_compression": _integer(
            data.get("image_compression"), f"{field}.image_compression", 0, 100
        ),
    }


def _lora_payload(value: Any, field: str) -> dict[str, Any]:
    data = _mapping(value, field)
    result = {
        "name": _text(data.get("name"), f"{field}.name"),
        "download_url": _text(data.get("download_url"), f"{field}.download_url"),
        "weight": _number(data.get("weight"), f"{field}.weight", -4.0, 4.0),
    }
    for optional in ("model_id", "version_id"):
        if data.get(optional) is not None:
            result[optional] = _integer(
                data[optional],
                f"{field}.{optional}",
                1,
                2**63 - 1,
            )
    return result


def validate_scene_edit(
    original_job: JobPayload,
    scene_id: int,
    document: Mapping[str, Any],
) -> ValidatedSceneEdit:
    data = _mapping(document, "scene")
    if data.get("job_id") != original_job.job_id or data.get("scene_id") != scene_id:
        raise ReviewValidationError("Scene identity cannot be changed.")
    character = _mapping(data.get("character"), "character")
    t2i = _mapping(data.get("t2i"), "t2i")
    i2v = _mapping(data.get("i2v"), "i2v")
    raw = deepcopy(dict(original_job.raw))
    raw_character = deepcopy(dict(raw["character"]))
    raw_character["name"] = _text(character.get("name"), "character.name")
    raw_character["series"] = _text(character.get("series"), "character.series")
    global_lora = _lora_payload(character.get("global_lora"), "character.global_lora")
    global_lora["base"] = _text(character.get("base_model"), "character.base_model")
    global_lora["recommended_weight"] = global_lora.pop("weight")
    raw_character["lora"] = global_lora
    raw["character"] = raw_character
    ltx_character = character.get("ltx_character_lora")
    raw["ltxv_character_lora"] = (
        None
        if ltx_character is None
        else _lora_payload(ltx_character, "character.ltx_character_lora")
    )

    raw_scene = next(
        item for item in raw["scenes"] if item.get("id") == scene_id
    )
    raw_scene["title"] = _text(data.get("title"), "title")
    raw_scene["estimated_sec"] = _number(
        data.get("estimated_seconds"),
        "estimated_seconds",
        1 / PRODUCTION_FPS,
        32.0,
    )
    raw_scene["t2i"] = {
        "prompt": _text(t2i.get("prompt"), "t2i.prompt"),
        "negative": _text(t2i.get("negative"), "t2i.negative"),
        "seed": _integer(t2i.get("seed"), "t2i.seed", 0, 2**64 - 1),
        "loras": [
            _lora_payload(item, f"t2i.loras[{index}]")
            for index, item in enumerate(_list(t2i.get("loras"), "t2i.loras"))
        ],
    }
    raw_scene["i2v"] = {
        "prompt": _text(i2v.get("prompt"), "i2v.prompt"),
        "negative": _text(i2v.get("negative"), "i2v.negative"),
        "seed": _integer(i2v.get("seed"), "i2v.seed", 0, 2**64 - 1),
        "loras": [
            _lora_payload(item, f"i2v.loras[{index}]")
            for index, item in enumerate(_list(i2v.get("loras"), "i2v.loras"))
        ],
    }
    temporal_continuation = i2v.get("temporal_continuation")
    continuity = i2v.get("continuity")
    segments = i2v.get("segments", [])
    if temporal_continuation is not None:
        continuation_document = deepcopy(
            dict(_mapping(temporal_continuation, "i2v.temporal_continuation"))
        )
        # ``estimated_seconds`` is the only editable duration source.  Accept
        # stale browser/revision documents, but never persist their hidden
        # continuation-specific duplicate into a new immutable revision.
        continuation_document.pop("requested_duration_seconds", None)
        raw_scene["i2v"]["continuation"] = continuation_document
    if continuity is not None:
        raw_scene["i2v"]["continuity"] = deepcopy(
            dict(_mapping(continuity, "i2v.continuity"))
        )
    normalized_segments: list[dict[str, Any]] = []
    for index, item in enumerate(_list(segments, "i2v.segments")):
        segment = deepcopy(dict(_mapping(item, f"i2v.segments[{index}]")))
        if segment.get("seed_override") is not None:
            # The browser deliberately keeps unsigned-64-bit seeds as decimal
            # text so JavaScript cannot round them through IEEE-754 Number.
            segment["seed_override"] = _integer(
                segment["seed_override"],
                f"i2v.segments[{index}].seed_override",
                0,
                2**64 - 1,
            )
        normalized_segments.append(segment)
    raw_scene["i2v"]["segments"] = normalized_segments
    try:
        revised_job = parse_job_payload(raw)
    except ContractValidationError as error:
        raise ReviewValidationError(str(error)) from error
    revised_scene = next(scene for scene in revised_job.scenes if scene.scene_id == scene_id)

    t2i_passes = tuple(
        _validated_t2i_pass(item, f"t2i.passes[{index}]")
        for index, item in enumerate(_list(t2i.get("passes"), "t2i.passes"))
    )
    expected_passes = 1 if revised_job.character.base_model.casefold() == "anima" else 2
    if len(t2i_passes) != expected_passes:
        raise ReviewValidationError(
            f"{revised_job.character.base_model} requires exactly {expected_passes} T2I pass(es)."
        )
    face_detailer = t2i.get("face_detailer")
    if revised_job.character.base_model.casefold() == "pony":
        detailer = dict(_mapping(face_detailer, "t2i.face_detailer"))
        detailer["enabled"] = bool(detailer.get("enabled", True))
        detailer["sampler"] = _text(detailer.get("sampler"), "t2i.face_detailer.sampler")
        detailer["scheduler"] = _text(
            detailer.get("scheduler"), "t2i.face_detailer.scheduler"
        )
        detailer["steps"] = _integer(
            detailer.get("steps"), "t2i.face_detailer.steps", 1, 200
        )
        detailer["cfg"] = _number(
            detailer.get("cfg"), "t2i.face_detailer.cfg", 0.0, 100.0
        )
        detailer["denoise"] = _number(
            detailer.get("denoise"), "t2i.face_detailer.denoise", 0.0, 1.0
        )
        face_detailer = detailer
    else:
        face_detailer = None

    first_pass = _validated_i2v_pass(i2v.get("first_pass"), "i2v.first_pass")
    second_pass = _validated_i2v_pass(i2v.get("second_pass"), "i2v.second_pass")
    chunking = dict(_mapping(i2v.get("chunking"), "i2v.chunking"))
    chunking["chunks"] = _integer(chunking.get("chunks"), "i2v.chunking.chunks", 1, 32)
    chunking["dimension_threshold"] = _integer(
        chunking.get("dimension_threshold"),
        "i2v.chunking.dimension_threshold",
        256,
        65536,
    )
    upscaler = dict(_mapping(i2v.get("spatial_upscaler"), "i2v.spatial_upscaler"))
    upscaler["model"] = _text(upscaler.get("model"), "i2v.spatial_upscaler.model")
    upscaler["tile_size"] = _integer(
        upscaler.get("tile_size"), "i2v.spatial_upscaler.tile_size", 1, 256
    )
    upscaler["overlap"] = _integer(
        upscaler.get("overlap"), "i2v.spatial_upscaler.overlap", 0, 255
    )
    upscaler["max_size_without_tiling"] = _integer(
        upscaler.get("max_size_without_tiling"),
        "i2v.spatial_upscaler.max_size_without_tiling",
        1,
        1024,
    )
    if upscaler["overlap"] >= upscaler["tile_size"]:
        raise ReviewValidationError("I2V tile overlap must be smaller than tile size.")

    canonical = scene_review_document(revised_job, revised_scene)
    canonical["t2i"]["passes"] = [dict(item) for item in t2i_passes]
    canonical["t2i"]["face_detailer"] = face_detailer
    canonical["i2v"]["first_pass"] = {**first_pass, "sigmas": list(first_pass["sigmas"])}
    canonical["i2v"]["second_pass"] = {**second_pass, "sigmas": list(second_pass["sigmas"])}
    canonical["i2v"]["chunking"] = chunking
    canonical["i2v"]["spatial_upscaler"] = upscaler
    return ValidatedSceneEdit(
        job=revised_job,
        scene=revised_scene,
        workflow=SceneWorkflowOverrides(
            t2i_passes=t2i_passes,
            face_detailer=face_detailer,
            i2v_first_pass=first_pass,
            i2v_second_pass=second_pass,
            chunking=chunking,
            spatial_upscaler=upscaler,
            temporal_continuation=(
                deepcopy(dict(revised_scene.i2v.continuation))
                if revised_scene.i2v.continuation is not None
                else None
            ),
            continuity=(
                deepcopy(dict(revised_scene.i2v.continuity))
                if revised_scene.i2v.continuity is not None
                else None
            ),
            segments=tuple(
                deepcopy(dict(item)) for item in revised_scene.i2v.segments
            ),
        ),
        document=json.loads(json.dumps(canonical)),
    )
