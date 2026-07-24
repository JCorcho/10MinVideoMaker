"""Strict, dependency-free validation for Grok job payloads."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse

from .constants import MAX_SCENE_SECONDS, frame_count_for_seconds

_JOB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_MAX_SEED = 0xFFFFFFFFFFFFFFFF


class ContractValidationError(ValueError):
    """Raised when an incoming payload cannot safely enter the pipeline."""


@dataclass(frozen=True)
class LoraSpec:
    name: str
    download_url: str
    weight: float
    model_id: int | None = None
    version_id: int | None = None


@dataclass(frozen=True)
class StageSpec:
    prompt: str
    negative: str
    seed: int
    loras: tuple[LoraSpec, ...]


@dataclass(frozen=True)
class CharacterSpec:
    name: str
    series: str
    base_model: str
    lora: LoraSpec


@dataclass(frozen=True)
class SceneSpec:
    scene_id: int
    title: str
    estimated_sec: float
    frame_count: int
    t2i: StageSpec
    i2v: StageSpec


@dataclass(frozen=True)
class JobPayload:
    job_id: str
    character: CharacterSpec
    ltxv_character_lora: LoraSpec | None
    scenes: tuple[SceneSpec, ...]
    raw: Mapping[str, Any]


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractValidationError(f"{field} must be an object.")
    return value


def _list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise ContractValidationError(f"{field} must be an array.")
    return value


def _required(mapping: Mapping[str, Any], field: str, context: str) -> Any:
    if field not in mapping:
        raise ContractValidationError(f"{context}.{field} is required.")
    return mapping[field]


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(f"{field} must be a non-empty string.")
    return value.strip()


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractValidationError(f"{field} must be a number.")
    return float(value)


def _seed(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= _MAX_SEED:
        raise ContractValidationError(f"{field} must be an integer from 0 to {_MAX_SEED}.")
    return value


def _optional_positive_integer(
    mapping: Mapping[str, Any],
    field: str,
    context: str,
) -> int | None:
    value = mapping.get(field)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ContractValidationError(f"{context}.{field} must be a positive integer when provided.")
    return value


def _https_url(value: Any, field: str) -> str:
    url = _string(value, field)
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ContractValidationError(f"{field} must be an absolute HTTPS URL.")
    return url


def _lora(value: Any, context: str, *, weight_field: str = "weight") -> LoraSpec:
    data = _mapping(value, context)
    weight = _number(_required(data, weight_field, context), f"{context}.{weight_field}")
    if not -4.0 <= weight <= 4.0:
        raise ContractValidationError(f"{context}.{weight_field} must be between -4.0 and 4.0.")
    download_url = _https_url(_required(data, "download_url", context), f"{context}.download_url")
    version_id = _optional_positive_integer(data, "version_id", context)
    url_version_id = civitai_version_id(download_url)
    if version_id is not None and url_version_id is not None and version_id != url_version_id:
        raise ContractValidationError(
            f"{context}.version_id does not match the Civitai download URL."
        )
    return LoraSpec(
        name=_string(_required(data, "name", context), f"{context}.name"),
        download_url=download_url,
        weight=weight,
        model_id=_optional_positive_integer(data, "model_id", context),
        version_id=version_id or url_version_id,
    )


def _stage(value: Any, context: str) -> StageSpec:
    data = _mapping(value, context)
    loras = tuple(_lora(item, f"{context}.loras[{index}]") for index, item in enumerate(_list(_required(data, "loras", context), f"{context}.loras")))
    return StageSpec(
        prompt=_string(_required(data, "prompt", context), f"{context}.prompt"),
        negative=_string(_required(data, "negative", context), f"{context}.negative"),
        seed=_seed(_required(data, "seed", context), f"{context}.seed"),
        loras=loras,
    )


def _scene(value: Any, index: int) -> SceneSpec:
    context = f"scenes[{index}]"
    data = _mapping(value, context)
    scene_id = _required(data, "id", context)
    if isinstance(scene_id, bool) or not isinstance(scene_id, int) or scene_id < 1:
        raise ContractValidationError(f"{context}.id must be a positive integer.")
    estimated_sec = _number(_required(data, "estimated_sec", context), f"{context}.estimated_sec")
    if not 0 < estimated_sec <= MAX_SCENE_SECONDS:
        raise ContractValidationError(f"{context}.estimated_sec must be greater than 0 and no more than {MAX_SCENE_SECONDS:g}.")
    return SceneSpec(
        scene_id=scene_id,
        title=_string(_required(data, "title", context), f"{context}.title"),
        estimated_sec=estimated_sec,
        frame_count=frame_count_for_seconds(estimated_sec),
        t2i=_stage(_required(data, "t2i", context), f"{context}.t2i"),
        i2v=_stage(_required(data, "i2v", context), f"{context}.i2v"),
    )


def parse_job_payload(value: Any) -> JobPayload:
    """Validate and normalize the exact nested contract delivered by Grok."""
    data = _mapping(value, "payload")
    job_id = _string(_required(data, "job_id", "payload"), "payload.job_id")
    if not _JOB_ID_RE.fullmatch(job_id):
        raise ContractValidationError("payload.job_id may contain only letters, numbers, dot, underscore, and hyphen.")

    character_data = _mapping(_required(data, "character", "payload"), "payload.character")
    character_lora_data = _mapping(_required(character_data, "lora", "payload.character"), "payload.character.lora")
    base_model = _string(_required(character_lora_data, "base", "payload.character.lora"), "payload.character.lora.base")
    if base_model.casefold() not in {"anima", "pony"}:
        raise ContractValidationError("payload.character.lora.base must be either Anima or Pony.")
    character = CharacterSpec(
        name=_string(_required(character_data, "name", "payload.character"), "payload.character.name"),
        series=_string(_required(character_data, "series", "payload.character"), "payload.character.series"),
        base_model=base_model,
        lora=_lora(character_lora_data, "payload.character.lora", weight_field="recommended_weight"),
    )
    ltxv_character_value = data.get("ltxv_character_lora")
    ltxv_character_lora = None if ltxv_character_value is None else _lora(ltxv_character_value, "payload.ltxv_character_lora")

    scenes = tuple(_scene(item, index) for index, item in enumerate(_list(_required(data, "scenes", "payload"), "payload.scenes")))
    if not scenes:
        raise ContractValidationError("payload.scenes must contain at least one scene.")
    ids = [scene.scene_id for scene in scenes]
    if len(ids) != len(set(ids)):
        raise ContractValidationError("payload.scenes contains duplicate scene ids.")

    return JobPayload(
        job_id=job_id,
        character=character,
        ltxv_character_lora=ltxv_character_lora,
        scenes=scenes,
        raw=data,
    )


def civitai_version_id(download_url: str) -> int | None:
    parsed = urlparse(download_url)
    if parsed.hostname not in {"civitai.com", "www.civitai.com"}:
        return None
    match = re.fullmatch(r"/api/download/models/(\d+)/?", parsed.path)
    return int(match.group(1)) if match else None


def lora_identity(lora: LoraSpec) -> str:
    """Return a stable asset identity independent of display-name differences."""
    if lora.version_id is not None:
        return f"civitai-version:{lora.version_id}"
    parsed = urlparse(lora.download_url)
    normalized_url = parsed._replace(
        scheme=parsed.scheme.casefold(),
        netloc=parsed.netloc.casefold(),
        fragment="",
    ).geturl()
    return f"url:{normalized_url}"


def unique_loras(loras: Iterable[LoraSpec]) -> tuple[LoraSpec, ...]:
    seen: set[str] = set()
    result: list[LoraSpec] = []
    for lora in loras:
        key = lora_identity(lora)
        if key not in seen:
            seen.add(key)
            result.append(lora)
    return tuple(result)


def effective_t2i_loras(scene: SceneSpec, character: CharacterSpec) -> tuple[LoraSpec, ...]:
    """Inject the global character LoRA once, even when Grok repeats it per scene."""
    return unique_loras((character.lora, *scene.t2i.loras))
