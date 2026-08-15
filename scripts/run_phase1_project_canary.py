"""Build and run an isolated phase-1 canary from historical Grok manifests."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tenminvideomaker.assembly import FfmpegAssembler, probe_video
from tenminvideomaker.assets import ComfyLoraAssetClient
from tenminvideomaker.chunk_assembly import SceneChunkAssembler
from tenminvideomaker.comfy_http import ComfyHttpClient
from tenminvideomaker.configuration import load_project_environment
from tenminvideomaker.contracts import (
    ContractValidationError,
    JobPayload,
    job_content_fingerprint,
    parse_job_payload,
)
from tenminvideomaker.qc_backend import LlamaCppHttpBackend
from tenminvideomaker.qc_config import QualityControlSettings
from tenminvideomaker.qc_contracts import QcCandidateState
from tenminvideomaker.qc_llama import LlamaCppProcess
from tenminvideomaker.artifacts import scene_clip_path
from tenminvideomaker.qc_controller import Phase1QcController
from tenminvideomaker.state_store import PipelineState, PipelineStateStore, SceneState
from tenminvideomaker.storage import StorageLayout
from tenminvideomaker.supervisor import PipelineSupervisor, SupervisorSettings


class CanarySafetyError(RuntimeError):
    """Raised when the canary attempts disallowed external effects."""


class CanarySafetyMailClient:
    """Fail-closed mail client used to guard all outbound mail-like effects."""

    def __init__(self, marker: dict[str, bool]):
        self._marker = marker

    @property
    def settings(self) -> Any:
        class _Settings:
            allowed_senders = ()

        return _Settings()

    def _mark(self, method_name: str) -> None:
        self._marker["external_effects_attempted"] = True
        raise CanarySafetyError(f"Canary safety block: {method_name} was called.")

    def send_request(self, *args: Any, **kwargs: Any) -> str:  # noqa: ARG002
        self._mark("mail.send_request")

    def unread_pipeline_messages(self, *args: Any, **kwargs: Any) -> list[object]:  # noqa: ARG002
        self._mark("mail.unread_pipeline_messages")

    def mark_seen(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        self._mark("mail.mark_seen")

    def request_message_id(self, *args: Any, **kwargs: Any) -> str:  # noqa: ARG002
        self._mark("mail.request_message_id")

    def request_was_sent(self, *args: Any, **kwargs: Any) -> bool:  # noqa: ARG002
        self._mark("mail.request_was_sent")

    def download_drive_json(self, *args: Any, **kwargs: Any) -> str:  # noqa: ARG002
        self._mark("mail.download_drive_json")

    def download(self, *args: Any, **kwargs: Any) -> object:  # noqa: ARG002
        self._mark("mail.download")

    def __getattr__(self, method_name: str) -> Any:
        if method_name.startswith("_"):
            raise AttributeError(method_name)
        return lambda *args: self._mark(f"mail.{method_name}")


class CanaryPipelineSupervisor(PipelineSupervisor):
    """Supervisor guardrail for a single canary run."""

    def _request_next_job(self, *, previous_job_id: str | None, succeeded: bool | None) -> None:  # noqa: ARG002
        raise CanarySafetyError("Canary safety block: _request_next_job was attempted.")

    def _finalize_qc_result(self, *args: Any, **kwargs: Any) -> None:
        raise CanarySafetyError(
            "Canary safety block: _finalize_qc_result was attempted."
        )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractValidationError(f"Failed to read {path}: {error}") from error
    if not isinstance(document, dict):
        raise ContractValidationError(f"{path} is not a JSON object.")
    return document


def _normalize_stage_seed(document: Mapping[str, Any], field: str) -> dict[str, Any]:
    if "seed" not in document:
        raise ContractValidationError(f"{field}.seed is required.")
    stage = dict(document)
    stage["seed"] = _ensure_int(stage["seed"], f"{field}.seed")
    return stage


def _mapping_value(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractValidationError(f"{field} must be an object.")
    return dict(value)


def _ensure_int(value: Any, field: str) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    raise ContractValidationError(f"{field} must be an integer.")


def _ensure_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractValidationError(f"{field} must be numeric.")
    return float(value)


def _ensure_text(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ContractValidationError(f"{field} must be a string.")
    return value.strip()


def _ensure_lora_weight(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ContractValidationError(f"{field} must be numeric.")
    if isinstance(value, str):
        if not value.strip():
            raise ContractValidationError(f"{field} must be numeric.")
        try:
            return float(value.strip())
        except ValueError as error:
            raise ContractValidationError(f"{field} must be numeric.") from error
    if not isinstance(value, (int, float)):
        raise ContractValidationError(f"{field} must be numeric.")
    return float(value)


def _normalize_optional_int(value: Any, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    raise ContractValidationError(f"{field} must be an integer if present.")


def _coerce_lora_document(
    value: Any,
    field: str,
    *,
    required_weight: bool = True,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ContractValidationError(f"{field} must be an object.")
    result = dict(value)
    if required_weight:
        if "weight" in result:
            result["weight"] = _ensure_lora_weight(result["weight"], f"{field}.weight")
        elif "recommended_weight" in result:
            result["weight"] = _ensure_lora_weight(
                result["recommended_weight"],
                f"{field}.recommended_weight",
            )
            result.pop("recommended_weight", None)
        else:
            raise ContractValidationError(
                f"{field} must include weight or recommended_weight."
            )
    if "model_id" in result:
        result["model_id"] = _normalize_optional_int(
            result["model_id"],
            f"{field}.model_id",
        )
    if "version_id" in result:
        result["version_id"] = _normalize_optional_int(
            result["version_id"],
            f"{field}.version_id",
        )
    if "name" in result:
        result["name"] = _ensure_text(result["name"], f"{field}.name")
    if "download_url" in result:
        result["download_url"] = _ensure_text(result["download_url"], f"{field}.download_url")
    return result


def _normalize_stage_document(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractValidationError(f"{field} must be an object.")
    stage = dict(value)
    if "seed" not in stage:
        raise ContractValidationError(f"{field}.seed is required.")
    stage["seed"] = _ensure_int(stage["seed"], f"{field}.seed")

    if "loras" in stage:
        if not isinstance(stage["loras"], list):
            raise ContractValidationError(f"{field}.loras must be an array.")
        stage["loras"] = [
            _coerce_lora_document(item, f"{field}.loras[{index}]")
            for index, item in enumerate(stage["loras"])
            if item is not None
        ]

    if "segments" in stage:
        segments = stage["segments"]
        if not isinstance(segments, list):
            raise ContractValidationError(f"{field}.segments must be an array.")
        normalized_segments: list[dict[str, Any]] = []
        for index, item in enumerate(segments):
            if not isinstance(item, Mapping):
                raise ContractValidationError(f"{field}.segments[{index}] must be an object.")
            segment = dict(item)
            if "seed_override" in segment and segment["seed_override"] is not None:
                segment["seed_override"] = _ensure_int(
                    segment["seed_override"],
                    f"{field}.segments[{index}].seed_override",
                )
            if "variation_index" in segment and segment["variation_index"] is not None:
                segment["variation_index"] = _ensure_int(
                    segment["variation_index"],
                    f"{field}.segments[{index}].variation_index",
                )
            if "new_transition_frames" in segment and segment["new_transition_frames"] is not None:
                segment["new_transition_frames"] = _ensure_int(
                    segment["new_transition_frames"],
                    f"{field}.segments[{index}].new_transition_frames",
                )
            if "requested_duration_seconds" in segment and segment["requested_duration_seconds"] is not None:
                segment["requested_duration_seconds"] = _ensure_number(
                    segment["requested_duration_seconds"],
                    f"{field}.segments[{index}].requested_duration_seconds",
                )
            normalized_segments.append(segment)
        stage["segments"] = normalized_segments

    return stage


def _coerce_character_signature(character: Mapping[str, Any]) -> tuple[str, str, str, dict[str, Any], dict[str, Any] | None]:
    base_model = _ensure_text(character.get("base_model"), "parameters.character.base_model")
    if not base_model:
        raise ContractValidationError("parameters.character.base_model must be set.")
    global_lora = _coerce_lora_document(
        character.get("global_lora"),
        "parameters.character.global_lora",
    )
    assert global_lora is not None
    ltx_character_lora = _coerce_lora_document(
        character.get("ltx_character_lora"),
        "parameters.character.ltx_character_lora",
        required_weight=False,
    )
    if ltx_character_lora is not None:
        if "weight" not in ltx_character_lora and "recommended_weight" in ltx_character_lora:
            ltx_character_lora["weight"] = _ensure_lora_weight(
                ltx_character_lora["recommended_weight"],
                "parameters.character.ltx_character_lora.recommended_weight",
            )
            ltx_character_lora.pop("recommended_weight", None)
    return (
        _ensure_text(character.get("name"), "parameters.character.name"),
        _ensure_text(character.get("series"), "parameters.character.series"),
        base_model,
        global_lora,
        ltx_character_lora,
    )


def _load_and_reconstruct_payload(
    source_root: Path,
    source_job_id: str,
    canary_job_id: str,
    scene_ids: tuple[int, ...],
) -> tuple[JobPayload, list[tuple[int, Path]]]:
    source_job_root = _resolve_source_job_root(source_root, source_job_id)
    selected_frames: list[tuple[int, Path]] = []
    scenes: list[dict[str, Any]] = []
    character_signature: tuple[str, str, str, dict[str, Any], dict[str, Any] | None] | None = None

    for scene_id in scene_ids:
        manifest_path = source_job_root / "scenes" / f"scene_{scene_id:04d}" / "revisions" / "0001" / "generation-manifest.json"
        if not manifest_path.is_file():
            raise ContractValidationError(f"Missing manifest: {manifest_path}")
        manifest = _read_json(manifest_path)
        parameters = manifest.get("parameters")
        if not isinstance(parameters, Mapping):
            raise ContractValidationError(f"{manifest_path} missing parameters object.")

        manifest_scene_id = _ensure_int(
            parameters.get("scene_id"),
            f"{manifest_path}: parameters.scene_id",
        )
        if manifest_scene_id != scene_id:
            raise ContractValidationError(
                f"{manifest_path} scene_id {manifest_scene_id} != requested {scene_id}."
            )

        character = parameters.get("character")
        if not isinstance(character, Mapping):
            raise ContractValidationError(
                f"{manifest_path} missing parameters.character."
            )
        signature = _coerce_character_signature(character)
        if character_signature is None:
            character_signature = signature
        else:
            if character_signature != signature:
                raise ContractValidationError(
                    f"{manifest_path}: character identity differs from previous selected scene."
                )

        scenes.append(
            {
                "id": scene_id,
                "title": _ensure_text(parameters.get("title"), f"{manifest_path}: parameters.title"),
                "estimated_sec": _ensure_number(
                    parameters.get("estimated_seconds"),
                    f"{manifest_path}: parameters.estimated_seconds",
                ),
                "t2i": _normalize_stage_seed(
                    _mapping_value(parameters.get("t2i"), f"{manifest_path}: parameters.t2i"),
                    f"{manifest_path}: parameters.t2i",
                ),
                "i2v": _normalize_stage_seed(
                    _mapping_value(parameters.get("i2v"), f"{manifest_path}: parameters.i2v"),
                    f"{manifest_path}: parameters.i2v",
                ),
            }
        )

        source_frame = (
            source_job_root
            / "scenes"
            / f"scene_{scene_id:04d}"
            / "revisions"
            / "0001"
            / "frame.png"
        )
        if not source_frame.is_file():
            raise ContractValidationError(f"Missing source revision-1 frame: {source_frame}")
        selected_frames.append((scene_id, source_frame))

    if character_signature is None:
        raise ContractValidationError("No scenes were selected.")

    source_name, source_series, source_base_model, source_global_lora, source_ltx = character_signature
    if not source_ltx:
        source_ltx = None

    character_lora = {
        "name": source_global_lora["name"],
        "download_url": source_global_lora["download_url"],
        "base": source_base_model,
        "recommended_weight": source_global_lora["weight"],
    }
    if source_global_lora.get("model_id") is not None:
        character_lora["model_id"] = source_global_lora["model_id"]
    if source_global_lora.get("version_id") is not None:
        character_lora["version_id"] = source_global_lora["version_id"]

    job_payload_raw: dict[str, Any] = {
        "job_id": canary_job_id,
        "character": {
            "name": source_name,
            "series": source_series,
            "lora": character_lora,
        },
        "ltxv_character_lora": source_ltx,
        "scenes": scenes,
        "source_job_id": source_job_id,
        "canary": True,
    }

    payload = parse_job_payload(job_payload_raw)
    return payload, selected_frames


def _resolve_source_job_root(source_root: Path, source_job_id: str) -> Path:
    probe_scene_id = 1
    direct_probe = (
        source_root / "scenes" / f"scene_{probe_scene_id:04d}" / "revisions" / "0001" / "generation-manifest.json"
    )
    if direct_probe.is_file():
        return source_root

    candidates = (
        source_root / source_job_id,
        source_root / "jobs" / source_job_id,
        source_root.parent / source_job_id if source_root.name.lower() == "jobs" else None,
    )
    for candidate in candidates:
        if candidate is None:
            continue
        manifest_candidate = (
            candidate
            / "scenes"
            / f"scene_{probe_scene_id:04d}"
            / "revisions"
            / "0001"
            / "generation-manifest.json"
        )
        if manifest_candidate.is_file():
            return candidate

    raise ContractValidationError(
        f"Could not resolve job root for source_job_id={source_job_id} in {source_root}"
    )



def _qc_owned_runtime_layout(
    qc_settings: QualityControlSettings,
    storage: StorageLayout,
) -> StorageLayout:
    runtime_root = (PROJECT_ROOT / "runtime" / "qc-owned").resolve()
    disallowed_roots = [
        storage.root,
        qc_settings.llama_vendor_root,
        qc_settings.llama_executable,
        qc_settings.llama_executable.parent if qc_settings.llama_executable is not None else None,
        qc_settings.model_path,
        qc_settings.model_path.parent if qc_settings.model_path is not None else None,
        qc_settings.projector_path,
        qc_settings.projector_path.parent if qc_settings.projector_path is not None else None,
    ]

    def _contains(root: Path, candidate: Path) -> bool:
        root_parts = tuple(part.casefold() for part in root.resolve().parts)
        candidate_parts = tuple(part.casefold() for part in candidate.resolve().parts)
        if len(candidate_parts) >= len(root_parts):
            return candidate_parts[: len(root_parts)] == root_parts
        return root_parts[: len(candidate_parts)] == candidate_parts

    for disallowed in [item for item in disallowed_roots if item is not None]:
        if _contains(disallowed, runtime_root) or _contains(runtime_root, disallowed):
            raise CanarySafetyError("Refusing to place QC runtime under shared asset paths.")
    return StorageLayout(runtime_root)


def _build_canary_supervisor(
    storage: StorageLayout,
    store: PipelineStateStore,
    qc_settings: QualityControlSettings,
    comfy_url: str,
    ffmpeg: str,
    ffprobe: str,
    external_tracker: dict[str, bool],
) -> CanaryPipelineSupervisor:
    comfy = ComfyHttpClient(comfy_url)
    settings = replace(
        SupervisorSettings.from_environment(),
        require_human_review=True,
    )
    qc_controller = Phase1QcController(
        store=store,
        layout=storage,
        settings=qc_settings,
        backend_factory=lambda: LlamaCppHttpBackend(
            qc_settings,
            LlamaCppProcess(qc_settings, _qc_owned_runtime_layout(qc_settings, storage)),
        ),
        prompt_root=PROJECT_ROOT / "prompts",
        ffmpeg_command=ffmpeg,
        ffprobe_command=ffprobe,
    )
    return CanaryPipelineSupervisor(
        store=store,
        mail_client=CanarySafetyMailClient(external_tracker),
        asset_manager=ComfyLoraAssetClient(comfy),
        comfy=comfy,
        assembler=FfmpegAssembler(
            storage.finals_root,
            ffmpeg_executable=ffmpeg,
        ),
        settings=settings,
        video_probe=partial(probe_video, ffprobe_executable=ffprobe),
        storage=storage,
        chunk_assembler=SceneChunkAssembler(
            ffmpeg_executable=ffmpeg,
            ffprobe_executable=ffprobe,
        ),
        qc_controller=qc_controller,
        delivery=None,
    )


def _collect_scene_records(store: PipelineStateStore, job_id: str) -> list[dict[str, object]]:
    scenes = []
    for record in store.scene_records(job_id):
        scenes.append(
            {
                "scene_id": record.scene_id,
                "state": record.state.value,
                "frame_path": record.frame_path,
                "video_path": record.video_path,
                "error": record.error,
                "t2i_attempts": record.t2i_attempts,
                "i2v_attempts": record.i2v_attempts,
            }
        )
    return scenes


def _candidate_decision(candidate: Any, store: PipelineStateStore) -> str:
    for evaluation in store.qc_evaluations(candidate.candidate_id):
        if evaluation.normalized_decision is not None:
            return str(evaluation.normalized_decision)
    if candidate.state in {QcCandidateState.PASS_PENDING_HUMAN, QcCandidateState.ACCEPTED}:
        return "PASS"
    if candidate.state == QcCandidateState.HOLD_FOR_REVIEW:
        return "HOLD"
    if candidate.state == QcCandidateState.SUPERSEDED:
        return "UNCERTAIN"
    return "UNCERTAIN"


def _collect_qc_summary(store: PipelineStateStore, job_id: str) -> tuple[
    list[dict[str, object]], dict[str, int], list[dict[str, object]]
]:
    per_scene: dict[int, list[dict[str, object]]] = {}
    candidates_by_scene: list[dict[str, object]] = []
    counts = {"PASS": 0, "FAIL": 0, "UNCERTAIN": 0, "HOLD": 0}

    for candidate in store.qc_candidates(job_id):
        decision = _candidate_decision(candidate, store)
        counts[decision] += 1
        candidates_by_scene.append(
            {
                "scene_id": candidate.scene_id,
                "revision": candidate.revision,
                "candidate_id": candidate.candidate_id,
                "tier": candidate.tier.value,
                "state": candidate.state.value,
                "decision": decision,
            }
        )
        per_scene.setdefault(candidate.scene_id, []).append(
            {
                "tier": candidate.tier.value,
                "state": candidate.state.value,
                "decision": decision,
                "candidate_id": candidate.candidate_id,
            }
        )

    per_scene_states = [
        {"scene_id": scene_id, "candidates": candidates}
        for scene_id, candidates in sorted(per_scene.items())
    ]
    return per_scene_states, counts, candidates_by_scene


def parse_scene_ids(raw: str) -> tuple[int, ...]:
    parts = [item.strip() for item in raw.split(",") if item.strip()]
    if not parts:
        raise ContractValidationError("--scene-ids cannot be empty.")
    values: list[int] = []
    for item in parts:
        if not item.isdigit():
            raise ContractValidationError(f"Scene id {item!r} is not an integer.")
        scene_id = int(item)
        if scene_id <= 0:
            raise ContractValidationError("Scene ids must be positive.")
        values.append(scene_id)
    if len(values) != len(set(values)):
        raise ContractValidationError("scene ids contains duplicates.")
    return tuple(values)


def _write_summary(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_selected_reconstruction(
    args: argparse.Namespace,
    scene_ids: tuple[int, ...],
) -> tuple[JobPayload, list[tuple[int, Path]], Path]:
    source_root = args.source_job_root.resolve()
    if not source_root.exists():
        raise ContractValidationError(f"Source job root does not exist: {source_root}")

    payload, selected_frames = _load_and_reconstruct_payload(
        source_root,
        args.source_job_id,
        args.canary_job_id,
        scene_ids,
    )
    return payload, selected_frames, source_root


def _copy_frame_records(
    canary_root: Path,
    canary_job_id: str,
    selected_frames: list[tuple[int, Path]],
) -> list[dict[str, str]]:
    layout = StorageLayout(canary_root)
    copied: list[dict[str, str]] = []
    for scene_id, source_frame in selected_frames:
        canary_frame = layout.scene_frame_path(canary_job_id, scene_id, 1)
        canary_frame.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_frame, canary_frame)
        copied.append(
            {
                "scene_id": str(scene_id),
                "source": str(source_frame),
                "canary": str(canary_frame),
            }
        )
    return copied


def _run_prepare(args: argparse.Namespace, scene_ids: tuple[int, ...]) -> dict[str, Any]:
    canary_root = args.canary_root.resolve()
    if canary_root.exists():
        raise CanarySafetyError(f"Canary root already exists: {canary_root}")

    payload, selected_frames, source_root = _load_selected_reconstruction(args, scene_ids)

    canary_layout = StorageLayout(canary_root)
    canary_layout.ensure()
    copied_frames = _copy_frame_records(canary_root, args.canary_job_id, selected_frames)

    store = PipelineStateStore(canary_layout.database_path)
    store.claim_job(payload, review_required=False)
    for scene in payload.scenes:
        canary_frame = canary_layout.scene_frame_path(payload.job_id, scene.scene_id, 1)
        store.set_scene_state(
            payload.job_id,
            scene.scene_id,
            SceneState.PENDING,
            frame_path=str(canary_frame),
            video_path=None,
        )

    return {
        "source_job_id": args.source_job_id,
        "source_job_root": str(source_root),
        "canary_job_id": payload.job_id,
        "canary_root": str(canary_root),
        "selected_scene_ids": list(scene_ids),
        "reconstructed_payload_fingerprint": job_content_fingerprint(payload),
        "copied_frames": copied_frames,
        "historical_video_copied": False,
        "qc_execute_status": "not_run",
        "runtime_seconds": 0.0,
        "external_effects_attempted": False,
        "canary_payload_loaded": True,
        "scene_records": _collect_scene_records(store, payload.job_id),
        "snapshot_state": str(store.snapshot().state),
    }


def _run_execute(args: argparse.Namespace, scene_ids: tuple[int, ...]) -> dict[str, Any]:
    canary_root = args.canary_root.resolve()
    if not canary_root.is_dir():
        raise CanarySafetyError(f"Canary root does not exist: {canary_root}")

    layout = StorageLayout(canary_root)
    store = PipelineStateStore(layout.database_path)
    payload = store.load_job(args.canary_job_id)

    reconstructed, _, source_root = _load_selected_reconstruction(
        args,
        tuple(scene.scene_id for scene in payload.scenes),
    )
    if job_content_fingerprint(payload) != job_content_fingerprint(reconstructed):
        raise CanarySafetyError("Reconstructed payload and canary DB payload mismatch.")

    project_environment = load_project_environment(PROJECT_ROOT, storage_layout=StorageLayout.configured())
    os.environ.update({key: value for key, value in project_environment.items() if key.startswith("TENMIN_")})
    if args.comfy_url:
        os.environ["TENMIN_COMFY_URL"] = args.comfy_url
    os.environ["TENMIN_STORAGE_ROOT"] = str(canary_root)
    effective_root = StorageLayout.configured().root.resolve()
    if effective_root != canary_root.resolve():
        raise CanarySafetyError(
            f"Refusing to run canary with non-isolated storage root: {effective_root} != {canary_root}"
        )
    probe_clip_path = scene_clip_path("canary_path_probe", 1)
    if not probe_clip_path.is_relative_to(canary_root):
        raise CanarySafetyError(
            "Canary path probe failed: clip path is not under canary root after "
            f"TENMIN_STORAGE_ROOT override: {probe_clip_path}"
        )

    ffmpeg = os.environ.get("TENMIN_FFMPEG", "ffmpeg")
    ffprobe = os.environ.get("TENMIN_FFPROBE", "ffprobe")
    comfy_url = os.environ.get("TENMIN_COMFY_URL", "http://127.0.0.1:8188")

    qc_settings = replace(
        QualityControlSettings.from_environment(project_environment),
        quality_control_enabled=True,
        auto_advance_pass=False,
    )
    qc_settings.validate_for_start()

    marker: dict[str, bool] = {"external_effects_attempted": False}
    supervisor = _build_canary_supervisor(
        layout,
        store,
        qc_settings,
        comfy_url=comfy_url,
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        external_tracker=marker,
    )

    start = time.perf_counter()
    qc_execute_status = "started"
    error: str | None = None
    try:
        supervisor.process_job(payload)
        qc_execute_status = "completed"
    except CanarySafetyError as exc:
        qc_execute_status = "blocked"
        error = str(exc)
    except Exception as exc:  # noqa: BLE001
        qc_execute_status = "failed"
        error = str(exc)

    snapshot = store.snapshot()
    per_scene_states, counts, per_scene_candidates = _collect_qc_summary(store, payload.job_id)
    summary = {
        "source_job_id": args.source_job_id,
        "source_job_root": str(source_root),
        "canary_job_id": payload.job_id,
        "canary_root": str(canary_root),
        "selected_scene_ids": [scene.scene_id for scene in payload.scenes],
        "reconstructed_payload_fingerprint": job_content_fingerprint(payload),
        "copied_frames": [],
        "historical_video_copied": False,
        "qc_execute_status": qc_execute_status,
        "runtime_seconds": time.perf_counter() - start,
        "external_effects_attempted": marker["external_effects_attempted"],
        "error": error,
        "snapshot_state": str(snapshot.state),
        "snapshot_job_id": snapshot.job_id,
        "snapshot_active_scene_id": snapshot.active_scene_id,
        "snapshot_error": snapshot.error,
        "per_scene_states": _collect_scene_records(store, payload.job_id),
        "per_scene_qc_candidate_states": per_scene_states,
        "per_scene_qc_candidates": per_scene_candidates,
        "qc_decision_counts": counts,
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay selected historical scenes into isolated canary storage."
    )
    parser.add_argument("--source-job-root", type=Path, required=True)
    parser.add_argument("--source-job-id", required=True)
    parser.add_argument("--canary-root", type=Path, required=True)
    parser.add_argument("--canary-job-id", required=True)
    parser.add_argument(
        "--scene-ids",
        type=parse_scene_ids,
        required=True,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--comfy-url", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary: dict[str, Any]
    if not args.scene_ids:
        raise SystemExit("--scene-ids cannot be empty.")
    try:
        if args.prepare_only:
            summary = _run_prepare(args, args.scene_ids)
        else:
            summary = _run_execute(args, args.scene_ids)
    except (CanarySafetyError, ContractValidationError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc

    output = args.canary_root.resolve() / "canary-summary.json"
    _write_summary(output, summary)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
