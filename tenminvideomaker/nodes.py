"""ComfyUI nodes exposing the shared 10MinVideoMaker services."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

from .assembly import AssemblyError, FfmpegAssembler, probe_video, validate_video_profile
from .artifacts import ArtifactError, save_scene_frame
from .assets import LocalLoraRequirement, LoraAssetManager
from .configuration import load_project_environment
from .constants import (
    I2V_DYNAMIC_BASE_MODEL,
    MANDATORY_I2V_LORAS,
    PRODUCTION_FPS,
    PRODUCTION_HEIGHT,
    PRODUCTION_WIDTH,
)
from .contracts import (
    ContractValidationError,
    JobPayload,
    effective_i2v_loras,
    effective_t2i_loras,
    parse_job_payload,
    unique_loras,
)
from .mail import GmailClient, GmailPollingService, GmailSettings, MailConfigurationError, MailTransportError
from .state_store import PipelineState, PipelineStateStore
from .storage import StorageLayout

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STORAGE = StorageLayout.configured()


def _store() -> PipelineStateStore:
    return PipelineStateStore(STORAGE.database_path)


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True)


def _job_from_json(job_json: str) -> JobPayload:
    try:
        raw = json.loads(job_json)
    except json.JSONDecodeError as error:
        raise ContractValidationError("job_json must contain valid JSON.") from error
    return parse_job_payload(raw)


class _AlwaysRun:
    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")


class TenMinValidateJobNode:
    CATEGORY = "10MinVideoMaker/Control"
    DESCRIPTION = "Strictly validates a Grok job payload before any asset or generation work begins."
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("job_id", "normalized_job_json", "scene_count", "width", "height", "fps")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"job_json": ("STRING", {"multiline": True, "default": ""})}}

    def execute(self, job_json: str):
        payload = _job_from_json(job_json)
        return (
            payload.job_id,
            _json(payload.raw),
            len(payload.scenes),
            PRODUCTION_WIDTH,
            PRODUCTION_HEIGHT,
            PRODUCTION_FPS,
        )


class TenMinPipelineStatusNode(_AlwaysRun):
    CATEGORY = "10MinVideoMaker/Control"
    DESCRIPTION = "Reads the durable pipeline state without accessing Gmail or generation backends."
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("state", "job_id", "error")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def execute(self):
        snapshot = _store().snapshot()
        return (snapshot.state.value, snapshot.job_id or "", snapshot.error or "")


class TenMinRequestGrokJobNode(_AlwaysRun):
    CATEGORY = "10MinVideoMaker/Gmail"
    DESCRIPTION = "Sends the exact Gmail request subject using environment-provided Gmail credentials."
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("message_id", "status")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "previous_job_id": ("STRING", {"default": ""}),
                "previous_job_succeeded": ("BOOLEAN", {"default": True}),
            }
        }

    def execute(self, previous_job_id: str, previous_job_succeeded: bool):
        try:
            client = GmailClient(GmailSettings.from_environment())
            message_id = client.send_request(
                previous_job_id=previous_job_id or None,
                succeeded=previous_job_succeeded if previous_job_id else None,
            )
        except (MailConfigurationError, MailTransportError) as error:
            raise RuntimeError(str(error)) from error
        _store().transition(PipelineState.WAITING_FOR_GROK)
        return (message_id, PipelineState.WAITING_FOR_GROK.value)


class TenMinPollGmailNode(_AlwaysRun):
    CATEGORY = "10MinVideoMaker/Gmail"
    DESCRIPTION = "Polls Gmail once. Use the supervisor for the five-minute schedule; this node never sleeps."
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = ("job_json", "job_id", "accepted")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def execute(self):
        try:
            client = GmailClient(GmailSettings.from_environment())
            payload = GmailPollingService(_store(), client).poll_once()
        except (MailConfigurationError, MailTransportError) as error:
            raise RuntimeError(str(error)) from error
        if payload is None:
            return ("", "", False)
        return (_json(payload.raw), payload.job_id, True)


class TenMinResolveLorasNode(_AlwaysRun):
    CATEGORY = "10MinVideoMaker/Assets"
    DESCRIPTION = "Checks/downloads payload LoRAs and verifies the mandatory local LTX LoRAs."
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING", "BOOLEAN")
    RETURN_NAMES = ("asset_results_json", "all_ready")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "job_json": ("STRING", {"multiline": True, "default": ""}),
                "stage": (["all", "t2i", "i2v"], {"default": "all"}),
            }
        }

    def execute(self, job_json: str, stage: str):
        import folder_paths

        payload = _job_from_json(job_json)
        t2i_loras = []
        i2v_loras = []
        if stage in {"all", "t2i"}:
            t2i_loras.extend(
                lora
                for scene in payload.scenes
                for lora in effective_t2i_loras(scene, payload.character)
            )
        if stage in {"all", "i2v"}:
            i2v_loras.extend(
                lora
                for scene in payload.scenes
                for lora in effective_i2v_loras(payload, scene)
            )

        manager = LoraAssetManager(
            folder_paths.get_folder_paths("loras"),
            RUNTIME_ROOT / "asset_manifest.json",
            visible_lora_names=folder_paths.get_filename_list("loras"),
            civitai_token=load_project_environment(PROJECT_ROOT).get(
                "TENMIN_CIVITAI_TOKEN",
                "",
            ),
        )
        results = [*manager.resolve_many(unique_loras(t2i_loras))]
        results.extend(
            manager.resolve_many(
                unique_loras(i2v_loras),
                expected_base_model=I2V_DYNAMIC_BASE_MODEL,
            )
        )
        if stage in {"all", "i2v"}:
            results.extend(
                manager.require_local(LocalLoraRequirement(filename, weight))
                for filename, weight in MANDATORY_I2V_LORAS
            )
        serializable = [
            {
                "name": result.name,
                "path": str(result.path) if result.path else None,
                "downloaded": result.downloaded,
                "error": result.error,
                "local_filename": result.local_filename,
                "base_model": result.base_model,
            }
            for result in results
        ]
        return (_json(serializable), all(result.succeeded for result in results))


class TenMinReleaseMemoryNode(_AlwaysRun):
    CATEGORY = "10MinVideoMaker/Runtime"
    DESCRIPTION = "Runs garbage collection and clears cached CUDA memory after a scene or completed job."
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def execute(self):
        collected = gc.collect()
        cuda_status = "CUDA unavailable"
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                cuda_status = "CUDA cache cleared"
        except (ImportError, RuntimeError) as error:
            cuda_status = f"CUDA cleanup skipped: {error}"
        return (f"Python objects collected: {collected}; {cuda_status}",)


class TenMinSaveSceneFrameNode(_AlwaysRun):
    CATEGORY = "10MinVideoMaker/Artifacts"
    DESCRIPTION = (
        f"Atomically caches the exact {PRODUCTION_WIDTH}x{PRODUCTION_HEIGHT} scene frame "
        "at a deterministic project path."
    )
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("frame_path",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "job_id": ("STRING", {"default": ""}),
                "scene_id": ("INT", {"default": 1, "min": 1}),
                "revision": ("INT", {"default": 1, "min": 1}),
            }
        }

    def execute(self, images, job_id: str, scene_id: int, revision: int = 1):
        try:
            path = save_scene_frame(images, job_id, scene_id, revision)
        except ArtifactError as error:
            raise RuntimeError(str(error)) from error
        return (str(path),)


class TenMinStitchClipsNode(_AlwaysRun):
    CATEGORY = "10MinVideoMaker/Assembly"
    DESCRIPTION = (
        f"FFprobes {PRODUCTION_WIDTH}x{PRODUCTION_HEIGHT}/{PRODUCTION_FPS} fps clips, "
        "then concatenates them to the final project output."
    )
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("final_video_path", "status")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "job_id": ("STRING", {"default": ""}),
                "clip_paths_json": ("STRING", {"multiline": True, "default": "[]"}),
            }
        }

    def execute(self, job_id: str, clip_paths_json: str):
        try:
            clip_paths = json.loads(clip_paths_json)
        except json.JSONDecodeError as error:
            raise AssemblyError("clip_paths_json must contain valid JSON.") from error
        if not isinstance(clip_paths, list) or not all(isinstance(path, str) for path in clip_paths):
            raise AssemblyError("clip_paths_json must be a JSON array of file paths.")
        validate_video_profile(probe_video(path) for path in clip_paths)
        final_path = FfmpegAssembler().stitch(
            job_id,
            clip_paths,
            STORAGE.temp_root / "concat",
        )
        return (str(final_path), "stitching complete")


NODE_CLASS_MAPPINGS = {
    "10MinVideoMaker_ValidateJob": TenMinValidateJobNode,
    "10MinVideoMaker_PipelineStatus": TenMinPipelineStatusNode,
    "10MinVideoMaker_RequestGrokJob": TenMinRequestGrokJobNode,
    "10MinVideoMaker_PollGmail": TenMinPollGmailNode,
    "10MinVideoMaker_ResolveLoras": TenMinResolveLorasNode,
    "10MinVideoMaker_ReleaseMemory": TenMinReleaseMemoryNode,
    "10MinVideoMaker_SaveSceneFrame": TenMinSaveSceneFrameNode,
    "10MinVideoMaker_StitchClips": TenMinStitchClipsNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "10MinVideoMaker_ValidateJob": "10Min Video Maker: Validate Job",
    "10MinVideoMaker_PipelineStatus": "10Min Video Maker: Pipeline Status",
    "10MinVideoMaker_RequestGrokJob": "10Min Video Maker: Request Grok Job",
    "10MinVideoMaker_PollGmail": "10Min Video Maker: Poll Gmail Once",
    "10MinVideoMaker_ResolveLoras": "10Min Video Maker: Resolve LoRAs",
    "10MinVideoMaker_ReleaseMemory": "10Min Video Maker: Release Memory",
    "10MinVideoMaker_SaveSceneFrame": "10Min Video Maker: Save Scene Frame",
    "10MinVideoMaker_StitchClips": "10Min Video Maker: Stitch Clips",
}
