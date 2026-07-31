"""ComfyUI nodes exposing the shared 10MinVideoMaker services."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import torch

from .assembly import AssemblyError, FfmpegAssembler, probe_video, validate_video_profile
from .artifacts import ArtifactError, save_scene_frame
from .assets import LocalLoraRequirement, LoraAssetManager
from .chunk_artifacts import (
    ChunkArtifactError,
    LATENT_CHECKPOINT_SCHEMA_VERSION,
    load_latent_checkpoint,
    save_latent_checkpoint,
    sha256_file,
)
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
from .storage import StorageError, StorageLayout

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STORAGE = StorageLayout.configured()
CHUNK_ARTIFACT_KINDS = [
    "stage1_handoff",
    "stage2_video",
    "stage2_audio",
]


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


def _chunk_checkpoint_fingerprint(
    job_id: str,
    scene_id: int,
    revision: int,
    chunk_index: int,
    attempt_number: int,
    artifact_kind: str,
    expected_temporal_tokens: int,
) -> str | float:
    """Return a verified content hash, or NaN so an invalid checkpoint is never cached."""
    try:
        checkpoint = STORAGE.chunk_checkpoint_path(
            job_id,
            scene_id,
            revision,
            chunk_index,
            attempt_number,
            artifact_kind,
        )
        manifest_path = STORAGE.chunk_checkpoint_manifest_path(
            job_id,
            scene_id,
            revision,
            chunk_index,
            attempt_number,
            artifact_kind,
        )
        if not checkpoint.is_file() or not manifest_path.is_file():
            return float("nan")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            return float("nan")
        expected_identity = {
            "schema_version": LATENT_CHECKPOINT_SCHEMA_VERSION,
            "artifact_kind": artifact_kind,
            "job_id": job_id,
            "scene_id": scene_id,
            "revision": revision,
            "chunk_index": chunk_index,
            "attempt_number": attempt_number,
            "checkpoint_path": str(checkpoint),
        }
        if any(manifest.get(field) != value for field, value in expected_identity.items()):
            return float("nan")
        declared_hash = manifest.get("sha256")
        if (
            not isinstance(declared_hash, str)
            or len(declared_hash) != 64
            or any(character not in "0123456789abcdef" for character in declared_hash)
            or manifest.get("byte_size") != checkpoint.stat().st_size
            or sha256_file(checkpoint) != declared_hash
        ):
            return float("nan")
        tensor_manifest = manifest.get("tensors", {})
        sample_shape = (
            tensor_manifest.get("samples", {}).get("shape")
            if isinstance(tensor_manifest, dict)
            else None
        )
        if artifact_kind != "stage2_audio" and (
            not isinstance(sample_shape, list)
            or len(sample_shape) < 3
            or sample_shape[2] != expected_temporal_tokens
        ):
            return float("nan")
        return f"{declared_hash}:{expected_temporal_tokens}"
    except (OSError, json.JSONDecodeError, StorageError):
        return float("nan")


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
            STORAGE.asset_manifest_path,
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


class TenMinSaveChunkLatentNode(_AlwaysRun):
    CATEGORY = "10MinVideoMaker/Artifacts"
    DESCRIPTION = (
        "Atomically checkpoints one plain LTX video or audio latent under the "
        "project-owned D-drive revision/chunk coordinates."
    )
    FUNCTION = "execute"
    RETURN_TYPES = ("LATENT", "STRING", "STRING")
    RETURN_NAMES = ("latent", "checkpoint_path", "checkpoint_sha256")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "job_id": ("STRING", {"default": ""}),
                "scene_id": ("INT", {"default": 1, "min": 1}),
                "revision": ("INT", {"default": 1, "min": 1}),
                "chunk_index": ("INT", {"default": 0, "min": 0}),
                "attempt_number": ("INT", {"default": 1, "min": 1}),
                "artifact_kind": (
                    CHUNK_ARTIFACT_KINDS,
                    {"default": "stage1_handoff"},
                ),
            }
        }

    def execute(
        self,
        latent,
        job_id: str,
        scene_id: int,
        revision: int,
        chunk_index: int,
        attempt_number: int,
        artifact_kind: str = "stage1_handoff",
    ):
        try:
            checkpoint, manifest = save_latent_checkpoint(
                STORAGE,
                latent,
                job_id=job_id,
                scene_id=scene_id,
                revision=revision,
                chunk_index=chunk_index,
                attempt_number=attempt_number,
                artifact_kind=artifact_kind,
            )
        except (ChunkArtifactError, StorageError) as error:
            raise RuntimeError(str(error)) from error
        return (latent, str(checkpoint), str(manifest["sha256"]))


class TenMinLoadChunkLatentNode:
    CATEGORY = "10MinVideoMaker/Artifacts"
    DESCRIPTION = (
        "Loads a finalized, SHA-256-verified LTX video or audio latent using "
        "project-owned revision/chunk coordinates."
    )
    FUNCTION = "execute"
    RETURN_TYPES = ("LATENT", "STRING", "STRING")
    RETURN_NAMES = ("latent", "checkpoint_path", "checkpoint_sha256")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "job_id": ("STRING", {"default": ""}),
                "scene_id": ("INT", {"default": 1, "min": 1}),
                "revision": ("INT", {"default": 1, "min": 1}),
                "chunk_index": ("INT", {"default": 0, "min": 0}),
                "attempt_number": ("INT", {"default": 1, "min": 1}),
                "artifact_kind": (
                    CHUNK_ARTIFACT_KINDS,
                    {"default": "stage1_handoff"},
                ),
                "expected_temporal_tokens": (
                    "INT",
                    {"default": 16, "min": 1},
                ),
            }
        }

    @classmethod
    def IS_CHANGED(
        cls,
        job_id: str,
        scene_id: int,
        revision: int,
        chunk_index: int,
        attempt_number: int,
        artifact_kind: str = "stage1_handoff",
        expected_temporal_tokens: int = 16,
    ):
        return _chunk_checkpoint_fingerprint(
            job_id,
            scene_id,
            revision,
            chunk_index,
            attempt_number,
            artifact_kind,
            expected_temporal_tokens,
        )

    def execute(
        self,
        job_id: str,
        scene_id: int,
        revision: int,
        chunk_index: int,
        attempt_number: int,
        artifact_kind: str = "stage1_handoff",
        expected_temporal_tokens: int = 16,
    ):
        try:
            latent, manifest = load_latent_checkpoint(
                STORAGE,
                job_id=job_id,
                scene_id=scene_id,
                revision=revision,
                chunk_index=chunk_index,
                attempt_number=attempt_number,
                artifact_kind=artifact_kind,
                expected_temporal_tokens=expected_temporal_tokens,
            )
            checkpoint = STORAGE.chunk_checkpoint_path(
                job_id,
                scene_id,
                revision,
                chunk_index,
                attempt_number,
                artifact_kind,
            )
        except (ChunkArtifactError, StorageError) as error:
            raise RuntimeError(str(error)) from error
        return (latent, str(checkpoint), str(manifest["sha256"]))


def _clone_conditioning_value(value: Any) -> Any:
    """Clone CONDITIONING containers and tensors without changing their meaning."""
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, list):
        return [_clone_conditioning_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_conditioning_value(item) for item in value)
    if isinstance(value, dict):
        return {key: _clone_conditioning_value(item) for key, item in value.items()}
    return value


class TenMinIsolateConditioningNode(_AlwaysRun):
    """Create a fresh copy before an LTX node may attach mutable guide state."""

    CATEGORY = "10MinVideoMaker/Continuation"
    DESCRIPTION = (
        "Clones text conditioning immediately before LTX conditioning so cached "
        "guide metadata cannot leak into another continuation prompt."
    )
    FUNCTION = "execute"
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "scope": ("STRING", {"default": "continuation"}),
            }
        }

    def execute(self, conditioning, scope: str):
        if not scope.strip():
            raise RuntimeError("scope must not be blank.")
        return (_clone_conditioning_value(conditioning),)


class TenMinIsolateModelNode(_AlwaysRun):
    """Clone one ModelPatcher wrapper before LTX may mutate its options."""

    CATEGORY = "10MinVideoMaker/Continuation"
    DESCRIPTION = (
        "Clones the ModelPatcher wrapper without duplicating model weights so "
        "LTX continuation state cannot leak through ComfyUI's model cache."
    )
    FUNCTION = "execute"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "scope": ("STRING", {"default": "continuation"}),
            }
        }

    def execute(self, model, scope: str):
        if not scope.strip():
            raise RuntimeError("scope must not be blank.")
        clone = getattr(model, "clone", None)
        if not callable(clone):
            raise RuntimeError("MODEL input does not support ModelPatcher.clone().")
        isolated = clone()
        if isolated is model:
            raise RuntimeError("ModelPatcher.clone() returned the cached model instance.")
        return (isolated,)


class TenMinFreshCheckpointNode(_AlwaysRun):
    """Construct a fresh checkpoint wrapper for one mutable LTX continuation phase."""

    CATEGORY = "10MinVideoMaker/Continuation"
    DESCRIPTION = (
        "Forces a fresh CheckpointLoaderSimple result so mutable LTX continuation "
        "state cannot survive ComfyUI's static loader cache."
    )
    FUNCTION = "execute"
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths

        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "scope": ("STRING", {"default": "continuation"}),
            }
        }

    def execute(self, ckpt_name: str, scope: str):
        if not scope.strip():
            raise RuntimeError("scope must not be blank.")
        # Call ComfyUI's public legacy loader surface rather than reimplementing
        # checkpoint parsing. `_AlwaysRun` gives each continuation phase a fresh
        # ModelPatcher/CLIP/VAE wrapper while ComfyUI may still retain weights.
        import nodes as comfy_nodes

        loader = getattr(comfy_nodes, "CheckpointLoaderSimple", None)
        if loader is None:
            raise RuntimeError("ComfyUI CheckpointLoaderSimple is unavailable.")
        result = loader().load_checkpoint(ckpt_name)
        if not isinstance(result, tuple) or len(result) != 3:
            raise RuntimeError("ComfyUI CheckpointLoaderSimple returned an invalid result.")
        return result


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
    "10MinVideoMaker_SaveChunkLatent": TenMinSaveChunkLatentNode,
    "10MinVideoMaker_LoadChunkLatent": TenMinLoadChunkLatentNode,
    "10MinVideoMaker_IsolateConditioning": TenMinIsolateConditioningNode,
    "10MinVideoMaker_IsolateModel": TenMinIsolateModelNode,
    "10MinVideoMaker_FreshCheckpoint": TenMinFreshCheckpointNode,
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
    "10MinVideoMaker_SaveChunkLatent": "10Min Video Maker: Save Chunk Latent",
    "10MinVideoMaker_LoadChunkLatent": "10Min Video Maker: Load Chunk Latent",
    "10MinVideoMaker_IsolateConditioning": "10Min Video Maker: Isolate Conditioning",
    "10MinVideoMaker_IsolateModel": "10Min Video Maker: Isolate Model",
    "10MinVideoMaker_FreshCheckpoint": "10Min Video Maker: Fresh Continuation Checkpoint",
    "10MinVideoMaker_StitchClips": "10Min Video Maker: Stitch Clips",
}
