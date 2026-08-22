"""Build scene-specific ComfyUI API workflows from the validated job contract."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .assets import predictable_lora_filename
from .constants import (
    I2V_BASE_HEIGHT,
    I2V_BASE_WIDTH,
    I2V_DYNAMIC_BASE_MODEL,
    I2V_FIRST_PASS_SAMPLER,
    I2V_FIRST_PASS_SIGMAS,
    I2V_SPATIAL_UPSCALER,
    I2V_UPSCALE_PASS_SAMPLER,
    I2V_UPSCALE_PASS_SIGMAS,
    LTX_CHECKPOINT,
    LTX_TEXT_ENCODER,
    MANDATORY_I2V_LORAS,
    PRODUCTION_FPS,
    PRODUCTION_HEIGHT,
    PRODUCTION_WIDTH,
)
from .contracts import (
    JobPayload,
    LoraSpec,
    SceneSpec,
    effective_i2v_loras,
    effective_t2i_loras,
    lora_identity,
)
from .delivery import DiscordDeliverySettings
from .qc_contracts import QcArtifactStage
from .review import SceneWorkflowOverrides
from .storage import StorageLayout

ANIMA_UNET = "CyberRealistic_AnimaSemi_V6.0.safetensors"
ANIMA_TEXT_ENCODER = "cyberrealisticAnima_v30_txt.safetensors"
ANIMA_VAE = "qwen_image_vae_cybv2.safetensors"
PONY_CHECKPOINT = "cyberrealisticPony_v180Coreshift.safetensors"

class WorkflowBuildError(ValueError):
    """Raised when a workflow cannot be built safely from the supplied scene."""


@dataclass(frozen=True)
class WorkflowBuild:
    api: dict[str, dict[str, Any]]
    output_node_id: str
    filename_prefix: str


class _Graph:
    def __init__(self, *, node_id_offset: int = 0) -> None:
        if isinstance(node_id_offset, bool) or not isinstance(node_id_offset, int):
            raise WorkflowBuildError("node_id_offset must be an integer.")
        if node_id_offset < 0:
            raise WorkflowBuildError("node_id_offset must be non-negative.")
        self.nodes: dict[str, dict[str, Any]] = {}
        self._next_id = node_id_offset + 1

    def add(self, class_type: str, title: str, **inputs: Any) -> str:
        node_id = str(self._next_id)
        self._next_id += 1
        self.nodes[node_id] = {
            "class_type": class_type,
            "_meta": {"title": title},
            "inputs": inputs,
        }
        return node_id

    @staticmethod
    def output(node_id: str, slot: int = 0) -> list[Any]:
        return [node_id, slot]


def _sigma_string(values: Iterable[float]) -> str:
    return ", ".join(f"{value:g}" for value in values)


def _scene_prefix(job_id: str, stage: str, scene_id: int) -> str:
    return f"10MinVideoMaker/{job_id}/{stage}/scene_{scene_id:04d}"


def _local_lora_filename(
    lora: LoraSpec,
    resolved_filenames: Mapping[str, str] | None,
) -> str:
    if resolved_filenames:
        resolved = resolved_filenames.get(lora_identity(lora))
        if resolved:
            return resolved
    return predictable_lora_filename(lora.name)


def _validated_i2v_lora_filename(
    lora: LoraSpec,
    resolved_filenames: Mapping[str, str] | None,
) -> str:
    key = f"i2v:{lora_identity(lora)}"
    resolved = resolved_filenames.get(key) if resolved_filenames else None
    if not resolved:
        raise WorkflowBuildError(
            f"Dynamic I2V LoRA {lora.name} has not been verified as "
            f"{I2V_DYNAMIC_BASE_MODEL}; refusing to attach it to the LTX model."
        )
    return resolved


def _apply_t2i_loras(
    graph: _Graph,
    model: list[Any],
    clip: list[Any],
    loras: Iterable[LoraSpec],
    resolved_filenames: Mapping[str, str] | None,
) -> tuple[list[Any], list[Any]]:
    for lora in loras:
        loader = graph.add(
            "LoraLoader",
            f"T2I LoRA: {lora.name}",
            model=model,
            clip=clip,
            lora_name=_local_lora_filename(lora, resolved_filenames),
            strength_model=lora.weight,
            strength_clip=lora.weight,
        )
        model = graph.output(loader, 0)
        clip = graph.output(loader, 1)
    return model, clip


def _watermarked_discord_media(
    graph: _Graph,
    images: list[Any],
    *,
    title: str,
) -> list[Any]:
    watermark = graph.add(
        "DaSiWa_Watermark",
        title,
        images=images,
        watermark_path="wm.png",
        position="bottom-right",
        scale=0.7,
        resampling="bicubic",
        transparency=0.4,
        rotation=0,
        padding_x=20,
        padding_y=20,
        optical_padding=False,
        optical_strength=0.4,
        random_switches=3,
        fade=False,
        fade_margin=0.1,
        randomize_position=False,
        random_seed=0,
    )
    return graph.output(watermark)


def _add_discord_image_delivery(
    graph: _Graph,
    images: list[Any],
    job: JobPayload,
    scene: SceneSpec,
    delivery: DiscordDeliverySettings | None,
) -> None:
    if delivery is None:
        return
    watermarked = _watermarked_discord_media(
        graph,
        images,
        title="Watermark Discord scene image",
    )
    graph.add(
        "DiscordSendSaveImage",
        "Send metadata-free watermarked image to Discord",
        images=watermarked,
        filename_prefix=f"10MinVideoMaker-{job.job_id}-scene-{scene.scene_id:04d}",
        overwrite_last=False,
        file_format="png",
        quality=100,
        lossless=True,
        save_output=False,
        show_preview=False,
        resize_to_power_of_2=False,
        resize_method="lanczos",
        include_format_in_message=False,
        group_batched_images=True,
        add_date=True,
        add_time=True,
        add_dimensions=True,
        send_to_discord=True,
        webhook_url=delivery.webhook_url,
        discord_message=(
            f"10MinVideoMaker job {job.job_id} — scene {scene.scene_id}: {scene.title}"
        ),
        include_prompts_in_message=False,
        send_workflow_json=False,
        save_cdn_urls=False,
        github_cdn_update=False,
        github_repo="",
        github_token="",
        github_file_path="cdn_urls.md",
    )


def _add_discord_video_delivery(
    graph: _Graph,
    images: list[Any],
    audio: list[Any],
    job: JobPayload,
    scene: SceneSpec,
    delivery: DiscordDeliverySettings | None,
) -> None:
    if delivery is None:
        return
    watermarked = _watermarked_discord_media(
        graph,
        images,
        title="Watermark Discord scene video",
    )
    graph.add(
        "DiscordSendSaveVideo",
        "Send metadata-free watermarked video to Discord",
        images=watermarked,
        audio=audio,
        filename_prefix=f"10MinVideoMaker-{job.job_id}-scene-{scene.scene_id:04d}",
        overwrite_last=False,
        format="video/h264-mp4",
        frame_rate=float(PRODUCTION_FPS),
        quality=65,
        loop_count=0,
        lossless=False,
        pingpong=False,
        save_output=False,
        include_video_info=True,
        add_date=True,
        add_time=True,
        add_dimensions=True,
        send_to_discord=True,
        webhook_url=delivery.webhook_url,
        discord_message=(
            f"10MinVideoMaker job {job.job_id} — scene {scene.scene_id}: {scene.title}"
        ),
        include_prompts_in_message=False,
        send_workflow_json=False,
        save_cdn_urls=False,
        github_cdn_update=False,
        github_repo="",
        github_token="",
        github_file_path="cdn_urls.md",
    )


def build_t2i_api_workflow(
    job: JobPayload,
    scene: SceneSpec,
    resolved_lora_filenames: Mapping[str, str] | None = None,
    *,
    delivery: DiscordDeliverySettings | None = None,
    overrides: SceneWorkflowOverrides | None = None,
    revision: int = 1,
) -> WorkflowBuild:
    """Build the matching Anima or Pony workflow for one scene."""
    graph = _Graph()
    family = job.character.base_model.casefold()
    if family == "anima":
        unet = graph.add(
            "UNETLoader",
            "Anima diffusion model",
            unet_name=ANIMA_UNET,
            weight_dtype="default",
        )
        clip_loader = graph.add(
            "CLIPLoader",
            "Anima text encoder",
            clip_name=ANIMA_TEXT_ENCODER,
            type="lumina2",
            device="default",
        )
        vae_loader = graph.add("VAELoader", "Anima VAE", vae_name=ANIMA_VAE)
        model = graph.output(unet)
        clip = graph.output(clip_loader)
        vae = graph.output(vae_loader)
    elif family == "pony":
        checkpoint = graph.add(
            "CheckpointLoaderSimple",
            "Pony checkpoint",
            ckpt_name=PONY_CHECKPOINT,
        )
        model = graph.output(checkpoint, 0)
        clip = graph.output(checkpoint, 1)
        vae = graph.output(checkpoint, 2)
    else:
        raise WorkflowBuildError(f"Unsupported T2I base model: {job.character.base_model}")

    model, clip = _apply_t2i_loras(
        graph,
        model,
        clip,
        effective_t2i_loras(scene, job.character),
        resolved_lora_filenames,
    )
    positive = graph.add(
        "CLIPTextEncode",
        "Scene positive prompt",
        text=scene.t2i.prompt,
        clip=clip,
    )
    negative = graph.add(
        "CLIPTextEncode",
        "Scene negative prompt",
        text=scene.t2i.negative,
        clip=clip,
    )
    latent_type = "EmptySD3LatentImage" if family == "anima" else "EmptyLatentImage"
    latent = graph.add(
        latent_type,
        f"{PRODUCTION_WIDTH}x{PRODUCTION_HEIGHT} production latent",
        width=PRODUCTION_WIDTH,
        height=PRODUCTION_HEIGHT,
        batch_size=1,
    )

    if family == "anima":
        pass_settings = (
            overrides.t2i_passes[0]
            if overrides is not None
            else {
                "sampler": "er_sde",
                "scheduler": "beta57",
                "steps": 30,
                "cfg": 4.5,
                "denoise": 1.0,
            }
        )
        sampled = graph.add(
            "KSampler",
            f"Anima {pass_settings['sampler']} {pass_settings['scheduler']}",
            model=model,
            seed=scene.t2i.seed,
            steps=pass_settings["steps"],
            cfg=pass_settings["cfg"],
            sampler_name=pass_settings["sampler"],
            scheduler=pass_settings["scheduler"],
            positive=graph.output(positive),
            negative=graph.output(negative),
            latent_image=graph.output(latent),
            denoise=pass_settings.get("denoise", 1.0),
        )
    else:
        pony_passes = (
            overrides.t2i_passes
            if overrides is not None
            else (
                {
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
                    "sampler": "res_5s_ode",
                    "scheduler": "karras",
                    "steps": 30,
                    "cfg": 6.0,
                    "start_step": 0,
                    "end_step": 30,
                    "add_noise": True,
                    "return_with_leftover_noise": False,
                },
            )
        )
        first_settings, second_settings = pony_passes
        first = graph.add(
            "KSamplerAdvanced",
            f"Pony pass 1: {first_settings['sampler']}",
            model=model,
            add_noise="enable" if first_settings.get("add_noise", True) else "disable",
            noise_seed=scene.t2i.seed,
            steps=first_settings["steps"],
            cfg=first_settings["cfg"],
            sampler_name=first_settings["sampler"],
            scheduler=first_settings["scheduler"],
            positive=graph.output(positive),
            negative=graph.output(negative),
            latent_image=graph.output(latent),
            start_at_step=first_settings.get("start_step", 0),
            end_at_step=first_settings.get("end_step", first_settings["steps"]),
            return_with_leftover_noise=(
                "enable"
                if first_settings.get("return_with_leftover_noise", False)
                else "disable"
            ),
        )
        sampled = graph.add(
            "KSamplerAdvanced",
            f"Pony pass 2: {second_settings['sampler']}",
            model=model,
            add_noise="enable" if second_settings.get("add_noise", True) else "disable",
            noise_seed=scene.t2i.seed,
            steps=second_settings["steps"],
            cfg=second_settings["cfg"],
            sampler_name=second_settings["sampler"],
            scheduler=second_settings["scheduler"],
            positive=graph.output(positive),
            negative=graph.output(negative),
            latent_image=graph.output(first),
            start_at_step=second_settings.get("start_step", 0),
            end_at_step=second_settings.get("end_step", second_settings["steps"]),
            return_with_leftover_noise=(
                "enable"
                if second_settings.get("return_with_leftover_noise", False)
                else "disable"
            ),
        )

    decoded = graph.add(
        "VAEDecode",
        "Decode initial frame",
        samples=graph.output(sampled),
        vae=vae,
    )
    final_image = graph.output(decoded)
    if family == "pony":
        detailer_settings = (
            overrides.face_detailer
            if overrides is not None and overrides.face_detailer is not None
            else {
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
        )
    if family == "pony" and detailer_settings.get("enabled", True):
        detector = graph.add(
            "UltralyticsDetectorProvider",
            "Pony face bbox detector",
            model_name=detailer_settings.get("detector", "bbox/face_yolov8m.pt"),
        )
        detailer = graph.add(
            "FaceDetailer",
            "Pony bbox face detailer",
            image=final_image,
            model=model,
            clip=clip,
            vae=vae,
            guide_size=detailer_settings.get("guide_size", 512),
            guide_size_for=True,
            max_size=detailer_settings.get("max_size", 1024),
            seed=scene.t2i.seed,
            steps=detailer_settings["steps"],
            cfg=detailer_settings["cfg"],
            sampler_name=detailer_settings["sampler"],
            scheduler=detailer_settings["scheduler"],
            positive=graph.output(positive),
            negative=graph.output(negative),
            denoise=detailer_settings["denoise"],
            feather=detailer_settings.get("feather", 5),
            noise_mask=True,
            force_inpaint=True,
            bbox_threshold=detailer_settings.get("bbox_threshold", 0.5),
            bbox_dilation=detailer_settings.get("bbox_dilation", 10),
            bbox_crop_factor=detailer_settings.get("bbox_crop_factor", 3.0),
            sam_detection_hint="center-1",
            sam_dilation=0,
            sam_threshold=0.93,
            sam_bbox_expansion=0,
            sam_mask_hint_threshold=0.7,
            sam_mask_hint_use_negative="False",
            drop_size=detailer_settings.get("drop_size", 120),
            bbox_detector=graph.output(detector, 0),
            wildcard="",
            cycle=1,
            inpaint_model=False,
            noise_mask_feather=detailer_settings.get("noise_mask_feather", 20),
            tiled_encode=False,
            tiled_decode=False,
        )
        final_image = graph.output(detailer, 0)
    saved = graph.add(
        "10MinVideoMaker_SaveSceneFrame",
        "Cache deterministic scene frame",
        images=final_image,
        job_id=job.job_id,
        scene_id=scene.scene_id,
        revision=revision,
    )
    _add_discord_image_delivery(graph, final_image, job, scene, delivery)
    return WorkflowBuild(
        api=graph.nodes,
        output_node_id=saved,
        filename_prefix=_scene_prefix(job.job_id, "frames", scene.scene_id),
    )


def _apply_i2v_loras(
    graph: _Graph,
    model: list[Any],
    job: JobPayload,
    scene: SceneSpec,
    resolved_filenames: Mapping[str, str] | None,
) -> list[Any]:
    for filename, weight in MANDATORY_I2V_LORAS:
        loader = graph.add(
            "LoraLoaderModelOnly",
            f"Mandatory I2V LoRA: {filename}",
            model=model,
            lora_name=filename,
            strength_model=weight,
        )
        model = graph.output(loader)
    for lora in effective_i2v_loras(job, scene):
        loader = graph.add(
            "LoraLoaderModelOnly",
            f"Scene I2V LoRA: {lora.name}",
            model=model,
            lora_name=_validated_i2v_lora_filename(lora, resolved_filenames),
            strength_model=lora.weight,
        )
        model = graph.output(loader)
    return model


def build_i2v_api_workflow(
    job: JobPayload,
    scene: SceneSpec,
    cached_frame_path: str | Path,
    resolved_lora_filenames: Mapping[str, str] | None = None,
    *,
    delivery: DiscordDeliverySettings | None = None,
    overrides: SceneWorkflowOverrides | None = None,
    artifact_stage: QcArtifactStage = QcArtifactStage.LEGACY_FINAL,
    revision: int = 1,
    attempt_number: int = 1,
) -> WorkflowBuild:
    """Build an explicit first-pass draft, approved final, or legacy graph."""
    frame_path = Path(cached_frame_path)
    if not frame_path.is_absolute():
        raise WorkflowBuildError("cached_frame_path must be absolute.")
    try:
        artifact_stage = QcArtifactStage(artifact_stage)
    except ValueError as error:
        raise WorkflowBuildError("Unknown I2V artifact stage.") from error
    if revision < 1 or attempt_number < 1:
        raise WorkflowBuildError("revision and attempt_number must be positive.")

    graph = _Graph()
    first_pass = (
        overrides.i2v_first_pass
        if overrides is not None
        else {
            "sampler": I2V_FIRST_PASS_SAMPLER,
            "sigmas": I2V_FIRST_PASS_SIGMAS,
            "cfg": 1.0,
            "reference_strength": 0.75,
            "image_strength": 0.75,
            "image_compression": 35,
        }
    )
    second_pass = (
        overrides.i2v_second_pass
        if overrides is not None
        else {
            "sampler": I2V_UPSCALE_PASS_SAMPLER,
            "sigmas": I2V_UPSCALE_PASS_SIGMAS,
            "cfg": 1.0,
            "reference_strength": 1.0,
            "image_strength": 1.0,
            "image_compression": 30,
        }
    )
    chunking = (
        overrides.chunking
        if overrides is not None
        else {"chunks": 2, "dimension_threshold": 4096}
    )
    upscaler_settings = (
        overrides.spatial_upscaler
        if overrides is not None
        else {
            "model": I2V_SPATIAL_UPSCALER,
            "tile_size": 11,
            "overlap": 6,
            "max_size_without_tiling": 22,
        }
    )
    checkpoint = graph.add(
        "CheckpointLoaderSimple",
        "LTX 2.3 checkpoint",
        ckpt_name=LTX_CHECKPOINT,
    )
    text_encoder = graph.add(
        "LTXAVTextEncoderLoader",
        "LTX 2.3 text encoder",
        text_encoder=LTX_TEXT_ENCODER,
        ckpt_name=LTX_CHECKPOINT,
        device="default",
    )
    audio_vae = graph.add(
        "LTXVAudioVAELoader",
        "LTX audio VAE",
        ckpt_name=LTX_CHECKPOINT,
    )

    model = _apply_i2v_loras(
        graph,
        graph.output(checkpoint, 0),
        job,
        scene,
        resolved_lora_filenames,
    )
    chunked = graph.add(
        "LTXVChunkFeedForward",
        "16 GB VRAM chunking",
        model=model,
        chunks=chunking["chunks"],
        dim_threshold=chunking["dimension_threshold"],
    )
    reference_enabled = graph.add(
        "LTXReferenceEnable",
        "Enable image reference conditioning",
        model=graph.output(chunked),
        zero_ref_timesteps=False,
        verbose=False,
    )

    positive = graph.add(
        "CLIPTextEncode",
        "I2V motion prompt",
        text=scene.i2v.prompt,
        clip=graph.output(text_encoder),
    )
    negative = graph.add(
        "CLIPTextEncode",
        "I2V negative prompt",
        text=scene.i2v.negative,
        clip=graph.output(text_encoder),
    )
    conditioning = graph.add(
        "LTXVConditioning",
        "24 fps conditioning",
        positive=graph.output(positive),
        negative=graph.output(negative),
        frame_rate=float(PRODUCTION_FPS),
    )

    source = graph.add(
        "VHS_LoadImagePath",
        "Load exact cached T2I frame",
        image=str(frame_path),
        custom_width=0,
        custom_height=0,
    )
    if artifact_stage == QcArtifactStage.FINAL:
        temporal_tokens = (scene.frame_count - 1) // 8 + 1
        draft_video = graph.add(
            "10MinVideoMaker_LoadChunkLatent",
            "Load exact approved first-pass video latent",
            job_id=job.job_id,
            scene_id=scene.scene_id,
            revision=revision,
            chunk_index=0,
            attempt_number=attempt_number,
            artifact_kind="stage1_handoff",
            expected_temporal_tokens=temporal_tokens,
            storage_root=str(StorageLayout.configured().root),
        )
        draft_audio = graph.add(
            "10MinVideoMaker_LoadChunkLatent",
            "Load exact approved first-pass audio latent",
            job_id=job.job_id,
            scene_id=scene.scene_id,
            revision=revision,
            chunk_index=0,
            attempt_number=attempt_number,
            artifact_kind="stage1_audio",
            expected_temporal_tokens=1,
            storage_root=str(StorageLayout.configured().root),
        )
        draft_video_latent = graph.output(draft_video)
        draft_audio_latent = graph.output(draft_audio)
    else:
        base_image = graph.add(
            "ImageScale",
            "Scale frame for first pass",
            image=graph.output(source),
            upscale_method="lanczos",
            width=I2V_BASE_WIDTH,
            height=I2V_BASE_HEIGHT,
            crop="disabled",
        )
        base_preprocessed = graph.add(
            "LTXVPreprocess",
            "Preprocess first-pass reference",
            image=graph.output(base_image),
            img_compression=first_pass["image_compression"],
        )
        base_latent = graph.add(
            "EmptyLTXVLatentVideo",
            "Half-resolution video latent",
            width=I2V_BASE_WIDTH,
            height=I2V_BASE_HEIGHT,
            length=scene.frame_count,
            batch_size=1,
        )
        first_i2v = graph.add(
            "LTXVImgToVideoInplaceKJ",
            "Inject cached frame into first pass",
            vae=graph.output(checkpoint, 2),
            latent=graph.output(base_latent),
            num_images="1",
            **{
                "num_images.strength_1": first_pass["image_strength"],
                "num_images.image_1": graph.output(base_preprocessed),
                "num_images.index_1": 0,
            },
        )
        empty_audio = graph.add(
            "LTXVEmptyLatentAudio",
            "Matching empty audio latent",
            frames_number=scene.frame_count,
            frame_rate=PRODUCTION_FPS,
            batch_size=1,
            audio_vae=graph.output(audio_vae),
        )
        first_av = graph.add(
            "LTXVConcatAVLatent",
            "First-pass AV latent",
            video_latent=graph.output(first_i2v),
            audio_latent=graph.output(empty_audio),
        )
        first_model = graph.add(
            "LTXReferenceConditioning",
            "First-pass reference model",
            model=graph.output(reference_enabled),
            vae=graph.output(checkpoint, 2),
            image=graph.output(base_preprocessed),
            target_latent=graph.output(first_i2v),
            strength=first_pass["reference_strength"],
            position_mode="reference",
            verbose=False,
        )
        first_sampler = graph.add(
            "KSamplerSelect",
            "I2V first-pass sampler",
            sampler_name=first_pass["sampler"],
        )
        first_sigmas = graph.add(
            "ManualSigmas",
            "Verified first-pass sigmas",
            sigmas=_sigma_string(first_pass["sigmas"]),
        )
        first_sampled = graph.add(
            "SamplerCustom",
            "First sampling pass",
            model=graph.output(first_model),
            add_noise=True,
            noise_seed=scene.i2v.seed,
            cfg=first_pass["cfg"],
            positive=graph.output(conditioning, 0),
            negative=graph.output(conditioning, 1),
            sampler=graph.output(first_sampler),
            sigmas=graph.output(first_sigmas),
            latent_image=graph.output(first_av),
        )
        first_split = graph.add(
            "LTXVSeparateAVLatent",
            "Split first-pass AV latent",
            av_latent=graph.output(first_sampled, 1),
        )
        draft_video_latent = graph.output(first_split, 0)
        draft_audio_latent = graph.output(first_split, 1)

        if artifact_stage == QcArtifactStage.DRAFT:
            saved_video = graph.add(
                "10MinVideoMaker_SaveChunkLatent",
                "Checkpoint approved-draft video latent",
                latent=draft_video_latent,
                job_id=job.job_id,
                scene_id=scene.scene_id,
                revision=revision,
                chunk_index=0,
                attempt_number=attempt_number,
                artifact_kind="stage1_handoff",
                storage_root=str(StorageLayout.configured().root),
            )
            saved_audio = graph.add(
                "10MinVideoMaker_SaveChunkLatent",
                "Checkpoint approved-draft audio latent",
                latent=draft_audio_latent,
                job_id=job.job_id,
                scene_id=scene.scene_id,
                revision=revision,
                chunk_index=0,
                attempt_number=attempt_number,
                artifact_kind="stage1_audio",
                storage_root=str(StorageLayout.configured().root),
            )
            draft_decoded_video = graph.add(
                "VAEDecode",
                "Decode first-pass draft video",
                samples=graph.output(saved_video),
                vae=graph.output(checkpoint, 2),
            )
            draft_decoded_audio = graph.add(
                "LTXVAudioVAEDecode",
                "Decode first-pass draft audio",
                samples=graph.output(saved_audio),
                audio_vae=graph.output(audio_vae),
            )
            filename_prefix = _scene_prefix(job.job_id, "drafts", scene.scene_id)
            combined = graph.add(
                "VHS_VideoCombine",
                f"Save {I2V_BASE_WIDTH}x{I2V_BASE_HEIGHT} first-pass QC draft",
                images=graph.output(draft_decoded_video),
                audio=graph.output(draft_decoded_audio),
                frame_rate=float(PRODUCTION_FPS),
                loop_count=0,
                filename_prefix=filename_prefix,
                format="video/h264-mp4",
                pingpong=False,
                save_output=False,
                pix_fmt="yuv420p",
                crf=23,
                save_metadata=True,
                trim_to_audio=False,
            )
            return WorkflowBuild(
                api=graph.nodes,
                output_node_id=combined,
                filename_prefix=filename_prefix,
            )
    upscaler = graph.add(
        "LatentUpscaleModelLoader",
        "LTX spatial x2 upscaler",
        model_name=upscaler_settings["model"],
    )
    upscaled_video = graph.add(
        "LTXVLatentUpsamplerTiled",
        "Tiled spatial upscale",
        samples=draft_video_latent,
        upscale_model=graph.output(upscaler),
        vae=graph.output(checkpoint, 2),
        tile_size=upscaler_settings["tile_size"],
        overlap=upscaler_settings["overlap"],
        max_size_for_no_tile=upscaler_settings["max_size_without_tiling"],
        rotate_for_landscape=False,
        debug=False,
    )

    full_image = graph.add(
        "ImageScale",
        "Scale exact frame to production size",
        image=graph.output(source),
        upscale_method="lanczos",
        width=PRODUCTION_WIDTH,
        height=PRODUCTION_HEIGHT,
        crop="disabled",
    )
    full_preprocessed = graph.add(
        "LTXVPreprocess",
        "Preprocess upscale-pass reference",
        image=graph.output(full_image),
        img_compression=second_pass["image_compression"],
    )
    second_i2v = graph.add(
        "LTXVImgToVideoInplaceKJ",
        "Reinject cached frame after spatial upscale",
        vae=graph.output(checkpoint, 2),
        latent=graph.output(upscaled_video),
        num_images="1",
        **{
            "num_images.strength_1": second_pass["image_strength"],
            "num_images.image_1": graph.output(full_preprocessed),
            "num_images.index_1": 0,
        },
    )
    second_av = graph.add(
        "LTXVConcatAVLatent",
        "Upscale-pass AV latent",
        video_latent=graph.output(second_i2v),
        audio_latent=draft_audio_latent,
    )
    second_model = graph.add(
        "LTXReferenceConditioning",
        "Upscale-pass reference model",
        model=graph.output(reference_enabled),
        vae=graph.output(checkpoint, 2),
        image=graph.output(full_preprocessed),
        target_latent=graph.output(second_i2v),
        strength=second_pass["reference_strength"],
        position_mode="reference",
        verbose=False,
    )
    second_sampler = graph.add(
        "KSamplerSelect",
        "I2V upscale-pass sampler",
        sampler_name=second_pass["sampler"],
    )
    second_sigmas = graph.add(
        "ManualSigmas",
        "Verified upscale-pass sigmas",
        sigmas=_sigma_string(second_pass["sigmas"]),
    )
    second_sampled = graph.add(
        "SamplerCustom",
        "Second sampling pass",
        model=graph.output(second_model),
        add_noise=True,
        noise_seed=scene.i2v.seed,
        cfg=second_pass["cfg"],
        positive=graph.output(conditioning, 0),
        negative=graph.output(conditioning, 1),
        sampler=graph.output(second_sampler),
        sigmas=graph.output(second_sigmas),
        latent_image=graph.output(second_av),
    )
    final_split = graph.add(
        "LTXVSeparateAVLatent",
        "Split final AV latent",
        av_latent=graph.output(second_sampled, 1),
    )
    decoded_video = graph.add(
        "VAEDecode",
        "Decode spatial-upscale video",
        samples=graph.output(final_split, 0),
        vae=graph.output(checkpoint, 2),
    )
    decoded_audio = graph.add(
        "LTXVAudioVAEDecode",
        "Decode generated audio",
        samples=graph.output(final_split, 1),
        audio_vae=graph.output(audio_vae),
    )
    filename_prefix = _scene_prefix(job.job_id, "clips", scene.scene_id)
    combined = graph.add(
        "VHS_VideoCombine",
        f"Save {PRODUCTION_WIDTH}x{PRODUCTION_HEIGHT} 24 fps scene clip",
        images=graph.output(decoded_video),
        audio=graph.output(decoded_audio),
        frame_rate=float(PRODUCTION_FPS),
        loop_count=0,
        filename_prefix=filename_prefix,
        format="video/h264-mp4",
        pingpong=False,
        save_output=False,
        pix_fmt="yuv420p",
        crf=19,
        save_metadata=True,
        trim_to_audio=False,
    )
    _add_discord_video_delivery(
        graph,
        graph.output(decoded_video),
        graph.output(decoded_audio),
        job,
        scene,
        delivery,
    )
    return WorkflowBuild(api=graph.nodes, output_node_id=combined, filename_prefix=filename_prefix)


def build_i2v_draft_api_workflow(
    job: JobPayload,
    scene: SceneSpec,
    cached_frame_path: str | Path,
    resolved_lora_filenames: Mapping[str, str] | None = None,
    *,
    overrides: SceneWorkflowOverrides | None = None,
    revision: int = 1,
    attempt_number: int = 1,
) -> WorkflowBuild:
    """Build only the cheap first LTX pass and durable handoff checkpoints."""
    return build_i2v_api_workflow(
        job,
        scene,
        cached_frame_path,
        resolved_lora_filenames,
        delivery=None,
        overrides=overrides,
        artifact_stage=QcArtifactStage.DRAFT,
        revision=revision,
        attempt_number=attempt_number,
    )


def build_i2v_final_api_workflow(
    job: JobPayload,
    scene: SceneSpec,
    cached_frame_path: str | Path,
    resolved_lora_filenames: Mapping[str, str] | None = None,
    *,
    overrides: SceneWorkflowOverrides | None = None,
    revision: int = 1,
    attempt_number: int = 1,
) -> WorkflowBuild:
    """Build the expensive pass from the exact approved draft checkpoints."""
    return build_i2v_api_workflow(
        job,
        scene,
        cached_frame_path,
        resolved_lora_filenames,
        delivery=None,
        overrides=overrides,
        artifact_stage=QcArtifactStage.FINAL,
        revision=revision,
        attempt_number=attempt_number,
    )


def validate_api_graph(workflow: Mapping[str, Mapping[str, Any]]) -> tuple[str, ...]:
    """Return structural graph errors without loading models or running ComfyUI."""
    errors: list[str] = []
    node_ids = set(workflow)
    if not node_ids:
        return ("workflow has no nodes",)
    for node_id, node in workflow.items():
        if not isinstance(node.get("class_type"), str) or not node["class_type"]:
            errors.append(f"node {node_id} has no class_type")
        inputs = node.get("inputs")
        if not isinstance(inputs, Mapping):
            errors.append(f"node {node_id} has no input mapping")
            continue
        for input_name, value in inputs.items():
            if (
                isinstance(value, list)
                and len(value) == 2
                and isinstance(value[0], str)
                and isinstance(value[1], int)
                and value[0] not in node_ids
            ):
                errors.append(f"node {node_id}.{input_name} references missing node {value[0]}")
    return tuple(errors)


def _literal_matches_scalar_type(value: Any, expected_type: str) -> bool | None:
    """Return scalar validity, or None when the type is routed/custom."""
    if expected_type == "INT":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "FLOAT":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "BOOLEAN":
        return isinstance(value, bool)
    if expected_type == "STRING":
        return isinstance(value, str)
    return None


def validate_against_object_info(
    workflow: Mapping[str, Mapping[str, Any]],
    object_info: Mapping[str, Mapping[str, Any]],
) -> tuple[str, ...]:
    """Validate classes, inputs, combo literals, output slots, and routed types."""
    errors = list(validate_api_graph(workflow))
    for node_id, node in workflow.items():
        class_type = node.get("class_type")
        info = object_info.get(class_type) if isinstance(class_type, str) else None
        if not info:
            errors.append(f"node {node_id} uses unavailable class {class_type}")
            continue
        input_schema = info.get("input", {})
        required = dict(input_schema.get("required", {}))
        optional = input_schema.get("optional", {})
        inputs = node.get("inputs", {})
        format_spec = required.get("format")
        selected_format = inputs.get("format")
        if (
            isinstance(format_spec, (list, tuple))
            and len(format_spec) > 1
            and isinstance(format_spec[1], Mapping)
            and isinstance(selected_format, str)
        ):
            formats = format_spec[1].get("formats", {})
            dynamic_fields = (
                formats.get(selected_format, [])
                if isinstance(formats, Mapping)
                else []
            )
            for field in dynamic_fields:
                if (
                    isinstance(field, (list, tuple))
                    and len(field) >= 2
                    and isinstance(field[0], str)
                ):
                    required[field[0]] = [
                        field[1],
                        field[2] if len(field) > 2 else {},
                    ]
        for name in required:
            if name not in inputs:
                errors.append(f"node {node_id} ({class_type}) is missing required input {name}")

        for input_name, value in inputs.items():
            expected_spec = required.get(input_name) or optional.get(input_name)
            if expected_spec is None and class_type == "LTXVImgToVideoInplaceKJ":
                if input_name.endswith(".image_1"):
                    expected_spec = ["IMAGE"]
                elif input_name.endswith(".strength_1"):
                    expected_spec = ["FLOAT"]
                elif input_name.endswith(".index_1"):
                    expected_spec = ["INT"]
            if expected_spec is None:
                errors.append(
                    f"node {node_id} ({class_type}) has unknown input {input_name}"
                )
                continue

            is_route = (
                isinstance(value, list)
                and len(value) == 2
                and isinstance(value[0], str)
                and isinstance(value[1], int)
                and value[0] in workflow
            )
            expected_type = expected_spec[0]
            if not is_route:
                combo_options = None
                if isinstance(expected_type, (list, tuple)):
                    combo_options = expected_type
                elif (
                    expected_type == "COMBO"
                    and len(expected_spec) > 1
                    and isinstance(expected_spec[1], Mapping)
                    and isinstance(expected_spec[1].get("options"), (list, tuple))
                ):
                    combo_options = expected_spec[1]["options"]
                if combo_options is not None and value not in combo_options:
                    errors.append(
                        f"node {node_id}.{input_name} has invalid combo value "
                        f"{value!r}"
                    )
                if isinstance(expected_type, str):
                    scalar_valid = _literal_matches_scalar_type(
                        value,
                        expected_type,
                    )
                    if scalar_valid is False:
                        errors.append(
                            f"node {node_id}.{input_name} expects a literal "
                            f"{expected_type}, got {type(value).__name__}"
                        )
                    elif (
                        scalar_valid is True
                        and expected_type in {"INT", "FLOAT"}
                        and len(expected_spec) > 1
                        and isinstance(expected_spec[1], Mapping)
                    ):
                        if (
                            expected_type == "FLOAT"
                            and isinstance(value, float)
                            and not math.isfinite(value)
                        ):
                            errors.append(
                                f"node {node_id}.{input_name} expects a finite "
                                f"FLOAT, got {value!r}"
                            )
                            continue
                        minimum = expected_spec[1].get("min")
                        maximum = expected_spec[1].get("max")
                        if isinstance(minimum, (int, float)) and value < minimum:
                            errors.append(
                                f"node {node_id}.{input_name} value {value!r} "
                                f"is below minimum {minimum!r}"
                            )
                        if isinstance(maximum, (int, float)) and value > maximum:
                            errors.append(
                                f"node {node_id}.{input_name} value {value!r} "
                                f"is above maximum {maximum!r}"
                            )
                continue
            source = workflow[value[0]]
            source_info = object_info.get(source.get("class_type"), {})
            outputs = source_info.get("output", [])
            if value[1] < 0 or value[1] >= len(outputs):
                errors.append(
                    f"node {node_id}.{input_name} uses invalid output slot {value[1]} "
                    f"from node {value[0]}"
                )
                continue
            actual_type = outputs[value[1]]
            if isinstance(expected_type, str) and expected_type != actual_type:
                errors.append(
                    f"node {node_id}.{input_name} expects {expected_type}, "
                    f"but node {value[0]} slot {value[1]} outputs {actual_type}"
                )
    return tuple(errors)
