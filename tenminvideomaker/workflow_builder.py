"""Build scene-specific ComfyUI API workflows from the validated job contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .assets import predictable_lora_filename
from .constants import (
    I2V_FIRST_PASS_SIGMAS,
    I2V_SAMPLER,
    I2V_SPATIAL_UPSCALER,
    I2V_UPSCALE_PASS_SIGMAS,
    MANDATORY_I2V_LORAS,
    PRODUCTION_FPS,
    PRODUCTION_HEIGHT,
    PRODUCTION_WIDTH,
)
from .contracts import (
    JobPayload,
    LoraSpec,
    SceneSpec,
    effective_t2i_loras,
    lora_identity,
    unique_loras,
)

ANIMA_UNET = "CyberRealistic_AnimaSemi_V6.0.safetensors"
ANIMA_TEXT_ENCODER = "cyberrealisticAnima_v30_txt.safetensors"
ANIMA_VAE = "qwen_image_vae_cybv2.safetensors"
PONY_CHECKPOINT = "cyberrealisticPony_v180Coreshift.safetensors"

LTX_CHECKPOINT = "10Eros_v1.4_fp8mixed_learned.safetensors"
LTX_TEXT_ENCODER = "gemma-3-12b-it-ablit-norms-biproj-fp8mixed.safetensors"
I2V_BASE_WIDTH = PRODUCTION_WIDTH // 2
I2V_BASE_HEIGHT = PRODUCTION_HEIGHT // 2


class WorkflowBuildError(ValueError):
    """Raised when a workflow cannot be built safely from the supplied scene."""


@dataclass(frozen=True)
class WorkflowBuild:
    api: dict[str, dict[str, Any]]
    output_node_id: str
    filename_prefix: str


class _Graph:
    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self._next_id = 1

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


def _dynamic_i2v_loras(job: JobPayload, scene: SceneSpec) -> tuple[LoraSpec, ...]:
    return unique_loras(
        (
            *((job.ltxv_character_lora,) if job.ltxv_character_lora else ()),
            *scene.i2v.loras,
        )
    )


def _local_lora_filename(
    lora: LoraSpec,
    resolved_filenames: Mapping[str, str] | None,
) -> str:
    if resolved_filenames:
        resolved = resolved_filenames.get(lora_identity(lora))
        if resolved:
            return resolved
    return predictable_lora_filename(lora.name)


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


def build_t2i_api_workflow(
    job: JobPayload,
    scene: SceneSpec,
    resolved_lora_filenames: Mapping[str, str] | None = None,
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
        "704x1248 production latent",
        width=PRODUCTION_WIDTH,
        height=PRODUCTION_HEIGHT,
        batch_size=1,
    )

    if family == "anima":
        sampled = graph.add(
            "KSampler",
            "Anima er_sde beta57",
            model=model,
            seed=scene.t2i.seed,
            steps=30,
            cfg=4.5,
            sampler_name="er_sde",
            scheduler="beta57",
            positive=graph.output(positive),
            negative=graph.output(negative),
            latent_image=graph.output(latent),
            denoise=1.0,
        )
    else:
        first = graph.add(
            "KSamplerAdvanced",
            "Pony pass 1: res_3m_ode",
            model=model,
            add_noise="enable",
            noise_seed=scene.t2i.seed,
            steps=30,
            cfg=6.0,
            sampler_name="res_3m_ode",
            scheduler="karras",
            positive=graph.output(positive),
            negative=graph.output(negative),
            latent_image=graph.output(latent),
            start_at_step=0,
            end_at_step=30,
            return_with_leftover_noise="disable",
        )
        sampled = graph.add(
            "KSamplerAdvanced",
            "Pony pass 2: res_5s_ode",
            model=model,
            add_noise="enable",
            noise_seed=scene.t2i.seed,
            steps=30,
            cfg=6.0,
            sampler_name="res_5s_ode",
            scheduler="karras",
            positive=graph.output(positive),
            negative=graph.output(negative),
            latent_image=graph.output(first),
            start_at_step=0,
            end_at_step=30,
            return_with_leftover_noise="disable",
        )

    decoded = graph.add(
        "VAEDecode",
        "Decode initial frame",
        samples=graph.output(sampled),
        vae=vae,
    )
    final_image = graph.output(decoded)
    if family == "pony":
        detector = graph.add(
            "UltralyticsDetectorProvider",
            "Pony face bbox detector",
            model_name="bbox/face_yolov8m.pt",
        )
        detailer = graph.add(
            "FaceDetailer",
            "Pony bbox face detailer",
            image=final_image,
            model=model,
            clip=clip,
            vae=vae,
            guide_size=512,
            guide_size_for=True,
            max_size=1024,
            seed=scene.t2i.seed,
            steps=20,
            cfg=5.0,
            sampler_name="dpmpp_2m_sde",
            scheduler="karras",
            positive=graph.output(positive),
            negative=graph.output(negative),
            denoise=0.38,
            feather=5,
            noise_mask=True,
            force_inpaint=True,
            bbox_threshold=0.5,
            bbox_dilation=10,
            bbox_crop_factor=3.0,
            sam_detection_hint="center-1",
            sam_dilation=0,
            sam_threshold=0.93,
            sam_bbox_expansion=0,
            sam_mask_hint_threshold=0.7,
            sam_mask_hint_use_negative="False",
            drop_size=120,
            bbox_detector=graph.output(detector, 0),
            wildcard="",
            cycle=1,
            inpaint_model=False,
            noise_mask_feather=20,
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
    )
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
    for lora in _dynamic_i2v_loras(job, scene):
        loader = graph.add(
            "LoraLoaderModelOnly",
            f"Scene I2V LoRA: {lora.name}",
            model=model,
            lora_name=_local_lora_filename(lora, resolved_filenames),
            strength_model=lora.weight,
        )
        model = graph.output(loader)
    return model


def build_i2v_api_workflow(
    job: JobPayload,
    scene: SceneSpec,
    cached_frame_path: str | Path,
    resolved_lora_filenames: Mapping[str, str] | None = None,
) -> WorkflowBuild:
    """Build the two-pass LCM LTX 2.3 graph for one exact cached T2I frame."""
    frame_path = Path(cached_frame_path)
    if not frame_path.is_absolute():
        raise WorkflowBuildError("cached_frame_path must be absolute.")

    graph = _Graph()
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
        chunks=2,
        dim_threshold=4096,
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
        img_compression=35,
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
            "num_images.strength_1": 0.75,
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
        strength=0.75,
        position_mode="reference",
        verbose=False,
    )
    first_sampler = graph.add("KSamplerSelect", "LCM first pass", sampler_name=I2V_SAMPLER)
    first_sigmas = graph.add(
        "ManualSigmas",
        "Verified first-pass sigmas",
        sigmas=_sigma_string(I2V_FIRST_PASS_SIGMAS),
    )
    first_sampled = graph.add(
        "SamplerCustom",
        "First LCM pass",
        model=graph.output(first_model),
        add_noise=True,
        noise_seed=scene.i2v.seed,
        cfg=1.0,
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
    upscaler = graph.add(
        "LatentUpscaleModelLoader",
        "LTX spatial x2 upscaler",
        model_name=I2V_SPATIAL_UPSCALER,
    )
    upscaled_video = graph.add(
        "LTXVLatentUpsamplerTiled",
        "Tiled spatial upscale",
        samples=graph.output(first_split, 0),
        upscale_model=graph.output(upscaler),
        vae=graph.output(checkpoint, 2),
        tile_size=11,
        overlap=6,
        max_size_for_no_tile=22,
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
        img_compression=30,
    )
    second_i2v = graph.add(
        "LTXVImgToVideoInplaceKJ",
        "Reinject cached frame after spatial upscale",
        vae=graph.output(checkpoint, 2),
        latent=graph.output(upscaled_video),
        num_images="1",
        **{
            "num_images.strength_1": 1.0,
            "num_images.image_1": graph.output(full_preprocessed),
            "num_images.index_1": 0,
        },
    )
    second_av = graph.add(
        "LTXVConcatAVLatent",
        "Upscale-pass AV latent",
        video_latent=graph.output(second_i2v),
        audio_latent=graph.output(first_split, 1),
    )
    second_model = graph.add(
        "LTXReferenceConditioning",
        "Upscale-pass reference model",
        model=graph.output(reference_enabled),
        vae=graph.output(checkpoint, 2),
        image=graph.output(full_preprocessed),
        target_latent=graph.output(second_i2v),
        strength=1.0,
        position_mode="reference",
        verbose=False,
    )
    second_sampler = graph.add("KSamplerSelect", "LCM upscale pass", sampler_name=I2V_SAMPLER)
    second_sigmas = graph.add(
        "ManualSigmas",
        "Verified upscale-pass sigmas",
        sigmas=_sigma_string(I2V_UPSCALE_PASS_SIGMAS),
    )
    second_sampled = graph.add(
        "SamplerCustom",
        "Second LCM pass",
        model=graph.output(second_model),
        add_noise=True,
        noise_seed=scene.i2v.seed,
        cfg=1.0,
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
    production_video = graph.add(
        "ImageScale",
        "Normalize decoded video to exact 704x1248",
        image=graph.output(decoded_video),
        upscale_method="lanczos",
        width=PRODUCTION_WIDTH,
        height=PRODUCTION_HEIGHT,
        crop="center",
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
        "Save 704x1248 24 fps scene clip",
        images=graph.output(production_video),
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
    return WorkflowBuild(api=graph.nodes, output_node_id=combined, filename_prefix=filename_prefix)


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


def validate_against_object_info(
    workflow: Mapping[str, Mapping[str, Any]],
    object_info: Mapping[str, Mapping[str, Any]],
) -> tuple[str, ...]:
    """Validate class names, required inputs, output slots, and routed types."""
    errors = list(validate_api_graph(workflow))
    for node_id, node in workflow.items():
        class_type = node.get("class_type")
        info = object_info.get(class_type) if isinstance(class_type, str) else None
        if not info:
            errors.append(f"node {node_id} uses unavailable class {class_type}")
            continue
        input_schema = info.get("input", {})
        required = input_schema.get("required", {})
        optional = input_schema.get("optional", {})
        inputs = node.get("inputs", {})
        for name in required:
            if name not in inputs:
                errors.append(f"node {node_id} ({class_type}) is missing required input {name}")

        for input_name, value in inputs.items():
            if not (
                isinstance(value, list)
                and len(value) == 2
                and isinstance(value[0], str)
                and isinstance(value[1], int)
                and value[0] in workflow
            ):
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
            expected_spec = required.get(input_name) or optional.get(input_name)
            if expected_spec is None and class_type == "LTXVImgToVideoInplaceKJ":
                if input_name.endswith(".image_1"):
                    expected_spec = ["IMAGE"]
                elif input_name.endswith(".strength_1"):
                    expected_spec = ["FLOAT"]
                elif input_name.endswith(".index_1"):
                    expected_spec = ["INT"]
            if expected_spec is None:
                continue
            expected_type = expected_spec[0]
            actual_type = outputs[value[1]]
            if isinstance(expected_type, str) and expected_type != actual_type:
                errors.append(
                    f"node {node_id}.{input_name} expects {expected_type}, "
                    f"but node {value[0]} slot {value[1]} outputs {actual_type}"
                )
    return tuple(errors)
