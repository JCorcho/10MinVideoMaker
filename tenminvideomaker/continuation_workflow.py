"""No-delivery ComfyUI worker graphs for LTX 2.3 temporal continuation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .constants import (
    I2V_BASE_HEIGHT,
    I2V_BASE_WIDTH,
    I2V_FIRST_PASS_SIGMAS,
    I2V_SAMPLER,
    I2V_SPATIAL_UPSCALER,
    I2V_UPSCALE_PASS_SIGMAS,
    PRODUCTION_FPS,
    PRODUCTION_HEIGHT,
    PRODUCTION_WIDTH,
)
from .continuation import (
    CAUSAL_REFINEMENT_PREROLL_FRAMES,
    ContinuationChunkPlan,
    SceneFramePlan,
    assembly_spans,
    handoff_latent_token_count,
)
from .contracts import JobPayload, SceneSpec
from .review import SceneWorkflowOverrides
from .workflow_builder import (
    LTX_CHECKPOINT,
    LTX_TEXT_ENCODER,
    WorkflowBuild,
    WorkflowBuildError,
    _Graph,
    _apply_i2v_loras,
    _sigma_string,
)


@dataclass(frozen=True)
class ContinuationStage2Build:
    """A refined raw chunk plus its durable final-resolution latent outputs."""

    workflow: WorkflowBuild
    video_checkpoint_node_id: str
    audio_checkpoint_node_id: str


def _validated_chunk(
    plan: SceneFramePlan,
    chunk: ContinuationChunkPlan,
) -> None:
    if (
        chunk.index < 0
        or chunk.index >= len(plan.chunks)
        or plan.chunks[chunk.index] != chunk
    ):
        raise WorkflowBuildError("Continuation chunk does not belong to its scene plan.")


def _pass_settings(
    overrides: SceneWorkflowOverrides | None,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    first = (
        overrides.i2v_first_pass
        if overrides is not None
        else {
            "sampler": I2V_SAMPLER,
            "sigmas": I2V_FIRST_PASS_SIGMAS,
            "cfg": 1.0,
            "reference_strength": 0.75,
            "image_strength": 0.75,
            "image_compression": 35,
        }
    )
    second = (
        overrides.i2v_second_pass
        if overrides is not None
        else {
            "sampler": I2V_SAMPLER,
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
    upscaler = (
        overrides.spatial_upscaler
        if overrides is not None
        else {
            "model": I2V_SPATIAL_UPSCALER,
            "tile_size": 11,
            "overlap": 6,
            "max_size_without_tiling": 22,
        }
    )
    return first, second, chunking, upscaler


def _model_stack(
    graph: _Graph,
    job: JobPayload,
    scene: SceneSpec,
    resolved_lora_filenames: Mapping[str, str] | None,
    chunking: Mapping[str, Any],
    *,
    needs_audio_vae: bool,
) -> tuple[str, str, str | None, list[Any]]:
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
    audio_vae = (
        graph.add(
            "LTXVAudioVAELoader",
            "LTX audio VAE",
            ckpt_name=LTX_CHECKPOINT,
        )
        if needs_audio_vae
        else None
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
    return checkpoint, text_encoder, audio_vae, graph.output(chunked)


def _conditioning(
    graph: _Graph,
    text_encoder: str,
    chunk: ContinuationChunkPlan,
) -> tuple[list[Any], list[Any]]:
    positive = graph.add(
        "CLIPTextEncode",
        f"Chunk {chunk.index + 1} motion prompt",
        text=chunk.prompt,
        clip=graph.output(text_encoder),
    )
    negative = graph.add(
        "CLIPTextEncode",
        f"Chunk {chunk.index + 1} negative prompt",
        text=chunk.negative,
        clip=graph.output(text_encoder),
    )
    conditioned = graph.add(
        "LTXVConditioning",
        "24 fps chunk conditioning",
        positive=graph.output(positive),
        negative=graph.output(negative),
        frame_rate=float(PRODUCTION_FPS),
    )
    return graph.output(conditioned, 0), graph.output(conditioned, 1)


def _pass_through_stg_guider(
    graph: _Graph,
    *,
    model: list[Any],
    positive: list[Any],
    negative: list[Any],
    cfg: float,
    sigmas: list[float] | tuple[float, ...],
) -> list[Any]:
    """Build the installed ExtendSampler's required zero-STG distilled guider."""
    sigma_values = tuple(float(value) for value in sigmas)
    if not sigma_values:
        raise WorkflowBuildError("Continuation guider requires at least one sigma.")
    repeated_cfg = ", ".join(str(float(cfg)) for _ in sigma_values)
    repeated_zero = ", ".join("0.0" for _ in sigma_values)
    repeated_one = ", ".join("1.0" for _ in sigma_values)
    repeated_layers = ", ".join("[29]" for _ in sigma_values)
    guider = graph.add(
        "STGGuiderAdvanced",
        "LCM pass-through guider (STG disabled)",
        model=model,
        positive=positive,
        negative=negative,
        skip_steps_sigma_threshold=100.0,
        cfg_star_rescale=False,
        sigmas=_sigma_string(sigma_values),
        cfg_values=repeated_cfg,
        stg_scale_values=repeated_zero,
        stg_rescale_values=repeated_one,
        stg_layers_indices=repeated_layers,
        apply_apg=False,
        apg_cfg_scale=1.0,
        eta=1.0,
        norm_threshold=0.0,
    )
    return graph.output(guider)


def _source_image(
    graph: _Graph,
    cached_frame_path: Path,
    *,
    width: int,
    height: int,
    compression: int,
    title_suffix: str,
) -> tuple[list[Any], list[Any]]:
    source = graph.add(
        "VHS_LoadImagePath",
        "Load exact cached T2I frame",
        image=str(cached_frame_path),
        custom_width=0,
        custom_height=0,
    )
    scaled = graph.add(
        "ImageScale",
        f"Scale frame for {title_suffix}",
        image=graph.output(source),
        upscale_method="lanczos",
        width=width,
        height=height,
        crop="disabled",
    )
    preprocessed = graph.add(
        "LTXVPreprocess",
        f"Preprocess {title_suffix} reference",
        image=graph.output(scaled),
        img_compression=compression,
    )
    return graph.output(scaled), graph.output(preprocessed)


def build_continuation_stage1_workflow(
    job: JobPayload,
    scene: SceneSpec,
    cached_frame_path: str | Path,
    plan: SceneFramePlan,
    chunk: ContinuationChunkPlan,
    *,
    revision: int,
    attempt_number: int,
    previous_attempt_number: int | None = None,
    resolved_lora_filenames: Mapping[str, str] | None = None,
    overrides: SceneWorkflowOverrides | None = None,
) -> WorkflowBuild:
    """Build one bounded low-resolution video-only continuation prompt."""
    _validated_chunk(plan, chunk)
    frame_path = Path(cached_frame_path)
    if not frame_path.is_absolute():
        raise WorkflowBuildError("cached_frame_path must be absolute.")
    if chunk.is_initial and previous_attempt_number is not None:
        raise WorkflowBuildError("Initial continuation chunk cannot have an upstream attempt.")
    if not chunk.is_initial and (
        previous_attempt_number is None or previous_attempt_number < 1
    ):
        raise WorkflowBuildError("Continuation chunk requires its accepted upstream attempt.")

    first, _second, chunking, _upscaler = _pass_settings(overrides)
    graph = _Graph()
    checkpoint, text_encoder, _audio_vae, model = _model_stack(
        graph,
        job,
        scene,
        resolved_lora_filenames,
        chunking,
        needs_audio_vae=False,
    )
    positive, negative = _conditioning(graph, text_encoder, chunk)
    sampler = graph.add(
        "KSamplerSelect",
        "First-pass LCM sampler",
        sampler_name=first["sampler"],
    )
    sigmas = graph.add(
        "ManualSigmas",
        "First-pass continuation sigmas",
        sigmas=_sigma_string(first["sigmas"]),
    )
    noise = graph.add(
        "RandomNoise",
        "Deterministic chunk noise",
        noise_seed=chunk.seed,
    )

    if chunk.is_initial:
        _scaled, preprocessed = _source_image(
            graph,
            frame_path,
            width=I2V_BASE_WIDTH,
            height=I2V_BASE_HEIGHT,
            compression=first["image_compression"],
            title_suffix="first continuation pass",
        )
        empty = graph.add(
            "EmptyLTXVLatentVideo",
            "121-frame-or-shorter first window",
            width=I2V_BASE_WIDTH,
            height=I2V_BASE_HEIGHT,
            length=chunk.model_window_frames,
            batch_size=1,
        )
        injected = graph.add(
            "LTXVImgToVideoInplaceKJ",
            "Inject cached scene frame",
            vae=graph.output(checkpoint, 2),
            latent=graph.output(empty),
            num_images="1",
            **{
                "num_images.strength_1": first["image_strength"],
                "num_images.image_1": preprocessed,
                "num_images.index_1": 0,
            },
        )
        reference_enabled = graph.add(
            "LTXReferenceEnable",
            "Enable initial-frame reference conditioning",
            model=model,
            zero_ref_timesteps=False,
            verbose=False,
        )
        sampling_model = graph.add(
            "LTXReferenceConditioning",
            "Initial-window reference model",
            model=graph.output(reference_enabled),
            vae=graph.output(checkpoint, 2),
            image=preprocessed,
            target_latent=graph.output(injected),
            strength=first["reference_strength"],
            position_mode="reference",
            verbose=False,
        )
        sampled = graph.add(
            "SamplerCustom",
            "Initial validated video-only LCM pass",
            model=graph.output(sampling_model),
            add_noise=True,
            noise_seed=chunk.seed,
            cfg=float(first["cfg"]),
            positive=positive,
            negative=negative,
            sampler=graph.output(sampler),
            sigmas=graph.output(sigmas),
            latent_image=graph.output(injected),
        )
        extended_result = graph.output(sampled, 1)
    else:
        previous = graph.add(
            "10MinVideoMaker_LoadChunkLatent",
            "Load accepted bounded first-pass handoff",
            job_id=job.job_id,
            scene_id=scene.scene_id,
            revision=revision,
            chunk_index=chunk.index - 1,
            attempt_number=previous_attempt_number,
            artifact_kind="stage1_handoff",
            expected_temporal_tokens=handoff_latent_token_count(
                plan.chunks[chunk.index - 1]
            ),
        )
        guider = _pass_through_stg_guider(
            graph,
            model=model,
            positive=positive,
            negative=negative,
            cfg=float(first["cfg"]),
            sigmas=first["sigmas"],
        )
        extended = graph.add(
            "LTXVExtendSampler",
            "Official latent-overlap continuation",
            model=model,
            vae=graph.output(checkpoint, 2),
            latents=graph.output(previous, 0),
            num_new_frames=chunk.new_transition_frames,
            frame_overlap=plan.overlap_pixel_frames,
            guider=guider,
            sampler=graph.output(sampler),
            sigmas=graph.output(sigmas),
            noise=graph.output(noise),
            strength=0.5,
        )
        extended_result = graph.output(extended, 0)

    # Keep one predecessor token for later full-resolution causal preroll, but
    # never persist the complete cumulative scene latent. The next extension
    # deliberately consumes the whole bounded handoff: the official node uses
    # its tail as overlap context, while the predecessor preserves causal token
    # interpretation during the later full-resolution decode.
    handoff_latent_count = handoff_latent_token_count(chunk)
    handoff = graph.add(
        "LTXVSelectLatents",
        "Persist bounded current window plus causal predecessor",
        samples=extended_result,
        start_index=-handoff_latent_count,
        end_index=-1,
    )
    saved = graph.add(
        "10MinVideoMaker_SaveChunkLatent",
        "Durably checkpoint bounded first-pass handoff",
        latent=graph.output(handoff),
        job_id=job.job_id,
        scene_id=scene.scene_id,
        revision=revision,
        chunk_index=chunk.index,
        attempt_number=attempt_number,
        artifact_kind="stage1_handoff",
    )
    return WorkflowBuild(
        api=graph.nodes,
        output_node_id=saved,
        filename_prefix=(
            f"10MinVideoMaker/{job.job_id}/continuation/"
            f"scene_{scene.scene_id:04d}/chunk_{chunk.index:04d}/stage1"
        ),
    )


def build_continuation_stage2_workflow(
    job: JobPayload,
    scene: SceneSpec,
    cached_frame_path: str | Path,
    plan: SceneFramePlan,
    chunk: ContinuationChunkPlan,
    *,
    revision: int,
    attempt_number: int,
    previous_attempt_number: int | None = None,
    previous_chunk_path: str | Path | None = None,
    resolved_lora_filenames: Mapping[str, str] | None = None,
    overrides: SceneWorkflowOverrides | None = None,
) -> ContinuationStage2Build:
    """Build one raw full-resolution AV refinement window without delivery."""
    _validated_chunk(plan, chunk)
    frame_path = Path(cached_frame_path)
    if not frame_path.is_absolute():
        raise WorkflowBuildError("cached_frame_path must be absolute.")
    if not chunk.is_initial and (
        previous_attempt_number is None or previous_attempt_number < 1
    ):
        raise WorkflowBuildError("Refined continuation requires the prior accepted attempt.")
    prior_path = Path(previous_chunk_path) if previous_chunk_path is not None else None
    if not chunk.is_initial and (prior_path is None or not prior_path.is_absolute()):
        raise WorkflowBuildError(
            "Refined continuation requires an absolute prior raw chunk path."
        )

    _first, second, chunking, upscaler_settings = _pass_settings(overrides)
    graph = _Graph()
    checkpoint, text_encoder, audio_vae, model = _model_stack(
        graph,
        job,
        scene,
        resolved_lora_filenames,
        chunking,
        needs_audio_vae=True,
    )
    assert audio_vae is not None
    positive, negative = _conditioning(graph, text_encoder, chunk)
    handoff = graph.add(
        "10MinVideoMaker_LoadChunkLatent",
        "Load accepted bounded first-pass handoff",
        job_id=job.job_id,
        scene_id=scene.scene_id,
        revision=revision,
        chunk_index=chunk.index,
        attempt_number=attempt_number,
        artifact_kind="stage1_handoff",
        expected_temporal_tokens=handoff_latent_token_count(chunk),
    )
    # A nonzero LTX latent token represents eight causal pixel frames. If it is
    # sliced into position zero directly, the decoder reinterprets that token
    # as LTX's special one-frame latent and corrupts the boundary. Include one
    # extra latent as an eight-frame sacrificial preroll; assembly removes it.
    causal_preroll_frames = (
        0 if chunk.is_initial else CAUSAL_REFINEMENT_PREROLL_FRAMES
    )
    upscaler = graph.add(
        "LatentUpscaleModelLoader",
        "LTX spatial x2 upscaler",
        model_name=upscaler_settings["model"],
    )
    upscaled = graph.add(
        "LTXVLatentUpsamplerTiled",
        "Tiled spatial refinement input",
        samples=graph.output(handoff, 0),
        upscale_model=graph.output(upscaler),
        vae=graph.output(checkpoint, 2),
        tile_size=upscaler_settings["tile_size"],
        overlap=upscaler_settings["overlap"],
        max_size_for_no_tile=upscaler_settings["max_size_without_tiling"],
        rotate_for_landscape=False,
        debug=False,
    )

    if chunk.is_initial:
        _scaled, preprocessed = _source_image(
            graph,
            frame_path,
            width=PRODUCTION_WIDTH,
            height=PRODUCTION_HEIGHT,
            compression=second["image_compression"],
            title_suffix="full-resolution refinement",
        )
        guided_video = graph.add(
            "LTXVImgToVideoInplaceKJ",
            "Reinject cached frame into first refined window",
            vae=graph.output(checkpoint, 2),
            latent=graph.output(upscaled),
            num_images="1",
            **{
                "num_images.strength_1": second["image_strength"],
                "num_images.image_1": preprocessed,
                "num_images.index_1": 0,
            },
        )
        reference_enabled = graph.add(
            "LTXReferenceEnable",
            "Enable initial refined reference conditioning",
            model=model,
            zero_ref_timesteps=False,
            verbose=False,
        )
        sampling_model = graph.add(
            "LTXReferenceConditioning",
            "Initial refined reference model",
            model=graph.output(reference_enabled),
            vae=graph.output(checkpoint, 2),
            image=preprocessed,
            target_latent=graph.output(guided_video),
            strength=second["reference_strength"],
            position_mode="reference",
            verbose=False,
        )
        video_latent = graph.output(guided_video)
        sampling_positive = positive
        sampling_negative = negative
        final_model = graph.output(sampling_model)
    else:
        previous_chunk = plan.chunks[chunk.index - 1]
        previous_preroll = (
            0
            if previous_chunk.is_initial
            else CAUSAL_REFINEMENT_PREROLL_FRAMES
        )
        previous_commit_boundary = assembly_spans(plan)[
            previous_chunk.index
        ].input_end_frame_exclusive
        # Delayed commit leaves the prior window's final 25 visible samples
        # provisional. They are the exact global overlap owned by this later
        # window. Account for the prior raw window's sacrificial causal
        # preroll, then guide the current upscaled latent at visible frame eight.
        guide_skip_frames = previous_preroll + previous_commit_boundary
        previous_video = graph.add(
            "VHS_LoadVideoPath",
            "Load prior 25-frame final-resolution overlap",
            video=str(prior_path),
            force_rate=float(PRODUCTION_FPS),
            custom_width=0,
            custom_height=0,
            frame_load_cap=25,
            skip_first_frames=guide_skip_frames,
            select_every_nth=1,
            format="None",
        )
        guided_video = graph.add(
            "LTXVAddGuide",
            "Guide refinement with prior 25-frame visible overlap",
            positive=positive,
            negative=negative,
            vae=graph.output(checkpoint, 2),
            latent=graph.output(upscaled),
            image=graph.output(previous_video, 0),
            # Raw frames 0..7 are the sacrificial causal-token preroll.
            # Align prior visible overlap to the first frame assembly keeps.
            frame_idx=CAUSAL_REFINEMENT_PREROLL_FRAMES,
            strength=1.0,
        )
        sampling_positive = graph.output(guided_video, 0)
        sampling_negative = graph.output(guided_video, 1)
        video_latent = graph.output(guided_video, 2)
        final_model = model

    empty_audio = graph.add(
        "LTXVEmptyLatentAudio",
        "Matching chunk audio latent",
        frames_number=chunk.model_window_frames + causal_preroll_frames,
        frame_rate=PRODUCTION_FPS,
        batch_size=1,
        audio_vae=graph.output(audio_vae),
    )
    av_latent = graph.add(
        "LTXVConcatAVLatent",
        "Chunk refinement AV latent",
        video_latent=video_latent,
        audio_latent=graph.output(empty_audio),
    )
    sampler = graph.add(
        "KSamplerSelect",
        "Second-pass LCM sampler",
        sampler_name=second["sampler"],
    )
    sigmas = graph.add(
        "ManualSigmas",
        "Second-pass refinement sigmas",
        sigmas=_sigma_string(second["sigmas"]),
    )
    sampled = graph.add(
        "SamplerCustom",
        "Refine one bounded AV continuation window",
        model=final_model,
        add_noise=True,
        noise_seed=chunk.seed,
        cfg=second["cfg"],
        positive=sampling_positive,
        negative=sampling_negative,
        sampler=graph.output(sampler),
        sigmas=graph.output(sigmas),
        latent_image=graph.output(av_latent),
    )
    split = graph.add(
        "LTXVSeparateAVLatent",
        "Split refined chunk AV latent",
        av_latent=graph.output(sampled, 1),
    )
    saved_video = graph.add(
        "10MinVideoMaker_SaveChunkLatent",
        "Checkpoint refined video latent",
        latent=graph.output(split, 0),
        job_id=job.job_id,
        scene_id=scene.scene_id,
        revision=revision,
        chunk_index=chunk.index,
        attempt_number=attempt_number,
        artifact_kind="stage2_video",
    )
    saved_audio = graph.add(
        "10MinVideoMaker_SaveChunkLatent",
        "Checkpoint refined audio latent",
        latent=graph.output(split, 1),
        job_id=job.job_id,
        scene_id=scene.scene_id,
        revision=revision,
        chunk_index=chunk.index,
        attempt_number=attempt_number,
        artifact_kind="stage2_audio",
    )
    decoded_video = graph.add(
        "LTXVSpatioTemporalTiledVAEDecode",
        "16 GB tiled decode of refined chunk",
        vae=graph.output(checkpoint, 2),
        latents=graph.output(saved_video, 0),
        spatial_tiles=4,
        spatial_overlap=1,
        temporal_tile_length=16,
        temporal_overlap=1,
        last_frame_fix=False,
        working_device="auto",
        working_dtype="auto",
    )
    decoded_audio = graph.add(
        "LTXVAudioVAEDecode",
        "Decode raw chunk audio",
        samples=graph.output(saved_audio, 0),
        audio_vae=graph.output(audio_vae),
    )
    filename_prefix = (
        f"10MinVideoMaker/{job.job_id}/continuation/"
        f"scene_{scene.scene_id:04d}/chunk_{chunk.index:04d}/attempt_{attempt_number:04d}"
    )
    combined = graph.add(
        "VHS_VideoCombine",
        "Save lossless raw unwatermarked continuation chunk",
        images=graph.output(decoded_video),
        audio=graph.output(decoded_audio),
        frame_rate=float(PRODUCTION_FPS),
        loop_count=0,
        filename_prefix=filename_prefix,
        format="video/ffv1-mkv",
        pingpong=False,
        save_output=False,
        level="3",
        coder="1",
        context="1",
        gop_size=1,
        slices="16",
        slicecrc="1",
        pix_fmt="yuv444p",
        save_metadata=False,
        trim_to_audio=False,
    )
    return ContinuationStage2Build(
        workflow=WorkflowBuild(
            api=graph.nodes,
            output_node_id=combined,
            filename_prefix=filename_prefix,
        ),
        video_checkpoint_node_id=saved_video,
        audio_checkpoint_node_id=saved_audio,
    )


def build_continuation_decode_workflow(
    job: JobPayload,
    scene: SceneSpec,
    plan: SceneFramePlan,
    chunk: ContinuationChunkPlan,
    *,
    revision: int,
    attempt_number: int,
) -> WorkflowBuild:
    """Decode/mux verified stage-two AV checkpoints without diffusion."""
    _validated_chunk(plan, chunk)
    graph = _Graph()
    checkpoint = graph.add(
        "CheckpointLoaderSimple",
        "LTX 2.3 checkpoint for checkpoint-only decode",
        ckpt_name=LTX_CHECKPOINT,
    )
    audio_vae = graph.add(
        "LTXVAudioVAELoader",
        "LTX audio VAE for checkpoint-only decode",
        ckpt_name=LTX_CHECKPOINT,
    )
    video_latent = graph.add(
        "10MinVideoMaker_LoadChunkLatent",
        "Load verified refined video latent",
        job_id=job.job_id,
        scene_id=scene.scene_id,
        revision=revision,
        chunk_index=chunk.index,
        attempt_number=attempt_number,
        artifact_kind="stage2_video",
        expected_temporal_tokens=handoff_latent_token_count(chunk),
    )
    audio_latent = graph.add(
        "10MinVideoMaker_LoadChunkLatent",
        "Load verified refined audio latent",
        job_id=job.job_id,
        scene_id=scene.scene_id,
        revision=revision,
        chunk_index=chunk.index,
        attempt_number=attempt_number,
        artifact_kind="stage2_audio",
        # Audio token geometry differs from video. The artifact-kind validator
        # checks its tensor descriptors/hash and deliberately ignores this
        # video-only temporal field.
        expected_temporal_tokens=1,
    )
    decoded_video = graph.add(
        "LTXVSpatioTemporalTiledVAEDecode",
        "Decode verified refined video checkpoint",
        vae=graph.output(checkpoint, 2),
        latents=graph.output(video_latent, 0),
        spatial_tiles=4,
        spatial_overlap=1,
        temporal_tile_length=16,
        temporal_overlap=1,
        last_frame_fix=False,
        working_device="auto",
        working_dtype="auto",
    )
    decoded_audio = graph.add(
        "LTXVAudioVAEDecode",
        "Decode verified refined audio checkpoint",
        samples=graph.output(audio_latent, 0),
        audio_vae=graph.output(audio_vae),
    )
    filename_prefix = (
        f"10MinVideoMaker/{job.job_id}/continuation/"
        f"scene_{scene.scene_id:04d}/chunk_{chunk.index:04d}/"
        f"attempt_{attempt_number:04d}_decode"
    )
    combined = graph.add(
        "VHS_VideoCombine",
        "Recover lossless raw unwatermarked continuation chunk",
        images=graph.output(decoded_video),
        audio=graph.output(decoded_audio),
        frame_rate=float(PRODUCTION_FPS),
        loop_count=0,
        filename_prefix=filename_prefix,
        format="video/ffv1-mkv",
        pingpong=False,
        save_output=False,
        level="3",
        coder="1",
        context="1",
        gop_size=1,
        slices="16",
        slicecrc="1",
        pix_fmt="yuv444p",
        save_metadata=False,
        trim_to_audio=False,
    )
    return WorkflowBuild(
        api=graph.nodes,
        output_node_id=combined,
        filename_prefix=filename_prefix,
    )


def build_assembled_scene_delivery_workflow(
    job: JobPayload,
    scene: SceneSpec,
    raw_scene_path: str | Path,
    webhook_url: str,
) -> WorkflowBuild:
    """Load an assembled raw scene and send only a watermarked Discord copy."""
    path = Path(raw_scene_path)
    if not path.is_absolute():
        raise WorkflowBuildError("raw_scene_path must be absolute.")
    if not webhook_url:
        raise WorkflowBuildError("Discord delivery requires a webhook URL.")
    graph = _Graph()
    loaded = graph.add(
        "VHS_LoadVideoPath",
        "Load assembled raw scene",
        video=str(path),
        force_rate=float(PRODUCTION_FPS),
        custom_width=0,
        custom_height=0,
        frame_load_cap=0,
        skip_first_frames=0,
        select_every_nth=1,
        format="None",
    )
    watermarked = graph.add(
        "DaSiWa_Watermark",
        "Watermark Discord-only scene copy",
        images=graph.output(loaded, 0),
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
    sender = graph.add(
        "DiscordSendSaveVideo",
        "Send metadata-free watermarked assembled scene",
        images=graph.output(watermarked),
        audio=graph.output(loaded, 2),
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
        webhook_url=webhook_url,
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
    return WorkflowBuild(
        api=graph.nodes,
        output_node_id=sender,
        filename_prefix=f"10MinVideoMaker-{job.job_id}-scene-{scene.scene_id:04d}",
    )
