"""Crash-resumable orchestration for bounded LTX continuation windows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Callable, Mapping

from .chunk_artifacts import (
    ChunkArtifactError,
    latent_checkpoint_is_valid,
    load_latent_checkpoint,
    sha256_file,
)
from .chunk_assembly import SceneChunkAssembler, SceneChunkAssemblyError
from .comfy_http import ComfyHttpClient, ComfyHttpError, find_video_output
from .constants import (
    I2V_BASE_HEIGHT,
    I2V_BASE_WIDTH,
    I2V_FIRST_PASS_SIGMAS,
    I2V_SAMPLER,
    I2V_SPATIAL_UPSCALER,
    I2V_UPSCALE_PASS_SIGMAS,
    MANDATORY_I2V_LORAS,
    PRODUCTION_FPS,
    PRODUCTION_HEIGHT,
    PRODUCTION_WIDTH,
)
from .continuation import (
    SceneFramePlan,
    assembly_spans,
    build_scene_frame_plan,
    chunk_plan_documents,
    handoff_latent_token_count,
    refinement_raw_frame_count,
)
from .continuation_workflow import (
    build_assembled_scene_delivery_workflow,
    build_continuation_decode_workflow,
    build_continuation_stage1_workflow,
    build_continuation_stage2_workflow,
)
from .contracts import JobPayload, SceneSpec, effective_i2v_loras
from .review import SceneWorkflowOverrides
from .state_store import (
    ChunkAttemptRecord,
    ChunkState,
    PipelineStateStore,
    SceneChunkRecord,
    StateTransitionError,
)
from .storage import StorageLayout, write_json_atomic
from .workflow_builder import LTX_CHECKPOINT, LTX_TEXT_ENCODER

LOGGER = logging.getLogger("10MinVideoMaker.continuation")

CONTINUATION_ATTEMPT_SCHEMA_VERSION = 3
CONTINUATION_DELIVERY_SCHEMA_VERSION = 1

CONTINUATION_CONTRACT_NODE_TYPES = (
    "10MinVideoMaker_LoadChunkLatent",
    "10MinVideoMaker_SaveChunkLatent",
    "10MinVideoMaker_IsolateConditioning",
    "CLIPTextEncode",
    "CheckpointLoaderSimple",
    "DaSiWa_Watermark",
    "DiscordSendSaveVideo",
    "EmptyLTXVLatentVideo",
    "ImageScale",
    "KSamplerSelect",
    "LTXAVTextEncoderLoader",
    "LTXReferenceConditioning",
    "LTXReferenceEnable",
    "LTXVAddGuide",
    "LTXVAudioVAEDecode",
    "LTXVAudioVAELoader",
    "LTXVChunkFeedForward",
    "LTXVConcatAVLatent",
    "LTXVConditioning",
    "LTXVCropGuides",
    "LTXVEmptyLatentAudio",
    "LTXVExtendSampler",
    "LTXVImgToVideoInplaceKJ",
    "LTXVLatentUpsamplerTiled",
    "LTXVPreprocess",
    "LTXVSelectLatents",
    "LTXVSeparateAVLatent",
    "LTXVSpatioTemporalTiledVAEDecode",
    "LatentUpscaleModelLoader",
    "LoraLoaderModelOnly",
    "ManualSigmas",
    "RandomNoise",
    "STGGuiderAdvanced",
    "SamplerCustom",
    "VHS_LoadImagePath",
    "VHS_LoadVideoPath",
    "VHS_VideoCombine",
)

CONTINUATION_IMPLEMENTATION_PATHS = (
    "scripts/run_gui.py",
    "scripts/run_supervisor.py",
    "scripts/setup_and_start.py",
    "scripts/validate_continuation_workflows.py",
    "tenminvideomaker/assembly.py",
    "tenminvideomaker/artifacts.py",
    "tenminvideomaker/assets.py",
    "tenminvideomaker/chunk_artifacts.py",
    "tenminvideomaker/chunk_assembly.py",
    "tenminvideomaker/comfy_http.py",
    "tenminvideomaker/configuration.py",
    "tenminvideomaker/constants.py",
    "tenminvideomaker/continuation.py",
    "tenminvideomaker/continuation_renderer.py",
    "tenminvideomaker/continuation_validation.py",
    "tenminvideomaker/continuation_workflow.py",
    "tenminvideomaker/contracts.py",
    "tenminvideomaker/delivery.py",
    "tenminvideomaker/gui_app.py",
    "tenminvideomaker/gui_service.py",
    "tenminvideomaker/nodes.py",
    "tenminvideomaker/ownership.py",
    "tenminvideomaker/review.py",
    "tenminvideomaker/server_api.py",
    "tenminvideomaker/state_store.py",
    "tenminvideomaker/storage.py",
    "tenminvideomaker/supervisor.py",
    "tenminvideomaker/workflow_builder.py",
)

CONTINUATION_CACHE_IMPLEMENTATION_PATHS = (
    "tenminvideomaker/assembly.py",
    "tenminvideomaker/chunk_artifacts.py",
    "tenminvideomaker/chunk_assembly.py",
    "tenminvideomaker/constants.py",
    "tenminvideomaker/continuation.py",
    "tenminvideomaker/continuation_renderer.py",
    "tenminvideomaker/continuation_workflow.py",
    "tenminvideomaker/contracts.py",
    "tenminvideomaker/nodes.py",
    "tenminvideomaker/review.py",
    "tenminvideomaker/storage.py",
    "tenminvideomaker/workflow_builder.py",
)


class ContinuationRenderError(ComfyHttpError):
    """Raised when a resumable continuation scene cannot safely advance."""


class ContinuationDeliveryError(ContinuationRenderError):
    """Raised when optional Discord delivery fails after raw scene completion."""


@dataclass(frozen=True)
class ContinuationRenderResult:
    scene_path: Path
    plan: SceneFramePlan
    chunk_paths: tuple[Path, ...]
    reused_scene_assembly: bool


@dataclass(frozen=True)
class ContinuationDeliveryResult:
    status: str
    prompt_id: str
    reused_prompt: bool


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _document_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _structural_input_spec(value: object) -> object:
    """Drop dynamic combo contents/defaults while retaining route structure."""
    if not isinstance(value, (list, tuple)) or not value:
        return value
    input_type = value[0]
    is_combo = isinstance(input_type, (list, tuple))
    normalized_type = (
        "COMBO"
        if is_combo
        else input_type
    )
    metadata: dict[str, Any] = {}
    if len(value) > 1 and isinstance(value[1], Mapping):
        if not is_combo and "default" in value[1]:
            metadata["default"] = value[1]["default"]
        for key in (
            "min",
            "max",
            "step",
            "round",
            "multiline",
            "forceInput",
            "defaultInput",
            "lazy",
            "rawLink",
        ):
            if key in value[1]:
                metadata[key] = value[1][key]
    return [normalized_type, metadata] if metadata else [normalized_type]


def _structural_node_contract(value: Mapping[str, Any]) -> Mapping[str, Any]:
    inputs = value.get("input", {})
    normalized_inputs: dict[str, Any] = {}
    if isinstance(inputs, Mapping):
        for group in ("required", "optional", "hidden"):
            members = inputs.get(group)
            if isinstance(members, Mapping):
                normalized_inputs[group] = {
                    str(name): _structural_input_spec(spec)
                    for name, spec in sorted(members.items())
                }
    normalized: dict[str, Any] = {
        "input": normalized_inputs,
        "output": value.get("output", []),
    }
    for key in ("output_name", "output_is_list", "output_node"):
        if key in value:
            normalized[key] = value[key]
    return normalized


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContinuationRenderError(f"Continuation manifest is unreadable: {path}") from error
    if not isinstance(value, Mapping):
        raise ContinuationRenderError(f"Continuation manifest is not an object: {path}")
    return value


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.is_file():
        existing = _read_json(path)
        if _canonical_json(existing) != _canonical_json(value):
            raise ContinuationRenderError(
                f"Immutable continuation plan changed on disk: {path}"
            )
        return
    write_json_atomic(path, value)


class ContinuationRenderer:
    """Render, checkpoint, validate, and assemble one scene revision.

    One chunk attempt owns both the bounded low-resolution handoff and
    its bounded full-resolution AV refinement. A chunk is accepted only after
    its checkpoints, lossless raw MKV, and final COMPLETE.json have all been verified.
    """

    def __init__(
        self,
        *,
        store: PipelineStateStore,
        storage: StorageLayout,
        comfy: ComfyHttpClient,
        assembler: SceneChunkAssembler,
        timeout_seconds: float,
        max_attempts: int,
        webhook_url: str | None = None,
    ):
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least one.")
        self.store = store
        self.storage = storage
        self.comfy = comfy
        self.assembler = assembler
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.webhook_url = webhook_url

    def render_scene(
        self,
        job: JobPayload,
        scene: SceneSpec,
        cached_frame_path: str | Path,
        destination: str | Path,
        *,
        revision: int,
        resolved_lora_filenames: Mapping[str, str] | None = None,
        overrides: SceneWorkflowOverrides | None = None,
        deliver_to_discord: bool = True,
        prompt_id_callback: Callable[[str], None] | None = None,
    ) -> ContinuationRenderResult:
        frame_path = Path(cached_frame_path).resolve(strict=False)
        if not frame_path.is_file():
            raise ContinuationRenderError(
                f"Continuation requires the cached scene frame: {frame_path}"
            )
        output_path = Path(destination).resolve(strict=False)
        self._require_active(job, scene, revision)
        plan = self.build_plan(job, scene, revision, overrides)
        self._persist_plan(job, scene, revision, plan)
        resolved = dict(resolved_lora_filenames or {})
        runtime_contract_sha256 = self.runtime_contract_sha256()

        chunk_paths: list[Path] = []
        accepted: list[ChunkAttemptRecord] = []
        for chunk in plan.chunks:
            self._require_active(job, scene, revision)
            selected = self._accepted_attempt(
                job,
                scene,
                revision,
                plan,
                chunk.index,
                frame_path,
                resolved,
                overrides,
                runtime_contract_sha256,
            )
            if selected is None:
                selected = self._render_chunk(
                    job,
                    scene,
                    frame_path,
                    revision,
                    plan,
                    chunk.index,
                    accepted[-1] if accepted else None,
                    resolved,
                    overrides,
                    runtime_contract_sha256,
                    prompt_id_callback,
                )
            accepted.append(selected)
            chunk_paths.append(Path(selected.video_path).resolve(strict=True))
            self._require_active(job, scene, revision)

        self._require_active(job, scene, revision)
        reused = self._scene_assembly_is_valid(
            job,
            scene,
            revision,
            plan,
            accepted,
            output_path,
        )
        if not reused:
            LOGGER.info(
                "Job %s scene %s revision %s: assembling %s validated continuation chunks.",
                job.job_id,
                scene.scene_id,
                revision,
                plan.chunk_count,
            )
            try:
                self.assembler.assemble(plan, chunk_paths, output_path)
            except SceneChunkAssemblyError as error:
                raise ContinuationRenderError(str(error)) from error
            self._require_active(job, scene, revision)
            self._write_scene_assembly_manifest(
                job,
                scene,
                revision,
                plan,
                accepted,
                output_path,
            )
        else:
            LOGGER.info(
                "Job %s scene %s revision %s: reusing verified assembled scene.",
                job.job_id,
                scene.scene_id,
                revision,
            )

        self._require_active(job, scene, revision)
        if deliver_to_discord and self.webhook_url:
            self._deliver_scene(
                job,
                scene,
                revision,
                plan,
                output_path,
                prompt_id_callback,
            )
        return ContinuationRenderResult(
            scene_path=output_path,
            plan=plan,
            chunk_paths=tuple(chunk_paths),
            reused_scene_assembly=reused,
        )

    def _require_active(
        self,
        job: JobPayload,
        scene: SceneSpec,
        revision: int,
    ) -> None:
        if not self.store.continuation_work_is_active(
            job.job_id,
            scene.scene_id,
            revision,
        ):
            raise ContinuationRenderError(
                f"Job {job.job_id} scene {scene.scene_id} revision {revision} "
                "was cancelled; late continuation work will not be committed."
            )

    def deliver_existing_scene(
        self,
        job: JobPayload,
        scene: SceneSpec,
        scene_path: str | Path,
        *,
        revision: int,
        overrides: SceneWorkflowOverrides | None = None,
        prompt_id_callback: Callable[[str], None] | None = None,
    ) -> ContinuationDeliveryResult | None:
        """Send a raw scene only after the caller has released generation models."""
        if not self.webhook_url:
            return None
        path = Path(scene_path).resolve(strict=True)
        self._require_active(job, scene, revision)
        plan = self.build_plan(job, scene, revision, overrides)
        return self._deliver_scene(
            job,
            scene,
            revision,
            plan,
            path,
            prompt_id_callback,
        )

    @staticmethod
    def build_plan(
        job: JobPayload,
        scene: SceneSpec,
        revision: int,
        overrides: SceneWorkflowOverrides | None,
    ) -> SceneFramePlan:
        continuation = (
            overrides.temporal_continuation
            if overrides is not None
            else scene.i2v.continuation
        )
        continuity = (
            overrides.continuity if overrides is not None else scene.i2v.continuity
        )
        segments = overrides.segments if overrides is not None else scene.i2v.segments
        requested_duration = scene.estimated_sec
        if continuation is not None:
            configured_duration = continuation.get("requested_duration_seconds")
            if isinstance(configured_duration, (int, float)) and not isinstance(
                configured_duration, bool
            ):
                requested_duration = float(configured_duration)
        return build_scene_frame_plan(
            job_id=job.job_id,
            scene_id=scene.scene_id,
            revision=revision,
            requested_duration_seconds=requested_duration,
            base_seed=scene.i2v.seed,
            fallback_prompt=scene.i2v.prompt,
            fallback_negative=scene.i2v.negative,
            continuity=continuity,
            raw_segments=segments,
        )

    # Compatibility for callers and saved tests from the initial beta.
    _build_plan = build_plan

    def _persist_plan(
        self,
        job: JobPayload,
        scene: SceneSpec,
        revision: int,
        plan: SceneFramePlan,
    ) -> None:
        plan_hash = plan.fingerprint()
        document = plan.to_document()
        self.store.ensure_continuation_plan(
            job.job_id,
            scene.scene_id,
            revision,
            plan_hash,
            document,
        )
        self.store.plan_chunks(
            job.job_id,
            scene.scene_id,
            revision,
            plan_hash,
            chunk_plan_documents(plan.chunks),
        )
        _write_immutable_json(
            self.storage.continuation_plan_path(
                job.job_id,
                scene.scene_id,
                revision,
            ),
            {"plan_hash": plan_hash, "plan": document},
        )

    def _accepted_attempt(
        self,
        job: JobPayload,
        scene: SceneSpec,
        revision: int,
        plan: SceneFramePlan,
        chunk_index: int,
        frame_path: Path,
        resolved_lora_filenames: Mapping[str, str],
        overrides: SceneWorkflowOverrides | None,
        runtime_contract_sha256: str,
    ) -> ChunkAttemptRecord | None:
        record = self.store.chunk_records(
            job.job_id,
            scene.scene_id,
            revision,
        )[chunk_index]
        attempts = self.store.chunk_attempts(
            job.job_id,
            scene.scene_id,
            revision,
            chunk_index,
        )
        expected_parameters = self._attempt_parameters(
            job,
            scene,
            frame_path,
            plan,
            chunk_index,
            resolved_lora_filenames,
            overrides,
            runtime_contract_sha256,
        )
        if record.accepted_attempt_number is not None:
            selected = next(
                (
                    attempt
                    for attempt in attempts
                    if attempt.attempt_number == record.accepted_attempt_number
                ),
                None,
            )
            if selected is not None and self._attempt_artifacts_are_valid(
                job,
                scene,
                revision,
                plan,
                record,
                selected,
                frame_path,
                resolved_lora_filenames,
                overrides,
                expected_parameters,
            ):
                return selected
            LOGGER.warning(
                "Job %s scene %s revision %s chunk %s: accepted artifacts failed "
                "verification; invalidating this chunk and its descendants.",
                job.job_id,
                scene.scene_id,
                revision,
                chunk_index,
            )
            self.store.invalidate_chunks_from(
                job.job_id,
                scene.scene_id,
                revision,
                chunk_index,
                reason="Accepted continuation artifacts failed integrity validation.",
            )
            return None

        # Recover the narrow crash window between finalizing COMPLETE.json and
        # selecting the already-complete attempt.
        for attempt in reversed(attempts):
            if attempt.state != ChunkState.COMPLETE:
                continue
            if self._attempt_artifacts_are_valid(
                job,
                scene,
                revision,
                plan,
                record,
                attempt,
                frame_path,
                resolved_lora_filenames,
                overrides,
                expected_parameters,
            ):
                self.store.select_chunk_attempt(
                    job.job_id,
                    scene.scene_id,
                    revision,
                    chunk_index,
                    attempt.attempt_number,
                    artifact_hash=attempt.artifact_hash,
                )
                return attempt
            try:
                self.store.update_chunk_attempt(
                    job.job_id,
                    scene.scene_id,
                    revision,
                    chunk_index,
                    attempt.attempt_number,
                    ChunkState.INVALIDATED,
                    error="Finalized continuation artifacts failed integrity validation.",
                )
            except StateTransitionError:
                LOGGER.warning(
                    "Could not invalidate corrupt unselected chunk attempt.",
                    exc_info=True,
                )
        return None

    def _render_chunk(
        self,
        job: JobPayload,
        scene: SceneSpec,
        frame_path: Path,
        revision: int,
        plan: SceneFramePlan,
        chunk_index: int,
        upstream: ChunkAttemptRecord | None,
        resolved_lora_filenames: Mapping[str, str],
        overrides: SceneWorkflowOverrides | None,
        runtime_contract_sha256: str,
        prompt_id_callback: Callable[[str], None] | None,
    ) -> ChunkAttemptRecord:
        chunk = plan.chunks[chunk_index]
        parameters = self._attempt_parameters(
            job,
            scene,
            frame_path,
            plan,
            chunk_index,
            resolved_lora_filenames,
            overrides,
            runtime_contract_sha256,
        )
        expected_upstream_hash = (
            upstream.artifact_hash if upstream is not None else None
        )
        active_states = {
            ChunkState.GENERATING_STAGE1,
            ChunkState.STAGE1_PERSISTING,
            ChunkState.STAGE1_COMPLETE,
            ChunkState.GENERATING_STAGE2,
            ChunkState.STAGE2_PERSISTING,
            ChunkState.DECODED,
            ChunkState.VALIDATING,
        }

        def immutable_inputs_match(attempt: ChunkAttemptRecord) -> bool:
            return (
                attempt.seed == chunk.seed
                and attempt.variation_index == chunk.variation_index
                and _document_hash(attempt.parameters) == _document_hash(parameters)
                and attempt.upstream_artifact_hash == expected_upstream_hash
            )

        def consumes_retry_budget(attempt: ChunkAttemptRecord) -> bool:
            return (
                attempt.state
                not in {
                    ChunkState.CANCELLED,
                    ChunkState.INVALIDATED,
                    ChunkState.STALE_UPSTREAM,
                }
                and immutable_inputs_match(attempt)
            )

        transport_recoveries = 0
        while True:
            record = self.store.chunk_records(
                job.job_id,
                scene.scene_id,
                revision,
            )[chunk_index]
            if record.state == ChunkState.FAILED_TERMINAL:
                raise ContinuationRenderError(
                    f"Chunk {chunk_index + 1}/{plan.chunk_count} exhausted "
                    f"{self.max_attempts} attempts: {record.error or 'unknown error'}"
                )
            existing = self.store.chunk_attempts(
                job.job_id,
                scene.scene_id,
                revision,
                chunk_index,
            )
            mismatched_active = next(
                (
                    attempt
                    for attempt in reversed(existing)
                    if attempt.state in active_states
                    and not immutable_inputs_match(attempt)
                ),
                None,
            )
            if mismatched_active is not None:
                for key in ("stage1_prompt_id", "stage2_prompt_id"):
                    prompt_id = mismatched_active.result.get(key)
                    if isinstance(prompt_id, str) and prompt_id:
                        self._cancel_owned_prompt_best_effort(prompt_id)
                self.store.update_chunk_attempt(
                    job.job_id,
                    scene.scene_id,
                    revision,
                    chunk_index,
                    mismatched_active.attempt_number,
                    ChunkState.INVALIDATED,
                    result=mismatched_active.result,
                    error=(
                        "Immutable continuation inputs changed before this "
                        "attempt could be reclaimed."
                    ),
                )
                continue
            budget_attempts = [
                attempt for attempt in existing if consumes_retry_budget(attempt)
            ]
            active = next(
                (
                    attempt
                    for attempt in reversed(budget_attempts)
                    if attempt.state in active_states
                ),
                None,
            )
            if active is None and len(budget_attempts) >= self.max_attempts:
                raise ContinuationRenderError(
                    f"Chunk {chunk_index + 1}/{plan.chunk_count} exhausted "
                    f"{self.max_attempts} attempts."
                )
            try:
                attempt = self.store.begin_chunk_attempt(
                    job.job_id,
                    scene.scene_id,
                    revision,
                    chunk_index,
                    seed=chunk.seed,
                    variation_index=chunk.variation_index,
                    parameters=parameters,
                    upstream_artifact_hash=expected_upstream_hash,
                )
            except StateTransitionError as error:
                raise ContinuationRenderError(str(error)) from error

            try:
                completed = self._execute_attempt(
                    job,
                    scene,
                    frame_path,
                    revision,
                    plan,
                    chunk_index,
                    attempt,
                    upstream,
                    resolved_lora_filenames,
                    overrides,
                    prompt_id_callback,
                )
                self.store.select_chunk_attempt(
                    job.job_id,
                    scene.scene_id,
                    revision,
                    chunk_index,
                    completed.attempt_number,
                    artifact_hash=completed.artifact_hash,
                )
                return completed
            except Exception as error:
                current_attempts = self.store.chunk_attempts(
                    job.job_id,
                    scene.scene_id,
                    revision,
                    chunk_index,
                )
                latest = current_attempts[-1]
                if latest.state == ChunkState.CANCELLED:
                    raise ContinuationRenderError(
                        f"Chunk {chunk_index + 1}/{plan.chunk_count} was cancelled."
                    ) from error
                comfy_alive = (
                    self.comfy.alive()
                    if isinstance(error, ComfyHttpError)
                    else True
                )
                if isinstance(error, ComfyHttpError) and not comfy_alive:
                    # Preserve active attempt/prompt ownership. Outer supervisor
                    # performs controlled ComfyUI restart, then reclaims it.
                    raise
                if (
                    isinstance(error, ComfyHttpError)
                    and comfy_alive
                    and transport_recoveries < self.max_attempts
                    and (
                        latest.result.get("stage1_prompt_id")
                        or latest.result.get("stage2_prompt_id")
                    )
                ):
                    transport_recoveries += 1
                    LOGGER.warning(
                        "Job %s scene %s revision %s chunk %s/%s lost a "
                        "ComfyUI transport/download step; reclaiming persisted "
                        "prompt state (%s/%s): %s",
                        job.job_id,
                        scene.scene_id,
                        revision,
                        chunk_index + 1,
                        plan.chunk_count,
                        transport_recoveries,
                        self.max_attempts,
                        error,
                    )
                    continue
                terminal = (
                    len(
                        [
                            candidate
                            for candidate in current_attempts
                            if consumes_retry_budget(candidate)
                        ]
                    )
                    >= self.max_attempts
                )
                failure_state = (
                    ChunkState.FAILED_TERMINAL
                    if terminal
                    else ChunkState.FAILED_RETRYABLE
                )
                try:
                    self.store.update_chunk_attempt(
                        job.job_id,
                        scene.scene_id,
                        revision,
                        chunk_index,
                        latest.attempt_number,
                        failure_state,
                        result=latest.result,
                        error=str(error),
                    )
                except StateTransitionError:
                    LOGGER.warning(
                        "Chunk failure state could not be persisted because its "
                        "lifecycle changed concurrently.",
                        exc_info=True,
                    )
                if terminal:
                    raise ContinuationRenderError(
                        f"Chunk {chunk_index + 1}/{plan.chunk_count} failed after "
                        f"{latest.attempt_number} attempts: {error}"
                    ) from error
                LOGGER.warning(
                    "Job %s scene %s revision %s chunk %s/%s attempt %s failed; "
                    "retrying deterministically: %s",
                    job.job_id,
                    scene.scene_id,
                    revision,
                    chunk_index + 1,
                    plan.chunk_count,
                    latest.attempt_number,
                    error,
                )

    def _execute_attempt(
        self,
        job: JobPayload,
        scene: SceneSpec,
        frame_path: Path,
        revision: int,
        plan: SceneFramePlan,
        chunk_index: int,
        attempt: ChunkAttemptRecord,
        upstream: ChunkAttemptRecord | None,
        resolved_lora_filenames: Mapping[str, str],
        overrides: SceneWorkflowOverrides | None,
        prompt_id_callback: Callable[[str], None] | None,
    ) -> ChunkAttemptRecord:
        chunk = plan.chunks[chunk_index]
        identity = {
            "job_id": job.job_id,
            "scene_id": scene.scene_id,
            "revision": revision,
            "chunk_index": chunk_index,
            "attempt_number": attempt.attempt_number,
        }
        result = dict(attempt.result)
        expected_temporal_tokens = handoff_latent_token_count(chunk)
        if not latent_checkpoint_is_valid(
            self.storage,
            **identity,
            artifact_kind="stage1_handoff",
            expected_temporal_tokens=expected_temporal_tokens,
        ):
            attempt = self.store.update_chunk_attempt(
                **identity,
                state=ChunkState.GENERATING_STAGE1,
                result=result,
            )
            LOGGER.info(
                "Job %s scene %s revision %s: chunk %s/%s first pass "
                "(attempt %s, %s new transitions).",
                job.job_id,
                scene.scene_id,
                revision,
                chunk_index + 1,
                plan.chunk_count,
                attempt.attempt_number,
                chunk.new_transition_frames,
            )
            stage1 = build_continuation_stage1_workflow(
                job,
                scene,
                frame_path,
                plan,
                chunk,
                revision=revision,
                attempt_number=attempt.attempt_number,
                previous_attempt_number=(
                    upstream.attempt_number if upstream is not None else None
                ),
                resolved_lora_filenames=resolved_lora_filenames,
                overrides=overrides,
            )
            _history, result = self._run_or_reclaim_prompt(
                workflow=stage1.api,
                identity=identity,
                attempt_state=ChunkState.GENERATING_STAGE1,
                result=result,
                prompt_key="stage1_prompt_id",
                workflow_hash_key="stage1_workflow_sha256",
                prompt_id_callback=prompt_id_callback,
            )
        _, stage1_manifest = load_latent_checkpoint(
            self.storage,
            **identity,
            artifact_kind="stage1_handoff",
            expected_temporal_tokens=expected_temporal_tokens,
        )
        result["stage1_checkpoint_sha256"] = stage1_manifest["sha256"]
        attempt = self.store.update_chunk_attempt(
            **identity,
            state=ChunkState.STAGE1_COMPLETE,
            result=result,
        )

        video_path = self.storage.chunk_video_path(**identity)
        stage2_video_valid = latent_checkpoint_is_valid(
            self.storage,
            **identity,
            artifact_kind="stage2_video",
            expected_temporal_tokens=expected_temporal_tokens,
        )
        stage2_audio_valid = latent_checkpoint_is_valid(
            self.storage,
            **identity,
            artifact_kind="stage2_audio",
        )

        def raw_video_is_valid() -> bool:
            if not video_path.is_file():
                return False
            try:
                self.assembler.validate_chunk(plan, chunk_index, video_path)
            except SceneChunkAssemblyError:
                return False
            return True

        stage2_latents_valid = stage2_video_valid and stage2_audio_valid
        raw_video_valid = raw_video_is_valid()
        if not stage2_latents_valid:
            attempt = self.store.update_chunk_attempt(
                **identity,
                state=ChunkState.GENERATING_STAGE2,
                result=result,
            )
            LOGGER.info(
                "Job %s scene %s revision %s: chunk %s/%s full-resolution "
                "refinement (attempt %s, %s raw frames).",
                job.job_id,
                scene.scene_id,
                revision,
                chunk_index + 1,
                plan.chunk_count,
                attempt.attempt_number,
                refinement_raw_frame_count(chunk),
            )
            stage2 = build_continuation_stage2_workflow(
                job,
                scene,
                frame_path,
                plan,
                chunk,
                revision=revision,
                attempt_number=attempt.attempt_number,
                previous_attempt_number=(
                    upstream.attempt_number if upstream is not None else None
                ),
                previous_chunk_path=(
                    upstream.video_path if upstream is not None else None
                ),
                resolved_lora_filenames=resolved_lora_filenames,
                overrides=overrides,
            )
            history, result = self._run_or_reclaim_prompt(
                workflow=stage2.workflow.api,
                identity=identity,
                attempt_state=ChunkState.GENERATING_STAGE2,
                result=result,
                prompt_key="stage2_prompt_id",
                workflow_hash_key="stage2_workflow_sha256",
                prompt_id_callback=prompt_id_callback,
            )
            metadata = find_video_output(
                history,
                stage2.workflow.output_node_id,
                expected_suffixes=(".mkv",),
            )
            self.comfy.download_output(metadata, video_path)
            raw_video_valid = raw_video_is_valid()

        if stage2_latents_valid and not raw_video_valid:
            # The full sampler may have completed and persisted both AV
            # latents while its downstream lossless encode is still queued or
            # recoverable from history. Reclaim that output before creating a
            # decode-only prompt.
            recovered_original_output = False
            stage2_prompt_id = result.get("stage2_prompt_id")
            if isinstance(stage2_prompt_id, str) and stage2_prompt_id:
                try:
                    history = self.comfy.completed_prompt(stage2_prompt_id)
                    if (
                        history is None
                        and self.comfy.prompt_is_queued(stage2_prompt_id)
                    ):
                        history = self.comfy.wait_for_prompt(
                            stage2_prompt_id,
                            timeout_seconds=self.timeout_seconds,
                        )
                except ComfyHttpError:
                    history = None
                    LOGGER.info(
                        "Original stage-two prompt cannot supply a valid raw "
                        "window; recovering from verified AV checkpoints."
                    )
                if history is not None:
                    stage2 = build_continuation_stage2_workflow(
                        job,
                        scene,
                        frame_path,
                        plan,
                        chunk,
                        revision=revision,
                        attempt_number=attempt.attempt_number,
                        previous_attempt_number=(
                            upstream.attempt_number if upstream is not None else None
                        ),
                        previous_chunk_path=(
                            upstream.video_path if upstream is not None else None
                        ),
                        resolved_lora_filenames=resolved_lora_filenames,
                        overrides=overrides,
                    )
                    try:
                        metadata = find_video_output(
                            history,
                            stage2.workflow.output_node_id,
                            expected_suffixes=(".mkv",),
                        )
                        self.comfy.download_output(metadata, video_path)
                        recovered_original_output = raw_video_is_valid()
                    except (ComfyHttpError, OSError):
                        LOGGER.warning(
                            "Completed stage-two output could not be restored; "
                            "falling back to checkpoint-only decode.",
                            exc_info=True,
                        )
            if not recovered_original_output:
                attempt = self.store.update_chunk_attempt(
                    **identity,
                    state=ChunkState.DECODED,
                    result=result,
                )
                decode = build_continuation_decode_workflow(
                    job,
                    scene,
                    plan,
                    chunk,
                    revision=revision,
                    attempt_number=attempt.attempt_number,
                )
                history, result = self._run_or_reclaim_prompt(
                    workflow=decode.api,
                    identity=identity,
                    attempt_state=ChunkState.DECODED,
                    result=result,
                    prompt_key="decode_prompt_id",
                    workflow_hash_key="decode_workflow_sha256",
                    prompt_id_callback=prompt_id_callback,
                )
                metadata = find_video_output(
                    history,
                    decode.output_node_id,
                    expected_suffixes=(".mkv",),
                )
                self.comfy.download_output(metadata, video_path)
            if not raw_video_is_valid():
                raise ContinuationRenderError(
                    "Checkpoint-only decode did not produce a valid raw chunk."
                )

        _, stage2_manifest = load_latent_checkpoint(
            self.storage,
            **identity,
            artifact_kind="stage2_video",
            expected_temporal_tokens=expected_temporal_tokens,
        )
        _, stage2_audio_manifest = load_latent_checkpoint(
            self.storage,
            **identity,
            artifact_kind="stage2_audio",
        )
        self.assembler.validate_chunk(plan, chunk_index, video_path)
        result["stage2_checkpoint_sha256"] = stage2_manifest["sha256"]
        result["stage2_audio_checkpoint_sha256"] = stage2_audio_manifest["sha256"]
        result["raw_video_sha256"] = sha256_file(video_path)
        attempt = self.store.update_chunk_attempt(
            **identity,
            state=ChunkState.VALIDATING,
            video_path=str(video_path),
            result=result,
        )

        complete_path = self.storage.chunk_attempt_manifest_path(**identity)
        span = assembly_spans(plan)[chunk_index]
        complete_document: dict[str, Any] = {
            "schema_version": CONTINUATION_ATTEMPT_SCHEMA_VERSION,
            **identity,
            "plan_hash": plan.fingerprint(),
            "upstream_artifact_hash": (
                upstream.artifact_hash if upstream is not None else None
            ),
            "seed": chunk.seed,
            "variation_index": chunk.variation_index,
            "prompt": chunk.prompt,
            "negative": chunk.negative,
            "attempt_parameters_sha256": _document_hash(attempt.parameters),
            "expected_raw_frames": refinement_raw_frame_count(chunk),
            "committed_master_frames": span.frame_count,
            "stage1_checkpoint": {
                "path": str(
                    self.storage.chunk_checkpoint_path(
                        **identity,
                        artifact_kind="stage1_handoff",
                    )
                ),
                "manifest_path": str(
                    self.storage.chunk_checkpoint_manifest_path(
                        **identity,
                        artifact_kind="stage1_handoff",
                    )
                ),
                "sha256": stage1_manifest["sha256"],
            },
            "stage2_checkpoint": {
                "path": str(
                    self.storage.chunk_checkpoint_path(
                        **identity,
                        artifact_kind="stage2_video",
                    )
                ),
                "manifest_path": str(
                    self.storage.chunk_checkpoint_manifest_path(
                        **identity,
                        artifact_kind="stage2_video",
                    )
                ),
                "sha256": stage2_manifest["sha256"],
            },
            "stage2_audio_checkpoint": {
                "path": str(
                    self.storage.chunk_checkpoint_path(
                        **identity,
                        artifact_kind="stage2_audio",
                    )
                ),
                "manifest_path": str(
                    self.storage.chunk_checkpoint_manifest_path(
                        **identity,
                        artifact_kind="stage2_audio",
                    )
                ),
                "sha256": stage2_audio_manifest["sha256"],
            },
            "raw_video": {
                "path": str(video_path),
                "sha256": result["raw_video_sha256"],
                "byte_size": video_path.stat().st_size,
            },
            "result": result,
        }
        write_json_atomic(complete_path, complete_document)
        complete_hash = sha256_file(complete_path)
        return self.store.update_chunk_attempt(
            **identity,
            state=ChunkState.COMPLETE,
            artifact_manifest_path=str(complete_path),
            artifact_hash=complete_hash,
            video_path=str(video_path),
            result=result,
        )

    def _run_or_reclaim_prompt(
        self,
        *,
        workflow: Mapping[str, Any],
        identity: Mapping[str, Any],
        attempt_state: ChunkState,
        result: Mapping[str, Any],
        prompt_key: str,
        workflow_hash_key: str,
        prompt_id_callback: Callable[[str], None] | None,
    ) -> tuple[Mapping[str, Any], dict[str, Any]]:
        """Persist prompt ownership before waiting and reclaim it after restart."""
        self._require_identity_active(identity)
        mutable_result = dict(result)
        prompt_id = mutable_result.get(prompt_key)
        history: Mapping[str, Any] | None = None
        if isinstance(prompt_id, str) and prompt_id:
            history = self.comfy.completed_prompt(prompt_id)
            if history is None and not self.comfy.prompt_is_queued(prompt_id):
                LOGGER.warning(
                    "Persisted ComfyUI prompt %s is absent from queue and history; "
                    "requeueing the same immutable chunk attempt.",
                    prompt_id,
                )
                prompt_id = None
        else:
            prompt_id = None

        if prompt_id is None:
            self._require_identity_active(identity)
            prompt_id = self.comfy.queue_prompt(workflow)
            mutable_result[prompt_key] = prompt_id
            mutable_result[workflow_hash_key] = _document_hash(workflow)
            try:
                # This write deliberately happens before the blocking wait. A
                # restarted GUI/supervisor can now wait on the exact owned
                # prompt or recover its successful history instead of
                # duplicating GPU work.
                self.store.update_chunk_attempt(
                    **identity,
                    state=attempt_state,
                    result=mutable_result,
                )
                if prompt_id_callback is not None:
                    prompt_id_callback(prompt_id)
                self._require_identity_active(identity)
            except BaseException:
                # /prompt and SQLite cannot share one transaction. Close the
                # narrow queue-before-persist window by cancelling only this
                # project-owned prompt whenever durable ownership cannot be
                # committed.
                self._cancel_owned_prompt_best_effort(prompt_id)
                raise

        if history is None:
            history = self.comfy.wait_for_prompt(
                prompt_id,
                timeout_seconds=self.timeout_seconds,
            )
        self._require_identity_active(identity)
        return history, mutable_result

    def _require_identity_active(self, identity: Mapping[str, Any]) -> None:
        try:
            job_id = str(identity["job_id"])
            scene_id = int(identity["scene_id"])
            revision = int(identity["revision"])
        except (KeyError, TypeError, ValueError) as error:
            raise ContinuationRenderError(
                "Continuation prompt identity is incomplete."
            ) from error
        if not self.store.continuation_work_is_active(
            job_id,
            scene_id,
            revision,
        ):
            raise ContinuationRenderError(
                f"Job {job_id} scene {scene_id} revision {revision} no longer "
                "owns this continuation prompt."
            )

    def _attempt_artifacts_are_valid(
        self,
        job: JobPayload,
        scene: SceneSpec,
        revision: int,
        plan: SceneFramePlan,
        chunk_record: SceneChunkRecord,
        attempt: ChunkAttemptRecord,
        frame_path: Path,
        resolved_lora_filenames: Mapping[str, str],
        overrides: SceneWorkflowOverrides | None,
        expected_parameters: Mapping[str, Any],
    ) -> bool:
        try:
            if (
                attempt.state != ChunkState.COMPLETE
                or _document_hash(attempt.parameters)
                != _document_hash(expected_parameters)
                or not attempt.artifact_hash
                or not attempt.artifact_manifest_path
                or not attempt.video_path
                or attempt.upstream_artifact_hash
                != (
                    None
                    if chunk_record.chunk_index == 0
                    else self.store.chunk_records(
                        job.job_id,
                        scene.scene_id,
                        revision,
                    )[chunk_record.chunk_index - 1].accepted_artifact_hash
                )
            ):
                return False
            identity = {
                "job_id": job.job_id,
                "scene_id": scene.scene_id,
                "revision": revision,
                "chunk_index": chunk_record.chunk_index,
                "attempt_number": attempt.attempt_number,
            }
            manifest_path = Path(attempt.artifact_manifest_path)
            if (
                manifest_path
                != self.storage.chunk_attempt_manifest_path(**identity)
                or not manifest_path.is_file()
                or sha256_file(manifest_path) != attempt.artifact_hash
            ):
                return False
            manifest = _read_json(manifest_path)
            expected = {
                **identity,
                "schema_version": CONTINUATION_ATTEMPT_SCHEMA_VERSION,
                "plan_hash": plan.fingerprint(),
                "upstream_artifact_hash": attempt.upstream_artifact_hash,
            }
            if any(manifest.get(key) != value for key, value in expected.items()):
                return False
            if (
                manifest.get("attempt_parameters_sha256")
                != _document_hash(expected_parameters)
                or manifest.get("result") != attempt.result
            ):
                return False
            upstream: ChunkAttemptRecord | None = None
            if chunk_record.chunk_index > 0:
                predecessor = self.store.chunk_records(
                    job.job_id,
                    scene.scene_id,
                    revision,
                )[chunk_record.chunk_index - 1]
                if predecessor.accepted_attempt_number is None:
                    return False
                upstream = next(
                    (
                        candidate
                        for candidate in self.store.chunk_attempts(
                            job.job_id,
                            scene.scene_id,
                            revision,
                            chunk_record.chunk_index - 1,
                        )
                        if candidate.attempt_number
                        == predecessor.accepted_attempt_number
                    ),
                    None,
                )
                if upstream is None or not upstream.video_path:
                    return False
            chunk = plan.chunks[chunk_record.chunk_index]
            stage1 = build_continuation_stage1_workflow(
                job,
                scene,
                frame_path,
                plan,
                chunk,
                revision=revision,
                attempt_number=attempt.attempt_number,
                previous_attempt_number=(
                    upstream.attempt_number if upstream is not None else None
                ),
                resolved_lora_filenames=resolved_lora_filenames,
                overrides=overrides,
            )
            stage2 = build_continuation_stage2_workflow(
                job,
                scene,
                frame_path,
                plan,
                chunk,
                revision=revision,
                attempt_number=attempt.attempt_number,
                previous_attempt_number=(
                    upstream.attempt_number if upstream is not None else None
                ),
                previous_chunk_path=(
                    upstream.video_path if upstream is not None else None
                ),
                resolved_lora_filenames=resolved_lora_filenames,
                overrides=overrides,
            )
            decode = build_continuation_decode_workflow(
                job,
                scene,
                plan,
                chunk,
                revision=revision,
                attempt_number=attempt.attempt_number,
            )
            if (
                attempt.result.get("stage1_workflow_sha256")
                != _document_hash(stage1.api)
                or attempt.result.get("stage2_workflow_sha256")
                != _document_hash(stage2.workflow.api)
                or (
                    "decode_workflow_sha256" in attempt.result
                    and attempt.result.get("decode_workflow_sha256")
                    != _document_hash(decode.api)
                )
            ):
                return False
            for artifact_kind in (
                "stage1_handoff",
                "stage2_video",
                "stage2_audio",
            ):
                load_latent_checkpoint(
                    self.storage,
                    **identity,
                    artifact_kind=artifact_kind,
                    expected_temporal_tokens=(
                        handoff_latent_token_count(
                            plan.chunks[chunk_record.chunk_index]
                        )
                        if artifact_kind != "stage2_audio"
                        else None
                    ),
                )
            video_path = Path(attempt.video_path)
            raw_video = manifest.get("raw_video")
            if (
                video_path != self.storage.chunk_video_path(**identity)
                or not isinstance(raw_video, Mapping)
                or raw_video.get("path") != str(video_path)
                or raw_video.get("byte_size") != video_path.stat().st_size
                or raw_video.get("sha256") != sha256_file(video_path)
            ):
                return False
            self.assembler.validate_chunk(
                plan,
                chunk_record.chunk_index,
                video_path,
            )
            return True
        except (
            ChunkArtifactError,
            ContinuationRenderError,
            OSError,
            SceneChunkAssemblyError,
            StateTransitionError,
        ):
            return False

    def _attempt_parameters(
        self,
        job: JobPayload,
        scene: SceneSpec,
        frame_path: Path,
        plan: SceneFramePlan,
        chunk_index: int,
        resolved_lora_filenames: Mapping[str, str],
        overrides: SceneWorkflowOverrides | None,
        runtime_contract_sha256: str,
    ) -> Mapping[str, Any]:
        return {
            "schema_version": CONTINUATION_ATTEMPT_SCHEMA_VERSION,
            "plan_hash": plan.fingerprint(),
            "chunk": asdict(plan.chunks[chunk_index]),
            "source_frame": {
                "path": str(frame_path),
                "sha256": sha256_file(frame_path),
                "byte_size": frame_path.stat().st_size,
            },
            "scene_i2v": {
                "prompt": scene.i2v.prompt,
                "negative": scene.i2v.negative,
                "seed": str(scene.i2v.seed),
                "loras": [
                    {
                        "name": lora.name,
                        "download_url": lora.download_url,
                        "weight": lora.weight,
                        "model_id": lora.model_id,
                        "version_id": lora.version_id,
                    }
                    for lora in effective_i2v_loras(job, scene)
                ],
            },
            "resolved_i2v_lora_filenames": {
                key: value
                for key, value in sorted(resolved_lora_filenames.items())
                if key.startswith("i2v:")
            },
            "workflow_overrides": (
                asdict(overrides) if overrides is not None else None
            ),
            "job_schema_version": job.schema_version,
            "runtime_identity": {
                "generation_implementation_sha256": (
                    self.cache_implementation_sha256()
                ),
                "node_contracts_sha256": runtime_contract_sha256,
                "checkpoint_filename": LTX_CHECKPOINT,
                "text_encoder_filename": LTX_TEXT_ENCODER,
                "spatial_upscaler_filename": I2V_SPATIAL_UPSCALER,
                "mandatory_loras": [
                    {"filename": filename, "weight": weight}
                    for filename, weight in MANDATORY_I2V_LORAS
                ],
                "production": {
                    "width": PRODUCTION_WIDTH,
                    "height": PRODUCTION_HEIGHT,
                    "fps": PRODUCTION_FPS,
                    "base_width": I2V_BASE_WIDTH,
                    "base_height": I2V_BASE_HEIGHT,
                    "sampler": I2V_SAMPLER,
                    "first_pass_sigmas": list(I2V_FIRST_PASS_SIGMAS),
                    "second_pass_sigmas": list(I2V_UPSCALE_PASS_SIGMAS),
                },
                # Shared model files are outside this repository's authorized
                # read boundary. Their exact hashes remain a mandatory
                # pre-auto-rollout validation gate; deterministic filenames,
                # Civitai IDs, workflow hashes, and live node contracts are
                # still pinned here for safe project-local cache invalidation.
                "shared_model_hash_policy": "external_validation_required",
            },
        }

    def runtime_contract_sha256(self) -> str:
        object_info = getattr(self.comfy, "object_info", None)
        if not callable(object_info):
            return _document_hash({"status": "unavailable-test-double"})
        contracts: dict[str, Any] = {}
        for node_type in CONTINUATION_CONTRACT_NODE_TYPES:
            document = object_info(node_type)
            if not isinstance(document, Mapping) or node_type not in document:
                raise ContinuationRenderError(
                    f"Live ComfyUI contract is missing required node {node_type}."
                )
            contracts[node_type] = _structural_node_contract(
                document[node_type]
            )
        return _document_hash(contracts)

    @staticmethod
    def cache_implementation_sha256() -> str:
        project_root = Path(__file__).resolve().parent.parent
        return _document_hash(
            {
                relative_path: sha256_file(project_root / relative_path)
                for relative_path in CONTINUATION_CACHE_IMPLEMENTATION_PATHS
            }
        )

    @staticmethod
    def implementation_sha256() -> str:
        project_root = Path(__file__).resolve().parent.parent
        return _document_hash(
            {
                relative_path: sha256_file(project_root / relative_path)
                for relative_path in CONTINUATION_IMPLEMENTATION_PATHS
            }
        )

    # Compatibility aliases for pre-rollout callers.
    def _runtime_contract_sha256(self) -> str:
        return self.runtime_contract_sha256()

    @staticmethod
    def _implementation_sha256() -> str:
        return ContinuationRenderer.implementation_sha256()

    def _scene_assembly_manifest_path(
        self,
        job: JobPayload,
        scene: SceneSpec,
        revision: int,
    ) -> Path:
        return (
            self.storage.scene_assembly_root(job.job_id, scene.scene_id, revision)
            / "COMPLETE.json"
        )

    def _scene_assembly_is_valid(
        self,
        job: JobPayload,
        scene: SceneSpec,
        revision: int,
        plan: SceneFramePlan,
        attempts: list[ChunkAttemptRecord],
        output_path: Path,
    ) -> bool:
        manifest_path = self._scene_assembly_manifest_path(
            job,
            scene,
            revision,
        )
        if not output_path.is_file() or not manifest_path.is_file():
            return False
        try:
            manifest = _read_json(manifest_path)
            if (
                manifest.get("plan_hash") != plan.fingerprint()
                or manifest.get("scene_path") != str(output_path)
                or manifest.get("scene_sha256") != sha256_file(output_path)
                or manifest.get("accepted_attempts")
                != [
                    {
                        "chunk_index": attempt.chunk_index,
                        "attempt_number": attempt.attempt_number,
                        "artifact_hash": attempt.artifact_hash,
                    }
                    for attempt in attempts
                ]
            ):
                return False
            self.assembler.validate_scene(plan, output_path)
            return True
        except (ContinuationRenderError, OSError, SceneChunkAssemblyError):
            return False

    def _write_scene_assembly_manifest(
        self,
        job: JobPayload,
        scene: SceneSpec,
        revision: int,
        plan: SceneFramePlan,
        attempts: list[ChunkAttemptRecord],
        output_path: Path,
    ) -> None:
        write_json_atomic(
            self._scene_assembly_manifest_path(job, scene, revision),
            {
                "schema_version": CONTINUATION_ATTEMPT_SCHEMA_VERSION,
                "job_id": job.job_id,
                "scene_id": scene.scene_id,
                "revision": revision,
                "plan_hash": plan.fingerprint(),
                "accepted_attempts": [
                    {
                        "chunk_index": attempt.chunk_index,
                        "attempt_number": attempt.attempt_number,
                        "artifact_hash": attempt.artifact_hash,
                    }
                    for attempt in attempts
                ],
                "scene_path": str(output_path),
                "scene_sha256": sha256_file(output_path),
                "timeline_output_frames": plan.timeline_output_frames,
            },
        )

    def _deliver_scene(
        self,
        job: JobPayload,
        scene: SceneSpec,
        revision: int,
        plan: SceneFramePlan,
        output_path: Path,
        prompt_id_callback: Callable[[str], None] | None,
    ) -> ContinuationDeliveryResult:
        self._require_active(job, scene, revision)
        delivery_marker = (
            self.storage.scene_assembly_root(job.job_id, scene.scene_id, revision)
            / "discord-delivery.json"
        )
        scene_hash = sha256_file(output_path)
        plan_hash = plan.fingerprint()
        delivery = build_assembled_scene_delivery_workflow(
            job,
            scene,
            output_path,
            self.webhook_url or "",
        )
        workflow_hash = _document_hash(delivery.api)
        signature = {
            "schema_version": CONTINUATION_DELIVERY_SCHEMA_VERSION,
            "job_id": job.job_id,
            "scene_id": scene.scene_id,
            "revision": revision,
            "scene_path": str(output_path),
            "scene_sha256": scene_hash,
            "plan_hash": plan_hash,
            "workflow_sha256": workflow_hash,
        }
        marker: Mapping[str, Any] | None = None
        if delivery_marker.is_file():
            try:
                marker = _read_json(delivery_marker)
            except ContinuationRenderError:
                marker = None

        if marker is not None and self._delivery_marker_matches(marker, signature):
            status = marker.get("status")
            prompt_id = marker.get("prompt_id")
            if status == "sent" and isinstance(prompt_id, str) and prompt_id:
                return ContinuationDeliveryResult(
                    status="sent",
                    prompt_id=prompt_id,
                    reused_prompt=True,
                )
            if status == "queued" and isinstance(prompt_id, str) and prompt_id:
                return self._reclaim_delivery_prompt(
                    job,
                    scene,
                    revision,
                    delivery_marker,
                    signature,
                    prompt_id,
                    prompt_id_callback,
                )
            if status == "ambiguous":
                raise ContinuationDeliveryError(
                    "Discord delivery may already have completed, but ComfyUI no "
                    "longer has authoritative queue/history state. Automatic resend "
                    "is blocked to prevent a duplicate post; raw scene remains valid."
                )

        if (
            marker is not None
            and marker.get("status") == "queued"
            and isinstance(marker.get("prompt_id"), str)
            and marker.get("prompt_id")
        ):
            # A changed scene/plan/workflow must not inherit the old send. Stop
            # that exact project-owned prompt when it is still cancellable,
            # then queue the new immutable delivery signature.
            self._cancel_owned_prompt_best_effort(str(marker["prompt_id"]))

        LOGGER.info(
            "Job %s scene %s revision %s: sending a Discord-only watermarked copy.",
            job.job_id,
            scene.scene_id,
            revision,
        )
        self._require_active(job, scene, revision)
        try:
            prompt_id = self.comfy.queue_prompt(delivery.api)
        except ComfyHttpError as error:
            self._write_delivery_marker_best_effort(
                delivery_marker,
                signature,
                status="failed",
                prompt_id=None,
                failure_reason="queue_rejected",
            )
            raise ContinuationDeliveryError(
                "Discord delivery could not be queued; raw scene output remains valid."
            ) from error

        queued_marker = self._delivery_marker_document(
            signature,
            status="queued",
            prompt_id=prompt_id,
        )
        try:
            # Persist immediately after /prompt returns. A restarted worker can
            # now reclaim this exact side-effecting Discord prompt.
            write_json_atomic(delivery_marker, queued_marker)
        except Exception as error:
            self._cancel_owned_prompt_best_effort(prompt_id)
            raise ContinuationDeliveryError(
                "Discord delivery was queued but its durable marker could not be "
                "written; the owned prompt was cancelled best-effort."
            ) from error

        try:
            self._require_active(job, scene, revision)
            if prompt_id_callback is not None:
                prompt_id_callback(prompt_id)
            self._require_active(job, scene, revision)
        except Exception as error:
            self._cancel_owned_prompt_best_effort(prompt_id)
            self._write_delivery_marker_best_effort(
                delivery_marker,
                signature,
                status="ambiguous",
                prompt_id=prompt_id,
                failure_reason="ownership_changed_before_wait",
            )
            raise ContinuationDeliveryError(
                "Discord delivery ownership changed before wait; automatic resend "
                "is blocked because completion is ambiguous."
            ) from error

        return self._wait_for_delivery_prompt(
            job,
            scene,
            revision,
            delivery_marker,
            signature,
            prompt_id,
            reused_prompt=False,
        )

    @staticmethod
    def _delivery_marker_matches(
        marker: Mapping[str, Any],
        signature: Mapping[str, Any],
    ) -> bool:
        return all(marker.get(key) == value for key, value in signature.items())

    @staticmethod
    def _delivery_marker_document(
        signature: Mapping[str, Any],
        *,
        status: str,
        prompt_id: str | None,
        failure_reason: str | None = None,
    ) -> dict[str, Any]:
        if status not in {"queued", "sent", "failed", "ambiguous"}:
            raise ValueError("Unsupported Discord delivery marker status.")
        document = {
            **signature,
            "status": status,
            "prompt_id": prompt_id,
        }
        if failure_reason is not None:
            document["failure_reason"] = failure_reason
        return document

    def _reclaim_delivery_prompt(
        self,
        job: JobPayload,
        scene: SceneSpec,
        revision: int,
        delivery_marker: Path,
        signature: Mapping[str, Any],
        prompt_id: str,
        prompt_id_callback: Callable[[str], None] | None,
    ) -> ContinuationDeliveryResult:
        self._require_active(job, scene, revision)
        try:
            history = self.comfy.completed_prompt(prompt_id)
        except ComfyHttpError as error:
            return self._resolve_delivery_prompt_error(
                job,
                scene,
                revision,
                delivery_marker,
                signature,
                prompt_id,
                error,
            )
        if history is not None:
            return self._commit_sent_delivery(
                job,
                scene,
                revision,
                delivery_marker,
                signature,
                prompt_id,
                reused_prompt=True,
            )
        try:
            queued = self.comfy.prompt_is_queued(prompt_id)
        except ComfyHttpError as error:
            # Queue/history transport is ambiguous. Keep the durable queued
            # marker so a later restart checks the same prompt instead of
            # duplicating the Discord side effect.
            raise ContinuationDeliveryError(
                "Discord delivery state could not be checked; its queued marker "
                "was preserved for restart-safe recovery."
            ) from error
        if not queued:
            self._write_delivery_marker_best_effort(
                delivery_marker,
                signature,
                status="ambiguous",
                prompt_id=prompt_id,
                failure_reason="prompt_missing",
            )
            raise ContinuationDeliveryError(
                "Persisted Discord delivery prompt is absent from ComfyUI queue "
                "and history; automatic resend is blocked to prevent a duplicate."
            )
        if prompt_id_callback is not None:
            prompt_id_callback(prompt_id)
        return self._wait_for_delivery_prompt(
            job,
            scene,
            revision,
            delivery_marker,
            signature,
            prompt_id,
            reused_prompt=True,
        )

    def _wait_for_delivery_prompt(
        self,
        job: JobPayload,
        scene: SceneSpec,
        revision: int,
        delivery_marker: Path,
        signature: Mapping[str, Any],
        prompt_id: str,
        *,
        reused_prompt: bool,
    ) -> ContinuationDeliveryResult:
        try:
            self.comfy.wait_for_prompt(
                prompt_id,
                timeout_seconds=self.timeout_seconds,
            )
        except ComfyHttpError as error:
            return self._resolve_delivery_prompt_error(
                job,
                scene,
                revision,
                delivery_marker,
                signature,
                prompt_id,
                error,
            )
        # Cancellation after the side effect leaves the queued marker intact.
        # If work is later resumed, successful history is reclaimed and no
        # second Discord send is queued.
        self._require_active(job, scene, revision)
        return self._commit_sent_delivery(
            job,
            scene,
            revision,
            delivery_marker,
            signature,
            prompt_id,
            reused_prompt=reused_prompt,
        )

    def _resolve_delivery_prompt_error(
        self,
        job: JobPayload,
        scene: SceneSpec,
        revision: int,
        delivery_marker: Path,
        signature: Mapping[str, Any],
        prompt_id: str,
        error: ComfyHttpError,
    ) -> ContinuationDeliveryResult:
        try:
            history = self.comfy.completed_prompt(prompt_id)
        except ComfyHttpError:
            history = None
        if history is not None:
            return self._commit_sent_delivery(
                job,
                scene,
                revision,
                delivery_marker,
                signature,
                prompt_id,
                reused_prompt=True,
            )
        try:
            still_queued = self.comfy.prompt_is_queued(prompt_id)
        except ComfyHttpError:
            still_queued = None
        if still_queued is not False:
            raise ContinuationDeliveryError(
                "Discord delivery wait failed with ambiguous ComfyUI state; the "
                "queued marker was preserved for restart-safe recovery."
            ) from error
        self._write_delivery_marker_best_effort(
            delivery_marker,
            signature,
            status="failed",
            prompt_id=prompt_id,
            failure_reason="prompt_failed",
        )
        raise ContinuationDeliveryError(
            "Discord delivery failed; raw scene output remains valid."
        ) from error

    def _commit_sent_delivery(
        self,
        job: JobPayload,
        scene: SceneSpec,
        revision: int,
        delivery_marker: Path,
        signature: Mapping[str, Any],
        prompt_id: str,
        *,
        reused_prompt: bool,
    ) -> ContinuationDeliveryResult:
        self._require_active(job, scene, revision)
        try:
            write_json_atomic(
                delivery_marker,
                self._delivery_marker_document(
                    signature,
                    status="sent",
                    prompt_id=prompt_id,
                ),
            )
        except Exception as error:
            raise ContinuationDeliveryError(
                "Discord delivery completed, but its sent marker could not be "
                "committed; the queued marker remains reclaimable."
            ) from error
        return ContinuationDeliveryResult(
            status="sent",
            prompt_id=prompt_id,
            reused_prompt=reused_prompt,
        )

    def _write_delivery_marker_best_effort(
        self,
        delivery_marker: Path,
        signature: Mapping[str, Any],
        *,
        status: str,
        prompt_id: str | None,
        failure_reason: str,
    ) -> None:
        try:
            write_json_atomic(
                delivery_marker,
                self._delivery_marker_document(
                    signature,
                    status=status,
                    prompt_id=prompt_id,
                    failure_reason=failure_reason,
                ),
            )
        except Exception:
            LOGGER.exception(
                "Could not persist Discord delivery marker state %s.",
                status,
            )

    def _cancel_owned_prompt_best_effort(self, prompt_id: str) -> None:
        try:
            cancel_owned = getattr(self.comfy, "cancel_owned_prompt", None)
            if callable(cancel_owned):
                cancel_owned(prompt_id)
            else:
                self.comfy.cancel_prompt(prompt_id)
        except Exception:
            LOGGER.exception(
                "Could not cancel project-owned ComfyUI prompt %s.",
                prompt_id,
            )
