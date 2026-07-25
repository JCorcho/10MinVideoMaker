"""Unattended pipeline supervisor built on the shared project services."""

from __future__ import annotations

from dataclasses import dataclass
import gc
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable, Mapping

from .artifacts import scene_clip_path, scene_frame_path
from .assembly import AssemblyError, FfmpegAssembler, probe_video, validate_video_profile
from .assets import AssetResolution, LocalLoraRequirement
from .comfy_http import ComfyHttpClient, ComfyHttpError, find_video_output
from .constants import I2V_DYNAMIC_BASE_MODEL, MANDATORY_I2V_LORAS
from .contracts import (
    JobPayload,
    LoraSpec,
    SceneSpec,
    effective_i2v_loras,
    effective_t2i_loras,
    lora_identity,
)
from .delivery import DiscordDeliverySettings
from .mail import GmailClient, GmailPollingService
from .state_store import PipelineState, PipelineStateStore, SceneState
from .workflow_builder import build_i2v_api_workflow, build_t2i_api_workflow

LOGGER = logging.getLogger("10MinVideoMaker.supervisor")


class FatalPipelineError(RuntimeError):
    """Raised when a project job cannot continue without external recovery."""


@dataclass(frozen=True)
class AssetPreparation:
    failures: dict[int, list[str]]
    resolved_filenames: dict[str, str]


@dataclass(frozen=True)
class SupervisorSettings:
    poll_interval_seconds: float = 300.0
    t2i_timeout_seconds: float = 3600.0
    i2v_timeout_seconds: float = 21600.0
    max_stage_attempts: int = 2
    status_interval_seconds: float = 15.0

    @classmethod
    def from_environment(cls) -> "SupervisorSettings":
        return cls(
            poll_interval_seconds=float(os.environ.get("TENMIN_POLL_SECONDS", "300")),
            t2i_timeout_seconds=float(os.environ.get("TENMIN_T2I_TIMEOUT_SECONDS", "3600")),
            i2v_timeout_seconds=float(os.environ.get("TENMIN_I2V_TIMEOUT_SECONDS", "21600")),
            max_stage_attempts=int(os.environ.get("TENMIN_MAX_STAGE_ATTEMPTS", "2")),
            status_interval_seconds=float(
                os.environ.get("TENMIN_STATUS_INTERVAL_SECONDS", "15")
            ),
        )

    def __post_init__(self) -> None:
        if self.poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive.")
        if self.t2i_timeout_seconds <= 0 or self.i2v_timeout_seconds <= 0:
            raise ValueError("stage timeouts must be positive.")
        if self.max_stage_attempts < 1:
            raise ValueError("max_stage_attempts must be at least 1.")
        if self.status_interval_seconds <= 0:
            raise ValueError("status_interval_seconds must be positive.")


class PipelineSupervisor:
    """Owns polling, resumable scene execution, assembly, and the self-healing loop."""

    def __init__(
        self,
        *,
        store: PipelineStateStore,
        mail_client: GmailClient,
        asset_manager: Any,
        comfy: ComfyHttpClient,
        assembler: FfmpegAssembler | None = None,
        settings: SupervisorSettings | None = None,
        restart_comfy: Callable[[], bool] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        frame_path_factory: Callable[[str, int], Path] = scene_frame_path,
        clip_path_factory: Callable[[str, int], Path] = scene_clip_path,
        video_probe: Callable[[str | Path], object] = probe_video,
        delivery: DiscordDeliverySettings | None = None,
    ):
        self.store = store
        self.mail_client = mail_client
        self.asset_manager = asset_manager
        self.comfy = comfy
        self.assembler = assembler or FfmpegAssembler()
        self.settings = settings or SupervisorSettings()
        self.restart_comfy = restart_comfy
        self._sleep = sleep
        self._frame_path = frame_path_factory
        self._clip_path = clip_path_factory
        self._video_probe = video_probe
        self.delivery = delivery

    def tick(self) -> None:
        """Advance the durable pipeline until it reaches a polling wait state."""
        snapshot = self.store.snapshot()
        LOGGER.debug(
            "Supervisor tick: state=%s job=%s scene=%s.",
            snapshot.state.value,
            snapshot.job_id or "none",
            snapshot.active_scene_id if snapshot.active_scene_id is not None else "none",
        )
        if snapshot.state == PipelineState.IDLE:
            self._request_next_job(previous_job_id=None, succeeded=None)
            return
        if snapshot.state == PipelineState.WAITING_FOR_GROK:
            LOGGER.info("Checking Gmail for an unread LTX_JOB_COMPLETE handoff.")
            payload = GmailPollingService(self.store, self.mail_client).poll_once()
            if payload:
                LOGGER.info(
                    "Accepted Gmail job %s with %s scene(s).",
                    payload.job_id,
                    len(payload.scenes),
                )
            else:
                LOGGER.info("No new valid job was found; remaining in waiting_for_grok.")
            return
        if not snapshot.job_id:
            raise FatalPipelineError(f"Pipeline state {snapshot.state} has no active job id.")
        if snapshot.state == PipelineState.ERROR:
            LOGGER.error(
                "Pipeline is paused in error for job %s: %s",
                snapshot.job_id,
                snapshot.error or "manual retry required",
            )
            return
        self.process_job(self.store.load_job(snapshot.job_id))

    def run_forever(self) -> None:
        """Poll every configured interval; long-running generation remains synchronous."""
        LOGGER.info(
            "10MinVideoMaker supervisor started; safe status heartbeat every %g seconds.",
            self.settings.status_interval_seconds,
        )
        stop_status = threading.Event()
        status_thread = threading.Thread(
            target=self._status_loop,
            args=(stop_status,),
            name="10MinVideoMaker-status",
            daemon=True,
        )
        status_thread.start()
        try:
            while True:
                try:
                    try:
                        self.tick()
                    except FatalPipelineError as error:
                        self._handle_fatal(error)
                    except Exception:
                        LOGGER.exception(
                            "Supervisor tick failed; durable state was preserved."
                        )
                    LOGGER.info(
                        "Next supervisor/mailbox check in %g seconds; heartbeat remains active.",
                        self.settings.poll_interval_seconds,
                    )
                    self._sleep(self.settings.poll_interval_seconds)
                except KeyboardInterrupt:
                    LOGGER.info("10MinVideoMaker supervisor stopped.")
                    return
        finally:
            stop_status.set()
            status_thread.join(timeout=1.0)

    def _status_loop(self, stop_status: threading.Event) -> None:
        self._log_status()
        while not stop_status.wait(self.settings.status_interval_seconds):
            self._log_status()

    def _log_status(self) -> None:
        """Log a redacted progress snapshot that is safe for the visible console."""
        try:
            snapshot = self.store.snapshot()
            try:
                running, pending = self.comfy.queue_counts()
                queue_text = f"running={running} pending={pending}"
            except (ComfyHttpError, AttributeError):
                queue_text = "unavailable"
            LOGGER.info(
                "STATUS | state=%s | job=%s | scene=%s | ComfyUI queue=%s",
                snapshot.state.value,
                snapshot.job_id or "none",
                snapshot.active_scene_id
                if snapshot.active_scene_id is not None
                else "none",
                queue_text,
            )
        except Exception:
            LOGGER.warning("STATUS | durable pipeline state could not be read.", exc_info=True)

    def process_job(self, job: JobPayload) -> None:
        LOGGER.info("Resolving assets for job %s.", job.job_id)
        self.store.transition(PipelineState.DOWNLOADING_ASSETS, job_id=job.job_id)
        preparation = self._resolve_assets(job)
        for scene_id, errors in preparation.failures.items():
            self.store.set_scene_state(
                job.job_id,
                scene_id,
                SceneState.FAILED,
                error="; ".join(errors),
            )
            LOGGER.error(
                "Job %s scene %s asset preparation failed: %s",
                job.job_id,
                scene_id,
                "; ".join(errors),
            )

        if len(preparation.failures) == len(job.scenes):
            error = (
                f"Asset preparation failed for all {len(job.scenes)} scene(s); "
                "correct the asset/authentication problem and retry the saved job."
            )
            self.store.transition(PipelineState.ERROR, job_id=job.job_id, error=error)
            LOGGER.error("%s Job %s remains saved and was not replaced.", error, job.job_id)
            return

        scene_by_id = {scene.scene_id: scene for scene in job.scenes}
        for record in self.store.scene_records(job.job_id):
            scene = scene_by_id[record.scene_id]
            if record.state == SceneState.SUCCEEDED and record.video_path and Path(record.video_path).is_file():
                continue
            if record.state == SceneState.FAILED:
                continue
            self._process_scene_with_retries(
                job,
                scene,
                preparation.resolved_filenames,
            )

        records = self.store.scene_records(job.job_id)
        successful = [
            record for record in records if record.state == SceneState.SUCCEEDED and record.video_path
        ]
        complete_success = len(successful) == len(records)
        if not successful:
            error = (
                f"Job {job.job_id} produced no successful scenes; "
                "the saved job is paused for manual retry."
            )
            self.store.transition(PipelineState.ERROR, job_id=job.job_id, error=error)
            LOGGER.error(error)
            self._release_memory()
            return
        self.store.transition(PipelineState.STITCHING, job_id=job.job_id)
        LOGGER.info(
            "Job %s: validating and stitching %s successful scene clip(s).",
            job.job_id,
            len(successful),
        )
        try:
            streams = [self._video_probe(record.video_path) for record in successful]
            validate_video_profile(streams)
            self.assembler.stitch(
                job.job_id,
                [record.video_path for record in successful],
                self.store.database_path.parent / "concat",
            )
        except AssemblyError as error:
            message = (
                f"Assembly failed for job {job.job_id}: {error} "
                "The saved job and completed scene clips were preserved."
            )
            self.store.transition(
                PipelineState.ERROR,
                job_id=job.job_id,
                error=message,
            )
            LOGGER.error(message)
            self._release_memory()
            return
        self._release_memory()
        self._request_next_job(previous_job_id=job.job_id, succeeded=complete_success)

    def _process_scene_with_retries(
        self,
        job: JobPayload,
        scene: SceneSpec,
        resolved_lora_filenames: Mapping[str, str],
    ) -> None:
        while True:
            try:
                self._process_scene(job, scene, resolved_lora_filenames)
                return
            except ComfyHttpError as error:
                if not self.comfy.alive():
                    raise FatalPipelineError(str(error)) from error
                record = next(
                    item
                    for item in self.store.scene_records(job.job_id)
                    if item.scene_id == scene.scene_id
                )
                attempts = record.i2v_attempts if record.frame_path else record.t2i_attempts
                if attempts < self.settings.max_stage_attempts:
                    LOGGER.warning(
                        "Retrying scene %s after attempt %s: %s",
                        scene.scene_id,
                        attempts,
                        error,
                    )
                    self.store.set_scene_state(
                        job.job_id,
                        scene.scene_id,
                        SceneState.PENDING,
                        error=str(error),
                    )
                    continue
                self.store.set_scene_state(
                    job.job_id,
                    scene.scene_id,
                    SceneState.FAILED,
                    error=str(error),
                )
                LOGGER.error("Scene %s exhausted its retry budget: %s", scene.scene_id, error)
                return
            except Exception as error:
                self.store.set_scene_state(
                    job.job_id,
                    scene.scene_id,
                    SceneState.FAILED,
                    error=str(error),
                )
                LOGGER.exception("Scene %s failed with a non-ComfyUI error.", scene.scene_id)
                return
            finally:
                self._release_memory()

    def _process_scene(
        self,
        job: JobPayload,
        scene: SceneSpec,
        resolved_lora_filenames: Mapping[str, str],
    ) -> None:
        record = next(
            item for item in self.store.scene_records(job.job_id) if item.scene_id == scene.scene_id
        )
        frame_path = self._frame_path(job.job_id, scene.scene_id)
        if record.frame_path and Path(record.frame_path).is_file():
            frame_path = Path(record.frame_path)
            LOGGER.info(
                "Job %s scene %s: reusing cached T2I frame.",
                job.job_id,
                scene.scene_id,
            )
        elif frame_path.is_file():
            self.store.set_scene_state(
                job.job_id,
                scene.scene_id,
                SceneState.PENDING,
                frame_path=str(frame_path),
            )
            LOGGER.info(
                "Job %s scene %s: recovered deterministic T2I frame from disk.",
                job.job_id,
                scene.scene_id,
            )
        else:
            if record.t2i_attempts >= self.settings.max_stage_attempts:
                raise ComfyHttpError("T2I attempt limit reached.")
            attempt = self.store.begin_scene_stage(
                job.job_id,
                scene.scene_id,
                PipelineState.RUNNING_T2I,
            )
            LOGGER.info(
                "Job %s scene %s: building and queueing T2I attempt %s.",
                job.job_id,
                scene.scene_id,
                attempt,
            )
            build = build_t2i_api_workflow(
                job,
                scene,
                resolved_lora_filenames,
                delivery=self.delivery,
            )
            prompt_id = self.comfy.queue_prompt(build.api)
            self.store.set_scene_prompt_id(job.job_id, scene.scene_id, prompt_id)
            self.comfy.wait_for_prompt(
                prompt_id,
                timeout_seconds=self.settings.t2i_timeout_seconds,
            )
            if not frame_path.is_file():
                raise ComfyHttpError(
                    f"T2I completed without deterministic frame {frame_path}."
                )
            self.store.set_scene_state(
                job.job_id,
                scene.scene_id,
                SceneState.PENDING,
                frame_path=str(frame_path),
            )
            LOGGER.info(
                "Job %s scene %s: T2I finished and cached the deterministic frame.",
                job.job_id,
                scene.scene_id,
            )
            self._release_memory()

        record = next(
            item for item in self.store.scene_records(job.job_id) if item.scene_id == scene.scene_id
        )
        clip_path = self._clip_path(job.job_id, scene.scene_id)
        if record.video_path and Path(record.video_path).is_file():
            self.store.set_scene_state(job.job_id, scene.scene_id, SceneState.SUCCEEDED)
            LOGGER.info(
                "Job %s scene %s: reusing completed scene clip.",
                job.job_id,
                scene.scene_id,
            )
            return
        if clip_path.is_file():
            self.store.set_scene_state(
                job.job_id,
                scene.scene_id,
                SceneState.SUCCEEDED,
                video_path=str(clip_path),
            )
            LOGGER.info(
                "Job %s scene %s: recovered deterministic scene clip from disk.",
                job.job_id,
                scene.scene_id,
            )
            return
        if record.i2v_attempts >= self.settings.max_stage_attempts:
            raise ComfyHttpError("I2V attempt limit reached.")

        attempt = self.store.begin_scene_stage(
            job.job_id,
            scene.scene_id,
            PipelineState.RUNNING_I2V,
        )
        LOGGER.info(
            "Job %s scene %s: building and queueing I2V attempt %s.",
            job.job_id,
            scene.scene_id,
            attempt,
        )
        build = build_i2v_api_workflow(
            job,
            scene,
            frame_path,
            resolved_lora_filenames,
            delivery=self.delivery,
        )
        prompt_id = self.comfy.queue_prompt(build.api)
        self.store.set_scene_prompt_id(job.job_id, scene.scene_id, prompt_id)
        history = self.comfy.wait_for_prompt(
            prompt_id,
            timeout_seconds=self.settings.i2v_timeout_seconds,
        )
        metadata = find_video_output(history)
        self.comfy.download_output(metadata, clip_path)
        self.store.set_scene_state(
            job.job_id,
            scene.scene_id,
            SceneState.SUCCEEDED,
            frame_path=str(frame_path),
            video_path=str(clip_path),
        )
        LOGGER.info(
            "Job %s scene %s: I2V finished and saved the deterministic clip.",
            job.job_id,
            scene.scene_id,
        )

    def _resolve_assets(self, job: JobPayload) -> AssetPreparation:
        failures: dict[int, dict[str, str]] = {
            scene.scene_id: {} for scene in job.scenes
        }
        cache: dict[tuple[str, str | None], AssetResolution] = {}
        resolved_filenames: dict[str, str] = {}

        def resolve(
            lora: LoraSpec,
            *,
            expected_base_model: str | None = None,
        ) -> AssetResolution:
            identity = lora_identity(lora)
            cache_key = (identity, expected_base_model)
            if cache_key not in cache:
                stage = "I2V LTX 2.x" if expected_base_model else "T2I"
                LOGGER.info("Asset check | stage=%s | LoRA=%s", stage, lora.name)
                cache[cache_key] = self.asset_manager.resolve_or_download(
                    lora,
                    expected_base_model=expected_base_model,
                )
                result = cache[cache_key]
                if result.succeeded and result.local_filename:
                    LOGGER.info(
                        "Asset ready | stage=%s | LoRA=%s | file=%s | source=%s",
                        stage,
                        lora.name,
                        result.local_filename,
                        "downloaded" if result.downloaded else "existing",
                    )
                    filename_key = (
                        f"i2v:{identity}"
                        if expected_base_model == I2V_DYNAMIC_BASE_MODEL
                        else identity
                    )
                    resolved_filenames[filename_key] = result.local_filename
                else:
                    LOGGER.warning(
                        "Asset failed | stage=%s | LoRA=%s | reason=%s",
                        stage,
                        lora.name,
                        result.error or "unknown asset error",
                    )
            else:
                LOGGER.debug(
                    "Asset cache hit | LoRA=%s | validation=%s",
                    lora.name,
                    expected_base_model or "T2I",
                )
            return cache[cache_key]

        def record_failure(scene_id: int, key: str, message: str) -> None:
            failures[scene_id][key] = message

        global_result = resolve(job.character.lora)
        if not global_result.succeeded:
            key = lora_identity(job.character.lora)
            for scene in job.scenes:
                record_failure(
                    scene.scene_id,
                    key,
                    global_result.error or f"Missing {job.character.lora.name}",
                )
        for filename, weight in MANDATORY_I2V_LORAS:
            LOGGER.info("Asset check | stage=I2V mandatory | LoRA=%s", filename)
            result = self.asset_manager.require_local(LocalLoraRequirement(filename, weight))
            if not result.succeeded:
                LOGGER.warning(
                    "Asset failed | stage=I2V mandatory | LoRA=%s | reason=%s",
                    filename,
                    result.error or "required local file is missing",
                )
                for scene in job.scenes:
                    record_failure(
                        scene.scene_id,
                        f"required:{filename.casefold()}",
                        result.error or f"Missing {filename}",
                    )
            else:
                LOGGER.info(
                    "Asset ready | stage=I2V mandatory | LoRA=%s | file=%s",
                    filename,
                    result.local_filename or filename,
                )

        for scene in job.scenes:
            for lora in effective_t2i_loras(scene, job.character):
                result = resolve(lora)
                if not result.succeeded:
                    record_failure(
                        scene.scene_id,
                        lora_identity(lora),
                        result.error or f"Missing {lora.name}",
                    )
            for lora in effective_i2v_loras(job, scene):
                result = resolve(
                    lora,
                    expected_base_model=I2V_DYNAMIC_BASE_MODEL,
                )
                if not result.succeeded:
                    record_failure(
                        scene.scene_id,
                        f"i2v:{lora_identity(lora)}",
                        result.error or f"Missing {lora.name}",
                    )
        return AssetPreparation(
            failures={
                scene_id: list(errors.values())
                for scene_id, errors in failures.items()
                if errors
            },
            resolved_filenames=resolved_filenames,
        )

    def _request_next_job(self, *, previous_job_id: str | None, succeeded: bool | None) -> None:
        self.mail_client.send_request(previous_job_id=previous_job_id, succeeded=succeeded)
        self.store.transition(PipelineState.WAITING_FOR_GROK, job_id=previous_job_id)
        LOGGER.info(
            "Requested the next Gmail job%s.",
            f" after {previous_job_id}" if previous_job_id else "",
        )

    def _release_memory(self) -> None:
        gc.collect()
        try:
            self.comfy.free_memory()
        except ComfyHttpError:
            LOGGER.warning("ComfyUI memory release request failed.", exc_info=True)

    def _handle_fatal(self, error: FatalPipelineError) -> None:
        snapshot = self.store.snapshot()
        self.store.transition(
            PipelineState.ERROR,
            job_id=snapshot.job_id,
            active_scene_id=snapshot.active_scene_id,
            error=str(error),
        )
        if self.restart_comfy and self.restart_comfy():
            if snapshot.job_id:
                self.store.requeue_unfinished_scenes(snapshot.job_id)
                self.store.transition(
                    PipelineState.DOWNLOADING_ASSETS,
                    job_id=snapshot.job_id,
                )
            LOGGER.warning("ComfyUI restarted; unfinished scenes will resume.")
        else:
            LOGGER.error("Fatal pipeline error requires recovery: %s", error)
