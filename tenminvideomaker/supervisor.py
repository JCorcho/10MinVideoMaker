"""Unattended pipeline supervisor built on the shared project services."""

from __future__ import annotations

from dataclasses import dataclass
import gc
import logging
import os
from pathlib import Path
import time
from typing import Callable

from .artifacts import scene_clip_path, scene_frame_path
from .assembly import FfmpegAssembler, probe_video, validate_video_profile
from .assets import AssetResolution, LocalLoraRequirement, LoraAssetManager
from .comfy_http import ComfyHttpClient, ComfyHttpError, find_video_output
from .constants import MANDATORY_I2V_LORAS
from .contracts import JobPayload, LoraSpec, SceneSpec, effective_t2i_loras
from .mail import GmailClient, GmailPollingService
from .state_store import PipelineState, PipelineStateStore, SceneState
from .workflow_builder import build_i2v_api_workflow, build_t2i_api_workflow

LOGGER = logging.getLogger("10MinVideoMaker.supervisor")


class FatalPipelineError(RuntimeError):
    """Raised when a project job cannot continue without external recovery."""


@dataclass(frozen=True)
class SupervisorSettings:
    poll_interval_seconds: float = 300.0
    t2i_timeout_seconds: float = 3600.0
    i2v_timeout_seconds: float = 21600.0
    max_stage_attempts: int = 2

    @classmethod
    def from_environment(cls) -> "SupervisorSettings":
        return cls(
            poll_interval_seconds=float(os.environ.get("TENMIN_POLL_SECONDS", "300")),
            t2i_timeout_seconds=float(os.environ.get("TENMIN_T2I_TIMEOUT_SECONDS", "3600")),
            i2v_timeout_seconds=float(os.environ.get("TENMIN_I2V_TIMEOUT_SECONDS", "21600")),
            max_stage_attempts=int(os.environ.get("TENMIN_MAX_STAGE_ATTEMPTS", "2")),
        )

    def __post_init__(self) -> None:
        if self.poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive.")
        if self.t2i_timeout_seconds <= 0 or self.i2v_timeout_seconds <= 0:
            raise ValueError("stage timeouts must be positive.")
        if self.max_stage_attempts < 1:
            raise ValueError("max_stage_attempts must be at least 1.")


class PipelineSupervisor:
    """Owns polling, resumable scene execution, assembly, and the self-healing loop."""

    def __init__(
        self,
        *,
        store: PipelineStateStore,
        mail_client: GmailClient,
        asset_manager: LoraAssetManager,
        comfy: ComfyHttpClient,
        assembler: FfmpegAssembler | None = None,
        settings: SupervisorSettings | None = None,
        restart_comfy: Callable[[], bool] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        frame_path_factory: Callable[[str, int], Path] = scene_frame_path,
        clip_path_factory: Callable[[str, int], Path] = scene_clip_path,
        video_probe: Callable[[str | Path], object] = probe_video,
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

    def tick(self) -> None:
        """Advance the durable pipeline until it reaches a polling wait state."""
        snapshot = self.store.snapshot()
        if snapshot.state == PipelineState.IDLE:
            self._request_next_job(previous_job_id=None, succeeded=None)
            return
        if snapshot.state == PipelineState.WAITING_FOR_GROK:
            GmailPollingService(self.store, self.mail_client).poll_once()
            return
        if not snapshot.job_id:
            raise FatalPipelineError(f"Pipeline state {snapshot.state} has no active job id.")
        if snapshot.state == PipelineState.ERROR:
            self.store.requeue_unfinished_scenes(snapshot.job_id)
        self.process_job(self.store.load_job(snapshot.job_id))

    def run_forever(self) -> None:
        """Poll every configured interval; long-running generation remains synchronous."""
        LOGGER.info("10MinVideoMaker supervisor started.")
        while True:
            try:
                self.tick()
            except KeyboardInterrupt:
                LOGGER.info("10MinVideoMaker supervisor stopped.")
                return
            except FatalPipelineError as error:
                self._handle_fatal(error)
            except Exception:
                LOGGER.exception("Supervisor tick failed; durable state was preserved.")
            self._sleep(self.settings.poll_interval_seconds)

    def process_job(self, job: JobPayload) -> None:
        self.store.transition(PipelineState.DOWNLOADING_ASSETS, job_id=job.job_id)
        failures = self._resolve_assets(job)
        for scene_id, errors in failures.items():
            self.store.set_scene_state(
                job.job_id,
                scene_id,
                SceneState.FAILED,
                error="; ".join(errors),
            )

        scene_by_id = {scene.scene_id: scene for scene in job.scenes}
        for record in self.store.scene_records(job.job_id):
            scene = scene_by_id[record.scene_id]
            if record.state == SceneState.SUCCEEDED and record.video_path and Path(record.video_path).is_file():
                continue
            if record.state == SceneState.FAILED:
                continue
            self._process_scene_with_retries(job, scene)

        records = self.store.scene_records(job.job_id)
        successful = [
            record for record in records if record.state == SceneState.SUCCEEDED and record.video_path
        ]
        complete_success = len(successful) == len(records)
        if successful:
            self.store.transition(PipelineState.STITCHING, job_id=job.job_id)
            streams = [self._video_probe(record.video_path) for record in successful]
            validate_video_profile(streams)
            self.assembler.stitch(
                job.job_id,
                [record.video_path for record in successful],
                self.store.database_path.parent / "concat",
            )
        self._release_memory()
        self._request_next_job(previous_job_id=job.job_id, succeeded=complete_success)

    def _process_scene_with_retries(self, job: JobPayload, scene: SceneSpec) -> None:
        while True:
            try:
                self._process_scene(job, scene)
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

    def _process_scene(self, job: JobPayload, scene: SceneSpec) -> None:
        record = next(
            item for item in self.store.scene_records(job.job_id) if item.scene_id == scene.scene_id
        )
        frame_path = self._frame_path(job.job_id, scene.scene_id)
        if record.frame_path and Path(record.frame_path).is_file():
            frame_path = Path(record.frame_path)
        elif frame_path.is_file():
            self.store.set_scene_state(
                job.job_id,
                scene.scene_id,
                SceneState.PENDING,
                frame_path=str(frame_path),
            )
        else:
            if record.t2i_attempts >= self.settings.max_stage_attempts:
                raise ComfyHttpError("T2I attempt limit reached.")
            self.store.begin_scene_stage(job.job_id, scene.scene_id, PipelineState.RUNNING_T2I)
            build = build_t2i_api_workflow(job, scene)
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
            self._release_memory()

        record = next(
            item for item in self.store.scene_records(job.job_id) if item.scene_id == scene.scene_id
        )
        clip_path = self._clip_path(job.job_id, scene.scene_id)
        if record.video_path and Path(record.video_path).is_file():
            self.store.set_scene_state(job.job_id, scene.scene_id, SceneState.SUCCEEDED)
            return
        if clip_path.is_file():
            self.store.set_scene_state(
                job.job_id,
                scene.scene_id,
                SceneState.SUCCEEDED,
                video_path=str(clip_path),
            )
            return
        if record.i2v_attempts >= self.settings.max_stage_attempts:
            raise ComfyHttpError("I2V attempt limit reached.")

        self.store.begin_scene_stage(job.job_id, scene.scene_id, PipelineState.RUNNING_I2V)
        build = build_i2v_api_workflow(job, scene, frame_path)
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

    def _resolve_assets(self, job: JobPayload) -> dict[int, list[str]]:
        failures: dict[int, list[str]] = {scene.scene_id: [] for scene in job.scenes}
        cache: dict[str, AssetResolution] = {}

        def resolve(lora: LoraSpec) -> AssetResolution:
            key = lora.name.casefold()
            if key not in cache:
                cache[key] = self.asset_manager.resolve_or_download(lora)
            return cache[key]

        global_result = resolve(job.character.lora)
        if not global_result.succeeded:
            for errors in failures.values():
                errors.append(global_result.error or f"Missing {job.character.lora.name}")
        if job.ltxv_character_lora:
            ltx_character = resolve(job.ltxv_character_lora)
            if not ltx_character.succeeded:
                for errors in failures.values():
                    errors.append(
                        ltx_character.error or f"Missing {job.ltxv_character_lora.name}"
                    )
        for filename, weight in MANDATORY_I2V_LORAS:
            result = self.asset_manager.require_local(LocalLoraRequirement(filename, weight))
            if not result.succeeded:
                for errors in failures.values():
                    errors.append(result.error or f"Missing {filename}")

        for scene in job.scenes:
            scene_loras = (
                *effective_t2i_loras(scene, job.character),
                *scene.i2v.loras,
            )
            for lora in scene_loras:
                result = resolve(lora)
                if not result.succeeded:
                    failures[scene.scene_id].append(result.error or f"Missing {lora.name}")
        return {scene_id: errors for scene_id, errors in failures.items() if errors}

    def _request_next_job(self, *, previous_job_id: str | None, succeeded: bool | None) -> None:
        self.mail_client.send_request(previous_job_id=previous_job_id, succeeded=succeeded)
        self.store.transition(PipelineState.WAITING_FOR_GROK, job_id=previous_job_id)

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
            LOGGER.warning("ComfyUI restarted; unfinished scenes will resume.")
        else:
            LOGGER.error("Fatal pipeline error requires recovery: %s", error)
