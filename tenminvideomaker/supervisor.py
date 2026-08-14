"""Unattended pipeline supervisor built on the shared project services."""

from __future__ import annotations

from dataclasses import dataclass
import gc
import hashlib
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable, Mapping, Sequence

from .artifacts import scene_clip_path, scene_frame_path
from .assembly import AssemblyError, FfmpegAssembler, probe_video, validate_video_profile
from .assets import AssetResolution, LocalLoraRequirement
from .chunk_assembly import SceneChunkAssembler
from .comfy_http import ComfyHttpClient, ComfyHttpError, find_video_output
from .constants import I2V_DYNAMIC_BASE_MODEL, MANDATORY_I2V_LORAS
from .continuation import (
    ContinuationPlanError,
    continuation_is_enabled,
    timeline_frame_count,
)
from .continuation_renderer import (
    ContinuationDeliveryError,
    ContinuationRenderError,
    ContinuationRenderer,
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
from .mail import GmailClient, GmailPollingService
from .qc_repair import RepairGenerationError
from .review import SceneWorkflowOverrides, scene_review_document, validate_scene_edit
from .state_store import (
    JobState,
    ManualFinalSceneSelection,
    PipelineState,
    PipelineStateStore,
    QcCandidateRecord,
    SceneState,
)
from .workflow_builder import build_i2v_api_workflow, build_t2i_api_workflow
from .storage import StorageLayout, write_json_atomic

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
    require_human_review: bool = False
    continuation_mode: str = "explicit"

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
            require_human_review=os.environ.get(
                "TENMIN_REQUIRE_HUMAN_REVIEW",
                "false",
            ).strip().casefold()
            in {"1", "true", "yes", "on"},
            continuation_mode=os.environ.get(
                "TENMIN_LTX_CONTINUATION_MODE",
                "explicit",
            ).strip().casefold(),
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
        if self.continuation_mode not in {"disabled", "explicit", "auto"}:
            raise ValueError(
                "continuation_mode must be disabled, explicit, or auto."
            )


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
        storage: StorageLayout | None = None,
        chunk_assembler: SceneChunkAssembler | None = None,
        qc_controller: Any | None = None,
        qc_finalization_checkpoint_hook: Callable[[str], None] | None = None,
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
        self.storage = storage
        self.chunk_assembler = chunk_assembler or SceneChunkAssembler()
        self.qc_controller = qc_controller
        self._qc_finalization_checkpoint_hook = (
            qc_finalization_checkpoint_hook or (lambda _point: None)
        )
        self.continuation_renderer = (
            ContinuationRenderer(
                store=self.store,
                storage=self.storage,
                comfy=self.comfy,
                assembler=self.chunk_assembler,
                timeout_seconds=self.settings.i2v_timeout_seconds,
                max_attempts=self.settings.max_stage_attempts,
                webhook_url=(
                    self.delivery.webhook_url if self.delivery is not None else None
                ),
            )
            if self.storage is not None
            else None
        )

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
            # After cancel/abandon, prefer an already-waiting handoff before
            # sending another Grok request. Normal job completion uses
            # _request_next_job directly and does not pass through idle.
            LOGGER.info(
                "Checking Gmail for an unread LTX_JOB_COMPLETE handoff before "
                "requesting another job."
            )
            payload = GmailPollingService(self.store, self.mail_client).poll_once(
                review_required=self.settings.require_human_review,
            )
            if payload:
                LOGGER.info(
                    "Accepted Gmail job %s with %s scene(s).",
                    payload.job_id,
                    len(payload.scenes),
                )
            else:
                LOGGER.info(
                    "No pending handoff was found; requesting the next Gmail job."
                )
                self._request_next_job(previous_job_id=None, succeeded=None)
            return
        if snapshot.state == PipelineState.WAITING_FOR_GROK:
            LOGGER.info("Checking Gmail for an unread LTX_JOB_COMPLETE handoff.")
            payload = GmailPollingService(self.store, self.mail_client).poll_once(
                review_required=self.settings.require_human_review,
            )
            if payload:
                LOGGER.info(
                    "Accepted Gmail job %s with %s scene(s).",
                    payload.job_id,
                    len(payload.scenes),
                )
            else:
                LOGGER.info("No new valid job was found; remaining in waiting_for_grok.")
            return
        if snapshot.state == PipelineState.AWAITING_REVIEW:
            LOGGER.info(
                "Job %s is awaiting human review in the supervisor GUI.",
                snapshot.job_id or "unknown",
            )
            return
        if snapshot.state == PipelineState.QC_BLOCKED:
            LOGGER.debug(
                "Job %s is on a durable QC hold: %s",
                snapshot.job_id or "unknown",
                snapshot.error or "incomplete pre-QC scene set",
            )
            return
        if snapshot.state == PipelineState.STITCHING and snapshot.job_id:
            plan = self.store.qc_finalization_plan(snapshot.job_id)
            if plan is not None:
                if (
                    self.qc_controller is not None
                    and not self.qc_controller.settings.quality_control_enabled
                ):
                    job = self.store.load_job(snapshot.job_id)
                    steps = self.store.qc_finalization_steps(snapshot.job_id)
                    if steps:
                        message = (
                            "QC was disabled after finalization side effects may have "
                            "started. The accepted QC plan will not be resumed; manual "
                            "reconciliation is required before restoring the immutable "
                            "pre-QC baseline."
                        )
                        self.store.transition(
                            PipelineState.QC_BLOCKED,
                            job_id=snapshot.job_id,
                            error=message,
                        )
                        LOGGER.error("Job %s: %s", snapshot.job_id, message)
                        return
                    baseline = self.store.original_final_selection(
                        snapshot.job_id,
                        [scene.scene_id for scene in job.scenes],
                    )
                    self._finalize_qc_disabled_selection(job, baseline)
                    return
                self._finalize_qc_job(
                    self.store.load_job(snapshot.job_id),
                    (),
                )
                return
        if snapshot.state in {
            PipelineState.RUNNING_QC,
            PipelineState.AWAITING_QC_REVIEW,
        }:
            if not snapshot.job_id:
                raise FatalPipelineError(
                    f"Pipeline state {snapshot.state} has no active job id."
                )
            if self.qc_controller is None:
                raise FatalPipelineError(
                    "A durable QC job cannot resume without the Phase-1 controller."
                )
            job = self.store.load_job(snapshot.job_id)
            result = self.qc_controller.run_epoch(job, self)
            if result.ready_for_finalization:
                self._finalize_qc_result(job, result.selection)
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
                        self.handle_fatal(error)
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
            self.shutdown()

    def shutdown(self) -> None:
        """Release only resources explicitly owned by this supervisor."""
        if self.qc_controller is not None:
            self.qc_controller.close()

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
        self.store.set_job_status(job.job_id, JobState.RUNNING)
        existing_records = {
            record.scene_id: record for record in self.store.scene_records(job.job_id)
        }
        scene_by_id = {scene.scene_id: scene for scene in job.scenes}
        for scene in job.scenes:
            record = existing_records[scene.scene_id]
            document = scene_review_document(job, scene)
            self.store.ensure_original_scene_revision(
                job.job_id,
                scene.scene_id,
                parameters=document,
                frame_path=record.frame_path,
                video_path=record.video_path,
            )
            if self.storage is not None:
                write_json_atomic(
                    self.storage.generation_manifest_path(
                        job.job_id,
                        scene.scene_id,
                        1,
                    ),
                    {
                        "job_id": job.job_id,
                        "scene_id": scene.scene_id,
                        "revision": 1,
                        "remake_mode": "image_and_video",
                        "parameters": document,
                        "frame_path": record.frame_path,
                        "video_path": record.video_path,
                        "status": record.state.value,
                    },
                )

        eligible_scene_ids = {scene.scene_id for scene in job.scenes}
        for scene in job.scenes:
            try:
                if self._automatic_uses_continuation(job.job_id, scene):
                    ContinuationRenderer.build_plan(job, scene, 1, None)
            except (ContinuationPlanError, ContinuationRenderError, ValueError) as error:
                message = f"Continuation preflight failed: {error}"
                eligible_scene_ids.discard(scene.scene_id)
                self.store.set_scene_state(
                    job.job_id,
                    scene.scene_id,
                    SceneState.FAILED,
                    error=message,
                )
                self._update_original_revision(
                    job,
                    scene,
                    state=SceneState.FAILED,
                    error=message,
                )
                LOGGER.error(
                    "Job %s scene %s rejected before assets or rendering: %s",
                    job.job_id,
                    scene.scene_id,
                    message,
                )

        if not eligible_scene_ids:
            error = (
                f"Continuation preflight failed for all {len(job.scenes)} scene(s); "
                "fix the saved scene timing/continuation data and retry the job."
            )
            self.store.transition(PipelineState.ERROR, job_id=job.job_id, error=error)
            self.store.set_job_status(job.job_id, JobState.FAILED)
            LOGGER.error("%s Job %s remains saved.", error, job.job_id)
            return

        LOGGER.info("Resolving assets for job %s.", job.job_id)
        self.store.transition(PipelineState.DOWNLOADING_ASSETS, job_id=job.job_id)
        preparation = self._resolve_assets(job, scene_ids=eligible_scene_ids)
        if self.store.snapshot().job_id != job.job_id:
            LOGGER.info(
                "Job %s asset resolution finished after a controlled cancellation; "
                "no scene was queued.",
                job.job_id,
            )
            return
        for scene_id, errors in preparation.failures.items():
            self.store.set_scene_state(
                job.job_id,
                scene_id,
                SceneState.FAILED,
                error="; ".join(errors),
            )
            self._update_original_revision(
                job,
                scene_by_id[scene_id],
                state=SceneState.FAILED,
                error="; ".join(errors),
            )
            LOGGER.error(
                "Job %s scene %s asset preparation failed: %s",
                job.job_id,
                scene_id,
                "; ".join(errors),
            )

        if len(preparation.failures) == len(eligible_scene_ids):
            error = (
                f"Asset preparation failed for all {len(eligible_scene_ids)} "
                "eligible scene(s); "
                "correct the asset/authentication problem and retry the saved job."
            )
            self.store.transition(PipelineState.ERROR, job_id=job.job_id, error=error)
            self.store.set_job_status(job.job_id, JobState.FAILED)
            LOGGER.error("%s Job %s remains saved and was not replaced.", error, job.job_id)
            return

        try:
            if not self._process_t2i_batch(
                job,
                scene_by_id,
                preparation.resolved_filenames,
            ):
                return
        finally:
            # Keep the image model resident until every required frame is complete,
            # then deliberately unload it before the LTX stage.
            self._release_memory()

        try:
            if not self._process_i2v_batch(
                job,
                scene_by_id,
                preparation.resolved_filenames,
            ):
                return
        finally:
            # Do not unload LTX between clips. Release it only after this job's
            # complete video stage has finished.
            self._release_memory()

        if (
            self.qc_controller is not None
            and self.qc_controller.settings.quality_control_enabled
        ):
            originals = self.qc_controller.register_original_candidates(job)
            if originals == ():
                return
            self.store.transition(PipelineState.RUNNING_QC, job_id=job.job_id)
            result = self.qc_controller.run_epoch(job, self)
            if result.ready_for_finalization:
                self._finalize_qc_result(job, result.selection)
            return

        try:
            if not self._deliver_i2v_batch(job, scene_by_id):
                return
        finally:
            # Delivery reloads raw scenes without the LTX model resident. Clear
            # each decoded/watermarked frame batch before final concatenation.
            self._release_memory()

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
            self.store.set_job_status(job.job_id, JobState.FAILED)
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
            assembled_path = self.assembler.stitch(
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
            self.store.set_job_status(job.job_id, JobState.FAILED)
            LOGGER.error(message)
            self._release_memory()
            return
        self.store.set_job_status(
            job.job_id,
            JobState.SUCCEEDED if complete_success else JobState.PARTIAL,
            final_path=str(assembled_path),
        )
        self._release_memory()
        self._request_next_job(previous_job_id=job.job_id, succeeded=complete_success)

    def render_qc_candidates(
        self,
        job: JobPayload,
        candidates: Sequence[tuple[QcCandidateRecord, Mapping[str, Any]]],
    ) -> None:
        """Render a whole repair batch while ComfyUI owns the generation epoch."""
        if not candidates:
            return
        if not self.comfy.alive():
            raise RepairGenerationError(
                candidates[0][0].candidate_id,
                "ComfyUI is unavailable at the QC repair generation boundary.",
                retryable=True,
            )
        scene_ids = {candidate.scene_id for candidate, _ in candidates}
        preparation = self._resolve_assets(job, scene_ids=scene_ids)
        if preparation.failures:
            raise RepairGenerationError(
                candidates[0][0].candidate_id,
                "QC repair asset resolution failed: "
                + "; ".join(
                    error
                    for errors in preparation.failures.values()
                    for error in errors
                ),
                retryable=False,
            )
        for candidate, document in candidates:
            revision = next(
                (
                    item
                    for item in self.store.scene_revisions(job.job_id, candidate.scene_id)
                    if item.revision == candidate.revision
                ),
                None,
            )
            if revision is None:
                raise RepairGenerationError(
                    candidate.candidate_id,
                    "QC repair lost its persisted scene revision.",
                    retryable=False,
                )
            if not revision.frame_path or not Path(revision.frame_path).is_file():
                raise RepairGenerationError(
                    candidate.candidate_id,
                    "QC repair lost its immutable starting frame.",
                    retryable=False,
                )
            try:
                validated = validate_scene_edit(job, candidate.scene_id, document)
            except Exception as error:
                raise RepairGenerationError(
                    candidate.candidate_id,
                    "QC repair revision document is invalid: " + str(error),
                    retryable=False,
                ) from error
            use_continuation = self._uses_continuation(
                validated.scene, validated.workflow
            )
            route = "continuation" if use_continuation else "legacy"
            destination = Path(candidate.source_video_path)
            if (
                candidate.state == candidate.state.GENERATING
                and destination.is_file()
            ):
                self._video_probe(destination)
                self.store.complete_qc_candidate_generation(
                    candidate.candidate_id,
                    source_video_path=str(destination),
                    source_video_sha256=self._sha256_path(destination),
                )
                continue
            self.store.update_scene_revision(
                job.job_id,
                candidate.scene_id,
                candidate.revision,
                state=SceneState.RUNNING,
            )
            self.store.set_qc_generation_owner(
                candidate.candidate_id,
                prompt_id=candidate.generation_prompt_id,
                prompt_stage=(
                    "i2v_continuation" if use_continuation else "i2v_legacy"
                ),
                route=route,
            )
            try:
                rendered = self.render_i2v_scene(
                    job=validated.job,
                    scene=validated.scene,
                    frame_path=revision.frame_path,
                    destination=destination,
                    resolved_lora_filenames=preparation.resolved_filenames,
                    revision=candidate.revision,
                    overrides=validated.workflow,
                    prompt_id_callback=lambda prompt_id, candidate_id=candidate.candidate_id,
                        route=route, use_continuation=use_continuation: (
                        self.store.set_qc_generation_owner(
                            candidate_id,
                            prompt_id=prompt_id,
                            prompt_stage=(
                                "i2v_continuation" if use_continuation else "i2v_legacy"
                            ),
                            route=route,
                        )
                    ),
                    existing_prompt_id=(
                        candidate.generation_prompt_id if not use_continuation else None
                    ),
                    deliver_to_discord=False,
                    continuation_route=use_continuation,
                )
                self._video_probe(rendered)
            except RepairGenerationError:
                raise
            except Exception as error:
                raise RepairGenerationError(
                    candidate.candidate_id,
                    "QC repair render failed: " + str(error),
                    retryable=True,
                ) from error
            self.store.complete_qc_candidate_generation(
                candidate.candidate_id,
                source_video_path=str(rendered),
                source_video_sha256=self._sha256_path(rendered),
            )

    @staticmethod
    def _sha256_path(path: str | Path) -> str:
        import hashlib

        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _finalize_qc_job(
        self,
        job: JobPayload,
        selection: Sequence[ManualFinalSceneSelection],
    ) -> None:
        """Resume the accepted-selection plan without replaying external effects."""
        required = tuple(sorted(scene.scene_id for scene in job.scenes))
        existing_plan = self.store.qc_finalization_plan(job.job_id)
        if existing_plan is None:
            if tuple(item.scene_id for item in selection) != required:
                raise FatalPipelineError(
                    "QC finalization selection does not contain every required scene."
                )
            try:
                final_path = str(self.assembler.final_path(job.job_id))
            except (AttributeError, AssemblyError) as error:
                raise FatalPipelineError(
                    "QC finalization requires a deterministic final artifact path."
                ) from error
        else:
            final_path = existing_plan.final_path
        plan = self.store.ensure_qc_finalization_plan(
            job.job_id,
            required,
            final_path=final_path,
        )
        planned_selection = tuple(
            ManualFinalSceneSelection(
                int(item["scene_id"]),
                int(item["revision"]),
                str(item["artifact_path"]),
            )
            for item in plan.selection
        )
        if selection and tuple(selection) != planned_selection:
            raise FatalPipelineError(
                "QC finalization selection differs from its durable committed plan."
            )
        if plan.state == "COMPLETED":
            self.store.transition(PipelineState.WAITING_FOR_GROK, job_id=job.job_id)
            return
        self.store.transition(PipelineState.STITCHING, job_id=job.job_id)
        self._qc_finalization_checkpoint_hook("plan_committed")
        self.store.advance_qc_finalization_plan(job.job_id, "DELIVERING")
        for item, selected in zip(plan.selection, planned_selection, strict=True):
            step_key = f"deliver-scene-{selected.scene_id}"
            operation_id = hashlib.sha256(
                (
                    f"{plan.plan_sha256}|SCENE_DELIVERY|{selected.scene_id}|"
                    f"{selected.revision}|{item['artifact_sha256']}"
                ).encode("utf-8")
            ).hexdigest()
            step = self.store.begin_qc_finalization_step(
                job.job_id,
                step_key,
                kind="SCENE_DELIVERY",
                evidence={
                    "candidate_id": item["candidate_id"],
                    "scene_id": selected.scene_id,
                    "revision": selected.revision,
                    "artifact_path": selected.video_path,
                    "artifact_sha256": item["artifact_sha256"],
                    "operation_id": operation_id,
                },
            )
            if step.state == "COMPLETED":
                if (step.receipt or {}).get("status") not in {
                    "CONFIRMED",
                    "NOT_CONFIGURED",
                }:
                    raise FatalPipelineError(
                        "Completed scene delivery lacks an authoritative receipt."
                    )
                continue
            if step.state in {"AMBIGUOUS", "FAILED"}:
                message = (
                    f"Scene {selected.scene_id} delivery is {step.state}; "
                    "manual reconciliation is required before finalization."
                )
                self.store.transition(
                    PipelineState.QC_BLOCKED, job_id=job.job_id, error=message
                )
                return
            if self.delivery is None:
                self.store.complete_qc_finalization_step(
                    job.job_id,
                    step_key,
                    receipt={
                        "status": "NOT_CONFIGURED",
                        "operation_id": operation_id,
                    },
                )
                continue
            step = self.store.mark_qc_finalization_step_dispatching(
                job.job_id,
                step_key,
                receipt={"operation_id": operation_id},
            )
            prompt_ids = (step.receipt or {}).get("prompt_ids") or []
            existing_prompt_id = prompt_ids[-1] if prompt_ids else None
            candidate_revision = next(
                revision
                for revision in self.store.scene_revisions(
                    job.job_id, selected.scene_id
                )
                if revision.revision == selected.revision
            )
            validated = validate_scene_edit(
                job, selected.scene_id, candidate_revision.parameters
            )
            try:
                delivery_result = self.deliver_scene_video(
                    job=validated.job,
                    scene=validated.scene,
                    scene_path=selected.video_path,
                    revision=selected.revision,
                    overrides=validated.workflow,
                    existing_prompt_id=(
                        str(existing_prompt_id) if existing_prompt_id else None
                    ),
                    require_authoritative=True,
                    prompt_id_callback=lambda prompt_id, step_key=step_key: (
                        self.store.bind_qc_finalization_step_prompt_id(
                            job.job_id,
                            step_key,
                            prompt_id,
                        )
                    ),
                )
            except ContinuationDeliveryError as error:
                self.store.set_qc_finalization_step_dispatch_state(
                    job.job_id,
                    step_key,
                    state=error.state,
                    receipt={
                        "operation_id": operation_id,
                        "failure_reason": str(error),
                    },
                )
                message = (
                    f"Scene {selected.scene_id} delivery is {error.state}; "
                    "assembly and mail are blocked pending manual reconciliation."
                )
                self.store.transition(
                    PipelineState.QC_BLOCKED, job_id=job.job_id, error=message
                )
                self._release_memory()
                return
            self._release_memory()
            if not delivery_result or not getattr(delivery_result, "prompt_id", None):
                self.store.set_qc_finalization_step_dispatch_state(
                    job.job_id,
                    step_key,
                    state="FAILED",
                    receipt={
                        "operation_id": operation_id,
                        "failure_reason": "missing_authoritative_delivery_receipt",
                    },
                )
                self.store.transition(
                    PipelineState.QC_BLOCKED,
                    job_id=job.job_id,
                    error="Configured scene delivery returned no authoritative receipt.",
                )
                return
            self._qc_finalization_checkpoint_hook(
                f"after_scene_delivery:{selected.scene_id}"
            )
            self.store.complete_qc_finalization_step(
                job.job_id,
                step_key,
                receipt={
                    "status": "CONFIRMED",
                    "operation_id": operation_id,
                    "prompt_id": getattr(delivery_result, "prompt_id", None),
                    "reused_prompt": getattr(delivery_result, "reused_prompt", None),
                    "delivery_marker_owned_by": "continuation_renderer",
                },
            )
        self.store.advance_qc_finalization_plan(job.job_id, "DELIVERED")
        self._qc_finalization_checkpoint_hook("deliveries_completed")

        stitch_step = self.store.begin_qc_finalization_step(
            job.job_id,
            "assemble-final",
            kind="ASSEMBLY",
            evidence={
                "plan_sha256": plan.plan_sha256,
                "ordered_artifact_sha256": [
                    item["artifact_sha256"] for item in plan.selection
                ],
                "final_path": plan.final_path,
            },
        )
        self.store.advance_qc_finalization_plan(job.job_id, "STITCHING")
        try:
            if stitch_step.state == "COMPLETED":
                receipt = stitch_step.receipt or {}
                assembled_path = Path(str(receipt["final_path"]))
                if (
                    not assembled_path.is_file()
                    or self._sha256_path(assembled_path) != receipt["final_sha256"]
                ):
                    raise AssemblyError(
                        "Durably checkpointed QC final artifact changed."
                    )
            else:
                streams = [
                    self._video_probe(item.video_path) for item in planned_selection
                ]
                validate_video_profile(streams)
                assembled_path = self.assembler.stitch_qc_plan(
                    job.job_id,
                    [item.video_path for item in planned_selection],
                    self.store.database_path.parent / "concat",
                    checkpoint_directory=(
                        self.store.database_path.parent / "qc-finalization"
                    ),
                    plan_sha256=plan.plan_sha256,
                )
                self._qc_finalization_checkpoint_hook("after_stitch_artifact")
                final_sha256 = self._sha256_path(assembled_path)
                stitch_step = self.store.complete_qc_finalization_step(
                    job.job_id,
                    "assemble-final",
                    receipt={
                        "final_path": str(Path(assembled_path).resolve()),
                        "final_sha256": final_sha256,
                        "plan_sha256": plan.plan_sha256,
                    },
                )
        except AssemblyError as error:
            message = f"Assembly failed for QC job {job.job_id}: {error}"
            self.store.transition(PipelineState.ERROR, job_id=job.job_id, error=message)
            self.store.set_job_status(job.job_id, JobState.FAILED)
            self._release_memory()
            return
        final_sha256 = str((stitch_step.receipt or {})["final_sha256"])
        plan = self.store.advance_qc_finalization_plan(
            job.job_id,
            "STITCHED",
            final_sha256=final_sha256,
        )
        self.store.set_job_status(
            job.job_id, JobState.SUCCEEDED, final_path=str(assembled_path)
        )
        plan = self.store.advance_qc_finalization_plan(
            job.job_id,
            "JOB_COMMITTED",
            final_sha256=final_sha256,
        )
        self._release_memory()
        self._qc_finalization_checkpoint_hook("after_job_success")

        request_id = f"qc-finalization-{plan.plan_sha256}"
        request_step = self.store.begin_qc_finalization_step(
            job.job_id,
            "request-next-job",
            kind="NEXT_JOB_REQUEST",
            evidence={
                "request_id": request_id,
                "previous_job_id": job.job_id,
                "succeeded": True,
            },
        )
        plan = self.store.advance_qc_finalization_plan(
            job.job_id,
            "NEXT_REQUEST_INTENT",
            next_request_id=request_id,
        )
        if request_step.state != "COMPLETED":
            already_sent = self.mail_client.request_was_sent(request_id)
            if already_sent:
                message_id = self.mail_client.request_message_id(request_id)
            elif request_step.state == "DISPATCHING":
                LOGGER.warning(
                    "QC next-job request %s is dispatch-ambiguous and is not yet "
                    "observable in Gmail; refusing an automatic resend.",
                    request_id,
                )
                return
            else:
                self.store.mark_qc_finalization_step_dispatching(
                    job.job_id,
                    "request-next-job",
                )
                message_id = self.mail_client.send_request(
                    previous_job_id=job.job_id,
                    succeeded=True,
                    request_id=request_id,
                )
                self._qc_finalization_checkpoint_hook("after_next_request")
            request_step = self.store.complete_qc_finalization_step(
                job.job_id,
                "request-next-job",
                receipt={
                    "request_id": request_id,
                    "message_id": message_id,
                },
            )
        message_id = str((request_step.receipt or {})["message_id"])
        self.store.advance_qc_finalization_plan(
            job.job_id,
            "COMPLETED",
            final_sha256=final_sha256,
            next_request_id=request_id,
            next_request_receipt=message_id,
        )
        self.store.transition(PipelineState.WAITING_FOR_GROK, job_id=job.job_id)

    def _finalize_qc_result(
        self,
        job: JobPayload,
        selection: Sequence[ManualFinalSceneSelection],
    ) -> None:
        if (
            self.qc_controller is not None
            and not self.qc_controller.settings.quality_control_enabled
        ):
            self._finalize_qc_disabled_selection(job, selection)
            return
        self._finalize_qc_job(job, selection)

    def _finalize_qc_disabled_selection(
        self,
        job: JobPayload,
        selection: Sequence[ManualFinalSceneSelection],
    ) -> None:
        """Preserve the pre-QC finalization path for the snapshotted baseline."""
        required = tuple(sorted(scene.scene_id for scene in job.scenes))
        if tuple(item.scene_id for item in selection) != required:
            raise FatalPipelineError(
                "QC kill-switch selection does not contain every required scene."
            )
        if self.delivery is not None:
            for item in selection:
                candidate_revision = next(
                    revision
                    for revision in self.store.scene_revisions(
                        job.job_id, item.scene_id
                    )
                    if revision.revision == item.revision
                )
                validated = validate_scene_edit(
                    job, item.scene_id, candidate_revision.parameters
                )
                self.deliver_scene_video(
                    job=validated.job,
                    scene=validated.scene,
                    scene_path=item.video_path,
                    revision=item.revision,
                    overrides=validated.workflow,
                )
                self._release_memory()
        self.store.transition(PipelineState.STITCHING, job_id=job.job_id)
        try:
            streams = [self._video_probe(item.video_path) for item in selection]
            validate_video_profile(streams)
            assembled_path = self.assembler.stitch(
                job.job_id,
                [item.video_path for item in selection],
                self.store.database_path.parent / "concat",
            )
        except AssemblyError as error:
            message = f"Assembly failed for kill-switched QC job {job.job_id}: {error}"
            self.store.transition(PipelineState.ERROR, job_id=job.job_id, error=message)
            self.store.set_job_status(job.job_id, JobState.FAILED)
            self._release_memory()
            return
        self.store.set_job_status(
            job.job_id,
            JobState.SUCCEEDED,
            final_path=str(assembled_path),
        )
        self._release_memory()
        self._request_next_job(previous_job_id=job.job_id, succeeded=True)

    def _process_t2i_batch(
        self,
        job: JobPayload,
        scene_by_id: Mapping[int, SceneSpec],
        resolved_lora_filenames: Mapping[str, str],
    ) -> Any:
        LOGGER.info(
            "Job %s: generating all required T2I frames before loading LTX.",
            job.job_id,
        )
        for record in self.store.scene_records(job.job_id):
            if record.state in {SceneState.FAILED, SceneState.CANCELLED}:
                continue
            if (
                record.state == SceneState.SUCCEEDED
                and record.video_path
                and Path(record.video_path).is_file()
            ):
                continue
            scene = scene_by_id[record.scene_id]
            self._process_scene_stage_with_retries(
                job,
                scene,
                PipelineState.RUNNING_T2I,
                resolved_lora_filenames,
            )
            if self.store.snapshot().job_id != job.job_id:
                LOGGER.info(
                    "Job %s left the active pipeline after a controlled cancellation.",
                    job.job_id,
                )
                return False
        return True

    def _process_i2v_batch(
        self,
        job: JobPayload,
        scene_by_id: Mapping[int, SceneSpec],
        resolved_lora_filenames: Mapping[str, str],
    ) -> bool:
        LOGGER.info(
            "Job %s: generating all required I2V clips from cached frames.",
            job.job_id,
        )
        for record in self.store.scene_records(job.job_id):
            if record.state in {SceneState.FAILED, SceneState.CANCELLED}:
                continue
            if (
                record.state == SceneState.SUCCEEDED
                and record.video_path
                and Path(record.video_path).is_file()
            ):
                continue
            scene = scene_by_id[record.scene_id]
            self._process_scene_stage_with_retries(
                job,
                scene,
                PipelineState.RUNNING_I2V,
                resolved_lora_filenames,
            )
            if self.store.snapshot().job_id != job.job_id:
                LOGGER.info(
                    "Job %s left the active pipeline after a controlled cancellation.",
                    job.job_id,
                )
                return False
        return True

    def _deliver_i2v_batch(
        self,
        job: JobPayload,
        scene_by_id: Mapping[int, SceneSpec],
    ) -> bool:
        """Deliver watermarked copies only after the shared LTX phase is unloaded."""
        if self.delivery is None or self.continuation_renderer is None:
            return True
        LOGGER.info(
            "Job %s: LTX generation is complete; sending Discord-only "
            "watermarked scene copies.",
            job.job_id,
        )
        for record in self.store.scene_records(job.job_id):
            if (
                record.state != SceneState.SUCCEEDED
                or not record.video_path
                or not Path(record.video_path).is_file()
            ):
                continue
            scene = scene_by_id[record.scene_id]
            self.deliver_scene_video(
                job=job,
                scene=scene,
                scene_path=record.video_path,
                revision=1,
                prompt_id_callback=lambda prompt_id, scene_id=record.scene_id: (
                    self.store.set_scene_prompt_id(
                        job.job_id,
                        scene_id,
                        prompt_id,
                        stage="delivery",
                    )
                ),
            )
            self._release_memory()
            if self.store.snapshot().job_id != job.job_id:
                LOGGER.info(
                    "Job %s left the active pipeline during delivery.",
                    job.job_id,
                )
                return False
        return True

    def _process_scene_stage_with_retries(
        self,
        job: JobPayload,
        scene: SceneSpec,
        pipeline_state: PipelineState,
        resolved_lora_filenames: Mapping[str, str],
    ) -> None:
        stage = "T2I" if pipeline_state == PipelineState.RUNNING_T2I else "I2V"
        while True:
            try:
                if pipeline_state == PipelineState.RUNNING_T2I:
                    self._process_t2i_stage(job, scene, resolved_lora_filenames)
                else:
                    self._process_i2v_stage(job, scene, resolved_lora_filenames)
                return
            except ComfyHttpError as error:
                if not self.comfy.alive():
                    raise FatalPipelineError(str(error)) from error
                snapshot = self.store.snapshot()
                record = next(
                    item
                    for item in self.store.scene_records(job.job_id)
                    if item.scene_id == scene.scene_id
                )
                if (
                    snapshot.job_id != job.job_id
                    or record.state == SceneState.CANCELLED
                ):
                    LOGGER.info(
                        "Job %s scene %s %s stopped after a controlled user cancellation.",
                        job.job_id,
                        scene.scene_id,
                        stage,
                    )
                    return
                self.store.clear_scene_prompt_id(
                    job.job_id,
                    scene.scene_id,
                )
                if (
                    pipeline_state == PipelineState.RUNNING_I2V
                    and self._automatic_uses_continuation(job.job_id, scene)
                ):
                    # Chunk attempts already own their bounded retry policy.
                    # Retrying here would resume the same scene attempt forever.
                    self.store.set_scene_state(
                        job.job_id,
                        scene.scene_id,
                        SceneState.FAILED,
                        error=str(error),
                    )
                    self._update_original_revision(
                        job,
                        scene,
                        state=SceneState.FAILED,
                        error=str(error),
                    )
                    LOGGER.error(
                        "Scene %s continuation stopped after its bounded chunk "
                        "retry policy: %s",
                        scene.scene_id,
                        error,
                    )
                    return
                attempts = (
                    record.t2i_attempts
                    if pipeline_state == PipelineState.RUNNING_T2I
                    else record.i2v_attempts
                )
                if attempts < self.settings.max_stage_attempts:
                    LOGGER.warning(
                        "Retrying scene %s %s after attempt %s: %s",
                        scene.scene_id,
                        stage,
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
                self._update_original_revision(
                    job,
                    scene,
                    state=SceneState.FAILED,
                    error=str(error),
                )
                LOGGER.error(
                    "Scene %s %s exhausted its retry budget: %s",
                    scene.scene_id,
                    stage,
                    error,
                )
                return
            except Exception as error:
                self.store.set_scene_state(
                    job.job_id,
                    scene.scene_id,
                    SceneState.FAILED,
                    error=str(error),
                )
                self._update_original_revision(
                    job,
                    scene,
                    state=SceneState.FAILED,
                    error=str(error),
                )
                LOGGER.exception(
                    "Scene %s %s failed with a non-ComfyUI error.",
                    scene.scene_id,
                    stage,
                )
                return

    def run_or_reclaim_prompt(
        self,
        workflow: Mapping[str, Any],
        *,
        timeout_seconds: float,
        existing_prompt_id: str | None = None,
        prompt_id_callback: Callable[[str], None] | None = None,
        active_check: Callable[[], bool] | None = None,
    ) -> Mapping[str, Any]:
        """Reclaim one persisted prompt or queue it with cancellation-safe ownership."""
        prompt_id = existing_prompt_id
        completed_prompt = getattr(self.comfy, "completed_prompt", None)
        prompt_is_queued = getattr(self.comfy, "prompt_is_queued", None)
        if prompt_id and callable(completed_prompt) and callable(prompt_is_queued):
            completed = completed_prompt(prompt_id)
            if completed is not None:
                LOGGER.info("Reclaimed completed ComfyUI prompt %s.", prompt_id)
                return completed
            if prompt_is_queued(prompt_id):
                LOGGER.info("Waiting for persisted ComfyUI prompt %s.", prompt_id)
                return self.comfy.wait_for_prompt(
                    prompt_id,
                    timeout_seconds=timeout_seconds,
                )
            LOGGER.warning(
                "Persisted ComfyUI prompt %s is absent from queue/history; "
                "requeueing the same deterministic workflow.",
                prompt_id,
            )
            prompt_id = None

        if prompt_id is None:
            prompt_id = self.comfy.queue_prompt(workflow)
            try:
                if prompt_id_callback is not None:
                    prompt_id_callback(prompt_id)
                if active_check is not None and not active_check():
                    raise ComfyHttpError(
                        "Prompt ownership was cancelled while ComfyUI accepted the prompt."
                    )
            except Exception:
                cancel = getattr(self.comfy, "cancel_owned_prompt", None)
                if callable(cancel):
                    try:
                        cancel(prompt_id)
                    except ComfyHttpError:
                        LOGGER.warning(
                            "Could not cancel prompt %s after ownership persistence "
                            "failed.",
                            prompt_id,
                            exc_info=True,
                        )
                raise
        return self.comfy.wait_for_prompt(
            prompt_id,
            timeout_seconds=timeout_seconds,
        )

    def _process_t2i_stage(
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
            self._update_original_revision(
                job,
                scene,
                state=SceneState.PENDING,
                frame_path=frame_path,
            )
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
            self._update_original_revision(
                job,
                scene,
                state=SceneState.PENDING,
                frame_path=frame_path,
            )
            LOGGER.info(
                "Job %s scene %s: recovered deterministic T2I frame from disk.",
                job.job_id,
                scene.scene_id,
            )
        else:
            resume_prompt = bool(
                record.prompt_id
                and record.prompt_stage == "t2i"
                and record.t2i_attempts > 0
            )
            if (
                not resume_prompt
                and record.t2i_attempts >= self.settings.max_stage_attempts
            ):
                raise ComfyHttpError("T2I attempt limit reached.")
            attempt = self.store.begin_scene_stage(
                job.job_id,
                scene.scene_id,
                PipelineState.RUNNING_T2I,
                resume=resume_prompt,
            )
            LOGGER.info(
                "Job %s scene %s: %s T2I attempt %s.",
                job.job_id,
                scene.scene_id,
                "reclaiming" if resume_prompt else "building and queueing",
                attempt,
            )
            build = build_t2i_api_workflow(
                job,
                scene,
                resolved_lora_filenames,
                delivery=self.delivery,
            )
            self.run_or_reclaim_prompt(
                build.api,
                timeout_seconds=self.settings.t2i_timeout_seconds,
                existing_prompt_id=record.prompt_id if resume_prompt else None,
                prompt_id_callback=lambda prompt_id: self.store.set_scene_prompt_id(
                    job.job_id,
                    scene.scene_id,
                    prompt_id,
                    stage="t2i",
                ),
                active_check=lambda: (
                    self.store.snapshot().job_id == job.job_id
                    and self.store.continuation_work_is_active(
                        job.job_id,
                        scene.scene_id,
                        1,
                    )
                ),
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
            self._update_original_revision(
                job,
                scene,
                state=SceneState.PENDING,
                frame_path=frame_path,
            )
            LOGGER.info(
                "Job %s scene %s: T2I finished and cached the deterministic frame.",
                job.job_id,
                scene.scene_id,
            )

    def render_i2v_scene(
        self,
        *,
        job: JobPayload,
        scene: SceneSpec,
        frame_path: str | Path,
        destination: str | Path,
        resolved_lora_filenames: Mapping[str, str],
        revision: int = 1,
        overrides: SceneWorkflowOverrides | None = None,
        prompt_id_callback: Callable[[str], None] | None = None,
        existing_prompt_id: str | None = None,
        deliver_to_discord: bool = True,
        continuation_route: bool | None = None,
    ) -> Path:
        """Use one shared legacy/continuation route for jobs and GUI remakes."""
        use_continuation = (
            self._uses_continuation(scene, overrides)
            if continuation_route is None
            else continuation_route
        )
        output_path = Path(destination)
        if use_continuation:
            if self.continuation_renderer is None:
                raise ComfyHttpError(
                    "Chunked continuation requires configured D-drive storage."
                )
            LOGGER.info(
                "Job %s scene %s revision %s: using resumable 121-frame "
                "LTX continuation.",
                job.job_id,
                scene.scene_id,
                revision,
            )
            return self.continuation_renderer.render_scene(
                job,
                scene,
                frame_path,
                output_path,
                revision=revision,
                resolved_lora_filenames=resolved_lora_filenames,
                overrides=overrides,
                deliver_to_discord=deliver_to_discord,
                prompt_id_callback=prompt_id_callback,
            ).scene_path

        LOGGER.info(
            "Job %s scene %s revision %s: using the legacy single-window "
            "I2V workflow.",
            job.job_id,
            scene.scene_id,
            revision,
        )
        build = build_i2v_api_workflow(
            job,
            scene,
            frame_path,
            resolved_lora_filenames,
            delivery=self.delivery if deliver_to_discord else None,
            overrides=overrides,
        )
        history = self.run_or_reclaim_prompt(
            build.api,
            timeout_seconds=self.settings.i2v_timeout_seconds,
            existing_prompt_id=existing_prompt_id,
            prompt_id_callback=prompt_id_callback,
            active_check=(
                lambda: self.store.continuation_work_is_active(
                    job.job_id,
                    scene.scene_id,
                    revision,
                )
            )
            if prompt_id_callback is not None
            else None,
        )
        metadata = find_video_output(history, build.output_node_id)
        return self.comfy.download_output(metadata, output_path)

    def _uses_continuation(
        self,
        scene: SceneSpec,
        overrides: SceneWorkflowOverrides | None = None,
    ) -> bool:
        """Route from the duration that the continuation renderer will use."""
        continuation = (
            overrides.temporal_continuation
            if overrides is not None
            else scene.i2v.continuation
        )
        requested_duration = scene.estimated_sec
        if continuation is not None:
            configured_duration = continuation.get("requested_duration_seconds")
            if isinstance(configured_duration, (int, float)) and not isinstance(
                configured_duration, bool
            ):
                requested_duration = float(configured_duration)
        return continuation_is_enabled(
            scene_frame_count=timeline_frame_count(requested_duration),
            continuation=continuation,
            mode=self.settings.continuation_mode,
        )

    def _automatic_uses_continuation(
        self,
        job_id: str,
        scene: SceneSpec,
    ) -> bool:
        """Keep an interrupted automatic scene on its originally selected route."""
        record = next(
            item
            for item in self.store.scene_records(job_id)
            if item.scene_id == scene.scene_id
        )
        if record.prompt_stage == "i2v_continuation":
            return True
        if record.prompt_stage == "i2v_legacy":
            return False
        if self.store.continuation_plan(job_id, scene.scene_id, 1) is not None:
            return True
        if record.i2v_attempts > 0:
            # A started pre-continuation scene without a continuation plan is
            # a durable legacy selection even if T2I recovery later replaced
            # the scene's current prompt owner.
            return False
        return self._uses_continuation(scene)

    def deliver_scene_video(
        self,
        *,
        job: JobPayload,
        scene: SceneSpec,
        scene_path: str | Path,
        revision: int,
        overrides: SceneWorkflowOverrides | None = None,
        prompt_id_callback: Callable[[str], None] | None = None,
        existing_prompt_id: str | None = None,
        require_authoritative: bool = False,
    ) -> bool:
        """Send a Discord-only copy after generation models have been released."""
        if self.delivery is None:
            return True
        if self.continuation_renderer is None:
            if require_authoritative:
                raise ContinuationDeliveryError(
                    "Configured Discord delivery has no continuation renderer.",
                    state="FAILED",
                )
            raise ComfyHttpError(
                "Discord scene delivery requires configured D-drive storage."
            )
        for delivery_attempt in range(1, 3):
            try:
                delivery_kwargs: dict[str, Any] = {
                    "revision": revision,
                    "overrides": overrides,
                    "prompt_id_callback": prompt_id_callback,
                }
                if existing_prompt_id is not None:
                    delivery_kwargs["existing_prompt_id"] = existing_prompt_id
                result = self.continuation_renderer.deliver_existing_scene(
                    job, scene, scene_path, **delivery_kwargs
                )
                return result or True
            except ContinuationDeliveryError as error:
                if error.state == "FAILED" and delivery_attempt < 2:
                    LOGGER.warning(
                        "Discord delivery failed for job %s scene %s revision %s; "
                        "reclaiming/retrying once: %s",
                        job.job_id,
                        scene.scene_id,
                        revision,
                        error,
                    )
                    continue
                if require_authoritative:
                    raise
                LOGGER.error(
                    "Discord delivery failed twice for job %s scene %s revision %s. "
                    "The raw unwatermarked scene remains valid and final assembly "
                    "will continue: %s",
                    job.job_id,
                    scene.scene_id,
                    revision,
                    error,
                )
                return False
        return False  # pragma: no cover - loop always returns.

    def _process_i2v_stage(
        self,
        job: JobPayload,
        scene: SceneSpec,
        resolved_lora_filenames: Mapping[str, str],
    ) -> None:
        record = next(
            item for item in self.store.scene_records(job.job_id) if item.scene_id == scene.scene_id
        )
        frame_path = (
            Path(record.frame_path)
            if record.frame_path and Path(record.frame_path).is_file()
            else self._frame_path(job.job_id, scene.scene_id)
        )
        if not frame_path.is_file():
            raise ComfyHttpError(
                f"I2V requires a cached T2I frame for scene {scene.scene_id}."
            )
        clip_path = self._clip_path(job.job_id, scene.scene_id)
        use_continuation = self._automatic_uses_continuation(job.job_id, scene)
        if (
            not use_continuation
            and record.video_path
            and Path(record.video_path).is_file()
        ):
            self.store.set_scene_state(job.job_id, scene.scene_id, SceneState.SUCCEEDED)
            self._update_original_revision(
                job,
                scene,
                state=SceneState.SUCCEEDED,
                frame_path=frame_path,
                video_path=Path(record.video_path),
            )
            LOGGER.info(
                "Job %s scene %s: reusing completed scene clip.",
                job.job_id,
                scene.scene_id,
            )
            return
        if not use_continuation and clip_path.is_file():
            self.store.set_scene_state(
                job.job_id,
                scene.scene_id,
                SceneState.SUCCEEDED,
                video_path=str(clip_path),
            )
            self._update_original_revision(
                job,
                scene,
                state=SceneState.SUCCEEDED,
                frame_path=frame_path,
                video_path=clip_path,
            )
            LOGGER.info(
                "Job %s scene %s: recovered deterministic scene clip from disk.",
                job.job_id,
                scene.scene_id,
            )
            return
        resume_legacy_prompt = bool(
            not use_continuation
            and record.prompt_id
            and record.prompt_stage == "i2v_legacy"
            and record.i2v_attempts > 0
        )
        if (
            not use_continuation
            and not resume_legacy_prompt
            and record.i2v_attempts >= self.settings.max_stage_attempts
        ):
            raise ComfyHttpError("I2V attempt limit reached.")

        resume_continuation = use_continuation and record.i2v_attempts > 0
        resume_stage = resume_continuation or resume_legacy_prompt
        prompt_stage = (
            "i2v_continuation" if use_continuation else "i2v_legacy"
        )
        attempt = self.store.begin_scene_stage(
            job.job_id,
            scene.scene_id,
            PipelineState.RUNNING_I2V,
            prompt_stage=prompt_stage,
            resume=resume_stage,
        )
        LOGGER.info(
            "Job %s scene %s: %s resumable I2V run %s.",
            job.job_id,
            scene.scene_id,
            "resuming" if resume_stage else "starting",
            attempt,
        )
        self.render_i2v_scene(
            job=job,
            scene=scene,
            frame_path=frame_path,
            destination=clip_path,
            resolved_lora_filenames=resolved_lora_filenames,
            revision=1,
            deliver_to_discord=False,
            existing_prompt_id=(
                record.prompt_id if resume_legacy_prompt else None
            ),
            prompt_id_callback=lambda prompt_id: self.store.set_scene_prompt_id(
                job.job_id,
                scene.scene_id,
                prompt_id,
                stage=prompt_stage,
            ),
            continuation_route=use_continuation,
        )
        self.store.set_scene_state(
            job.job_id,
            scene.scene_id,
            SceneState.SUCCEEDED,
            frame_path=str(frame_path),
            video_path=str(clip_path),
        )
        self._update_original_revision(
            job,
            scene,
            state=SceneState.SUCCEEDED,
            frame_path=frame_path,
            video_path=clip_path,
        )
        LOGGER.info(
            "Job %s scene %s: I2V finished and saved the deterministic clip.",
            job.job_id,
            scene.scene_id,
        )

    def _update_original_revision(
        self,
        job: JobPayload,
        scene: SceneSpec,
        *,
        state: SceneState,
        frame_path: str | Path | None = None,
        video_path: str | Path | None = None,
        error: str | None = None,
    ) -> None:
        """Keep revision one and its human-readable manifest in sync."""
        frame_text = str(frame_path) if frame_path is not None else None
        video_text = str(video_path) if video_path is not None else None
        self.store.update_scene_revision(
            job.job_id,
            scene.scene_id,
            1,
            state=state,
            frame_path=frame_text,
            video_path=video_text,
            error=error,
        )
        if self.storage is None:
            return
        revision = next(
            item
            for item in self.store.scene_revisions(job.job_id, scene.scene_id)
            if item.revision == 1
        )
        write_json_atomic(
            self.storage.generation_manifest_path(job.job_id, scene.scene_id, 1),
            {
                "job_id": job.job_id,
                "scene_id": scene.scene_id,
                "revision": 1,
                "remake_mode": "image_and_video",
                "parameters": revision.parameters,
                "frame_path": revision.frame_path,
                "video_path": revision.video_path,
                "status": revision.state.value,
                "error": revision.error,
            },
        )

    def _resolve_assets(
        self,
        job: JobPayload,
        *,
        scene_ids: set[int] | None = None,
    ) -> AssetPreparation:
        selected_scenes = tuple(
            scene
            for scene in job.scenes
            if scene_ids is None or scene.scene_id in scene_ids
        )
        if scene_ids is not None and {scene.scene_id for scene in selected_scenes} != scene_ids:
            raise ValueError("Asset resolution requested an unknown scene.")
        failures: dict[int, dict[str, str]] = {
            scene.scene_id: {} for scene in selected_scenes
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
            for scene in selected_scenes:
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
                for scene in selected_scenes:
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

        for scene in selected_scenes:
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

    def release_memory(self) -> None:
        """Release ComfyUI models only at a deliberate pipeline boundary."""
        self._release_memory()

    def _release_memory(self) -> None:
        gc.collect()
        try:
            self.comfy.free_memory()
        except ComfyHttpError:
            LOGGER.warning("ComfyUI memory release request failed.", exc_info=True)

    def handle_fatal(self, error: FatalPipelineError) -> None:
        """Enter durable error state, then use configured controlled recovery."""
        snapshot = self.store.snapshot()
        self.store.transition(
            PipelineState.ERROR,
            job_id=snapshot.job_id,
            active_scene_id=snapshot.active_scene_id,
            error=str(error),
        )
        if self.restart_comfy and self.restart_comfy():
            if snapshot.job_id and snapshot.state in {
                PipelineState.DOWNLOADING_ASSETS,
                PipelineState.RUNNING_T2I,
                PipelineState.RUNNING_I2V,
                PipelineState.STITCHING,
            }:
                self.store.requeue_unfinished_scenes(snapshot.job_id)
                self.store.transition(
                    PipelineState.DOWNLOADING_ASSETS,
                    job_id=snapshot.job_id,
                )
                LOGGER.warning("ComfyUI restarted; unfinished scenes will resume.")
                return
            if snapshot.state in {
                PipelineState.IDLE,
                PipelineState.WAITING_FOR_GROK,
                PipelineState.AWAITING_REVIEW,
                PipelineState.QC_BLOCKED,
            }:
                self.store.transition(
                    snapshot.state,
                    job_id=snapshot.job_id,
                )
                LOGGER.warning(
                    "ComfyUI restarted; restored non-render pipeline state %s.",
                    snapshot.state.value,
                )
                return
            if snapshot.state in {
                PipelineState.RUNNING_QC,
                PipelineState.AWAITING_QC_REVIEW,
            }:
                self.store.transition(
                    PipelineState.RUNNING_QC,
                    job_id=snapshot.job_id,
                )
                LOGGER.warning(
                    "ComfyUI restarted; restored the durable QC controller epoch."
                )
                return
            LOGGER.error(
                "ComfyUI restarted, but pipeline remains in error because its "
                "previous state was %s.",
                snapshot.state.value,
            )
        else:
            LOGGER.error("Fatal pipeline error requires recovery: %s", error)

    # Compatibility for callers from earlier project revisions.
    _handle_fatal = handle_fatal
