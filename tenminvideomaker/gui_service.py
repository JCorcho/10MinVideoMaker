"""GUI-hosted single-owner supervisor and remake-batch execution."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import threading
from typing import Any, Mapping

from .assembly import AssemblyError, validate_video_profile
from .comfy_http import ComfyHttpError, find_video_output
from .review import ReviewValidationError, ValidatedSceneEdit, validate_scene_edit
from .state_store import (
    PipelineState,
    PipelineStateStore,
    ManualFinalRecord,
    ManualFinalState,
    RemakeBatchRecord,
    RemakeBatchState,
    RemakeItemRecord,
    RemakeMode,
    SceneRevision,
    SceneState,
    StateTransitionError,
)
from .storage import StorageLayout, write_json_atomic
from .supervisor import PipelineSupervisor
from .workflow_builder import build_i2v_api_workflow, build_t2i_api_workflow


LOGGER = logging.getLogger("10MinVideoMaker.gui")
ACTIVE_RENDER_STATES = frozenset(
    {
        PipelineState.DOWNLOADING_ASSETS,
        PipelineState.RUNNING_T2I,
        PipelineState.RUNNING_I2V,
        PipelineState.STITCHING,
    }
)
# States where the singleton pipeline is holding a project that should not
# accept a replacement job until abandoned/cancelled or finished.
CANCELLABLE_PROJECT_STATES = frozenset(
    {
        *ACTIVE_RENDER_STATES,
        PipelineState.ERROR,
        PipelineState.AWAITING_REVIEW,
    }
)


class GuiServiceError(RuntimeError):
    """Raised when a GUI command cannot be applied safely."""


@dataclass(frozen=True)
class _PreparedRemakeItem:
    item: RemakeItemRecord
    revision: SceneRevision
    edit: ValidatedSceneEdit
    resolved_lora_filenames: Mapping[str, str]
    frame_path: Path
    clip_path: Path
    manifest_path: Path
    manifest: dict[str, Any]


class SupervisorController:
    """Run Gmail/automatic work and queued remakes through one worker."""

    def __init__(
        self,
        supervisor: PipelineSupervisor,
        storage: StorageLayout,
        *,
        idle_wait_seconds: float = 2.0,
    ):
        self.supervisor = supervisor
        self.store: PipelineStateStore = supervisor.store
        self.storage = storage
        self.idle_wait_seconds = idle_wait_seconds
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_error: str | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._worker,
            name="10MinVideoMaker-GUI-supervisor",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def wake(self) -> None:
        self._wake.set()

    def active_render(self) -> bool:
        return self.store.snapshot().state in ACTIVE_RENDER_STATES

    def approve_job(self, job_id: str) -> None:
        self.store.approve_job(job_id)
        self.wake()

    def queue_batch(self, batch_id: str, collision_policy: str) -> None:
        if collision_policy == "interrupt_current" and self.active_render():
            self.interrupt_current_job()
        self.store.queue_remake_batch(batch_id, collision_policy)
        self.wake()

    def queue_manual_final(self, job_id: str) -> ManualFinalRecord:
        request = self.store.queue_manual_final(job_id)
        self.wake()
        return request

    def interrupt_current_job(self) -> tuple[str, ...]:
        snapshot = self.store.snapshot()
        if snapshot.state not in ACTIVE_RENDER_STATES or not snapshot.job_id:
            return ()
        cancelled = self.supervisor.comfy.cancel_project_prompts()
        self.store.abandon_job(
            snapshot.job_id,
            reason="Interrupted from the GUI so a remake batch could run immediately.",
        )
        LOGGER.warning(
            "Interrupted automatic job %s for a GUI remake batch; cancelled prompts=%s.",
            snapshot.job_id,
            len(cancelled),
        )
        return cancelled

    def can_cancel_current_project(self) -> bool:
        snapshot = self.store.snapshot()
        return bool(
            snapshot.job_id and snapshot.state in CANCELLABLE_PROJECT_STATES
        )

    def cancel_current_project(self) -> dict[str, Any]:
        """Abandon the held project so the worker can accept the next Gmail job.

        History, payloads, and completed scenes remain. Only this project's
        ComfyUI prompts are cancelled when a render is active.
        """
        snapshot = self.store.snapshot()
        if not snapshot.job_id or snapshot.state not in CANCELLABLE_PROJECT_STATES:
            raise GuiServiceError(
                "There is no held project to cancel. The pipeline is already free "
                "to check email or wait for the next handoff."
            )
        job_id = snapshot.job_id
        cancelled: tuple[str, ...] = ()
        if snapshot.state in ACTIVE_RENDER_STATES:
            cancelled = self.supervisor.comfy.cancel_project_prompts()
        self.store.abandon_job(
            job_id,
            reason="Cancelled from the GUI so the pipeline could advance to the next job.",
        )
        self.wake()
        LOGGER.warning(
            "Cancelled project %s from the GUI; pipeline is idle for the next job "
            "(cancelled prompts=%s).",
            job_id,
            len(cancelled),
        )
        return {
            "job_id": job_id,
            "cancelled_prompts": list(cancelled),
            "pipeline_state": self.store.snapshot().state.value,
        }

    def status_document(self) -> dict[str, Any]:
        snapshot = self.store.snapshot()
        manual_final = self.store.latest_manual_final_any()
        try:
            running, pending = self.supervisor.comfy.queue_counts()
            comfy_healthy = True
        except ComfyHttpError:
            running = pending = 0
            comfy_healthy = False
        return {
            "controller_running": self.running,
            "pipeline_state": snapshot.state.value,
            "job_id": snapshot.job_id,
            "active_scene_id": snapshot.active_scene_id,
            "pipeline_error": snapshot.error,
            "last_controller_error": self.last_error,
            "comfyui_healthy": comfy_healthy,
            "comfyui_running": running,
            "comfyui_pending": pending,
            "active_render": snapshot.state in ACTIVE_RENDER_STATES,
            "can_cancel_current_project": self.can_cancel_current_project(),
            "hold_new_jobs_for_review": self.supervisor.settings.require_human_review,
            "manual_final": (
                {
                    "job_id": manual_final.job_id,
                    "state": manual_final.state.value,
                    "error": manual_final.error,
                    "output_available": bool(manual_final.output_path),
                }
                if manual_final is not None
                else None
            ),
        }

    def _worker(self) -> None:
        LOGGER.info("GUI supervisor worker started.")
        while not self._stop.is_set():
            try:
                manual_final = self.store.next_queued_manual_final()
                batch = self.store.next_queued_remake_batch()
                if manual_final is not None and not self.active_render():
                    self._run_manual_final(manual_final)
                elif batch is not None and self._batch_can_run(batch):
                    self._run_remake_batch(batch)
                else:
                    self.supervisor.tick()
                self._last_error = None
            except Exception as error:
                self._last_error = str(error)
                LOGGER.exception("GUI supervisor worker tick failed; state was preserved.")
            self._wake.wait(self._wait_interval())
            self._wake.clear()
        LOGGER.info("GUI supervisor worker stopped.")

    def _wait_interval(self) -> float:
        state = self.store.snapshot().state
        if state in {
            PipelineState.WAITING_FOR_GROK,
            PipelineState.AWAITING_REVIEW,
            PipelineState.ERROR,
        }:
            return self.supervisor.settings.poll_interval_seconds
        return self.idle_wait_seconds

    def _batch_can_run(self, batch: RemakeBatchRecord) -> bool:
        snapshot = self.store.snapshot()
        if snapshot.state not in ACTIVE_RENDER_STATES:
            return True
        return batch.collision_policy == "interrupt_current"

    def _run_manual_final(self, request: ManualFinalRecord) -> None:
        """Concatenate an explicit snapshot without changing automated assembly behavior."""
        self.store.set_manual_final_state(request.request_id, ManualFinalState.RUNNING)
        try:
            selection = self.store.manual_final_selection(request.request_id)
            clips: list[Path] = []
            storage_root = self.storage.root.resolve()
            for item in selection:
                clip = Path(item.video_path).resolve()
                try:
                    clip.relative_to(storage_root)
                except ValueError as error:
                    raise GuiServiceError(
                        f"Manual final clip for scene {item.scene_id} is outside project storage."
                    ) from error
                clips.append(clip)
            streams = [self.supervisor._video_probe(path) for path in clips]
            validate_video_profile(streams)
            output_path = self.supervisor.assembler.stitch(
                request.job_id,
                clips,
                self.storage.temp_root / "manual-final-concat",
            )
            self.store.set_job_final_path(request.job_id, str(output_path))
            self.store.set_manual_final_state(
                request.request_id,
                ManualFinalState.SUCCEEDED,
                output_path=str(output_path),
            )
            LOGGER.info(
                "Manual final completed | job=%s scenes=%s.",
                request.job_id,
                len(selection),
            )
        except (AssemblyError, GuiServiceError, OSError, StateTransitionError) as error:
            self.store.set_manual_final_state(
                request.request_id,
                ManualFinalState.FAILED,
                error=str(error),
            )
            LOGGER.error("Manual final failed | job=%s | reason=%s", request.job_id, error)

    def _run_remake_batch(self, batch: RemakeBatchRecord) -> None:
        self.store.set_remake_batch_state(batch.batch_id, RemakeBatchState.RUNNING)
        prepared: list[_PreparedRemakeItem] = []
        for item in self.store.remake_items(batch.batch_id):
            if self._stop.is_set():
                break
            if item.state == SceneState.SUCCEEDED:
                continue
            self.store.set_remake_item_state(
                batch.batch_id,
                item.position,
                SceneState.RUNNING,
            )
            try:
                prepared.append(self._prepare_remake_item(item))
            except Exception as error:
                self._mark_remake_failure(batch, item, error)

        # Group frames by their T2I model family. This handles a mixed Anima/Pony
        # edit batch without returning to LTX between individual image remakes.
        image_remakes = sorted(
            (
                item
                for item in prepared
                if item.revision.remake_mode == RemakeMode.IMAGE_AND_VIDEO
            ),
            key=lambda item: (item.edit.job.character.base_model.casefold(), item.item.position),
        )
        try:
            for item in image_remakes:
                if self._stop.is_set():
                    break
                try:
                    self._run_remake_t2i(item)
                except Exception as error:
                    self._mark_remake_failure(batch, item.item, error, prepared=item)
        finally:
            if image_remakes:
                self.supervisor.release_memory()

        # All successful image remakes and all video-only remakes now share the
        # same LTX stage, so the LTX model stays resident across every clip.
        failed_positions = {
            item.position
            for item in self.store.remake_items(batch.batch_id)
            if item.state == SceneState.FAILED
        }
        i2v_remakes = [
            item for item in prepared if item.item.position not in failed_positions
        ]
        try:
            for item in i2v_remakes:
                if self._stop.is_set():
                    break
                try:
                    self._run_remake_i2v(item)
                except Exception as error:
                    self._mark_remake_failure(batch, item.item, error, prepared=item)
                else:
                    self.store.set_remake_item_state(
                        batch.batch_id,
                        item.item.position,
                        SceneState.SUCCEEDED,
                    )
        finally:
            if i2v_remakes:
                self.supervisor.release_memory()

        items = self.store.remake_items(batch.batch_id)
        succeeded = sum(item.state == SceneState.SUCCEEDED for item in items)
        failed = sum(item.state == SceneState.FAILED for item in items)
        if failed == 0 and succeeded == len(items):
            state = RemakeBatchState.SUCCEEDED
        elif succeeded:
            state = RemakeBatchState.PARTIAL
        else:
            state = RemakeBatchState.FAILED
        self.store.set_remake_batch_state(batch.batch_id, state)

    def _prepare_remake_item(
        self,
        item: RemakeItemRecord,
    ) -> _PreparedRemakeItem:
        revision = next(
            (
                candidate
                for candidate in self.store.scene_revisions(item.job_id, item.scene_id)
                if candidate.revision == item.revision
            ),
            None,
        )
        if revision is None:
            raise GuiServiceError(
                f"Missing revision {item.revision} for scene {item.scene_id}."
            )
        original_job = self.store.load_job(item.job_id)
        try:
            edit = validate_scene_edit(original_job, item.scene_id, revision.parameters)
        except ReviewValidationError as error:
            raise GuiServiceError(str(error)) from error
        preparation = self.supervisor._resolve_assets(
            edit.job,
            scene_ids={item.scene_id},
        )
        if preparation.failures:
            raise GuiServiceError("; ".join(preparation.failures.get(item.scene_id, [])))

        frame_path = (
            Path(revision.frame_path)
            if revision.remake_mode == RemakeMode.VIDEO_ONLY and revision.frame_path
            else self.storage.scene_frame_path(
                item.job_id,
                item.scene_id,
                item.revision,
            )
        )
        if revision.remake_mode == RemakeMode.VIDEO_ONLY and not frame_path.is_file():
            raise GuiServiceError(
                "Video-only remake requires an existing cached starting frame."
            )
        clip_path = self.storage.scene_clip_path(
            item.job_id,
            item.scene_id,
            item.revision,
        )
        manifest_path = self.storage.generation_manifest_path(
            item.job_id,
            item.scene_id,
            item.revision,
        )
        manifest: dict[str, Any] = {
            "job_id": item.job_id,
            "scene_id": item.scene_id,
            "revision": item.revision,
            "remake_mode": revision.remake_mode.value,
            "parameters": edit.document,
            "resolved_lora_filenames": preparation.resolved_filenames,
            "frame_path": str(frame_path),
            "video_path": str(clip_path),
            "status": "running",
        }
        write_json_atomic(manifest_path, manifest)
        self.store.update_scene_revision(
            item.job_id,
            item.scene_id,
            item.revision,
            state=SceneState.RUNNING,
            frame_path=str(frame_path) if frame_path.is_file() else None,
        )
        return _PreparedRemakeItem(
            item=item,
            revision=revision,
            edit=edit,
            resolved_lora_filenames=preparation.resolved_filenames,
            frame_path=frame_path,
            clip_path=clip_path,
            manifest_path=manifest_path,
            manifest=manifest,
        )

    def _run_remake_t2i(self, item: _PreparedRemakeItem) -> None:
        build = build_t2i_api_workflow(
            item.edit.job,
            item.edit.scene,
            item.resolved_lora_filenames,
            delivery=self.supervisor.delivery,
            overrides=item.edit.workflow,
            revision=item.item.revision,
        )
        prompt_id = self.supervisor.comfy.queue_prompt(build.api)
        self.supervisor.comfy.wait_for_prompt(
            prompt_id,
            timeout_seconds=self.supervisor.settings.t2i_timeout_seconds,
        )
        if not item.frame_path.is_file():
            raise GuiServiceError(
                f"T2I completed without revision frame {item.frame_path}."
            )
        self.store.update_scene_revision(
            item.item.job_id,
            item.item.scene_id,
            item.item.revision,
            state=SceneState.RUNNING,
            frame_path=str(item.frame_path),
        )

    def _run_remake_i2v(self, item: _PreparedRemakeItem) -> None:
        build = build_i2v_api_workflow(
            item.edit.job,
            item.edit.scene,
            item.frame_path,
            item.resolved_lora_filenames,
            delivery=self.supervisor.delivery,
            overrides=item.edit.workflow,
        )
        prompt_id = self.supervisor.comfy.queue_prompt(build.api)
        history = self.supervisor.comfy.wait_for_prompt(
            prompt_id,
            timeout_seconds=self.supervisor.settings.i2v_timeout_seconds,
        )
        self.supervisor.comfy.download_output(
            find_video_output(history, build.output_node_id),
            item.clip_path,
        )
        self.store.update_scene_revision(
            item.item.job_id,
            item.item.scene_id,
            item.item.revision,
            state=SceneState.SUCCEEDED,
            frame_path=str(item.frame_path),
            video_path=str(item.clip_path),
        )
        item.manifest["status"] = "succeeded"
        write_json_atomic(item.manifest_path, item.manifest)

    def _mark_remake_failure(
        self,
        batch: RemakeBatchRecord,
        item: RemakeItemRecord,
        error: Exception,
        *,
        prepared: _PreparedRemakeItem | None = None,
    ) -> None:
        message = str(error)
        self.store.set_remake_item_state(
            batch.batch_id,
            item.position,
            SceneState.FAILED,
            error=message,
        )
        self.store.update_scene_revision(
            item.job_id,
            item.scene_id,
            item.revision,
            state=SceneState.FAILED,
            error=message,
        )
        if prepared is not None:
            prepared.manifest["status"] = "failed"
            prepared.manifest["error"] = message
            write_json_atomic(prepared.manifest_path, prepared.manifest)
        LOGGER.exception(
            "Remake failed | batch=%s job=%s scene=%s revision=%s.",
            batch.batch_id,
            item.job_id,
            item.scene_id,
            item.revision,
        )
