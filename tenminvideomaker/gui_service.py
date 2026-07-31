"""GUI-hosted single-owner supervisor and remake-batch execution."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
import threading
from typing import Any, Mapping

from .assembly import AssemblyError, validate_video_profile
from .chunk_artifacts import sha256_file
from .comfy_http import ComfyHttpError
from .continuation import ContinuationPlanError
from .continuation_renderer import ContinuationRenderError, ContinuationRenderer
from .review import ReviewValidationError, ValidatedSceneEdit, validate_scene_edit
from .state_store import (
    ChunkState,
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
from .supervisor import FatalPipelineError, PipelineSupervisor
from .workflow_builder import build_t2i_api_workflow


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
    continuation_route: bool
    video_checkpoint_valid: bool


def _document_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(document) if isinstance(document, Mapping) else {}


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
        self._active_remake_revision: tuple[str, int, int] | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def start(self) -> None:
        if self.running:
            return
        recovered = self.store.recover_interrupted_remake_batches()
        if recovered:
            LOGGER.warning(
                "Recovered %s interrupted remake batch(es) for checkpoint-safe resume.",
                len(recovered),
            )
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
        self.store.abandon_job(
            snapshot.job_id,
            reason="Interrupted from the GUI so a remake batch could run immediately.",
        )
        cancelled = self.supervisor.comfy.cancel_project_prompts()
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
        self.store.abandon_job(
            job_id,
            reason="Cancelled from the GUI so the pipeline could advance to the next job.",
        )
        if snapshot.state in ACTIVE_RENDER_STATES:
            cancelled = self.supervisor.comfy.cancel_project_prompts()
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
            "continuation_mode": self.supervisor.settings.continuation_mode,
            "chunk_progress": self.active_chunk_progress_document(),
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

    def chunk_progress_document(
        self,
        job_id: str,
        scene_id: int,
        revision: int,
    ) -> dict[str, Any] | None:
        """Project durable chunk state into a small human-readable status."""
        plan = self.store.continuation_plan(job_id, scene_id, revision)
        if plan is None:
            return None
        chunks = self.store.chunk_records(job_id, scene_id, revision)
        progress = self.store.chunk_progress(job_id, scene_id, revision)
        revision_record = next(
            (
                item
                for item in self.store.scene_revisions(job_id, scene_id)
                if item.revision == revision
            ),
            None,
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
        priority_states = (
            {ChunkState.FAILED_TERMINAL},
            {ChunkState.FAILED_RETRYABLE},
            {ChunkState.READY},
            {
                ChunkState.CANCELLED,
                ChunkState.INVALIDATED,
                ChunkState.STALE_UPSTREAM,
                ChunkState.BLOCKED_UPSTREAM,
            },
        )
        fallback = None
        for states in priority_states:
            fallback = next(
                (chunk for chunk in chunks if chunk.state in states),
                None,
            )
            if fallback is not None:
                break
        current = next(
            (chunk for chunk in chunks if chunk.state in active_states),
            fallback or (chunks[-1] if chunks else None),
        )
        state = current.state if current is not None else None
        chunk_states = {chunk.state for chunk in chunks}
        output_available = bool(
            revision_record is not None
            and revision_record.state == SceneState.SUCCEEDED
            and revision_record.video_path
        )
        if (
            progress.total_count > 0
            and progress.complete_count == progress.total_count
            and output_available
        ):
            phase = "complete"
        elif (
            ChunkState.FAILED_TERMINAL in chunk_states
            or (
                revision_record is not None
                and revision_record.state == SceneState.FAILED
            )
        ):
            phase = "failed"
        elif (
            ChunkState.CANCELLED in chunk_states
            or (
                revision_record is not None
                and revision_record.state == SceneState.CANCELLED
            )
        ):
            phase = "cancelled"
        elif chunk_states.intersection(
            {ChunkState.INVALIDATED, ChunkState.STALE_UPSTREAM}
        ):
            phase = "invalidated"
        elif state in {
            ChunkState.GENERATING_STAGE1,
            ChunkState.STAGE1_PERSISTING,
            ChunkState.STAGE1_COMPLETE,
            ChunkState.READY,
            ChunkState.FAILED_RETRYABLE,
        }:
            phase = "first_pass"
        elif state in {
            ChunkState.GENERATING_STAGE2,
            ChunkState.STAGE2_PERSISTING,
        }:
            phase = "upscale_refinement"
        elif state in {ChunkState.DECODED, ChunkState.VALIDATING}:
            phase = "validation"
        elif (
            progress.total_count > 0
            and progress.complete_count == progress.total_count
        ):
            phase = "assembly"
        else:
            phase = "waiting"
        attempts = [
            attempt
            for chunk in chunks
            for attempt in self.store.chunk_attempts(
                job_id,
                scene_id,
                revision,
                chunk.chunk_index,
            )
        ]
        attempts_by_chunk: dict[int, list[Any]] = {}
        for attempt in attempts:
            attempts_by_chunk.setdefault(attempt.chunk_index, []).append(attempt)

        def attempt_document(attempt: Any) -> dict[str, Any]:
            parameters = (
                attempt.parameters
                if isinstance(attempt.parameters, Mapping)
                else {}
            )
            runtime = parameters.get("runtime_identity")
            runtime = runtime if isinstance(runtime, Mapping) else {}
            result = attempt.result if isinstance(attempt.result, Mapping) else {}
            return {
                "attempt_number": attempt.attempt_number,
                "state": attempt.state.value,
                "seed": str(attempt.seed),
                "variation_index": attempt.variation_index,
                "upstream_artifact_hash": attempt.upstream_artifact_hash,
                "artifact_hash": attempt.artifact_hash,
                "artifact_manifest_path": attempt.artifact_manifest_path,
                "video_path": attempt.video_path,
                "stage1_prompt_id": result.get("stage1_prompt_id"),
                "stage2_prompt_id": result.get("stage2_prompt_id"),
                "stage1_workflow_sha256": result.get("stage1_workflow_sha256"),
                "stage2_workflow_sha256": result.get("stage2_workflow_sha256"),
                "stage1_checkpoint_sha256": result.get(
                    "stage1_checkpoint_sha256"
                ),
                "stage2_checkpoint_sha256": result.get(
                    "stage2_checkpoint_sha256"
                ),
                "raw_video_sha256": result.get("raw_video_sha256"),
                "implementation_sha256": runtime.get("implementation_sha256"),
                "node_contracts_sha256": runtime.get("node_contracts_sha256"),
                "checkpoint_filename": runtime.get("checkpoint_filename"),
                "text_encoder_filename": runtime.get("text_encoder_filename"),
                "spatial_upscaler_filename": runtime.get(
                    "spatial_upscaler_filename"
                ),
                "mandatory_loras": runtime.get("mandatory_loras"),
                "production": runtime.get("production"),
                "error": attempt.error,
            }

        chunk_details = []
        for chunk in chunks:
            planned = chunk.chunk if isinstance(chunk.chunk, Mapping) else {}
            chunk_details.append(
                {
                    "chunk_number": chunk.chunk_index + 1,
                    "chunk_index": chunk.chunk_index,
                    "state": chunk.state.value,
                    "accepted_attempt_number": chunk.accepted_attempt_number,
                    "accepted_artifact_hash": chunk.accepted_artifact_hash,
                    "prompt": planned.get("prompt"),
                    "negative": planned.get("negative"),
                    "seed": (
                        str(planned["seed"])
                        if planned.get("seed") is not None
                        else None
                    ),
                    "variation_index": planned.get("variation_index"),
                    "model_window_frames": planned.get("model_window_frames"),
                    "new_transition_frames": planned.get(
                        "new_transition_frames"
                    ),
                    "global_window_start_frame": planned.get(
                        "global_window_start_frame"
                    ),
                    "global_window_end_frame_exclusive": planned.get(
                        "global_window_end_frame_exclusive"
                    ),
                    "global_new_start_frame": planned.get(
                        "global_new_start_frame"
                    ),
                    "global_new_end_frame_exclusive": planned.get(
                        "global_new_end_frame_exclusive"
                    ),
                    "segment_indices": planned.get("segment_indices"),
                    "prompt_segmentation_quality": planned.get(
                        "prompt_segmentation_quality"
                    ),
                    "attempts": [
                        attempt_document(attempt)
                        for attempt in attempts_by_chunk.get(
                            chunk.chunk_index,
                            (),
                        )
                    ],
                    "error": chunk.error,
                }
            )
        return {
            "job_id": job_id,
            "scene_id": scene_id,
            "revision": revision,
            "current_chunk": (
                current.chunk_index + 1
                if current is not None
                else progress.complete_count
            ),
            "total_chunks": progress.total_count,
            "completed_chunks": progress.complete_count,
            "phase": phase,
            "state": state.value if state is not None else None,
            "failed_attempts": sum(
                attempt.state
                in {ChunkState.FAILED_RETRYABLE, ChunkState.FAILED_TERMINAL}
                for attempt in attempts
            ),
            # A partial first attempt is normal forward progress, not evidence
            # of a process restart.  Do not claim a resume without a durable
            # retry signal.
            "resumed": any(
                attempt.attempt_number > 1 for attempt in attempts
            ),
            "timeline_frames": plan.plan.get("timeline_output_frames"),
            "generation_master_frames": plan.plan.get(
                "generation_master_frames"
            ),
            "strategy": plan.plan.get("strategy"),
            "overlap_pixel_frames": plan.plan.get("overlap_pixel_frames"),
            "chunks": chunk_details,
            "output_available": output_available,
        }

    def active_chunk_progress_document(self) -> dict[str, Any] | None:
        if self._active_remake_revision is not None:
            return self.chunk_progress_document(*self._active_remake_revision)
        snapshot = self.store.snapshot()
        if snapshot.job_id and snapshot.active_scene_id is not None:
            return self.chunk_progress_document(
                snapshot.job_id,
                snapshot.active_scene_id,
                1,
            )
        return None

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
            except FatalPipelineError as error:
                self._last_error = str(error)
                self.supervisor.handle_fatal(error)
                if self.supervisor.comfy.alive():
                    recovered = self.store.recover_interrupted_remake_batches()
                    if recovered:
                        LOGGER.warning(
                            "Controlled ComfyUI recovery requeued %s interrupted "
                            "remake batch(es) with durable prompt/checkpoint state.",
                            len(recovered),
                        )
                        self._wake.set()
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
            if item.state in {
                SceneState.SUCCEEDED,
                SceneState.FAILED,
                SceneState.CANCELLED,
            }:
                continue
            self.store.set_remake_item_state(
                batch.batch_id,
                item.position,
                SceneState.RUNNING,
            )
            try:
                prepared.append(self._prepare_remake_item(item))
            except FatalPipelineError:
                raise
            except Exception as error:
                self._raise_if_comfy_unavailable(error)
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
                except FatalPipelineError:
                    raise
                except Exception as error:
                    self._raise_if_comfy_unavailable(error)
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
        generated_i2v: list[_PreparedRemakeItem] = []
        try:
            for item in i2v_remakes:
                if self._stop.is_set():
                    break
                try:
                    self._run_remake_i2v(item)
                except FatalPipelineError:
                    raise
                except Exception as error:
                    self._raise_if_comfy_unavailable(error)
                    self._mark_remake_failure(batch, item.item, error, prepared=item)
                else:
                    generated_i2v.append(item)
        finally:
            if i2v_remakes:
                # Keep LTX resident through every requested video remake, then
                # unload it before decoding complete scenes for watermarking.
                self.supervisor.release_memory()

        for item in generated_i2v:
            if self._stop.is_set():
                break
            try:
                self._complete_remake_i2v(item)
            except FatalPipelineError:
                raise
            except Exception as error:
                self._raise_if_comfy_unavailable(error)
                self._mark_remake_failure(batch, item.item, error, prepared=item)
            else:
                self.store.set_remake_item_state(
                    batch.batch_id,
                    item.item.position,
                    SceneState.SUCCEEDED,
                )
            finally:
                if self.supervisor.delivery is not None:
                    # The Discord-only branch decodes and watermarks a complete
                    # scene. Release that frame batch before loading the next.
                    self.supervisor.release_memory()

        if self._stop.is_set():
            LOGGER.info(
                "Remake batch %s stopped mid-run; RUNNING state remains for "
                "checkpoint-safe startup recovery.",
                batch.batch_id,
            )
            return
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
        continuation_route = self._remake_uses_continuation(item, edit)
        try:
            if continuation_route:
                ContinuationRenderer.build_plan(
                    edit.job,
                    edit.scene,
                    item.revision,
                    edit.workflow,
                )
        except (ContinuationPlanError, ContinuationRenderError, ValueError) as error:
            raise GuiServiceError(
                f"Continuation preflight failed before assets or rendering: {error}"
            ) from error
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
        existing_manifest = _read_json_object(manifest_path)
        video_checkpoint_valid = self._remake_video_checkpoint_is_valid(
            item=item,
            revision=revision,
            edit=edit,
            continuation_route=continuation_route,
            frame_path=frame_path,
            clip_path=clip_path,
            manifest=existing_manifest,
        )
        if video_checkpoint_valid:
            resolved_lora_filenames: Mapping[str, str] = {}
            manifest = existing_manifest
        else:
            preparation = self.supervisor._resolve_assets(
                edit.job,
                scene_ids={item.scene_id},
            )
            if preparation.failures:
                raise GuiServiceError(
                    "; ".join(preparation.failures.get(item.scene_id, []))
                )
            resolved_lora_filenames = preparation.resolved_filenames
            manifest = {
                "job_id": item.job_id,
                "scene_id": item.scene_id,
                "revision": item.revision,
                "remake_mode": revision.remake_mode.value,
                "parameters": edit.document,
                "parameters_sha256": _document_sha256(edit.document),
                "resolved_lora_filenames": resolved_lora_filenames,
                "frame_path": str(frame_path),
                "video_path": str(clip_path),
                "route": (
                    "continuation" if continuation_route else "legacy"
                ),
                "status": "running",
            }
            write_json_atomic(manifest_path, manifest)
        self.store.update_scene_revision(
            item.job_id,
            item.scene_id,
            item.revision,
            state=SceneState.RUNNING,
            frame_path=str(frame_path) if frame_path.is_file() else None,
            video_path=str(clip_path) if video_checkpoint_valid else None,
        )
        return _PreparedRemakeItem(
            item=item,
            revision=revision,
            edit=edit,
            resolved_lora_filenames=resolved_lora_filenames,
            frame_path=frame_path,
            clip_path=clip_path,
            manifest_path=manifest_path,
            manifest=manifest,
            continuation_route=continuation_route,
            video_checkpoint_valid=video_checkpoint_valid,
        )

    def _remake_video_checkpoint_is_valid(
        self,
        *,
        item: RemakeItemRecord,
        revision: SceneRevision,
        edit: ValidatedSceneEdit,
        continuation_route: bool,
        frame_path: Path,
        clip_path: Path,
        manifest: Mapping[str, Any],
    ) -> bool:
        """Accept only a fully bound, probed raw-video remake checkpoint."""
        if (
            manifest.get("status") != "video_generated"
            or manifest.get("job_id") != item.job_id
            or manifest.get("scene_id") != item.scene_id
            or manifest.get("revision") != item.revision
            or manifest.get("route")
            != ("continuation" if continuation_route else "legacy")
            or manifest.get("parameters_sha256")
            != _document_sha256(edit.document)
        ):
            return False
        try:
            expected_frame = frame_path.resolve()
            expected_clip = clip_path.resolve()
            if (
                Path(str(manifest.get("frame_path", ""))).resolve()
                != expected_frame
                or Path(str(manifest.get("video_path", ""))).resolve()
                != expected_clip
                or (
                    revision.video_path is not None
                    and Path(revision.video_path).resolve() != expected_clip
                )
                or not expected_frame.is_file()
                or not expected_clip.is_file()
                or manifest.get("frame_sha256") != sha256_file(expected_frame)
                or manifest.get("video_sha256") != sha256_file(expected_clip)
            ):
                return False
            validate_video_profile((self.supervisor._video_probe(expected_clip),))
        except (AssemblyError, OSError, ValueError):
            return False
        return True

    def _remake_uses_continuation(
        self,
        item: RemakeItemRecord,
        edit: ValidatedSceneEdit,
    ) -> bool:
        """Keep an interrupted remake revision on its selected I2V route."""
        if item.prompt_stage == "i2v_continuation":
            return True
        if item.prompt_stage == "i2v_legacy":
            return False
        if (
            self.store.continuation_plan(
                item.job_id,
                item.scene_id,
                item.revision,
            )
            is not None
        ):
            return True
        return self.supervisor._uses_continuation(edit.scene, edit.workflow)

    def _run_remake_t2i(self, item: _PreparedRemakeItem) -> None:
        if item.frame_path.is_file():
            if item.item.prompt_stage == "t2i":
                self.store.clear_remake_item_prompt(
                    item.item.batch_id,
                    item.item.position,
                    expected_stage="t2i",
                )
            self.store.update_scene_revision(
                item.item.job_id,
                item.item.scene_id,
                item.item.revision,
                state=SceneState.RUNNING,
                frame_path=str(item.frame_path),
            )
            return
        build = build_t2i_api_workflow(
            item.edit.job,
            item.edit.scene,
            item.resolved_lora_filenames,
            delivery=self.supervisor.delivery,
            overrides=item.edit.workflow,
            revision=item.item.revision,
        )
        self.supervisor.run_or_reclaim_prompt(
            build.api,
            timeout_seconds=self.supervisor.settings.t2i_timeout_seconds,
            existing_prompt_id=(
                item.item.prompt_id
                if item.item.prompt_stage == "t2i"
                else None
            ),
            prompt_id_callback=lambda prompt_id: self.store.set_remake_item_prompt_id(
                item.item.batch_id,
                item.item.position,
                prompt_id,
                stage="t2i",
            ),
            active_check=lambda: self.store.continuation_work_is_active(
                item.item.job_id,
                item.item.scene_id,
                item.item.revision,
            ),
        )
        self.store.clear_remake_item_prompt(
            item.item.batch_id,
            item.item.position,
            expected_stage="t2i",
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
        current_item = next(
            candidate
            for candidate in self.store.remake_items(item.item.batch_id)
            if candidate.position == item.item.position
        )
        continuation_route = item.continuation_route
        prompt_stage = (
            "i2v_continuation"
            if continuation_route
            else "i2v_legacy"
        )
        if item.video_checkpoint_valid:
            LOGGER.info(
                "Remake checkpoint recovered | job=%s scene=%s revision=%s.",
                item.item.job_id,
                item.item.scene_id,
                item.item.revision,
            )
            return
        self._active_remake_revision = (
            item.item.job_id,
            item.item.scene_id,
            item.item.revision,
        )
        try:
            self.supervisor.render_i2v_scene(
                job=item.edit.job,
                scene=item.edit.scene,
                frame_path=item.frame_path,
                destination=item.clip_path,
                resolved_lora_filenames=item.resolved_lora_filenames,
                revision=item.item.revision,
                overrides=item.edit.workflow,
                deliver_to_discord=False,
                existing_prompt_id=(
                    current_item.prompt_id
                    if current_item.prompt_stage == "i2v_legacy"
                    else None
                ),
                prompt_id_callback=lambda prompt_id: self.store.set_remake_item_prompt_id(
                    item.item.batch_id,
                    item.item.position,
                    prompt_id,
                    stage=prompt_stage,
                ),
                continuation_route=continuation_route,
            )
        finally:
            self._active_remake_revision = None
        validate_video_profile((self.supervisor._video_probe(item.clip_path),))
        item.manifest.update(
            {
                "status": "video_generated",
                "route": (
                    "continuation" if continuation_route else "legacy"
                ),
                "parameters_sha256": _document_sha256(item.edit.document),
                "frame_path": str(item.frame_path),
                "frame_sha256": sha256_file(item.frame_path),
                "video_path": str(item.clip_path),
                "video_sha256": sha256_file(item.clip_path),
            }
        )
        write_json_atomic(item.manifest_path, item.manifest)
        self.store.update_scene_revision(
            item.item.job_id,
            item.item.scene_id,
            item.item.revision,
            state=SceneState.RUNNING,
            frame_path=str(item.frame_path),
            video_path=str(item.clip_path),
        )

    def _complete_remake_i2v(self, item: _PreparedRemakeItem) -> None:
        """Deliver after LTX unload, then atomically expose the revision as complete."""
        delivered = self.supervisor.deliver_scene_video(
            job=item.edit.job,
            scene=item.edit.scene,
            scene_path=item.clip_path,
            revision=item.item.revision,
            overrides=item.edit.workflow,
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
        item.manifest["discord_delivery"] = (
            "sent" if delivered else "failed_raw_preserved"
        )
        write_json_atomic(item.manifest_path, item.manifest)

    def _raise_if_comfy_unavailable(self, error: Exception) -> None:
        """Promote transport loss so the worker uses controlled ComfyUI recovery."""
        if isinstance(error, ComfyHttpError) and not self.supervisor.comfy.alive():
            raise FatalPipelineError(
                f"ComfyUI became unavailable during a GUI remake: {error}"
            ) from error

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
