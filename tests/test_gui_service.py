from __future__ import annotations

import copy
from fractions import Fraction
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from tenminvideomaker.assets import AssetResolution
from tenminvideomaker.comfy_http import ComfyHttpError
from tenminvideomaker.contracts import parse_job_payload
from tenminvideomaker.continuation import (
    build_scene_frame_plan,
    chunk_plan_documents,
)
from tenminvideomaker.gui_service import GuiServiceError, SupervisorController
from tenminvideomaker.review import scene_review_document
from tenminvideomaker.state_store import (
    PipelineState,
    PipelineStateStore,
    ManualFinalState,
    RemakeBatchState,
    RemakeMode,
    SceneState,
    ChunkState,
)
from tenminvideomaker.storage import StorageLayout
from tenminvideomaker.supervisor import (
    FatalPipelineError,
    PipelineSupervisor,
    SupervisorSettings,
)
from tenminvideomaker.assembly import VideoStreamInfo

from test_contracts import payload


class RemakeAssetManager:
    def resolve_or_download(self, lora, *, expected_base_model=None):
        filename = f"installed/{lora.name}.safetensors"
        return AssetResolution(
            lora.name,
            Path(filename),
            downloaded=False,
            local_filename=filename,
        )

    def require_local(self, requirement):
        return AssetResolution(
            requirement.filename,
            Path(requirement.filename),
            downloaded=False,
            local_filename=requirement.filename,
        )


class RemakeStageRecordingComfy:
    def __init__(self, storage: StorageLayout):
        self.storage = storage
        self.stages: list[str] = []
        self._prompts: dict[str, tuple[str, dict]] = {}
        self.free_calls = 0

    def queue_prompt(self, workflow):
        prompt_id = f"prompt-{len(self.stages) + 1}"
        saver = next(
            (
                node
                for node in workflow.values()
                if node["class_type"] == "10MinVideoMaker_SaveSceneFrame"
            ),
            None,
        )
        stage = "t2i" if saver is not None else "i2v"
        self.stages.append(stage)
        self._prompts[prompt_id] = (stage, saver or {})
        return prompt_id

    def wait_for_prompt(self, prompt_id, *, timeout_seconds):
        stage, saver = self._prompts[prompt_id]
        if stage == "t2i":
            frame = self.storage.scene_frame_path(
                saver["inputs"]["job_id"],
                saver["inputs"]["scene_id"],
                saver["inputs"]["revision"],
            )
            frame.parent.mkdir(parents=True, exist_ok=True)
            frame.write_bytes(b"png")
            return {"outputs": {}}
        return {
            "outputs": {
                "36": {
                    "gifs": [
                        {
                            "filename": "scene.mp4",
                            "subfolder": "10MinVideoMaker/test",
                            "type": "temp",
                        }
                    ]
                }
            }
        }

    def download_output(self, metadata, destination):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"mp4")
        return destination

    def completed_prompt(self, prompt_id):
        return None

    def prompt_is_queued(self, prompt_id):
        return prompt_id in self._prompts

    def free_memory(self):
        self.free_calls += 1


class RestartReclaimComfy(RemakeStageRecordingComfy):
    """Lose transport after queue, then reclaim the same prompt after restart."""

    def __init__(self, storage: StorageLayout):
        super().__init__(storage)
        self.alive_flag = True
        self.waited_prompt_ids: list[str] = []
        self.stop_event = None

    def alive(self):
        return self.alive_flag

    def completed_prompt(self, prompt_id):
        if not self.alive_flag:
            raise ComfyHttpError("ComfyUI connection lost")
        return None

    def prompt_is_queued(self, prompt_id):
        if not self.alive_flag:
            raise ComfyHttpError("ComfyUI connection lost")
        return prompt_id in self._prompts

    def wait_for_prompt(self, prompt_id, *, timeout_seconds):
        self.waited_prompt_ids.append(prompt_id)
        if len(self.waited_prompt_ids) == 1:
            self.alive_flag = False
            raise ComfyHttpError("ComfyUI connection lost")
        result = super().wait_for_prompt(
            prompt_id,
            timeout_seconds=timeout_seconds,
        )
        if self.stop_event is not None:
            self.stop_event.set()
        return result


class GuiServiceTests(unittest.TestCase):
    def _create_interrupted_video_checkpoint(
        self,
        root: Path,
    ):
        storage = StorageLayout(root / "storage")
        storage.ensure()
        store = PipelineStateStore(storage.database_path)
        job = parse_job_payload(payload())
        scene = job.scenes[0]
        store.claim_job(job)
        frame = storage.scene_frame_path(job.job_id, scene.scene_id)
        frame.parent.mkdir(parents=True, exist_ok=True)
        frame.write_bytes(b"cached-frame")
        store.set_scene_state(
            job.job_id,
            scene.scene_id,
            SceneState.SUCCEEDED,
            frame_path=str(frame),
        )
        store.ensure_original_scene_revision(
            job.job_id,
            scene.scene_id,
            parameters=scene_review_document(job, scene),
            frame_path=str(frame),
        )
        batch_id, _created = store.create_remake_batch(
            [
                (
                    job.job_id,
                    scene.scene_id,
                    RemakeMode.VIDEO_ONLY,
                    scene_review_document(job, scene),
                )
            ]
        )
        store.queue_remake_batch(batch_id, "after_current")
        store.set_remake_batch_state(batch_id, RemakeBatchState.RUNNING)
        store.set_remake_item_state(batch_id, 1, SceneState.RUNNING)
        first_comfy = RemakeStageRecordingComfy(storage)
        valid_probe = lambda path: VideoStreamInfo(
            Path(path),
            768,
            1344,
            Fraction(24, 1),
        )
        supervisor = PipelineSupervisor(
            store=store,
            mail_client=Mock(),
            asset_manager=RemakeAssetManager(),
            comfy=first_comfy,
            settings=SupervisorSettings(1, 10, 10, 2),
            storage=storage,
            video_probe=valid_probe,
        )
        controller = SupervisorController(supervisor, storage)
        item = store.remake_items(batch_id)[0]
        prepared = controller._prepare_remake_item(item)
        controller._run_remake_i2v(prepared)

        self.assertEqual(first_comfy.stages, ["i2v"])
        self.assertEqual(
            store.recover_interrupted_remake_batches(),
            (batch_id,),
        )
        return storage, store, job, batch_id, valid_probe

    def test_start_recovers_interrupted_remakes_before_worker_launch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events: list[str] = []
            supervisor = Mock()
            supervisor.store.recover_interrupted_remake_batches.side_effect = lambda: (
                events.append("recover"),
                ("batch-1",),
            )[1]
            controller = SupervisorController(
                supervisor,
                StorageLayout(root / "storage"),
            )
            fake_thread = Mock()
            fake_thread.start.side_effect = lambda: events.append("start")

            with patch(
                "tenminvideomaker.gui_service.threading.Thread",
                return_value=fake_thread,
            ):
                controller.start()

            self.assertEqual(events, ["recover", "start"])
            supervisor.store.recover_interrupted_remake_batches.assert_called_once_with()

    def test_worker_routes_fatal_pipeline_error_through_controlled_recovery(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = PipelineStateStore(root / "pipeline.sqlite3")
            supervisor = Mock()
            supervisor.store = store
            supervisor.settings.poll_interval_seconds = 1
            controller = SupervisorController(
                supervisor,
                StorageLayout(root / "storage"),
                idle_wait_seconds=0,
            )
            fatal = FatalPipelineError("ComfyUI server unavailable")

            def fail_once():
                controller._stop.set()
                raise fatal

            supervisor.tick.side_effect = fail_once

            controller._worker()

            supervisor.handle_fatal.assert_called_once_with(fatal)
            self.assertEqual(controller.last_error, str(fatal))

    def test_remake_fatal_error_escapes_item_failure_handler_for_worker_recovery(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = PipelineStateStore(root / "pipeline.sqlite3")
            job = parse_job_payload(payload())
            store.claim_job(job)
            batch_id, _created = store.create_remake_batch(
                [
                    (
                        job.job_id,
                        job.scenes[0].scene_id,
                        RemakeMode.IMAGE_AND_VIDEO,
                        scene_review_document(job, job.scenes[0]),
                    )
                ]
            )
            store.queue_remake_batch(batch_id, "after_current")
            batch = store.next_queued_remake_batch()
            item = store.remake_items(batch_id)[0]
            supervisor = Mock()
            supervisor.store = store
            controller = SupervisorController(
                supervisor,
                StorageLayout(root / "storage"),
            )
            prepared = Mock()
            prepared.item = item
            prepared.revision.remake_mode = RemakeMode.IMAGE_AND_VIDEO
            prepared.edit.job.character.base_model = "Anima"
            fatal = FatalPipelineError("ComfyUI restart required")

            with (
                patch.object(
                    controller,
                    "_prepare_remake_item",
                    return_value=prepared,
                ),
                patch.object(
                    controller,
                    "_run_remake_t2i",
                    side_effect=fatal,
                ),
            ):
                with self.assertRaises(FatalPipelineError) as raised:
                    controller._run_remake_batch(batch)

            self.assertIs(raised.exception, fatal)
            self.assertEqual(
                store.remake_items(batch_id)[0].state,
                SceneState.RUNNING,
            )
            self.assertEqual(
                next(
                    candidate
                    for candidate in store.list_remake_batches()
                    if candidate.batch_id == batch_id
                ).state,
                RemakeBatchState.RUNNING,
            )

    def test_remake_server_loss_recovers_batch_and_reclaims_prompt_without_requeue(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            storage = StorageLayout(root / "storage")
            storage.ensure()
            store = PipelineStateStore(storage.database_path)
            job = parse_job_payload(payload())
            scene = job.scenes[0]
            store.claim_job(job)
            store.ensure_original_scene_revision(
                job.job_id,
                scene.scene_id,
                parameters=scene_review_document(job, scene),
            )
            store.abandon_job(job.job_id, reason="historical remake test")
            batch_id, created = store.create_remake_batch(
                [
                    (
                        job.job_id,
                        scene.scene_id,
                        RemakeMode.IMAGE_AND_VIDEO,
                        scene_review_document(job, scene),
                    )
                ]
            )
            revision = created[0][2]
            store.queue_remake_batch(batch_id, "after_current")
            comfy = RestartReclaimComfy(storage)
            restart_calls: list[str] = []

            def restart_comfy():
                restart_calls.append("restart")
                comfy.alive_flag = True
                return True

            supervisor = PipelineSupervisor(
                store=store,
                mail_client=Mock(),
                asset_manager=RemakeAssetManager(),
                comfy=comfy,
                settings=SupervisorSettings(1, 10, 10, 2),
                restart_comfy=restart_comfy,
                storage=storage,
            )
            controller = SupervisorController(
                supervisor,
                storage,
                idle_wait_seconds=0,
            )
            comfy.stop_event = controller._stop

            controller._worker()

            self.assertEqual(restart_calls, ["restart"])
            self.assertEqual(comfy.stages, ["t2i"])
            self.assertEqual(
                comfy.waited_prompt_ids,
                ["prompt-1", "prompt-1"],
            )
            self.assertTrue(
                storage.scene_frame_path(
                    job.job_id,
                    scene.scene_id,
                    revision,
                ).is_file()
            )
            item = store.remake_items(batch_id)[0]
            self.assertEqual(item.state, SceneState.RUNNING)
            self.assertIsNone(item.prompt_id)
            self.assertIsNone(item.prompt_stage)
            self.assertEqual(
                next(
                    candidate
                    for candidate in store.list_remake_batches()
                    if candidate.batch_id == batch_id
                ).state,
                RemakeBatchState.RUNNING,
            )
            self.assertEqual(store.snapshot().state, PipelineState.IDLE)

    def test_remake_recovery_uses_verified_post_video_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            storage, store, _job, batch_id, valid_probe = (
                self._create_interrupted_video_checkpoint(root)
            )
            # The atomic manifest is the commit marker even if the process
            # stopped before mirroring its path into SQLite.
            with store._connection() as connection:
                connection.execute(
                    """
                    UPDATE scene_revisions SET video_path = NULL
                    WHERE job_id = ? AND scene_id = 1 AND revision = 2
                    """,
                    (_job.job_id,),
                )
            second_comfy = RemakeStageRecordingComfy(storage)
            supervisor = PipelineSupervisor(
                store=store,
                mail_client=Mock(),
                asset_manager=RemakeAssetManager(),
                comfy=second_comfy,
                settings=SupervisorSettings(1, 10, 10, 2),
                storage=storage,
                video_probe=valid_probe,
            )
            controller = SupervisorController(supervisor, storage)

            controller._run_remake_batch(store.next_queued_remake_batch())

            self.assertEqual(second_comfy.stages, [])
            item = store.remake_items(batch_id)[0]
            self.assertEqual(item.state, SceneState.SUCCEEDED)
            self.assertIsNone(item.prompt_id)
            self.assertIsNone(item.prompt_stage)
            self.assertEqual(
                next(
                    batch
                    for batch in store.list_remake_batches()
                    if batch.batch_id == batch_id
                ).state,
                RemakeBatchState.SUCCEEDED,
            )

    def test_corrupt_post_video_checkpoint_is_regenerated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            storage, store, job, batch_id, valid_probe = (
                self._create_interrupted_video_checkpoint(root)
            )
            clip = storage.scene_clip_path(job.job_id, 1, 2)
            clip.write_bytes(b"corrupted-after-checkpoint")
            second_comfy = RemakeStageRecordingComfy(storage)
            supervisor = PipelineSupervisor(
                store=store,
                mail_client=Mock(),
                asset_manager=RemakeAssetManager(),
                comfy=second_comfy,
                settings=SupervisorSettings(1, 10, 10, 2),
                storage=storage,
                video_probe=valid_probe,
            )
            controller = SupervisorController(supervisor, storage)

            controller._run_remake_batch(store.next_queued_remake_batch())

            self.assertEqual(second_comfy.stages, ["i2v"])
            self.assertEqual(
                store.remake_items(batch_id)[0].state,
                SceneState.SUCCEEDED,
            )

    def test_remake_continuation_preflight_runs_before_asset_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            storage = StorageLayout(root / "storage")
            storage.ensure()
            store = PipelineStateStore(storage.database_path)
            job = parse_job_payload(payload())
            scene = job.scenes[0]
            store.claim_job(job)
            document = scene_review_document(job, scene)
            document["i2v"]["temporal_continuation"] = {"enabled": True}
            document["i2v"]["segments"] = [
                {
                    "index": 0,
                    "requested_duration_seconds": 5.0,
                    "positive_prompt": "An incomplete temporal beat.",
                }
            ]
            batch_id, _created = store.create_remake_batch(
                [
                    (
                        job.job_id,
                        scene.scene_id,
                        RemakeMode.IMAGE_AND_VIDEO,
                        document,
                    )
                ]
            )
            item = store.remake_items(batch_id)[0]
            assets = Mock(wraps=RemakeAssetManager())
            supervisor = PipelineSupervisor(
                store=store,
                mail_client=Mock(),
                asset_manager=assets,
                comfy=Mock(),
                settings=SupervisorSettings(
                    1,
                    10,
                    10,
                    2,
                    continuation_mode="explicit",
                ),
                storage=storage,
            )
            controller = SupervisorController(supervisor, storage)

            with self.assertRaisesRegex(
                GuiServiceError,
                "Continuation preflight failed before assets or rendering",
            ):
                controller._prepare_remake_item(item)

            assets.resolve_or_download.assert_not_called()
            assets.require_local.assert_not_called()
            supervisor.comfy.queue_prompt.assert_not_called()

    def test_chunk_progress_projects_durable_phase_and_resume_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            storage = StorageLayout(root / "storage")
            store = PipelineStateStore(storage.database_path)
            job = parse_job_payload(payload())
            store.claim_job(job)
            scene = job.scenes[0]
            plan = build_scene_frame_plan(
                job_id=job.job_id,
                scene_id=scene.scene_id,
                revision=1,
                requested_duration_seconds=10,
                base_seed=scene.i2v.seed,
                fallback_prompt=scene.i2v.prompt,
                fallback_negative=scene.i2v.negative,
            )
            store.ensure_continuation_plan(
                job.job_id,
                scene.scene_id,
                1,
                plan.fingerprint(),
                plan.to_document(),
            )
            store.plan_chunks(
                job.job_id,
                scene.scene_id,
                1,
                plan.fingerprint(),
                chunk_plan_documents(plan.chunks),
            )
            supervisor = PipelineSupervisor(
                store=store,
                mail_client=Mock(),
                asset_manager=Mock(),
                comfy=Mock(),
                settings=SupervisorSettings(1, 10, 10, 2),
                storage=storage,
            )
            controller = SupervisorController(supervisor, storage)
            supervisor.comfy.queue_counts.return_value = (0, 0)
            self.assertEqual(
                controller.status_document()["continuation_mode"],
                "explicit",
            )

            ready = controller.chunk_progress_document(
                job.job_id,
                scene.scene_id,
                1,
            )
            self.assertEqual(ready["phase"], "first_pass")
            self.assertEqual(ready["current_chunk"], 1)
            self.assertEqual(ready["total_chunks"], 2)
            self.assertFalse(ready["resumed"])
            self.assertEqual(ready["timeline_frames"], 240)
            self.assertEqual(ready["generation_master_frames"], 241)

            attempt = store.begin_chunk_attempt(
                job.job_id,
                scene.scene_id,
                1,
                0,
                seed=plan.chunks[0].seed,
                parameters={"plan": plan.fingerprint()},
            )
            store.update_chunk_attempt(
                job.job_id,
                scene.scene_id,
                1,
                0,
                attempt.attempt_number,
                ChunkState.GENERATING_STAGE2,
            )
            refining = controller.chunk_progress_document(
                job.job_id,
                scene.scene_id,
                1,
            )
            self.assertEqual(refining["phase"], "upscale_refinement")
            self.assertFalse(refining["resumed"])

            # A normal multi-chunk run is not a "resume" merely because earlier
            # chunks completed.  Once the durable scene revision has its final
            # clip, historical UI status must say complete rather than assembly.
            store.ensure_original_scene_revision(
                job.job_id,
                scene.scene_id,
                parameters=scene_review_document(job, scene),
            )
            for chunk in plan.chunks:
                attempts = store.chunk_attempts(
                    job.job_id,
                    scene.scene_id,
                    1,
                    chunk.index,
                )
                active = attempts[-1] if attempts else store.begin_chunk_attempt(
                    job.job_id,
                    scene.scene_id,
                    1,
                    chunk.index,
                    seed=chunk.seed,
                    parameters={"plan": plan.fingerprint()},
                )
                if active.state != ChunkState.COMPLETE:
                    active = store.update_chunk_attempt(
                        job.job_id,
                        scene.scene_id,
                        1,
                        chunk.index,
                        active.attempt_number,
                        ChunkState.COMPLETE,
                        artifact_hash=f"artifact-{chunk.index}",
                        video_path=f"chunk-{chunk.index}.mp4",
                    )
                store.select_chunk_attempt(
                    job.job_id,
                    scene.scene_id,
                    1,
                    chunk.index,
                    active.attempt_number,
                    artifact_hash=active.artifact_hash,
                )
            assembling = controller.chunk_progress_document(
                job.job_id,
                scene.scene_id,
                1,
            )
            self.assertEqual(assembling["phase"], "assembly")
            self.assertFalse(assembling["resumed"])

            store.update_scene_revision(
                job.job_id,
                scene.scene_id,
                1,
                state=SceneState.SUCCEEDED,
                video_path="scene.mp4",
            )
            complete = controller.chunk_progress_document(
                job.job_id,
                scene.scene_id,
                1,
            )
            self.assertEqual(complete["phase"], "complete")
            self.assertTrue(complete["output_available"])
            self.assertFalse(complete["resumed"])

    def test_manual_final_uses_latest_selected_revision_without_generation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            storage = StorageLayout(root / "storage")
            store = PipelineStateStore(storage.database_path)
            job = parse_job_payload(payload())
            store.claim_job(job)
            original = storage.scene_clip_path(job.job_id, 1, 1)
            remake = storage.scene_clip_path(job.job_id, 1, 2)
            for clip in (original, remake):
                clip.parent.mkdir(parents=True, exist_ok=True)
                clip.write_bytes(b"mp4")
            store.set_scene_state(
                job.job_id,
                1,
                SceneState.SUCCEEDED,
                video_path=str(original),
            )
            store.create_scene_revision(
                job.job_id,
                1,
                remake_mode=RemakeMode.IMAGE_AND_VIDEO,
                parameters={"version": 1},
                state=SceneState.SUCCEEDED,
                video_path=str(original),
            )
            store.create_scene_revision(
                job.job_id,
                1,
                remake_mode=RemakeMode.VIDEO_ONLY,
                parameters={"version": 2},
                state=SceneState.SUCCEEDED,
                video_path=str(remake),
            )

            class RecordingAssembler:
                def __init__(self):
                    self.clips = ()

                def stitch(self, job_id, clips, _concat_directory):
                    self.clips = tuple(Path(item) for item in clips)
                    output = storage.final_path(job_id)
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_bytes(b"final")
                    return output

            assembler = RecordingAssembler()
            comfy = Mock()
            supervisor = PipelineSupervisor(
                store=store,
                mail_client=Mock(),
                asset_manager=Mock(),
                comfy=comfy,
                assembler=assembler,
                settings=SupervisorSettings(1, 10, 10, 2),
                storage=storage,
                video_probe=lambda path: VideoStreamInfo(
                    Path(path), 768, 1344, Fraction(24, 1)
                ),
            )
            controller = SupervisorController(supervisor, storage)
            request = controller.queue_manual_final(job.job_id)

            controller._run_manual_final(request)

            self.assertEqual(assembler.clips, (remake,))
            self.assertEqual(
                store.latest_manual_final(job.job_id).state,
                ManualFinalState.SUCCEEDED,
            )
            self.assertEqual(store.list_jobs()[0].final_path, str(storage.final_path(job.job_id)))
            comfy.queue_prompt.assert_not_called()

    def test_interrupt_cancels_only_project_prompts_and_preserves_job_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = PipelineStateStore(root / "pipeline.sqlite3")
            job = parse_job_payload(payload())
            store.claim_job(job)
            store.begin_scene_stage(
                job.job_id,
                1,
                PipelineState.RUNNING_I2V,
                prompt_id="project-prompt",
            )
            supervisor = Mock()
            supervisor.store = store
            events: list[str] = []
            original_abandon = store.abandon_job
            store.abandon_job = Mock(
                side_effect=lambda *args, **kwargs: (
                    events.append("abandon"),
                    original_abandon(*args, **kwargs),
                )[1]
            )
            supervisor.comfy.cancel_project_prompts.side_effect = lambda: (
                events.append("cancel"),
                ("project-prompt",),
            )[1]
            controller = SupervisorController(
                supervisor,
                StorageLayout(root / "storage"),
            )

            cancelled = controller.interrupt_current_job()

            self.assertEqual(cancelled, ("project-prompt",))
            self.assertEqual(events, ["abandon", "cancel"])
            supervisor.comfy.cancel_project_prompts.assert_called_once_with()
            self.assertEqual(store.snapshot().state, PipelineState.IDLE)
            self.assertEqual(store.scene_records(job.job_id)[0].state, SceneState.CANCELLED)
            self.assertEqual(store.load_job(job.job_id).job_id, job.job_id)

    def test_cancel_current_project_from_error_preserves_history_and_wakes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = PipelineStateStore(root / "pipeline.sqlite3")
            job = parse_job_payload(payload())
            store.claim_job(job)
            store.set_scene_state(
                job.job_id,
                1,
                SceneState.FAILED,
                error="frame size mismatch",
            )
            store.transition(
                PipelineState.ERROR,
                job_id=job.job_id,
                error="no successful scenes",
            )
            supervisor = Mock()
            supervisor.store = store
            controller = SupervisorController(
                supervisor,
                StorageLayout(root / "storage"),
            )
            controller.wake = Mock()

            self.assertTrue(controller.can_cancel_current_project())
            result = controller.cancel_current_project()

            self.assertEqual(result["job_id"], job.job_id)
            self.assertEqual(result["pipeline_state"], PipelineState.IDLE.value)
            self.assertEqual(result["cancelled_prompts"], [])
            supervisor.comfy.cancel_project_prompts.assert_not_called()
            self.assertEqual(store.snapshot().state, PipelineState.IDLE)
            self.assertIsNone(store.snapshot().job_id)
            self.assertEqual(
                store.scene_records(job.job_id)[0].state,
                SceneState.CANCELLED,
            )
            self.assertEqual(store.load_job(job.job_id).job_id, job.job_id)
            controller.wake.assert_called_once_with()
            self.assertFalse(controller.can_cancel_current_project())

    def test_cancel_current_project_rejects_when_pipeline_already_free(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = PipelineStateStore(root / "pipeline.sqlite3")
            supervisor = Mock()
            supervisor.store = store
            controller = SupervisorController(
                supervisor,
                StorageLayout(root / "storage"),
            )

            with self.assertRaisesRegex(GuiServiceError, "no held project"):
                controller.cancel_current_project()

    def test_remake_batch_generates_all_frames_before_any_video(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            storage = StorageLayout(root / "storage")
            store = PipelineStateStore(storage.database_path)
            raw = payload()
            for scene_id in range(2, 6):
                scene = copy.deepcopy(raw["scenes"][0])
                scene["id"] = scene_id
                scene["title"] = f"Batched remake {scene_id}"
                scene["t2i"]["seed"] += scene_id
                scene["i2v"]["seed"] += scene_id
                raw["scenes"].append(scene)
            job = parse_job_payload(raw)
            store.claim_job(job)
            cached_video_only_frame = storage.scene_frame_path(job.job_id, 5)
            cached_video_only_frame.parent.mkdir(parents=True, exist_ok=True)
            cached_video_only_frame.write_bytes(b"png")
            store.set_scene_state(
                job.job_id,
                5,
                SceneState.PENDING,
                frame_path=str(cached_video_only_frame),
            )
            items = [
                (
                    job.job_id,
                    scene.scene_id,
                    (
                        RemakeMode.IMAGE_AND_VIDEO
                        if scene.scene_id < 5
                        else RemakeMode.VIDEO_ONLY
                    ),
                    scene_review_document(job, scene),
                )
                for scene in job.scenes
            ]
            batch_id, _ = store.create_remake_batch(items)
            store.queue_remake_batch(batch_id, "after_current")
            comfy = RemakeStageRecordingComfy(storage)
            supervisor = PipelineSupervisor(
                store=store,
                mail_client=Mock(),
                asset_manager=RemakeAssetManager(),
                comfy=comfy,
                settings=SupervisorSettings(1, 10, 10, 2),
                storage=storage,
                video_probe=lambda path: VideoStreamInfo(
                    Path(path),
                    768,
                    1344,
                    Fraction(24, 1),
                ),
            )
            supervisor.delivery = Mock()
            supervisor.deliver_scene_video = Mock(
                side_effect=lambda **_kwargs: comfy.stages.append("delivery")
            )
            controller = SupervisorController(supervisor, storage)
            batch = store.next_queued_remake_batch()
            self.assertIsNotNone(batch)

            controller._run_remake_batch(batch)

            self.assertEqual(
                comfy.stages,
                [
                    "t2i",
                    "t2i",
                    "t2i",
                    "t2i",
                    "i2v",
                    "i2v",
                    "i2v",
                    "i2v",
                    "i2v",
                    "delivery",
                    "delivery",
                    "delivery",
                    "delivery",
                    "delivery",
                ],
            )
            self.assertEqual(comfy.free_calls, 7)
            self.assertTrue(
                all(
                    call.kwargs["scene_path"].is_file()
                    and call.kwargs["revision"] >= 1
                    for call in supervisor.deliver_scene_video.call_args_list
                )
            )
            self.assertTrue(
                all(
                    item.state == SceneState.SUCCEEDED
                    for item in store.remake_items(batch_id)
                )
            )
            self.assertEqual(
                next(
                    item
                    for item in store.list_remake_batches()
                    if item.batch_id == batch_id
                ).state,
                RemakeBatchState.SUCCEEDED,
            )


if __name__ == "__main__":
    unittest.main()
