from __future__ import annotations

import copy
from fractions import Fraction
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from tenminvideomaker.assets import AssetResolution
from tenminvideomaker.contracts import parse_job_payload
from tenminvideomaker.gui_service import SupervisorController
from tenminvideomaker.review import scene_review_document
from tenminvideomaker.state_store import (
    PipelineState,
    PipelineStateStore,
    ManualFinalState,
    RemakeBatchState,
    RemakeMode,
    SceneState,
)
from tenminvideomaker.storage import StorageLayout
from tenminvideomaker.supervisor import PipelineSupervisor, SupervisorSettings
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

    def free_memory(self):
        self.free_calls += 1


class GuiServiceTests(unittest.TestCase):
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
            supervisor.comfy.cancel_project_prompts.return_value = ("project-prompt",)
            controller = SupervisorController(
                supervisor,
                StorageLayout(root / "storage"),
            )

            cancelled = controller.interrupt_current_job()

            self.assertEqual(cancelled, ("project-prompt",))
            supervisor.comfy.cancel_project_prompts.assert_called_once_with()
            self.assertEqual(store.snapshot().state, PipelineState.IDLE)
            self.assertEqual(store.scene_records(job.job_id)[0].state, SceneState.CANCELLED)
            self.assertEqual(store.load_job(job.job_id).job_id, job.job_id)

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
            )
            controller = SupervisorController(supervisor, storage)
            batch = store.next_queued_remake_batch()
            self.assertIsNotNone(batch)

            controller._run_remake_batch(batch)

            self.assertEqual(
                comfy.stages,
                ["t2i", "t2i", "t2i", "t2i", "i2v", "i2v", "i2v", "i2v", "i2v"],
            )
            self.assertEqual(comfy.free_calls, 2)
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
