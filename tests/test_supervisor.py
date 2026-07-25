from __future__ import annotations

import copy
from fractions import Fraction
from pathlib import Path
import tempfile
import unittest

from tenminvideomaker.assembly import VideoStreamInfo
from tenminvideomaker.assets import AssetResolution
from tenminvideomaker.comfy_http import ComfyHttpError
from tenminvideomaker.contracts import parse_job_payload
from tenminvideomaker.state_store import PipelineState, PipelineStateStore, SceneState
from tenminvideomaker.supervisor import PipelineSupervisor, SupervisorSettings

from test_contracts import payload


class FakeMailClient:
    def __init__(self):
        self.requests = []

    def send_request(self, *, previous_job_id=None, succeeded=None):
        self.requests.append((previous_job_id, succeeded))
        return "message-id"


class FakeAssetManager:
    def resolve_or_download(self, lora):
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


class OneMissingAssetManager(FakeAssetManager):
    def resolve_or_download(self, lora):
        if lora.name == "Missing Scene LoRA":
            return AssetResolution(lora.name, None, downloaded=False, error="download failed")
        return super().resolve_or_download(lora)


class AllMissingAssetManager(FakeAssetManager):
    def __init__(self):
        self.resolve_calls = 0

    def resolve_or_download(self, lora):
        self.resolve_calls += 1
        return AssetResolution(
            lora.name,
            None,
            downloaded=False,
            error="Civitai authentication required",
        )


class FakeComfy:
    def __init__(self, frame_path: Path):
        self.frame_path = frame_path
        self.workflows = []
        self.free_calls = 0

    def queue_prompt(self, workflow):
        self.workflows.append(workflow)
        return f"prompt-{len(self.workflows)}"

    def wait_for_prompt(self, prompt_id, *, timeout_seconds):
        if prompt_id == "prompt-1":
            self.frame_path.parent.mkdir(parents=True, exist_ok=True)
            self.frame_path.write_bytes(b"png")
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

    def alive(self):
        return True


class FakeAssembler:
    def __init__(self, final_path: Path):
        self.final_path = final_path
        self.calls = []

    def stitch(self, job_id, clips, concat_directory):
        self.calls.append((job_id, list(clips), Path(concat_directory)))
        self.final_path.write_bytes(b"final")
        return self.final_path


class RetryOnceComfy(FakeComfy):
    def wait_for_prompt(self, prompt_id, *, timeout_seconds):
        if prompt_id == "prompt-1":
            raise ComfyHttpError("transient sampler failure")
        if prompt_id == "prompt-2":
            self.frame_path.parent.mkdir(parents=True, exist_ok=True)
            self.frame_path.write_bytes(b"png")
            return {"outputs": {}}
        return super().wait_for_prompt("prompt-2", timeout_seconds=timeout_seconds)


class SupervisorTests(unittest.TestCase):
    def test_process_job_runs_t2i_then_i2v_stitches_and_requests_next(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            job = parse_job_payload(payload())
            store = PipelineStateStore(root / "pipeline.sqlite3")
            store.claim_job(job)
            frame = root / "frames" / "scene_0001.png"
            clip = root / "clips" / "scene_0001.mp4"
            mail = FakeMailClient()
            comfy = FakeComfy(frame)
            assembler = FakeAssembler(root / "final.mp4")
            supervisor = PipelineSupervisor(
                store=store,
                mail_client=mail,
                asset_manager=FakeAssetManager(),
                comfy=comfy,
                assembler=assembler,
                settings=SupervisorSettings(
                    poll_interval_seconds=1,
                    t2i_timeout_seconds=10,
                    i2v_timeout_seconds=10,
                    max_stage_attempts=2,
                ),
                frame_path_factory=lambda _job, _scene: frame,
                clip_path_factory=lambda _job, _scene: clip,
                video_probe=lambda path: VideoStreamInfo(
                    Path(path),
                    704,
                    1248,
                    Fraction(24, 1),
                ),
            )

            supervisor.process_job(job)

            self.assertEqual(len(comfy.workflows), 2)
            self.assertIn(
                "10MinVideoMaker_SaveSceneFrame",
                {node["class_type"] for node in comfy.workflows[0].values()},
            )
            t2i_lora = next(
                node
                for node in comfy.workflows[0].values()
                if node["class_type"] == "LoraLoader"
            )
            self.assertEqual(
                t2i_lora["inputs"]["lora_name"],
                "installed/Elsa Frozen Anima.safetensors",
            )
            image_loader = next(
                node
                for node in comfy.workflows[1].values()
                if node["class_type"] == "VHS_LoadImagePath"
            )
            self.assertEqual(image_loader["inputs"]["image"], str(frame))
            record = store.scene_records(job.job_id)[0]
            self.assertEqual(record.state, SceneState.SUCCEEDED)
            self.assertEqual(record.frame_path, str(frame))
            self.assertEqual(record.video_path, str(clip))
            self.assertEqual(store.snapshot().state, PipelineState.WAITING_FOR_GROK)
            self.assertEqual(mail.requests, [(job.job_id, True)])
            self.assertEqual(len(assembler.calls), 1)
            self.assertGreaterEqual(comfy.free_calls, 3)

    def test_transient_comfy_failure_retries_only_the_unfinished_stage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            job = parse_job_payload(payload())
            store = PipelineStateStore(root / "pipeline.sqlite3")
            store.claim_job(job)
            frame = root / "frames" / "scene_0001.png"
            clip = root / "clips" / "scene_0001.mp4"
            comfy = RetryOnceComfy(frame)
            supervisor = PipelineSupervisor(
                store=store,
                mail_client=FakeMailClient(),
                asset_manager=FakeAssetManager(),
                comfy=comfy,
                assembler=FakeAssembler(root / "final.mp4"),
                settings=SupervisorSettings(1, 10, 10, 2),
                frame_path_factory=lambda _job, _scene: frame,
                clip_path_factory=lambda _job, _scene: clip,
                video_probe=lambda path: VideoStreamInfo(
                    Path(path), 704, 1248, Fraction(24, 1)
                ),
            )

            supervisor.process_job(job)

            record = store.scene_records(job.job_id)[0]
            self.assertEqual(record.state, SceneState.SUCCEEDED)
            self.assertEqual(record.t2i_attempts, 2)
            self.assertEqual(record.i2v_attempts, 1)
            self.assertEqual(len(comfy.workflows), 3)

    def test_scene_asset_failure_does_not_abort_other_scenes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            raw = payload()
            second = copy.deepcopy(raw["scenes"][0])
            second["id"] = 2
            second["title"] = "Missing asset scene"
            second["i2v"]["loras"] = [
                {
                    "name": "Missing Scene LoRA",
                    "download_url": "https://example.invalid/missing.safetensors",
                    "weight": 0.8,
                }
            ]
            raw["scenes"].append(second)
            job = parse_job_payload(raw)
            store = PipelineStateStore(root / "pipeline.sqlite3")
            store.claim_job(job)
            frame = root / "frames" / "scene_0001.png"
            clip = root / "clips" / "scene_0001.mp4"
            mail = FakeMailClient()
            assembler = FakeAssembler(root / "final.mp4")
            supervisor = PipelineSupervisor(
                store=store,
                mail_client=mail,
                asset_manager=OneMissingAssetManager(),
                comfy=FakeComfy(frame),
                assembler=assembler,
                settings=SupervisorSettings(1, 10, 10, 2),
                frame_path_factory=lambda _job, scene: frame if scene == 1 else root / "unused.png",
                clip_path_factory=lambda _job, scene: clip if scene == 1 else root / "unused.mp4",
                video_probe=lambda path: VideoStreamInfo(
                    Path(path), 704, 1248, Fraction(24, 1)
                ),
            )

            supervisor.process_job(job)

            records = store.scene_records(job.job_id)
            self.assertEqual([record.state for record in records], [SceneState.SUCCEEDED, SceneState.FAILED])
            self.assertIn("download failed", records[1].error)
            self.assertEqual(len(assembler.calls), 1)
            self.assertEqual(mail.requests, [(job.job_id, False)])

    def test_all_asset_failures_pause_saved_job_without_requesting_another(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            job = parse_job_payload(payload())
            store = PipelineStateStore(root / "pipeline.sqlite3")
            store.claim_job(job)
            mail = FakeMailClient()
            assets = AllMissingAssetManager()
            comfy = FakeComfy(root / "unused.png")
            supervisor = PipelineSupervisor(
                store=store,
                mail_client=mail,
                asset_manager=assets,
                comfy=comfy,
                assembler=FakeAssembler(root / "final.mp4"),
                settings=SupervisorSettings(1, 10, 10, 2),
            )

            supervisor.process_job(job)

            snapshot = store.snapshot()
            self.assertEqual(snapshot.state, PipelineState.ERROR)
            self.assertEqual(snapshot.job_id, job.job_id)
            self.assertIn("Asset preparation failed for all", snapshot.error)
            self.assertEqual(
                store.scene_states(job.job_id),
                {1: SceneState.FAILED},
            )
            self.assertEqual(mail.requests, [])
            self.assertEqual(comfy.workflows, [])

            calls_before_paused_tick = assets.resolve_calls
            supervisor.tick()
            self.assertEqual(assets.resolve_calls, calls_before_paused_tick)
            self.assertEqual(mail.requests, [])

    def test_assembly_profile_failure_pauses_and_preserves_completed_clip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            job = parse_job_payload(payload())
            store = PipelineStateStore(root / "pipeline.sqlite3")
            store.claim_job(job)
            frame = root / "frame.png"
            clip = root / "scene.mp4"
            frame.write_bytes(b"png")
            clip.write_bytes(b"mp4")
            store.set_scene_state(
                job.job_id,
                1,
                SceneState.SUCCEEDED,
                frame_path=str(frame),
                video_path=str(clip),
            )
            mail = FakeMailClient()
            assembler = FakeAssembler(root / "final.mp4")
            supervisor = PipelineSupervisor(
                store=store,
                mail_client=mail,
                asset_manager=FakeAssetManager(),
                comfy=FakeComfy(frame),
                assembler=assembler,
                settings=SupervisorSettings(1, 10, 10, 2),
                video_probe=lambda path: VideoStreamInfo(
                    Path(path), 704, 1216, Fraction(24, 1)
                ),
            )

            supervisor.process_job(job)

            snapshot = store.snapshot()
            self.assertEqual(snapshot.state, PipelineState.ERROR)
            self.assertIn("expected 704x1248", snapshot.error)
            self.assertEqual(
                store.scene_states(job.job_id),
                {1: SceneState.SUCCEEDED},
            )
            self.assertTrue(clip.is_file())
            self.assertEqual(assembler.calls, [])
            self.assertEqual(mail.requests, [])


if __name__ == "__main__":
    unittest.main()
