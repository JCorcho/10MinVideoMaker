from __future__ import annotations

import copy
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tenminvideomaker.assembly import VideoStreamInfo
from tenminvideomaker.assets import AssetResolution
from tenminvideomaker.comfy_http import (
    ComfyHttpError,
    ComfyPromptDispatchAmbiguousError,
)
from tenminvideomaker.constants import PRODUCTION_HEIGHT, PRODUCTION_WIDTH
from tenminvideomaker.continuation_renderer import (
    ContinuationDeliveryError,
    ContinuationRenderError,
)
from tenminvideomaker.contracts import parse_job_payload
from tenminvideomaker.qc_contracts import QcCandidateState, QcTier
from tenminvideomaker.qc_repair import RepairGenerationError
from tenminvideomaker.delivery import DiscordDeliverySettings
from tenminvideomaker.state_store import (
    ManualFinalSceneSelection,
    PipelineState,
    PipelineStateStore,
    RemakeMode,
    SceneState,
)
from tenminvideomaker.storage import StorageLayout
from tenminvideomaker.supervisor import (
    FatalPipelineError,
    PipelineSupervisor,
    SupervisorSettings,
)
from tenminvideomaker.review import scene_review_document

from test_contracts import payload


class FakeMailClient:
    def __init__(self):
        self.requests = []
        self.unread_messages = []
        self.sent_request_ids = set()
        self.actual_send_count = 0

    def send_request(self, *, previous_job_id=None, succeeded=None, request_id=None):
        self.requests.append((previous_job_id, succeeded))
        self.actual_send_count += 1
        if request_id is not None:
            self.sent_request_ids.add(request_id)
        return f"message-id-{request_id or 'legacy'}"

    def request_was_sent(self, request_id):
        return request_id in self.sent_request_ids

    def request_message_id(self, request_id):
        return f"message-id-{request_id}"

    def unread_pipeline_messages(self):
        return list(self.unread_messages)

    def mark_seen(self, uid):
        return None


class FakeAssetManager:
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


class OneMissingAssetManager(FakeAssetManager):
    def resolve_or_download(self, lora, *, expected_base_model=None):
        if lora.name == "Missing Scene LoRA":
            return AssetResolution(lora.name, None, downloaded=False, error="download failed")
        return super().resolve_or_download(
            lora,
            expected_base_model=expected_base_model,
        )


class AllMissingAssetManager(FakeAssetManager):
    def __init__(self):
        self.resolve_calls = 0

    def resolve_or_download(self, lora, *, expected_base_model=None):
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
        workflow = self.workflows[int(prompt_id.rsplit("-", 1)[1]) - 1]
        output_node = next(
            node_id for node_id, node in workflow.items()
            if node.get("class_type") == "VHS_VideoCombine"
        ) if any(
            node.get("class_type") == "VHS_VideoCombine" for node in workflow.values()
        ) else "36"
        return {
            "outputs": {
                output_node: {
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

    def queue_counts(self):
        return 1, 2


class FakeAssembler:
    def __init__(self, final_path: Path):
        self._final_path = final_path
        self.calls = []

    def final_path(self, job_id):
        del job_id
        return self._final_path

    def stitch(self, job_id, clips, concat_directory):
        self.calls.append((job_id, list(clips), Path(concat_directory)))
        self._final_path.write_bytes(b"final")
        return self._final_path

    def stitch_qc_plan(
        self,
        job_id,
        clips,
        concat_directory,
        *,
        checkpoint_directory,
        plan_sha256,
    ):
        workspace = Path(checkpoint_directory) / job_id / plan_sha256
        receipt = workspace / "assembly.json"
        if not receipt.is_file():
            workspace.mkdir(parents=True, exist_ok=True)
            self.stitch(job_id, clips, concat_directory)
            receipt.write_text(
                json.dumps({"artifact_sha256": hashlib.sha256(b"final").hexdigest()}),
                encoding="utf-8",
            )
        elif not self._final_path.is_file():
            self._final_path.write_bytes(b"final")
        return self._final_path


class RetryOnceComfy(FakeComfy):
    def wait_for_prompt(self, prompt_id, *, timeout_seconds):
        if prompt_id == "prompt-1":
            raise ComfyHttpError("transient sampler failure")
        if prompt_id == "prompt-2":
            self.frame_path.parent.mkdir(parents=True, exist_ok=True)
            self.frame_path.write_bytes(b"png")
            return {"outputs": {}}
        return super().wait_for_prompt("prompt-2", timeout_seconds=timeout_seconds)


class ReclaimingComfy(FakeComfy):
    def __init__(self, frame_path: Path, *, stage: str):
        super().__init__(frame_path)
        self.stage = stage
        self.persisted_prompt_id = "persisted-prompt"
        self.queue_calls = 0
        self.waited_prompt_ids: list[str] = []

    def queue_prompt(self, workflow):
        self.queue_calls += 1
        return super().queue_prompt(workflow)

    def completed_prompt(self, prompt_id):
        return None

    def prompt_is_queued(self, prompt_id):
        return prompt_id == self.persisted_prompt_id

    def wait_for_prompt(self, prompt_id, *, timeout_seconds):
        del timeout_seconds
        self.waited_prompt_ids.append(prompt_id)
        if self.stage == "t2i":
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


class StageRecordingComfy(FakeComfy):
    def __init__(self, frame_paths: list[Path]):
        super().__init__(frame_paths[0])
        self._frame_paths = iter(frame_paths)
        self.stages: list[str] = []
        self._stages_by_prompt: dict[str, str] = {}

    def queue_prompt(self, workflow):
        prompt_id = super().queue_prompt(workflow)
        stage = (
            "t2i"
            if any(
                node["class_type"] == "10MinVideoMaker_SaveSceneFrame"
                for node in workflow.values()
            )
            else "i2v"
        )
        self.stages.append(stage)
        self._stages_by_prompt[prompt_id] = stage
        return prompt_id

    def wait_for_prompt(self, prompt_id, *, timeout_seconds):
        if self._stages_by_prompt[prompt_id] == "t2i":
            frame = next(self._frame_paths)
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


class SupervisorTests(unittest.TestCase):
    @staticmethod
    def _accept_qc_scenes(store, job, root):
        selection = []
        for scene in sorted(job.scenes, key=lambda item: item.scene_id):
            clip = root / f"scene-{scene.scene_id}.mp4"
            clip.write_bytes(f"scene-{scene.scene_id}".encode())
            frame = root / f"scene-{scene.scene_id}.png"
            frame.write_bytes(f"frame-{scene.scene_id}".encode())
            store.set_scene_state(
                job.job_id,
                scene.scene_id,
                SceneState.SUCCEEDED,
                video_path=str(clip),
            )
            store.ensure_original_scene_revision(
                job.job_id,
                scene.scene_id,
                parameters=scene_review_document(job, scene),
                frame_path=str(frame),
                video_path=str(clip),
            )
            candidate = store.ensure_qc_candidate(
                candidate_id=f"accepted-{scene.scene_id}",
                job_id=job.job_id,
                scene_id=scene.scene_id,
                revision=1,
                tier=QcTier.ORIGINAL,
                parent_candidate_id=None,
                source_video_path=str(clip),
                source_video_sha256=hashlib.sha256(clip.read_bytes()).hexdigest(),
                original_prompt=scene.i2v.prompt,
                current_prompt=scene.i2v.prompt,
                original_seed=scene.i2v.seed,
                current_seed=scene.i2v.seed,
                negative_prompt=scene.i2v.negative,
                negative_prompt_sha256=hashlib.sha256(
                    scene.i2v.negative.encode()
                ).hexdigest(),
                state=QcCandidateState.ACCEPTED,
                next_action=None,
            )
            selection.append(
                ManualFinalSceneSelection(
                    scene.scene_id,
                    candidate.revision,
                    candidate.source_video_path,
                )
            )
        return tuple(selection)

    def test_qc_finalization_uses_deterministic_scene_order_not_payload_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            raw = payload()
            second = copy.deepcopy(raw["scenes"][0])
            second["id"] = 2
            second["title"] = "Second scene"
            second["t2i"]["seed"] += 1
            second["i2v"]["seed"] += 1
            raw["scenes"] = [second, raw["scenes"][0]]
            job = parse_job_payload(raw)
            store = PipelineStateStore(root / "pipeline.sqlite3")
            store.claim_job(job)
            selection = self._accept_qc_scenes(store, job, root)
            clips = [Path(item.video_path) for item in selection]
            assembler = FakeAssembler(root / "final.mp4")
            mail = FakeMailClient()
            supervisor = PipelineSupervisor(
                store=store,
                mail_client=mail,
                asset_manager=FakeAssetManager(),
                comfy=FakeComfy(root / "unused-frame.png"),
                assembler=assembler,
                settings=SupervisorSettings(),
                video_probe=lambda path: VideoStreamInfo(
                    Path(path),
                    PRODUCTION_WIDTH,
                    PRODUCTION_HEIGHT,
                    Fraction(24, 1),
                ),
            )
            supervisor._finalize_qc_job(job, selection)

            self.assertEqual(assembler.calls[0][1], [str(item) for item in clips])
            self.assertEqual(mail.requests, [(job.job_id, True)])
            delivery_step = next(
                item
                for item in store.qc_finalization_steps(job.job_id)
                if item.kind == "SCENE_DELIVERY"
            )
            self.assertEqual(delivery_step.receipt["status"], "NOT_CONFIGURED")

    def test_configured_qc_delivery_failure_blocks_assembly_and_mail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            job = parse_job_payload(payload())
            storage = StorageLayout(root / "storage")
            storage.ensure()
            store = PipelineStateStore(storage.database_path)
            store.claim_job(job)
            selection = self._accept_qc_scenes(store, job, root)
            assembler = FakeAssembler(root / "final.mp4")
            mail = FakeMailClient()
            supervisor = PipelineSupervisor(
                store=store,
                mail_client=mail,
                asset_manager=FakeAssetManager(),
                comfy=FakeComfy(root / "unused.png"),
                assembler=assembler,
                settings=SupervisorSettings(),
                storage=storage,
                delivery=DiscordDeliverySettings(
                    "https://discord.com/api/webhooks/123456789/test-token"
                ),
                video_probe=lambda path: VideoStreamInfo(
                    Path(path), PRODUCTION_WIDTH, PRODUCTION_HEIGHT, Fraction(24, 1)
                ),
            )
            renderer = Mock()
            renderer.deliver_existing_scene.side_effect = ContinuationDeliveryError(
                "definitively rejected", state="FAILED"
            )
            supervisor.continuation_renderer = renderer

            supervisor._finalize_qc_job(job, selection)

            self.assertEqual(renderer.deliver_existing_scene.call_count, 2)
            self.assertEqual(assembler.calls, [])
            self.assertEqual(mail.requests, [])
            self.assertEqual(store.snapshot().state, PipelineState.QC_BLOCKED)
            delivery_step = next(
                item
                for item in store.qc_finalization_steps(job.job_id)
                if item.kind == "SCENE_DELIVERY"
            )
            self.assertEqual(delivery_step.state, "FAILED")

    def test_ambiguous_qc_delivery_is_never_redispatched(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            job = parse_job_payload(payload())
            storage = StorageLayout(root / "storage")
            storage.ensure()
            store = PipelineStateStore(storage.database_path)
            store.claim_job(job)
            selection = self._accept_qc_scenes(store, job, root)
            assembler = FakeAssembler(root / "final.mp4")
            supervisor = PipelineSupervisor(
                store=store,
                mail_client=FakeMailClient(),
                asset_manager=FakeAssetManager(),
                comfy=FakeComfy(root / "unused.png"),
                assembler=assembler,
                settings=SupervisorSettings(),
                storage=storage,
                delivery=DiscordDeliverySettings(
                    "https://discord.com/api/webhooks/123456789/test-token"
                ),
                video_probe=lambda path: VideoStreamInfo(
                    Path(path), PRODUCTION_WIDTH, PRODUCTION_HEIGHT, Fraction(24, 1)
                ),
            )
            renderer = Mock()
            renderer.deliver_existing_scene.side_effect = ContinuationDeliveryError(
                "acceptance unknown", state="AMBIGUOUS"
            )
            supervisor.continuation_renderer = renderer

            supervisor._finalize_qc_job(job, selection)
            supervisor.tick()

            self.assertEqual(renderer.deliver_existing_scene.call_count, 1)
            self.assertEqual(assembler.calls, [])
            self.assertEqual(store.snapshot().state, PipelineState.QC_BLOCKED)

    def test_qc_disabled_result_commits_legacy_baseline_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            job = parse_job_payload(payload())
            store = PipelineStateStore(root / "pipeline.sqlite3")
            store.claim_job(job)
            selection = self._accept_qc_scenes(store, job, root)
            assembler = FakeAssembler(root / "final.mp4")
            mail = FakeMailClient()
            supervisor = PipelineSupervisor(
                store=store,
                mail_client=mail,
                asset_manager=FakeAssetManager(),
                comfy=FakeComfy(root / "unused.png"),
                assembler=assembler,
                settings=SupervisorSettings(),
                qc_controller=SimpleNamespace(
                    settings=SimpleNamespace(quality_control_enabled=False)
                ),
                video_probe=lambda path: VideoStreamInfo(
                    Path(path), PRODUCTION_WIDTH, PRODUCTION_HEIGHT, Fraction(24, 1)
                ),
            )

            supervisor._finalize_qc_result(job, selection)

            plan = store.qc_finalization_plan(job.job_id)
            self.assertEqual(plan.selection_mode, "LEGACY_BASELINE")
            self.assertEqual(plan.state, "COMPLETED")
            self.assertEqual(len(assembler.calls), 1)
            self.assertEqual(mail.requests, [(job.job_id, True)])
            self.assertEqual(store.snapshot().state, PipelineState.WAITING_FOR_GROK)

    def test_never_qc_job_keeps_legacy_finalization_without_qc_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            job = parse_job_payload(payload())
            store = PipelineStateStore(root / "pipeline.sqlite3")
            store.claim_job(job)
            clip = root / "legacy.mp4"
            clip.write_bytes(b"legacy")
            store.set_scene_state(
                job.job_id, 1, SceneState.SUCCEEDED, video_path=str(clip)
            )
            store.ensure_original_scene_revision(
                job.job_id,
                1,
                parameters=scene_review_document(job, job.scenes[0]),
                video_path=str(clip),
            )
            supervisor = PipelineSupervisor(
                store=store,
                mail_client=FakeMailClient(),
                asset_manager=FakeAssetManager(),
                comfy=FakeComfy(root / "unused.png"),
                assembler=FakeAssembler(root / "final.mp4"),
                settings=SupervisorSettings(),
                video_probe=lambda path: VideoStreamInfo(
                    Path(path), PRODUCTION_WIDTH, PRODUCTION_HEIGHT, Fraction(24, 1)
                ),
            )

            supervisor._finalize_qc_disabled_selection(
                job, store.legacy_final_selection(job.job_id, [1])
            )

            self.assertIsNone(store.qc_finalization_plan(job.job_id))
            self.assertEqual(store.snapshot().state, PipelineState.WAITING_FOR_GROK)

    def test_legacy_baseline_plan_never_substitutes_accepted_repair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            job = parse_job_payload(payload())
            store = PipelineStateStore(root / "pipeline.sqlite3")
            store.claim_job(job)
            baseline = self._accept_qc_scenes(store, job, root)[0]
            repair_path = root / "accepted-a1.mp4"
            repair_path.write_bytes(b"accepted-a1")
            document = scene_review_document(job, job.scenes[0])
            revision = store.create_scene_revision(
                job.job_id,
                1,
                remake_mode=RemakeMode.VIDEO_ONLY,
                parameters=document,
                state=SceneState.SUCCEEDED,
                video_path=str(repair_path),
            )
            repair = store.ensure_qc_candidate(
                candidate_id="accepted-a1",
                job_id=job.job_id,
                scene_id=1,
                revision=revision,
                tier=QcTier.A1,
                parent_candidate_id="accepted-1",
                source_video_path=str(repair_path),
                source_video_sha256=hashlib.sha256(repair_path.read_bytes()).hexdigest(),
                original_prompt=job.scenes[0].i2v.prompt,
                current_prompt=job.scenes[0].i2v.prompt,
                original_seed=job.scenes[0].i2v.seed,
                current_seed=job.scenes[0].i2v.seed + 1,
                negative_prompt=job.scenes[0].i2v.negative,
                negative_prompt_sha256=hashlib.sha256(
                    job.scenes[0].i2v.negative.encode()
                ).hexdigest(),
                state=QcCandidateState.ACCEPTED,
                next_action=None,
            )
            store.set_qc_candidate_state(
                "accepted-1", QcCandidateState.SUPERSEDED, next_action=None
            )
            supervisor = PipelineSupervisor(
                store=store,
                mail_client=FakeMailClient(),
                asset_manager=FakeAssetManager(),
                comfy=FakeComfy(root / "unused.png"),
                assembler=FakeAssembler(root / "final.mp4"),
                settings=SupervisorSettings(),
                video_probe=lambda path: VideoStreamInfo(
                    Path(path), PRODUCTION_WIDTH, PRODUCTION_HEIGHT, Fraction(24, 1)
                ),
            )

            supervisor._finalize_qc_disabled_selection(job, (baseline,))

            plan = store.qc_finalization_plan(job.job_id)
            self.assertEqual(plan.selection_mode, "LEGACY_BASELINE")
            self.assertEqual(plan.selection[0]["revision"], baseline.revision)
            self.assertEqual(plan.selection[0]["artifact_path"], baseline.video_path)
            self.assertNotEqual(plan.selection[0]["candidate_id"], repair.candidate_id)

    def test_qc_disabled_restart_abandons_side_effect_free_plan_for_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            job = parse_job_payload(payload())
            store = PipelineStateStore(root / "pipeline.sqlite3")
            store.claim_job(job)
            self._accept_qc_scenes(store, job, root)
            assembler = FakeAssembler(root / "final.mp4")
            mail = FakeMailClient()
            store.ensure_qc_finalization_plan(
                job.job_id,
                [scene.scene_id for scene in job.scenes],
                final_path=str(assembler.final_path(job.job_id)),
            )
            store.transition(PipelineState.STITCHING, job_id=job.job_id)
            supervisor = PipelineSupervisor(
                store=store,
                mail_client=mail,
                asset_manager=FakeAssetManager(),
                comfy=FakeComfy(root / "unused.png"),
                assembler=assembler,
                settings=SupervisorSettings(),
                qc_controller=SimpleNamespace(
                    settings=SimpleNamespace(quality_control_enabled=False)
                ),
                video_probe=lambda path: VideoStreamInfo(
                    Path(path), PRODUCTION_WIDTH, PRODUCTION_HEIGHT, Fraction(24, 1)
                ),
            )

            supervisor.tick()

            self.assertEqual(len(assembler.calls), 1)
            self.assertEqual(mail.requests, [(job.job_id, True)])
            plan = store.qc_finalization_plan(job.job_id)
            self.assertEqual(plan.selection_mode, "LEGACY_BASELINE")
            self.assertEqual(plan.state, "COMPLETED")
            self.assertEqual(store.snapshot().state, PipelineState.WAITING_FOR_GROK)

    def test_qc_disabled_restart_holds_ambiguous_finalization_side_effect(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            job = parse_job_payload(payload())
            store = PipelineStateStore(root / "pipeline.sqlite3")
            store.claim_job(job)
            selection = self._accept_qc_scenes(store, job, root)
            assembler = FakeAssembler(root / "final.mp4")
            mail = FakeMailClient()
            store.ensure_qc_finalization_plan(
                job.job_id,
                [scene.scene_id for scene in job.scenes],
                final_path=str(assembler.final_path(job.job_id)),
            )
            store.begin_qc_finalization_step(
                job.job_id,
                "deliver-scene-1",
                kind="SCENE_DELIVERY",
                evidence={
                    "scene_id": selection[0].scene_id,
                    "artifact_path": selection[0].video_path,
                },
            )
            store.transition(PipelineState.STITCHING, job_id=job.job_id)
            supervisor = PipelineSupervisor(
                store=store,
                mail_client=mail,
                asset_manager=FakeAssetManager(),
                comfy=FakeComfy(root / "unused.png"),
                assembler=assembler,
                settings=SupervisorSettings(),
                qc_controller=SimpleNamespace(
                    settings=SimpleNamespace(quality_control_enabled=False)
                ),
                video_probe=lambda path: VideoStreamInfo(
                    Path(path), PRODUCTION_WIDTH, PRODUCTION_HEIGHT, Fraction(24, 1)
                ),
            )

            supervisor.tick()

            self.assertEqual(assembler.calls, [])
            self.assertEqual(mail.requests, [])
            self.assertEqual(store.snapshot().state, PipelineState.QC_BLOCKED)
            self.assertIn("manual reconciliation", store.snapshot().error)

    def test_qc_finalization_resumes_each_crash_boundary_without_duplicate_side_effects(self) -> None:
        crash_points = (
            "plan_committed",
            "after_scene_delivery:1",
            "deliveries_completed",
            "after_stitch_artifact",
            "after_job_success",
            "after_next_request",
        )
        for crash_point in crash_points:
            with self.subTest(crash_point=crash_point), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                raw = payload()
                if crash_point == "after_scene_delivery:1":
                    second = copy.deepcopy(raw["scenes"][0])
                    second["id"] = 2
                    second["title"] = "Second accepted scene"
                    second["t2i"]["seed"] += 1
                    second["i2v"]["seed"] += 1
                    raw["scenes"].append(second)
                job = parse_job_payload(raw)
                storage = StorageLayout(root / "storage")
                storage.ensure()
                store = PipelineStateStore(storage.database_path)
                store.claim_job(job)
                selection = self._accept_qc_scenes(store, job, root)
                assembler = FakeAssembler(root / "final.mp4")
                mail = FakeMailClient()

                class IdempotentRenderer:
                    def __init__(self):
                        self.actual_sends = 0
                        self.sent = set()

                    def deliver_existing_scene(self, _job, scene, _path, **_kwargs):
                        if scene.scene_id not in self.sent:
                            self.sent.add(scene.scene_id)
                            self.actual_sends += 1
                        return SimpleNamespace(
                            status="sent", prompt_id=f"prompt-{scene.scene_id}"
                        )

                renderer = IdempotentRenderer()
                fired = []

                def crash_hook(point):
                    if point == crash_point and not fired:
                        fired.append(point)
                        raise RuntimeError(f"crash at {point}")

                def build(hook=None):
                    supervisor = PipelineSupervisor(
                        store=store,
                        mail_client=mail,
                        asset_manager=FakeAssetManager(),
                        comfy=FakeComfy(root / "unused.png"),
                        assembler=assembler,
                        settings=SupervisorSettings(),
                        storage=storage,
                        delivery=DiscordDeliverySettings(
                            "https://discord.com/api/webhooks/123456789/test-token"
                        ),
                        video_probe=lambda path: VideoStreamInfo(
                            Path(path), PRODUCTION_WIDTH, PRODUCTION_HEIGHT, Fraction(24, 1)
                        ),
                        qc_finalization_checkpoint_hook=hook,
                    )
                    supervisor.continuation_renderer = renderer
                    return supervisor

                with self.assertRaisesRegex(RuntimeError, "crash at"):
                    build(crash_hook)._finalize_qc_job(job, selection)

                build().tick()

                self.assertEqual(renderer.actual_sends, len(job.scenes))
                self.assertEqual(len(assembler.calls), 1)
                self.assertEqual(mail.actual_send_count, 1)
                self.assertEqual(store.snapshot().state, PipelineState.WAITING_FOR_GROK)
                self.assertEqual(store.list_jobs()[0].status.value, "succeeded")
                plan = store.qc_finalization_plan(job.job_id)
                self.assertEqual(plan.state, "COMPLETED")
                self.assertEqual(
                    plan.final_sha256,
                    hashlib.sha256(b"final").hexdigest(),
                )
                delivery_steps = [
                    item
                    for item in store.qc_finalization_steps(job.job_id)
                    if item.kind == "SCENE_DELIVERY"
                ]
                self.assertEqual(len(delivery_steps), len(job.scenes))
                self.assertTrue(
                    all(item.receipt["prompt_id"] for item in delivery_steps)
                )

    def test_legacy_baseline_plan_resumes_all_crash_boundaries(self) -> None:
        crash_points = (
            "plan_committed",
            "after_scene_delivery:1",
            "deliveries_completed",
            "after_stitch_artifact",
            "after_job_success",
            "after_next_request",
        )
        for crash_point in crash_points:
            with self.subTest(crash_point=crash_point), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                job = parse_job_payload(payload())
                storage = StorageLayout(root / "storage")
                storage.ensure()
                store = PipelineStateStore(storage.database_path)
                store.claim_job(job)
                selection = self._accept_qc_scenes(store, job, root)
                assembler = FakeAssembler(root / "final.mp4")
                mail = FakeMailClient()

                class IdempotentRenderer:
                    def __init__(self):
                        self.actual_sends = 0
                        self.sent = set()

                    def deliver_existing_scene(self, _job, scene, _path, **_kwargs):
                        if scene.scene_id not in self.sent:
                            self.sent.add(scene.scene_id)
                            self.actual_sends += 1
                        return SimpleNamespace(
                            status="sent", prompt_id=f"prompt-{scene.scene_id}"
                        )

                renderer = IdempotentRenderer()
                fired = []

                def crash_hook(point):
                    if point == crash_point and not fired:
                        fired.append(point)
                        raise RuntimeError(f"crash at {point}")

                def build(hook=None):
                    supervisor = PipelineSupervisor(
                        store=store,
                        mail_client=mail,
                        asset_manager=FakeAssetManager(),
                        comfy=FakeComfy(root / "unused.png"),
                        assembler=assembler,
                        settings=SupervisorSettings(),
                        storage=storage,
                        delivery=DiscordDeliverySettings(
                            "https://discord.com/api/webhooks/123456789/test-token"
                        ),
                        qc_controller=SimpleNamespace(
                            settings=SimpleNamespace(quality_control_enabled=False)
                        ),
                        video_probe=lambda path: VideoStreamInfo(
                            Path(path), PRODUCTION_WIDTH, PRODUCTION_HEIGHT, Fraction(24, 1)
                        ),
                        qc_finalization_checkpoint_hook=hook,
                    )
                    supervisor.continuation_renderer = renderer
                    return supervisor

                with self.assertRaisesRegex(RuntimeError, "crash at"):
                    build(crash_hook)._finalize_qc_job(
                        job,
                        selection,
                        selection_mode="LEGACY_BASELINE",
                    )

                build().tick()

                plan = store.qc_finalization_plan(job.job_id)
                self.assertEqual(plan.selection_mode, "LEGACY_BASELINE")
                self.assertEqual(plan.state, "COMPLETED")
                self.assertEqual(renderer.actual_sends, 1)
                self.assertEqual(len(assembler.calls), 1)
                self.assertEqual(mail.actual_send_count, 1)
                self.assertEqual(store.snapshot().state, PipelineState.WAITING_FOR_GROK)

    def test_legacy_baseline_ambiguous_mail_send_holds_without_resend(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            job = parse_job_payload(payload())
            storage = StorageLayout(root / "storage")
            storage.ensure()
            store = PipelineStateStore(storage.database_path)
            store.claim_job(job)
            selection = self._accept_qc_scenes(store, job, root)

            class AmbiguousMail(FakeMailClient):
                def send_request(self, **kwargs):
                    self.actual_send_count += 1
                    raise TimeoutError("connection lost after SMTP submission")

            mail = AmbiguousMail()

            def build():
                return PipelineSupervisor(
                    store=store,
                    mail_client=mail,
                    asset_manager=FakeAssetManager(),
                    comfy=FakeComfy(root / "unused.png"),
                    assembler=FakeAssembler(root / "final.mp4"),
                    settings=SupervisorSettings(),
                    storage=storage,
                    qc_controller=SimpleNamespace(
                        settings=SimpleNamespace(quality_control_enabled=False)
                    ),
                    video_probe=lambda path: VideoStreamInfo(
                        Path(path), PRODUCTION_WIDTH, PRODUCTION_HEIGHT, Fraction(24, 1)
                    ),
                )

            build()._finalize_qc_job(
                job,
                selection,
                selection_mode="LEGACY_BASELINE",
            )

            self.assertEqual(mail.actual_send_count, 1)
            self.assertEqual(store.snapshot().state, PipelineState.QC_BLOCKED)
            step = next(
                item
                for item in store.qc_finalization_steps(job.job_id)
                if item.step_key == "request-next-job"
            )
            self.assertEqual(step.state, "AMBIGUOUS")
            self.assertIn("SMTP submission", step.receipt["error"])

            build().tick()

            self.assertEqual(mail.actual_send_count, 1)
            self.assertEqual(store.snapshot().state, PipelineState.QC_BLOCKED)

    def test_qc_enabled_inserts_epoch_after_complete_i2v_batch_and_blocks_stitch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            job = parse_job_payload(payload())
            store = PipelineStateStore(root / "pipeline.sqlite3")
            store.claim_job(job)
            frame = root / "frame.png"
            clip = root / "clip.mp4"
            comfy = FakeComfy(frame)
            assembler = FakeAssembler(root / "final.mp4")
            calls: list[str] = []

            class FakeQcController:
                settings = SimpleNamespace(quality_control_enabled=True)

                def register_original_candidates(self, _job):
                    calls.append("register")

                def run_epoch(self, _job, _supervisor):
                    calls.append("epoch")
                    return SimpleNamespace(ready_for_finalization=False, selection=())

            supervisor = PipelineSupervisor(
                store=store,
                mail_client=FakeMailClient(),
                asset_manager=FakeAssetManager(),
                comfy=comfy,
                assembler=assembler,
                settings=SupervisorSettings(1, 10, 10, 2),
                frame_path_factory=lambda _job, _scene: frame,
                clip_path_factory=lambda _job, _scene: clip,
                qc_controller=FakeQcController(),
            )

            supervisor.process_job(job)

            self.assertEqual(calls, ["register", "epoch"])
            self.assertEqual(store.snapshot().state, PipelineState.RUNNING_QC)
            self.assertEqual(assembler.calls, [])
            self.assertEqual(len(comfy.workflows), 2)

    def test_qc_restart_state_delegates_without_replaying_generation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            job = parse_job_payload(payload())
            store = PipelineStateStore(root / "pipeline.sqlite3")
            store.claim_job(job)
            store.transition(PipelineState.AWAITING_QC_REVIEW, job_id=job.job_id)
            calls = []

            class FakeQcController:
                settings = SimpleNamespace(quality_control_enabled=False)

                def run_epoch(self, _job, _supervisor):
                    calls.append("epoch")
                    return SimpleNamespace(ready_for_finalization=False, selection=())

            comfy = FakeComfy(root / "unused.png")
            supervisor = PipelineSupervisor(
                store=store,
                mail_client=FakeMailClient(),
                asset_manager=FakeAssetManager(),
                comfy=comfy,
                settings=SupervisorSettings(1, 10, 10, 2),
                qc_controller=FakeQcController(),
            )

            supervisor.tick()

            self.assertEqual(calls, ["epoch"])
            self.assertEqual(comfy.workflows, [])

    def test_qc_blocked_ticks_do_not_repeat_registration_or_controller_work(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            job = parse_job_payload(payload())
            store = PipelineStateStore(root / "pipeline.sqlite3")
            store.claim_job(job)
            store.transition(
                PipelineState.QC_BLOCKED,
                job_id=job.job_id,
                error="QC blocked: pre-QC selection is missing scene(s) 02",
            )
            calls: list[str] = []

            class FakeQcController:
                settings = SimpleNamespace(quality_control_enabled=True)

                def run_epoch(self, *_args):
                    calls.append("epoch")
                    raise AssertionError("blocked QC must not run")

                def register_original_candidates(self, *_args):
                    calls.append("register")
                    raise AssertionError("blocked QC must not register")

            supervisor = PipelineSupervisor(
                store=store,
                mail_client=FakeMailClient(),
                asset_manager=FakeAssetManager(),
                comfy=FakeComfy(root / "unused.png"),
                settings=SupervisorSettings(1, 10, 10, 2),
                qc_controller=FakeQcController(),
            )

            supervisor.tick()
            supervisor.tick()

            self.assertEqual(calls, [])
            self.assertEqual(store.snapshot().state, PipelineState.QC_BLOCKED)

    def test_fatal_restart_resumes_only_active_automatic_job(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            job = parse_job_payload(payload())
            store = PipelineStateStore(root / "pipeline.sqlite3")
            store.claim_job(job)
            supervisor = PipelineSupervisor(
                store=store,
                mail_client=FakeMailClient(),
                asset_manager=FakeAssetManager(),
                comfy=FakeComfy(root / "unused.png"),
                settings=SupervisorSettings(1, 10, 10, 2),
                restart_comfy=lambda: True,
            )

            with patch.object(
                store,
                "requeue_unfinished_scenes",
                wraps=store.requeue_unfinished_scenes,
            ) as requeue:
                supervisor.handle_fatal(FatalPipelineError("server stopped"))

            self.assertEqual(store.snapshot().state, PipelineState.DOWNLOADING_ASSETS)
            requeue.assert_called_once_with(job.job_id)

    def test_qc_repair_generation_fails_before_queue_when_comfy_is_dead(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            job = parse_job_payload(payload())
            store = PipelineStateStore(root / "pipeline.sqlite3")
            store.claim_job(job)
            comfy = FakeComfy(root / "unused.png")
            comfy.alive = lambda: False
            supervisor = PipelineSupervisor(
                store=store,
                mail_client=FakeMailClient(),
                asset_manager=FakeAssetManager(),
                comfy=comfy,
                settings=SupervisorSettings(1, 10, 10, 2),
            )

            with self.assertRaisesRegex(RepairGenerationError, "unavailable") as raised:
                supervisor.render_qc_candidates(
                    job, ((SimpleNamespace(candidate_id="candidate-a1-test"), {}),)
                )

            self.assertEqual(comfy.workflows, [])
            self.assertTrue(raised.exception.retryable)

    def test_qc_repair_ambiguous_prompt_acceptance_is_not_retried(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            job = parse_job_payload(payload())
            frame = root / "frame.png"
            frame.write_bytes(b"frame")
            store = Mock()
            store.scene_revisions.return_value = (
                SimpleNamespace(revision=2, frame_path=str(frame)),
            )
            supervisor = PipelineSupervisor(
                store=store,
                mail_client=FakeMailClient(),
                asset_manager=FakeAssetManager(),
                comfy=FakeComfy(root / "unused.png"),
                settings=SupervisorSettings(1, 10, 10, 2),
            )
            supervisor._resolve_assets = Mock(
                return_value=SimpleNamespace(failures={}, resolved_filenames={})
            )
            supervisor._uses_continuation = Mock(return_value=False)
            supervisor.render_i2v_scene = Mock(
                side_effect=ComfyPromptDispatchAmbiguousError(
                    "acceptance cannot be disproved"
                )
            )
            candidate = SimpleNamespace(
                candidate_id="candidate-a1-ambiguous",
                scene_id=1,
                revision=2,
                state=QcCandidateState.PENDING_GENERATION,
                generation_prompt_id=None,
                source_video_path=str(root / "a1.mp4"),
            )

            with self.assertRaises(RepairGenerationError) as raised:
                supervisor.render_qc_candidates(
                    job,
                    ((candidate, scene_review_document(job, job.scenes[0])),),
                )

            self.assertFalse(raised.exception.retryable)

    def test_fatal_restart_restores_qc_epoch_without_replaying_originals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            job = parse_job_payload(payload())
            store = PipelineStateStore(root / "pipeline.sqlite3")
            store.claim_job(job)
            store.transition(PipelineState.RUNNING_QC, job_id=job.job_id)
            supervisor = PipelineSupervisor(
                store=store,
                mail_client=FakeMailClient(),
                asset_manager=FakeAssetManager(),
                comfy=FakeComfy(root / "unused.png"),
                settings=SupervisorSettings(1, 10, 10, 2),
                restart_comfy=lambda: True,
            )

            with patch.object(store, "requeue_unfinished_scenes") as requeue:
                supervisor.handle_fatal(FatalPipelineError("server stopped"))

            self.assertEqual(store.snapshot().state, PipelineState.RUNNING_QC)
            requeue.assert_not_called()

    def test_fatal_restart_restores_waiting_state_without_requeueing_old_job(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            job = parse_job_payload(payload())
            store = PipelineStateStore(root / "pipeline.sqlite3")
            store.claim_job(job)
            store.transition(PipelineState.WAITING_FOR_GROK, job_id=job.job_id)
            supervisor = PipelineSupervisor(
                store=store,
                mail_client=FakeMailClient(),
                asset_manager=FakeAssetManager(),
                comfy=FakeComfy(root / "unused.png"),
                settings=SupervisorSettings(1, 10, 10, 2),
                restart_comfy=lambda: True,
            )

            with patch.object(
                store,
                "requeue_unfinished_scenes",
                wraps=store.requeue_unfinished_scenes,
            ) as requeue:
                supervisor.handle_fatal(FatalPipelineError("server stopped"))

            snapshot = store.snapshot()
            self.assertEqual(snapshot.state, PipelineState.WAITING_FOR_GROK)
            self.assertEqual(snapshot.job_id, job.job_id)
            requeue.assert_not_called()

    def test_fatal_restart_restores_idle_state_without_claiming_work(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            store = PipelineStateStore(root / "pipeline.sqlite3")
            supervisor = PipelineSupervisor(
                store=store,
                mail_client=FakeMailClient(),
                asset_manager=FakeAssetManager(),
                comfy=FakeComfy(root / "unused.png"),
                settings=SupervisorSettings(1, 10, 10, 2),
                restart_comfy=lambda: True,
            )

            supervisor.handle_fatal(FatalPipelineError("server stopped"))

            snapshot = store.snapshot()
            self.assertEqual(snapshot.state, PipelineState.IDLE)
            self.assertIsNone(snapshot.job_id)

    def test_automatic_t2i_reclaims_persisted_prompt_without_duplicate_queue(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            job = parse_job_payload(payload())
            scene = job.scenes[0]
            storage = StorageLayout(root / "storage")
            storage.ensure()
            store = PipelineStateStore(storage.database_path)
            store.claim_job(job)
            store.ensure_original_scene_revision(
                job.job_id,
                scene.scene_id,
                parameters={"job_id": job.job_id, "scene_id": scene.scene_id},
            )
            frame = storage.scene_frame_path(job.job_id, scene.scene_id, 1)
            comfy = ReclaimingComfy(frame, stage="t2i")
            supervisor = PipelineSupervisor(
                store=store,
                mail_client=FakeMailClient(),
                asset_manager=FakeAssetManager(),
                comfy=comfy,
                settings=SupervisorSettings(1, 10, 10, 2),
                storage=storage,
                frame_path_factory=lambda job_id, scene_id: storage.scene_frame_path(
                    job_id,
                    scene_id,
                    1,
                ),
                clip_path_factory=lambda job_id, scene_id: storage.scene_clip_path(
                    job_id,
                    scene_id,
                    1,
                ),
            )
            resolved = supervisor._resolve_assets(job).resolved_filenames
            store.begin_scene_stage(
                job.job_id,
                scene.scene_id,
                PipelineState.RUNNING_T2I,
            )
            store.set_scene_prompt_id(
                job.job_id,
                scene.scene_id,
                comfy.persisted_prompt_id,
                stage="t2i",
            )

            supervisor._process_t2i_stage(job, scene, resolved)

            record = store.scene_records(job.job_id)[0]
            self.assertEqual(record.t2i_attempts, 1)
            self.assertEqual(record.state, SceneState.PENDING)
            self.assertTrue(frame.is_file())
            self.assertEqual(comfy.queue_calls, 0)
            self.assertEqual(
                comfy.waited_prompt_ids,
                [comfy.persisted_prompt_id],
            )

    def test_legacy_i2v_reclaims_persisted_prompt_without_duplicate_queue(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            job = parse_job_payload(payload())
            scene = job.scenes[0]
            storage = StorageLayout(root / "storage")
            storage.ensure()
            store = PipelineStateStore(storage.database_path)
            store.claim_job(job)
            store.ensure_original_scene_revision(
                job.job_id,
                scene.scene_id,
                parameters={"job_id": job.job_id, "scene_id": scene.scene_id},
            )
            frame = storage.scene_frame_path(job.job_id, scene.scene_id, 1)
            frame.parent.mkdir(parents=True, exist_ok=True)
            frame.write_bytes(b"png")
            store.set_scene_state(
                job.job_id,
                scene.scene_id,
                SceneState.PENDING,
                frame_path=str(frame),
            )
            comfy = ReclaimingComfy(frame, stage="i2v")
            supervisor = PipelineSupervisor(
                store=store,
                mail_client=FakeMailClient(),
                asset_manager=FakeAssetManager(),
                comfy=comfy,
                settings=SupervisorSettings(1, 10, 10, 2),
                storage=storage,
                frame_path_factory=lambda job_id, scene_id: storage.scene_frame_path(
                    job_id,
                    scene_id,
                    1,
                ),
                clip_path_factory=lambda job_id, scene_id: storage.scene_clip_path(
                    job_id,
                    scene_id,
                    1,
                ),
            )
            resolved = supervisor._resolve_assets(job).resolved_filenames
            store.begin_scene_stage(
                job.job_id,
                scene.scene_id,
                PipelineState.RUNNING_I2V,
            )
            store.set_scene_prompt_id(
                job.job_id,
                scene.scene_id,
                comfy.persisted_prompt_id,
                stage="i2v_legacy",
            )

            supervisor._process_i2v_stage(job, scene, resolved)

            record = store.scene_records(job.job_id)[0]
            self.assertEqual(record.i2v_attempts, 1)
            self.assertEqual(record.state, SceneState.SUCCEEDED)
            self.assertTrue(
                storage.scene_clip_path(job.job_id, scene.scene_id, 1).is_file()
            )
            self.assertEqual(comfy.queue_calls, 0)
            self.assertEqual(
                comfy.waited_prompt_ids,
                [comfy.persisted_prompt_id],
            )

    def test_continuation_terminal_error_is_not_retried_by_outer_scene_loop(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            document = payload()
            document["scenes"][0]["estimated_sec"] = 10
            document["scenes"][0]["i2v"]["loras"] = []
            document["scenes"][0]["i2v"]["continuation"] = {"enabled": True}
            job = parse_job_payload(document)
            scene = job.scenes[0]
            storage = StorageLayout(root / "storage")
            storage.ensure()
            store = PipelineStateStore(storage.database_path)
            store.claim_job(job)
            store.ensure_original_scene_revision(
                job.job_id,
                scene.scene_id,
                parameters={"job_id": job.job_id, "scene_id": scene.scene_id},
            )
            frame = storage.scene_frame_path(job.job_id, scene.scene_id, 1)
            frame.parent.mkdir(parents=True, exist_ok=True)
            frame.write_bytes(b"frame")
            store.set_scene_state(
                job.job_id,
                scene.scene_id,
                SceneState.PENDING,
                frame_path=str(frame),
            )
            supervisor = PipelineSupervisor(
                store=store,
                mail_client=FakeMailClient(),
                asset_manager=FakeAssetManager(),
                comfy=FakeComfy(frame),
                settings=SupervisorSettings(
                    poll_interval_seconds=1,
                    t2i_timeout_seconds=10,
                    i2v_timeout_seconds=10,
                    max_stage_attempts=2,
                    continuation_mode="explicit",
                ),
                storage=storage,
            )
            continuation = Mock()
            continuation.render_scene.side_effect = [
                ContinuationRenderError("chunk retry budget exhausted"),
                AssertionError("outer scene loop retried continuation"),
            ]
            supervisor.continuation_renderer = continuation

            supervisor._process_scene_stage_with_retries(
                job,
                scene,
                PipelineState.RUNNING_I2V,
                {},
            )

            record = store.scene_records(job.job_id)[0]
            self.assertEqual(record.state, SceneState.FAILED)
            self.assertEqual(record.i2v_attempts, 1)
            self.assertIn("retry budget exhausted", record.error)
            self.assertEqual(continuation.render_scene.call_count, 1)

    def test_continuation_route_uses_effective_requested_duration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            document = payload()
            document["scenes"][0]["estimated_sec"] = 5
            document["scenes"][0]["i2v"]["loras"] = []
            document["scenes"][0]["i2v"]["continuation"] = {
                "enabled": True,
                "requested_duration_seconds": 10,
            }
            job = parse_job_payload(document)
            scene = job.scenes[0]
            storage = StorageLayout(root / "storage")
            storage.ensure()
            store = PipelineStateStore(storage.database_path)
            frame = storage.scene_frame_path(job.job_id, scene.scene_id, 1)
            frame.parent.mkdir(parents=True, exist_ok=True)
            frame.write_bytes(b"png")
            comfy = StageRecordingComfy([frame])
            supervisor = PipelineSupervisor(
                store=store,
                mail_client=FakeMailClient(),
                asset_manager=FakeAssetManager(),
                comfy=comfy,
                settings=SupervisorSettings(
                    continuation_mode="explicit",
                    i2v_timeout_seconds=10,
                ),
                storage=storage,
                frame_path_factory=lambda job_id, scene_id: storage.scene_frame_path(
                    job_id,
                    scene_id,
                    1,
                ),
                clip_path_factory=lambda job_id, scene_id: storage.scene_clip_path(
                    job_id,
                    scene_id,
                    1,
                ),
            )
            continuation = Mock()
            destination = storage.scene_clip_path(job.job_id, scene.scene_id, 1)
            continuation.render_scene.return_value = SimpleNamespace(
                scene_path=destination
            )
            supervisor.continuation_renderer = continuation

            self.assertEqual(
                supervisor.render_i2v_scene(
                    job=job,
                    scene=scene,
                    frame_path=frame,
                    destination=destination,
                    resolved_lora_filenames={},
                ),
                destination,
            )
            continuation.render_scene.assert_called_once()
            self.assertEqual(comfy.workflows, [])

            shorter_override = copy.deepcopy(document)
            shorter_override["scenes"][0]["estimated_sec"] = 10
            shorter_override["scenes"][0]["i2v"]["continuation"][
                "requested_duration_seconds"
            ] = 5
            legacy_job = parse_job_payload(shorter_override)
            legacy_scene = legacy_job.scenes[0]
            legacy_destination = storage.scene_clip_path(
                legacy_job.job_id,
                legacy_scene.scene_id,
                2,
            )
            supervisor.render_i2v_scene(
                job=legacy_job,
                scene=legacy_scene,
                frame_path=frame,
                destination=legacy_destination,
                resolved_lora_filenames={},
                revision=2,
                deliver_to_discord=False,
            )

            continuation.render_scene.assert_called_once()
            self.assertEqual(comfy.stages, ["i2v"])
            self.assertTrue(legacy_destination.is_file())

    def test_existing_continuation_destination_is_revalidated_by_renderer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            document = payload()
            document["scenes"][0]["estimated_sec"] = 10
            document["scenes"][0]["i2v"]["loras"] = []
            document["scenes"][0]["i2v"]["continuation"] = {"enabled": True}
            job = parse_job_payload(document)
            scene = job.scenes[0]
            storage = StorageLayout(root / "storage")
            storage.ensure()
            store = PipelineStateStore(storage.database_path)
            store.claim_job(job)
            store.ensure_original_scene_revision(
                job.job_id,
                scene.scene_id,
                parameters={},
            )
            frame = storage.scene_frame_path(job.job_id, scene.scene_id, 1)
            frame.parent.mkdir(parents=True, exist_ok=True)
            frame.write_bytes(b"png")
            store.set_scene_state(
                job.job_id,
                scene.scene_id,
                SceneState.PENDING,
                frame_path=str(frame),
            )
            destination = storage.scene_clip_path(job.job_id, scene.scene_id, 1)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"unverified-assembly")
            supervisor = PipelineSupervisor(
                store=store,
                mail_client=FakeMailClient(),
                asset_manager=FakeAssetManager(),
                comfy=FakeComfy(frame),
                settings=SupervisorSettings(
                    continuation_mode="explicit",
                    i2v_timeout_seconds=10,
                ),
                storage=storage,
                frame_path_factory=lambda job_id, scene_id: storage.scene_frame_path(
                    job_id,
                    scene_id,
                    1,
                ),
                clip_path_factory=lambda job_id, scene_id: storage.scene_clip_path(
                    job_id,
                    scene_id,
                    1,
                ),
            )
            continuation = Mock()
            continuation.render_scene.return_value = SimpleNamespace(
                scene_path=destination
            )
            supervisor.continuation_renderer = continuation

            supervisor._process_i2v_stage(job, scene, {})

            continuation.render_scene.assert_called_once()
            record = store.scene_records(job.job_id)[0]
            self.assertEqual(record.state, SceneState.SUCCEEDED)
            self.assertEqual(record.i2v_attempts, 1)

    def test_interrupted_continuation_resumes_without_spending_scene_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            document = payload()
            document["scenes"][0]["estimated_sec"] = 10
            document["scenes"][0]["i2v"]["loras"] = []
            document["scenes"][0]["i2v"]["continuation"] = {"enabled": True}
            job = parse_job_payload(document)
            scene = job.scenes[0]
            storage = StorageLayout(root / "storage")
            storage.ensure()
            store = PipelineStateStore(storage.database_path)
            store.claim_job(job)
            store.ensure_original_scene_revision(
                job.job_id,
                scene.scene_id,
                parameters={"job_id": job.job_id, "scene_id": scene.scene_id},
            )
            frame = storage.scene_frame_path(job.job_id, scene.scene_id, 1)
            frame.parent.mkdir(parents=True, exist_ok=True)
            frame.write_bytes(b"frame")
            destination = storage.scene_clip_path(job.job_id, scene.scene_id, 1)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"raw-scene")
            supervisor = PipelineSupervisor(
                store=store,
                mail_client=FakeMailClient(),
                asset_manager=FakeAssetManager(),
                comfy=FakeComfy(frame),
                settings=SupervisorSettings(
                    continuation_mode="explicit",
                    i2v_timeout_seconds=10,
                ),
                storage=storage,
                frame_path_factory=lambda job_id, scene_id: storage.scene_frame_path(
                    job_id,
                    scene_id,
                    1,
                ),
                clip_path_factory=lambda job_id, scene_id: storage.scene_clip_path(
                    job_id,
                    scene_id,
                    1,
                ),
            )
            continuation = Mock()
            continuation.render_scene.side_effect = [
                RuntimeError("simulated supervisor restart"),
                SimpleNamespace(scene_path=destination),
            ]
            supervisor.continuation_renderer = continuation

            with self.assertRaisesRegex(RuntimeError, "supervisor restart"):
                supervisor._process_i2v_stage(job, scene, {})
            self.assertEqual(
                store.scene_records(job.job_id)[0].i2v_attempts,
                1,
            )
            supervisor._process_i2v_stage(job, scene, {})

            record = store.scene_records(job.job_id)[0]
            self.assertEqual(record.state, SceneState.SUCCEEDED)
            self.assertEqual(record.i2v_attempts, 1)
            self.assertEqual(continuation.render_scene.call_count, 2)

    def test_interrupted_i2v_route_does_not_change_with_launch_setting(self) -> None:
        cases = (
            ("i2v_legacy", "explicit", False),
            ("i2v_continuation", "disabled", True),
        )
        for prompt_stage, continuation_mode, expected in cases:
            with self.subTest(
                prompt_stage=prompt_stage,
                continuation_mode=continuation_mode,
            ), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                document = payload()
                document["scenes"][0]["estimated_sec"] = 10
                document["scenes"][0]["i2v"]["loras"] = []
                document["scenes"][0]["i2v"]["continuation"] = {"enabled": True}
                job = parse_job_payload(document)
                scene = job.scenes[0]
                storage = StorageLayout(root / "storage")
                storage.ensure()
                store = PipelineStateStore(storage.database_path)
                store.claim_job(job)
                store.begin_scene_stage(
                    job.job_id,
                    scene.scene_id,
                    PipelineState.RUNNING_I2V,
                    prompt_stage=prompt_stage,
                )
                supervisor = PipelineSupervisor(
                    store=store,
                    mail_client=FakeMailClient(),
                    asset_manager=FakeAssetManager(),
                    comfy=FakeComfy(root / "frame.png"),
                    settings=SupervisorSettings(
                        poll_interval_seconds=1,
                        t2i_timeout_seconds=10,
                        i2v_timeout_seconds=10,
                        max_stage_attempts=2,
                        continuation_mode=continuation_mode,
                    ),
                    storage=storage,
                )

                self.assertEqual(
                    supervisor._automatic_uses_continuation(
                        job.job_id,
                        scene,
                    ),
                    expected,
                )

    def test_explicit_legacy_route_outranks_a_stale_continuation_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            document = payload()
            document["scenes"][0]["estimated_sec"] = 10
            document["scenes"][0]["i2v"]["continuation"] = {"enabled": True}
            job = parse_job_payload(document)
            scene = job.scenes[0]
            storage = StorageLayout(root / "storage")
            storage.ensure()
            store = PipelineStateStore(storage.database_path)
            store.claim_job(job)
            store.ensure_continuation_plan(
                job.job_id,
                scene.scene_id,
                1,
                "stale-plan",
                {"strategy": "ltx23_latent_overlap_v1"},
            )
            store.begin_scene_stage(
                job.job_id,
                scene.scene_id,
                PipelineState.RUNNING_I2V,
                prompt_stage="i2v_legacy",
            )
            supervisor = PipelineSupervisor(
                store=store,
                mail_client=FakeMailClient(),
                asset_manager=FakeAssetManager(),
                comfy=FakeComfy(root / "frame.png"),
                settings=SupervisorSettings(
                    continuation_mode="explicit",
                    i2v_timeout_seconds=10,
                ),
                storage=storage,
            )

            self.assertFalse(
                supervisor._automatic_uses_continuation(job.job_id, scene)
            )

    def test_t2i_recovery_does_not_erase_prior_legacy_route(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            document = payload()
            document["scenes"][0]["estimated_sec"] = 10
            document["scenes"][0]["i2v"]["continuation"] = {"enabled": True}
            job = parse_job_payload(document)
            scene = job.scenes[0]
            storage = StorageLayout(root / "storage")
            storage.ensure()
            store = PipelineStateStore(storage.database_path)
            store.claim_job(job)
            store.begin_scene_stage(
                job.job_id,
                scene.scene_id,
                PipelineState.RUNNING_I2V,
                prompt_stage="i2v_legacy",
            )
            store.begin_scene_stage(
                job.job_id,
                scene.scene_id,
                PipelineState.RUNNING_T2I,
            )
            supervisor = PipelineSupervisor(
                store=store,
                mail_client=FakeMailClient(),
                asset_manager=FakeAssetManager(),
                comfy=FakeComfy(root / "frame.png"),
                settings=SupervisorSettings(
                    continuation_mode="explicit",
                    i2v_timeout_seconds=10,
                ),
                storage=storage,
            )

            self.assertFalse(
                supervisor._automatic_uses_continuation(job.job_id, scene)
            )

    def test_shared_i2v_route_uses_continuation_only_for_eligible_long_scene(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            document = payload()
            document["scenes"][0]["estimated_sec"] = 10
            document["scenes"][0]["i2v"]["loras"] = []
            document["scenes"][0]["i2v"]["continuation"] = {"enabled": True}
            job = parse_job_payload(document)
            scene = job.scenes[0]
            storage = StorageLayout(root / "storage")
            storage.ensure()
            store = PipelineStateStore(storage.database_path)
            store.claim_job(job)
            frame = storage.scene_frame_path(job.job_id, scene.scene_id, 1)
            frame.parent.mkdir(parents=True, exist_ok=True)
            frame.write_bytes(b"png")
            comfy = StageRecordingComfy([frame])
            supervisor = PipelineSupervisor(
                store=store,
                mail_client=FakeMailClient(),
                asset_manager=FakeAssetManager(),
                comfy=comfy,
                settings=SupervisorSettings(
                    poll_interval_seconds=1,
                    t2i_timeout_seconds=10,
                    i2v_timeout_seconds=10,
                    max_stage_attempts=2,
                    continuation_mode="explicit",
                ),
                storage=storage,
            )
            continuation = Mock()
            destination = storage.scene_clip_path(job.job_id, scene.scene_id, 1)
            continuation.render_scene.return_value = SimpleNamespace(
                scene_path=destination
            )
            supervisor.continuation_renderer = continuation

            result = supervisor.render_i2v_scene(
                job=job,
                scene=scene,
                frame_path=frame,
                destination=destination,
                resolved_lora_filenames={},
            )

            self.assertEqual(result, destination)
            continuation.render_scene.assert_called_once()
            self.assertEqual(comfy.workflows, [])

            short_document = copy.deepcopy(document)
            short_document["scenes"][0]["estimated_sec"] = 5
            short_job = parse_job_payload(short_document)
            short_scene = short_job.scenes[0]
            short_destination = storage.scene_clip_path(
                short_job.job_id,
                short_scene.scene_id,
                2,
            )
            supervisor.render_i2v_scene(
                job=short_job,
                scene=short_scene,
                frame_path=frame,
                destination=short_destination,
                resolved_lora_filenames={},
                revision=2,
                deliver_to_discord=False,
            )
            self.assertEqual(comfy.stages, ["i2v"])
            self.assertTrue(short_destination.is_file())

    def test_status_heartbeat_logs_redacted_pipeline_and_queue_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            job = parse_job_payload(payload())
            store = PipelineStateStore(root / "pipeline.sqlite3")
            store.claim_job(job)
            supervisor = PipelineSupervisor(
                store=store,
                mail_client=FakeMailClient(),
                asset_manager=FakeAssetManager(),
                comfy=FakeComfy(root / "unused.png"),
                settings=SupervisorSettings(status_interval_seconds=7),
            )

            with self.assertLogs("10MinVideoMaker.supervisor", level="INFO") as logs:
                supervisor._log_status()

            output = "\n".join(logs.output)
            self.assertIn("STATUS | state=downloading_assets", output)
            self.assertIn(f"job={job.job_id}", output)
            self.assertIn("scene=none", output)
            self.assertIn("ComfyUI queue=running=1 pending=2", output)
            self.assertNotIn("secret-prompt", output)

    def test_status_interval_loads_from_environment_and_must_be_positive(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TENMIN_POLL_SECONDS": "300",
                "TENMIN_T2I_TIMEOUT_SECONDS": "3600",
                "TENMIN_I2V_TIMEOUT_SECONDS": "21600",
                "TENMIN_MAX_STAGE_ATTEMPTS": "2",
                "TENMIN_STATUS_INTERVAL_SECONDS": "9",
                "TENMIN_LTX_CONTINUATION_MODE": "auto",
            },
            clear=True,
        ):
            settings = SupervisorSettings.from_environment()
            self.assertEqual(settings.status_interval_seconds, 9)
            self.assertEqual(settings.continuation_mode, "auto")
            self.assertFalse(settings.require_human_review)
        with patch.dict(
            os.environ,
            {"TENMIN_REQUIRE_HUMAN_REVIEW": "true"},
            clear=True,
        ):
            self.assertTrue(SupervisorSettings.from_environment().require_human_review)
        with self.assertRaisesRegex(ValueError, "status_interval_seconds"):
            SupervisorSettings(status_interval_seconds=0)
        with self.assertRaisesRegex(ValueError, "continuation_mode"):
            SupervisorSettings(continuation_mode="unknown")

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
                    PRODUCTION_WIDTH,
                    PRODUCTION_HEIGHT,
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

    def test_process_job_batches_all_t2i_before_i2v(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            raw = payload()
            second = copy.deepcopy(raw["scenes"][0])
            second["id"] = 2
            second["title"] = "Second batched scene"
            second["t2i"]["seed"] += 1
            second["i2v"]["seed"] += 1
            raw["scenes"].append(second)
            job = parse_job_payload(raw)
            store = PipelineStateStore(root / "pipeline.sqlite3")
            store.claim_job(job)
            frames = [
                root / "frames" / "scene_0001.png",
                root / "frames" / "scene_0002.png",
            ]
            comfy = StageRecordingComfy(frames)
            supervisor = PipelineSupervisor(
                store=store,
                mail_client=FakeMailClient(),
                asset_manager=FakeAssetManager(),
                comfy=comfy,
                assembler=FakeAssembler(root / "final.mp4"),
                settings=SupervisorSettings(1, 10, 10, 2),
                frame_path_factory=lambda _job, scene_id: root
                / "frames"
                / f"scene_{scene_id:04d}.png",
                clip_path_factory=lambda _job, scene_id: root
                / "clips"
                / f"scene_{scene_id:04d}.mp4",
                video_probe=lambda path: VideoStreamInfo(
                    Path(path), PRODUCTION_WIDTH, PRODUCTION_HEIGHT, Fraction(24, 1)
                ),
            )

            supervisor.process_job(job)

            self.assertEqual(comfy.stages, ["t2i", "t2i", "i2v", "i2v"])
            self.assertEqual(comfy.free_calls, 4)
            self.assertEqual(
                [record.state for record in store.scene_records(job.job_id)],
                [SceneState.SUCCEEDED, SceneState.SUCCEEDED],
            )

    def test_discord_video_delivery_runs_after_generation_as_a_separate_batch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            job = parse_job_payload(payload())
            storage = StorageLayout(root / "storage")
            storage.ensure()
            store = PipelineStateStore(storage.database_path)
            store.claim_job(job)
            clip = storage.scene_clip_path(job.job_id, 1, 1)
            clip.parent.mkdir(parents=True, exist_ok=True)
            clip.write_bytes(b"raw-mp4")
            store.set_scene_state(
                job.job_id,
                1,
                SceneState.SUCCEEDED,
                video_path=str(clip),
            )
            comfy = FakeComfy(root / "unused.png")
            supervisor = PipelineSupervisor(
                store=store,
                mail_client=FakeMailClient(),
                asset_manager=FakeAssetManager(),
                comfy=comfy,
                settings=SupervisorSettings(1, 10, 10, 2),
                storage=storage,
                delivery=DiscordDeliverySettings(
                    "https://discord.com/api/webhooks/123456789/test-token"
                ),
            )
            renderer = Mock()
            supervisor.continuation_renderer = renderer

            self.assertTrue(
                supervisor._deliver_i2v_batch(job, {1: job.scenes[0]})
            )

            renderer.deliver_existing_scene.assert_called_once_with(
                job,
                job.scenes[0],
                str(clip),
                revision=1,
                overrides=None,
                prompt_id_callback=unittest.mock.ANY,
            )
            self.assertEqual(comfy.free_calls, 1)

    def test_discord_failure_is_bounded_and_does_not_block_raw_final(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            job = parse_job_payload(payload())
            storage = StorageLayout(root / "storage")
            storage.ensure()
            store = PipelineStateStore(storage.database_path)
            store.claim_job(job)
            clip = storage.scene_clip_path(job.job_id, 1, 1)
            clip.parent.mkdir(parents=True, exist_ok=True)
            clip.write_bytes(b"raw-unwatermarked-mp4")
            store.set_scene_state(
                job.job_id,
                1,
                SceneState.SUCCEEDED,
                video_path=str(clip),
            )
            supervisor = PipelineSupervisor(
                store=store,
                mail_client=FakeMailClient(),
                asset_manager=FakeAssetManager(),
                comfy=FakeComfy(root / "unused.png"),
                settings=SupervisorSettings(1, 10, 10, 2),
                storage=storage,
                delivery=DiscordDeliverySettings(
                    "https://discord.com/api/webhooks/123456789/test-token"
                ),
            )
            renderer = Mock()
            renderer.deliver_existing_scene.side_effect = ContinuationDeliveryError(
                "Discord unavailable", state="FAILED"
            )
            supervisor.continuation_renderer = renderer

            self.assertTrue(
                supervisor._deliver_i2v_batch(job, {1: job.scenes[0]})
            )
            self.assertEqual(renderer.deliver_existing_scene.call_count, 2)
            self.assertEqual(clip.read_bytes(), b"raw-unwatermarked-mp4")
            self.assertEqual(
                store.scene_records(job.job_id)[0].state,
                SceneState.SUCCEEDED,
            )

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
                    Path(path), PRODUCTION_WIDTH, PRODUCTION_HEIGHT, Fraction(24, 1)
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
                    Path(path), PRODUCTION_WIDTH, PRODUCTION_HEIGHT, Fraction(24, 1)
                ),
            )

            supervisor.process_job(job)

            records = store.scene_records(job.job_id)
            self.assertEqual([record.state for record in records], [SceneState.SUCCEEDED, SceneState.FAILED])
            self.assertIn("download failed", records[1].error)
            self.assertEqual(len(assembler.calls), 1)
            self.assertEqual(mail.requests, [(job.job_id, False)])

    def test_remake_asset_resolution_is_limited_to_the_selected_scene(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            raw = payload()
            second = copy.deepcopy(raw["scenes"][0])
            second["id"] = 2
            second["title"] = "Unrelated missing asset scene"
            second["i2v"]["loras"] = [
                {
                    "name": "Missing Scene LoRA",
                    "download_url": "https://example.invalid/missing.safetensors",
                    "weight": 0.8,
                }
            ]
            raw["scenes"].append(second)
            job = parse_job_payload(raw)
            supervisor = PipelineSupervisor(
                store=PipelineStateStore(root / "pipeline.sqlite3"),
                mail_client=FakeMailClient(),
                asset_manager=OneMissingAssetManager(),
                comfy=FakeComfy(root / "unused.png"),
            )

            preparation = supervisor._resolve_assets(job, scene_ids={1})

            self.assertEqual(preparation.failures, {})
            self.assertNotIn("Missing Scene LoRA", " ".join(preparation.resolved_filenames))

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

    def test_malformed_continuation_fails_before_assets_or_comfy_queue(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            document = payload()
            document["scenes"][0]["i2v"]["continuation"] = {"enabled": True}
            document["scenes"][0]["i2v"]["segments"] = [
                {
                    "index": 0,
                    "requested_duration_seconds": 5.0,
                    "positive_prompt": "A short beat that does not cover the scene.",
                }
            ]
            job = parse_job_payload(document)
            store = PipelineStateStore(root / "pipeline.sqlite3")
            store.claim_job(job)
            assets = Mock(wraps=FakeAssetManager())
            comfy = FakeComfy(root / "unused.png")
            supervisor = PipelineSupervisor(
                store=store,
                mail_client=FakeMailClient(),
                asset_manager=assets,
                comfy=comfy,
                settings=SupervisorSettings(
                    1,
                    10,
                    10,
                    2,
                    continuation_mode="explicit",
                ),
            )

            supervisor.process_job(job)

            snapshot = store.snapshot()
            self.assertEqual(snapshot.state, PipelineState.ERROR)
            self.assertIn("Continuation preflight failed for all", snapshot.error)
            record = store.scene_records(job.job_id)[0]
            self.assertEqual(record.state, SceneState.FAILED)
            self.assertIn("complete scene timeline", record.error)
            assets.resolve_or_download.assert_not_called()
            assets.require_local.assert_not_called()
            self.assertEqual(comfy.workflows, [])

    def test_idle_tick_requests_only_when_no_pending_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            store = PipelineStateStore(root / "pipeline.sqlite3")
            mail = FakeMailClient()
            supervisor = PipelineSupervisor(
                store=store,
                mail_client=mail,
                asset_manager=FakeAssetManager(),
                comfy=FakeComfy(root / "unused.png"),
                settings=SupervisorSettings(1, 10, 10, 2),
            )

            self.assertEqual(store.snapshot().state, PipelineState.IDLE)
            supervisor.tick()

            self.assertEqual(mail.requests, [(None, None)])
            self.assertEqual(store.snapshot().state, PipelineState.WAITING_FOR_GROK)

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
                    Path(path), PRODUCTION_WIDTH, PRODUCTION_HEIGHT - 32, Fraction(24, 1)
                ),
            )

            supervisor.process_job(job)

            snapshot = store.snapshot()
            self.assertEqual(snapshot.state, PipelineState.ERROR)
            self.assertIn(
                f"expected {PRODUCTION_WIDTH}x{PRODUCTION_HEIGHT}",
                snapshot.error,
            )
            self.assertEqual(
                store.scene_states(job.job_id),
                {1: SceneState.SUCCEEDED},
            )
            self.assertTrue(clip.is_file())
            self.assertEqual(assembler.calls, [])
            self.assertEqual(mail.requests, [])


if __name__ == "__main__":
    unittest.main()
