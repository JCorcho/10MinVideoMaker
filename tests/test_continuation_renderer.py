from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tenminvideomaker.chunk_assembly import SceneChunkAssemblyError
from tenminvideomaker.comfy_http import ComfyHttpError
from tenminvideomaker.continuation_renderer import (
    CONTINUATION_CACHE_IMPLEMENTATION_PATHS,
    CONTINUATION_CONTRACT_NODE_TYPES,
    CONTINUATION_IMPLEMENTATION_PATHS,
    ContinuationDeliveryError,
    ContinuationRenderer,
)
from tenminvideomaker.continuation_workflow import ContinuationStage2Build
from tenminvideomaker.contracts import parse_job_payload
from tenminvideomaker.state_store import ChunkState, PipelineStateStore
from tenminvideomaker.storage import StorageLayout
from tenminvideomaker.workflow_builder import WorkflowBuild

from test_contracts import payload


class _FakeChunkAssembler:
    def __init__(self):
        self.assembly_calls = 0
        self.extractions = []

    def extract_frame(self, source, frame_index, destination):
        destination = Path(destination)
        if not destination.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"lossless-extracted-frame")
            self.extractions.append((Path(source), frame_index, destination))
        return destination

    def validate_chunk(self, _plan, _chunk_index, path):
        if not Path(path).is_file():
            raise SceneChunkAssemblyError("missing fake chunk")
        return object()

    def assemble(self, _plan, raw_chunks, destination):
        if not all(Path(path).is_file() for path in raw_chunks):
            raise SceneChunkAssemblyError("missing fake chunk")
        self.assembly_calls += 1
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"assembled-raw-scene")
        return destination

    def validate_scene(self, _plan, path):
        if not Path(path).is_file():
            raise SceneChunkAssemblyError("missing fake scene")
        return object()


class _FakeComfy:
    def __init__(self, checkpoints: set[tuple[int, int, str]]):
        self.checkpoints = checkpoints
        self.workflows: list[dict] = []
        self._by_prompt: dict[str, dict] = {}
        self._history: dict[str, dict] = {}
        self.cancelled_prompt_ids: list[str] = []

    def queue_prompt(self, workflow):
        prompt_id = f"prompt-{len(self.workflows) + 1}"
        self.workflows.append(workflow)
        self._by_prompt[prompt_id] = workflow
        return prompt_id

    def wait_for_prompt(self, prompt_id, *, timeout_seconds):
        del timeout_seconds
        workflow = self._by_prompt[prompt_id]
        node = workflow["1"]
        if node["class_type"] == "FakeDelivery":
            history = {"outputs": {}, "status": {"completed": True, "status_str": "success"}}
            self._history[prompt_id] = history
            return history
        chunk = node["inputs"]["chunk"]
        attempt = node["inputs"]["attempt"]
        if node["class_type"] == "FakeStage1":
            self.checkpoints.add((chunk, attempt, "stage1_handoff"))
            history = {"outputs": {}}
            self._history[prompt_id] = history
            return history
        if node["class_type"] == "FakeStage2":
            self.checkpoints.add((chunk, attempt, "stage2_video"))
            self.checkpoints.add((chunk, attempt, "stage2_audio"))
        history = {
            "outputs": {
                "1": {
                    "gifs": [
                        {
                            "filename": "fake.mkv",
                            "subfolder": "continuation",
                            "type": "temp",
                        }
                    ]
                }
            }
        }
        self._history[prompt_id] = history
        return history

    def completed_prompt(self, prompt_id):
        return self._history.get(prompt_id)

    def prompt_is_queued(self, prompt_id):
        return prompt_id in self._by_prompt and prompt_id not in self._history

    def cancel_owned_prompt(self, prompt_id):
        self.cancelled_prompt_ids.append(prompt_id)
        self._by_prompt.pop(prompt_id, None)
        return True

    def cancel_prompt(self, prompt_id):
        self.cancel_owned_prompt(prompt_id)

    def download_output(self, _metadata, destination):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"raw-window")
        return destination


def _stage1_build(*_args, **kwargs):
    chunk = _args[4]
    return WorkflowBuild(
        api={
            "1": {
                "class_type": "FakeStage1",
                "inputs": {
                    "chunk": chunk.index,
                    "attempt": kwargs["attempt_number"],
                },
            }
        },
        output_node_id="1",
        filename_prefix="fake-stage1",
    )


def _stage2_build(*_args, **kwargs):
    chunk = _args[4]
    workflow = WorkflowBuild(
        api={
            "1": {
                "class_type": "FakeStage2",
                "inputs": {
                    "chunk": chunk.index,
                    "attempt": kwargs["attempt_number"],
                },
            }
        },
        output_node_id="1",
        filename_prefix="fake-stage2",
    )
    return ContinuationStage2Build(
        workflow=workflow,
        video_checkpoint_node_id="1",
        audio_checkpoint_node_id="1",
    )


def _decode_build(*_args, **kwargs):
    chunk = _args[3]
    return WorkflowBuild(
        api={
            "1": {
                "class_type": "FakeDecode",
                "inputs": {
                    "chunk": chunk.index,
                    "attempt": kwargs["attempt_number"],
                },
            }
        },
        output_node_id="1",
        filename_prefix="fake-decode",
    )


def _delivery_build(*_args, **_kwargs):
    return WorkflowBuild(
        api={
            "1": {
                "class_type": "FakeDelivery",
                "inputs": {"delivery_revision": 1},
            }
        },
        output_node_id="1",
        filename_prefix="fake-delivery",
    )


class ContinuationRendererTests(unittest.TestCase):
    def test_runtime_identity_hashes_every_representative_graph_contract(self) -> None:
        class ContractComfy:
            def __init__(self) -> None:
                self.queried: list[str] = []

            def object_info(self, node_type: str):
                self.queried.append(node_type)
                return {node_type: {"input": {}, "output": []}}

        renderer = object.__new__(ContinuationRenderer)
        renderer.comfy = ContractComfy()

        digest = renderer.runtime_contract_sha256()

        self.assertEqual(len(digest), 64)
        self.assertEqual(
            renderer.comfy.queried,
            list(CONTINUATION_CONTRACT_NODE_TYPES),
        )
        self.assertIn("LTXVAddGuide", renderer.comfy.queried)
        self.assertIn("DiscordSendSaveVideo", renderer.comfy.queried)

    def test_runtime_identity_ignores_dynamic_combo_membership(self) -> None:
        class ContractComfy:
            def __init__(self, choices):
                self.choices = choices

            def object_info(self, node_type: str):
                return {
                    node_type: {
                        "input": {
                            "required": {
                                "model_name": [
                                    list(self.choices),
                                    {"default": self.choices[0]},
                                ]
                            }
                        },
                        "output": ["MODEL"],
                    }
                }

        renderer = object.__new__(ContinuationRenderer)
        renderer.comfy = ContractComfy(("one.safetensors",))
        first = renderer.runtime_contract_sha256()
        renderer.comfy = ContractComfy(
            ("one.safetensors", "new-unrelated-lora.safetensors")
        )
        second = renderer.runtime_contract_sha256()

        self.assertEqual(first, second)

    def test_runtime_identity_tracks_output_names_and_noncombo_defaults(self) -> None:
        class ContractComfy:
            def __init__(self, *, output_name, default):
                self.output_name = output_name
                self.default = default

            def object_info(self, node_type: str):
                return {
                    node_type: {
                        "input": {
                            "required": {
                                "strength": [
                                    "FLOAT",
                                    {"default": self.default},
                                ]
                            }
                        },
                        "output": ["LATENT", "LATENT"],
                        "output_name": list(self.output_name),
                    }
                }

        renderer = object.__new__(ContinuationRenderer)
        renderer.comfy = ContractComfy(
            output_name=("video", "audio"),
            default=0.5,
        )
        baseline = renderer.runtime_contract_sha256()
        renderer.comfy = ContractComfy(
            output_name=("audio", "video"),
            default=0.5,
        )
        renamed = renderer.runtime_contract_sha256()
        renderer.comfy = ContractComfy(
            output_name=("video", "audio"),
            default=0.75,
        )
        changed_default = renderer.runtime_contract_sha256()

        self.assertNotEqual(baseline, renamed)
        self.assertNotEqual(baseline, changed_default)

    def test_implementation_identity_covers_generation_routing_and_recovery(self) -> None:
        required = {
            "scripts/run_gui.py",
            "scripts/run_supervisor.py",
            "scripts/setup_and_start.py",
            "tenminvideomaker/artifacts.py",
            "tenminvideomaker/continuation_validation.py",
            "tenminvideomaker/contracts.py",
            "tenminvideomaker/gui_app.py",
            "tenminvideomaker/gui_service.py",
            "tenminvideomaker/server_api.py",
            "tenminvideomaker/state_store.py",
            "tenminvideomaker/storage.py",
            "tenminvideomaker/supervisor.py",
        }
        self.assertTrue(required.issubset(CONTINUATION_IMPLEMENTATION_PATHS))

        project_root = Path(__file__).resolve().parents[1]
        with patch(
            "tenminvideomaker.continuation_renderer.sha256_file",
            return_value="f" * 64,
        ) as hasher:
            digest = ContinuationRenderer.implementation_sha256()

        self.assertEqual(len(digest), 64)
        self.assertEqual(
            [
                call.args[0].relative_to(project_root).as_posix()
                for call in hasher.call_args_list
            ],
            list(CONTINUATION_IMPLEMENTATION_PATHS),
        )

    def test_runtime_contract_identity_uses_native_ltx_spatial_refinement(self) -> None:
        self.assertIn("LTXVLatentUpsamplerTiled", CONTINUATION_CONTRACT_NODE_TYPES)
        self.assertNotIn("UpscaleModelLoader", CONTINUATION_CONTRACT_NODE_TYPES)
        self.assertNotIn("ImageUpscaleWithModel", CONTINUATION_CONTRACT_NODE_TYPES)

    def test_cache_identity_covers_only_generation_affecting_code(self) -> None:
        self.assertIn(
            "tenminvideomaker/continuation_workflow.py",
            CONTINUATION_CACHE_IMPLEMENTATION_PATHS,
        )
        self.assertIn(
            "tenminvideomaker/chunk_artifacts.py",
            CONTINUATION_CACHE_IMPLEMENTATION_PATHS,
        )
        self.assertNotIn(
            "tenminvideomaker/gui_app.py",
            CONTINUATION_CACHE_IMPLEMENTATION_PATHS,
        )
        self.assertNotIn(
            "scripts/setup_and_start.py",
            CONTINUATION_CACHE_IMPLEMENTATION_PATHS,
        )

    def _fixture(self, root: Path):
        document = copy.deepcopy(payload())
        document["scenes"][0]["estimated_sec"] = 10
        document["scenes"][0]["i2v"]["continuation"] = {"enabled": True}
        job = parse_job_payload(document)
        scene = job.scenes[0]
        storage = StorageLayout(root / "storage")
        storage.ensure()
        store = PipelineStateStore(storage.database_path)
        store.claim_job(job)
        frame = storage.scene_frame_path(job.job_id, scene.scene_id, 1)
        frame.parent.mkdir(parents=True, exist_ok=True)
        frame.write_bytes(b"frame")
        return job, scene, storage, store, frame

    def test_attempt_parameters_do_not_record_removed_pixel_upscaler(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            job, scene, _storage, _store, frame = self._fixture(Path(directory))
            renderer = ContinuationRenderer(
                store=_store,
                storage=_storage,
                comfy=_FakeComfy(set()),
                assembler=_FakeChunkAssembler(),
                timeout_seconds=10,
                max_attempts=2,
            )
            plan = renderer._build_plan(job, scene, 1, None)

            parameters = renderer._attempt_parameters(
                job,
                scene,
                frame,
                plan,
                0,
                {},
                None,
                "f" * 64,
            )

            self.assertNotIn(
                "continuation_video_upscaler_filename",
                parameters["runtime_identity"],
            )

    def test_renders_exact_frame_chunks_then_reuses_verified_lineage_and_scene(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job, scene, storage, store, frame = self._fixture(root)
            checkpoints: set[tuple[int, int, str]] = set()
            comfy = _FakeComfy(checkpoints)
            assembler = _FakeChunkAssembler()
            renderer = ContinuationRenderer(
                store=store,
                storage=storage,
                comfy=comfy,
                assembler=assembler,
                timeout_seconds=10,
                max_attempts=2,
            )
            destination = storage.scene_clip_path(job.job_id, scene.scene_id, 1)

            def checkpoint_valid(_layout, **coordinates):
                return (
                    coordinates["chunk_index"],
                    coordinates["attempt_number"],
                    coordinates["artifact_kind"],
                ) in checkpoints

            def load_checkpoint(_layout, **coordinates):
                key = (
                    coordinates["chunk_index"],
                    coordinates["attempt_number"],
                    coordinates["artifact_kind"],
                )
                if key not in checkpoints:
                    raise AssertionError(f"unexpected missing checkpoint {key}")
                return ({}, {"sha256": "-".join(map(str, key))})

            with (
                patch(
                    "tenminvideomaker.continuation_renderer."
                    "build_continuation_stage1_workflow",
                    side_effect=_stage1_build,
                ),
                patch(
                    "tenminvideomaker.continuation_renderer."
                    "build_continuation_stage2_workflow",
                    side_effect=_stage2_build,
                ),
                patch(
                    "tenminvideomaker.continuation_renderer."
                    "latent_checkpoint_is_valid",
                    side_effect=checkpoint_valid,
                ),
                patch(
                    "tenminvideomaker.continuation_renderer.load_latent_checkpoint",
                    side_effect=load_checkpoint,
                ),
            ):
                first = renderer.render_scene(
                    job,
                    scene,
                    frame,
                    destination,
                    revision=1,
                    deliver_to_discord=False,
                )
                second = renderer.render_scene(
                    job,
                    scene,
                    frame,
                    destination,
                    revision=1,
                    deliver_to_discord=False,
                )

            self.assertEqual(first.plan.chunk_count, 2)
            self.assertEqual(len(comfy.workflows), 4)
            self.assertEqual(len(assembler.extractions), 1)
            self.assertEqual(assembler.extractions[0][1], 120)
            self.assertIn("input_frames", str(assembler.extractions[0][2]))
            self.assertEqual(assembler.assembly_calls, 1)
            self.assertFalse(first.reused_scene_assembly)
            self.assertTrue(second.reused_scene_assembly)
            progress = store.chunk_progress(job.job_id, scene.scene_id, 1)
            self.assertEqual(progress.total_count, 2)
            self.assertEqual(progress.complete_count, 2)
            self.assertTrue(destination.is_file())

    def test_server_down_preserves_active_attempt_without_spending_retry_budget(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job, scene, storage, store, frame = self._fixture(root)
            comfy = _FakeComfy(set())
            comfy.alive = lambda: False
            renderer = ContinuationRenderer(
                store=store,
                storage=storage,
                comfy=comfy,
                assembler=_FakeChunkAssembler(),
                timeout_seconds=10,
                max_attempts=2,
            )
            plan = renderer._build_plan(job, scene, 1, None)
            renderer._persist_plan(job, scene, 1, plan)

            with patch.object(
                renderer,
                "_execute_attempt",
                side_effect=ComfyHttpError("connection refused"),
            ) as execute:
                for _restart in range(2):
                    with self.assertRaisesRegex(ComfyHttpError, "connection refused"):
                        renderer._render_chunk(
                            job,
                            scene,
                            frame,
                            1,
                            plan,
                            0,
                            None,
                            {},
                            None,
                            "f" * 64,
                            None,
                        )

            attempts = store.chunk_attempts(job.job_id, scene.scene_id, 1, 0)
            self.assertEqual(len(attempts), 1)
            self.assertEqual(attempts[0].attempt_number, 1)
            self.assertEqual(attempts[0].state, ChunkState.GENERATING_STAGE1)
            self.assertEqual(
                store.chunk_records(job.job_id, scene.scene_id, 1)[0].state,
                ChunkState.GENERATING_STAGE1,
            )
            self.assertEqual(execute.call_count, 2)

    def test_stage2_av_checkpoints_resume_with_decode_only_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job, scene, storage, store, frame = self._fixture(root)
            checkpoints = {
                (0, 1, "stage1_handoff"),
                (0, 1, "stage2_video"),
                (0, 1, "stage2_audio"),
            }
            comfy = _FakeComfy(checkpoints)
            renderer = ContinuationRenderer(
                store=store,
                storage=storage,
                comfy=comfy,
                assembler=_FakeChunkAssembler(),
                timeout_seconds=10,
                max_attempts=2,
            )
            plan = renderer._build_plan(job, scene, 1, None)
            renderer._persist_plan(job, scene, 1, plan)
            parameters = renderer._attempt_parameters(
                job,
                scene,
                frame,
                plan,
                0,
                {},
                None,
                "f" * 64,
            )
            attempt = store.begin_chunk_attempt(
                job.job_id,
                scene.scene_id,
                1,
                0,
                seed=plan.chunks[0].seed,
                parameters=parameters,
            )

            def checkpoint_valid(_layout, **coordinates):
                return (
                    coordinates["chunk_index"],
                    coordinates["attempt_number"],
                    coordinates["artifact_kind"],
                ) in checkpoints

            def load_checkpoint(_layout, **coordinates):
                key = (
                    coordinates["chunk_index"],
                    coordinates["attempt_number"],
                    coordinates["artifact_kind"],
                )
                if key not in checkpoints:
                    raise AssertionError(f"unexpected missing checkpoint {key}")
                return ({}, {"sha256": "-".join(map(str, key))})

            with (
                patch(
                    "tenminvideomaker.continuation_renderer."
                    "latent_checkpoint_is_valid",
                    side_effect=checkpoint_valid,
                ),
                patch(
                    "tenminvideomaker.continuation_renderer.load_latent_checkpoint",
                    side_effect=load_checkpoint,
                ),
                patch(
                    "tenminvideomaker.continuation_renderer."
                    "build_continuation_decode_workflow",
                    side_effect=_decode_build,
                ),
            ):
                completed = renderer._execute_attempt(
                    job,
                    scene,
                    frame,
                    1,
                    plan,
                    0,
                    attempt,
                    None,
                    {},
                    None,
                    None,
                )

            self.assertEqual(
                [workflow["1"]["class_type"] for workflow in comfy.workflows],
                ["FakeDecode"],
            )
            self.assertEqual(completed.state, ChunkState.COMPLETE)
            self.assertIn(
                "stage2_audio_checkpoint_sha256",
                completed.result,
            )
            self.assertTrue(Path(completed.video_path).is_file())

    def test_failed_stage2_history_falls_back_to_decode_only_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job, scene, storage, store, frame = self._fixture(root)
            checkpoints = {
                (0, 1, "stage1_handoff"),
                (0, 1, "stage2_video"),
                (0, 1, "stage2_audio"),
            }
            comfy = _FakeComfy(checkpoints)
            renderer = ContinuationRenderer(
                store=store,
                storage=storage,
                comfy=comfy,
                assembler=_FakeChunkAssembler(),
                timeout_seconds=10,
                max_attempts=2,
            )
            plan = renderer._build_plan(job, scene, 1, None)
            renderer._persist_plan(job, scene, 1, plan)
            parameters = renderer._attempt_parameters(
                job,
                scene,
                frame,
                plan,
                0,
                {},
                None,
                "f" * 64,
            )
            attempt = store.begin_chunk_attempt(
                job.job_id,
                scene.scene_id,
                1,
                0,
                seed=plan.chunks[0].seed,
                parameters=parameters,
            )
            attempt = store.update_chunk_attempt(
                job.job_id,
                scene.scene_id,
                1,
                0,
                attempt.attempt_number,
                ChunkState.GENERATING_STAGE2,
                result={"stage2_prompt_id": "failed-stage2"},
            )

            def checkpoint_valid(_layout, **coordinates):
                return (
                    coordinates["chunk_index"],
                    coordinates["attempt_number"],
                    coordinates["artifact_kind"],
                ) in checkpoints

            def load_checkpoint(_layout, **coordinates):
                key = (
                    coordinates["chunk_index"],
                    coordinates["attempt_number"],
                    coordinates["artifact_kind"],
                )
                return ({}, {"sha256": "-".join(map(str, key))})

            original_completed_prompt = comfy.completed_prompt

            def completed_prompt(prompt_id):
                if prompt_id == "failed-stage2":
                    raise ComfyHttpError("stage-two encode failed")
                return original_completed_prompt(prompt_id)

            comfy.completed_prompt = completed_prompt
            with (
                patch(
                    "tenminvideomaker.continuation_renderer."
                    "latent_checkpoint_is_valid",
                    side_effect=checkpoint_valid,
                ),
                patch(
                    "tenminvideomaker.continuation_renderer.load_latent_checkpoint",
                    side_effect=load_checkpoint,
                ),
                patch(
                    "tenminvideomaker.continuation_renderer."
                    "build_continuation_decode_workflow",
                    side_effect=_decode_build,
                ),
            ):
                completed = renderer._execute_attempt(
                    job,
                    scene,
                    frame,
                    1,
                    plan,
                    0,
                    attempt,
                    None,
                    {},
                    None,
                    None,
                )

            self.assertEqual(
                [workflow["1"]["class_type"] for workflow in comfy.workflows],
                ["FakeDecode"],
            )
            self.assertEqual(completed.state, ChunkState.COMPLETE)

    def test_corrupt_accepted_manifest_invalidates_and_regenerates_descendants(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job, scene, storage, store, frame = self._fixture(root)
            checkpoints: set[tuple[int, int, str]] = set()
            comfy = _FakeComfy(checkpoints)
            renderer = ContinuationRenderer(
                store=store,
                storage=storage,
                comfy=comfy,
                assembler=_FakeChunkAssembler(),
                timeout_seconds=10,
                max_attempts=2,
            )
            destination = storage.scene_clip_path(job.job_id, scene.scene_id, 1)

            def checkpoint_valid(_layout, **coordinates):
                return (
                    coordinates["chunk_index"],
                    coordinates["attempt_number"],
                    coordinates["artifact_kind"],
                ) in checkpoints

            def load_checkpoint(_layout, **coordinates):
                key = (
                    coordinates["chunk_index"],
                    coordinates["attempt_number"],
                    coordinates["artifact_kind"],
                )
                return ({}, {"sha256": "-".join(map(str, key))})

            patches = (
                patch(
                    "tenminvideomaker.continuation_renderer."
                    "build_continuation_stage1_workflow",
                    side_effect=_stage1_build,
                ),
                patch(
                    "tenminvideomaker.continuation_renderer."
                    "build_continuation_stage2_workflow",
                    side_effect=_stage2_build,
                ),
                patch(
                    "tenminvideomaker.continuation_renderer."
                    "latent_checkpoint_is_valid",
                    side_effect=checkpoint_valid,
                ),
                patch(
                    "tenminvideomaker.continuation_renderer.load_latent_checkpoint",
                    side_effect=load_checkpoint,
                ),
            )
            with patches[0], patches[1], patches[2], patches[3]:
                renderer.render_scene(
                    job,
                    scene,
                    frame,
                    destination,
                    revision=1,
                    deliver_to_discord=False,
                )
                first_manifest = storage.chunk_attempt_manifest_path(
                    job.job_id,
                    scene.scene_id,
                    1,
                    0,
                    1,
                )
                first_manifest.write_text("corrupt", encoding="utf-8")
                renderer.render_scene(
                    job,
                    scene,
                    frame,
                    destination,
                    revision=1,
                    deliver_to_discord=False,
                )

            self.assertEqual(len(comfy.workflows), 8)
            self.assertEqual(
                [
                    chunk.accepted_attempt_number
                    for chunk in store.chunk_records(job.job_id, scene.scene_id, 1)
                ],
                [2, 2],
            )

    def test_prompt_id_is_persisted_before_wait_and_reclaimed_from_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job, scene, storage, store, frame = self._fixture(root)
            comfy = _FakeComfy(set())
            renderer = ContinuationRenderer(
                store=store,
                storage=storage,
                comfy=comfy,
                assembler=_FakeChunkAssembler(),
                timeout_seconds=10,
                max_attempts=2,
            )
            plan = renderer._build_plan(job, scene, 1, None)
            renderer._persist_plan(job, scene, 1, plan)
            attempt = store.begin_chunk_attempt(
                job.job_id,
                scene.scene_id,
                1,
                0,
                seed=plan.chunks[0].seed,
                parameters={"test": True},
            )
            identity = {
                "job_id": job.job_id,
                "scene_id": scene.scene_id,
                "revision": 1,
                "chunk_index": 0,
                "attempt_number": attempt.attempt_number,
            }
            workflow = {"1": {"class_type": "FakeStage1", "inputs": {"chunk": 0, "attempt": 1}}}

            with patch.object(
                comfy,
                "wait_for_prompt",
                side_effect=RuntimeError("simulated supervisor crash"),
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated"):
                    renderer._run_or_reclaim_prompt(
                        workflow=workflow,
                        identity=identity,
                        attempt_state=store.chunk_attempts(
                            job.job_id, scene.scene_id, 1, 0
                        )[0].state,
                        result={},
                        prompt_key="stage1_prompt_id",
                        workflow_hash_key="stage1_workflow_sha256",
                        prompt_id_callback=None,
                    )

            persisted = store.chunk_attempts(job.job_id, scene.scene_id, 1, 0)[0]
            prompt_id = persisted.result["stage1_prompt_id"]
            self.assertEqual(len(comfy.workflows), 1)
            comfy._history[prompt_id] = {"outputs": {}}
            history, _result = renderer._run_or_reclaim_prompt(
                workflow=workflow,
                identity=identity,
                attempt_state=persisted.state,
                result=persisted.result,
                prompt_key="stage1_prompt_id",
                workflow_hash_key="stage1_workflow_sha256",
                prompt_id_callback=None,
            )
            self.assertEqual(history, {"outputs": {}})
            self.assertEqual(len(comfy.workflows), 1)

    def test_generation_prompt_is_cancelled_when_ownership_persist_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job, scene, storage, store, _frame = self._fixture(root)
            comfy = _FakeComfy(set())
            renderer = ContinuationRenderer(
                store=store,
                storage=storage,
                comfy=comfy,
                assembler=_FakeChunkAssembler(),
                timeout_seconds=10,
                max_attempts=2,
            )
            plan = renderer._build_plan(job, scene, 1, None)
            renderer._persist_plan(job, scene, 1, plan)
            attempt = store.begin_chunk_attempt(
                job.job_id,
                scene.scene_id,
                1,
                0,
                seed=plan.chunks[0].seed,
                parameters={"test": True},
            )
            identity = {
                "job_id": job.job_id,
                "scene_id": scene.scene_id,
                "revision": 1,
                "chunk_index": 0,
                "attempt_number": attempt.attempt_number,
            }
            workflow = {
                "1": {
                    "class_type": "FakeStage1",
                    "inputs": {"chunk": 0, "attempt": 1},
                }
            }

            with patch.object(
                store,
                "update_chunk_attempt",
                side_effect=OSError("simulated SQLite write failure"),
            ):
                with self.assertRaisesRegex(OSError, "SQLite write failure"):
                    renderer._run_or_reclaim_prompt(
                        workflow=workflow,
                        identity=identity,
                        attempt_state=attempt.state,
                        result={},
                        prompt_key="stage1_prompt_id",
                        workflow_hash_key="stage1_workflow_sha256",
                        prompt_id_callback=None,
                    )

            self.assertEqual(comfy.cancelled_prompt_ids, ["prompt-1"])

    def _delivery_fixture(self, root: Path):
        job, scene, storage, store, _frame = self._fixture(root)
        scene_path = storage.scene_clip_path(job.job_id, scene.scene_id, 1)
        scene_path.parent.mkdir(parents=True, exist_ok=True)
        scene_path.write_bytes(b"raw-unwatermarked-scene")
        comfy = _FakeComfy(set())
        renderer = ContinuationRenderer(
            store=store,
            storage=storage,
            comfy=comfy,
            assembler=_FakeChunkAssembler(),
            timeout_seconds=10,
            max_attempts=2,
            webhook_url="https://discord.invalid/test",
        )
        marker = (
            storage.scene_assembly_root(job.job_id, scene.scene_id, 1)
            / "discord-delivery.json"
        )
        return job, scene, renderer, comfy, scene_path, marker

    def test_delivery_crash_after_queue_leaves_complete_queued_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            job, scene, renderer, comfy, scene_path, marker = self._delivery_fixture(
                Path(directory)
            )
            with (
                patch(
                    "tenminvideomaker.continuation_renderer."
                    "build_assembled_scene_delivery_workflow",
                    side_effect=_delivery_build,
                ),
                patch.object(
                    comfy,
                    "wait_for_prompt",
                    side_effect=RuntimeError("simulated process crash"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated process crash"):
                    renderer.deliver_existing_scene(
                        job,
                        scene,
                        scene_path,
                        revision=1,
                    )

            document = json.loads(marker.read_text(encoding="utf-8"))
            self.assertEqual(document["status"], "queued")
            self.assertEqual(document["prompt_id"], "prompt-1")
            self.assertEqual(document["job_id"], job.job_id)
            self.assertEqual(document["scene_id"], scene.scene_id)
            self.assertEqual(document["revision"], 1)
            self.assertEqual(len(document["scene_sha256"]), 64)
            self.assertEqual(len(document["plan_hash"]), 64)
            self.assertEqual(len(document["workflow_sha256"]), 64)

    def test_delivery_reclaims_queued_prompt_after_restart_without_duplicate_send(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            job, scene, renderer, comfy, scene_path, marker = self._delivery_fixture(
                Path(directory)
            )
            with (
                patch(
                    "tenminvideomaker.continuation_renderer."
                    "build_assembled_scene_delivery_workflow",
                    side_effect=_delivery_build,
                ),
                patch.object(
                    comfy,
                    "wait_for_prompt",
                    side_effect=RuntimeError("simulated process crash"),
                ),
            ):
                with self.assertRaises(RuntimeError):
                    renderer.deliver_existing_scene(
                        job,
                        scene,
                        scene_path,
                        revision=1,
                    )

            with patch(
                "tenminvideomaker.continuation_renderer."
                "build_assembled_scene_delivery_workflow",
                side_effect=_delivery_build,
            ):
                result = renderer.deliver_existing_scene(
                    job,
                    scene,
                    scene_path,
                    revision=1,
                )

            self.assertIsNotNone(result)
            self.assertTrue(result.reused_prompt)
            self.assertEqual(result.prompt_id, "prompt-1")
            self.assertEqual(len(comfy.workflows), 1)
            self.assertEqual(
                json.loads(marker.read_text(encoding="utf-8"))["status"],
                "sent",
            )

    def test_delivery_reclaims_completed_prompt_after_precommit_crash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            job, scene, renderer, comfy, scene_path, marker = self._delivery_fixture(
                Path(directory)
            )
            with (
                patch(
                    "tenminvideomaker.continuation_renderer."
                    "build_assembled_scene_delivery_workflow",
                    side_effect=_delivery_build,
                ),
                patch.object(
                    renderer,
                    "_commit_sent_delivery",
                    side_effect=RuntimeError("crash before sent marker"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "sent marker"):
                    renderer.deliver_existing_scene(
                        job,
                        scene,
                        scene_path,
                        revision=1,
                    )

            self.assertIn("prompt-1", comfy._history)
            self.assertEqual(
                json.loads(marker.read_text(encoding="utf-8"))["status"],
                "queued",
            )
            with patch(
                "tenminvideomaker.continuation_renderer."
                "build_assembled_scene_delivery_workflow",
                side_effect=_delivery_build,
            ):
                result = renderer.deliver_existing_scene(
                    job,
                    scene,
                    scene_path,
                    revision=1,
                )

            self.assertTrue(result.reused_prompt)
            self.assertEqual(len(comfy.workflows), 1)
            self.assertEqual(
                json.loads(marker.read_text(encoding="utf-8"))["status"],
                "sent",
            )

    def test_delivery_sent_marker_skips_duplicate_send(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            job, scene, renderer, comfy, scene_path, marker = self._delivery_fixture(
                Path(directory)
            )
            with patch(
                "tenminvideomaker.continuation_renderer."
                "build_assembled_scene_delivery_workflow",
                side_effect=_delivery_build,
            ):
                first = renderer.deliver_existing_scene(
                    job,
                    scene,
                    scene_path,
                    revision=1,
                )
                second = renderer.deliver_existing_scene(
                    job,
                    scene,
                    scene_path,
                    revision=1,
                )

            self.assertFalse(first.reused_prompt)
            self.assertTrue(second.reused_prompt)
            self.assertEqual(first.prompt_id, second.prompt_id)
            self.assertEqual(len(comfy.workflows), 1)
            self.assertEqual(
                json.loads(marker.read_text(encoding="utf-8"))["status"],
                "sent",
            )

    def test_delivery_stale_marker_requeues_current_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            job, scene, renderer, comfy, scene_path, marker = self._delivery_fixture(
                Path(directory)
            )
            with patch(
                "tenminvideomaker.continuation_renderer."
                "build_assembled_scene_delivery_workflow",
                side_effect=_delivery_build,
            ):
                renderer.deliver_existing_scene(
                    job,
                    scene,
                    scene_path,
                    revision=1,
                )
                stale = json.loads(marker.read_text(encoding="utf-8"))
                stale["workflow_sha256"] = "0" * 64
                marker.write_text(json.dumps(stale), encoding="utf-8")
                result = renderer.deliver_existing_scene(
                    job,
                    scene,
                    scene_path,
                    revision=1,
                )

            self.assertEqual(len(comfy.workflows), 2)
            self.assertEqual(result.prompt_id, "prompt-2")
            current = json.loads(marker.read_text(encoding="utf-8"))
            self.assertEqual(current["status"], "sent")
            self.assertNotEqual(current["workflow_sha256"], "0" * 64)

    def test_delivery_marker_write_failure_cancels_owned_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            job, scene, renderer, comfy, scene_path, marker = self._delivery_fixture(
                Path(directory)
            )
            with (
                patch(
                    "tenminvideomaker.continuation_renderer."
                    "build_assembled_scene_delivery_workflow",
                    side_effect=_delivery_build,
                ),
                patch(
                    "tenminvideomaker.continuation_renderer.write_json_atomic",
                    side_effect=OSError("disk full"),
                ),
            ):
                with self.assertRaisesRegex(
                    ContinuationDeliveryError,
                    "durable marker",
                ):
                    renderer.deliver_existing_scene(
                        job,
                        scene,
                        scene_path,
                        revision=1,
                    )

            self.assertEqual(comfy.cancelled_prompt_ids, ["prompt-1"])
            self.assertFalse(marker.exists())

    def test_delivery_failure_is_dedicated_and_marks_failed_without_deleting_raw(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            job, scene, renderer, comfy, scene_path, marker = self._delivery_fixture(
                Path(directory)
            )
            with (
                patch(
                    "tenminvideomaker.continuation_renderer."
                    "build_assembled_scene_delivery_workflow",
                    side_effect=_delivery_build,
                ),
                patch.object(
                    comfy,
                    "wait_for_prompt",
                    side_effect=ComfyHttpError("Discord node failed"),
                ),
                patch.object(
                    comfy,
                    "completed_prompt",
                    side_effect=ComfyHttpError("Discord node failed"),
                ),
                patch.object(comfy, "prompt_is_queued", return_value=False),
            ):
                with self.assertRaisesRegex(
                    ContinuationDeliveryError,
                    "raw scene output remains valid",
                ):
                    renderer.deliver_existing_scene(
                        job,
                        scene,
                        scene_path,
                        revision=1,
                    )

            self.assertEqual(scene_path.read_bytes(), b"raw-unwatermarked-scene")
            document = json.loads(marker.read_text(encoding="utf-8"))
            self.assertEqual(document["status"], "failed")
            self.assertEqual(document["prompt_id"], "prompt-1")
            self.assertEqual(document["failure_reason"], "prompt_failed")
