from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tenminvideomaker.contracts import parse_job_payload
from tenminvideomaker.qc_config import QualityControlSettings
from tenminvideomaker.qc_backend import (
    BackendIdentity,
    RepairPlannerResponse,
    VisionJudgeEvaluation,
)
from tenminvideomaker.qc_contracts import (
    QcCandidateState,
    QcDecision,
    QcHumanDecision,
    QcTier,
    evaluation_idempotency_key,
    parse_judge_response,
)
from tenminvideomaker.qc_video import SampledFrame, SampledVideo, VideoMetadata
from tenminvideomaker.qc_controller import Phase1QcController, QcControllerError
from tenminvideomaker.review import scene_review_document
from tenminvideomaker.state_store import PipelineStateStore, SceneState
from tenminvideomaker.storage import StorageLayout

from test_contracts import payload


class Phase1QcControllerRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.layout = StorageLayout(self.root)
        self.store = PipelineStateStore(self.layout.database_path)
        self.job = parse_job_payload(payload())
        self.store.claim_job(self.job)
        self.video = self.root / "original.mp4"
        self.frame = self.root / "frame.png"
        self.video.write_bytes(b"non-production-video")
        self.frame.write_bytes(b"frame")
        self.document = scene_review_document(self.job, self.job.scenes[0])
        self.store.set_scene_state(
            self.job.job_id,
            1,
            SceneState.SUCCEEDED,
            frame_path=str(self.frame),
            video_path=str(self.video),
        )
        self.store.ensure_original_scene_revision(
            self.job.job_id,
            1,
            parameters=self.document,
            frame_path=str(self.frame),
            video_path=str(self.video),
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def controller(self, *, auto: bool = False) -> Phase1QcController:
        settings = replace(
            QualityControlSettings(),
            quality_control_enabled=True,
            auto_advance_pass=auto,
        )
        return Phase1QcController(
            store=self.store,
            layout=self.layout,
            settings=settings,
            backend_factory=lambda: None,
            prompt_root=Path(__file__).parents[1] / "prompts",
        )

    def _epoch_controller(self, events: list[str], *, auto: bool):
        identity = BackendIdentity(
            evaluator_id="production-qc",
            evaluator_version="phase1-v1",
            backend_family="llama.cpp",
            backend_version="llama.cpp 2.28.2",
            executable_path="C:/llama-server.exe",
            executable_sha256="1" * 64,
            model_path="C:/model.gguf",
            model_sha256="2" * 64,
            model_id="Qwen3.6 27B",
            quantization="IQ3_M GGUF",
            projector_path="C:/mmproj.gguf",
            projector_sha256="3" * 64,
            projector_precision="FP16",
            gpu_uuid="GPU-stable",
            gpu_name="NVIDIA GeForce RTX 4080 SUPER",
            effective_args=("--host", "127.0.0.1"),
            effective_config_sha256="4" * 64,
            owned_pid=123,
            stdout_log_path="stdout.log",
            stderr_log_path="stderr.log",
        )

        class Backend:
            def start(self):
                events.append("qwen-start")
                return identity

            def evaluate(self, request):
                events.append("judge")
                return VisionJudgeEvaluation(
                    parse_judge_response(
                        '{"decision":"PASS","confidence":0.99,"summary":"clean","errors":[]}'
                    )
                )

            def close(self):
                events.append("qwen-close")

        sampled = SampledVideo(
            VideoMetadata(24.0, 4, 1.0),
            2.0,
            tuple(
                SampledFrame(index, index / 2, self.root / f"{index}.jpg", b"jpeg")
                for index in range(4)
            ),
        )
        settings = replace(
            QualityControlSettings(),
            quality_control_enabled=True,
            auto_advance_pass=auto,
        )
        return Phase1QcController(
            store=self.store,
            layout=self.layout,
            settings=settings,
            backend_factory=Backend,
            prompt_root=Path(__file__).parents[1] / "prompts",
            sample_video=lambda *args, **kwargs: sampled,
        )

    def _candidate(self):
        return self.controller().register_original_candidates(self.job)[0]

    def _evaluation(self, candidate_id: str, decision: QcDecision):
        candidate = self.store.qc_candidate(candidate_id)
        identity = {
            "evaluator_id": "production-qc",
            "evaluator_version": "phase1-v1",
            "backend_family": "llama.cpp",
            "backend_version": "2.28.2",
            "executable_path": "C:/llama-server.exe",
            "executable_sha256": "1" * 64,
            "model_id": "Qwen3.6 27B",
            "model_path": "C:/model.gguf",
            "model_sha256": "2" * 64,
            "quantization": "IQ3_M GGUF",
            "projector_path": "C:/mmproj.gguf",
            "projector_sha256": "3" * 64,
            "projector_precision": "FP16",
            "gpu_uuid": "GPU-stable",
            "gpu_name": "NVIDIA GeForce RTX 4080 SUPER",
        }
        key = evaluation_idempotency_key(
            source_video_sha256=candidate.source_video_sha256,
            evaluator_id=identity["evaluator_id"],
            evaluator_version=identity["evaluator_version"],
            backend_version=identity["backend_version"],
            executable_sha256=identity["executable_sha256"],
            model_sha256=identity["model_sha256"],
            projector_sha256=identity["projector_sha256"],
            effective_config_sha256="4" * 64,
            prompt_sha256="5" * 64,
        )
        evaluation = self.store.begin_qc_evaluation(
            evaluation_id=(
                "evaluation-"
                + decision.value.lower()
                + "-"
                + candidate_id[-8:]
            ),
            idempotency_key=key,
            candidate_id=candidate_id,
            source_video_path=candidate.source_video_path,
            source_video_sha256=candidate.source_video_sha256,
            evaluator_identity=identity,
            effective_config={},
            effective_config_sha256="4" * 64,
            prompt_version="production_ltx_video_qc_v1",
            prompt_sha256="5" * 64,
            sampling_config={"fps": 2.0},
            window_config={"frames": 4},
        )
        return self.store.complete_qc_evaluation(
            evaluation.evaluation_id,
            raw_result="raw",
            normalized_decision=decision,
            suspect_windows=[],
            strong_window_count=0,
            frame_accounting={},
            evidence_manifest_path=str(
                Path(candidate.source_video_path).parent
                / "qc"
                / "evaluations"
                / evaluation.evaluation_id
                / "result.json"
            ),
            evidence_manifest_sha256="6" * 64,
            next_action="route",
        )

    def test_pass_requires_human_by_default_but_auto_policy_accepts(self) -> None:
        candidate = self._candidate()
        self._evaluation(candidate.candidate_id, QcDecision.PASS)
        routed = self.controller().route_completed_evaluation(
            self.job, candidate.candidate_id
        )
        self.assertEqual(routed.state, QcCandidateState.PASS_PENDING_HUMAN)

        # Policy change is intentionally evaluated from durable evidence.
        routed = self.controller(auto=True).route_completed_evaluation(
            self.job, candidate.candidate_id
        )
        self.assertEqual(routed.state, QcCandidateState.ACCEPTED)

    def test_uncertain_holds_without_creating_retry(self) -> None:
        candidate = self._candidate()
        self._evaluation(candidate.candidate_id, QcDecision.UNCERTAIN)
        routed = self.controller().route_completed_evaluation(
            self.job, candidate.candidate_id
        )
        self.assertEqual(routed.state, QcCandidateState.HOLD_FOR_REVIEW)
        self.assertEqual(len(self.store.qc_candidates(self.job.job_id)), 1)

    def test_original_fail_creates_exactly_one_durable_a1(self) -> None:
        candidate = self._candidate()
        self._evaluation(candidate.candidate_id, QcDecision.FAIL)
        first = self.controller().route_completed_evaluation(
            self.job, candidate.candidate_id
        )
        second = self.controller().route_completed_evaluation(
            self.job, candidate.candidate_id
        )
        candidates = self.store.qc_candidates(self.job.job_id)
        self.assertEqual(first.tier, QcTier.A1)
        self.assertEqual(second.candidate_id, first.candidate_id)
        self.assertEqual([item.tier for item in candidates], [QcTier.ORIGINAL, QcTier.A1])
        self.assertEqual(first.state, QcCandidateState.PENDING_GENERATION)

    def test_b1_never_loops_after_fail(self) -> None:
        candidate = self._candidate()
        self.store.set_qc_candidate_state(
            candidate.candidate_id,
            QcCandidateState.HOLD_FOR_REVIEW,
            next_action="test",
        )
        # A B1 terminal result cannot be transformed into another retry tier.
        with self.store._connection() as connection:
            connection.execute(
                "UPDATE qc_candidates SET tier = ? WHERE candidate_id = ?",
                (QcTier.B1.value, candidate.candidate_id),
            )
        self._evaluation(candidate.candidate_id, QcDecision.FAIL)
        routed = self.controller().route_completed_evaluation(
            self.job, candidate.candidate_id
        )
        self.assertEqual(routed.state, QcCandidateState.HOLD_FOR_REVIEW)
        self.assertEqual(len(self.store.qc_candidates(self.job.job_id)), 1)

    def test_restart_reconstructs_b1_after_repair_persisted_before_candidate(self) -> None:
        original = self._candidate()
        self._evaluation(original.candidate_id, QcDecision.FAIL)
        a1 = self.controller().route_completed_evaluation(
            self.job, original.candidate_id
        )
        a1_video = Path(a1.source_video_path)
        a1_video.parent.mkdir(parents=True, exist_ok=True)
        a1_video.write_bytes(b"a1-non-production-video")
        a1 = self.store.complete_qc_candidate_generation(
            a1.candidate_id,
            source_video_path=str(a1_video),
            source_video_sha256=hashlib.sha256(a1_video.read_bytes()).hexdigest(),
        )
        self._evaluation(a1.candidate_id, QcDecision.FAIL)

        class Planner:
            def plan_repair(_self, request):
                source = request.source_identity
                return RepairPlannerResponse(
                    raw_text=(
                        '{"schema_version":1,"source":{'
                        f'"candidate_id":"{source["candidate_id"]}",'
                        f'"candidate_sha256":"{source["candidate_sha256"]}",'
                        f'"evaluation_id":"{source["evaluation_id"]}",'
                        f'"repair_input_sha256":"{request.repair_input_sha256}"'
                        f',"source_revision":{source["source_revision"]}'
                        f',"source_document_sha256":"{source["source_document_sha256"]}"'
                        '},"patch":{"i2v":{"prompt":"movement prompt with steadier hand motion"}},'
                        '"summary":"stabilize the visible hand motion"}'
                    )
                )

        controller = self.controller()
        with patch.object(
            self.store,
            "ensure_b1_candidate_revision",
            side_effect=RuntimeError("simulated crash after repair persistence"),
        ):
            held = controller.route_completed_evaluation(
                self.job,
                a1.candidate_id,
                backend=Planner(),
                planner_identity={"backend": "fake"},
            )
        self.assertEqual(held.state, QcCandidateState.HOLD_FOR_REVIEW)
        self.assertEqual(self.store.qc_repairs(a1.candidate_id)[0].status, "ACCEPTED")
        self.assertFalse(
            any(
                item.tier == QcTier.B1
                for item in self.store.qc_candidates(self.job.job_id)
            )
        )

        recovered = self.controller().route_completed_evaluation(
            self.job, a1.candidate_id
        )

        self.assertEqual(recovered.tier, QcTier.B1)
        self.assertEqual(recovered.state, QcCandidateState.PENDING_GENERATION)
        self.assertEqual(
            self.store.qc_candidate(a1.candidate_id).state,
            QcCandidateState.SUPERSEDED,
        )

    def test_restart_never_redispatches_ambiguous_b1_planner_claim(self) -> None:
        original = self._candidate()
        self._evaluation(original.candidate_id, QcDecision.FAIL)
        a1 = self.controller().route_completed_evaluation(
            self.job, original.candidate_id
        )
        a1_video = Path(a1.source_video_path)
        a1_video.parent.mkdir(parents=True, exist_ok=True)
        a1_video.write_bytes(b"a1-non-production-video")
        self.store.complete_qc_candidate_generation(
            a1.candidate_id,
            source_video_path=str(a1_video),
            source_video_sha256=hashlib.sha256(a1_video.read_bytes()).hexdigest(),
        )
        self._evaluation(a1.candidate_id, QcDecision.FAIL)
        calls = 0

        class CrashingPlanner:
            def plan_repair(_self, _request):
                nonlocal calls
                calls += 1
                raise KeyboardInterrupt("simulated process death after dispatch")

        with self.assertRaises(KeyboardInterrupt):
            self.controller().route_completed_evaluation(
                self.job,
                a1.candidate_id,
                backend=CrashingPlanner(),
                planner_identity={"backend": "fake", "owned_pid": 999},
            )
        claim = self.store.qc_repair_planner_claim(a1.candidate_id)
        self.assertIsNotNone(claim)
        self.assertEqual(claim.state, "CLAIMED")

        held = self.controller().route_completed_evaluation(
            self.job,
            a1.candidate_id,
            backend=CrashingPlanner(),
            planner_identity={"backend": "fake"},
        )

        self.assertEqual(calls, 1)
        self.assertEqual(held.state, QcCandidateState.HOLD_FOR_REVIEW)
        repair = self.store.qc_repairs(a1.candidate_id)[0]
        self.assertEqual(repair.status, "REJECTED")
        self.assertEqual(
            repair.reason, "planner_invocation_ambiguous_after_restart"
        )
        self.assertEqual(
            self.store.qc_repair_planner_claim(a1.candidate_id).state,
            "FAILED_CLOSED",
        )

    def test_isolated_end_to_end_epoch_releases_generation_before_qwen_and_closes(self) -> None:
        events: list[str] = []

        class Comfy:
            def queue_counts(self):
                events.append("queue-idle")
                return 0, 0

            def free_memory(self):
                events.append("comfy-free")

        supervisor = type(
            "Supervisor",
            (),
            {"comfy": Comfy(), "release_memory": lambda self: None},
        )()
        result = self._epoch_controller(events, auto=False).run_epoch(
            self.job, supervisor
        )

        self.assertFalse(result.ready_for_finalization)
        self.assertTrue(result.waiting_for_human)
        self.assertEqual(
            self.store.qc_candidates(self.job.job_id)[0].state,
            QcCandidateState.PASS_PENDING_HUMAN,
        )
        self.assertEqual(
            events,
            ["queue-idle", "comfy-free", "queue-idle", "qwen-start", "judge", "qwen-close"],
        )

    def test_auto_pass_epoch_resumes_finalization_only_after_durable_acceptance(self) -> None:
        events: list[str] = []

        class Comfy:
            def queue_counts(self):
                return 0, 0

            def free_memory(self):
                pass

        supervisor = type(
            "Supervisor",
            (),
            {"comfy": Comfy(), "release_memory": lambda self: None},
        )()
        result = self._epoch_controller(events, auto=True).run_epoch(
            self.job, supervisor
        )

        self.assertTrue(result.ready_for_finalization)
        self.assertEqual(result.selection[0].revision, 1)
        self.assertEqual(
            self.store.qc_candidates(self.job.job_id)[0].state,
            QcCandidateState.ACCEPTED,
        )
        self.assertEqual(events[-1], "qwen-close")

    def test_backend_factory_failure_uses_bounded_infrastructure_budget(self) -> None:
        candidate = self._candidate()
        controller = self.controller()
        controller.backend_factory = lambda: (_ for _ in ()).throw(
            RuntimeError("factory failed")
        )

        class Comfy:
            def queue_counts(self):
                return 0, 0

            def free_memory(self):
                pass

        supervisor = type(
            "Supervisor",
            (),
            {"comfy": Comfy(), "release_memory": lambda self: None},
        )()

        result = controller.run_epoch(self.job, supervisor)

        persisted = self.store.qc_candidate(candidate.candidate_id)
        self.assertFalse(result.ready_for_finalization)
        self.assertEqual(persisted.infrastructure_failure_count, 2)
        self.assertEqual(persisted.state, QcCandidateState.HOLD_FOR_REVIEW)

    def test_failed_prior_backend_cleanup_blocks_repair_generation(self) -> None:
        original = self._candidate()
        self._evaluation(original.candidate_id, QcDecision.FAIL)
        self.controller().route_completed_evaluation(self.job, original.candidate_id)
        events: list[str] = []

        class StuckBackend:
            def close(self):
                events.append("close")
                raise RuntimeError("port remains open")

        class Supervisor:
            comfy = object()

            def render_qc_candidates(self, *_args):
                events.append("render")

            def release_memory(self):
                events.append("free")

        controller = self.controller()
        controller._active_backend = StuckBackend()

        with self.assertRaisesRegex(RuntimeError, "port remains open"):
            controller.run_epoch(self.job, Supervisor())

        self.assertEqual(events, ["close"])
        self.assertIsNotNone(controller._active_backend)

    def test_unknown_qc_port_owner_blocks_repair_generation_after_restart(self) -> None:
        original = self._candidate()
        self._evaluation(original.candidate_id, QcDecision.FAIL)
        a1 = self.controller().route_completed_evaluation(
            self.job, original.candidate_id
        )
        events: list[str] = []
        controller = Phase1QcController(
            store=self.store,
            layout=self.layout,
            settings=replace(
                QualityControlSettings(), quality_control_enabled=True
            ),
            backend_factory=lambda: None,
            prompt_root=Path(__file__).parents[1] / "prompts",
            qc_port_open=lambda: True,
        )
        supervisor = type(
            "Supervisor",
            (),
            {
                "render_qc_candidates": lambda *_args: events.append("render"),
                "release_memory": lambda *_args: events.append("free"),
            },
        )()

        with self.assertRaisesRegex(QcControllerError, "unverified process"):
            controller.run_epoch(self.job, supervisor)

        self.assertEqual(events, [])
        self.assertEqual(
            self.store.qc_candidate(a1.candidate_id).state,
            QcCandidateState.PENDING_GENERATION,
        )

    def test_end_to_end_original_fail_a1_fail_b1_pass_is_restart_safe(self) -> None:
        events: list[str] = []
        responses = iter(
            [
                # ORIGINAL: two independently strong FAIL windows.
                '{"decision":"FAIL","confidence":0.95,"summary":"fusion",'
                '"errors":[{"category":"topology","severity":4,"confidence":0.95,'
                '"start_time_seconds":0.0,"end_time_seconds":1.5,'
                '"description":"visible hand fusion","evidence":"hand boundaries merge"}]}',
                '{"decision":"FAIL","confidence":0.95,"summary":"fusion persists",'
                '"errors":[{"category":"topology","severity":4,"confidence":0.95,'
                '"start_time_seconds":2.0,"end_time_seconds":3.5,'
                '"description":"visible hand fusion","evidence":"hand boundaries merge again"}]}',
                # A1: same bounded failure.
                '{"decision":"FAIL","confidence":0.95,"summary":"fusion",'
                '"errors":[{"category":"topology","severity":4,"confidence":0.95,'
                '"start_time_seconds":0.0,"end_time_seconds":1.5,'
                '"description":"visible hand fusion","evidence":"hand boundaries merge"}]}',
                '{"decision":"FAIL","confidence":0.95,"summary":"fusion persists",'
                '"errors":[{"category":"topology","severity":4,"confidence":0.95,'
                '"start_time_seconds":2.0,"end_time_seconds":3.5,'
                '"description":"visible hand fusion","evidence":"hand boundaries merge again"}]}',
                # B1: all windows pass.
                '{"decision":"PASS","confidence":0.99,"summary":"clean","errors":[]}',
                '{"decision":"PASS","confidence":0.99,"summary":"clean","errors":[]}',
            ]
        )
        identity = BackendIdentity(
            evaluator_id="production-qc",
            evaluator_version="phase1-v1",
            backend_family="llama.cpp",
            backend_version="llama.cpp 2.28.2",
            executable_path="C:/llama-server.exe",
            executable_sha256="1" * 64,
            model_path="C:/model.gguf",
            model_sha256="2" * 64,
            model_id="Qwen3.6 27B",
            quantization="IQ3_M GGUF",
            projector_path="C:/mmproj.gguf",
            projector_sha256="3" * 64,
            projector_precision="FP16",
            gpu_uuid="GPU-stable",
            gpu_name="NVIDIA GeForce RTX 4080 SUPER",
            effective_args=("--host", "127.0.0.1"),
            effective_config_sha256="4" * 64,
            owned_pid=123,
            stdout_log_path="stdout.log",
            stderr_log_path="stderr.log",
        )

        class Backend:
            def start(self):
                events.append("qwen-start")
                return identity

            def evaluate(self, _request):
                events.append("judge")
                return VisionJudgeEvaluation(parse_judge_response(next(responses)))

            def plan_repair(self, request):
                events.append("planner")
                source = request.source_identity
                return RepairPlannerResponse(
                    raw_text=(
                        '{"schema_version":1,"source":{'
                        f'"candidate_id":"{source["candidate_id"]}",'
                        f'"candidate_sha256":"{source["candidate_sha256"]}",'
                        f'"evaluation_id":"{source["evaluation_id"]}",'
                        f'"repair_input_sha256":"{request.repair_input_sha256}"'
                        f',"source_revision":{source["source_revision"]}'
                        f',"source_document_sha256":"{source["source_document_sha256"]}"'
                        '},"patch":{"i2v":{"prompt":"movement prompt with stable separated hand motion"}},'
                        '"summary":"stabilize the concrete hand defect"}'
                    )
                )

            def close(self):
                events.append("qwen-close")

        sampled = SampledVideo(
            VideoMetadata(24.0, 8, 4.0),
            2.0,
            tuple(
                SampledFrame(index, index / 2, self.root / f"e2e-{index}.jpg", b"jpeg")
                for index in range(8)
            ),
        )
        controller = Phase1QcController(
            store=self.store,
            layout=self.layout,
            settings=replace(
                QualityControlSettings(), quality_control_enabled=True
            ),
            backend_factory=Backend,
            prompt_root=Path(__file__).parents[1] / "prompts",
            sample_video=lambda *args, **kwargs: sampled,
        )

        class Comfy:
            def queue_counts(self):
                events.append("queue-idle")
                return 0, 0

            def free_memory(self):
                events.append("comfy-free")

        class Supervisor:
            comfy = Comfy()

            def render_qc_candidates(_self, _job, candidates):
                events.append("render-batch")
                for candidate, _document in candidates:
                    destination = Path(candidate.source_video_path)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(candidate.tier.value.encode())
                    self.store.complete_qc_candidate_generation(
                        candidate.candidate_id,
                        source_video_path=str(destination),
                        source_video_sha256=hashlib.sha256(
                            destination.read_bytes()
                        ).hexdigest(),
                    )

            def release_memory(_self):
                events.append("generation-free")

        result = controller.run_epoch(self.job, Supervisor())

        self.assertFalse(result.ready_for_finalization)
        b1 = next(
            item
            for item in self.store.qc_candidates(self.job.job_id)
            if item.tier == QcTier.B1
        )
        self.assertEqual(b1.state, QcCandidateState.PASS_PENDING_HUMAN)
        self.assertEqual(len(self.store.qc_repairs(b1.parent_candidate_id)), 1)
        close_positions = [
            index for index, event in enumerate(events) if event == "qwen-close"
        ]
        render_positions = [
            index for index, event in enumerate(events) if event == "render-batch"
        ]
        self.assertLess(close_positions[0], render_positions[0])
        self.assertLess(close_positions[1], render_positions[1])

        restarted_store = PipelineStateStore(self.layout.database_path)
        restarted_store.decide_qc_candidate(
            job_id=self.job.job_id,
            scene_id=1,
            candidate_id=b1.candidate_id,
            decision=QcHumanDecision.APPROVE,
            note=None,
        )
        restarted = Phase1QcController(
            store=restarted_store,
            layout=self.layout,
            settings=replace(
                QualityControlSettings(), quality_control_enabled=True
            ),
            backend_factory=lambda: None,
            prompt_root=Path(__file__).parents[1] / "prompts",
        )
        final = restarted.run_epoch(self.job, Supervisor())
        self.assertTrue(final.ready_for_finalization)
        self.assertEqual(final.selection[0].revision, b1.revision)


if __name__ == "__main__":
    unittest.main()
