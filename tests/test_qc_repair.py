from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from tenminvideomaker.contracts import parse_job_payload
from tenminvideomaker.qc_contracts import (
    QcCandidateState,
    QcDecision,
    QcTier,
    canonical_json,
    derive_retry_seed,
    evaluation_idempotency_key,
)
from tenminvideomaker.qc_repair import (
    B1PatchValidationError,
    apply_b1_patch,
    build_a1_document,
    parse_and_validate_b1_patch,
    schedule_a1_retry,
    schedule_b1_retry,
)
from tenminvideomaker.review import scene_review_document
from tenminvideomaker.state_store import (
    PipelineStateStore,
    RemakeMode,
    SceneState,
    StateTransitionError,
)
from tenminvideomaker.storage import StorageLayout
from test_contracts import payload


class A1RetryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        self.layout = StorageLayout(root / "storage")
        self.store = PipelineStateStore(self.layout.database_path)
        self.job = parse_job_payload(payload())
        self.scene = self.job.scenes[0]
        self.document = scene_review_document(self.job, self.scene)
        self.store.claim_job(self.job)
        frame = self.layout.scene_frame_path(self.job.job_id, 1, 1)
        video = self.layout.scene_clip_path(self.job.job_id, 1, 1)
        frame.parent.mkdir(parents=True, exist_ok=True)
        frame.write_bytes(b"frame")
        video.write_bytes(b"video")
        self.store.set_scene_state(
            self.job.job_id,
            1,
            SceneState.SUCCEEDED,
            frame_path=str(frame),
            video_path=str(video),
        )
        self.store.ensure_original_scene_revision(
            self.job.job_id,
            1,
            parameters=self.document,
            frame_path=str(frame),
            video_path=str(video),
        )
        self.original = self.store.ensure_qc_candidate(
            candidate_id="candidate-original",
            job_id=self.job.job_id,
            scene_id=1,
            revision=1,
            tier=QcTier.ORIGINAL,
            parent_candidate_id=None,
            source_video_path=str(video),
            source_video_sha256=hashlib.sha256(b"video").hexdigest(),
            original_prompt=self.document["i2v"]["prompt"],
            current_prompt=self.document["i2v"]["prompt"],
            original_seed=int(self.document["i2v"]["seed"]),
            current_seed=int(self.document["i2v"]["seed"]),
            negative_prompt=self.document["i2v"]["negative"],
            negative_prompt_sha256=hashlib.sha256(
                self.document["i2v"]["negative"].encode("utf-8")
            ).hexdigest(),
            state=QcCandidateState.PENDING_QC,
            next_action="create_a1",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_a1_document_changes_only_seed_and_preserves_prompt_exactly(self) -> None:
        seed = 123456789
        revised = build_a1_document(self.document, seed=seed)

        self.assertEqual(revised["i2v"]["prompt"], self.document["i2v"]["prompt"])
        self.assertIsNot(revised, self.document)
        expected = deepcopy(self.document)
        expected["i2v"]["seed"] = str(seed)
        self.assertEqual(revised, expected)

    def test_a1_creates_one_video_only_revision_and_pending_candidate(self) -> None:
        retry = schedule_a1_retry(
            self.store,
            self.layout,
            original_job=self.job,
            source_candidate_id=self.original.candidate_id,
            source_document=self.document,
        )

        expected_seed = derive_retry_seed(
            job_id=self.job.job_id,
            scene_id=1,
            source_revision=1,
            original_seed=self.original.original_seed,
            tier=QcTier.A1,
            used_seeds=(self.original.current_seed,),
        )
        self.assertEqual(retry.seed, expected_seed)
        self.assertNotEqual(retry.seed, self.original.current_seed)
        self.assertEqual(retry.candidate.tier, QcTier.A1)
        self.assertEqual(retry.candidate.parent_candidate_id, self.original.candidate_id)
        self.assertEqual(retry.candidate.revision, 2)
        self.assertEqual(retry.candidate.state, QcCandidateState.PENDING_GENERATION)
        self.assertIsNone(retry.candidate.source_video_sha256)
        self.assertEqual(
            retry.candidate.source_video_path,
            str(self.layout.scene_clip_path(self.job.job_id, 1, 2)),
        )
        revisions = self.store.scene_revisions(self.job.job_id, 1)
        self.assertEqual(tuple(item.revision for item in revisions), (2, 1))
        self.assertEqual(revisions[0].remake_mode.value, "video_only")
        self.assertEqual(revisions[0].frame_path, revisions[1].frame_path)
        self.assertEqual(revisions[0].parameters, retry.document)

    def test_duplicate_a1_request_is_idempotent_and_does_not_create_retry_loop(self) -> None:
        first = schedule_a1_retry(
            self.store,
            self.layout,
            original_job=self.job,
            source_candidate_id=self.original.candidate_id,
            source_document=self.document,
        )
        second = schedule_a1_retry(
            PipelineStateStore(self.layout.database_path),
            self.layout,
            original_job=self.job,
            source_candidate_id=self.original.candidate_id,
            source_document=self.document,
        )

        self.assertEqual(second, first)
        self.assertEqual(
            len(self.store.qc_candidates(self.job.job_id, scene_id=1)), 2
        )
        self.assertEqual(len(self.store.scene_revisions(self.job.job_id, 1)), 2)

    def test_a1_seed_excludes_every_seed_in_revision_lineage(self) -> None:
        first_seed = derive_retry_seed(
            job_id=self.job.job_id,
            scene_id=1,
            source_revision=1,
            original_seed=self.original.original_seed,
            tier=QcTier.A1,
            used_seeds=(self.original.current_seed,),
        )
        intervening = deepcopy(self.document)
        intervening["i2v"]["seed"] = str(first_seed)
        self.store.create_scene_revision(
            self.job.job_id,
            1,
            remake_mode=RemakeMode.VIDEO_ONLY,
            parameters=intervening,
        )

        retry = schedule_a1_retry(
            self.store,
            self.layout,
            original_job=self.job,
            source_candidate_id=self.original.candidate_id,
            source_document=self.document,
        )

        self.assertNotEqual(retry.seed, first_seed)
        self.assertEqual(retry.candidate.revision, 3)

    def test_generation_completion_adds_hash_without_overwriting_original(self) -> None:
        retry = schedule_a1_retry(
            self.store,
            self.layout,
            original_job=self.job,
            source_candidate_id=self.original.candidate_id,
            source_document=self.document,
        )
        Path(retry.candidate.source_video_path).parent.mkdir(parents=True, exist_ok=True)
        Path(retry.candidate.source_video_path).write_bytes(b"a1-video")

        a1_hash = hashlib.sha256(b"a1-video").hexdigest()
        completed = self.store.complete_qc_candidate_generation(
            retry.candidate.candidate_id,
            source_video_path=retry.candidate.source_video_path,
            source_video_sha256=a1_hash,
        )

        self.assertEqual(completed.state, QcCandidateState.PENDING_QC)
        self.assertEqual(completed.source_video_sha256, a1_hash)
        self.assertEqual(
            self.original.source_video_sha256,
            hashlib.sha256(b"video").hexdigest(),
        )
        with self.assertRaises(StateTransitionError):
            self.store.complete_qc_candidate_generation(
                retry.candidate.candidate_id,
                source_video_path=retry.candidate.source_video_path,
                source_video_sha256="d" * 64,
            )

    def test_changed_baseline_starting_frame_blocks_a1_without_candidate(self) -> None:
        identity = self.store.qc_candidate_source_identity(
            self.original.candidate_id
        )
        Path(identity["source_frame_path"]).write_bytes(b"mutated-frame")

        with self.assertRaisesRegex(StateTransitionError, "starting frame changed"):
            schedule_a1_retry(
                self.store,
                self.layout,
                original_job=self.job,
                source_candidate_id=self.original.candidate_id,
                source_document=self.document,
            )

        self.assertFalse(
            any(
                item.tier == QcTier.A1
                for item in self.store.qc_candidates(self.job.job_id, 1)
            )
        )

    def test_changed_baseline_revision_document_blocks_a1(self) -> None:
        mutated = deepcopy(self.document)
        mutated["t2i"]["prompt"] += " out-of-band mutation"
        connection = sqlite3.connect(self.layout.database_path)
        try:
            connection.execute(
                """
                UPDATE scene_revisions SET parameters_json = ?
                WHERE job_id = ? AND scene_id = 1 AND revision = 1
                """,
                (canonical_json(mutated), self.job.job_id),
            )
            connection.commit()
        finally:
            connection.close()

        with self.assertRaisesRegex(StateTransitionError, "document changed"):
            schedule_a1_retry(
                self.store,
                self.layout,
                original_job=self.job,
                source_candidate_id=self.original.candidate_id,
                source_document=mutated,
            )

    def test_changed_baseline_video_bytes_still_block_a1(self) -> None:
        Path(self.original.source_video_path).write_bytes(b"mutated-video")

        with self.assertRaisesRegex(StateTransitionError, "source video changed"):
            schedule_a1_retry(
                self.store,
                self.layout,
                original_job=self.job,
                source_candidate_id=self.original.candidate_id,
                source_document=self.document,
            )


class B1PatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.job = parse_job_payload(payload())
        self.document = scene_review_document(self.job, self.job.scenes[0])
        self.seed = 987654321
        self.input_hash = "1" * 64
        self.candidate_hash = "2" * 64
        self.candidate_id = "candidate-a1"
        self.evaluation_id = "evaluation-a1"
        self.source_revision = 2
        self.source_document_sha256 = hashlib.sha256(
            canonical_json(self.document).encode("utf-8")
        ).hexdigest()

    def raw(self, prompt: str | None = None, **extra: object) -> str:
        value = {
            "schema_version": 1,
            "source": {
                "candidate_id": self.candidate_id,
                "candidate_sha256": self.candidate_hash,
                "evaluation_id": self.evaluation_id,
                "repair_input_sha256": self.input_hash,
                "source_revision": self.source_revision,
                "source_document_sha256": self.source_document_sha256,
            },
            "patch": {"i2v": {"prompt": (
                "A smaller, defect-focused motion instruction."
                if prompt is None
                else prompt
            )}},
            "summary": "Tighten the visible hand transition without adding action.",
        }
        value.update(extra)
        return json.dumps(value)

    def parse(self, raw_text: str, **kwargs: object):
        return parse_and_validate_b1_patch(
            raw_text,
            source_document=self.document,
            required_seed=self.seed,
            repair_input_hash=self.input_hash,
            current_candidate_hash=self.candidate_hash,
            current_candidate_id=self.candidate_id,
            evaluation_id=self.evaluation_id,
            source_revision=self.source_revision,
            source_document_sha256=self.source_document_sha256,
            **kwargs,
        )

    def test_valid_patch_changes_only_i2v_prompt_and_controller_seed(self) -> None:
        patch = self.parse(self.raw())
        result = apply_b1_patch(self.job, 1, self.document, patch)

        expected = deepcopy(self.document)
        expected["i2v"]["prompt"] = patch.prompt
        expected["i2v"]["seed"] = str(self.seed)
        self.assertEqual(result.document, expected)
        self.assertEqual(result.document["i2v"]["negative"], self.document["i2v"]["negative"])
        self.assertEqual(result.document["t2i"], self.document["t2i"])
        self.assertEqual(result.document["character"], self.document["character"])
        self.assertEqual(result.document["scene_context"], self.document["scene_context"])
        self.assertEqual(result.document["job_context"], self.document["job_context"])

    def test_seed_or_unknown_model_mutation_is_rejected(self) -> None:
        value = json.loads(self.raw())
        value["patch"]["i2v"]["seed"] = 123
        with self.assertRaises(B1PatchValidationError):
            self.parse(json.dumps(value))
        value = json.loads(self.raw())
        value["patch"]["negative"] = "changed"
        with self.assertRaises(B1PatchValidationError):
            self.parse(json.dumps(value))

    def test_stale_source_identity_or_hash_is_rejected(self) -> None:
        value = json.loads(self.raw())
        value["source"]["candidate_sha256"] = "3" * 64
        with self.assertRaises(B1PatchValidationError):
            self.parse(json.dumps(value))
        value = json.loads(self.raw())
        value["source"]["candidate_id"] = "candidate-old"
        with self.assertRaises(B1PatchValidationError):
            self.parse(json.dumps(value))
        value = json.loads(self.raw())
        value["source"]["source_revision"] = self.source_revision + 1
        with self.assertRaises(B1PatchValidationError):
            self.parse(json.dumps(value))
        value = json.loads(self.raw())
        value["source"]["source_document_sha256"] = "4" * 64
        with self.assertRaises(B1PatchValidationError):
            self.parse(json.dumps(value))

    def test_malformed_refusal_blank_and_unchanged_are_rejected(self) -> None:
        for raw_text in (
            "not json",
            "I cannot help with that",
            self.raw("   "),
            self.raw(self.document["i2v"]["prompt"]),
        ):
            with self.subTest(raw_text=raw_text[:20]):
                with self.assertRaises(B1PatchValidationError):
                    self.parse(raw_text)

    def test_duplicate_prompt_seed_config_attempt_is_rejected(self) -> None:
        prompt = "A smaller, defect-focused motion instruction."
        with self.assertRaises(B1PatchValidationError):
            self.parse(
                self.raw(prompt),
                prior_attempts=((prompt, self.seed, "config-v1"),),
                generation_config_hash="config-v1",
            )

    def test_required_fixed_prompt_content_cannot_be_lost(self) -> None:
        with self.assertRaises(B1PatchValidationError):
            self.parse(
                self.raw("Only the camera moves gently."),
                required_prompt_fragments=("keeps holding the red umbrella",),
            )

    def test_explicit_segment_prompt_scene_fails_closed_as_b1_inapplicable(self) -> None:
        self.document["i2v"]["segments"] = [
            {"start_seconds": 0.0, "end_seconds": 1.0, "positive_prompt": "locked"}
        ]
        with self.assertRaises(B1PatchValidationError):
            self.parse(self.raw())


class B1DurabilityTests(A1RetryTests):
    def _completed_a1_failure(self):
        retry = schedule_a1_retry(
            self.store,
            self.layout,
            original_job=self.job,
            source_candidate_id=self.original.candidate_id,
            source_document=self.document,
        )
        path = Path(retry.candidate.source_video_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"a1-video")
        a1 = self.store.complete_qc_candidate_generation(
            retry.candidate.candidate_id,
            source_video_path=str(path),
            source_video_sha256=hashlib.sha256(b"a1-video").hexdigest(),
        )
        identity = {
            "evaluator_id": "production-qc",
            "evaluator_version": "phase1-v1",
            "backend_family": "llama.cpp",
            "backend_version": "2.28.2",
            "executable_path": "C:/llama.exe",
            "executable_sha256": "1" * 64,
            "model_id": "Qwen",
            "model_path": "C:/model.gguf",
            "model_sha256": "2" * 64,
            "quantization": "IQ3_M",
            "projector_path": "C:/mmproj.gguf",
            "projector_sha256": "3" * 64,
            "projector_precision": "FP16",
            "gpu_uuid": "GPU-test",
            "gpu_name": "RTX 4080 SUPER",
        }
        key = evaluation_idempotency_key(
            source_video_sha256=a1.source_video_sha256,
            evaluator_id=identity["evaluator_id"],
            evaluator_version=identity["evaluator_version"],
            backend_version=identity["backend_version"],
            executable_sha256=identity["executable_sha256"],
            model_sha256=identity["model_sha256"],
            projector_sha256=identity["projector_sha256"],
            effective_config_sha256="4" * 64,
            prompt_sha256="5" * 64,
        )
        self.store.begin_qc_evaluation(
            evaluation_id="evaluation-a1",
            idempotency_key=key,
            candidate_id=a1.candidate_id,
            source_video_path=a1.source_video_path,
            source_video_sha256=a1.source_video_sha256,
            evaluator_identity=identity,
            effective_config={},
            effective_config_sha256="4" * 64,
            prompt_version="v1",
            prompt_sha256="5" * 64,
            sampling_config={},
            window_config={},
        )
        manifest = self.layout.qc_evaluation_manifest_path(
            self.job.job_id, 1, a1.revision, "evaluation-a1"
        )
        self.store.complete_qc_evaluation(
            "evaluation-a1",
            raw_result="fail",
            normalized_decision=QcDecision.FAIL,
            suspect_windows=[{"start": 1.0, "end": 2.0}],
            strong_window_count=2,
            frame_accounting={},
            evidence_manifest_path=str(manifest),
            evidence_manifest_sha256="6" * 64,
            next_action="plan_b1",
        )
        revision = next(
            item for item in self.store.scene_revisions(self.job.job_id, 1)
            if item.revision == a1.revision
        )
        return a1, revision.parameters

    def test_b1_plan_and_revision_are_durable_exactly_once(self) -> None:
        a1, document = self._completed_a1_failure()
        input_hash = "7" * 64
        raw_output = json.dumps({
            "schema_version": 1,
            "source": {
                "candidate_id": a1.candidate_id,
                "candidate_sha256": a1.source_video_sha256,
                "evaluation_id": "evaluation-a1",
                "repair_input_sha256": input_hash,
                "source_revision": a1.revision,
                "source_document_sha256": hashlib.sha256(
                    canonical_json(document).encode("utf-8")
                ).hexdigest(),
            },
            "patch": {"i2v": {"prompt": "A minimal corrected hand transition."}},
            "summary": "Correct the visible hand transition.",
        })
        kwargs = dict(
            original_job=self.job,
            source_candidate_id=a1.candidate_id,
            evaluation_id="evaluation-a1",
            source_document=document,
            raw_output=raw_output,
            planner_identity={"backend": "llama.cpp", "model": "Qwen", "prompt_version": "v1", "prompt_sha256": "8" * 64},
            repair_input_hash=input_hash,
        )
        first = schedule_b1_retry(self.store, self.layout, **kwargs)
        second = schedule_b1_retry(
            PipelineStateStore(self.layout.database_path), self.layout, **kwargs
        )

        self.assertEqual(second.candidate, first.candidate)
        self.assertEqual(first.candidate.tier, QcTier.B1)
        self.assertEqual(
            self.store.qc_candidate_source_identity(first.candidate.candidate_id),
            PipelineStateStore(self.layout.database_path).qc_candidate_source_identity(
                first.candidate.candidate_id
            ),
        )
        self.assertEqual(first.candidate.state, QcCandidateState.PENDING_GENERATION)
        self.assertNotEqual(first.seed, a1.current_seed)
        self.assertEqual(len(self.store.qc_repairs(a1.candidate_id)), 1)
        self.assertEqual(
            len([c for c in self.store.qc_candidates(self.job.job_id, 1) if c.tier == QcTier.B1]),
            1,
        )

    def test_changed_baseline_locked_document_blocks_b1_descendant(self) -> None:
        a1, document = self._completed_a1_failure()
        mutated = deepcopy(self.document)
        mutated["character"]["name"] = "out-of-band mutation"
        connection = sqlite3.connect(self.layout.database_path)
        try:
            connection.execute(
                """
                UPDATE scene_revisions SET parameters_json = ?
                WHERE job_id = ? AND scene_id = 1 AND revision = 1
                """,
                (canonical_json(mutated), self.job.job_id),
            )
            connection.commit()
        finally:
            connection.close()

        with self.assertRaisesRegex(StateTransitionError, "document changed"):
            schedule_b1_retry(
                self.store,
                self.layout,
                original_job=self.job,
                source_candidate_id=a1.candidate_id,
                evaluation_id="evaluation-a1",
                source_document=document,
                raw_output="",
                planner_identity={"backend": "test"},
                repair_input_hash="7" * 64,
            )
    def test_rejected_b1_persists_the_parsed_forbidden_patch(self) -> None:
        a1, document = self._completed_a1_failure()
        input_hash = "9" * 64
        parsed_patch = {
            "i2v": {
                "prompt": "A minimal corrected hand transition.",
                "seed": 123,
            }
        }
        raw_output = json.dumps(
            {
                "schema_version": 1,
                "source": {
                    "candidate_id": a1.candidate_id,
                    "candidate_sha256": a1.source_video_sha256,
                    "evaluation_id": "evaluation-a1",
                    "repair_input_sha256": input_hash,
                    "source_revision": a1.revision,
                    "source_document_sha256": hashlib.sha256(
                        canonical_json(document).encode("utf-8")
                    ).hexdigest(),
                },
                "patch": parsed_patch,
                "summary": "Invalid model-owned seed proposal.",
            }
        )

        result = schedule_b1_retry(
            self.store,
            self.layout,
            original_job=self.job,
            source_candidate_id=a1.candidate_id,
            evaluation_id="evaluation-a1",
            source_document=document,
            raw_output=raw_output,
            planner_identity={"backend": "fake"},
            repair_input_hash=input_hash,
        )

        self.assertIsNone(result.candidate)
        self.assertEqual(result.repair.status, "REJECTED")
        self.assertEqual(result.repair.proposed_patch, parsed_patch)
        manifest = json.loads(
            Path(result.repair.evidence_manifest_path).read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["parsed_patch"], parsed_patch)
        self.assertIsNone(manifest["validated_patch"])


if __name__ == "__main__":
    unittest.main()
