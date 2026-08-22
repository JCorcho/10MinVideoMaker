from __future__ import annotations

import unittest
import hashlib
from pathlib import Path
import tempfile

from tenminvideomaker.adaptive_qc import (
    AdaptivePlanError, apply_repair_plan, next_active_strategy,
    start_frame_defect, validate_repair_plan,
)
from tenminvideomaker.qc_contracts import QcTier
from tenminvideomaker.contracts import parse_job_payload
from tenminvideomaker.qc_contracts import QcArtifactStage, QcCandidateState
from tenminvideomaker.review import scene_review_document
from tenminvideomaker.state_store import PipelineStateStore, SceneState, StateTransitionError
from test_contracts import payload


def plan(**overrides):
    value = {
        "schema_version": "adaptive_repair_plan_v1",
        "authority_tier": "A",
        "strategy": "reduced_motion_pressure",
        "hypothesis": "Stronger source conditioning should stabilize ownership.",
        "preserve_start_frame": True,
        "changes": [{"path": "i2v.first_pass.preset", "operation": "replace", "value": "reduced_motion_pressure"}],
        "failure_addressed": ["hands"],
        "difference_from_prior_attempts": "Changes conditioning, not only seed.",
        "expected_effect": "Stable geometry.",
    }
    value.update(overrides)
    return value


class AdaptiveQcTests(unittest.TestCase):
    def setUp(self):
        self.document = {"i2v": {"seed": "1", "prompt": "motion", "negative": "adult-only safety", "first_pass": {"sampler": "lcm", "sigmas": [1.0, 0.0], "cfg": 1.0, "reference_strength": 0.75, "image_strength": 0.75, "image_compression": 35}}, "t2i": {"prompt": "frame"}, "scene_context": {"camera": "static", "staging": "center"}}

    def test_valid_preset_changes_real_controls_and_controller_seed(self):
        parsed = validate_repair_plan(plan(), minimum_authority="A")
        result, changes = apply_repair_plan(self.document, parsed, controller_seed=2)
        self.assertEqual(result["i2v"]["seed"], "2")
        self.assertEqual(result["i2v"]["first_pass"]["reference_strength"], 0.85)
        self.assertIn("i2v.first_pass.image_strength", changes)

    def test_plan_rejects_noop_unsupported_preset_and_backward_authority(self):
        for value, minimum in (
            (plan(changes=[]), "A"),
            (plan(changes=[{"path": "workflow.node.1", "operation": "replace", "value": "x"}]), "A"),
            (plan(changes=[{"path": "i2v.first_pass.preset", "operation": "replace", "value": "invented"}]), "A"),
            (plan(), "B"),
        ):
            with self.subTest(value=value, minimum=minimum), self.assertRaises(AdaptivePlanError):
                validate_repair_plan(value, minimum_authority=minimum)

    def test_repeated_strategy_is_rejected(self):
        with self.assertRaises(AdaptivePlanError):
            validate_repair_plan(plan(), minimum_authority="A", failed_strategies=["reduced_motion_pressure"])

    def test_start_frame_defect_skips_to_b2_and_temporal_defect_starts_a1(self):
        early = [{"category": "hands", "start_time_seconds": 0.0}]
        temporal = [{"category": "motion_continuity", "start_time_seconds": 2.0}]
        self.assertTrue(start_frame_defect(early))
        self.assertEqual(next_active_strategy(attempted=[], evidence=early)[0], QcTier.B2)
        self.assertEqual(next_active_strategy(attempted=[], evidence=temporal)[0], QcTier.A1)
        self.assertEqual(next_active_strategy(attempted=["new_seed"], evidence=temporal)[0], QcTier.A2)

    def test_all_active_strategies_defer_instead_of_terminal_hold(self):
        attempted = ["new_seed", "reduced_motion_pressure", "constrained_prompt_repair", "regenerate_start_frame", "current_scene_shot_redesign", "current_scene_semantic_replan"]
        self.assertIsNone(next_active_strategy(attempted=attempted, evidence=[])[0])


class AdaptivePersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.store = PipelineStateStore(root / "pipeline.sqlite3")
        self.job = parse_job_payload(payload())
        self.store.claim_job(self.job)
        self.frame = root / "frame.png"
        self.video = root / "draft.mp4"
        self.frame.write_bytes(b"frame")
        self.video.write_bytes(b"draft")
        self.document = scene_review_document(self.job, self.job.scenes[0])
        self.store.set_scene_state(self.job.job_id, 1, SceneState.SUCCEEDED, frame_path=str(self.frame), video_path=str(self.video))
        self.store.ensure_original_scene_revision(self.job.job_id, 1, parameters=self.document, frame_path=str(self.frame), video_path=str(self.video))
        self.original = self.store.ensure_qc_candidate(
            candidate_id="candidate-original", job_id=self.job.job_id, scene_id=1,
            revision=1, tier=QcTier.ORIGINAL, parent_candidate_id=None,
            source_video_path=str(self.video), source_video_sha256=hashlib.sha256(b"draft").hexdigest(),
            original_prompt=self.document["i2v"]["prompt"], current_prompt=self.document["i2v"]["prompt"],
            original_seed=int(self.document["i2v"]["seed"]), current_seed=int(self.document["i2v"]["seed"]),
            negative_prompt=self.document["i2v"]["negative"],
            negative_prompt_sha256=hashlib.sha256(self.document["i2v"]["negative"].encode()).hexdigest(),
            state=QcCandidateState.PENDING_QC, next_action="evaluate",
            artifact_stage=QcArtifactStage.DRAFT, recipe_sha256=None,
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_adaptive_candidate_and_attempt_are_atomic_and_recipe_unique(self):
        document = dict(self.document)
        document = {**document, "i2v": {**document["i2v"], "seed": "987654321", "first_pass": {**document["i2v"]["first_pass"], "reference_strength": 0.85, "image_strength": 0.85}}}
        candidate = self.store.create_adaptive_candidate_revision(
            parent_candidate_id=self.original.candidate_id, tier=QcTier.A2,
            parameters=document, frame_path=str(self.frame), source_video_path=str(Path(self.temp.name) / "a2-draft.mp4"),
            authority_tier="A", strategy="reduced_motion_pressure", defect_family="hands",
            intervention={"fields_changed": {"i2v.first_pass.reference_strength": {"before": 0.75, "after": 0.85}}},
        )
        attempts = self.store.adaptive_attempts(self.job.job_id, 1)
        self.assertEqual((candidate.recipe_sha256,), tuple(item.recipe_sha256 for item in attempts))
        self.assertEqual(attempts[0].strategy, "reduced_motion_pressure")
        with self.assertRaises(StateTransitionError):
            self.store.create_adaptive_candidate_revision(
                parent_candidate_id=self.original.candidate_id, tier=QcTier.C,
                parameters=document, frame_path=str(self.frame), source_video_path=str(Path(self.temp.name) / "duplicate.mp4"),
                authority_tier="C", strategy="current_scene_shot_redesign", defect_family="hands",
                intervention={"fields_changed": {}},
            )

    def test_pause_and_resume_preserve_candidate_history(self):
        paused = self.store.pause_qc_candidate(self.original.candidate_id)
        self.assertEqual(paused.state, QcCandidateState.PAUSED)
        resumed = self.store.resume_qc_candidate(self.original.candidate_id)
        self.assertEqual(resumed.state, QcCandidateState.DEFERRED_AUTOMATED_REPAIR)
        self.assertEqual(resumed.next_action, "resume_adaptive")


if __name__ == "__main__":
    unittest.main()
