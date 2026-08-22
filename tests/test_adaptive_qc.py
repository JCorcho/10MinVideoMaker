from __future__ import annotations

import unittest

from tenminvideomaker.adaptive_qc import (
    AdaptivePlanError, apply_repair_plan, next_active_strategy,
    start_frame_defect, validate_repair_plan,
)
from tenminvideomaker.qc_contracts import QcTier


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


if __name__ == "__main__":
    unittest.main()
