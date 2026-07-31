from __future__ import annotations

import unittest

from tenminvideomaker.continuation_acceptance import (
    build_acceptance_job,
    build_acceptance_plans,
)
from tenminvideomaker.continuation import build_scene_frame_plan

from test_contracts import payload


class ContinuationAcceptanceTests(unittest.TestCase):
    def test_builds_one_safe_two_window_job_from_selected_source_scene(self) -> None:
        job = build_acceptance_job(
            payload(),
            source_scene_id=1,
            acceptance_job_id="continuation-acceptance-test",
        )

        self.assertEqual(job.job_id, "continuation-acceptance-test")
        self.assertEqual(len(job.scenes), 1)
        self.assertEqual(job.scenes[0].estimated_sec, 10.0)
        self.assertEqual(len(job.scenes[0].i2v.segments), 2)
        self.assertIn("clearly adult", job.scenes[0].i2v.segments[0]["positive_prompt"])
        self.assertEqual(job.scenes[0].i2v.loras, ())

    def test_diagnostic_initial_window_reuses_latent_overlap_beat_seed(self) -> None:
        job = build_acceptance_job(
            payload(),
            source_scene_id=1,
            acceptance_job_id="continuation-acceptance-test",
        )
        plans = build_acceptance_plans(job, revision=1)

        self.assertEqual(plans.full.chunk_count, 2)
        self.assertEqual(plans.base.model_window_frames, 121)
        self.assertEqual(plans.latent_overlap.model_window_frames, 121)
        self.assertTrue(plans.diagnostic.is_initial)
        self.assertEqual(plans.diagnostic.seed, plans.latent_overlap.seed)
        self.assertEqual(plans.diagnostic.prompt, plans.latent_overlap.prompt)
        self.assertEqual(plans.diagnostic.negative, plans.latent_overlap.negative)

    def test_can_build_a_three_window_exact_frame_accumulation_fixture(self) -> None:
        job = build_acceptance_job(
            payload(),
            source_scene_id=1,
            acceptance_job_id="continuation-acceptance-long-test",
            duration_seconds=15.0,
        )
        scene = job.scenes[0]
        plan = build_scene_frame_plan(
            job_id=job.job_id,
            scene_id=scene.scene_id,
            revision=1,
            requested_duration_seconds=scene.estimated_sec,
            base_seed=scene.i2v.seed,
            fallback_prompt=scene.i2v.prompt,
            fallback_negative=scene.i2v.negative,
            continuity=scene.i2v.continuity,
            raw_segments=scene.i2v.segments,
        )

        self.assertEqual(job.scenes[0].estimated_sec, 15.0)
        self.assertEqual(plan.chunk_count, 3)
        self.assertEqual([chunk.model_window_frames for chunk in plan.chunks], [121] * 3)

    def test_safe_acceptance_preserves_source_continuity_and_style_anchors(self) -> None:
        raw = payload()
        raw["scenes"][0]["i2v"]["continuity"] = {
            "identity_anchors": ["same silver-haired adult traveler"],
            "wardrobe_anchors": ["same buttoned teal coat"],
            "environment_anchors": [
                "same blue-and-gold corridor",
                "cel-shaded game-render style, never photorealistic",
            ],
            "camera_axis": "source camera axis",
            "screen_direction": "source screen direction",
        }

        job = build_acceptance_job(
            raw,
            source_scene_id=1,
            acceptance_job_id="continuation-acceptance-style-test",
        )
        continuity = job.scenes[0].i2v.continuity

        self.assertIsNotNone(continuity)
        assert continuity is not None
        self.assertIn(
            "same silver-haired adult traveler",
            continuity["identity_anchors"],
        )
        self.assertIn("same buttoned teal coat", continuity["wardrobe_anchors"])
        self.assertIn(
            "cel-shaded game-render style, never photorealistic",
            continuity["environment_anchors"],
        )
        self.assertEqual(continuity["camera_axis"], "Preserve the lateral tracking axis.")
        self.assertEqual(
            continuity["screen_direction"],
            "Movement continues toward screen right.",
        )


if __name__ == "__main__":
    unittest.main()
