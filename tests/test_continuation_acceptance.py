from __future__ import annotations

import unittest

from tenminvideomaker.continuation_acceptance import (
    build_acceptance_job,
    build_acceptance_plans,
)

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
        self.assertEqual(job.scenes[0].estimated_sec, 9.0)
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


if __name__ == "__main__":
    unittest.main()
