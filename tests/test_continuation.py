from __future__ import annotations

import unittest

from tenminvideomaker.continuation import (
    ContinuationPlanError,
    assembly_spans,
    build_scene_frame_plan,
    continuation_is_enabled,
    derived_chunk_seed,
    generation_transition_count,
    handoff_latent_token_count,
    materialize_segments,
    refinement_raw_frame_count,
    timeline_frame_count,
    transition_contributions,
)


class ContinuationPlanTests(unittest.TestCase):
    def _plan(self, seconds: float, **overrides):
        values = {
            "job_id": "20260729-continue",
            "scene_id": 1,
            "revision": 1,
            "requested_duration_seconds": seconds,
            "base_seed": 2472348977,
            "fallback_prompt": "An adult woman walks steadily toward screen right.",
            "fallback_negative": "text, watermark, body distortion",
        }
        values.update(overrides)
        return build_scene_frame_plan(**values)

    def test_exact_timeline_and_generation_master_math(self) -> None:
        cases = (
            (5.0, 120, 121, (120,)),
            (10.0, 240, 241, (120, 120)),
            (20.0, 480, 481, (120, 120, 120, 120)),
            (30.0, 720, 721, (120, 120, 120, 120, 120, 120)),
            (32.0, 768, 769, (120, 120, 120, 120, 120, 120, 48)),
        )
        for seconds, timeline, master, contributions in cases:
            with self.subTest(seconds=seconds):
                plan = self._plan(seconds)
                self.assertEqual(plan.timeline_output_frames, timeline)
                self.assertEqual(plan.generation_master_frames, master)
                self.assertEqual(
                    tuple(chunk.new_transition_frames for chunk in plan.chunks),
                    contributions,
                )
                self.assertEqual(
                    plan.chunks[-1].cumulative_master_frames,
                    master,
                )

    def test_continuation_windows_are_121_or_short_valid_8n_plus_1(self) -> None:
        plan = self._plan(30.0)
        self.assertEqual(plan.chunk_count, 6)
        for chunk in plan.chunks:
            self.assertGreaterEqual(chunk.model_window_frames, 25)
            self.assertEqual((chunk.model_window_frames - 1) % 8, 0)
        self.assertEqual(
            [chunk.model_window_frames for chunk in plan.chunks],
            [121, 121, 121, 121, 121, 121],
        )
        self.assertEqual(
            [chunk.global_window_start_frame for chunk in plan.chunks],
            [0, 120, 240, 360, 480, 600],
        )

    def test_exact_last_frame_handoff_drops_only_duplicate_frame_zero(self) -> None:
        plan = self._plan(30.0)
        spans = assembly_spans(plan)
        self.assertEqual(
            [span.frame_count for span in spans],
            [121, 120, 120, 120, 120, 120],
        )
        self.assertEqual(
            [span.input_start_frame for span in spans],
            [0, 1, 1, 1, 1, 1],
        )
        self.assertEqual(spans[0].master_start_frame, 0)
        self.assertEqual(
            spans[-1].master_end_frame_exclusive,
            plan.generation_master_frames,
        )
        self.assertEqual(
            sum(span.frame_count for span in spans),
            plan.generation_master_frames,
        )
        self.assertEqual(
            [refinement_raw_frame_count(chunk) for chunk in plan.chunks],
            [121, 121, 121, 121, 121, 121],
        )
        self.assertEqual(
            [handoff_latent_token_count(chunk) for chunk in plan.chunks],
            [16, 16, 16, 16, 16, 16],
        )

    def test_rounding_never_under_generates_requested_timeline(self) -> None:
        plan = self._plan(18.37)
        self.assertEqual(plan.timeline_output_frames, round(18.37 * 24))
        self.assertGreaterEqual(
            plan.generation_master_frames,
            plan.timeline_output_frames,
        )
        self.assertLessEqual(
            plan.generation_master_frames - plan.timeline_output_frames,
            7,
        )

    def test_seed_derivation_is_stable_distinct_and_unsigned_64_bit(self) -> None:
        first = derived_chunk_seed(
            job_id="job",
            scene_id=2,
            revision=3,
            base_seed=4,
            chunk_index=5,
            prompt="prompt",
        )
        self.assertEqual(
            first,
            derived_chunk_seed(
                job_id="job",
                scene_id=2,
                revision=3,
                base_seed=4,
                chunk_index=5,
                prompt="prompt",
            ),
        )
        self.assertNotEqual(
            first,
            derived_chunk_seed(
                job_id="job",
                scene_id=2,
                revision=3,
                base_seed=4,
                chunk_index=6,
                prompt="prompt",
            ),
        )
        self.assertGreaterEqual(first, 0)
        self.assertLessEqual(first, 0xFFFFFFFFFFFFFFFF)

    def test_explicit_segments_map_to_overlapping_model_windows(self) -> None:
        segments = [
            {
                "index": 0,
                "requested_duration_seconds": 5.0,
                "positive_prompt": "She begins walking.",
                "negative_prompt_additions": ["motion reset"],
            },
            {
                "index": 1,
                "requested_duration_seconds": 5.0,
                "positive_prompt": "She turns while continuing forward.",
                "negative_prompt_additions": [],
            },
        ]
        plan = self._plan(
            10.0,
            raw_segments=segments,
            continuity={
                "identity_anchors": ["The same 28-year-old adult woman."],
                "camera_axis": "Keep the camera south of the action axis.",
            },
        )
        self.assertEqual(plan.chunks[0].segment_indices, (0, 1))
        self.assertEqual(plan.chunks[1].segment_indices, (1,))
        self.assertIn("same 28-year-old", plan.chunks[1].prompt)
        self.assertIn("Continue seamlessly", plan.chunks[1].prompt)
        self.assertIn("motion reset", plan.chunks[0].negative)

    def test_legacy_prompt_is_reused_without_semantic_splitting(self) -> None:
        plan = self._plan(10.0)
        self.assertTrue(
            all(
                chunk.prompt_segmentation_quality == "fallback_reused_prompt"
                for chunk in plan.chunks
            )
        )
        self.assertTrue(
            all(
                "An adult woman walks steadily" in chunk.prompt
                for chunk in plan.chunks
            )
        )

    def test_invalid_frame_and_segment_values_are_rejected(self) -> None:
        with self.assertRaises(ContinuationPlanError):
            timeline_frame_count(0)
        with self.assertRaises(ContinuationPlanError):
            generation_transition_count(0)
        with self.assertRaises(ContinuationPlanError):
            transition_contributions(121)
        with self.assertRaises(ContinuationPlanError):
            materialize_segments(
                [
                    {
                        "requested_duration_seconds": 1,
                        "positive_prompt": "",
                    }
                ],
                timeline_frames=24,
            )
        with self.assertRaisesRegex(
            ContinuationPlanError,
            "cover the complete scene timeline",
        ):
            materialize_segments(
                [
                    {
                        "requested_duration_seconds": 4,
                        "positive_prompt": "First incomplete beat.",
                    }
                ],
                timeline_frames=240,
            )
        with self.assertRaisesRegex(
            ContinuationPlanError,
            "cover the complete scene timeline",
        ):
            materialize_segments(
                [
                    {
                        "requested_duration_seconds": 11,
                        "positive_prompt": "Overlong beat.",
                    }
                ],
                timeline_frames=240,
            )

    def test_segment_coverage_allows_only_frame_rounding_drift(self) -> None:
        segments = materialize_segments(
            [
                {
                    "requested_duration_seconds": 1.01,
                    "positive_prompt": "First beat.",
                },
                {
                    "requested_duration_seconds": 0.99,
                    "positive_prompt": "Second beat.",
                },
            ],
            timeline_frames=49,
        )
        self.assertEqual(segments[0].start_frame, 0)
        self.assertEqual(segments[-1].end_frame_exclusive, 49)

    def test_segment_materialization_rejects_ambiguous_timing(self) -> None:
        with self.assertRaisesRegex(
            ContinuationPlanError,
            "exactly one of requested_duration_seconds or new_transition_frames",
        ):
            materialize_segments(
                [
                    {
                        "requested_duration_seconds": 5.0,
                        "new_transition_frames": 120,
                        "positive_prompt": "Ambiguous beat.",
                    }
                ],
                timeline_frames=120,
            )

    def test_rollout_mode_requires_long_scene_and_respects_explicit_opt_out(self):
        self.assertFalse(
            continuation_is_enabled(
                scene_frame_count=121,
                continuation={"enabled": True},
                mode="auto",
            )
        )
        self.assertFalse(
            continuation_is_enabled(
                scene_frame_count=241,
                continuation=None,
                mode="explicit",
            )
        )
        self.assertTrue(
            continuation_is_enabled(
                scene_frame_count=241,
                continuation={"enabled": True},
                mode="explicit",
            )
        )
        self.assertTrue(
            continuation_is_enabled(
                scene_frame_count=241,
                continuation=None,
                mode="auto",
            )
        )
        self.assertFalse(
            continuation_is_enabled(
                scene_frame_count=241,
                continuation={"enabled": False},
                mode="auto",
            )
        )


if __name__ == "__main__":
    unittest.main()
