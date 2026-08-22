from __future__ import annotations

from copy import deepcopy
import json
import unittest

from tenminvideomaker.contracts import parse_job_payload
from tenminvideomaker.review import (
    ReviewValidationError,
    scene_review_document,
    validate_scene_edit,
)

from test_contracts import payload


class ReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.job = parse_job_payload(payload())
        self.scene = self.job.scenes[0]

    def test_document_exposes_effective_sampling_and_preserves_large_seed_as_text(self) -> None:
        document = scene_review_document(self.job, self.scene)
        self.assertEqual(document["t2i"]["passes"][0]["sampler"], "er_sde")
        self.assertEqual(document["i2v"]["first_pass"]["sampler"], "euler_ancestral")
        self.assertEqual(
            document["i2v"]["second_pass"]["sampler"],
            "euler_ancestral_cfg_pp",
        )
        self.assertEqual(
            document["production_profile"]["ltx_checkpoint"],
            "10Eros_v1.5_fp8mixed_experimental_learned.safetensors",
        )
        self.assertEqual(document["i2v"]["second_pass"]["sigmas"][-1], 0.0)
        self.assertIsInstance(document["i2v"]["seed"], str)
        self.assertTrue(document["production_profile"]["locked"])

    def test_sampler_sigma_prompt_seed_and_lora_edits_validate(self) -> None:
        document = scene_review_document(self.job, self.scene)
        document["i2v"]["first_pass"]["sampler"] = "euler"
        document["i2v"]["first_pass"]["sigmas"] = [1.0, 0.5, 0.0]
        document["i2v"]["prompt"] = "Edited motion prompt"
        document["i2v"]["seed"] = "18446744073709551615"
        document["i2v"]["loras"].append(
            {
                "name": "Motion",
                "download_url": "https://civitai.com/api/download/models/987654",
                "weight": 0.7,
            }
        )
        result = validate_scene_edit(self.job, self.scene.scene_id, document)
        self.assertEqual(result.workflow.i2v_first_pass["sampler"], "euler")
        self.assertEqual(result.workflow.i2v_first_pass["sigmas"], (1.0, 0.5, 0.0))
        self.assertEqual(result.scene.i2v.seed, 2**64 - 1)
        self.assertEqual(result.scene.i2v.prompt, "Edited motion prompt")

    def test_image_only_mode_is_absent_and_invalid_sigmas_fail(self) -> None:
        document = scene_review_document(self.job, self.scene)
        document["i2v"]["first_pass"]["sigmas"] = [0.5, 1.0, 0.0]
        with self.assertRaisesRegex(ReviewValidationError, "non-increasing"):
            validate_scene_edit(self.job, self.scene.scene_id, document)

    def test_continuation_beats_round_trip_through_scene_revision(self) -> None:
        data = payload()
        data["schema_version"] = "2"
        data["scenes"][0]["i2v"].update(
            {
                "continuation": {
                    "enabled": True,
                    "fps": 24,
                    "base_window_transition_frames": 120,
                    "overlap_transition_frames": 24,
                },
                "continuity": {
                    "identity_anchors": ["The same 28-year-old adult woman."],
                    "screen_direction": "Continue toward screen right.",
                },
                "segments": [
                    {
                        "index": 0,
                        "requested_duration_seconds": 5.0,
                        "positive_prompt": "She begins walking.",
                        "negative_prompt_additions": [],
                    }
                ],
            }
        )
        job = parse_job_payload(data)
        scene = job.scenes[0]
        document = scene_review_document(job, scene)
        document["i2v"]["segments"][0]["positive_prompt"] = (
            "She begins walking at a steady pace."
        )
        document["i2v"]["segments"][0]["seed_override"] = (
            "18446744073709551615"
        )

        result = validate_scene_edit(job, scene.scene_id, document)

        self.assertEqual(
            result.scene.i2v.segments[0]["positive_prompt"],
            "She begins walking at a steady pace.",
        )
        self.assertEqual(
            result.workflow.continuity["screen_direction"],
            "Continue toward screen right.",
        )
        self.assertTrue(result.workflow.temporal_continuation["enabled"])
        self.assertEqual(
            result.scene.i2v.segments[0]["seed_override"],
            2**64 - 1,
        )
        self.assertEqual(
            result.document["i2v"]["segments"][0]["seed_override"],
            str(2**64 - 1),
        )
        encoded = json.dumps(result.document)
        self.assertIsInstance(
            json.loads(encoded)["i2v"]["segments"][0]["seed_override"],
            str,
        )
        self.assertEqual(
            result.document["i2v"]["segments"][0]["positive_prompt"],
            "She begins walking at a steady pace.",
        )

    def test_visible_estimated_seconds_replaces_legacy_hidden_duration(self) -> None:
        data = payload()
        data["schema_version"] = "2"
        data["scenes"][0]["estimated_sec"] = 18
        data["scenes"][0]["i2v"]["continuation"] = {
            "enabled": True,
            "fps": 24,
            "base_window_transition_frames": 120,
            "overlap_transition_frames": 24,
            "requested_duration_seconds": 10,
        }
        job = parse_job_payload(data)
        scene = job.scenes[0]

        document = scene_review_document(job, scene)

        self.assertEqual(document["estimated_seconds"], 10)
        self.assertNotIn(
            "requested_duration_seconds",
            document["i2v"]["temporal_continuation"],
        )
        self.assertEqual(
            document["production_profile"]["timeline_output_frames"],
            240,
        )
        self.assertEqual(
            document["production_profile"]["generation_master_frames"],
            241,
        )

        document["estimated_seconds"] = 12
        # Simulate a stale browser revision that still carries the retired
        # hidden duration.  Validation must not let it override the visible
        # scene field in the new immutable revision.
        document["i2v"]["temporal_continuation"][
            "requested_duration_seconds"
        ] = 30
        result = validate_scene_edit(job, scene.scene_id, document)

        self.assertEqual(result.scene.estimated_sec, 12)
        self.assertNotIn(
            "requested_duration_seconds",
            result.workflow.temporal_continuation,
        )
        self.assertNotIn(
            "requested_duration_seconds",
            result.document["i2v"]["temporal_continuation"],
        )
        self.assertEqual(
            result.document["production_profile"]["timeline_output_frames"],
            288,
        )
        self.assertEqual(
            result.document["production_profile"]["generation_master_frames"],
            289,
        )


if __name__ == "__main__":
    unittest.main()
