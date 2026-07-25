from __future__ import annotations

from copy import deepcopy
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
        self.assertEqual(document["i2v"]["first_pass"]["sampler"], "lcm")
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


if __name__ == "__main__":
    unittest.main()
