from __future__ import annotations

import unittest

from tenminvideomaker.constants import frame_count_for_seconds, seconds_for_frame_count
from tenminvideomaker.contracts import ContractValidationError, effective_t2i_loras, parse_job_payload


def payload() -> dict:
    return {
        "job_id": "20260724-1610",
        "character": {
            "name": "Elsa",
            "series": "Frozen",
            "lora": {
                "name": "Elsa Frozen Anima",
                "base": "Anima",
                "download_url": "https://civitai.com/api/download/models/3184055",
                "recommended_weight": 0.85,
            },
        },
        "ltxv_character_lora": None,
        "scenes": [
            {
                "id": 1,
                "title": "Portrait",
                "estimated_sec": 18,
                "t2i": {
                    "prompt": "portrait prompt",
                    "negative": "bad quality",
                    "seed": 2472348977,
                    "loras": [
                        {
                            "name": "Elsa Frozen Anima",
                            "download_url": "https://civitai.com/api/download/models/3184055",
                            "weight": 0.85,
                        }
                    ],
                },
                "i2v": {
                    "prompt": "movement prompt",
                    "negative": "watermark",
                    "seed": 2472348977,
                    "loras": [],
                },
            }
        ],
    }


class ContractTests(unittest.TestCase):
    def test_valid_payload_uses_ltx_frame_rule(self) -> None:
        job = parse_job_payload(payload())
        self.assertEqual(job.character.base_model, "Anima")
        self.assertEqual(job.character.lora.weight, 0.85)
        self.assertEqual(job.scenes[0].frame_count, 433)
        self.assertEqual(seconds_for_frame_count(job.scenes[0].frame_count), 18.0)

    def test_duration_is_rejected_before_generation(self) -> None:
        data = payload()
        data["scenes"][0]["estimated_sec"] = 32.01
        with self.assertRaisesRegex(ContractValidationError, "no more than 32"):
            parse_job_payload(data)

    def test_global_character_lora_is_not_applied_twice(self) -> None:
        job = parse_job_payload(payload())
        loras = effective_t2i_loras(job.scenes[0], job.character)
        self.assertEqual([lora.name for lora in loras], ["Elsa Frozen Anima"])

    def test_non_https_asset_url_is_rejected(self) -> None:
        data = payload()
        data["character"]["lora"]["download_url"] = "http://example.invalid/lora"
        with self.assertRaisesRegex(ContractValidationError, "HTTPS"):
            parse_job_payload(data)

    def test_duplicate_scene_ids_are_rejected(self) -> None:
        data = payload()
        duplicate = payload()["scenes"][0].copy()
        data["scenes"].append(duplicate)
        with self.assertRaisesRegex(ContractValidationError, "duplicate scene ids"):
            parse_job_payload(data)

    def test_frame_count_rejects_out_of_range_duration(self) -> None:
        with self.assertRaises(ValueError):
            frame_count_for_seconds(33)

    def test_character_lora_requires_exact_base_and_recommended_weight_fields(self) -> None:
        data = payload()
        data["character"]["lora"]["weight"] = data["character"]["lora"].pop("recommended_weight")
        with self.assertRaisesRegex(ContractValidationError, "recommended_weight"):
            parse_job_payload(data)

        data = payload()
        data["character"]["lora"]["base"] = "Unknown"
        with self.assertRaisesRegex(ContractValidationError, "Anima or Pony"):
            parse_job_payload(data)


if __name__ == "__main__":
    unittest.main()
