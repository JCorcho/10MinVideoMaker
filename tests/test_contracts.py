from __future__ import annotations

import unittest

from tenminvideomaker.constants import frame_count_for_seconds, seconds_for_frame_count
from tenminvideomaker.contracts import (
    ContractValidationError,
    effective_i2v_loras,
    effective_t2i_loras,
    lora_identity,
    parse_job_payload,
)


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
        self.assertEqual(job.schema_version, "1")
        self.assertEqual(job.character.base_model, "Anima")
        self.assertEqual(job.character.lora.weight, 0.85)
        self.assertEqual(job.scenes[0].frame_count, 433)
        self.assertEqual(seconds_for_frame_count(job.scenes[0].frame_count), 18.0)

    def test_optional_continuation_contract_is_typed_without_breaking_legacy(self) -> None:
        data = payload()
        data["schema_version"] = "2"
        data["scenes"][0]["i2v"].update(
            {
                "continuation": {
                    "enabled": True,
                    "strategy": "ltx23_latent_overlap_v1",
                    "fps": 24,
                    "base_window_transition_frames": 120,
                    "overlap_transition_frames": 24,
                    "seed_policy": "derived_v1",
                },
                "continuity": {
                    "identity_anchors": ["The same 28-year-old adult woman."],
                    "camera_axis": "Keep the camera south of the action axis.",
                },
                "segments": [
                    {
                        "index": 0,
                        "requested_duration_seconds": 5.0,
                        "positive_prompt": "She begins walking toward screen right.",
                        "negative_prompt_additions": ["motion reset"],
                        "variation_index": 0,
                    }
                ],
            }
        )

        job = parse_job_payload(data)

        self.assertEqual(job.schema_version, "2")
        self.assertTrue(job.scenes[0].i2v.continuation["enabled"])
        self.assertEqual(
            job.scenes[0].i2v.continuity["camera_axis"],
            "Keep the camera south of the action axis.",
        )
        self.assertEqual(job.scenes[0].i2v.segments[0]["index"], 0)

    def test_continuation_profile_cannot_change_fixed_production_math(self) -> None:
        data = payload()
        data["scenes"][0]["i2v"]["continuation"] = {
            "enabled": True,
            "fps": 30,
        }
        with self.assertRaisesRegex(ContractValidationError, "must be the production value 24"):
            parse_job_payload(data)

        data = payload()
        data["scenes"][0]["i2v"]["continuation"] = {
            "enabled": True,
            "overlap_transition_frames": 16,
        }
        with self.assertRaisesRegex(ContractValidationError, "must be 24"):
            parse_job_payload(data)

        data = payload()
        data["scenes"][0]["i2v"]["continuation"] = {
            "enabled": True,
            "strategy": "unknown_strategy",
        }
        with self.assertRaisesRegex(
            ContractValidationError,
            "ltx23_latent_overlap_v1",
        ):
            parse_job_payload(data)

    def test_invalid_continuation_segments_are_rejected(self) -> None:
        data = payload()
        data["scenes"][0]["i2v"]["segments"] = [
            {
                "index": 0,
                "positive_prompt": "Beat",
                "new_transition_frames": 25,
            }
        ]
        with self.assertRaisesRegex(ContractValidationError, "multiple of 8"):
            parse_job_payload(data)

    def test_continuation_segment_timing_sources_are_mutually_exclusive(self) -> None:
        data = payload()
        data["scenes"][0]["i2v"]["segments"] = [
            {
                "index": 0,
                "positive_prompt": "Beat",
                "requested_duration_seconds": 5.0,
                "new_transition_frames": 120,
            }
        ]
        with self.assertRaisesRegex(
            ContractValidationError,
            "exactly one of requested_duration_seconds or new_transition_frames",
        ):
            parse_job_payload(data)

    def test_segment_ltx_loras_are_rejected_until_routing_is_supported(self) -> None:
        data = payload()
        data["scenes"][0]["i2v"]["segments"] = [
            {
                "index": 0,
                "positive_prompt": "Beat",
                "requested_duration_seconds": 18.0,
                "ltx_loras": [
                    {
                        "name": "Segment motion",
                        "download_url": "https://civitai.com/api/download/models/987654",
                        "weight": 0.7,
                    }
                ],
            }
        ]
        with self.assertRaisesRegex(
            ContractValidationError,
            "segments\\[0\\]\\.ltx_loras is not supported",
        ):
            parse_job_payload(data)

    def test_duration_is_rejected_before_generation(self) -> None:
        data = payload()
        data["scenes"][0]["estimated_sec"] = 32.01
        with self.assertRaisesRegex(ContractValidationError, "no more than 32"):
            parse_job_payload(data)

    def test_global_character_lora_is_not_applied_twice(self) -> None:
        job = parse_job_payload(payload())
        loras = effective_t2i_loras(job.scenes[0], job.character)
        self.assertEqual([lora.name for lora in loras], ["Elsa Frozen Anima"])

    def test_same_civitai_version_deduplicates_even_when_names_differ(self) -> None:
        data = payload()
        data["scenes"][0]["t2i"]["loras"][0]["name"] = "Different display name"
        data["scenes"][0]["t2i"]["loras"][0]["weight"] = 0.25
        job = parse_job_payload(data)
        loras = effective_t2i_loras(job.scenes[0], job.character)
        self.assertEqual(len(loras), 1)
        self.assertEqual(loras[0].name, "Elsa Frozen Anima")
        self.assertEqual(loras[0].weight, 0.85)
        self.assertEqual(loras[0].version_id, 3184055)
        self.assertEqual(lora_identity(loras[0]), "civitai-version:3184055")

    def test_t2i_lora_alias_is_quarantined_from_i2v_candidates(self) -> None:
        data = payload()
        data["scenes"][0]["i2v"]["loras"] = [
            {
                "name": "Character alias accidentally repeated for video",
                "download_url": "https://civitai.com/api/download/models/3184055",
                "weight": 0.8,
            },
            {
                "name": "Actual LTX motion",
                "download_url": "https://civitai.com/api/download/models/3082662",
                "weight": 0.7,
            },
        ]
        job = parse_job_payload(data)
        loras = effective_i2v_loras(job, job.scenes[0])
        self.assertEqual([lora.name for lora in loras], ["Actual LTX motion"])

    def test_explicit_civitai_version_must_match_download_url(self) -> None:
        data = payload()
        data["scenes"][0]["t2i"]["loras"][0]["version_id"] = 123
        with self.assertRaisesRegex(ContractValidationError, "does not match"):
            parse_job_payload(data)

    def test_digit_only_civitai_ids_are_normalized_from_gmail_json(self) -> None:
        data = payload()
        data["character"]["lora"]["version_id"] = "3184055"
        data["character"]["lora"]["model_id"] = "2847192"

        job = parse_job_payload(data)

        self.assertEqual(job.character.lora.version_id, 3184055)
        self.assertEqual(job.character.lora.model_id, 2847192)

    def test_noncanonical_civitai_id_strings_remain_rejected(self) -> None:
        for invalid_id in ("", " 3184055", "+3184055", "3184055.0", "version-3184055"):
            with self.subTest(invalid_id=invalid_id):
                data = payload()
                data["character"]["lora"]["version_id"] = invalid_id
                with self.assertRaisesRegex(ContractValidationError, "positive integer"):
                    parse_job_payload(data)

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
