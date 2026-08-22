from __future__ import annotations

from pathlib import Path
import unittest

from tenminvideomaker.constants import (
    I2V_BASE_HEIGHT,
    I2V_BASE_WIDTH,
    I2V_FIRST_PASS_SIGMAS,
    I2V_SPATIAL_UPSCALER,
    I2V_UPSCALE_PASS_SIGMAS,
    LTX_SPATIAL_DIMENSION_STEP,
    PRODUCTION_HEIGHT,
    PRODUCTION_WIDTH,
)
from tenminvideomaker.contracts import lora_identity, parse_job_payload
from tenminvideomaker.delivery import DiscordDeliverySettings
from tenminvideomaker.review import scene_review_document, validate_scene_edit
from tenminvideomaker.workflow_builder import (
    WorkflowBuildError,
    build_i2v_api_workflow,
    build_i2v_draft_api_workflow,
    build_i2v_final_api_workflow,
    build_t2i_api_workflow,
    validate_against_object_info,
    validate_api_graph,
)

from test_contracts import payload


def nodes_of_type(workflow, class_type):
    return [node for node in workflow.values() if node["class_type"] == class_type]


class WorkflowBuilderTests(unittest.TestCase):
    def test_qc_draft_omits_expensive_second_pass_and_checkpoints_handoff(self) -> None:
        frame = Path(r"D:\LTX_Supervisor_Storage\canary\frame.png")
        build = build_i2v_draft_api_workflow(
            self.job,
            self.scene,
            frame,
            revision=3,
            attempt_number=2,
        )
        titles = {node.get("_meta", {}).get("title") for node in build.api.values()}
        self.assertIn("First sampling pass", titles)
        self.assertIn("Checkpoint approved-draft video latent", titles)
        self.assertIn("Checkpoint approved-draft audio latent", titles)
        self.assertNotIn("Second sampling pass", titles)
        self.assertFalse(nodes_of_type(build.api, "LatentUpscaleModelLoader"))
        self.assertFalse(nodes_of_type(build.api, "LTXVLatentUpsamplerTiled"))
        self.assertEqual(len(nodes_of_type(build.api, "SamplerCustom")), 1)

    def test_qc_final_loads_approved_handoff_and_contains_only_second_pass(self) -> None:
        frame = Path(r"D:\LTX_Supervisor_Storage\canary\frame.png")
        build = build_i2v_final_api_workflow(
            self.job,
            self.scene,
            frame,
            revision=3,
            attempt_number=2,
        )
        titles = {node.get("_meta", {}).get("title") for node in build.api.values()}
        self.assertNotIn("First sampling pass", titles)
        self.assertIn("Second sampling pass", titles)
        self.assertIn("Load exact approved first-pass video latent", titles)
        self.assertIn("Load exact approved first-pass audio latent", titles)
        self.assertEqual(len(nodes_of_type(build.api, "SamplerCustom")), 1)
        self.assertTrue(nodes_of_type(build.api, "LatentUpscaleModelLoader"))
        self.assertTrue(nodes_of_type(build.api, "LTXVLatentUpsamplerTiled"))

    def test_reviewed_i2v_sampler_and_sigmas_reach_both_workflow_passes(self) -> None:
        job = parse_job_payload(payload())
        scene = job.scenes[0]
        document = scene_review_document(job, scene)
        document["i2v"]["first_pass"]["sampler"] = "euler"
        document["i2v"]["first_pass"]["sigmas"] = [1.0, 0.5, 0.0]
        document["i2v"]["second_pass"]["sampler"] = "heun"
        document["i2v"]["second_pass"]["sigmas"] = [0.5, 0.0]
        edit = validate_scene_edit(job, scene.scene_id, document)
        build = build_i2v_api_workflow(
            edit.job,
            edit.scene,
            Path(r"D:\LTX_Supervisor_Storage\jobs\job\frame.png"),
            {},
            overrides=edit.workflow,
        )
        samplers = nodes_of_type(build.api, "KSamplerSelect")
        sigmas = nodes_of_type(build.api, "ManualSigmas")
        self.assertEqual(
            [node["inputs"]["sampler_name"] for node in samplers],
            ["euler", "heun"],
        )
        self.assertEqual(
            [node["inputs"]["sigmas"] for node in sigmas],
            ["1, 0.5, 0", "0.5, 0"],
        )

    def setUp(self) -> None:
        self.job = parse_job_payload(payload())
        self.scene = self.job.scenes[0]
        self.delivery = DiscordDeliverySettings(
            "https://discord.com" + "/api/webhooks/123456789/test-token"
        )

    def test_anima_uses_reference_sampler_and_fixed_output_size(self) -> None:
        build = build_t2i_api_workflow(self.job, self.scene)
        self.assertEqual(validate_api_graph(build.api), ())
        sampler = nodes_of_type(build.api, "KSampler")[0]["inputs"]
        self.assertEqual((sampler["sampler_name"], sampler["scheduler"]), ("er_sde", "beta57"))
        self.assertEqual((sampler["steps"], sampler["cfg"]), (30, 4.5))
        latent = nodes_of_type(build.api, "EmptySD3LatentImage")[0]["inputs"]
        self.assertEqual((latent["width"], latent["height"]), (PRODUCTION_WIDTH, PRODUCTION_HEIGHT))
        self.assertFalse(nodes_of_type(build.api, "KSamplerSelect"))
        self.assertFalse(nodes_of_type(build.api, "FaceDetailer"))
        self.assertFalse(nodes_of_type(build.api, "UltralyticsDetectorProvider"))
        self.assertEqual(build.api[build.output_node_id]["class_type"], "10MinVideoMaker_SaveSceneFrame")

    def test_pony_uses_both_exact_reference_sampler_passes(self) -> None:
        raw = payload()
        raw["character"]["lora"]["base"] = "Pony"
        job = parse_job_payload(raw)
        build = build_t2i_api_workflow(job, job.scenes[0])
        samplers = nodes_of_type(build.api, "KSamplerAdvanced")
        self.assertEqual(
            [(node["inputs"]["sampler_name"], node["inputs"]["scheduler"]) for node in samplers],
            [("res_3m_ode", "karras"), ("res_5s_ode", "karras")],
        )
        self.assertTrue(all(node["inputs"]["steps"] == 30 for node in samplers))
        self.assertTrue(all(node["inputs"]["cfg"] == 6.0 for node in samplers))
        first_id = next(
            node_id
            for node_id, node in build.api.items()
            if node is samplers[0]
        )
        self.assertEqual(samplers[1]["inputs"]["latent_image"], [first_id, 0])
        detector = nodes_of_type(build.api, "UltralyticsDetectorProvider")[0]
        self.assertEqual(detector["inputs"]["model_name"], "bbox/face_yolov8m.pt")
        detailer = nodes_of_type(build.api, "FaceDetailer")[0]["inputs"]
        self.assertEqual(
            {
                "guide_size": detailer["guide_size"],
                "guide_size_for": detailer["guide_size_for"],
                "max_size": detailer["max_size"],
                "seed": detailer["seed"],
                "steps": detailer["steps"],
                "cfg": detailer["cfg"],
                "sampler_name": detailer["sampler_name"],
                "scheduler": detailer["scheduler"],
                "denoise": detailer["denoise"],
                "bbox_threshold": detailer["bbox_threshold"],
                "bbox_dilation": detailer["bbox_dilation"],
                "bbox_crop_factor": detailer["bbox_crop_factor"],
                "drop_size": detailer["drop_size"],
            },
            {
                "guide_size": 512,
                "guide_size_for": True,
                "max_size": 1024,
                "seed": job.scenes[0].t2i.seed,
                "steps": 20,
                "cfg": 5.0,
                "sampler_name": "dpmpp_2m_sde",
                "scheduler": "karras",
                "denoise": 0.38,
                "bbox_threshold": 0.5,
                "bbox_dilation": 10,
                "bbox_crop_factor": 3.0,
                "drop_size": 120,
            },
        )
        detector_id = next(
            node_id
            for node_id, node in build.api.items()
            if node is detector
        )
        self.assertEqual(detailer["bbox_detector"], [detector_id, 0])
        detailer_id = next(
            node_id
            for node_id, node in build.api.items()
            if node["class_type"] == "FaceDetailer"
        )
        saver = build.api[build.output_node_id]
        self.assertEqual(saver["inputs"]["images"], [detailer_id, 0])
        self.assertEqual(validate_api_graph(build.api), ())

    def test_i2v_uses_tenstrip_v5_samplers_sigmas_upscaler_and_chunking(self) -> None:
        build = build_i2v_api_workflow(
            self.job,
            self.scene,
            Path(r"D:\output\10minfinals\.work\20260724-1610\frames\scene_0001.png"),
        )
        self.assertEqual(validate_api_graph(build.api), ())
        samplers = nodes_of_type(build.api, "KSamplerSelect")
        self.assertEqual(
            [node["inputs"]["sampler_name"] for node in samplers],
            ["euler_ancestral", "euler_ancestral_cfg_pp"],
        )
        self.assertEqual(
            I2V_FIRST_PASS_SIGMAS,
            (1.0, 0.955, 0.893, 0.812, 0.715, 0.603, 0.482, 0.241, 0.121, 0.0),
        )
        self.assertEqual(I2V_UPSCALE_PASS_SIGMAS, (0.92, 0.725, 0.421875, 0.0))
        sigmas = [node["inputs"]["sigmas"] for node in nodes_of_type(build.api, "ManualSigmas")]
        self.assertEqual(
            sigmas,
            [
                ", ".join(f"{value:g}" for value in I2V_FIRST_PASS_SIGMAS),
                ", ".join(f"{value:g}" for value in I2V_UPSCALE_PASS_SIGMAS),
            ],
        )
        upscaler = nodes_of_type(build.api, "LatentUpscaleModelLoader")[0]
        self.assertEqual(upscaler["inputs"]["model_name"], I2V_SPATIAL_UPSCALER)
        chunking = nodes_of_type(build.api, "LTXVChunkFeedForward")[0]["inputs"]
        self.assertEqual((chunking["chunks"], chunking["dim_threshold"]), (2, 4096))

    def test_i2v_reuses_exact_cached_frame_and_fixed_timeline(self) -> None:
        frame_path = Path(r"D:\output\10minfinals\.work\20260724-1610\frames\scene_0001.png")
        build = build_i2v_api_workflow(self.job, self.scene, frame_path)
        loaded = nodes_of_type(build.api, "VHS_LoadImagePath")[0]["inputs"]
        self.assertEqual(loaded["image"], str(frame_path))
        latent = nodes_of_type(build.api, "EmptyLTXVLatentVideo")[0]["inputs"]
        self.assertEqual(
            (latent["width"], latent["height"], latent["length"]),
            (I2V_BASE_WIDTH, I2V_BASE_HEIGHT, self.scene.frame_count),
        )
        self.assertEqual((I2V_BASE_WIDTH, I2V_BASE_HEIGHT), (384, 672))
        self.assertTrue(
            all(
                dimension % LTX_SPATIAL_DIMENSION_STEP == 0
                for dimension in (
                    PRODUCTION_WIDTH,
                    PRODUCTION_HEIGHT,
                    I2V_BASE_WIDTH,
                    I2V_BASE_HEIGHT,
                )
            )
        )
        combine = nodes_of_type(build.api, "VHS_VideoCombine")[0]["inputs"]
        self.assertEqual(combine["frame_rate"], 24.0)
        self.assertFalse(combine["save_output"])
        self.assertEqual(build.api[build.output_node_id]["class_type"], "VHS_VideoCombine")

    def test_t2i_discord_branch_watermarks_full_quality_without_second_save(self) -> None:
        build = build_t2i_api_workflow(
            self.job,
            self.scene,
            delivery=self.delivery,
        )
        watermark = nodes_of_type(build.api, "DaSiWa_Watermark")[0]
        sender = nodes_of_type(build.api, "DiscordSendSaveImage")[0]
        sender_id = sender["inputs"]["images"][0]
        self.assertIs(build.api[sender_id], watermark)
        self.assertEqual(
            {
                "watermark_path": watermark["inputs"]["watermark_path"],
                "position": watermark["inputs"]["position"],
                "scale": watermark["inputs"]["scale"],
                "transparency": watermark["inputs"]["transparency"],
            },
            {
                "watermark_path": "wm.png",
                "position": "bottom-right",
                "scale": 0.7,
                "transparency": 0.4,
            },
        )
        self.assertEqual(sender["inputs"]["quality"], 100)
        self.assertTrue(sender["inputs"]["lossless"])
        self.assertTrue(sender["inputs"]["send_to_discord"])
        self.assertFalse(sender["inputs"]["save_output"])
        self.assertFalse(sender["inputs"]["include_prompts_in_message"])
        self.assertFalse(sender["inputs"]["send_workflow_json"])
        saver = build.api[build.output_node_id]
        self.assertNotEqual(saver["inputs"]["images"], sender["inputs"]["images"])

    def test_i2v_discord_branch_uses_watermark_audio_and_quality_65_only(self) -> None:
        frame_path = Path(r"D:\output\10minfinals\.work\job\frames\scene_0001.png")
        build = build_i2v_api_workflow(
            self.job,
            self.scene,
            frame_path,
            delivery=self.delivery,
        )
        watermark = nodes_of_type(build.api, "DaSiWa_Watermark")[0]
        sender = nodes_of_type(build.api, "DiscordSendSaveVideo")[0]
        combine = nodes_of_type(build.api, "VHS_VideoCombine")[0]
        self.assertEqual(sender["inputs"]["images"][0], next(
            node_id for node_id, node in build.api.items() if node is watermark
        ))
        self.assertNotEqual(sender["inputs"]["images"], combine["inputs"]["images"])
        self.assertEqual(sender["inputs"]["audio"], combine["inputs"]["audio"])
        self.assertEqual(sender["inputs"]["format"], "video/h264-mp4")
        self.assertEqual(sender["inputs"]["frame_rate"], 24.0)
        self.assertEqual(sender["inputs"]["quality"], 65)
        self.assertTrue(sender["inputs"]["send_to_discord"])
        self.assertFalse(sender["inputs"]["save_output"])
        self.assertFalse(sender["inputs"]["include_prompts_in_message"])
        self.assertFalse(sender["inputs"]["send_workflow_json"])

    def test_i2v_decodes_directly_to_exact_production_geometry(self) -> None:
        build = build_i2v_api_workflow(
            self.job,
            self.scene,
            Path(r"D:\output\10minfinals\.work\20260724-1610\frames\scene_0001.png"),
        )
        self.assertFalse(
            any(
                node.get("_meta", {}).get("title", "").startswith("Normalize decoded video")
                for node in nodes_of_type(build.api, "ImageScale")
            )
        )
        decode_id = next(
            node_id for node_id, node in build.api.items() if node["class_type"] == "VAEDecode"
        )
        combine = nodes_of_type(build.api, "VHS_VideoCombine")[0]
        self.assertEqual(combine["inputs"]["images"], [decode_id, 0])

    def test_i2v_mandatory_loras_are_always_first_and_exact(self) -> None:
        build = build_i2v_api_workflow(
            self.job,
            self.scene,
            Path(r"D:\output\10minfinals\.work\20260724-1610\frames\scene_0001.png"),
        )
        loras = nodes_of_type(build.api, "LoraLoaderModelOnly")
        self.assertEqual(
            (loras[0]["inputs"]["lora_name"], loras[0]["inputs"]["strength_model"]),
            ("LTX2.3_DMD_hybrid_v2.safetensors", 1.0),
        )
        self.assertNotIn(
            "JoyAI-Echo-content_r256.safetensors",
            [node["inputs"]["lora_name"] for node in loras],
        )

    def test_i2v_quarantines_t2i_alias_and_loads_only_validated_dynamic_ltx(self) -> None:
        raw = payload()
        raw["scenes"][0]["i2v"]["loras"] = [
            {
                "name": "Character alias",
                "download_url": "https://civitai.com/api/download/models/3184055",
                "weight": 0.8,
            },
            {
                "name": "LTX motion",
                "download_url": "https://civitai.com/api/download/models/3082662",
                "weight": 0.7,
            },
        ]
        job = parse_job_payload(raw)
        motion = job.scenes[0].i2v.loras[1]
        build = build_i2v_api_workflow(
            job,
            job.scenes[0],
            Path(r"D:\output\10minfinals\.work\job\frames\scene_0001.png"),
            {f"i2v:{lora_identity(motion)}": "ltx/Motion.safetensors"},
        )
        loras = nodes_of_type(build.api, "LoraLoaderModelOnly")
        self.assertEqual(
            [node["inputs"]["lora_name"] for node in loras],
            [
                "LTX2.3_DMD_hybrid_v2.safetensors",
                "ltx/Motion.safetensors",
            ],
        )
        self.assertNotIn(
            "Character",
            " ".join(node["_meta"]["title"] for node in loras),
        )

    def test_i2v_refuses_unvalidated_dynamic_lora_mapping(self) -> None:
        raw = payload()
        raw["scenes"][0]["i2v"]["loras"] = [
            {
                "name": "Unvalidated motion",
                "download_url": "https://civitai.com/api/download/models/3082662",
                "weight": 0.7,
            }
        ]
        job = parse_job_payload(raw)
        with self.assertRaisesRegex(WorkflowBuildError, "has not been verified"):
            build_i2v_api_workflow(
                job,
                job.scenes[0],
                Path(r"D:\output\10minfinals\.work\job\frames\scene_0001.png"),
            )

    def test_resolved_comfy_filename_is_used_for_dynamic_lora(self) -> None:
        resolved_name = r"characters\Elsa-canonical.safetensors"
        build = build_t2i_api_workflow(
            self.job,
            self.scene,
            {lora_identity(self.job.character.lora): resolved_name},
        )
        lora = nodes_of_type(build.api, "LoraLoader")[0]
        self.assertEqual(lora["inputs"]["lora_name"], resolved_name)

    def test_graph_validator_reports_dangling_references(self) -> None:
        errors = validate_api_graph(
            {
                "1": {
                    "class_type": "PreviewImage",
                    "inputs": {"images": ["missing", 0]},
                }
            }
        )
        self.assertEqual(errors, ("node 1.images references missing node missing",))

    def test_live_contract_validator_checks_required_inputs_and_types(self) -> None:
        graph = {
            "1": {"class_type": "Source", "inputs": {}},
            "2": {"class_type": "Sink", "inputs": {"value": ["1", 0]}},
        }
        object_info = {
            "Source": {"input": {"required": {}}, "output": ["STRING"]},
            "Sink": {"input": {"required": {"value": ["INT"], "label": ["STRING"]}}, "output": []},
        }
        self.assertEqual(
            validate_against_object_info(graph, object_info),
            (
                "node 2 (Sink) is missing required input label",
                "node 2.value expects INT, but node 1 slot 0 outputs STRING",
            ),
        )

    def test_live_contract_validator_rejects_unknown_inputs_and_combo_literals(self) -> None:
        graph = {
            "1": {
                "class_type": "Choice",
                "inputs": {"mode": "removed-value", "extra": 1},
            }
        }
        object_info = {
            "Choice": {
                "input": {
                    "required": {
                        "mode": [["current-value", "other-value"]],
                    }
                },
                "output": [],
            }
        }
        self.assertEqual(
            validate_against_object_info(graph, object_info),
            (
                "node 1.mode has invalid combo value 'removed-value'",
                "node 1 (Choice) has unknown input extra",
            ),
        )

    def test_live_contract_validator_supports_modern_combo_schema(self) -> None:
        graph = {
            "1": {
                "class_type": "Choice",
                "inputs": {"mode": "current-value"},
            }
        }
        object_info = {
            "Choice": {
                "input": {
                    "required": {
                        "mode": [
                            "COMBO",
                            {"options": ["current-value", "other-value"]},
                        ],
                    }
                },
                "output": [],
            }
        }
        self.assertEqual(validate_against_object_info(graph, object_info), ())
        graph["1"]["inputs"]["mode"] = "removed-value"
        self.assertEqual(
            validate_against_object_info(graph, object_info),
            ("node 1.mode has invalid combo value 'removed-value'",),
        )

    def test_live_contract_validator_rejects_scalar_type_and_bounds(self) -> None:
        graph = {
            "1": {
                "class_type": "LTXVAddGuide",
                "inputs": {
                    "frame_idx": 10_000,
                    "strength": "1.0",
                    "enabled": 1,
                    "finite": float("nan"),
                },
            }
        }
        object_info = {
            "LTXVAddGuide": {
                "input": {
                    "required": {
                        "frame_idx": ["INT", {"min": 0, "max": 9_999}],
                        "strength": ["FLOAT", {"min": 0.0, "max": 1.0}],
                        "enabled": ["BOOLEAN"],
                        "finite": ["FLOAT", {"min": 0.0, "max": 1.0}],
                    }
                },
                "output": [],
            }
        }

        self.assertEqual(
            validate_against_object_info(graph, object_info),
            (
                "node 1.frame_idx value 10000 is above maximum 9999",
                "node 1.strength expects a literal FLOAT, got str",
                "node 1.enabled expects a literal BOOLEAN, got int",
                "node 1.finite expects a finite FLOAT, got nan",
            ),
        )

    def test_live_contract_validator_supports_vhs_format_fields(self) -> None:
        graph = {
            "1": {
                "class_type": "VHS_VideoCombine",
                "inputs": {
                    "format": "video/h264-mp4",
                    "pix_fmt": "yuv420p",
                    "crf": 19,
                },
            }
        }
        object_info = {
            "VHS_VideoCombine": {
                "input": {
                    "required": {
                        "format": [
                            ["video/h264-mp4"],
                            {
                                "formats": {
                                    "video/h264-mp4": [
                                        ["pix_fmt", ["yuv420p", "yuv420p10le"]],
                                        ["crf", "INT", {"default": 19}],
                                    ]
                                }
                            },
                        ]
                    }
                },
                "output": [],
            }
        }
        self.assertEqual(validate_against_object_info(graph, object_info), ())
        del graph["1"]["inputs"]["crf"]
        self.assertEqual(
            validate_against_object_info(graph, object_info),
            (
                "node 1 (VHS_VideoCombine) is missing required input crf",
            ),
        )


if __name__ == "__main__":
    unittest.main()
