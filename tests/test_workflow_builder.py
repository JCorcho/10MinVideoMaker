from __future__ import annotations

from pathlib import Path
import unittest

from tenminvideomaker.constants import (
    I2V_FIRST_PASS_SIGMAS,
    I2V_SPATIAL_UPSCALER,
    I2V_UPSCALE_PASS_SIGMAS,
)
from tenminvideomaker.contracts import lora_identity, parse_job_payload
from tenminvideomaker.workflow_builder import (
    I2V_BASE_HEIGHT,
    I2V_BASE_WIDTH,
    WorkflowBuildError,
    build_i2v_api_workflow,
    build_t2i_api_workflow,
    validate_against_object_info,
    validate_api_graph,
)

from test_contracts import payload


def nodes_of_type(workflow, class_type):
    return [node for node in workflow.values() if node["class_type"] == class_type]


class WorkflowBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.job = parse_job_payload(payload())
        self.scene = self.job.scenes[0]

    def test_anima_uses_reference_sampler_and_fixed_output_size(self) -> None:
        build = build_t2i_api_workflow(self.job, self.scene)
        self.assertEqual(validate_api_graph(build.api), ())
        sampler = nodes_of_type(build.api, "KSampler")[0]["inputs"]
        self.assertEqual((sampler["sampler_name"], sampler["scheduler"]), ("er_sde", "beta57"))
        self.assertEqual((sampler["steps"], sampler["cfg"]), (30, 4.5))
        latent = nodes_of_type(build.api, "EmptySD3LatentImage")[0]["inputs"]
        self.assertEqual((latent["width"], latent["height"]), (704, 1248))
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

    def test_i2v_has_two_lcm_passes_verified_sigmas_upscaler_and_chunking(self) -> None:
        build = build_i2v_api_workflow(
            self.job,
            self.scene,
            Path(r"D:\output\10minfinals\.work\20260724-1610\frames\scene_0001.png"),
        )
        self.assertEqual(validate_api_graph(build.api), ())
        samplers = nodes_of_type(build.api, "KSamplerSelect")
        self.assertEqual([node["inputs"]["sampler_name"] for node in samplers], ["lcm", "lcm"])
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
        combine = nodes_of_type(build.api, "VHS_VideoCombine")[0]["inputs"]
        self.assertEqual(combine["frame_rate"], 24.0)
        self.assertFalse(combine["save_output"])
        self.assertEqual(build.api[build.output_node_id]["class_type"], "VHS_VideoCombine")

    def test_i2v_normalizes_decoded_frames_to_exact_production_geometry(self) -> None:
        build = build_i2v_api_workflow(
            self.job,
            self.scene,
            Path(r"D:\output\10minfinals\.work\20260724-1610\frames\scene_0001.png"),
        )
        normalized = next(
            node
            for node in nodes_of_type(build.api, "ImageScale")
            if node.get("_meta", {}).get("title")
            == "Normalize decoded video to exact 704x1248"
        )
        self.assertEqual(
            (
                normalized["inputs"]["width"],
                normalized["inputs"]["height"],
                normalized["inputs"]["upscale_method"],
                normalized["inputs"]["crop"],
            ),
            (704, 1248, "lanczos", "center"),
        )
        decode_id = normalized["inputs"]["image"][0]
        self.assertEqual(build.api[decode_id]["class_type"], "VAEDecode")
        combine = nodes_of_type(build.api, "VHS_VideoCombine")[0]
        self.assertEqual(combine["inputs"]["images"][0], next(
            node_id for node_id, node in build.api.items() if node is normalized
        ))

    def test_i2v_mandatory_loras_are_always_first_and_exact(self) -> None:
        build = build_i2v_api_workflow(
            self.job,
            self.scene,
            Path(r"D:\output\10minfinals\.work\20260724-1610\frames\scene_0001.png"),
        )
        loras = nodes_of_type(build.api, "LoraLoaderModelOnly")
        self.assertEqual(
            [(node["inputs"]["lora_name"], node["inputs"]["strength_model"]) for node in loras[:2]],
            [
                ("LTX2.3_DMD_reshaped_r256.safetensors", 1.0),
                ("JoyAI-Echo-content_r256.safetensors", 0.5),
            ],
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
                "LTX2.3_DMD_reshaped_r256.safetensors",
                "JoyAI-Echo-content_r256.safetensors",
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


if __name__ == "__main__":
    unittest.main()
