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
        self.assertEqual(build.api[build.output_node_id]["class_type"], "10MinVideoMaker_SaveSceneFrame")

    def test_pony_uses_both_exact_reference_sampler_passes(self) -> None:
        raw = payload()
        raw["character"]["lora"]["base"] = "Pony"
        job = parse_job_payload(raw)
        build = build_t2i_api_workflow(job, job.scenes[0])
        samplers = nodes_of_type(build.api, "KSamplerAdvanced")
        self.assertEqual(
            [(node["inputs"]["sampler_name"], node["inputs"]["scheduler"]) for node in samplers],
            [("res_5s_ode", "karras"), ("res_3m_ode", "karras")],
        )
        self.assertTrue(all(node["inputs"]["steps"] == 30 for node in samplers))
        self.assertTrue(all(node["inputs"]["cfg"] == 6.0 for node in samplers))
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
