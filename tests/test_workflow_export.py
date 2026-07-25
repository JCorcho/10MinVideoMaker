from __future__ import annotations

import unittest

from tenminvideomaker.workflow_export import api_to_gui_workflow, inspect_gui_workflow


class WorkflowExportTests(unittest.TestCase):
    def test_layout_has_no_overlaps_and_group_covers_all_nodes(self) -> None:
        api = {
            "1": {"class_type": "Source", "_meta": {"title": "Source A"}, "inputs": {"value": 1}},
            "2": {"class_type": "Source", "_meta": {"title": "Source B"}, "inputs": {"value": 2}},
            "3": {
                "class_type": "Merge",
                "_meta": {"title": "Merge"},
                "inputs": {"left": ["1", 0], "right": ["2", 0]},
            },
        }
        info = {
            "Source": {
                "input": {"required": {"value": ["INT", {"default": 0}]}, "optional": {}},
                "input_order": {"required": ["value"]},
                "output": ["INT"],
                "output_name": ["value"],
            },
            "Merge": {
                "input": {
                    "required": {"left": ["INT"], "right": ["INT"]},
                    "optional": {},
                },
                "input_order": {"required": ["left", "right"]},
                "output": ["INT"],
                "output_name": ["value"],
            },
        }
        workflow = api_to_gui_workflow(api, info, title="Test group")
        inspection = inspect_gui_workflow(workflow)
        self.assertEqual(inspection["overlaps"], [])
        self.assertEqual(inspection["out_of_group"], [])
        self.assertEqual(len(workflow["links"]), 2)
        self.assertEqual(workflow["groups"][0]["title"], "Test group")

    def test_sampler_seed_control_widget_preserves_canvas_field_alignment(self) -> None:
        api = {
            "1": {
                "class_type": "KSamplerAdvanced",
                "inputs": {
                    "model": ["2", 0],
                    "add_noise": "enable",
                    "noise_seed": 123,
                    "steps": 30,
                    "cfg": 6.0,
                    "sampler_name": "res_3m_ode",
                    "scheduler": "karras",
                    "positive": ["3", 0],
                    "negative": ["4", 0],
                    "latent_image": ["5", 0],
                    "start_at_step": 0,
                    "end_at_step": 30,
                    "return_with_leftover_noise": "disable",
                },
            },
            "2": {"class_type": "Model", "inputs": {}},
            "3": {"class_type": "Conditioning", "inputs": {}},
            "4": {"class_type": "Conditioning", "inputs": {}},
            "5": {"class_type": "Latent", "inputs": {}},
        }
        info = {
            "KSamplerAdvanced": {
                "input": {
                    "required": {
                        "model": ["MODEL"],
                        "add_noise": [["enable", "disable"]],
                        "noise_seed": ["INT", {"control_after_generate": True}],
                        "steps": ["INT"],
                        "cfg": ["FLOAT"],
                        "sampler_name": [["res_3m_ode"]],
                        "scheduler": [["karras"]],
                        "positive": ["CONDITIONING"],
                        "negative": ["CONDITIONING"],
                        "latent_image": ["LATENT"],
                        "start_at_step": ["INT"],
                        "end_at_step": ["INT"],
                        "return_with_leftover_noise": [["disable", "enable"]],
                    }
                },
                "input_order": {"required": list(api["1"]["inputs"])},
                "output": ["LATENT"],
                "output_name": ["LATENT"],
            },
            "Model": {"input": {"required": {}}, "output": ["MODEL"]},
            "Conditioning": {"input": {"required": {}}, "output": ["CONDITIONING"]},
            "Latent": {"input": {"required": {}}, "output": ["LATENT"]},
        }
        workflow = api_to_gui_workflow(api, info, title="Sampler widgets")
        sampler = next(node for node in workflow["nodes"] if node["type"] == "KSamplerAdvanced")
        self.assertEqual(
            sampler["widgets_values"],
            ["enable", 123, "fixed", 30, 6.0, "res_3m_ode", "karras", 0, 30, "disable"],
        )

    def test_face_detailer_seed_control_preserves_canvas_field_alignment(self) -> None:
        api = {
            "1": {
                "class_type": "FaceDetailer",
                "inputs": {"seed": 456, "steps": 20, "cfg": 5.0},
            }
        }
        info = {
            "FaceDetailer": {
                "input": {
                    "required": {
                        "seed": ["INT"],
                        "steps": ["INT"],
                        "cfg": ["FLOAT"],
                    }
                },
                "input_order": {"required": ["seed", "steps", "cfg"]},
                "output": [],
            }
        }
        workflow = api_to_gui_workflow(api, info, title="Detailer widgets")
        detailer = workflow["nodes"][0]
        self.assertEqual(detailer["widgets_values"], [456, "fixed", 20, 5.0])


if __name__ == "__main__":
    unittest.main()
