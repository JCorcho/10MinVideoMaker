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


if __name__ == "__main__":
    unittest.main()
