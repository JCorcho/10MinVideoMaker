from __future__ import annotations

import unittest


class ContinuationWorkflowValidationScriptTests(unittest.TestCase):
    def test_builds_initial_later_final_and_decoded_guide_graphs(self) -> None:
        from tenminvideomaker.continuation_renderer import (
            CONTINUATION_CONTRACT_NODE_TYPES,
        )
        from scripts.validate_continuation_workflows import (
            build_representative_workflows,
        )

        workflows = build_representative_workflows()

        self.assertEqual(
            set(workflows),
            {
                "stage1_initial",
                "stage1_later",
                "stage1_final",
                "stage2_initial",
                "stage2_decoded_guide",
                "stage2_later",
                "stage2_final",
                "decode",
                "delivery",
            },
        )
        classes = {
            name: {node["class_type"] for node in workflow.values()}
            for name, workflow in workflows.items()
        }
        self.assertNotIn("LTXVExtendSampler", classes["stage1_initial"])
        self.assertIn("LTXVExtendSampler", classes["stage1_later"])
        self.assertIn("LTXVExtendSampler", classes["stage1_final"])
        self.assertIn("LTXVAddGuide", classes["stage2_decoded_guide"])
        self.assertIn("10MinVideoMaker_LoadChunkLatent", classes["decode"])
        self.assertNotIn("SamplerCustom", classes["decode"])
        self.assertIn("DiscordSendSaveVideo", classes["delivery"])
        self.assertEqual(
            {
                node_type
                for workflow_classes in classes.values()
                for node_type in workflow_classes
            },
            set(CONTINUATION_CONTRACT_NODE_TYPES),
        )

    def test_live_validation_only_reads_required_node_contracts(self) -> None:
        from scripts.validate_continuation_workflows import validate_live_workflows

        class FakeComfy:
            def __init__(self) -> None:
                self.queried = []

            def object_info(self, node_type):
                self.queried.append(node_type)
                return {
                    node_type: {
                        "input": {
                            "required": {
                                "value": [["ok"], {"default": "ok"}],
                            }
                        },
                        "output": [],
                    }
                }

        comfy = FakeComfy()
        errors = validate_live_workflows(
            comfy,
            workflows={
                "sample": {
                    "1": {
                        "class_type": "ReadOnlyNode",
                        "inputs": {"value": "ok"},
                    }
                }
            },
        )

        self.assertEqual(errors, {})
        self.assertEqual(comfy.queried, ["ReadOnlyNode"])

    def test_live_validation_reports_prompt_contract_errors(self) -> None:
        from scripts.validate_continuation_workflows import validate_live_workflows

        class FakeComfy:
            @staticmethod
            def object_info(node_type):
                return {
                    node_type: {
                        "input": {
                            "required": {
                                "value": [["allowed"], {"default": "allowed"}],
                            }
                        },
                        "output": [],
                    }
                }

        errors = validate_live_workflows(
            FakeComfy(),
            workflows={
                "invalid": {
                    "1": {
                        "class_type": "ReadOnlyNode",
                        "inputs": {"value": "rejected"},
                    }
                }
            },
        )

        self.assertIn("invalid", errors)
        self.assertTrue(
            any("invalid combo value" in error for error in errors["invalid"])
        )


if __name__ == "__main__":
    unittest.main()
