from __future__ import annotations

import unittest

from tenminvideomaker.nodes import (
    NODE_CLASS_MAPPINGS,
    TenMinPipelineStatusNode,
    TenMinValidateJobNode,
)

from test_contracts import payload


class NodeSurfaceTests(unittest.TestCase):
    def test_all_project_nodes_are_registered(self) -> None:
        self.assertEqual(
            set(NODE_CLASS_MAPPINGS),
            {
                "10MinVideoMaker_ValidateJob",
                "10MinVideoMaker_PipelineStatus",
                "10MinVideoMaker_RequestGrokJob",
                "10MinVideoMaker_PollGmail",
                "10MinVideoMaker_ResolveLoras",
                "10MinVideoMaker_ReleaseMemory",
                "10MinVideoMaker_SaveSceneFrame",
                "10MinVideoMaker_StitchClips",
            },
        )

    def test_validate_node_exposes_fixed_production_profile(self) -> None:
        import json

        result = TenMinValidateJobNode().execute(json.dumps(payload()))
        self.assertEqual(result[0], "20260724-1610")
        self.assertEqual(result[2:], (1, 704, 1248, 24))

    def test_status_node_is_forced_to_reexecute(self) -> None:
        self.assertNotEqual(
            TenMinPipelineStatusNode.IS_CHANGED(),
            TenMinPipelineStatusNode.IS_CHANGED(),
        )


if __name__ == "__main__":
    unittest.main()
