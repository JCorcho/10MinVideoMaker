from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import torch

from tenminvideomaker.nodes import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    TenMinLoadChunkLatentNode,
    TenMinPipelineStatusNode,
    TenMinResolveLorasNode,
    TenMinSaveChunkLatentNode,
    TenMinValidateJobNode,
)
from tenminvideomaker.storage import StorageLayout

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
                "10MinVideoMaker_SaveChunkLatent",
                "10MinVideoMaker_LoadChunkLatent",
                "10MinVideoMaker_IsolateConditioning",
                "10MinVideoMaker_IsolateModel",
                "10MinVideoMaker_StitchClips",
            },
        )
        self.assertEqual(
            set(NODE_CLASS_MAPPINGS),
            set(NODE_DISPLAY_NAME_MAPPINGS),
        )

    def test_validate_node_exposes_fixed_production_profile(self) -> None:
        import json

        result = TenMinValidateJobNode().execute(json.dumps(payload()))
        self.assertEqual(result[0], "20260724-1610")
        self.assertEqual(result[2:], (1, 768, 1344, 24))

    def test_status_node_is_forced_to_reexecute(self) -> None:
        self.assertNotEqual(
            TenMinPipelineStatusNode.IS_CHANGED(),
            TenMinPipelineStatusNode.IS_CHANGED(),
        )

    def test_conditioning_isolator_clones_tensors_and_never_reuses_cache(self) -> None:
        node_type = NODE_CLASS_MAPPINGS["10MinVideoMaker_IsolateConditioning"]
        source = [
            (
                torch.ones((1, 2)),
                {"pooled_output": torch.ones((1,)), "nested": {"value": 1}},
            )
        ]

        result = node_type().execute(source, "test scope")[0]

        self.assertIsNot(result, source)
        self.assertIsNot(result[0][0], source[0][0])
        self.assertIsNot(result[0][1], source[0][1])
        self.assertIsNot(result[0][1]["pooled_output"], source[0][1]["pooled_output"])
        result[0][0].zero_()
        self.assertTrue(torch.equal(source[0][0], torch.ones((1, 2))))
        self.assertNotEqual(
            node_type.IS_CHANGED(source, "test scope"),
            node_type.IS_CHANGED(source, "test scope"),
        )

    def test_model_isolator_clones_model_patcher_and_never_reuses_cache(self) -> None:
        class FakeModelPatcher:
            def clone(self):
                return FakeModelPatcher()

        node_type = NODE_CLASS_MAPPINGS["10MinVideoMaker_IsolateModel"]
        source = FakeModelPatcher()

        result = node_type().execute(source, "test scope")[0]

        self.assertIsInstance(result, FakeModelPatcher)
        self.assertIsNot(result, source)
        self.assertNotEqual(
            node_type.IS_CHANGED(source, "test scope"),
            node_type.IS_CHANGED(source, "test scope"),
        )

    def test_chunk_latent_nodes_round_trip_by_coordinates_and_hash(self) -> None:
        tests_root = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(dir=tests_root) as temporary:
            layout = StorageLayout(Path(temporary))
            latent = {
                "samples": torch.arange(
                    128 * 3 * 2 * 2,
                    dtype=torch.float32,
                ).reshape(1, 128, 3, 2, 2),
                "downscale_ratio_spacial": 32.0,
            }
            with patch("tenminvideomaker.nodes.STORAGE", layout):
                saved = TenMinSaveChunkLatentNode().execute(
                    latent,
                    "job-1",
                    2,
                    3,
                    4,
                    5,
                    "stage1_handoff",
                )
                self.assertIs(saved[0], latent)
                self.assertEqual(len(saved[2]), 64)
                self.assertEqual(
                    saved[1],
                    str(
                        layout.chunk_checkpoint_path(
                            "job-1",
                            2,
                            3,
                            4,
                            5,
                            "stage1_handoff",
                        )
                    ),
                )
                self.assertEqual(
                    TenMinLoadChunkLatentNode.IS_CHANGED(
                        "job-1",
                        2,
                        3,
                        4,
                        5,
                        "stage1_handoff",
                        3,
                    ),
                    f"{saved[2]}:3",
                )
                loaded = TenMinLoadChunkLatentNode().execute(
                    "job-1",
                    2,
                    3,
                    4,
                    5,
                    "stage1_handoff",
                    3,
                )

            torch.testing.assert_close(loaded[0]["samples"], latent["samples"])
            self.assertEqual(loaded[0]["downscale_ratio_spacial"], 32.0)
            self.assertEqual(loaded[1:], saved[1:])

    def test_load_chunk_latent_cache_rejects_corrupt_checkpoint(self) -> None:
        tests_root = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(dir=tests_root) as temporary:
            layout = StorageLayout(Path(temporary))
            latent = {"samples": torch.zeros((1, 128, 2, 1, 1))}
            with patch("tenminvideomaker.nodes.STORAGE", layout):
                saved = TenMinSaveChunkLatentNode().execute(
                    latent,
                    "job-2",
                    1,
                    1,
                    0,
                    1,
                )
                Path(saved[1]).write_bytes(Path(saved[1]).read_bytes() + b"corrupt")
                fingerprint = TenMinLoadChunkLatentNode.IS_CHANGED(
                    "job-2",
                    1,
                    1,
                    0,
                    1,
                )
                self.assertTrue(math.isnan(fingerprint))
                with self.assertRaisesRegex(RuntimeError, "byte size"):
                    TenMinLoadChunkLatentNode().execute(
                        "job-2",
                        1,
                        1,
                        0,
                        1,
                    )

    def test_chunk_latent_nodes_do_not_accept_arbitrary_paths(self) -> None:
        tests_root = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(dir=tests_root) as temporary:
            layout = StorageLayout(Path(temporary))
            with patch("tenminvideomaker.nodes.STORAGE", layout):
                with self.assertRaisesRegex(RuntimeError, "unsafe path"):
                    TenMinSaveChunkLatentNode().execute(
                        {"samples": torch.zeros((1, 128, 1, 1, 1))},
                        "../outside",
                        1,
                        1,
                        0,
                        1,
                    )
                self.assertTrue(
                    math.isnan(
                        TenMinLoadChunkLatentNode.IS_CHANGED(
                            "../outside",
                            1,
                            1,
                            0,
                            1,
                        )
                    )
                )

    def test_save_chunk_latent_is_output_node_and_always_reexecutes(self) -> None:
        self.assertTrue(TenMinSaveChunkLatentNode.OUTPUT_NODE)
        self.assertNotEqual(
            TenMinSaveChunkLatentNode.IS_CHANGED(),
            TenMinSaveChunkLatentNode.IS_CHANGED(),
        )

    def test_resolve_loras_uses_configured_storage_manifest(self) -> None:
        fake_folder_paths = SimpleNamespace(
            get_folder_paths=lambda _kind: [r"C:\models\loras"],
            get_filename_list=lambda _kind: [],
        )
        manager = MagicMock()
        manager.resolve_many.return_value = []
        with (
            patch.dict(sys.modules, {"folder_paths": fake_folder_paths}),
            patch(
                "tenminvideomaker.nodes.LoraAssetManager",
                return_value=manager,
            ) as manager_type,
            patch(
                "tenminvideomaker.nodes.load_project_environment",
                return_value={},
            ),
        ):
            result = TenMinResolveLorasNode().execute(
                json.dumps(payload()),
                "t2i",
            )

        self.assertEqual(result, ("[]", True))
        self.assertEqual(
            manager_type.call_args.args[1],
            StorageLayout.configured().asset_manifest_path,
        )


if __name__ == "__main__":
    unittest.main()
