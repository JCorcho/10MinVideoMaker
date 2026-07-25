from __future__ import annotations

import unittest

from tenminvideomaker.artifacts import ArtifactError, scene_clip_path, scene_frame_path


class ArtifactTests(unittest.TestCase):
    def test_scene_paths_are_deterministic_and_project_scoped(self) -> None:
        self.assertEqual(
            str(scene_frame_path("20260724-1610", 3)),
            (
                r"D:\LTX_Supervisor_Storage\jobs\20260724-1610\scenes"
                r"\scene_0003\revisions\0001\frame.png"
            ),
        )
        self.assertEqual(
            str(scene_clip_path("20260724-1610", 3)),
            (
                r"D:\LTX_Supervisor_Storage\jobs\20260724-1610\scenes"
                r"\scene_0003\revisions\0001\video.mp4"
            ),
        )

    def test_artifact_paths_reject_traversal(self) -> None:
        with self.assertRaises(ArtifactError):
            scene_frame_path("../outside", 1)
        with self.assertRaises(ArtifactError):
            scene_clip_path("safe", 0)


if __name__ == "__main__":
    unittest.main()
