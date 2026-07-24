from __future__ import annotations

import unittest

from tenminvideomaker.artifacts import ArtifactError, scene_clip_path, scene_frame_path


class ArtifactTests(unittest.TestCase):
    def test_scene_paths_are_deterministic_and_project_scoped(self) -> None:
        self.assertEqual(
            str(scene_frame_path("20260724-1610", 3)),
            r"D:\output\10minfinals\.work\20260724-1610\frames\scene_0003.png",
        )
        self.assertEqual(
            str(scene_clip_path("20260724-1610", 3)),
            r"D:\output\10minfinals\.work\20260724-1610\clips\scene_0003.mp4",
        )

    def test_artifact_paths_reject_traversal(self) -> None:
        with self.assertRaises(ArtifactError):
            scene_frame_path("../outside", 1)
        with self.assertRaises(ArtifactError):
            scene_clip_path("safe", 0)


if __name__ == "__main__":
    unittest.main()
