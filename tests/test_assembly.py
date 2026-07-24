from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import subprocess
import tempfile
import unittest

from tenminvideomaker.assembly import AssemblyError, FfmpegAssembler, VideoStreamInfo, concat_list_text, validate_video_profile


class AssemblyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.clip = self.root / "scene 1.mp4"
        self.clip.write_bytes(b"not-a-real-video")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_profile_accepts_only_fixed_geometry_and_rate(self) -> None:
        stream = VideoStreamInfo(self.clip, 704, 1248, Fraction(24, 1))
        self.assertEqual(validate_video_profile([stream]), (stream,))
        with self.assertRaisesRegex(AssemblyError, "expected 704x1248"):
            validate_video_profile([VideoStreamInfo(self.clip, 720, 1248, Fraction(24, 1))])

    def test_concat_list_uses_absolute_escaped_paths(self) -> None:
        text = concat_list_text([self.clip])
        self.assertIn("file '", text)
        self.assertIn("scene 1.mp4", text)

    def test_stitch_uses_copy_concat_and_requires_output(self) -> None:
        output_root = self.root / "finals"

        def runner(command, **_kwargs):
            self.assertIn("-f", command)
            self.assertIn("concat", command)
            self.assertIn("copy", command)
            Path(command[-1]).write_bytes(b"stitched")
            return subprocess.CompletedProcess(command, 0, "", "")

        assembler = FfmpegAssembler(output_root, runner=runner)
        output = assembler.stitch("job-1", [self.clip], self.root / "runtime")
        self.assertEqual(output, output_root / "job-1_final.mp4")
        self.assertTrue(output.is_file())

    def test_final_path_rejects_traversal(self) -> None:
        with self.assertRaises(AssemblyError):
            FfmpegAssembler(self.root).final_path("../job")


if __name__ == "__main__":
    unittest.main()
