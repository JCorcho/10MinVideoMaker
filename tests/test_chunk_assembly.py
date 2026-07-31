from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from tenminvideomaker.chunk_assembly import (
    SceneChunkAssembler,
    SceneChunkAssemblyError,
    chunk_slices,
)
from tenminvideomaker.continuation import (
    build_scene_frame_plan,
    refinement_raw_frame_count,
)


def _plan(seconds: float = 30.0):
    return build_scene_frame_plan(
        job_id="20260729-assembly",
        scene_id=1,
        revision=1,
        requested_duration_seconds=seconds,
        base_seed=1234,
        fallback_prompt="The same adult woman continues walking.",
        fallback_negative="body distortion",
    )


def _probe_json(
    frames: int,
    *,
    width: int = 768,
    height: int = 1344,
    audio: bool = True,
    output: bool = False,
    codec: str | None = None,
    pixel_format: str | None = None,
) -> str:
    streams = [
        {
            "codec_type": "video",
            "codec_name": codec or ("h264" if output else "ffv1"),
            "profile": "High" if output else "Main",
            "width": width,
            "height": height,
            "pix_fmt": pixel_format or ("yuv420p" if output else "yuv444p"),
            "r_frame_rate": "24/1",
            "avg_frame_rate": "24/1",
            "nb_read_frames": str(frames),
        }
    ]
    if audio:
        streams.append(
            {
                "codec_type": "audio",
                "codec_name": "aac",
                "sample_rate": "48000",
                "channels": 2,
            }
        )
    return json.dumps({"streams": streams})


class SceneChunkAssemblerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _chunks(self, plan):
        paths = []
        for chunk in plan.chunks:
            path = self.root / f"chunk_{chunk.index:04d}.mkv"
            path.write_bytes(b"raw")
            paths.append(path)
        return paths

    def _probe_runner(self, plan, temporary_frames: int | None = None):
        expected_by_name = {
            f"chunk_{chunk.index:04d}.mkv": refinement_raw_frame_count(chunk)
            for chunk in plan.chunks
        }

        def runner(command, **_kwargs):
            path = Path(command[-1])
            frames = expected_by_name.get(
                path.name,
                plan.timeline_output_frames
                if temporary_frames is None
                else temporary_frames,
            )
            return subprocess.CompletedProcess(
                command,
                0,
                _probe_json(frames, output=path.name not in expected_by_name),
                "",
            )

        return runner

    def test_30_second_command_uses_exact_spans_preroll_and_encode(self) -> None:
        plan = _plan()
        chunks = self._chunks(plan)
        destination = self.root / "scene.mp4"
        commands = []

        def ffmpeg_runner(command, **_kwargs):
            commands.append(command)
            Path(command[-1]).write_bytes(b"assembled")
            return subprocess.CompletedProcess(command, 0, "", "")

        assembler = SceneChunkAssembler(
            ffmpeg_runner=ffmpeg_runner,
            ffprobe_runner=self._probe_runner(plan),
        )
        self.assertEqual(assembler.assemble(plan, chunks, destination), destination)
        self.assertEqual(
            [(item.start_frame, item.end_frame_exclusive) for item in chunk_slices(plan)],
            [
                (0, 104),
                (8, 104),
                (8, 104),
                (8, 104),
                (8, 104),
                (8, 104),
                (8, 104),
                (8, 49),
            ],
        )
        self.assertEqual(
            [
                (item.audio_start_frame, item.audio_end_frame_exclusive)
                for item in chunk_slices(plan)
            ],
            [
                (0, 104),
                (16, 112),
                (16, 112),
                (16, 112),
                (16, 112),
                (16, 112),
                (16, 112),
                (16, 57),
            ],
        )

        command = commands[0]
        filters = command[command.index("-filter_complex") + 1]
        self.assertIn("[0:v]trim=start_frame=0:end_frame=104", filters)
        self.assertIn("[1:v]trim=start_frame=8:end_frame=104", filters)
        self.assertIn("[7:v]trim=start_frame=8:end_frame=49", filters)
        self.assertIn("[1:a]atrim=start=0.666666667:end=4.666666667", filters)
        self.assertIn("[7:a]atrim=start=0.666666667:end=2.375000000", filters)
        self.assertIn("concat=n=8:v=1:a=1[vcat][acat]", filters)
        self.assertIn("[vcat]trim=start_frame=0:end_frame=720", filters)
        self.assertIn("[acat]atrim=duration=30.000000000", filters)
        self.assertIn("afade=t=out", filters)
        self.assertIn("afade=t=in", filters)
        self.assertEqual(command[command.index("-crf") + 1], "19")
        self.assertEqual(command[command.index("-g") + 1], "48")
        self.assertEqual(command[command.index("-keyint_min") + 1], "48")
        self.assertEqual(command[command.index("-pix_fmt") + 1], "yuv420p")
        self.assertEqual(command[command.index("-c:a") + 1], "aac")
        self.assertTrue(destination.is_file())

    def test_ffmpeg_failure_preserves_existing_destination_atomically(self) -> None:
        plan = _plan(5.0)
        chunks = self._chunks(plan)
        destination = self.root / "scene.mp4"
        destination.write_bytes(b"previous-good-scene")

        def ffmpeg_runner(command, **_kwargs):
            return subprocess.CompletedProcess(command, 1, "", "encode failed")

        assembler = SceneChunkAssembler(
            ffmpeg_runner=ffmpeg_runner,
            ffprobe_runner=self._probe_runner(plan),
        )
        with self.assertRaisesRegex(SceneChunkAssemblyError, "encode failed"):
            assembler.assemble(plan, chunks, destination)
        self.assertEqual(destination.read_bytes(), b"previous-good-scene")
        self.assertEqual(list(self.root.glob(".*.assembling-*.mp4")), [])

    def test_missing_audio_and_wrong_input_profile_are_rejected(self) -> None:
        plan = _plan(5.0)
        chunks = self._chunks(plan)
        destination = self.root / "scene.mp4"
        ffmpeg_calls = []

        def ffmpeg_runner(command, **_kwargs):
            ffmpeg_calls.append(command)
            return subprocess.CompletedProcess(command, 0, "", "")

        def missing_audio(command, **_kwargs):
            return subprocess.CompletedProcess(
                command,
                0,
                _probe_json(121, audio=False),
                "",
            )

        with self.assertRaisesRegex(SceneChunkAssemblyError, "no audio stream"):
            SceneChunkAssembler(
                ffmpeg_runner=ffmpeg_runner,
                ffprobe_runner=missing_audio,
            ).assemble(plan, chunks, destination)

        def wrong_profile(command, **_kwargs):
            return subprocess.CompletedProcess(
                command,
                0,
                _probe_json(121, width=704),
                "",
            )

        with self.assertRaisesRegex(SceneChunkAssemblyError, "expected 768x1344"):
            SceneChunkAssembler(
                ffmpeg_runner=ffmpeg_runner,
                ffprobe_runner=wrong_profile,
            ).assemble(plan, chunks, destination)

        def lossy_chunk(command, **_kwargs):
            return subprocess.CompletedProcess(
                command,
                0,
                _probe_json(121, codec="h264", pixel_format="yuv420p"),
                "",
            )

        with self.assertRaisesRegex(SceneChunkAssemblyError, "lossless FFV1"):
            SceneChunkAssembler(
                ffmpeg_runner=ffmpeg_runner,
                ffprobe_runner=lossy_chunk,
            ).assemble(plan, chunks, destination)
        self.assertEqual(ffmpeg_calls, [])

    def test_exact_decoded_frame_validation_prevents_replacement(self) -> None:
        plan = _plan(5.0)
        chunks = self._chunks(plan)
        destination = self.root / "scene.mp4"
        destination.write_bytes(b"previous-good-scene")

        def ffmpeg_runner(command, **_kwargs):
            Path(command[-1]).write_bytes(b"wrong-frame-count")
            return subprocess.CompletedProcess(command, 0, "", "")

        assembler = SceneChunkAssembler(
            ffmpeg_runner=ffmpeg_runner,
            ffprobe_runner=self._probe_runner(
                plan,
                temporary_frames=plan.timeline_output_frames - 1,
            ),
        )
        with self.assertRaisesRegex(
            SceneChunkAssemblyError,
            "expected exactly 120",
        ):
            assembler.assemble(plan, chunks, destination)
        self.assertEqual(destination.read_bytes(), b"previous-good-scene")
        self.assertEqual(list(self.root.glob(".*.assembling-*.mp4")), [])


if __name__ == "__main__":
    unittest.main()
