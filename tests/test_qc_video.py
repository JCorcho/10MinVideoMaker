from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from tenminvideomaker.qc_video import (
    SampledFrame,
    SampledVideo,
    VideoMetadata,
    build_frame_accounting,
    chronological_windows,
    sample_video_frames,
    select_frame_indices,
    shifted_confirmation_window,
)


def sampled(count: int) -> SampledVideo:
    frames = tuple(
        SampledFrame(
            source_index=index * 12,
            timestamp_seconds=index * 0.5,
            image_path=Path(f"frame-{index}.jpg"),
            image_bytes=f"frame-{index}".encode("ascii"),
        )
        for index in range(count)
    )
    return SampledVideo(
        metadata=VideoMetadata(
            source_fps=24.0,
            source_frame_count=max(1, count * 12),
            duration_seconds=max(0.5, count * 0.5),
        ),
        target_fps=2.0,
        frames=frames,
    )


class QcVideoTests(unittest.TestCase):
    def test_exact_lab_two_fps_selection_is_deterministic(self) -> None:
        selection = select_frame_indices(240, 24.0, 10.0, 2.0)

        self.assertEqual(selection.indices, tuple(range(0, 240, 12)))
        self.assertEqual(selection.timestamps_seconds[1], 0.5)
        self.assertEqual(selection.effective_fps, 2.0)

    def test_selection_suppresses_duplicates_and_uses_actual_timestamps(self) -> None:
        selection = select_frame_indices(
            4,
            1.0,
            4.0,
            2.0,
            actual_timestamps=(0.0, 1.1, 2.2, 3.3),
        )

        self.assertEqual(selection.indices, (0, 1, 2, 3))
        self.assertEqual(selection.timestamps_seconds, (0.0, 1.1, 2.2, 3.3))

    def test_windows_are_chronological_and_keep_short_final_window(self) -> None:
        windows = chronological_windows(sampled(10), frame_count=4)

        self.assertEqual(tuple(len(item.frames) for item in windows), (4, 4, 2))
        self.assertEqual(windows[0].source_frame_indices, (0, 12, 24, 36))
        self.assertEqual(windows[-1].timestamps_seconds, (4.0, 4.5))

    def test_shifted_confirmation_matches_validated_overlap_formula(self) -> None:
        video = sampled(12)
        windows = chronological_windows(video, frame_count=4)

        shifted = shifted_confirmation_window(video, windows[1], frame_count=4)

        self.assertIsNotNone(shifted)
        assert shifted is not None
        self.assertEqual(shifted.source_frame_indices, (24, 36, 48, 60))
        self.assertEqual(shifted.confirmation_of_window, 2)
        self.assertNotEqual(shifted.source_frame_indices, windows[1].source_frame_indices)

    def test_short_clip_cannot_fabricate_independent_confirmation(self) -> None:
        video = sampled(4)
        suspect = chronological_windows(video, frame_count=4)[0]

        self.assertIsNone(
            shifted_confirmation_window(video, suspect, frame_count=4)
        )

    def test_frame_accounting_separates_unique_coverage_and_confirmation_exposure(self) -> None:
        video = sampled(12)
        planned = chronological_windows(video, frame_count=4)
        shifted = shifted_confirmation_window(video, planned[0], frame_count=4)

        accounting = build_frame_accounting(
            video,
            planned_windows=planned,
            processed_windows=planned[:2],
            confirmation=shifted,
            early_exit=True,
            early_exit_reason="two strong windows",
        )

        self.assertEqual(accounting["planned_window_count"], 3)
        self.assertEqual(accounting["processed_window_count"], 2)
        self.assertEqual(accounting["unique_selected_frames_inspected"], 8)
        self.assertEqual(accounting["confirmation_frame_exposures"], 4)
        self.assertEqual(accounting["frame_count_represented_in_model_input"], 12)
        self.assertTrue(accounting["early_exit_applied"])

    def test_ffprobe_and_ffmpeg_wrapper_extracts_exact_selected_indices(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "candidate.mp4"
            video.write_bytes(b"video")
            commands: list[list[str]] = []

            def runner(command, **kwargs):
                commands.append(list(command))
                if command[0] == "ffprobe-test":
                    frames = [
                        {"best_effort_timestamp_time": str(index / 24)}
                        for index in range(24)
                    ]
                    return SimpleNamespace(
                        returncode=0,
                        stdout=(
                            '{"streams":[{"avg_frame_rate":"24/1",'
                            '"r_frame_rate":"24/1","nb_frames":"24",'
                            '"duration":"1.0"}],"format":{"duration":"1.0"},'
                            f'"frames":{__import__("json").dumps(frames)}}}'
                        ),
                        stderr="",
                    )
                output = Path(command[-1])
                output.parent.mkdir(parents=True, exist_ok=True)
                for index in range(1, 3):
                    output.with_name(f"frame-{index:06d}.jpg").write_bytes(b"jpeg")
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            result = sample_video_frames(
                video,
                target_fps=2.0,
                ffprobe_command="ffprobe-test",
                ffmpeg_command="ffmpeg-test",
                temporary_root=root / "temp",
                run_command=runner,
            )

            self.assertEqual(
                tuple(frame.source_index for frame in result.frames), (0, 12)
            )
            self.assertIn("eq(n\\,0)+eq(n\\,12)", commands[1][7])


if __name__ == "__main__":
    unittest.main()
