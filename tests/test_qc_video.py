from __future__ import annotations

from pathlib import Path
import hashlib
import json
import tempfile
from types import SimpleNamespace
import unittest

from tenminvideomaker.qc_video import (
    _benchmark_resize,
    _encode_benchmark_jpeg,
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
    def test_frozen_lab_preprocessing_reference_matches_production(self) -> None:
        import cv2
        import numpy as np

        reference = json.loads(
            (
                Path(__file__).parent
                / "fixtures"
                / "qc_lab_preprocessing_reference_v1.json"
            ).read_text(encoding="utf-8")
        )
        source = reference["source_generator"]
        y, x = np.indices((source["height"], source["width"]), dtype=np.uint32)
        rgb = np.stack(
            (
                ((x * 3 + y * 5 + 17) % 256).astype(np.uint8),
                ((x * 7 + y * 11 + 29) % 256).astype(np.uint8),
                ((x * 13 + y * 17 + 43) % 256).astype(np.uint8),
            ),
            axis=2,
        )

        resized_rgb = _benchmark_resize(rgb)
        selection = select_frame_indices(
            reference["sampling"]["source_frame_count"],
            reference["sampling"]["source_fps"],
            reference["sampling"]["duration_seconds"],
            reference["sampling"]["target_fps"],
        )

        self.assertEqual(
            (resized_rgb.shape[1], resized_rgb.shape[0]),
            (reference["resize"]["width"], reference["resize"]["height"]),
        )
        self.assertEqual(
            hashlib.sha256(resized_rgb.tobytes()).hexdigest(),
            reference["resize"]["rgb_sha256"],
        )
        self.assertEqual(list(selection.indices), reference["sampling"]["indices"])
        self.assertEqual(
            list(selection.timestamps_seconds),
            reference["sampling"]["timestamps_seconds"],
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "synthetic-source.png"
            self.assertTrue(
                cv2.imwrite(str(source_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            )
            payload, width, height, digest = _encode_benchmark_jpeg(
                source_path, root / "production.jpg"
            )
        self.assertEqual((width, height), (512, 896))
        if cv2.__version__ == reference["opencv_version"]:
            self.assertEqual(len(payload), reference["jpeg"]["length"])
            self.assertEqual(digest, reference["jpeg"]["sha256"])

    def test_reviewed_lab_fixture_indices_timestamps_dimensions_and_windows(self) -> None:
        # Read-only lab artifact:
        # results/20260813T223649Z-718671/{video-metadata,frame-selection}.json
        selection = select_frame_indices(601, 24.0, 25.041667, 2.0)
        frames = tuple(
            SampledFrame(
                source_index=index,
                timestamp_seconds=timestamp,
                image_path=Path(f"frame-{position:06d}.jpg"),
                image_bytes=b"jpeg",
                width=512,
                height=896,
            )
            for position, (index, timestamp) in enumerate(
                zip(selection.indices, selection.timestamps_seconds), 1
            )
        )
        video = SampledVideo(
            VideoMetadata(24.0, 601, 25.041667),
            2.0,
            frames,
        )

        self.assertEqual(selection.indices, tuple(range(0, 601, 12)))
        self.assertEqual(
            selection.timestamps_seconds,
            tuple(index / 24 for index in range(0, 601, 12)),
        )
        self.assertEqual(len(frames), 51)
        self.assertTrue(all((item.width, item.height) == (512, 896) for item in frames))
        self.assertEqual(
            tuple(len(item.frames) for item in chronological_windows(video, frame_count=4)),
            (4,) * 12 + (3,),
        )

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
        self.assertEqual(
            accounting["confirmation_independence_rule"],
            "shifted_indices_and_at_least_one_new_image_sha256_required",
        )
        self.assertEqual(
            accounting["strong_window_independence_rule"],
            "each_additional_strong_window_requires_at_least_one_new_image_sha256",
        )
        self.assertTrue(accounting["confirmation_is_independent"])
        self.assertTrue(accounting["early_exit_applied"])

    def test_repeated_image_bytes_cannot_form_shifted_confirmation(self) -> None:
        video = sampled(12)
        repeated = b"identical-jpeg"
        frozen = SampledVideo(
            video.metadata,
            video.target_fps,
            tuple(
                SampledFrame(
                    item.source_index,
                    item.timestamp_seconds,
                    item.image_path,
                    repeated,
                )
                for item in video.frames
            ),
        )

        self.assertIsNone(
            shifted_confirmation_window(
                frozen,
                chronological_windows(frozen)[1],
                frame_count=4,
            )
        )

    def test_ffprobe_and_ffmpeg_wrapper_extracts_exact_selected_indices(self) -> None:
        import cv2
        import numpy as np

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
                y, x = np.indices((1344, 768))
                source = np.stack(
                    (
                        (x % 256).astype(np.uint8),
                        (y % 256).astype(np.uint8),
                        ((x + y) % 256).astype(np.uint8),
                    ),
                    axis=2,
                )
                for index in range(1, 3):
                    self.assertTrue(
                        cv2.imwrite(
                            str(output.with_name(f"source-{index:06d}.png")),
                            source,
                        )
                    )
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
            filter_index = commands[1].index("-vf") + 1
            self.assertIn("eq(n\\,0)+eq(n\\,12)", commands[1][filter_index])
            self.assertIn("-noautorotate", commands[1])
            self.assertEqual(
                commands[1][commands[1].index("-pix_fmt") + 1], "rgb24"
            )
            self.assertEqual(
                result.preprocessing,
                {
                    "version": "vlm-qc-lab-f634ca2-image-v1",
                    "validated_lab_commit": "f634ca2ab7ca95ddd9abde7fe840031eba0696f4",
                    "decoder": "ffmpeg_selected_png_rgb24",
                    "orientation": "encoded_pixels_no_autorotate",
                    "color_pipeline": "rgb24_to_opencv_bgr_to_jpeg",
                    "resize_interpolation": "opencv_inter_area",
                    "max_short_edge": 512,
                    "max_pixels": 458752,
                    "dimension_multiple": 16,
                    "jpeg_quality": 88,
                },
            )
            decoded = cv2.imread(str(result.frames[0].image_path), cv2.IMREAD_COLOR)
            self.assertEqual(decoded.shape[:2], (896, 512))
            self.assertEqual(result.frames[0].width, 512)
            self.assertEqual(result.frames[0].height, 896)

            original = cv2.imread(
                str(
                    result.frames[0].image_path.parent.parent
                    / "decoded"
                    / "source-000001.png"
                ),
                cv2.IMREAD_COLOR,
            )
            expected_pixels = cv2.resize(
                original, (512, 896), interpolation=cv2.INTER_AREA
            )
            ok, expected_jpeg = cv2.imencode(
                ".jpg",
                expected_pixels,
                [cv2.IMWRITE_JPEG_QUALITY, 88],
            )
            self.assertTrue(ok)
            self.assertEqual(result.frames[0].bytes(), expected_jpeg.tobytes())
            self.assertEqual(
                result.frames[0].image_sha256,
                hashlib.sha256(expected_jpeg.tobytes()).hexdigest(),
            )

            windows = chronological_windows(result, frame_count=4)
            self.assertEqual(tuple(len(item.frames) for item in windows), (2,))
            self.assertEqual(windows[0].source_frame_indices, (0, 12))


if __name__ == "__main__":
    unittest.main()
