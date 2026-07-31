from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch


class RunContinuationAcceptanceScriptTests(unittest.TestCase):
    def test_acceptance_runner_parses_source_and_timeout_options(self) -> None:
        from scripts.run_continuation_acceptance import argument_parser

        args = argument_parser().parse_args(
            [
                "--source-job-id",
                "source-job",
                "--source-scene-id",
                "7",
                "--timeout-seconds",
                "900",
            ]
        )

        self.assertEqual(args.source_job_id, "source-job")
        self.assertEqual(args.source_scene_id, 7)
        self.assertEqual(args.timeout_seconds, 900.0)
        self.assertFalse(args.dry_run)

    def test_acceptance_runner_accepts_file_source(self) -> None:
        from scripts.run_continuation_acceptance import argument_parser

        args = argument_parser().parse_args(
            [
                "--source-payload-file",
                r"D:\safe\source.json",
                "--source-frame",
                r"D:\safe\frame.png",
                "--source-scene-id",
                "1",
            ]
        )

        self.assertIsNone(args.source_job_id)
        self.assertEqual(args.source_payload_file, Path(r"D:\safe\source.json"))
        self.assertEqual(args.source_frame, Path(r"D:\safe\frame.png"))

    def test_decoded_guide_constants_select_a_17_frame_base_span(self) -> None:
        from scripts import run_continuation_acceptance as acceptance

        self.assertEqual(acceptance.ACCEPTANCE_SCHEMA_VERSION, 4)
        self.assertEqual(acceptance.DIAGNOSTIC_GUIDE_FRAME_COUNT, 17)
        self.assertEqual(acceptance.BASE_DIAGNOSTIC_GUIDE_START_FRAME_INDEX, 96)

    def test_case_metrics_use_requested_guide_frame_span(self) -> None:
        from scripts import run_continuation_acceptance as acceptance

        def extracted(_video: Path, _frame: int, destination: Path) -> Path:
            return destination

        with (
            patch.object(acceptance, "_extract_frame", side_effect=extracted) as extract,
            patch.object(acceptance, "_probe_video", return_value={}),
            patch.object(acceptance, "_image_difference", return_value={}),
            patch.object(acceptance, "_flow_discontinuity", return_value={}),
        ):
            acceptance._case_metrics(
                run_root=Path(r"D:\LTX_Supervisor_Storage\acceptance\run"),
                base_video=Path(r"D:\base.mkv"),
                case_name="decoded_17_frame",
                case_video=Path(r"D:\case.mkv"),
                guide_start_frame_index=96,
                guide_frame_count=17,
            )

        self.assertEqual(
            [call.args[1] for call in extract.call_args_list],
            [96, 111, 112, 0, 16, 17],
        )

    def test_latent_overlap_metrics_use_the_direct_handoff_production_seam(self) -> None:
        from scripts import run_continuation_acceptance as acceptance

        def extracted(_video: Path, _frame: int, destination: Path) -> Path:
            return destination

        with (
            patch.object(acceptance, "_extract_frame", side_effect=extracted) as extract,
            patch.object(acceptance, "_probe_video", return_value={}),
            patch.object(acceptance, "_image_difference", return_value={}),
            patch.object(acceptance, "_flow_discontinuity", return_value={}),
        ):
            report = acceptance._case_metrics(
                run_root=Path(r"D:\LTX_Supervisor_Storage\acceptance\run"),
                base_video=Path(r"D:\base.mkv"),
                case_name="latent_overlap",
                case_video=Path(r"D:\case.mkv"),
                guide_start_frame_index=96,
                guide_frame_count=25,
            )

        self.assertEqual(
            [call.args[1] for call in extract.call_args_list],
            [102, 103, 8, 9],
        )
        self.assertEqual(
            report["production_seam"],
            {"base_end_frame": 103, "continuation_start_frame": 8},
        )


if __name__ == "__main__":
    unittest.main()
