from __future__ import annotations

import unittest


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


if __name__ == "__main__":
    unittest.main()
