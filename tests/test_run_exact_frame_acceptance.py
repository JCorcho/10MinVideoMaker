from __future__ import annotations

import unittest


class ExactFrameAcceptanceRunnerTests(unittest.TestCase):
    def test_defaults_to_a_three_chunk_accumulation_proof(self) -> None:
        from scripts.run_exact_frame_acceptance import argument_parser

        args = argument_parser().parse_args(
            [
                "--source-payload-file",
                r"D:\safe\source.json",
                "--source-frame",
                r"D:\safe\frame.png",
                "--source-scene-id",
                "1",
                "--run-id",
                "exact-frame-acceptance-20260731-130000",
            ]
        )

        self.assertEqual(args.duration_seconds, 15.0)


if __name__ == "__main__":
    unittest.main()
