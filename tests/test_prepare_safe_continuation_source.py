from __future__ import annotations

import unittest


class PrepareSafeContinuationSourceTests(unittest.TestCase):
    def test_parser_has_bounded_project_owned_defaults(self) -> None:
        from scripts.prepare_safe_continuation_source import argument_parser

        args = argument_parser().parse_args([])

        self.assertEqual(args.timeout_seconds, 900.0)
        self.assertEqual(args.revision, 1)
        self.assertTrue(args.payload.name.endswith("safe_continuation_source.json"))
        self.assertEqual(
            args.local_lora_filename,
            "Tsunade_-_Naruto_-_Anima_LORA.safetensors",
        )


if __name__ == "__main__":
    unittest.main()
