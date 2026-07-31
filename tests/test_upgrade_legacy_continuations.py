from __future__ import annotations

import unittest


class UpgradeLegacyContinuationsTests(unittest.TestCase):
    def test_upgrade_is_dry_run_unless_apply_is_explicit(self) -> None:
        from scripts.upgrade_legacy_continuations import argument_parser

        parser = argument_parser()
        self.assertFalse(parser.parse_args([]).apply)
        self.assertTrue(parser.parse_args(["--apply"]).apply)


if __name__ == "__main__":
    unittest.main()
