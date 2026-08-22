from __future__ import annotations

import json
import math
import unittest

from tenminvideomaker.qc_feedback import (
    HUMAN_QC_FEEDBACK_VERSION,
    HumanQcFeedbackError,
    canonicalize_human_qc_feedback,
    parse_human_qc_feedback_note,
)


class HumanQcFeedbackTests(unittest.TestCase):
    def test_canonicalizes_structured_feedback_with_server_owned_fields(self) -> None:
        note = canonicalize_human_qc_feedback(
            {
                "category": "policy_mismatch",
                "note": "  Intentional visual design.  ",
                "playback_timestamp_seconds": 6.426,
            },
            action_context="automatic_hold_override",
        )

        self.assertEqual(
            note,
            '{"action_context":"automatic_hold_override","category":"policy_mismatch",'
            '"note":"Intentional visual design.","playback_timestamp_seconds":6.43,'
            '"version":"human_qc_feedback_v1"}',
        )
        parsed = json.loads(note)
        self.assertEqual(parsed["version"], HUMAN_QC_FEEDBACK_VERSION)

    def test_no_feedback_is_null_except_required_audit_context(self) -> None:
        self.assertIsNone(
            canonicalize_human_qc_feedback(None, action_context="pass_approval")
        )
        self.assertEqual(
            canonicalize_human_qc_feedback(
                None,
                action_context="automatic_hold_override",
                force_record=True,
            ),
            '{"action_context":"automatic_hold_override","category":null,'
            '"note":null,"playback_timestamp_seconds":null,'
            '"version":"human_qc_feedback_v1"}',
        )

    def test_rejects_unknown_fields_invalid_categories_and_invalid_timestamps(self) -> None:
        with self.assertRaisesRegex(HumanQcFeedbackError, "Unsupported feedback field"):
            canonicalize_human_qc_feedback(
                {"version": "client-controlled"}, action_context="pass_approval"
            )
        with self.assertRaisesRegex(HumanQcFeedbackError, "Unknown feedback category"):
            canonicalize_human_qc_feedback(
                {"category": "wrong"}, action_context="pass_approval"
            )
        for value in (-1, math.nan, math.inf, "4.2"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    HumanQcFeedbackError, "playback_timestamp_seconds"
                ):
                    canonicalize_human_qc_feedback(
                        {"playback_timestamp_seconds": value},
                        action_context="pass_approval",
                    )

    def test_rejects_overlong_notes_and_normalizes_blank_values(self) -> None:
        with self.assertRaisesRegex(HumanQcFeedbackError, "at most 4000"):
            canonicalize_human_qc_feedback(
                {"note": "x" * 4001}, action_context="pass_hold"
            )
        self.assertIsNone(
            canonicalize_human_qc_feedback(
                {"category": None, "note": "   ", "playback_timestamp_seconds": None},
                action_context="pass_hold",
            )
        )

    def test_parses_structured_null_and_legacy_notes_without_mutating_history(self) -> None:
        structured = canonicalize_human_qc_feedback(
            {"category": "agree_with_qwen", "note": "Correct quarantine."},
            action_context="pass_hold",
        )
        self.assertEqual(
            parse_human_qc_feedback_note(structured),
            {
                "version": "human_qc_feedback_v1",
                "action_context": "pass_hold",
                "category": "agree_with_qwen",
                "note": "Correct quarantine.",
                "playback_timestamp_seconds": None,
                "legacy": False,
            },
        )
        self.assertEqual(
            parse_human_qc_feedback_note(None),
            {
                "version": None,
                "action_context": None,
                "category": None,
                "note": None,
                "playback_timestamp_seconds": None,
                "legacy": False,
            },
        )
        self.assertEqual(
            parse_human_qc_feedback_note("manual_override_of_automatic_hold"),
            {
                "version": None,
                "action_context": None,
                "category": None,
                "note": "manual_override_of_automatic_hold",
                "playback_timestamp_seconds": None,
                "legacy": True,
            },
        )


if __name__ == "__main__":
    unittest.main()
