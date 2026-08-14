from __future__ import annotations

import json
import unittest

from tenminvideomaker.qc_contracts import (
    JudgeWindowResult,
    QcCandidateState,
    QcDecision,
    QcError,
    QcEvidencePolicy,
    QcHumanDecision,
    QcTier,
    derive_retry_seed,
    is_strong_evidence,
    normalize_window_results,
    parse_judge_response,
)


def response(
    decision: str,
    *,
    errors: list[dict[str, object]] | None = None,
    confidence: float = 0.9,
) -> str:
    return json.dumps(
        {
            "decision": decision,
            "confidence": confidence,
            "summary": "window result",
            "errors": errors or [],
        }
    )


def error(
    *, severity: int = 3, confidence: float = 0.85, evidence: str = "visible fusion"
) -> dict[str, object]:
    return {
        "category": "topology",
        "severity": severity,
        "confidence": confidence,
        "start_time_seconds": 0.5,
        "end_time_seconds": 1.0,
        "description": "two visible boundaries fuse",
        "evidence": evidence,
    }


def window(number: int, raw: str, start: float | None = None) -> JudgeWindowResult:
    start = float(number - 1) * 2.0 if start is None else start
    return JudgeWindowResult(
        window_number=number,
        source_frame_indices=(number * 4 - 4, number * 4 - 3, number * 4 - 2, number * 4 - 1),
        timestamps_seconds=(start, start + 0.5, start + 1.0, start + 1.5),
        response=parse_judge_response(raw),
    )


class QcContractTests(unittest.TestCase):
    def test_enums_round_trip_and_invalid_decision_is_rejected(self) -> None:
        for enum_type in (
            QcDecision,
            QcTier,
            QcCandidateState,
            QcHumanDecision,
        ):
            for member in enum_type:
                self.assertEqual(enum_type(member.value), member)
        with self.assertRaises(ValueError):
            QcDecision("ACCEPT")

    def test_valid_pass_and_uncertain_parse(self) -> None:
        passed = parse_judge_response(response("PASS"))
        uncertain = parse_judge_response(response("UNCERTAIN", confidence=0.4))

        self.assertEqual(passed.decision, QcDecision.PASS)
        self.assertEqual(uncertain.decision, QcDecision.UNCERTAIN)
        self.assertEqual(passed.parse_status, "parsed")

    def test_pass_with_reported_error_reconciles_to_fail(self) -> None:
        parsed = parse_judge_response(response("PASS", errors=[error()]))

        self.assertEqual(parsed.model_decision, QcDecision.PASS)
        self.assertEqual(parsed.decision, QcDecision.FAIL)
        self.assertIn("normalized to FAIL", parsed.decision_reconciliation or "")

    def test_bare_fail_becomes_uncertain_and_cannot_be_strong(self) -> None:
        raw = response("FAIL")
        parsed = parse_judge_response(raw)

        self.assertEqual(parsed.model_decision, QcDecision.FAIL)
        self.assertEqual(parsed.decision, QcDecision.UNCERTAIN)
        self.assertEqual(parsed.raw_text, raw)
        normalized = normalize_window_results((window(1, raw),), None, QcEvidencePolicy())
        self.assertEqual(normalized.decision, QcDecision.UNCERTAIN)
        self.assertEqual(normalized.strong_window_count, 0)

    def test_malformed_and_refusal_results_remain_auditable(self) -> None:
        for raw in ("not json", "I cannot review this content."):
            with self.subTest(raw=raw):
                parsed = parse_judge_response(raw)
                self.assertEqual(parsed.raw_text, raw)
                self.assertEqual(parsed.parse_status, "malformed")
                self.assertIsNone(parsed.decision)
                normalized = normalize_window_results(
                    (window(1, raw),), None, QcEvidencePolicy()
                )
                self.assertEqual(normalized.decision, QcDecision.UNCERTAIN)

    def test_schema_bounds_and_required_fields_are_strict(self) -> None:
        cases = (
            '{"decision":"NOPE","confidence":0.9,"summary":"x","errors":[]}',
            '{"decision":"PASS","confidence":1.1,"summary":"x","errors":[]}',
            '{"decision":"PASS","confidence":0.9,"errors":[]}',
            response("FAIL", errors=[{**error(), "severity": 6}]),
            response("FAIL", errors=[{**error(), "confidence": -0.1}]),
            response("FAIL", errors=[{**error(), "start_time_seconds": 2.0, "end_time_seconds": 1.0}]),
        )
        for raw in cases:
            with self.subTest(raw=raw):
                self.assertEqual(parse_judge_response(raw).parse_status, "malformed")

    def test_strong_evidence_requires_all_three_thresholds(self) -> None:
        policy = QcEvidencePolicy()
        self.assertTrue(is_strong_evidence(QcError.from_mapping(error()), policy))
        self.assertFalse(
            is_strong_evidence(QcError.from_mapping(error(severity=2)), policy)
        )
        self.assertFalse(
            is_strong_evidence(QcError.from_mapping(error(confidence=0.84)), policy)
        )
        self.assertFalse(
            is_strong_evidence(QcError.from_mapping(error(evidence="  ")), policy)
        )

    def test_one_strong_window_needs_shifted_confirmation(self) -> None:
        suspect = window(1, response("FAIL", errors=[error()]))
        passed = window(2, response("PASS"))

        unconfirmed = normalize_window_results(
            (suspect, passed), None, QcEvidencePolicy()
        )
        self.assertEqual(unconfirmed.decision, QcDecision.UNCERTAIN)
        self.assertEqual(unconfirmed.strong_window_count, 1)

        shifted_pass = window(99, response("PASS"), start=0.5).as_confirmation_of(1)
        cleared = normalize_window_results(
            (suspect, passed), shifted_pass, QcEvidencePolicy()
        )
        self.assertEqual(cleared.decision, QcDecision.PASS)

    def test_two_independent_strong_windows_fail_early(self) -> None:
        first = window(1, response("FAIL", errors=[error()]))
        second = window(2, response("FAIL", errors=[error()]))

        normalized = normalize_window_results(
            (first, second), None, QcEvidencePolicy()
        )

        self.assertEqual(normalized.decision, QcDecision.FAIL)
        self.assertTrue(normalized.automatic_early_fail)
        self.assertEqual(normalized.strong_window_count, 2)
        self.assertEqual(normalized.suspect_window_numbers, (1, 2))

    def test_shifted_strong_confirmation_is_independent_evidence(self) -> None:
        suspect = window(2, response("FAIL", errors=[error()]))
        shifted = window(99, response("FAIL", errors=[error()]), start=1.0).as_confirmation_of(2)

        normalized = normalize_window_results(
            (window(1, response("PASS")), suspect), shifted, QcEvidencePolicy()
        )

        self.assertEqual(normalized.decision, QcDecision.FAIL)
        self.assertEqual(normalized.strong_window_count, 2)
        self.assertFalse(normalized.automatic_early_fail)

    def test_window_order_and_timestamps_are_preserved(self) -> None:
        windows = (window(1, response("PASS")), window(2, response("UNCERTAIN")))
        normalized = normalize_window_results(windows, None, QcEvidencePolicy())

        self.assertEqual(normalized.decision, QcDecision.UNCERTAIN)
        self.assertEqual(normalized.windows, windows)
        self.assertEqual(normalized.windows[1].timestamps_seconds, (2.0, 2.5, 3.0, 3.5))

    def test_retry_seed_is_deterministic_unique_and_tier_separated(self) -> None:
        kwargs = {
            "job_id": "job-1",
            "scene_id": 2,
            "source_revision": 1,
            "original_seed": 18446744073709551615,
        }
        a1 = derive_retry_seed(**kwargs, tier=QcTier.A1)
        self.assertEqual(a1, derive_retry_seed(**kwargs, tier=QcTier.A1))
        self.assertNotEqual(a1, kwargs["original_seed"])
        self.assertNotEqual(a1, derive_retry_seed(**kwargs, tier=QcTier.B1))
        self.assertGreaterEqual(a1, 0)
        self.assertLessEqual(a1, 2**64 - 1)


if __name__ == "__main__":
    unittest.main()
