"""Strict contracts and deterministic normalization for production VLM QC."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence


class QcDecision(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNCERTAIN = "UNCERTAIN"


class QcTier(StrEnum):
    ORIGINAL = "ORIGINAL"
    A1 = "A1"
    B1 = "B1"


class QcCandidateState(StrEnum):
    PENDING_GENERATION = "PENDING_GENERATION"
    GENERATING = "GENERATING"
    PENDING_QC = "PENDING_QC"
    QC_RUNNING = "QC_RUNNING"
    PASS_PENDING_HUMAN = "PASS_PENDING_HUMAN"
    ACCEPTED = "ACCEPTED"
    HOLD_FOR_REVIEW = "HOLD_FOR_REVIEW"
    SUPERSEDED = "SUPERSEDED"


class QcHumanDecision(StrEnum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    HOLD = "HOLD"


@dataclass(frozen=True)
class QcEvidencePolicy:
    minimum_severity: int = 3
    minimum_confidence: float = 0.85
    minimum_strong_windows: int = 2

    def __post_init__(self) -> None:
        if isinstance(self.minimum_severity, bool) or not 1 <= self.minimum_severity <= 5:
            raise ValueError("minimum_severity must be between 1 and 5.")
        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be between 0 and 1.")
        if self.minimum_strong_windows < 2:
            raise ValueError("minimum_strong_windows must be at least two.")


def _finite_number(value: Any, field: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric.")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{field} is outside its allowed range.")
    return result


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be text.")
    return value


@dataclass(frozen=True)
class QcError:
    category: str
    severity: int
    confidence: float
    start_time_seconds: float
    end_time_seconds: float
    description: str
    evidence: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "QcError":
        required = {
            "category",
            "severity",
            "confidence",
            "start_time_seconds",
            "end_time_seconds",
            "description",
            "evidence",
        }
        if set(value) != required:
            raise ValueError("Each error must contain exactly the versioned QC error fields.")
        severity = value["severity"]
        if isinstance(severity, bool) or not isinstance(severity, int) or not 1 <= severity <= 5:
            raise ValueError("error.severity must be an integer between 1 and 5.")
        start = _finite_number(value["start_time_seconds"], "error.start_time_seconds", 0.0, float("inf"))
        end = _finite_number(value["end_time_seconds"], "error.end_time_seconds", 0.0, float("inf"))
        if end < start:
            raise ValueError("error.end_time_seconds cannot precede its start time.")
        return cls(
            category=_text(value["category"], "error.category"),
            severity=severity,
            confidence=_finite_number(value["confidence"], "error.confidence", 0.0, 1.0),
            start_time_seconds=start,
            end_time_seconds=end,
            description=_text(value["description"], "error.description"),
            evidence=_text(value["evidence"], "error.evidence"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "category": self.category,
            "severity": self.severity,
            "confidence": self.confidence,
            "start_time_seconds": self.start_time_seconds,
            "end_time_seconds": self.end_time_seconds,
            "description": self.description,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class JudgeResponse:
    decision: QcDecision | None
    confidence: float | None
    summary: str | None
    errors: tuple[QcError, ...]
    raw_text: str
    parse_status: str
    parse_error: str | None = None
    model_decision: QcDecision | None = None
    decision_reconciliation: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "decision": self.decision.value if self.decision else None,
            "confidence": self.confidence,
            "summary": self.summary,
            "errors": [item.to_dict() for item in self.errors],
            "raw_text": self.raw_text,
            "parse_status": self.parse_status,
            "parse_error": self.parse_error,
            "model_decision": self.model_decision.value if self.model_decision else None,
            "decision_reconciliation": self.decision_reconciliation,
        }


def _malformed(raw_text: str, error: str) -> JudgeResponse:
    return JudgeResponse(None, None, None, (), raw_text, "malformed", error)


def parse_judge_response(raw_text: str) -> JudgeResponse:
    """Parse one judge response while preserving every raw byte as text evidence."""
    if not isinstance(raw_text, str):
        raise TypeError("raw_text must be a string.")
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start < 0 or end <= start:
        return _malformed(raw_text, "No JSON object found.")
    try:
        payload = json.loads(raw_text[start : end + 1])
        if not isinstance(payload, dict):
            raise ValueError("Top-level JSON value is not an object.")
        if set(payload) != {"decision", "confidence", "summary", "errors"}:
            raise ValueError("Response must contain exactly the versioned top-level fields.")
        model_decision = QcDecision(payload["decision"])
        confidence = _finite_number(payload["confidence"], "confidence", 0.0, 1.0)
        summary = _text(payload["summary"], "summary")
        raw_errors = payload["errors"]
        if not isinstance(raw_errors, list):
            raise ValueError("errors must be an array.")
        errors = tuple(
            QcError.from_mapping(item)
            if isinstance(item, Mapping)
            else (_ for _ in ()).throw(ValueError("Each error must be an object."))
            for item in raw_errors
        )
    except (ValueError, TypeError, json.JSONDecodeError) as error:
        return _malformed(raw_text, str(error))

    decision = model_decision
    reconciliation = None
    if errors and model_decision != QcDecision.FAIL:
        decision = QcDecision.FAIL
        reconciliation = (
            f"Model reported {len(errors)} defect(s) but labeled the result "
            f"{model_decision.value}; final decision normalized to FAIL."
        )
    elif not errors and model_decision == QcDecision.FAIL:
        decision = QcDecision.UNCERTAIN
        reconciliation = (
            "Model labeled the result FAIL but supplied no defect evidence; "
            "final decision normalized to UNCERTAIN."
        )
    return JudgeResponse(
        decision,
        confidence,
        summary,
        errors,
        raw_text,
        "parsed",
        None,
        model_decision,
        reconciliation,
    )


def is_strong_evidence(error: QcError, policy: QcEvidencePolicy) -> bool:
    return bool(
        error.severity >= policy.minimum_severity
        and error.confidence >= policy.minimum_confidence
        and error.evidence.strip()
    )


@dataclass(frozen=True)
class JudgeWindowResult:
    window_number: int
    source_frame_indices: tuple[int, ...]
    timestamps_seconds: tuple[float, ...]
    response: JudgeResponse
    confirmation_of_window: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.window_number, bool) or self.window_number < 1:
            raise ValueError("window_number must be positive.")
        if not self.source_frame_indices or len(self.source_frame_indices) > 4:
            raise ValueError("A QC window must contain one to four frames.")
        if len(self.source_frame_indices) != len(self.timestamps_seconds):
            raise ValueError("Frame indices and timestamps must have equal lengths.")
        if tuple(sorted(self.source_frame_indices)) != self.source_frame_indices:
            raise ValueError("Source frame indices must remain chronological.")
        if tuple(sorted(self.timestamps_seconds)) != self.timestamps_seconds:
            raise ValueError("Timestamps must remain chronological.")
        if any(value < 0 for value in self.timestamps_seconds):
            raise ValueError("Timestamps cannot be negative.")

    def as_confirmation_of(self, window_number: int) -> "JudgeWindowResult":
        if window_number < 1:
            raise ValueError("The confirmed window number must be positive.")
        return replace(self, confirmation_of_window=window_number)

    @property
    def strong_errors(self) -> tuple[QcError, ...]:
        return self.response.errors


@dataclass(frozen=True)
class NormalizedEvaluation:
    decision: QcDecision
    windows: tuple[JudgeWindowResult, ...]
    confirmation: JudgeWindowResult | None
    strong_window_count: int
    suspect_window_numbers: tuple[int, ...]
    suspect_time_ranges: tuple[tuple[float, float], ...]
    strong_errors: tuple[QcError, ...]
    automatic_early_fail: bool
    normalization_reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "decision": self.decision.value,
            "strong_window_count": self.strong_window_count,
            "suspect_window_numbers": list(self.suspect_window_numbers),
            "suspect_time_ranges": [list(item) for item in self.suspect_time_ranges],
            "strong_errors": [item.to_dict() for item in self.strong_errors],
            "automatic_early_fail": self.automatic_early_fail,
            "normalization_reason": self.normalization_reason,
        }


def _window_strong_errors(
    window: JudgeWindowResult, policy: QcEvidencePolicy
) -> tuple[QcError, ...]:
    return tuple(item for item in window.response.errors if is_strong_evidence(item, policy))


def normalize_window_results(
    windows: Sequence[JudgeWindowResult],
    confirmation: JudgeWindowResult | None,
    policy: QcEvidencePolicy,
) -> NormalizedEvaluation:
    ordered = tuple(windows)
    if not ordered:
        raise ValueError("At least one judge window is required.")
    if tuple(item.window_number for item in ordered) != tuple(
        sorted(item.window_number for item in ordered)
    ):
        raise ValueError("Judge windows must be supplied chronologically.")

    normal_strong = tuple(
        (item, _window_strong_errors(item, policy))
        for item in ordered
        if _window_strong_errors(item, policy)
    )
    confirming = False
    confirmation_errors: tuple[QcError, ...] = ()
    if confirmation is not None:
        if len(normal_strong) != 1:
            raise ValueError("Shifted confirmation is valid only for one suspect window.")
        suspect = normal_strong[0][0]
        if confirmation.confirmation_of_window != suspect.window_number:
            raise ValueError("Shifted confirmation must identify its suspect window.")
        if confirmation.source_frame_indices == suspect.source_frame_indices:
            raise ValueError("Confirmation must use a shifted frame window.")
        confirmation_errors = _window_strong_errors(confirmation, policy)
        confirming = bool(confirmation_errors)

    strong_count = len(normal_strong) + int(confirming)
    suspects = [item for item, _ in normal_strong]
    if confirming and confirmation is not None:
        suspects.append(confirmation)
    suspect_numbers = tuple(item.window_number for item in suspects)
    suspect_ranges = tuple(
        (item.timestamps_seconds[0], item.timestamps_seconds[-1]) for item in suspects
    )
    strong_errors = tuple(
        error for _, errors in normal_strong for error in errors
    ) + confirmation_errors
    automatic_early_fail = len(normal_strong) >= policy.minimum_strong_windows

    if strong_count >= policy.minimum_strong_windows:
        decision = QcDecision.FAIL
        reason = (
            "Two distinct normal windows supplied strong evidence."
            if automatic_early_fail
            else "A shifted fresh review independently confirmed the lone suspect window."
        )
    elif normal_strong:
        if confirmation is not None and confirmation.response.decision == QcDecision.PASS:
            remaining = tuple(
                item.response.decision
                for item in ordered
                if item.window_number != normal_strong[0][0].window_number
            )
            decision = (
                QcDecision.UNCERTAIN
                if QcDecision.UNCERTAIN in remaining or None in remaining
                else QcDecision.PASS
            )
            reason = "Shifted review did not confirm the lone strong suspect."
        else:
            decision = QcDecision.UNCERTAIN
            reason = "One strong suspect window lacks independent confirmation."
    else:
        decisions = tuple(item.response.decision for item in ordered)
        if confirmation is not None:
            decisions += (confirmation.response.decision,)
        if None in decisions or QcDecision.UNCERTAIN in decisions:
            decision = QcDecision.UNCERTAIN
            reason = "At least one window was uncertain or malformed."
        elif decisions and all(item == QcDecision.PASS for item in decisions):
            decision = QcDecision.PASS
            reason = "All inspected windows passed without strong defect evidence."
        else:
            decision = QcDecision.UNCERTAIN
            reason = "Window results did not support a trusted PASS or FAIL."
    return NormalizedEvaluation(
        decision=decision,
        windows=ordered,
        confirmation=confirmation,
        strong_window_count=strong_count,
        suspect_window_numbers=suspect_numbers,
        suspect_time_ranges=suspect_ranges,
        strong_errors=strong_errors,
        automatic_early_fail=automatic_early_fail,
        normalization_reason=reason,
    )


def canonical_json(value: Mapping[str, Any] | Sequence[Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_sha256(value: Mapping[str, Any] | Sequence[Any]) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def derive_retry_seed(
    *,
    job_id: str,
    scene_id: int,
    source_revision: int,
    original_seed: int,
    tier: QcTier,
    used_seeds: Iterable[int] = (),
) -> int:
    """Derive a stable uint64 seed, retrying only deterministic collisions."""
    if tier not in {QcTier.A1, QcTier.B1}:
        raise ValueError("Retry seeds are defined only for A1 and B1.")
    if not job_id or isinstance(scene_id, bool) or scene_id < 1:
        raise ValueError("A stable job and positive scene identity are required.")
    if isinstance(source_revision, bool) or source_revision < 1:
        raise ValueError("source_revision must be positive.")
    if isinstance(original_seed, bool) or not 0 <= original_seed <= 2**64 - 1:
        raise ValueError("original_seed must be an unsigned 64-bit integer.")
    excluded = {original_seed, *used_seeds}
    counter = 0
    while True:
        identity = canonical_json(
            {
                "job_id": job_id,
                "scene_id": scene_id,
                "source_revision": source_revision,
                "original_seed": str(original_seed),
                "tier": tier.value,
                "collision_counter": counter,
            }
        ).encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(identity).digest()[:8], "big")
        if seed not in excluded:
            return seed
        counter += 1
