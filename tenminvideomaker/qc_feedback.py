"""Validated human QC feedback stored in the existing decision note column."""

from __future__ import annotations

import json
import math
from typing import Any, Mapping


HUMAN_QC_FEEDBACK_VERSION = "human_qc_feedback_v1"
QC_FEEDBACK_EXPORT_VERSION = "qc_feedback_export_v1"
MAX_HUMAN_QC_NOTE_LENGTH = 4000
ALLOWED_HUMAN_QC_CATEGORIES = frozenset(
    {
        "agree_with_qwen",
        "policy_mismatch",
        "qwen_perception_error",
        "qwen_missed_defect",
        "acceptable_minor_issue",
        "other",
    }
)
_ALLOWED_FEEDBACK_FIELDS = frozenset(
    {"category", "note", "playback_timestamp_seconds"}
)


class HumanQcFeedbackError(ValueError):
    """Raised when operator-supplied QC feedback violates the stable contract."""


def canonicalize_human_qc_feedback(
    feedback: Mapping[str, Any] | None,
    *,
    action_context: str,
    force_record: bool = False,
) -> str | None:
    """Return deterministic JSON for durable storage, or ``None`` when empty."""
    if not isinstance(action_context, str) or not action_context.strip():
        raise HumanQcFeedbackError("action_context must be non-empty text.")
    if feedback is None:
        feedback = {}
    if not isinstance(feedback, Mapping):
        raise HumanQcFeedbackError("feedback must be an object or null.")
    unknown = sorted(set(feedback) - _ALLOWED_FEEDBACK_FIELDS)
    if unknown:
        raise HumanQcFeedbackError(
            f"Unsupported feedback field: {', '.join(unknown)}."
        )

    category = feedback.get("category")
    if category is not None:
        if not isinstance(category, str):
            raise HumanQcFeedbackError("category must be text or null.")
        category = category.strip() or None
        if category is not None and category not in ALLOWED_HUMAN_QC_CATEGORIES:
            raise HumanQcFeedbackError(f"Unknown feedback category: {category}.")

    note = feedback.get("note")
    if note is not None:
        if not isinstance(note, str):
            raise HumanQcFeedbackError("note must be text or null.")
        note = note.strip() or None
        if note is not None and len(note) > MAX_HUMAN_QC_NOTE_LENGTH:
            raise HumanQcFeedbackError(
                f"note must contain at most {MAX_HUMAN_QC_NOTE_LENGTH} characters."
            )

    timestamp = feedback.get("playback_timestamp_seconds")
    if timestamp is not None:
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
            raise HumanQcFeedbackError(
                "playback_timestamp_seconds must be a finite nonnegative number or null."
            )
        timestamp = float(timestamp)
        if not math.isfinite(timestamp) or timestamp < 0:
            raise HumanQcFeedbackError(
                "playback_timestamp_seconds must be a finite nonnegative number or null."
            )
        timestamp = round(timestamp, 2)

    if not force_record and category is None and note is None and timestamp is None:
        return None

    document = {
        "action_context": action_context.strip(),
        "category": category,
        "note": note,
        "playback_timestamp_seconds": timestamp,
        "version": HUMAN_QC_FEEDBACK_VERSION,
    }
    return json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def parse_human_qc_feedback_note(note: str | None) -> dict[str, Any]:
    """Parse current structured notes while preserving legacy text verbatim."""
    empty = {
        "version": None,
        "action_context": None,
        "category": None,
        "note": None,
        "playback_timestamp_seconds": None,
        "legacy": False,
    }
    if note is None:
        return empty
    try:
        value = json.loads(note)
    except (json.JSONDecodeError, TypeError):
        return {**empty, "note": str(note), "legacy": True}
    if not isinstance(value, Mapping) or value.get("version") != HUMAN_QC_FEEDBACK_VERSION:
        return {**empty, "note": str(note), "legacy": True}
    return {
        "version": HUMAN_QC_FEEDBACK_VERSION,
        "action_context": value.get("action_context"),
        "category": value.get("category"),
        "note": value.get("note"),
        "playback_timestamp_seconds": value.get("playback_timestamp_seconds"),
        "legacy": False,
    }
