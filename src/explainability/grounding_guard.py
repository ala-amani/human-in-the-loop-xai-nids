"""Fail-closed numeric grounding checks for advisory LLM text.

These checks do not prove semantic accuracy. They prevent unsupported numerical
conversions from being displayed as facts and trigger a deterministic fallback.
Source numeric features and validator labels remain the authoritative values.
"""
from __future__ import annotations

import re
from typing import Any

_PERCENT = re.compile(r"(?<![\w.])(-?\d+(?:\.\d+)?)\s*(?:%|percent\b)", flags=re.I)
_NUMBER = re.compile(r"(?<![\w.])(-?\d+(?:\.\d+)?)(?![\w.])")
_UNSUPPORTED_CONVERSION = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:x|×|times)\s*(?:the\s+)?(?:mean|average)\b"
    r"|\b\d+(?:\.\d+)?\s*(?:%|percent)\s+(?:above|below|higher|lower)\b"
    r"|\b(?:above|below)\s+(?:the\s+)?(?:mean|average)\s+by\s+\d"
    r"|\b\d+(?:\.\d+)?\s*percentage[- ]points?\b",
    flags=re.I,
)
_UNGROUNDED_THRESHOLD = re.compile(r"\b(?:threshold|cutoff)\s*(?:of|at|is|=|:)\s*\d", flags=re.I)


def numeric_grounding_issues(text: str, event: Any, validation: Any | None) -> list[str]:
    """Identify numeric claims that cannot safely be inferred from the given event.

    This intentionally favors false alarms and conservative fallback over
    exposing invented or transformed numbers to analysts.
    """
    issues: list[str] = []
    if _UNSUPPORTED_CONVERSION.search(text):
        issues.append("unsupported_percentage_or_multiplier_conversion")
    if _UNGROUNDED_THRESHOLD.search(text):
        issues.append("invented_threshold")

    confidence_percent = 100.0 * float(event.confidence)
    allowed = [float(event.confidence), confidence_percent]
    for item in event.evidence:
        for key in ("observed_value", "contribution", "baseline"):
            value = getattr(item, key, None)
            if isinstance(value, (int, float)):
                allowed.append(float(value))
    if validation is not None:
        for key in ("reliability_score", "historical_precision", "evidence_agreement", "reviewed_alerts"):
            value = getattr(validation, key, None)
            if isinstance(value, (int, float)):
                allowed.append(float(value))

    for match in _PERCENT.finditer(text):
        value = float(match.group(1))
        context = text[max(0, match.start() - 65):min(len(text), match.end() + 65)].lower()
        # Only the classifier confidence may be expressed as a percentage by the LLM.
        # Other percentages are constructed, if needed, by the trusted UI.
        if abs(value - confidence_percent) > 0.061 or (
            "confidence" not in context or "reliability" in context
        ):
            issues.append("percentage_not_proven_by_classifier_input")

    for match in _NUMBER.finditer(text):
        value = float(match.group(1))
        if not any(abs(value - x) <= max(0.006, abs(x) * 0.001) for x in allowed):
            issues.append("numeric_value_not_in_supplied_evidence")
            break

    return sorted(set(issues))


def review_claim_issues(recommendation: str, validation: Any | None) -> list[str]:
    """Reject priority language contradicting the deterministic validator."""
    if validation is None:
        return []
    recommendation = recommendation.lower()
    priority = getattr(validation, "review_priority", None)
    if priority == "routine" and re.search(r"\b(?:high.priority|urgent|prioriti[sz]e|immediate)\b", recommendation):
        return ["review_priority_mismatch"]
    if priority == "high" and re.search(r"\b(?:routine|low.priority)\b", recommendation):
        return ["review_priority_mismatch"]
    return []
