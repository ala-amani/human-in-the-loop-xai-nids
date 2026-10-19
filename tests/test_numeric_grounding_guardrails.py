"""Numeric grounding regression tests based on the recorded Qwen3 failure mode."""
import json
from dataclasses import asdict, replace
from pathlib import Path

from src.explainability.generative_explainer import (
    AnalystNarrative, AttackEvent, FeatureEvidence, OllamaNarrativeClient,
    _check_narrative, _verified_fallback,
)
from src.explainability.grounding_guard import numeric_grounding_issues, review_claim_issues
from src.pipeline.upstream_nsl_rf_ai import format_llm_commentary
from src.validation.human_feedback import ValidationAssessment


def event_and_review():
    event = AttackEvent("DoS", 0.86, [
        FeatureEvidence("dst_host_srv_count", 1.77, 0.24, "supports prediction", 0.0),
        FeatureEvidence("dst_host_rerror_rate", -0.23, -0.08, "opposes prediction", 0.0),
        FeatureEvidence("dst_host_diff_srv_rate", 0.11, 0.03, "supports prediction", 0.0),
    ], "fixture-dos")
    assessment = ValidationAssessment(
        reliability_score=0.757, verdict="likely_correct", evidence_agreement=0.77,
        historical_precision=0.5, reviewed_alerts=0, review_priority="routine", reasons=[],
    )
    return event, assessment


def base_narrative():
    return AnalystNarrative(
        title="DoS alert for human review",
        summary="The classifier reported a possible service-disruption pattern.",
        observed_signs=["Some SHAP features support the prediction while others oppose it."],
        reasoning="SHAP values reflect the prediction, not causality.",
        confidence_note="Classifier confidence does not prove an attack.",
        recommended_checks=["Inspect network logs and maintenance records."],
        limitations="Raw packets and network addresses are not supplied.",
        validation_verdict="Likely correct according to the heuristic validator, not verified ground truth.",
        review_recommendation="The validator assigns routine review; do not block automatically.",
        human_review_question="Is this prediction correct, a false positive, or a wrong attack class?",
    )


def test_archived_incorrect_standardized_conversions_rejected():
    event, validation = event_and_review()
    wrong = "Recent connections (1.77x average); rejection rate (23% below mean)"
    issues = numeric_grounding_issues(wrong, event, validation)
    assert "unsupported_percentage_or_multiplier_conversion" in issues


def test_unsupported_probability_and_review_priority_rejected():
    event, validation = event_and_review()
    original = base_narrative()
    bad = replace(
        original,
        observed_signs=[*original.observed_signs, "Rejected errors were 23 percent lower than the mean."],
        review_recommendation="Prioritize urgent review with high priority.",
    )
    issues = _check_narrative(bad, event, validation)
    assert "unsupported_percentage_or_multiplier_conversion" in issues
    assert "review_priority_mismatch" in issues


def test_readable_deterministic_fallback_uses_exact_evidence():
    event, validation = event_and_review()
    explanation = _verified_fallback(event, validation)
    rendered = format_llm_commentary(explanation)
    assert explanation.source == "deterministic_fallback"
    assert "VERIFIED RULE-BASED FALLBACK" in rendered
    assert "1.770" in rendered and "-0.230" in rendered
    assert "23% below" not in rendered
    assert "review priority: routine" in rendered
    assert "not a calibrated probability" in rendered


def test_verified_classifier_confidence_allowed():
    event, validation = event_and_review()
    assert numeric_grounding_issues("Classifier confidence: 86%. This does not prove an attack.", event, validation) == []


def test_conflicting_or_invented_numeric_percent_rejected():
    event, validation = event_and_review()
    assert "percentage_not_proven_by_classifier_input" in numeric_grounding_issues(
        "Reliability 75.7% means the alert is probably correct.", event, validation
    )
    assert "numeric_value_not_in_supplied_evidence" in numeric_grounding_issues(
        "A threshold of 0.99 was exceeded.", event, validation
    )


def test_llm_client_fails_closed_on_misleading_generated_narrative(monkeypatch):
    event, validation = event_and_review()
    bad = base_narrative()
    bad.observed_signs.append("Rejected errors are 23% below mean.")
    raw = asdict(bad)
    raw.pop("source", None)
    response = {"model": "qwen3:4b", "message": {"content": json.dumps(raw)}}

    class Response:
        def __enter__(self): return self
        def __exit__(self, *_): return False
        def read(self): return json.dumps(response).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: Response())
    client = OllamaNarrativeClient()
    result = client.explain(event, validation)
    assert result.source == "deterministic_fallback"
    assert "unsupported_percentage_or_multiplier_conversion" in client.last_grounding_issues
    assert "23%" not in format_llm_commentary(result)


def test_llm_client_does_not_replace_valid_nonnumeric_explanation(monkeypatch):
    event, validation = event_and_review()
    raw = asdict(base_narrative())
    raw.pop("source", None)
    response = {"model": "qwen3:4b", "message": {"content": json.dumps(raw)}}

    class Response:
        def __enter__(self): return self
        def __exit__(self, *_): return False
        def read(self): return json.dumps(response).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: Response())
    client = OllamaNarrativeClient()
    result = client.explain(event, validation)
    assert result.source == "llm"
    assert client.last_grounding_issues == []


def test_interactive_chat_withholds_unsupported_numeric_claims(monkeypatch):
    event, validation = event_and_review()
    text = "The rejection error rate is 23 percent below average."
    response = {"model": "qwen3:4b", "message": {"content": text}}

    class Response:
        def __enter__(self): return self
        def __exit__(self, *_): return False
        def read(self): return json.dumps(response).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: Response())
    client = OllamaNarrativeClient()
    answer = client.chat(event, validation, [], "Identify the SHAP features and causality.")
    assert "withheld" in answer
    assert "23 percent" not in answer
    assert "dst_host_srv_count" in answer
    assert client.last_raw_chat_content == text


def test_missing_review_history_is_not_called_empirical_precision():
    event, validation = event_and_review()
    narrative = _verified_fallback(event, validation)
    assert "prior, not measured precision" in narrative.confidence_note

def test_exact_archived_qwen_dos_error_is_rejected():
    """Regression: exercise the actual recorded erroneous DoS explanation."""
    file = (Path(__file__).resolve().parents[1]
            / "experiments/2026-10-19-live-ollama-audit/run_outputs_guarded/live_model_outputs.json")
    record = json.loads(file.read_text(encoding="utf-8"))
    entry = next(item for item in record["records"] if item["scenario"] == "wifi_access_point_syn_burst")
    raw = entry["event"]
    event = AttackEvent(
        predicted_class=raw["predicted_class"], confidence=raw["confidence"],
        evidence=[FeatureEvidence(**row) for row in raw["evidence"]], flow_id=raw["flow_id"]
    )
    validation = ValidationAssessment(**entry["assessment"])
    generated = next(item["structured"] for item in entry["responses"]
                     if item["kind"] == "structured_narrative")
    narrative = AnalystNarrative(**generated)
    issues = _check_narrative(narrative, event, validation)
    assert "unsupported_percentage_or_multiplier_conversion" in issues
    assert "review_priority_mismatch" in issues


def test_interactive_chat_withholds_inconsistent_review_priority(monkeypatch):
    event, validation = event_and_review()
    raw_answer = "This alert requires urgent high priority review immediately."
    response = {"model": "qwen3:4b", "message": {"content": raw_answer}}

    class Response:
        def __enter__(self): return self
        def __exit__(self, *_): return False
        def read(self): return json.dumps(response).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: Response())
    client = OllamaNarrativeClient()
    answer = client.chat(event, validation, [], "Is the alert proven?")
    assert "withheld" in answer
    assert "urgent high priority" not in answer
    assert "review_priority_mismatch" in client.last_chat_grounding_issues
