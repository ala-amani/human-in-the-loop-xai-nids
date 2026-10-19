"""Regression tests for LLM narrative visibility in the analyst interface."""

from src.explainability.generative_explainer import AnalystNarrative
from src.pipeline.upstream_nsl_rf_ai import format_llm_commentary


def test_generated_narrative_is_visible_in_ui_text() -> None:
    narrative = AnalystNarrative(
        title="Evidence requires review",
        summary="Model output has conflicting feature signals.",
        observed_signs=["Feature A supports", "Feature B opposes"],
        reasoning="Both signs should be verified.",
        confidence_note="Not a calibrated probability.",
        recommended_checks=["Check maintenance window", "Inspect router logs"],
        limitations="Traffic context is not supplied.",
        validation_verdict="uncertain",
        review_recommendation="high priority",
        human_review_question="Does an authorized scan explain this alert?",
    )
    text = format_llm_commentary(narrative)

    assert "LOCAL LLM COMMENTARY (GENERATED; ADVISORY)" in text
    assert "Model output has conflicting feature signals." in text
    assert "Feature A supports; Feature B opposes" in text
    assert "Check maintenance window; Inspect router logs" in text
    assert "Does an authorized scan explain this alert?" in text


def test_missing_narrative_is_explicit() -> None:
    text = format_llm_commentary(None)
    assert "No generated commentary available" in text
    assert "GENERATED; ADVISORY" not in text
