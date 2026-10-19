"""Regression coverage for sequential synthetic alert review."""

import json
import sys
from pathlib import Path

import pandas as pd

from src.explainability.generative_explainer import AnalystNarrative, AttackEvent
from src.pipeline import synthetic_lab_evaluation as lab
from src.validation.human_feedback import ValidationAssessment


def test_all_scenarios_show_each_non_normal_alert(monkeypatch, tmp_path: Path) -> None:
    """Run the orchestration with deterministic stubs, without benchmark data or Ollama."""
    scenarios = ["normal_web", "wifi_ap_syn_burst", "router_port_probe", "authorized_router_health_check"]
    frames = pd.DataFrame({"scenario": scenarios, "synthetic_label": ["Normal", "DoS", "Probe", "Normal"]})
    calls = []

    class FixedValidator:
        def assess(self, event):
            return ValidationAssessment(
                reliability_score=0.5,
                verdict="uncertain",
                evidence_agreement=0.5,
                historical_precision=0.5,
                reviewed_alerts=0,
                review_priority="high",
                reasons=[],
            )

    narrative = AnalystNarrative(
        title="Review required",
        summary="Synthetic exercise.",
        observed_signs=[],
        reasoning="Evidence requires adjudication.",
        confidence_note="Uncalibrated.",
        recommended_checks=[],
        limitations="Synthetic scenario.",
        validation_verdict="uncertain",
        review_recommendation="review",
        human_review_question="Does context support the alert?",
    )

    def fake_evaluate(row, namespace):
        predicted = "Normal" if row["scenario"] == "normal_web" else "DoS"
        return AttackEvent(predicted_class=predicted, confidence=0.7, evidence=[], flow_id=row["scenario"]), {}

    monkeypatch.setattr(lab.pd, "read_csv", lambda *_args, **_kwargs: frames)
    monkeypatch.setattr(lab, "execute_published_prediction_block", lambda *_args: {})
    monkeypatch.setattr(lab, "FeedbackStore", lambda *_args: object())
    monkeypatch.setattr(lab, "AlertValidator", lambda *_args: FixedValidator())
    monkeypatch.setattr(lab, "OllamaNarrativeClient", lambda *_args: object())
    monkeypatch.setattr(lab, "evaluate_flow", fake_evaluate)
    monkeypatch.setattr(lab, "explain_if_attack", lambda event, client, validation=None: None if event.predicted_class == "Normal" else narrative)
    monkeypatch.setattr(lab, "show_attack_alert", lambda event, *_args: calls.append(event.flow_id))

    output = tmp_path / "scenario_results.json"
    argv = ["synthetic_lab_evaluation", "--flows", "scenarios.csv", "--data-root", str(tmp_path), "--upstream-script", "upstream.py", "--scenario", "all", "--output", str(output), "--feedback-store", str(tmp_path / "feedback.jsonl"), "--ollama", "--show-alert"]
    monkeypatch.setattr(sys, "argv", argv)

    lab.main()

    assert calls == scenarios[1:]
    recorded = json.loads(output.read_text(encoding="utf-8"))
    assert len(recorded) == 4
    assert recorded[0]["narrative"] is None
    assert all(item["narrative"] is not None for item in recorded[1:])
