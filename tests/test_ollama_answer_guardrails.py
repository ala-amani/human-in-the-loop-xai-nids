"""Regression tests for grounded and user-facing local-model chat responses."""

import json
import urllib.request

import pytest

from src.explainability.generative_explainer import AttackEvent, FeatureEvidence, OllamaNarrativeClient, strip_ollama_thinking


def test_internal_reasoning_is_not_returned_to_analyst() -> None:
    """Exclude tagged internal model text from the final user-facing answer."""
    assert strip_ollama_thinking("Private thoughts\\n</think>\\nFinal analyst guidance") == "Final analyst guidance"
    assert strip_ollama_thinking("Final analyst guidance") == "Final analyst guidance"


def test_missing_final_model_response_is_rejected() -> None:
    """Do not surface a chain of thought when the model returned no final answer."""
    with pytest.raises(RuntimeError, match="no user-facing answer"):
        strip_ollama_thinking("<think>private intermediate reasoning")


def test_chat_preserves_raw_audit_and_returns_only_final(monkeypatch) -> None:
    """Validate prompt constraints, raw audit retention, and final-answer extraction."""
    seen = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return json.dumps({"message": {"content": "Intermediate reasoning\\n</think>\\nNo source IP is present."}}).encode()

    def fake_urlopen(request, timeout):
        seen["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    event = AttackEvent("Probe", 0.51, [FeatureEvidence("diff_srv_rate", 1.0, 0.2, "supports prediction", 0.0)])
    client = OllamaNarrativeClient(model="qwen3:4b")
    answer = client.chat(event, None, [], "What is the source IP?")
    assert answer == "No source IP is present."
    assert "Intermediate reasoning" in client.last_raw_chat_content
    assert seen["payload"]["think"] is False
    assert seen["payload"]["options"]["temperature"] == 0
    assert "SHAP contribution values are model attribution units" in seen["payload"]["messages"][0]["content"]
    assert "Beta(1,1) prior" in seen["payload"]["messages"][0]["content"]
