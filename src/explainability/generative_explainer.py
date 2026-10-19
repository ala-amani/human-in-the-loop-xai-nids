"""Grounded analyst narratives for attacks detected by an IDS model.

The language model is downstream of the classifier and XAI layer: it explains
supplied evidence, but it never decides whether traffic is an attack.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Protocol, Sequence

from src.explainability.grounding_guard import numeric_grounding_issues, review_claim_issues


@dataclass(frozen=True)
class FeatureEvidence:
    feature: str
    observed_value: float | int | str
    contribution: float
    direction: str
    baseline: float | int | str | None = None


@dataclass(frozen=True)
class AttackEvent:
    predicted_class: str
    confidence: float
    evidence: Sequence[FeatureEvidence]
    flow_id: str | None = None


@dataclass(frozen=True)
class AnalystNarrative:
    title: str
    summary: str
    observed_signs: list[str]
    reasoning: str
    confidence_note: str
    recommended_checks: list[str]
    limitations: str
    validation_verdict: str
    review_recommendation: str
    human_review_question: str
    source: str = "llm"


class NarrativeClient(Protocol):
    def explain(
        self, event: AttackEvent, validation: Any | None = None, language: str = "English"
    ) -> AnalystNarrative: ...


NARRATIVE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string"},
        "observed_signs": {"type": "array", "items": {"type": "string"}},
        "reasoning": {"type": "string"},
        "confidence_note": {"type": "string"},
        "recommended_checks": {"type": "array", "items": {"type": "string"}},
        "limitations": {"type": "string"},
        "validation_verdict": {"type": "string"},
        "review_recommendation": {"type": "string"},
        "human_review_question": {"type": "string"},
    },
    "required": [
        "title", "summary", "observed_signs", "reasoning", "confidence_note",
        "recommended_checks", "limitations", "validation_verdict",
        "review_recommendation", "human_review_question",
    ],
}

NSL_KDD_FEATURE_GLOSSARY = {
    "dst_host_srv_count": "count of recent connections to the destination host that used the same service",
    "dst_host_rerror_rate": "rate of rejected-connection errors in recent traffic to the destination host",
    "dst_host_diff_srv_rate": "rate of different services used in recent traffic to the destination host",
    "diff_srv_rate": "rate of connections to different services in the short-term traffic window",
    "dst_host_same_src_port_rate": "rate of recent destination-host connections using the same source port",
}

ATTACK_GLOSSARY = {
    "Probe": "network reconnaissance or scanning intended to discover reachable hosts, ports, or services",
    "DoS": "an attempt to disrupt service availability through abnormal load or requests",
    "R2L": "a remote user attempting to obtain unauthorized local access",
    "U2R": "a normal user attempting to gain administrative privileges",
}


def strip_ollama_thinking(text: str) -> str:
    """Return only the user-facing answer; retain raw output separately for audit."""
    content = str(text).strip()
    if "</think>" in content:
        content = content.rsplit("</think>", 1)[-1].strip()
    if "<think>" in content:
        content = content.split("<think>", 1)[0].strip()
    if not content:
        raise RuntimeError("Ollama provided no user-facing answer")
    return content


def _verified_fallback(event: AttackEvent, validation: Any | None) -> AnalystNarrative:
    """Human-readable, deterministic account of the exact available evidence."""
    signs = [
        (
            f"{item.feature}: standardized input {float(item.observed_value):+.3f}; "
            f"signed SHAP contribution {float(item.contribution):+.4f} "
            f"({'supports' if item.contribution >= 0 else 'opposes'} the prediction)."
        )
        for item in event.evidence
    ]
    verdict = getattr(validation, "verdict", "not assessed")
    priority = getattr(validation, "review_priority", "requires human review")
    score = getattr(validation, "reliability_score", None)
    score_note = (
        f" Validator heuristic score: {float(score):.3f}, not a calibrated probability."
        if score is not None else ""
    )
    reviews = getattr(validation, "reviewed_alerts", None)
    prior_note = (
        " There are zero past human reviews: the Beta(1,1) estimate is a prior, not measured precision."
        if reviews == 0 else ""
    )
    return AnalystNarrative(
        title="Evidence summary (verified deterministic fallback)",
        summary=(
            f"The IDS predicted {event.predicted_class} with classifier confidence "
            f"{event.confidence * 100:.2f}%. This is not a verified attack."
        ),
        observed_signs=signs,
        reasoning=(
            "Positive signed SHAP contributions support the classifier prediction and "
            "negative contributions oppose it. These are attribution units, not "
            "probability changes or causal evidence."
        ),
        confidence_note="Classifier confidence does not establish ground truth." + score_note + prior_note,
        recommended_checks=[
            "Compare the alert with packet capture, router logs and authorized maintenance records.",
            "Have an analyst verify the outcome; never automatically change firewall policy.",
        ],
        limitations="No packet payload, source IP or specific destination port is supplied in this evidence.",
        validation_verdict=f"Deterministic validator verdict: {verdict}.",
        review_recommendation=f"Deterministic validator review priority: {priority}.",
        human_review_question="Is the alert correct, a false positive, or the wrong attack class?",
        source="deterministic_fallback",
    )


def _verified_chat_fallback(event: AttackEvent, validation: Any | None, question: str) -> str:
    """Give only input-derived facts when LLM numeric claims fail checks."""
    q = question.lower()
    intro = "The generated chat reply was withheld because its numeric claims were not fully grounded. "
    if "ip" in q or "payload" in q or "port" in q:
        return intro + "The supplied SHAP evidence contains no source IP, destination port or packet payload."
    if "block" in q or "proven" in q or "firewall" in q:
        return intro + (
            f"The predicted class is {event.predicted_class}; it is not proof of an attack. "
            "A researcher must review network evidence. Do not automatically block traffic or alter firewall policy."
        )
    if "shap" in q or "caus" in q or "feature" in q:
        positive = sorted((x for x in event.evidence if x.contribution > 0),
                          key=lambda x: x.contribution, reverse=True)
        negative = sorted((x for x in event.evidence if x.contribution < 0),
                          key=lambda x: x.contribution)
        supporting = (f"{positive[0].feature} ({positive[0].contribution:+.4f})"
                      if positive else "none in the supplied evidence")
        opposing = (f"{negative[0].feature} ({negative[0].contribution:+.4f})"
                    if negative else "none in the supplied evidence")
        return intro + (
            f"Strongest supporting feature: {supporting}. Strongest opposing feature: {opposing}. "
            "SHAP describes the classifier, not causality."
        )
    return intro + "Please review the signed SHAP values and deterministic validator result before making any decision."


def _check_narrative(
    narrative: AnalystNarrative, event: AttackEvent, validation: Any | None
) -> list[str]:
    parts = [
        narrative.title, narrative.summary, *narrative.observed_signs,
        narrative.reasoning, narrative.confidence_note, *narrative.recommended_checks,
        narrative.limitations, narrative.validation_verdict,
        narrative.review_recommendation, narrative.human_review_question,
    ]
    issues = numeric_grounding_issues("\n".join(parts), event, validation)
    issues.extend(review_claim_issues(narrative.review_recommendation, validation))
    expected = getattr(validation, "verdict", None)
    if expected:
        verdict_text = narrative.validation_verdict.lower().replace("_", " ")
        for candidate in ("likely correct", "likely incorrect", "uncertain"):
            if candidate != expected.replace("_", " ") and candidate in verdict_text:
                issues.append("validator_verdict_mismatch")
    return sorted(set(issues))


class OllamaNarrativeClient:
    """Generate a grounded narrative through a local Ollama chat endpoint."""

    def __init__(
        self,
        model: str = "qwen3:4b",
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 180.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.last_metrics: dict[str, float | int | str] = {}
        self.last_raw_chat_content: str = ""
        self.last_grounding_issues: list[str] = []
        self.last_chat_grounding_issues: list[str] = []

    def explain(
        self, event: AttackEvent, validation: Any | None = None, language: str = "English"
    ) -> AnalystNarrative:
        evidence_json = json.dumps(asdict(event), ensure_ascii=False, indent=2)
        schema_json = json.dumps(NARRATIVE_SCHEMA, ensure_ascii=False)
        context_json = json.dumps(
            {
                "predicted_attack_meaning": ATTACK_GLOSSARY.get(event.predicted_class, event.predicted_class),
                "feature_meanings": {
                    item.feature: NSL_KDD_FEATURE_GLOSSARY.get(item.feature, item.feature)
                    for item in event.evidence
                },
                "value_scale": (
                    "Observed values are standardized model inputs. Positive means above the training average; "
                    "negative means below it. They are not raw packet counts or percentages."
                ),
            },
            ensure_ascii=False,
        )
        validation_json = json.dumps(asdict(validation) if validation else None, ensure_ascii=False, indent=2)
        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "format": NARRATIVE_SCHEMA,
            "options": {"temperature": 0, "seed": 42},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You explain network IDS alerts to a security analyst. The classifier "
                        "already made the prediction. Use only supplied values and attribution "
                        "evidence. Never invent packets, thresholds, causal claims, or certainty. "
                        "Clearly distinguish evidence supporting the prediction from evidence "
                        "opposing it. Recommend verification, never automatic shutdown. Write fluent, plain, "
                        "professional English for a general IT analyst. Avoid ambiguous or unusual terminology. "
                        "Describe Probe as network reconnaissance or scanning. Explain each "
                        "technical feature using the supplied glossary. State which signs support and which oppose "
                        "the prediction. Treat confidence near 51 percent as low and borderline. A probability is "
                        "confidence, never confirmation. Explain the independent validation verdict and whether the "
                        "analyst should prioritize the alert for review. Ask the analyst to label the alert as correct, "
                        "false positive, or wrong attack class. Never describe the validation estimate as ground truth. The reliability score is an uncalibrated heuristic, not a probability. If no prior alerts have been reviewed, the Beta(1,1) historical-precision estimate of 0.5 is a prior, not measured classifier precision. Never claim that half of alerts were wrong when there are zero reviews. SHAP contributions are signed attribution units, never percentage-point changes in attack probability. Do not invent decision thresholds or extra numeric claims."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Write the narrative in {language}. Return only JSON matching this schema: "
                        f"{schema_json}\nPlain-language context:\n{context_json}\nEvent evidence:\n{evidence_json}"
                        f"\nIndependent validation assessment:\n{validation_json}"
                    ),
                },
            ],
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama is unavailable at {self.base_url}: {exc}") from exc
        content = body["message"]["content"]
        self.last_metrics = {
            "provider": "Ollama",
            "model": str(body.get("model", self.model)),
            "total_duration_seconds": float(body.get("total_duration", 0)) / 1_000_000_000,
            "load_duration_seconds": float(body.get("load_duration", 0)) / 1_000_000_000,
            "prompt_tokens": int(body.get("prompt_eval_count", 0)),
            "generated_tokens": int(body.get("eval_count", 0)),
        }
        narrative = AnalystNarrative(**json.loads(content))
        self.last_grounding_issues = _check_narrative(narrative, event, validation)
        if self.last_grounding_issues:
            return _verified_fallback(event, validation)
        return narrative

    def chat(
        self, event: AttackEvent, validation: Any, history: list[dict[str, str]], question: str
    ) -> str:
        """Continue a grounded English conversation about one detected event."""
        event_json = json.dumps(asdict(event), ensure_ascii=False, indent=2)
        validation_json = json.dumps(asdict(validation) if validation else None, ensure_ascii=False, indent=2)
        messages = [{
            "role": "system",
            "content": (
                "You are assisting a network security analyst with one IDS alert. Answer in clear, "
                "concise English. Use only the supplied classifier prediction and SHAP evidence. "
                "Do not invent packet contents, IP addresses, thresholds, or facts that are absent. Do not provide internal reasoning or thought transcripts. Provide only a concise final answer. SHAP contribution values are model attribution units, never percentage-point changes in classification probability. A historical precision of 0.5 with zero reviewed alerts is an uninformative Beta(1,1) prior, NOT 50 percent empirically correct or incorrect alerts. The reliability score is an uncalibrated heuristic, NOT a predicted probability of the alert being correct. Do not invent confidence cutoffs or thresholds. "
                "Explain technical terms plainly. Say when the available evidence cannot answer a "
                "question. SHAP explains model influence, not causality. Recommend verification; "
                "never claim the attack is proven and never order an automatic shutdown. Explain that the validator "
                "estimates reliability and only human feedback establishes the reviewed outcome. Validated feedback "
                "is returned to the IDS retraining workflow, not used to modify firewall policy.\n\n"
                f"Fixed event evidence:\n{event_json}\n\nValidation assessment:\n{validation_json}"
            ),
        }, *history[-8:], {"role": "user", "content": question}]
        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "options": {"temperature": 0, "seed": 42},
            "messages": messages,
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama is unavailable at {self.base_url}: {exc}") from exc
        self.last_raw_chat_content = str(body["message"]["content"])
        answer = strip_ollama_thinking(self.last_raw_chat_content)
        self.last_chat_grounding_issues = numeric_grounding_issues(answer, event, validation)
        if self.last_chat_grounding_issues:
            return _verified_chat_fallback(event, validation, question)
        return answer


def explain_if_attack(
    event: AttackEvent,
    client: NarrativeClient,
    validation: Any | None = None,
    benign_labels: Sequence[str] = ("normal", "benign"),
    language: str = "English",
) -> AnalystNarrative | None:
    """Call the narrative AI only after a non-benign prediction."""

    normalized = event.predicted_class.strip().casefold()
    if normalized in {label.strip().casefold() for label in benign_labels}:
        return None
    if not event.evidence:
        raise ValueError("An attack narrative requires SHAP/LIME feature evidence")
    return client.explain(event, validation=validation, language=language)
