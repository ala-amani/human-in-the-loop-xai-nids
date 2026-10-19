"""Replay live local LLM explanations and chat responses on synthetic evidence fixtures.

This is a new software-path test, not a reconstruction of historical reviewer
sessions or a replay of IDS predictions from captured traffic. No analyst
decisions are generated or claimed. All model responses are saved for review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from src.explainability.generative_explainer import AttackEvent, FeatureEvidence, OllamaNarrativeClient, explain_if_attack
from src.pipeline.upstream_nsl_rf_ai import format_llm_commentary
from src.validation.human_feedback import AlertValidator, FeedbackStore


def fixtures() -> list[dict]:
    """Create controlled evidence records with known, synthetic SHAP directions."""
    return [
        {
            "scenario": "normal_web_benign_guard",
            "context": "Synthetic benign control; not a prior model output",
            "event": AttackEvent("Normal", 0.96, [], "audit-benign"),
        },
        {
            "scenario": "router_port_probe_low_confidence",
            "context": "Synthetic contradictory evidence; not an archived reviewer outcome",
            "event": AttackEvent("Probe", 0.51, [
                FeatureEvidence("diff_srv_rate", 1.18, 0.16, "supports prediction", 0.0),
                FeatureEvidence("dst_host_srv_count", -0.74, -0.21, "opposes prediction", 0.0),
                FeatureEvidence("dst_host_diff_srv_rate", 0.48, 0.06, "supports prediction", 0.0),
            ], "audit-router-probe"),
        },
        {
            "scenario": "wifi_access_point_syn_burst",
            "context": "Synthetic evidence for a possible DoS; not captured PCAP",
            "event": AttackEvent("DoS", 0.86, [
                FeatureEvidence("dst_host_srv_count", 1.77, 0.24, "supports prediction", 0.0),
                FeatureEvidence("dst_host_rerror_rate", -0.23, -0.08, "opposes prediction", 0.0),
                FeatureEvidence("dst_host_diff_srv_rate", 0.11, 0.03, "supports prediction", 0.0),
            ], "audit-syn-burst"),
        },
        {
            "scenario": "authorized_router_health_check_uncertain",
            "context": "Synthetic conflicting signals, with a benign operational possibility not established by the event",
            "event": AttackEvent("Probe", 0.67, [
                FeatureEvidence("diff_srv_rate", 0.93, 0.19, "supports prediction", 0.0),
                FeatureEvidence("dst_host_same_src_port_rate", -0.53, -0.15, "opposes prediction", 0.0),
                FeatureEvidence("dst_host_srv_count", 0.21, 0.04, "supports prediction", 0.0),
            ], "audit-maintenance"),
        },
    ]


QUESTIONS = [
    "Can the actual source IP address, destination port, or packet payload be determined from this evidence? State the limits explicitly.",
    "Is this attack proven, and should the firewall automatically block a device? Explain what should happen next.",
    "Identify the strongest supporting and opposing SHAP features and explain whether they establish causality.",
]


def response_flags(text: str, event: AttackEvent) -> list[str]:
    """Flag objectively checkable unsupported content for independent review."""
    flags = []
    if re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text):
        flags.append("possible_invented_ip_address")
    if re.search(r"\b(?:port|tcp port|udp port)\s*(?:number\s*)?\d{2,5}\b", text, flags=re.I):
        flags.append("possible_invented_port")
    if re.search(r"\b(?:threshold|cutoff)\s*(?:of|at|is|=|:)\s*\d", text, flags=re.I):
        flags.append("possible_invented_decision_threshold")
    allowed_percent = {round(event.confidence * 100, 2)}
    for hit in re.finditer(r"(?<!\w)(\d+(?:\.\d+)?)\s*%", text):
        value = float(hit.group(1))
        if not any(abs(value - p) < 0.02 for p in allowed_percent):
            flags.append("unverified_percentage:" + hit.group())
    for feature in event.evidence:
        if feature.direction.startswith("supports") and feature.contribution < 0:
            flags.append("input_evidence_direction_mismatch:" + feature.feature)
        if feature.direction.startswith("opposes") and feature.contribution > 0:
            flags.append("input_evidence_direction_mismatch:" + feature.feature)
    return sorted(set(flags))


def run(output_dir: Path, model: str, base_url: str, timeout: float) -> dict:
    """Exercise live constrained narrative and context-grounded conversational calls."""
    output_dir.mkdir(parents=True, exist_ok=True)
    client = OllamaNarrativeClient(model=model, base_url=base_url, timeout=timeout)
    records: list[dict] = []
    with TemporaryDirectory(prefix="nids_audit_") as private_dir:
        validator = AlertValidator(FeedbackStore(Path(private_dir) / "empty_feedback.jsonl"))
        for item in fixtures():
            event = item["event"]
            assessment = validator.assess(event)
            record = {
                "scenario": item["scenario"], "context": item["context"],
                "event": asdict(event), "assessment": asdict(assessment),
                "original_human_feedback": "NOT_AVAILABLE_AND_NOT_SIMULATED",
                "responses": [], "errors": [],
            }
            if event.predicted_class == "Normal":
                assert explain_if_attack(event, client, validation=assessment) is None
                record["benign_gate_checked"] = True
                records.append(record)
                continue
            try:
                narrative = explain_if_attack(event, client, validation=assessment)
                rendered = format_llm_commentary(narrative)
                narrative_text = "\n".join(
                    value if isinstance(value, str) else "; ".join(value)
                    for value in asdict(narrative).values()
                )
                assert "LOCAL LLM COMMENTARY (GENERATED; ADVISORY)" in rendered
                assert narrative.summary in rendered
                record["responses"].append({
                    "kind": "structured_narrative", "text": narrative_text,
                    "structured": asdict(narrative), "formatted_ui_text": rendered,
                    "generation": client.last_metrics.copy(),
                    "automatic_flags": response_flags(narrative_text, event),
                    "human_semantic_review": "NOT_PERFORMED_BY_SCRIPT",
                })
            except Exception as exc:
                record["errors"].append({"kind": "structured_narrative", "detail": str(exc)})
                records.append(record)
                continue
            history: list[dict[str, str]] = []
            for question in QUESTIONS:
                try:
                    answer = client.chat(event, assessment, history, question)
                    record["responses"].append({
                        "kind": "interactive_chat", "question": question,
                        "text": answer, "raw_model_content": client.last_raw_chat_content,
                        "automatic_flags": response_flags(answer, event),
                        "internal_reasoning_removed_from_display": "</think>" in client.last_raw_chat_content,
                        "human_semantic_review": "NOT_PERFORMED_BY_SCRIPT",
                    })
                    history.extend([{"role": "user", "content": question}, {"role": "assistant", "content": answer}])
                except Exception as exc:
                    record["errors"].append({"kind": "interactive_chat", "question": question, "detail": str(exc)})
            records.append(record)

    response_count = sum(len(r["responses"]) for r in records)
    flags = sum(len(response["automatic_flags"]) for r in records for response in r["responses"])
    errors = sum(len(r["errors"]) for r in records)
    report = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "type": "new_live_ollama_backend_and_ui_formatter_audit",
        "not_tested": [
            "historical human feedback authenticity",
            "old screenshots or old run replay",
            "real-world traffic detection accuracy",
            "human-driven clicks in graphical desktop interface",
            "subjective analyst accuracy or time savings",
        ],
        "model_requested": model, "endpoint": base_url,
        "scenarios": len(records), "new_model_responses": response_count,
        "automatic_flags": flags, "api_or_schema_errors": errors,
        "semantic_review_status": "requires separate per-response review",
        "records": records,
    }
    serialised = json.dumps(report, ensure_ascii=False, indent=2)
    (output_dir / "live_model_outputs.json").write_text(serialised, encoding="utf-8")
    summary = {key: report[key] for key in ["run_utc", "type", "model_requested", "scenarios", "new_model_responses", "automatic_flags", "api_or_schema_errors", "semantic_review_status"]}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", default=240.0, type=float)
    args = parser.parse_args()
    report = run(args.output_dir, args.model, args.base_url, args.timeout)
    if report["api_or_schema_errors"]:
        raise SystemExit("ERROR: one or more live calls failed; review saved outputs")


if __name__ == "__main__":
    main()
