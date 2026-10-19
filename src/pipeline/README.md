# End-to-end pipeline

The pipeline connects NSL-KDD preprocessing, predictions, signed SHAP evidence, constrained commentary, researcher-entered alert decisions, and append-only feedback.

## Commentary shown in the interface

The runtime GUI now renders the actual structured `AnalystNarrative` output under the heading **LOCAL LLM COMMENTARY (GENERATED; ADVISORY)**, separated from the deterministic **RULE-BASED REVIEW CONTEXT**. If no generated narrative is available, the interface explicitly states that no generated commentary is available. These strings do not assign ground truth or modify the IDS.

The archived figures from the original four-scenario synthetic demo show the earlier rule-based explanation display. Those historical screenshots are not evidence that the GUI displayed a generated LLM response, and they cannot substantiate an LLM-specific improvement in reviewer performance.

The synthetic experiment involved one researcher-operator and four deterministic fixtures, of which three produced alert decisions. This is not an independently sampled analyst study. The separate full NSL-KDD feedback experiment uses benchmark ground-truth labels, not human reviews.

## Tests

```bash
python -m pytest -q tests/test_llm_ui_commentary.py
python -m pytest -q tests/test_synthetic_lab.py
python experiments/2026-10-16-full-dataset-feedback/verify_recorded_results.py
```

The archived-result consistency checker does not retrain models from raw datasets. End-to-end local Qwen/Ollama replay and independently audited human-review logs require the external data and runtime environment.
