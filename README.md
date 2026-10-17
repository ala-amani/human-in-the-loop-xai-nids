# Human-in-the-Loop Explainable Intrusion Detection

Research implementation of an explainable network intrusion-detection workflow. The repository combines the published XAI-NIDS reference implementation with alert validation, grounded LLM interpretation, explicit analyst adjudication, and controlled IDS retraining from validated feedback.

## Repository layout

- `src/` — maintained implementation
  - `src/explainability/` — SHAP evidence records and the schema-constrained LLM interpretation
  - `src/validation/` — reliability estimation, append-only analyst feedback, and controlled candidate retraining
  - `src/pipeline/` — the NSL-KDD Random Forest pipeline, the analyst interface, and the synthetic local-network evaluation
  - `src/data/` — NSL-KDD loading and attack-family mapping
- `examples/` — runnable scripts, including the deterministic synthetic MikroTik packet generator
- `tests/` — automated tests (`python -m pytest`)
- `XAI_NIDS/` — upstream reference implementation of the base study
- `code/base_article_code/` — preserved baseline implementation
- `code/ai_extension_code/` — earlier standalone copy of the extension, kept for reference; `src/` is the maintained version
- `docs/datasets/` — dataset provenance and schemas (no datasets are committed)

## Workflow

1. The IDS classifies a network-flow record.
2. SHAP ranks the features supporting and opposing the prediction.
3. An independent validation score estimates whether the alert should be reviewed.
4. A grounded language model converts the selected evidence into an English analyst narrative.
5. The analyst records `Correct`, `False positive`, or `Wrong attack class` with a reason.
6. Reviewed feedback updates the class-history term of later reliability estimates and supplies labelled rows for controlled IDS retraining; a candidate model is promoted only after holdout evaluation and explicit approval.

## Quick start

See [`code/README.md`](code/README.md) for installation and execution instructions. The interactive runs reported in the manuscript used a local Ollama model (`qwen3:4b`), so no traffic data left the host; the reported LLM execution path uses Ollama. Credentials, datasets, packet captures, private logs, and manuscript files are intentionally excluded from this repository.

## Reproducibility

Dataset provenance and expected schemas are documented under `docs/datasets/`. The synthetic local-network generator uses fixed seeds and reproducible scenario definitions. Official dataset files and private generated artifacts must be produced locally and are not versioned; archived aggregate feedback results and scripts are included in the experiment package below.

## Journal manuscript: matched-seed NSL-KDD feedback evaluation

The full-dataset **automatic-feedback** experiments for Section 6.9 of the revised journal manuscript, including archived output files, five matched Random Forest seeds (42–46), and duplicate/overlap sensitivity analysis, are in [`experiments/2026-10-16-full-dataset-feedback/`](experiments/2026-10-16-full-dataset-feedback/README.md). The archived consistency checker runs without dataset files:

```bash
python experiments/2026-10-16-full-dataset-feedback/verify_recorded_results.py
```

The simulated feedback is generated from **NSL-KDD benchmark labels**, not human reviews or LLM labels. In a separate synthetic four-scenario integration test, a person reviewed three alerts. The five matched-seed benchmark experiment found a mean relative false-positive reduction of **32.7%** and wrong-class error reduction of **93.1%**; the false-positive intervention also reduced binary attack recall by **0.59 percentage points** on average. The experiment does not demonstrate an independent detection benefit from the LLM or deployment-scale efficacy. The original IDS, SHAP, LLM, and review implementation has not been modified to obtain these estimates.

See the experiment README for data requirements, protocol, duplicate exclusion, and the distinction between reported seed ranges and statistical confidence intervals.
