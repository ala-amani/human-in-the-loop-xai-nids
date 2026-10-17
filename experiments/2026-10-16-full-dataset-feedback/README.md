# Full-dataset NSL-KDD automatic-feedback experiment

This folder packages the experiment that produced the aggregate feedback results reported in
the manuscript (Section 6.9), together with a controlled re-evaluation that removes a random-seed
confound found in the original run, and the leakage checks used to validate the design.

**Nothing in `original/` has been modified.** Those four files are preserved copies of
artifacts produced on 2026-09-11. The controlled re-evaluation and sensitivity analysis were
performed separately on 2026-10-16; their recorded outputs are included in this folder.

The raw NSL-KDD files are **not** included here. They are read from the data folder listed below.

## Layout

| Path | What it is |
| --- | --- |
| `original/full_dataset_auto_feedback.py` | The script that produced the reported numbers (verbatim copy of `tmp/full_dataset_auto_feedback.py`) |
| `original/full_dataset_feedback_results.json` | Original results, written 2026-09-11 13:04 |
| `original/full_dataset_feedback_folds.csv` | Original per-fold records (5 folds x 2 variants) |
| `original/before_after_error_rates.png` | Historical single-run figure; **not** Figure 6 in the revised manuscript |
| `controlled/controlled_feedback_experiment.py` | Matched-seed re-evaluation with a per-seed no-feedback control |
| `controlled/controlled_results.json` | Controlled results for seeds 42-46 |
| `controlled/controlled_folds.json` | Controlled per-fold records |
| `controlled/run.log` | Console output of the controlled run |
| `analysis/duplicate_overlap_check.py` | Duplicate and cross-fold overlap checks |
| `analysis/duplicate_overlap_report.json` | Output of those checks |
| `analysis/comparison_table.csv` | Original versus controlled metrics |
| `sensitivity/duplicate_sensitivity_experiment.py` | Five sensitivity scenarios across five seeds |
| `sensitivity/sensitivity_results.json` | Detailed sensitivity outcomes |
| `sensitivity/per_seed_table.csv` | Per-seed metrics for each scenario |
| `sensitivity/scenario_summary.csv` | Aggregate sensitivity summary |
| `METHODOLOGICAL_REPORT.md` | Historical audit dated 2026-10-16, with the pre-revision wording retained |
| `verify_recorded_results.py` | Check JSON output consistency against Section 6.9 without datasets |

## Environment

Python 3.12 with the repository's virtual environment:

```
numpy >= 1.26
pandas >= 2.2
scikit-learn >= 1.5
scipy          (binomtest, used for the exact McNemar test)
matplotlib     (figure only)
```

## Data

```
<data root>/KDDTrain+.txt      125,973 rows
<data root>/KDDTest+.txt        22,544 rows
```

Supply the directory containing those two files via `--data-root`.
Use a path on your own machine. The files are not redistributed in this repository.

## Reproducing the original run

```bash
python experiments/2026-10-16-full-dataset-feedback/original/full_dataset_auto_feedback.py \
  --data-root "<data root>" \
  --output-dir <output folder> \
  --seed 42 --folds 5
```

Configuration actually used for the reported numbers: `seed=42`, `folds=5`,
`RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=2, n_jobs=-1)`,
fold split `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` over the test set.
Note the seed asymmetry in that script: the baseline is fitted with `random_state=42` while each
fold's feedback model is fitted with `random_state=42+fold`, so seed and feedback change together.

## Reproducing the controlled run

```bash
python experiments/2026-10-16-full-dataset-feedback/controlled/controlled_feedback_experiment.py \
  --data-root "<data root>" \
  --output-dir experiments/2026-10-16-full-dataset-feedback/controlled \
  --seeds 42 43 44 45 46 --folds 5
```

The controlled script imports the loading, preprocessing, mask, metric, Wilson-interval and exact
McNemar functions from the original script, so the definitions cannot drift. For every seed it fits

* a **no-feedback control** on the training rows only, and
* the two feedback models on training rows plus the corrected rows of that fold's feedback partition,

all with the same `random_state` and the same hyperparameters, and evaluates them on the same
held-out fold. Runtime: about 25 s per seed on this machine.

## Leakage checks

```bash
python experiments/2026-10-16-full-dataset-feedback/analysis/duplicate_overlap_check.py \
  --data-root "<data root>" \
  --output-dir experiments/2026-10-16-full-dataset-feedback/analysis
```

The controlled run also asserts, for every fold, that the holdout rows cover the test set exactly
once, that feedback and holdout indices are disjoint, and that every corrected row lies inside the
feedback partition. Those three guards are recorded in `controlled/controlled_results.json`.

## What the feedback is

The feedback in this experiment is **automatic**: baseline errors are identified by comparing the
baseline predictions with the benchmark labels of KDDTest+, and those benchmark labels are returned
to the model. No analyst reviewed these rows and the interactive interface is not involved. The
human-reviewed material in the manuscript is the separate four-scenario synthetic local-network
study, which uses different data and a different path.

## Reproducing the duplicate sensitivity analysis

Run from the repository root (the `--original-script` argument is optional because each script now resolves the immutable original within this package):

```bash
python experiments/2026-10-16-full-dataset-feedback/sensitivity/duplicate_sensitivity_experiment.py \
  --data-root "<data root>" \
  --output-dir experiments/2026-10-16-full-dataset-feedback/sensitivity \
  --seeds 42 43 44 45 46 --folds 5
python experiments/2026-10-16-full-dataset-feedback/sensitivity/summarise_sensitivity.py \
  --results experiments/2026-10-16-full-dataset-feedback/sensitivity/sensitivity_results.json \
  --output-dir experiments/2026-10-16-full-dataset-feedback/sensitivity
```

## Check archived outputs without datasets

```bash
python experiments/2026-10-16-full-dataset-feedback/verify_recorded_results.py
```

This command checks saved numbers, fold counts and leakage guards only. It does not retrain
the models or independently verify the original NSL-KDD execution. Full reproduction requires
obtaining the official NSL-KDD files and rerunning the three experiment scripts above.

## Reporting and provenance

* Matched-seed comparisons over all 22,544 official test records are the main estimates. Duplicate-exclusion sensitivity analyses are supplementary robustness results, not a replacement primary cohort.
* Mean relative reduction across five seeds is distinct from relative reduction calculated using pooled means. Seed minima and maxima are not statistical confidence intervals.
* Figure 6 in the revised manuscript uses the five matched-seed estimates and displays the range over seeds; the archived `original/before_after_error_rates.png` is **not** the revised figure.
* The historical `original/` outputs remain unchanged. This is the public reproducibility package for the revised manuscript; `METHODOLOGICAL_REPORT.md` remains a dated audit.
* Recorded outputs were verified for internal numerical consistency. Full reproduction requires the separate official dataset files and compatible package versions; these files are not included.
