> **Historical verification report (2026-10-16).** This report describes the state **before** the
> revised manuscript and public reproducibility package were prepared. References below to
> unedited manuscript text, an unpublished package, or the old Figure 6 are historical notes,
> not the current state. The revised manuscript reports the matched-seed results (32.7% and
> 93.1%). See `README.md` and `verify_recorded_results.py` for current usage.

# Methodological verification of the full-dataset NSL-KDD feedback experiment

Prepared 2026-10-16. Nothing in the manuscript was edited and nothing was pushed to GitHub.
The original experiment files were preserved unchanged; every new number in this report comes from
runs that write only into `controlled/` and `sensitivity/`.

---

## 1. What the original run was, and where its evidence lives

| Item | Location | Evidence |
| --- | --- | --- |
| Script | `original/full_dataset_auto_feedback.py` (copy of `tmp/full_dataset_auto_feedback.py`) | sha256 starts `92099009c473f7a6` |
| Results | `original/full_dataset_feedback_results.json` | written 2026-09-11 13:04, sha256 starts `6bbc814f4495` |
| Per-fold records | `original/full_dataset_feedback_folds.csv` | 10 rows (5 folds x 2 variants) |
| Figure | `original/before_after_error_rates.png` | manuscript Fig. 6 |
| Input data | `2_raw/NSL-KDD/KDDTrain+.txt`, `KDDTest+.txt` | 125,973 and 22,544 rows |

Every number in Section 6.9 of the manuscript (23 values and four Wilson intervals) matches this
JSON exactly. The results were produced by executing the script, not by hand calculation. No stdout
log of the original execution was retained; the JSON, CSV and figure are the execution evidence.

**The feedback was automatic.** Baseline errors are located by comparing predictions with the
benchmark labels of KDDTest+ (`masks(yq, baseline_pred)`), and those benchmark labels are appended
to the training data (`y_aug = concatenate([ytr, yq[corrected_idx]])`). No analyst reviewed these
rows and the interactive interface is not involved.

---

## 2. Finding 1 - the original run changed the random seed together with the feedback

In `full_dataset_auto_feedback.py` the baseline is fitted with `random_state=42` while each fold's
feedback model is fitted with `random_state=42+fold`. Seed and feedback therefore moved together.

The controlled re-run fits a **no-feedback control** and the feedback models with the same
`random_state` and the same hyperparameters, on the same fold split, and scores them on the same
rows. Five seeds (42-46) were used so that run-to-run variation is visible.

| Metric | Original run | Controlled, all rows (mean of 5 seeds) | Seed-to-seed range |
| --- | --- | --- | --- |
| False-positive rate | 2.55% -> 1.61% (-37.1%) | 2.58% -> 1.74% (-32.7%) | -28.2% to -36.7% |
| Wrong-attack-class rate | 1.72% -> 0.12% (-93.2%) | 1.68% -> 0.11% (-93.1%) | -88.3% to -97.2% |
| Attack recall, FP variant | -0.67 pp | -0.59 pp | -0.99 to -0.03 pp |
| Accuracy, FP variant | +0.15 pp | +0.01 pp | -0.19 to +0.12 pp |
| Attack recall, wrong-class variant | +1.19 pp | +0.34 pp | -0.28 to +1.20 pp |
| Accuracy, wrong-class variant | +1.58 pp | +1.09 pp | +0.90 to +1.61 pp |

With the seed held fixed at 42 on both sides, the false-positive reduction is **28.2%**, not 37.1%.
The wrong-attack-class reduction is unaffected by the confound.

---

## 3. Finding 2 - duplicate and overlap sensitivity

Counts on raw feature vectors (label and difficulty excluded):

| Quantity | Count | Share of test set |
| --- | --- | --- |
| Duplicate rows inside KDDTest+ | 57 | 0.25% |
| Duplicate rows inside KDDTrain+ | 16 | - |
| Test rows whose feature vector also appears in KDDTrain+ | 664 | 2.95% |
| ... of which the label differs from the training copy | 50 | 0.22% |
| Holdout rows with an exact duplicate inside their own feedback partition | 90 | 0.40% |
| ... carrying the **same** label (the only route for label transfer) | 6 | 0.027% |
| ... carrying a different label (conflicting supervision) | 84 | 0.37% |

The controlled comparison was repeated under five exclusion rules, applied identically to the
control and the feedback models:

| Scenario | Rule | Rows evaluated |
| --- | --- | --- |
| A | all test rows | 22,544 |
| B | duplicate rows inside KDDTest+ removed | 22,487 |
| C | test rows that also occur in KDDTrain+ removed | 21,880 |
| D | B and C combined | 21,833 |
| E | D, plus cross-fold duplicates removed from evaluation **and** withheld from the feedback set | 21,796 |

Mean relative change across the five seeds:

| Scenario | False-positive rate | Wrong-attack-class rate |
| --- | --- | --- |
| A all rows | 2.58% -> 1.74% (-32.7%) | 1.68% -> 0.11% (-93.1%) |
| B no test duplicates | 2.59% -> 1.74% (-32.7%) | 1.68% -> 0.11% (-93.1%) |
| C no train overlap | 2.41% -> 1.56% (-35.4%) | 1.75% -> 0.12% (-93.1%) |
| D both removed | 2.42% -> 1.56% (-35.4%) | 1.76% -> 0.12% (-93.1%) |
| E strict | 2.42% -> 1.56% (-35.4%) | 1.76% -> 0.12% (-93.1%) |

**No reduction and no reversal.** Removing duplicated and train-overlapping rows leaves the
wrong-attack-class effect unchanged and makes the false-positive effect slightly larger, because the
excluded rows are ones the baseline already classified well. Duplicate overlap is therefore not a
plausible explanation for either improvement.

---

## 4. Per-seed results

Scenario A (all 22,544 rows). Rates are per-run values; the spread across seeds is run-to-run
variation of the random forest, **not** a confidence interval.

### False-positive feedback

| Seed | Baseline FPR | Feedback FPR | Relative | Recall before -> after | Accuracy before -> after | Corrected / introduced |
| --- | --- | --- | --- | --- | --- | --- |
| 42 | 2.55% | 1.83% | -28.2% | 57.20% -> 57.17% | 73.55% -> 73.62% | 72 / 2 |
| 43 | 2.58% | 1.71% | -33.9% | 56.87% -> 56.30% | 73.44% -> 73.56% | 85 / 0 |
| 44 | 2.63% | 1.74% | -33.7% | 57.11% -> 56.30% | 73.60% -> 73.41% | 86 / 0 |
| 45 | 2.58% | 1.78% | -31.1% | 57.60% -> 57.07% | 73.81% -> 73.89% | 78 / 0 |
| 46 | 2.55% | 1.62% | -36.7% | 58.26% -> 57.27% | 74.04% -> 73.99% | 92 / 1 |

### Wrong-attack-class feedback

| Seed | Baseline rate | Feedback rate | Relative | Recall before -> after | Accuracy before -> after | Corrected / introduced |
| --- | --- | --- | --- | --- | --- | --- |
| 42 | 1.72% | 0.09% | -94.6% | 57.20% -> 58.40% | 73.55% -> 75.16% | 209 / 0 |
| 43 | 1.57% | 0.15% | -90.5% | 56.87% -> 57.19% | 73.44% -> 74.44% | 195 / 13 |
| 44 | 1.50% | 0.08% | -94.8% | 57.11% -> 57.29% | 73.60% -> 74.53% | 185 / 2 |
| 45 | 1.66% | 0.19% | -88.3% | 57.60% -> 57.89% | 73.81% -> 74.81% | 211 / 23 |
| 46 | 1.94% | 0.05% | -97.2% | 58.26% -> 57.98% | 74.04% -> 74.94% | 245 / 3 |

Scenario E (strict, 21,796 rows) gives the same picture with slightly larger false-positive effects:
-30.6%, -36.6%, -36.4%, -33.6% and -39.7% for seeds 42-46, and identical wrong-class values
(-94.6%, -90.5%, -94.8%, -88.3%, -97.2%). Full per-seed numbers for all five scenarios are in
`sensitivity/per_seed_table.csv`.

---

## 5. How to report the uncertainty

Two different quantities appear above and must not be conflated:

* **Wilson 95% intervals** (in the JSON files) are statistical intervals for the error counts of a
  single run, e.g. 2.55% (95% CI 2.26%-2.89%).
* **Seed ranges** (minimum and maximum over seeds 42-46) describe how much the result moves when the
  random forest is refitted with a different seed. They are not confidence intervals and should be
  described as "range over five seeds".

---

## 6. Leakage verification

The controlled run asserts, and records in `controlled/controlled_results.json`, that for every fold

* the holdout partitions cover the test set exactly once,
* feedback and holdout indices are disjoint,
* every corrected row lies inside the feedback partition.

All three hold. Preprocessing is fitted on the training rows only (`fit_transform(X_train)` followed
by `transform(X_test)`), and the baseline is fitted on training rows only. No evaluated row's own
label enters the model that predicts it. Residual routes, quantified in Section 3, are negligible:
six rows in the entire test set (0.027%) could in principle receive a label through an exact
duplicate, and scenario E removes them without changing the conclusions.

What remains is not leakage but a boundary on interpretation: the feedback rows and the evaluation
rows are drawn from the same benchmark test set, KDDTest+ deliberately contains attack types absent
from KDDTrain+, and the correction is applied exactly to the rows the metric scores. The experiment
therefore measures what returning verified labels does **inside this benchmark**, and is not a
measurement of generalisation to independently captured traffic.

---

## 7. Separation from the human-reviewed material

| File | Records | Modified | Used by the automated experiment? |
| --- | --- | --- | --- |
| `4_results/feedback/synthetic_local_lab_feedback.jsonl` | 3 (confirmed DoS, corrected to Probe, rejected false positive) | 2026-08-24 15:31 | no |
| `4_results/feedback/human_review_feedback.jsonl` | 2 (NSL-KDD worked examples of Sections 6.7 and 6.8) | 2026-08-24 14:46 | no |
| `4_results/feedback/benchmark_feedback.jsonl` | 2 (same two rows, benchmark reviewer) | 2026-08-24 13:45 | no |

`full_dataset_auto_feedback.py` reads only `KDDTrain+.txt` and `KDDTest+.txt` and writes only its own
JSON, CSV and figure. It contains no reference to the feedback store, to `FeedbackStore`, or to any
analyst record. The three human-adjudicated alerts predate the automated run by 18 days and remain a
separate strand of evidence.

---

## 8. Statements in the manuscript that need attention

No manuscript file was edited. The following items are affected:

| Location | Statement | Issue |
| --- | --- | --- |
| Abstract | "reduced the held-out false-positive rate from 2.55% to 1.61% (37.1% relative)" | produced with mismatched seeds; matched-seed value is 32.7% on all rows and 35.4% after duplicate exclusion, range 28-40% |
| Abstract | "while improving recall and accuracy" (wrong-class variant) | accuracy holds (+1.09 pp mean); the recall gain does not reproduce (-0.28 to +1.20 pp) |
| 6.9 | "a 37.1% relative reduction" | same as above |
| 6.9 | "corrected 92 errors and introduced none" | matched-seed range is 72-92 corrected, 0-2 introduced |
| 6.9 | "binary attack recall increased from 57.20% to 58.39% and overall accuracy from 73.55% to 75.13%" | accuracy reproduces; the recall increase is seed-dependent |
| 6.9 | "overall accuracy moved from 73.55% to 73.70%" (FP variant) | matched-seed mean change is +0.01 pp, i.e. no change |
| 8 Discussion | "the second improved recall and accuracy together" | restrict to accuracy |
| 9 Conclusion | "reduced FPR from 2.55% to 1.61%" | same as the abstract |

The wrong-attack-class result (-93.1% mean, -88.3% to -97.2% across seeds, unchanged under every
exclusion rule) is reproduced and defensible as reported. The false-positive result is real and
survives every exclusion, but its magnitude should be stated from the matched-seed runs.

---

## 9. Reproducibility

See `README.md` for the environment, the data paths and the exact commands for the original run,
the controlled run, the duplicate checks and the sensitivity analysis. The package deliberately
excludes the raw NSL-KDD files, analyst records and credentials.
