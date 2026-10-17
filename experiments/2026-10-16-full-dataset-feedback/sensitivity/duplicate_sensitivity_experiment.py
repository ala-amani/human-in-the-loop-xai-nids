"""Duplicate-overlap sensitivity analysis for the NSL-KDD automatic-feedback experiment.

Repeats the matched-seed controlled comparison under progressively stricter exclusion rules,
so that the reported improvements can be read without the contribution of duplicated rows:

  A  all rows                      reference, identical to the controlled run
  B  minus duplicate test rows     later copies of an exact duplicate inside KDDTest+ are dropped
  C  minus train-overlap rows      test rows whose feature vector also appears in KDDTrain+
  D  minus both
  E  strict                        D, plus: holdout rows that have an exact duplicate inside their
                                   own feedback partition are dropped from the evaluation AND those
                                   duplicates are withheld from the feedback set during training

Rules are applied identically to the no-feedback control and to the feedback models: both are
fitted with the same random_state and the same hyperparameters and are scored on exactly the same
rows. Scenarios A-D reuse one set of feedback models per seed; scenario E needs its own models
because its exclusion changes the training data.

Nothing produced by earlier runs is read, modified or overwritten.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

SCENARIOS = {
    "A_all_rows": "all test rows",
    "B_no_test_duplicates": "duplicate rows inside KDDTest+ removed from evaluation",
    "C_no_train_overlap": "test rows whose feature vector also occurs in KDDTrain+ removed",
    "D_no_duplicates_or_overlap": "B and C combined",
    "E_strict_cross_fold": "D, plus cross-fold duplicate rows excluded from evaluation and from the feedback set",
}


def load_original(path: Path):
    spec = importlib.util.spec_from_file_location("original_experiment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def densify(matrix):
    return matrix.toarray() if hasattr(matrix, "toarray") else matrix


def subset_report(orig, y_true, y_pred_control, y_pred_feedback, variant: str) -> dict:
    control = orig.metrics(y_true, y_pred_control)
    feedback = orig.metrics(y_true, y_pred_feedback)
    control["false_positive_rate_ci95_wilson"] = orig.wilson(control["false_positives"], control["normal_samples"])
    control["wrong_attack_class_rate_ci95_wilson"] = orig.wilson(control["wrong_attack_class"], control["attack_samples"])
    feedback["false_positive_rate_ci95_wilson"] = orig.wilson(feedback["false_positives"], feedback["normal_samples"])
    feedback["wrong_attack_class_rate_ci95_wilson"] = orig.wilson(feedback["wrong_attack_class"], feedback["attack_samples"])
    normal = y_true == "Normal"
    before, after = orig.masks(y_true, y_pred_control), orig.masks(y_true, y_pred_feedback)
    if variant == "false_positive_feedback":
        paired = orig.paired_change(before["false_positive"][normal], after["false_positive"][normal])
    else:
        paired = orig.paired_change(before["wrong_attack_class"][~normal], after["wrong_attack_class"][~normal])
    return {"evaluated_rows": int(len(y_true)), "control": control, "feedback": feedback, "paired_vs_control": paired}


def run(data_root: Path, original_script: Path, output_dir: Path, seeds: list[int], folds: int = 5) -> dict:
    orig = load_original(original_script)
    X_train, y_train = orig.load_file(data_root / "KDDTrain+.txt")
    X_test, y_test = orig.load_file(data_root / "KDDTest+.txt")
    key_train = X_train.astype(str).agg("|".join, axis=1)
    key_test = X_test.astype(str).agg("|".join, axis=1)

    prep = orig.preprocessor(X_train)
    Xt = densify(prep.fit_transform(X_train))
    Xq = densify(prep.transform(X_test))
    ytr, yq = y_train.to_numpy(), y_test.to_numpy()

    dup_in_test = key_test.duplicated().to_numpy()                 # later copies only
    in_train = key_test.isin(set(key_train)).to_numpy()
    splits = list(StratifiedKFold(n_splits=folds, shuffle=True, random_state=42).split(Xq, yq))

    # rows whose label could, in principle, travel through an exact duplicate inside the same fold
    cross_fold_dup = np.zeros(len(yq), dtype=bool)
    feedback_blocked = {}
    keys = key_test.to_numpy()                      # plain array lookups; .iloc per row is far too slow here
    for fold, (feedback_idx, holdout_idx) in enumerate(splits, start=1):
        feedback_keys = set(keys[feedback_idx])
        holdout_keys = set(keys[holdout_idx])
        hits = [j for j in holdout_idx if keys[j] in feedback_keys]
        cross_fold_dup[hits] = True
        feedback_blocked[fold] = {j for j in feedback_idx if keys[j] in holdout_keys}

    keep = {
        "A_all_rows": np.ones(len(yq), dtype=bool),
        "B_no_test_duplicates": ~dup_in_test,
        "C_no_train_overlap": ~in_train,
        "D_no_duplicates_or_overlap": ~(dup_in_test | in_train),
        "E_strict_cross_fold": ~(dup_in_test | in_train | cross_fold_dup),
    }
    counts = {name: int(mask.sum()) for name, mask in keep.items()}

    results = {
        "design": ("matched-seed control versus feedback under five duplicate-overlap exclusion rules; "
                   "identical rules, seeds and hyperparameters on both sides"),
        "scenarios": SCENARIOS, "rows_evaluated_per_scenario": counts,
        "excluded_counts": {"duplicate_rows_inside_test": int(dup_in_test.sum()),
                            "test_rows_also_in_train": int(in_train.sum()),
                            "cross_fold_duplicate_holdout_rows": int(cross_fold_dup.sum())},
        "fold_split": "StratifiedKFold(n_splits=%d, shuffle=True, random_state=42) over the test set" % folds,
        "seeds": seeds, "per_seed": {}, "note_on_ranges": (
            "Across-seed minimum and maximum describe run-to-run variation of the random forest; they are not "
            "confidence intervals. Wilson 95% intervals reported inside each scenario are the statistical "
            "intervals for a single run's counts."),
    }

    for seed in seeds:
        started = time.time()
        control = orig.classifier(seed).fit(Xt, ytr)
        control_pred = control.predict(Xq)
        control_masks = orig.masks(yq, control_pred)
        variants = {"false_positive_feedback": control_masks["false_positive"],
                    "wrong_attack_class_feedback": control_masks["wrong_attack_class"]}
        pred_plain = {name: np.empty(len(yq), dtype=object) for name in variants}
        pred_strict = {name: np.empty(len(yq), dtype=object) for name in variants}
        feedback_rows = {name: {"plain": 0, "strict": 0} for name in variants}

        for fold, (feedback_idx, holdout_idx) in enumerate(splits, start=1):
            blocked = feedback_blocked[fold]
            for name, eligible in variants.items():
                corrected = feedback_idx[eligible[feedback_idx]]
                for tag, rows, store in (("plain", corrected, pred_plain),
                                         ("strict", np.array([j for j in corrected if j not in blocked], dtype=int), pred_strict)):
                    X_aug = np.vstack([Xt, Xq[rows]]) if len(rows) else Xt
                    y_aug = np.concatenate([ytr, yq[rows]]) if len(rows) else ytr
                    model = orig.classifier(seed).fit(X_aug, y_aug)
                    store[name][holdout_idx] = model.predict(Xq[holdout_idx])
                    feedback_rows[name][tag] += int(len(rows))

        entry = {"feedback_rows_used": feedback_rows, "scenarios": {}}
        for scenario, mask in keep.items():
            source = pred_strict if scenario == "E_strict_cross_fold" else pred_plain
            entry["scenarios"][scenario] = {
                name: subset_report(orig, yq[mask], control_pred[mask], source[name][mask], name)
                for name in variants
            }
        entry["seconds"] = round(time.time() - started, 1)
        results["per_seed"][str(seed)] = entry
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "sensitivity_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        a = entry["scenarios"]["A_all_rows"]["false_positive_feedback"]
        e = entry["scenarios"]["E_strict_cross_fold"]["false_positive_feedback"]
        print("seed %d done in %.0f s | FPR A %.4f->%.4f | FPR E %.4f->%.4f"
              % (seed, entry["seconds"], a["control"]["false_positive_rate"], a["feedback"]["false_positive_rate"],
                 e["control"]["false_positive_rate"], e["feedback"]["false_positive_rate"]), flush=True)

    # tidy table: one row per seed, scenario, variant
    rows = []
    for seed in seeds:
        for scenario in keep:
            for variant in ("false_positive_feedback", "wrong_attack_class_feedback"):
                r = results["per_seed"][str(seed)]["scenarios"][scenario][variant]
                rows.append({
                    "seed": seed, "scenario": scenario, "variant": variant, "evaluated_rows": r["evaluated_rows"],
                    "control_false_positive_rate": r["control"]["false_positive_rate"],
                    "feedback_false_positive_rate": r["feedback"]["false_positive_rate"],
                    "control_wrong_attack_class_rate": r["control"]["wrong_attack_class_rate"],
                    "feedback_wrong_attack_class_rate": r["feedback"]["wrong_attack_class_rate"],
                    "control_attack_recall_binary": r["control"]["attack_recall_binary"],
                    "feedback_attack_recall_binary": r["feedback"]["attack_recall_binary"],
                    "control_accuracy": r["control"]["accuracy"], "feedback_accuracy": r["feedback"]["accuracy"],
                    "corrected_errors": r["paired_vs_control"]["corrected_errors"],
                    "introduced_errors": r["paired_vs_control"]["introduced_errors"],
                    "mcnemar_exact_p": r["paired_vs_control"]["mcnemar_exact_p"],
                })
    pd.DataFrame(rows).to_csv(output_dir / "sensitivity_table.csv", index=False)
    (output_dir / "sensitivity_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--original-script", type=Path, default=Path(__file__).resolve().parents[1] / "original" / "full_dataset_auto_feedback.py")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    out = run(args.data_root, args.original_script, args.output_dir, args.seeds, args.folds)
    print(json.dumps(out["rows_evaluated_per_scenario"], indent=2))
