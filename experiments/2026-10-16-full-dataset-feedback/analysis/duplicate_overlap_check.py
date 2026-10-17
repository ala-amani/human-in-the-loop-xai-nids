"""Leakage checks for the NSL-KDD automatic-feedback experiment.

Counts, on the raw feature vectors (label and difficulty columns excluded):

  * duplicate rows inside KDDTrain+ and inside KDDTest+;
  * test rows whose feature vector also appears in the training file;
  * for the fold split used by the experiment, holdout rows that have an exact
    duplicate inside their own feedback partition - the only way a holdout row's
    label could reach its candidate model without the row itself being used.

No model is trained and nothing from the original run is read or modified.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold


def load_original(path: Path):
    spec = importlib.util.spec_from_file_location("original_experiment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(data_root: Path, original_script: Path, output_dir: Path, folds: int = 5) -> dict:
    orig = load_original(original_script)
    X_train, y_train = orig.load_file(data_root / "KDDTrain+.txt")
    X_test, y_test = orig.load_file(data_root / "KDDTest+.txt")
    key_train = X_train.astype(str).agg("|".join, axis=1)
    key_test = X_test.astype(str).agg("|".join, axis=1)

    train_labels = pd.Series(y_train.values, index=key_train.values)
    train_labels = train_labels[~train_labels.index.duplicated()]
    in_train = key_test.isin(set(key_train))
    conflicting = int(sum(1 for k, lab in zip(key_test[in_train], y_test[in_train.values])
                          if train_labels.get(k) != lab))

    report = {
        "train_rows": int(len(key_train)), "test_rows": int(len(key_test)),
        "duplicate_rows_within_train": int(key_train.duplicated().sum()),
        "duplicate_rows_within_test": int(key_test.duplicated().sum()),
        "test_rows_also_present_in_train": int(in_train.sum()),
        "test_rows_also_present_in_train_with_different_label": conflicting,
        "fold_split": "StratifiedKFold(n_splits=%d, shuffle=True, random_state=42) over the test set" % folds,
        "folds": [],
    }

    splits = list(StratifiedKFold(n_splits=folds, shuffle=True, random_state=42).split(X_test, y_test))
    total_pairs = total_conflict = 0
    for fold, (feedback_idx, holdout_idx) in enumerate(splits, start=1):
        feedback_keys = {}
        for j in feedback_idx:
            feedback_keys.setdefault(key_test.iloc[j], y_test.iloc[j])
        hits = [j for j in holdout_idx if key_test.iloc[j] in feedback_keys]
        conflict = sum(1 for j in hits if feedback_keys[key_test.iloc[j]] != y_test.iloc[j])
        total_pairs += len(hits)
        total_conflict += conflict
        report["folds"].append({
            "fold": fold, "holdout_rows": int(len(holdout_idx)),
            "holdout_rows_with_duplicate_in_feedback_partition": int(len(hits)),
            "of_which_label_differs": int(conflict),
            "same_label_duplicates": int(len(hits) - conflict),
        })
    report["totals"] = {
        "holdout_rows_with_duplicate_in_feedback_partition": int(total_pairs),
        "share_of_test_set": float(total_pairs / len(key_test)),
        "same_label_duplicates": int(total_pairs - total_conflict),
        "share_of_test_set_same_label": float((total_pairs - total_conflict) / len(key_test)),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "duplicate_overlap_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--original-script", type=Path, default=Path(__file__).resolve().parents[1] / "original" / "full_dataset_auto_feedback.py")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(run(args.data_root, args.original_script, args.output_dir, args.folds), indent=2))
