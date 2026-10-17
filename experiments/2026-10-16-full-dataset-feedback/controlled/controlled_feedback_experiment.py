"""Controlled re-evaluation of the NSL-KDD automatic-feedback experiment.

The original run (tmp/full_dataset_auto_feedback.py, executed 2026-09-11) compared a
baseline fitted with random_state=42 against feedback models fitted with random_state
42+fold, so the random seed and the feedback were changed at the same time. This script
removes that confound:

  * the fold split is the one the original used, StratifiedKFold(5, shuffle, seed 42);
  * for every seed S, a no-feedback control and the feedback models all use random_state=S
    and identical hyperparameters, so the only difference between them is the feedback data;
  * the whole comparison is repeated over several seeds, which gives an honest picture of
    how large the improvement is relative to seed-to-seed variation.

Loading, preprocessing, error masks, metrics, Wilson intervals and the exact McNemar test
are imported from the original script, so the definitions cannot drift. Nothing the original
run produced is read, modified or overwritten here.

Usage (from the repository root):

    python experiments/2026-10-16-full-dataset-feedback/controlled/controlled_feedback_experiment.py \
        --data-root "<path>/detection data/2_raw/NSL-KDD" \
                --output-dir experiments/2026-10-16-full-dataset-feedback/controlled \
        --seeds 42 43 44 45 46
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


def load_original(path: Path):
    spec = importlib.util.spec_from_file_location("original_experiment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def densify(matrix):
    return matrix.toarray() if hasattr(matrix, "toarray") else matrix


def run(data_root: Path, original_script: Path, output_dir: Path, seeds: list[int], folds: int = 5) -> dict:
    orig = load_original(original_script)
    X_train, y_train = orig.load_file(data_root / "KDDTrain+.txt")
    X_test, y_test = orig.load_file(data_root / "KDDTest+.txt")
    prep = orig.preprocessor(X_train)                 # fitted on training rows only
    Xt = densify(prep.fit_transform(X_train))
    Xq = densify(prep.transform(X_test))
    ytr, yq = y_train.to_numpy(), y_test.to_numpy()

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    splits = list(splitter.split(Xq, yq))

    # ── leakage guards that must hold for every fold, recorded in the output
    guards = {"holdout_rows_cover_test_set_once": True, "feedback_and_holdout_disjoint": True,
              "corrected_rows_inside_feedback_partition": True}
    covered = np.zeros(len(yq), dtype=int)
    for feedback_idx, holdout_idx in splits:
        covered[holdout_idx] += 1
        if set(feedback_idx) & set(holdout_idx):
            guards["feedback_and_holdout_disjoint"] = False
    guards["holdout_rows_cover_test_set_once"] = bool((covered == 1).all())

    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {
        "design": ("per-seed no-feedback control and feedback models trained with identical hyperparameters "
                   "and identical random_state; fold split identical to the original run"),
        "training_rows": int(len(ytr)), "test_rows": int(len(yq)), "folds": folds,
        "fold_split": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42) over the test set",
        "seeds": seeds, "leakage_guards": guards, "per_seed": {}, "fold_records": [],
    }

    for seed in seeds:
        started = time.time()
        control = orig.classifier(seed).fit(Xt, ytr)        # train rows only
        control_pred = control.predict(Xq)
        control_masks = orig.masks(yq, control_pred)
        variants = {"false_positive_feedback": control_masks["false_positive"],
                    "wrong_attack_class_feedback": control_masks["wrong_attack_class"]}
        feedback_pred = {name: np.empty(len(yq), dtype=object) for name in variants}

        for fold, (feedback_idx, holdout_idx) in enumerate(splits, start=1):
            for name, eligible in variants.items():
                corrected_idx = feedback_idx[eligible[feedback_idx]]
                if set(corrected_idx) - set(feedback_idx):
                    guards["corrected_rows_inside_feedback_partition"] = False
                X_aug = np.vstack([Xt, Xq[corrected_idx]])
                y_aug = np.concatenate([ytr, yq[corrected_idx]])
                model = orig.classifier(seed).fit(X_aug, y_aug)        # same seed as the control
                pred = model.predict(Xq[holdout_idx])
                feedback_pred[name][holdout_idx] = pred
                results["fold_records"].append({
                    "seed": seed, "fold": fold, "variant": name,
                    "feedback_rows": int(len(corrected_idx)),
                    "control": orig.metrics(yq[holdout_idx], control_pred[holdout_idx]),
                    "feedback": orig.metrics(yq[holdout_idx], pred),
                })

        entry = {"control": orig.metrics(yq, control_pred), "feedback": {}, "paired_vs_control": {}}
        entry["control"]["false_positive_rate_ci95"] = orig.wilson(entry["control"]["false_positives"],
                                                                   entry["control"]["normal_samples"])
        entry["control"]["wrong_attack_class_rate_ci95"] = orig.wilson(entry["control"]["wrong_attack_class"],
                                                                       entry["control"]["attack_samples"])
        normal, attack = yq == "Normal", yq != "Normal"
        for name, pred in feedback_pred.items():
            m = orig.metrics(yq, pred)
            m["false_positive_rate_ci95"] = orig.wilson(m["false_positives"], m["normal_samples"])
            m["wrong_attack_class_rate_ci95"] = orig.wilson(m["wrong_attack_class"], m["attack_samples"])
            entry["feedback"][name] = m
            after = orig.masks(yq, pred)
            if name == "false_positive_feedback":
                entry["paired_vs_control"][name] = orig.paired_change(control_masks["false_positive"][normal],
                                                                      after["false_positive"][normal])
            else:
                entry["paired_vs_control"][name] = orig.paired_change(control_masks["wrong_attack_class"][attack],
                                                                      after["wrong_attack_class"][attack])
        entry["seconds"] = round(time.time() - started, 1)
        results["per_seed"][str(seed)] = entry
        (output_dir / "controlled_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        pd.DataFrame(results["fold_records"]).to_json(output_dir / "controlled_folds.json", orient="records", indent=1)
        print("seed %d done in %.0f s | control FPR %.4f -> feedback FPR %.4f | control WC %.4f -> feedback WC %.4f"
              % (seed, entry["seconds"], entry["control"]["false_positive_rate"],
                 entry["feedback"]["false_positive_feedback"]["false_positive_rate"],
                 entry["control"]["wrong_attack_class_rate"],
                 entry["feedback"]["wrong_attack_class_feedback"]["wrong_attack_class_rate"]), flush=True)

    # ── across-seed summary, so the effect can be read against seed-to-seed variation
    def spread(values):
        a = np.asarray(values, dtype=float)
        return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
                "min": float(a.min()), "max": float(a.max())}

    summary = {}
    for key, variant in (("false_positive_rate", "false_positive_feedback"),
                         ("wrong_attack_class_rate", "wrong_attack_class_feedback"),
                         ("attack_recall_binary", "false_positive_feedback"),
                         ("accuracy", "false_positive_feedback")):
        ctrl = [results["per_seed"][str(s)]["control"][key] for s in seeds]
        fb = [results["per_seed"][str(s)]["feedback"][variant][key] for s in seeds]
        summary["%s__%s" % (key, variant)] = {
            "control": spread(ctrl), "feedback": spread(fb),
            "delta_per_seed": [float(f - c) for c, f in zip(ctrl, fb)],
            "relative_change_per_seed": [float((f - c) / c) if c else None for c, f in zip(ctrl, fb)],
        }
    for key in ("attack_recall_binary", "accuracy"):
        ctrl = [results["per_seed"][str(s)]["control"][key] for s in seeds]
        fb = [results["per_seed"][str(s)]["feedback"]["wrong_attack_class_feedback"][key] for s in seeds]
        summary["%s__wrong_attack_class_feedback" % key] = {
            "control": spread(ctrl), "feedback": spread(fb),
            "delta_per_seed": [float(f - c) for c, f in zip(ctrl, fb)],
            "relative_change_per_seed": [float((f - c) / c) if c else None for c, f in zip(ctrl, fb)],
        }
    results["across_seed_summary"] = summary
    (output_dir / "controlled_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
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
    print(json.dumps(out["across_seed_summary"], indent=2))
