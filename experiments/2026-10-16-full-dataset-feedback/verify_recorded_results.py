"""Validate archived numerical claims without requiring the NSL-KDD input files.

This checks recorded JSON outputs, not independent recomputation from raw data.
To reproduce the experiment, execute the controlled and sensitivity scripts
against official KDDTrain+.txt and KDDTest+.txt as described in README.md.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEEDS = [42, 43, 44, 45, 46]


def close(actual: float, expected: float, tolerance: float = 0.00055) -> None:
    assert math.isclose(actual, expected, abs_tol=tolerance), (actual, expected)


def verify() -> None:
    controlled = json.loads((HERE / "controlled/controlled_results.json").read_text(encoding="utf-8"))
    sensitivity = json.loads((HERE / "sensitivity/sensitivity_results.json").read_text(encoding="utf-8"))
    overlap = json.loads((HERE / "analysis/duplicate_overlap_report.json").read_text(encoding="utf-8"))
    assert controlled["training_rows"] == 125973
    assert controlled["test_rows"] == 22544
    assert controlled["seeds"] == SEEDS
    assert controlled["folds"] == 5
    assert all(controlled["leakage_guards"].values())
    assert len(controlled["fold_records"]) == len(SEEDS) * 5 * 2
    assert set(controlled["per_seed"]) == {str(s) for s in SEEDS}
    assert sensitivity["seeds"] == SEEDS
    assert sensitivity["rows_evaluated_per_scenario"]["E_strict_cross_fold"] == 21796
    assert overlap["train_rows"] == 125973 and overlap["test_rows"] == 22544

    for metric, variant, expected_before, expected_after, expected_relative in [
        ("false_positive_rate", "false_positive_feedback", 0.0258, 0.0174, 0.327),
        ("wrong_attack_class_rate", "wrong_attack_class_feedback", 0.0168, 0.0011, 0.931),
    ]:
        rows = [controlled["per_seed"][str(s)] for s in SEEDS]
        before = [row["control"][metric] for row in rows]
        after = [row["feedback"][variant][metric] for row in rows]
        mean_before = sum(before) / len(before)
        mean_after = sum(after) / len(after)
        relative = sum((b - a) / b for a, b in zip(after, before)) / len(before)
        close(mean_before, expected_before)
        close(mean_after, expected_after)
        close(relative, expected_relative, tolerance=0.0007)
        reported = controlled["across_seed_summary"][f"{metric}__{variant}"]
        close(reported["control"]["mean"], mean_before, 1e-12)
        close(reported["feedback"]["mean"], mean_after, 1e-12)

    fp_recall = controlled["across_seed_summary"]["attack_recall_binary__false_positive_feedback"]
    wc_accuracy = controlled["across_seed_summary"]["accuracy__wrong_attack_class_feedback"]
    close(sum(fp_recall["delta_per_seed"]) / 5, -0.0059)
    close(sum(wc_accuracy["delta_per_seed"]) / 5, 0.0109)

    for variant, metric, expected in [
        ("false_positive_feedback", "false_positive_rate", 0.354),
        ("wrong_attack_class_feedback", "wrong_attack_class_rate", 0.931),
    ]:
        ratios = []
        for seed in SEEDS:
            entry = sensitivity["per_seed"][str(seed)]["scenarios"]["E_strict_cross_fold"][variant]
            before = entry["control"][metric]
            after = entry["feedback"][metric]
            ratios.append((before - after) / before)
        close(sum(ratios) / len(ratios), expected, tolerance=0.0007)

    print("PASS: archived matched-seed and strict-sensitivity results agree with manuscript Section 6.9")
    print("NOTE: this checks saved outputs; raw NSL-KDD files are required to rerun experiments")


if __name__ == "__main__":
    verify()
