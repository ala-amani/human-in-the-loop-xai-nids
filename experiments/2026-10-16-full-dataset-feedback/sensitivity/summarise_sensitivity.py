"""Readable summaries of the duplicate-overlap sensitivity analysis.

Produces, from sensitivity_results.json:
  * per_seed_table.csv   - one row per seed, scenario and variant, with every requested metric
  * scenario_summary.csv - across-seed mean, minimum and maximum per scenario

Across-seed minimum and maximum describe run-to-run variation of the random forest. They are not
confidence intervals; the Wilson 95% intervals in the JSON are the statistical intervals.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from pathlib import Path

VARIANTS = ("false_positive_feedback", "wrong_attack_class_feedback")
RATE_KEY = {"false_positive_feedback": "false_positive_rate", "wrong_attack_class_feedback": "wrong_attack_class_rate"}


def main(results_path: Path, output_dir: Path) -> None:
    data = json.loads(results_path.read_text(encoding="utf-8"))
    seeds = [str(s) for s in data["seeds"]]
    scenarios = list(data["scenarios"])

    rows = []
    for seed in seeds:
        for scenario in scenarios:
            for variant in VARIANTS:
                r = data["per_seed"][seed]["scenarios"][scenario][variant]
                c, f, p = r["control"], r["feedback"], r["paired_vs_control"]
                rows.append({
                    "seed": seed, "scenario": scenario, "variant": variant, "evaluated_rows": r["evaluated_rows"],
                    "baseline_fpr": round(c["false_positive_rate"], 6), "feedback_fpr": round(f["false_positive_rate"], 6),
                    "baseline_wrong_class_rate": round(c["wrong_attack_class_rate"], 6),
                    "feedback_wrong_class_rate": round(f["wrong_attack_class_rate"], 6),
                    "baseline_attack_recall": round(c["attack_recall_binary"], 6),
                    "feedback_attack_recall": round(f["attack_recall_binary"], 6),
                    "baseline_accuracy": round(c["accuracy"], 6), "feedback_accuracy": round(f["accuracy"], 6),
                    "corrected_errors": p["corrected_errors"], "introduced_errors": p["introduced_errors"],
                    "mcnemar_exact_p": p["mcnemar_exact_p"],
                    "relative_change_target_metric_pct": round(
                        (f[RATE_KEY[variant]] - c[RATE_KEY[variant]]) / c[RATE_KEY[variant]] * 100, 2)
                    if c[RATE_KEY[variant]] else None,
                })
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "per_seed_table.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for scenario in scenarios:
        for variant in VARIANTS:
            sel = [r for r in rows if r["scenario"] == scenario and r["variant"] == variant]
            rel = [r["relative_change_target_metric_pct"] for r in sel if r["relative_change_target_metric_pct"] is not None]
            key_c = "baseline_fpr" if variant == "false_positive_feedback" else "baseline_wrong_class_rate"
            key_f = "feedback_fpr" if variant == "false_positive_feedback" else "feedback_wrong_class_rate"
            summary.append({
                "scenario": scenario, "variant": variant, "evaluated_rows": sel[0]["evaluated_rows"],
                "baseline_rate_mean": round(st.mean(r[key_c] for r in sel), 6),
                "feedback_rate_mean": round(st.mean(r[key_f] for r in sel), 6),
                "relative_change_pct_mean": round(st.mean(rel), 2),
                "relative_change_pct_min_across_seeds": round(min(rel), 2),
                "relative_change_pct_max_across_seeds": round(max(rel), 2),
                "corrected_errors_min": min(r["corrected_errors"] for r in sel),
                "corrected_errors_max": max(r["corrected_errors"] for r in sel),
                "introduced_errors_min": min(r["introduced_errors"] for r in sel),
                "introduced_errors_max": max(r["introduced_errors"] for r in sel),
                "recall_delta_mean": round(st.mean(r["feedback_attack_recall"] - r["baseline_attack_recall"] for r in sel), 6),
                "accuracy_delta_mean": round(st.mean(r["feedback_accuracy"] - r["baseline_accuracy"] for r in sel), 6),
            })
    with open(output_dir / "scenario_summary.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    print("rows evaluated per scenario:", data["rows_evaluated_per_scenario"])
    print("\n%-28s %-26s %9s %9s %10s %14s" % ("scenario", "variant", "baseline", "feedback", "mean rel.", "seed range"))
    for s in summary:
        print("%-28s %-26s %8.4f %9.4f %9.1f%% %7.1f..%+.1f%%"
              % (s["scenario"], s["variant"], s["baseline_rate_mean"], s["feedback_rate_mean"],
                 s["relative_change_pct_mean"], s["relative_change_pct_min_across_seeds"],
                 s["relative_change_pct_max_across_seeds"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    main(args.results, args.output_dir)
