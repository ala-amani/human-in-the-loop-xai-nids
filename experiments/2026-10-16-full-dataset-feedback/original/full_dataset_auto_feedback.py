"""Local-only, leakage-safe automatic feedback experiment for NSL-KDD.

This file is intentionally kept under tmp/ and is not part of the GitHub codebase.
It simulates validated human labels so that every baseline error in each feedback
partition can be returned to the IDS without opening the interactive GUI.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty",
]
DOS = {"neptune", "back", "land", "pod", "smurf", "teardrop", "mailbomb", "apache2", "processtable", "udpstorm", "worm"}
PROBE = {"ipsweep", "nmap", "portsweep", "satan", "mscan", "saint"}
R2L = {"ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy", "warezclient", "warezmaster", "sendmail", "named", "snmpgetattack", "snmpguess", "xlock", "xsnoop", "httptunnel"}
U2R = {"buffer_overflow", "loadmodule", "perl", "rootkit", "ps", "sqlattack", "xterm"}


def family(value: str) -> str:
    value = value.strip().rstrip(".").casefold()
    if value == "normal": return "Normal"
    if value in DOS: return "DoS"
    if value in PROBE: return "Probe"
    if value in R2L: return "R2L"
    if value in U2R: return "U2R"
    raise ValueError(value)


def load_file(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.read_csv(path, names=COLUMNS)
    y = frame.pop("label").map(family).reset_index(drop=True)
    return frame.drop(columns=["difficulty"]).reset_index(drop=True), y


def preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", MinMaxScaler())]), numeric),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])


def classifier(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=2, random_state=seed, n_jobs=-1)


def masks(y: np.ndarray, pred: np.ndarray) -> dict[str, np.ndarray]:
    normal = y == "Normal"
    attack = ~normal
    return {
        "false_positive": normal & (pred != "Normal"),
        "wrong_attack_class": attack & (pred != "Normal") & (pred != y),
        "false_negative": attack & (pred == "Normal"),
    }


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    m = masks(y, pred)
    n_normal = int((y == "Normal").sum())
    n_attack = int((y != "Normal").sum())
    return {
        "samples": int(len(y)),
        "normal_samples": n_normal,
        "attack_samples": n_attack,
        "false_positives": int(m["false_positive"].sum()),
        "false_positive_rate": float(m["false_positive"].sum() / n_normal),
        "wrong_attack_class": int(m["wrong_attack_class"].sum()),
        "wrong_attack_class_rate": float(m["wrong_attack_class"].sum() / n_attack),
        "false_negatives": int(m["false_negative"].sum()),
        "false_negative_rate": float(m["false_negative"].sum() / n_attack),
        "attack_recall_binary": float(recall_score(y != "Normal", pred != "Normal")),
        "accuracy": float(accuracy_score(y, pred)),
    }


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def paired_change(before_error: np.ndarray, after_error: np.ndarray) -> dict[str, float | int]:
    corrected = int((before_error & ~after_error).sum())
    introduced = int((~before_error & after_error).sum())
    discordant = corrected + introduced
    p_value = float(binomtest(min(corrected, introduced), discordant, 0.5).pvalue) if discordant else 1.0
    return {"corrected_errors": corrected, "introduced_errors": introduced, "mcnemar_exact_p": p_value}


def run(data_root: Path, output_dir: Path, seed: int = 42, folds: int = 5) -> dict:
    X_train, y_train = load_file(data_root / "KDDTrain+.txt")
    X_test, y_test = load_file(data_root / "KDDTest+.txt")
    prep = preprocessor(X_train)
    Xt = prep.fit_transform(X_train)
    Xq = prep.transform(X_test)
    ytr = y_train.to_numpy()
    yq = y_test.to_numpy()

    base = classifier(seed).fit(Xt, ytr)
    baseline_pred = base.predict(Xq)
    base_masks = masks(yq, baseline_pred)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    variants = {"false_positive_feedback": base_masks["false_positive"], "wrong_attack_class_feedback": base_masks["wrong_attack_class"]}
    post_predictions = {name: np.empty(len(yq), dtype=object) for name in variants}
    fold_records: list[dict] = []

    for fold, (feedback_idx, holdout_idx) in enumerate(splitter.split(Xq, yq), start=1):
        for name, eligible in variants.items():
            corrected_idx = feedback_idx[eligible[feedback_idx]]
            X_aug = np.vstack([Xt.toarray() if hasattr(Xt, "toarray") else Xt, Xq[corrected_idx].toarray() if hasattr(Xq, "toarray") else Xq[corrected_idx]])
            y_aug = np.concatenate([ytr, yq[corrected_idx]])
            model = classifier(seed + fold).fit(X_aug, y_aug)
            pred = model.predict(Xq[holdout_idx])
            post_predictions[name][holdout_idx] = pred
            fold_records.append({"fold": fold, "variant": name, "feedback_errors": int(len(corrected_idx)), **metrics(yq[holdout_idx], pred)})

    before = metrics(yq, baseline_pred)
    before["false_positive_rate_ci95"] = wilson(before["false_positives"], before["normal_samples"])
    before["wrong_attack_class_rate_ci95"] = wilson(before["wrong_attack_class"], before["attack_samples"])
    after = {}
    for name, pred in post_predictions.items():
        after[name] = metrics(yq, pred)
        after[name]["false_positive_rate_ci95"] = wilson(after[name]["false_positives"], after[name]["normal_samples"])
        after[name]["wrong_attack_class_rate_ci95"] = wilson(after[name]["wrong_attack_class"], after[name]["attack_samples"])

    normal = yq == "Normal"
    attack = ~normal
    paired = {
        "false_positive_feedback": paired_change(
            base_masks["false_positive"][normal],
            masks(yq, post_predictions["false_positive_feedback"])["false_positive"][normal],
        ),
        "wrong_attack_class_feedback": paired_change(
            base_masks["wrong_attack_class"][attack],
            masks(yq, post_predictions["wrong_attack_class_feedback"])["wrong_attack_class"][attack],
        ),
    }

    result = {
        "protocol": "5-fold cross-fitted automatic analyst-feedback simulation over the complete official NSL-KDD test set",
        "training_rows": int(len(ytr)), "test_rows": int(len(yq)), "seed": seed, "folds": folds,
        "baseline": before, "after": after,
        "unique_baseline_false_positives": int(base_masks["false_positive"].sum()),
        "unique_baseline_wrong_attack_class": int(base_masks["wrong_attack_class"].sum()),
        "paired_tests": paired, "folds_detail": fold_records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "full_dataset_feedback_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(fold_records).to_csv(output_dir / "full_dataset_feedback_folds.csv", index=False)

    panels = [
        ("False-positive feedback", before, after["false_positive_feedback"], "false_positive_rate", "false_positive_rate_ci95"),
        ("Wrong attack-class feedback", before, after["wrong_attack_class_feedback"], "wrong_attack_class_rate", "wrong_attack_class_rate_ci95"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.4))
    for ax, (title, b, a, key, ci_key) in zip(axes, panels):
        values = np.array([b[key], a[key]]) * 100
        cis = [b[ci_key], a[ci_key]]
        errors = np.array([[v - lo * 100 for v, (lo, hi) in zip(values, cis)], [hi * 100 - v for v, (lo, hi) in zip(values, cis)]])
        bars = ax.bar([0, 1], values, color=["#4472C4", "#70AD47"], width=.58, yerr=errors, capsize=5)
        ax.bar_label(bars, fmt="%.2f%%", padding=7, fontsize=10)
        ax.set_xticks([0, 1], ["Before", "After"]); ax.set_title(title, fontsize=11); ax.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("Rate (%)")
    fig.suptitle("Cross-fitted NSL-KDD error rates with 95% confidence intervals", fontsize=13)
    fig.tight_layout(); fig.savefig(output_dir / "before_after_error_rates.png", dpi=220); plt.close(fig)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(run(args.data_root, args.output_dir, args.seed, args.folds), indent=2))
