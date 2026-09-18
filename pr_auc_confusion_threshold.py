#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Priority 11: PR-AUC + confusion matrix + decision threshold
===========================================================

Reviewer request
----------------
"Report AUC-PR, the confusion matrix and the decision threshold,
especially if classes are imbalanced."

This analysis does NOT retrain the model. It uses the per-match mean predicted
probability across the 10 CatBoost random-seed runs generated in Priority 4.

Default input:
    /data2/zhangyiyi/beach_ai/final_catboost_v1/
    auc_uncertainty_delong_output/
    baseline_test_predictions_mean_10seeds.csv

Required columns:
    y
    catboost_probability_mean_10seeds
"""

import os
import json
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    auc,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

DEFAULT_PREDICTIONS = (
    "/data2/zhangyiyi/beach_ai/final_catboost_v1/"
    "auc_uncertainty_delong_output/"
    "baseline_test_predictions_mean_10seeds.csv"
)
DEFAULT_OUTDIR = (
    "/data2/zhangyiyi/beach_ai/final_catboost_v1/"
    "pr_auc_confusion_output"
)
PROB_COL_DEFAULT = "catboost_probability_mean_10seeds"
THRESHOLD_DEFAULT = 0.50


def load_predictions(path, prob_col):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Prediction file not found: {path}\n"
            "Use --predictions /path/to/file.csv if your file is elsewhere."
        )
    df = pd.read_csv(path)
    required = ["y", prob_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing columns: " + ", ".join(missing) +
            "\nAvailable columns: " + ", ".join(df.columns.astype(str))
        )

    df = df.copy()
    df["y"] = pd.to_numeric(df["y"], errors="raise").astype(int)
    df[prob_col] = pd.to_numeric(df[prob_col], errors="raise").astype(float)

    if not set(df["y"].unique()).issubset({0, 1}):
        raise ValueError("y must contain only 0 and 1.")
    if df[prob_col].isna().any():
        raise ValueError(f"{prob_col} contains missing values.")
    if ((df[prob_col] < 0) | (df[prob_col] > 1)).any():
        raise ValueError("Predicted probabilities must be in [0,1].")
    return df


def compute_metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y, p)
    pr_auc = auc(recall_curve, precision_curve)
    ap = average_precision_score(y, p)

    prevalence = float(np.mean(y))

    return {
        "n_test": int(len(y)),
        "positive_n": int(np.sum(y == 1)),
        "negative_n": int(np.sum(y == 0)),
        "positive_prevalence": prevalence,
        "decision_threshold": float(threshold),
        "pr_auc": float(pr_auc),
        "average_precision": float(ap),
        "no_skill_pr_baseline": prevalence,
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "f1_score": float(f1_score(y, pred, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }, (precision_curve, recall_curve, pr_thresholds), pred, cm


def make_table(m):
    return pd.DataFrame([
        ["Held-out test set, N", f"{m['n_test']:,}"],
        ["Team A wins, n (%)", f"{m['positive_n']:,} ({100*m['positive_prevalence']:.1f}%)"],
        ["Decision threshold", f"{m['decision_threshold']:.2f}"],
        ["PR-AUC", f"{m['pr_auc']:.4f}"],
        ["Average precision", f"{m['average_precision']:.4f}"],
        ["Accuracy", f"{m['accuracy']:.4f}"],
        ["Precision", f"{m['precision']:.4f}"],
        ["Recall (sensitivity)", f"{m['recall_sensitivity']:.4f}"],
        ["Specificity", f"{m['specificity']:.4f}"],
        ["F1-score", f"{m['f1_score']:.4f}"],
        ["True negatives (TN)", f"{m['tn']:,}"],
        ["False positives (FP)", f"{m['fp']:,}"],
        ["False negatives (FN)", f"{m['fn']:,}"],
        ["True positives (TP)", f"{m['tp']:,}"],
    ], columns=["Metric", "Test value"])


def save_confusion_matrix(cm, outdir, threshold):
    labels = ["Team B win (0)", "Team A win (1)"]
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(
        cm, row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0
    ) * 100

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(cm, aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted outcome")
    ax.set_ylabel("Observed outcome")
    ax.set_title(f"Confusion matrix on held-out test set (threshold = {threshold:.2f})")

    for i in range(2):
        for j in range(2):
            ax.text(
                j, i,
                f"{cm[i, j]:,}\n({cm_pct[i, j]:.1f}%)",
                ha="center", va="center"
            )

    fig.colorbar(im, ax=ax, label="Number of matches")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_confusion_matrix_test.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(outdir, "fig_confusion_matrix_test.pdf"), bbox_inches="tight")
    plt.close(fig)


def save_pr_curve(precision_curve, recall_curve, pr_auc, prevalence, outdir):
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.plot(
        recall_curve, precision_curve,
        linewidth=2,
        label=f"CatBoost (PR-AUC = {pr_auc:.3f})"
    )
    ax.axhline(
        y=prevalence,
        linestyle="--",
        linewidth=1.2,
        label=f"No-skill baseline = {prevalence:.3f}"
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curve on held-out test set")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_precision_recall_curve_test.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(outdir, "fig_precision_recall_curve_test.pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Priority 11: PR-AUC, confusion matrix, and threshold."
    )
    parser.add_argument("--predictions", default=DEFAULT_PREDICTIONS)
    parser.add_argument("--prob-col", default=PROB_COL_DEFAULT)
    parser.add_argument("--threshold", type=float, default=THRESHOLD_DEFAULT)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    if not 0 <= args.threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 82)
    print("PRIORITY 11: PR-AUC + CONFUSION MATRIX + DECISION THRESHOLD")
    print("=" * 82)
    print("Predictions:", args.predictions)
    print("Probability column:", args.prob_col)
    print("Threshold:", args.threshold)
    print("Output:", args.outdir)

    df = load_predictions(args.predictions, args.prob_col)
    y = df["y"].to_numpy()
    p = df[args.prob_col].to_numpy()

    m, pr_data, pred, cm = compute_metrics(y, p, args.threshold)
    precision_curve, recall_curve, pr_thresholds = pr_data

    table = make_table(m)
    table.to_csv(
        os.path.join(args.outdir, "supplementary_table_pr_confusion_threshold.csv"),
        index=False
    )

    cm_df = pd.DataFrame(
        cm,
        index=["Observed Team B win (0)", "Observed Team A win (1)"],
        columns=["Predicted Team B win (0)", "Predicted Team A win (1)"],
    )
    cm_df.to_csv(os.path.join(args.outdir, "confusion_matrix_counts.csv"))

    pred_out = pd.DataFrame({
        "y": y,
        "probability_mean_10seeds": p,
        "predicted_class_threshold_0.50": pred,
    })
    if "date" in df.columns:
        pred_out.insert(0, "date", df["date"].astype(str).to_numpy())
    pred_out.to_csv(
        os.path.join(args.outdir, "test_predictions_with_threshold.csv"),
        index=False
    )

    pr_df = pd.DataFrame({
        "recall": recall_curve,
        "precision": precision_curve,
    })
    pr_df["threshold"] = np.nan
    if len(pr_thresholds):
        pr_df.loc[:len(pr_thresholds)-1, "threshold"] = pr_thresholds
    pr_df.to_csv(os.path.join(args.outdir, "pr_curve_data.csv"), index=False)

    save_confusion_matrix(cm, args.outdir, args.threshold)
    save_pr_curve(
        precision_curve, recall_curve,
        m["pr_auc"], m["positive_prevalence"], args.outdir
    )

    config = {
        "prediction_file": args.predictions,
        "probability_column": args.prob_col,
        "decision_threshold": args.threshold,
        "positive_class": "Team A win (y=1)",
        "negative_class": "Team B win / Team A loss (y=0)",
        "prediction_definition": "Mean CatBoost probability across 10 random-seed runs",
        "pr_auc_definition": "Trapezoidal area under the precision-recall curve",
        "average_precision_also_reported": True,
        "metrics": m,
    }
    with open(os.path.join(args.outdir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    readme = f"""
Priority 11: PR-AUC + confusion matrix + threshold

No model retraining was performed.
Predictions are the mean CatBoost probabilities across 10 random seeds.
Positive class: Team A win (y=1).
Decision threshold: {args.threshold:.2f}.

Main outputs:
- supplementary_table_pr_confusion_threshold.csv
- confusion_matrix_counts.csv
- fig_confusion_matrix_test.png/pdf

Optional:
- fig_precision_recall_curve_test.png/pdf
- pr_curve_data.csv
- test_predictions_with_threshold.csv

Recommended manuscript presentation:
Use one supplementary table plus the confusion-matrix figure.
The PR curve is optional.
""".strip()
    with open(os.path.join(args.outdir, "README_priority11.txt"), "w", encoding="utf-8") as f:
        f.write(readme + "\n")

    print("\nRESULTS")
    print(table.to_string(index=False))
    print("\nCONFUSION MATRIX")
    print(cm_df.to_string())
    print("\nSaved to:", args.outdir)


if __name__ == "__main__":
    main()
