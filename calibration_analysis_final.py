#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Priority 7: Calibration analysis for the retained CatBoost model
================================================================

This script addresses Reviewer 2's calibration comment:

"Utility for probabilistic decision support is claimed, but only log-loss is
reported. Please add reliability curves and Brier score / ECE; if calibration
is poor, evaluate Platt or isotonic recalibration."

Inputs
------
Two CSV files containing the mean predicted probability across the 10 CatBoost
random-seed runs:

1) Validation predictions
2) Test predictions

Required columns in each file:
    y
    probability_mean_across_seeds

Design
------
1. Evaluate the original CatBoost probability calibration on validation and test:
   - AUC
   - Brier score
   - ECE (10 equally spaced probability bins)
   - Log-loss
   - Reliability curves

2. Fit post-hoc recalibration ONLY on validation predictions:
   - Platt scaling
   - Isotonic regression

3. Apply the frozen calibrators to the untouched test set.

4. Compare original vs Platt vs isotonic on the test set.

Outputs
-------
Tables:
    supplementary_table_Sx_original_calibration.csv
    supplementary_table_Sy_recalibration_sensitivity.csv

Calibration-bin data:
    calibration_bins_original_validation.csv
    calibration_bins_original_test.csv
    calibration_bins_platt_test.csv
    calibration_bins_isotonic_test.csv

Figures:
    fig_SxA_original_validation_test_reliability.png/pdf
    fig_SxB_test_recalibration_reliability.png/pdf

Additional outputs:
    test_predictions_with_recalibration.csv
    calibration_run_config.json
    README_priority7_calibration.txt

Notes
-----
- The original CatBoost remains the primary model.
- Recalibration is treated as a supplementary sensitivity analysis.
- ECE uses 10 equally spaced probability bins.
- Platt/isotonic fitting uses validation data only; test labels are never used
  to fit the calibrators.
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
    roc_auc_score,
    brier_score_loss,
    log_loss,
)
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression


# ============================================================
# SETTINGS
# ============================================================

N_BINS = 10
EPS = 1e-6


# ============================================================
# DATA UTILITIES
# ============================================================

def load_predictions(path: str):
    df = pd.read_csv(path)

    required = ["y", "probability_mean_across_seeds"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path} is missing required columns: {missing}"
        )

    y = pd.to_numeric(df["y"], errors="raise").astype(int).to_numpy()
    p = pd.to_numeric(
        df["probability_mean_across_seeds"],
        errors="raise"
    ).astype(float).to_numpy()

    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError(
            f"Target in {path} must be binary 0/1. Found: {np.unique(y)}"
        )

    if np.any(~np.isfinite(p)):
        raise ValueError(f"Non-finite probabilities found in {path}.")

    if np.any((p < 0) | (p > 1)):
        raise ValueError(f"Probabilities outside [0, 1] found in {path}.")

    return df, y, p


# ============================================================
# CALIBRATION METRICS
# ============================================================

def calibration_bins_equal_width(y_true, prob, n_bins=N_BINS):
    """
    Equal-width bins over [0,1].

    Returns:
        ece
        dataframe with:
            bin
            lower_bound
            upper_bound
            n
            weight
            mean_predicted_probability
            observed_event_rate
            absolute_gap
            signed_gap_observed_minus_predicted
    """
    y_true = np.asarray(y_true)
    prob = np.asarray(prob, dtype=float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)

    # Bins:
    # [0.0,0.1), [0.1,0.2), ..., [0.9,1.0]
    bin_ids = np.digitize(
        prob,
        edges[1:-1],
        right=False
    )

    rows = []
    ece = 0.0
    n_total = len(y_true)

    for b in range(n_bins):
        mask = bin_ids == b
        n = int(mask.sum())

        lower = edges[b]
        upper = edges[b + 1]

        if n == 0:
            rows.append({
                "bin": b + 1,
                "lower_bound": lower,
                "upper_bound": upper,
                "n": 0,
                "weight": 0.0,
                "mean_predicted_probability": np.nan,
                "observed_event_rate": np.nan,
                "absolute_gap": np.nan,
                "signed_gap_observed_minus_predicted": np.nan,
            })
            continue

        mean_pred = float(prob[mask].mean())
        observed = float(y_true[mask].mean())
        abs_gap = abs(mean_pred - observed)
        signed_gap = observed - mean_pred
        weight = n / n_total

        ece += weight * abs_gap

        rows.append({
            "bin": b + 1,
            "lower_bound": lower,
            "upper_bound": upper,
            "n": n,
            "weight": weight,
            "mean_predicted_probability": mean_pred,
            "observed_event_rate": observed,
            "absolute_gap": abs_gap,
            "signed_gap_observed_minus_predicted": signed_gap,
        })

    return float(ece), pd.DataFrame(rows)


def evaluate_calibration(y_true, prob, label):
    prob_clip = np.clip(prob, EPS, 1 - EPS)
    ece, bins_df = calibration_bins_equal_width(
        y_true,
        prob_clip,
        n_bins=N_BINS
    )

    metrics = {
        "Method": label,
        "AUC": float(roc_auc_score(y_true, prob_clip)),
        "Brier score": float(brier_score_loss(y_true, prob_clip)),
        "ECE": float(ece),
        "Log-loss": float(log_loss(y_true, prob_clip)),
        "Observed prevalence": float(np.mean(y_true)),
        "Mean predicted probability": float(np.mean(prob_clip)),
    }

    return metrics, bins_df


# ============================================================
# RECALIBRATION
# ============================================================

def probability_to_logit(prob):
    p = np.clip(np.asarray(prob, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1.0 - p))


def fit_platt(validation_y, validation_prob):
    """
    Classical Platt-style logistic recalibration:
    logistic regression of observed outcome on the logit of the model probability.
    """
    x = probability_to_logit(validation_prob).reshape(-1, 1)

    # Very weak regularization approximates ordinary logistic calibration.
    model = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=5000
    )
    model.fit(x, validation_y)
    return model


def apply_platt(model, prob):
    x = probability_to_logit(prob).reshape(-1, 1)
    return model.predict_proba(x)[:, 1]


def fit_isotonic(validation_y, validation_prob):
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(validation_prob, validation_y)
    return model


# ============================================================
# PLOTTING
# ============================================================

def plot_original_validation_test(
    valid_bins,
    test_bins,
    out_png,
    out_pdf,
):
    """
    Supplementary Figure SxA:
    Original CatBoost reliability on validation vs test.
    """
    v = valid_bins.dropna(
        subset=["mean_predicted_probability", "observed_event_rate"]
    )
    t = test_bins.dropna(
        subset=["mean_predicted_probability", "observed_event_rate"]
    )

    plt.figure(figsize=(6.5, 6.0))

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.2,
        label="Perfect calibration"
    )

    plt.plot(
        v["mean_predicted_probability"],
        v["observed_event_rate"],
        marker="o",
        linewidth=1.8,
        label="Validation"
    )

    plt.plot(
        t["mean_predicted_probability"],
        t["observed_event_rate"],
        marker="s",
        linewidth=1.8,
        label="Test"
    )

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed Team A win probability")
    plt.title("Original CatBoost calibration")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(loc="upper left")
    plt.tight_layout()

    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()


def plot_test_recalibration(
    original_bins,
    platt_bins,
    isotonic_bins,
    out_png,
    out_pdf,
    show_isotonic=True,
):
    """
    Supplementary Figure SxB:
    Test-set reliability before/after recalibration.
    """
    raw = original_bins.dropna(
        subset=["mean_predicted_probability", "observed_event_rate"]
    )
    platt = platt_bins.dropna(
        subset=["mean_predicted_probability", "observed_event_rate"]
    )
    iso = isotonic_bins.dropna(
        subset=["mean_predicted_probability", "observed_event_rate"]
    )

    plt.figure(figsize=(6.5, 6.0))

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.2,
        label="Perfect calibration"
    )

    plt.plot(
        raw["mean_predicted_probability"],
        raw["observed_event_rate"],
        marker="o",
        linewidth=1.8,
        label="Original CatBoost"
    )

    plt.plot(
        platt["mean_predicted_probability"],
        platt["observed_event_rate"],
        marker="s",
        linewidth=1.8,
        label="Platt scaling"
    )

    if show_isotonic:
        plt.plot(
            iso["mean_predicted_probability"],
            iso["observed_event_rate"],
            marker="^",
            linewidth=1.8,
            label="Isotonic regression"
        )

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed Team A win probability")
    plt.title("Test-set calibration before and after recalibration")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(loc="upper left")
    plt.tight_layout()

    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()


# ============================================================
# TABLE FORMATTING
# ============================================================

def round_table(df, digits=4):
    out = df.copy()

    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].round(digits)

    return out


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Priority 7 calibration analysis for retained CatBoost."
    )

    parser.add_argument(
        "--valid-predictions",
        type=str,
        default=(
            "/data2/zhangyiyi/beach_ai/final_catboost_v1/"
            "priority6_global_subgroup_output/"
            "priority6_global_valid_predictions_mean_10seeds.csv"
        ),
    )

    parser.add_argument(
        "--test-predictions",
        type=str,
        default=(
            "/data2/zhangyiyi/beach_ai/final_catboost_v1/"
            "priority6_global_subgroup_output/"
            "priority6_global_test_predictions_mean_10seeds.csv"
        ),
    )

    parser.add_argument(
        "--outdir",
        type=str,
        default=(
            "/data2/zhangyiyi/beach_ai/final_catboost_v1/"
            "priority7_calibration_output"
        ),
    )

    parser.add_argument(
        "--hide-isotonic-in-figure",
        action="store_true",
        help=(
            "If specified, Figure SxB will show only Original + Platt + "
            "perfect-calibration line. Isotonic metrics remain in the table."
        ),
    )

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 90)
    print("PRIORITY 7: CALIBRATION ANALYSIS")
    print("=" * 90)

    valid_df, y_valid, p_valid = load_predictions(
        args.valid_predictions
    )
    test_df, y_test, p_test = load_predictions(
        args.test_predictions
    )

    print(f"Validation n = {len(y_valid):,}")
    print(f"Test n       = {len(y_test):,}")

    # --------------------------------------------------------
    # Original model
    # --------------------------------------------------------
    valid_original_metrics, valid_original_bins = evaluate_calibration(
        y_valid,
        p_valid,
        "Original CatBoost"
    )

    test_original_metrics, test_original_bins = evaluate_calibration(
        y_test,
        p_test,
        "Original CatBoost"
    )

    # --------------------------------------------------------
    # Platt scaling: fit VALIDATION ONLY
    # --------------------------------------------------------
    platt_model = fit_platt(y_valid, p_valid)

    p_valid_platt = apply_platt(platt_model, p_valid)
    p_test_platt = apply_platt(platt_model, p_test)

    valid_platt_metrics, valid_platt_bins = evaluate_calibration(
        y_valid,
        p_valid_platt,
        "Platt scaling"
    )

    test_platt_metrics, test_platt_bins = evaluate_calibration(
        y_test,
        p_test_platt,
        "Platt scaling"
    )

    # --------------------------------------------------------
    # Isotonic: fit VALIDATION ONLY
    # --------------------------------------------------------
    isotonic_model = fit_isotonic(y_valid, p_valid)

    p_valid_iso = isotonic_model.predict(p_valid)
    p_test_iso = isotonic_model.predict(p_test)

    valid_iso_metrics, valid_iso_bins = evaluate_calibration(
        y_valid,
        p_valid_iso,
        "Isotonic regression"
    )

    test_iso_metrics, test_iso_bins = evaluate_calibration(
        y_test,
        p_test_iso,
        "Isotonic regression"
    )

    # --------------------------------------------------------
    # Supplementary Table Sx:
    # Original retained CatBoost on validation vs test
    # --------------------------------------------------------
    table_sx = pd.DataFrame([
        {
            "Dataset": "Validation",
            "AUC": valid_original_metrics["AUC"],
            "Brier score": valid_original_metrics["Brier score"],
            "ECE": valid_original_metrics["ECE"],
            "Log-loss": valid_original_metrics["Log-loss"],
        },
        {
            "Dataset": "Test",
            "AUC": test_original_metrics["AUC"],
            "Brier score": test_original_metrics["Brier score"],
            "ECE": test_original_metrics["ECE"],
            "Log-loss": test_original_metrics["Log-loss"],
        },
    ])

    table_sx = round_table(table_sx, 4)

    table_sx.to_csv(
        os.path.join(
            args.outdir,
            "supplementary_table_Sx_original_calibration.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Supplementary Table Sy:
    # Test recalibration sensitivity analysis
    # --------------------------------------------------------
    table_sy = pd.DataFrame([
        {
            "Method": "Original CatBoost",
            "AUC": test_original_metrics["AUC"],
            "Brier score": test_original_metrics["Brier score"],
            "ECE": test_original_metrics["ECE"],
            "Log-loss": test_original_metrics["Log-loss"],
        },
        {
            "Method": "Platt scaling",
            "AUC": test_platt_metrics["AUC"],
            "Brier score": test_platt_metrics["Brier score"],
            "ECE": test_platt_metrics["ECE"],
            "Log-loss": test_platt_metrics["Log-loss"],
        },
        {
            "Method": "Isotonic regression",
            "AUC": test_iso_metrics["AUC"],
            "Brier score": test_iso_metrics["Brier score"],
            "ECE": test_iso_metrics["ECE"],
            "Log-loss": test_iso_metrics["Log-loss"],
        },
    ])

    table_sy = round_table(table_sy, 4)

    table_sy.to_csv(
        os.path.join(
            args.outdir,
            "supplementary_table_Sy_recalibration_sensitivity.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Calibration-bin CSVs
    # --------------------------------------------------------
    valid_original_bins.to_csv(
        os.path.join(
            args.outdir,
            "calibration_bins_original_validation.csv"
        ),
        index=False
    )

    test_original_bins.to_csv(
        os.path.join(
            args.outdir,
            "calibration_bins_original_test.csv"
        ),
        index=False
    )

    test_platt_bins.to_csv(
        os.path.join(
            args.outdir,
            "calibration_bins_platt_test.csv"
        ),
        index=False
    )

    test_iso_bins.to_csv(
        os.path.join(
            args.outdir,
            "calibration_bins_isotonic_test.csv"
        ),
        index=False
    )

    # Optional validation calibrated-bin files for audit
    valid_platt_bins.to_csv(
        os.path.join(
            args.outdir,
            "calibration_bins_platt_validation.csv"
        ),
        index=False
    )

    valid_iso_bins.to_csv(
        os.path.join(
            args.outdir,
            "calibration_bins_isotonic_validation.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Figures
    # --------------------------------------------------------
    plot_original_validation_test(
        valid_original_bins,
        test_original_bins,
        out_png=os.path.join(
            args.outdir,
            "fig_SxA_original_validation_test_reliability.png"
        ),
        out_pdf=os.path.join(
            args.outdir,
            "fig_SxA_original_validation_test_reliability.pdf"
        ),
    )

    plot_test_recalibration(
        test_original_bins,
        test_platt_bins,
        test_iso_bins,
        out_png=os.path.join(
            args.outdir,
            "fig_SxB_test_recalibration_reliability.png"
        ),
        out_pdf=os.path.join(
            args.outdir,
            "fig_SxB_test_recalibration_reliability.pdf"
        ),
        show_isotonic=(not args.hide_isotonic_in_figure),
    )

    # --------------------------------------------------------
    # Save recalibrated test predictions
    # --------------------------------------------------------
    test_out = test_df.copy()
    test_out["prob_original"] = p_test
    test_out["prob_platt"] = p_test_platt
    test_out["prob_isotonic"] = p_test_iso

    test_out.to_csv(
        os.path.join(
            args.outdir,
            "test_predictions_with_recalibration.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------
    config = {
        "validation_predictions": args.valid_predictions,
        "test_predictions": args.test_predictions,
        "outdir": args.outdir,
        "n_bins_ece": N_BINS,
        "ece_binning": "10 equally spaced probability bins",
        "probability_source": (
            "Mean predicted probability across the 10 retained CatBoost "
            "random-seed runs"
        ),
        "platt_fit_dataset": "validation only",
        "isotonic_fit_dataset": "validation only",
        "test_used_for_calibrator_fitting": False,
        "platt_intercept": float(platt_model.intercept_[0]),
        "platt_slope": float(platt_model.coef_[0][0]),
        "show_isotonic_in_figure": (
            not args.hide_isotonic_in_figure
        ),
    }

    with open(
        os.path.join(
            args.outdir,
            "calibration_run_config.json"
        ),
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            config,
            f,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # README
    # --------------------------------------------------------
    readme = f"""
Priority 7 Calibration Analysis
===============================

Validation observations: {len(y_valid)}
Test observations: {len(y_test)}

Primary calibration assessment
------------------------------
The retained CatBoost model is evaluated without recalibration using:
- AUC
- Brier score
- ECE
- Log-loss
- Reliability curves

ECE definition
--------------
ECE uses {N_BINS} equally spaced probability bins over [0,1].
For each bin, the absolute difference between mean predicted probability and
observed event rate is weighted by the proportion of observations in that bin.

Recalibration sensitivity analysis
----------------------------------
Platt scaling and isotonic regression are fitted ONLY on validation predictions
and labels. The fitted calibrators are then applied to the untouched test set.

Supplementary Table Sx
----------------------
supplementary_table_Sx_original_calibration.csv

Supplementary Table Sy
----------------------
supplementary_table_Sy_recalibration_sensitivity.csv

Supplementary Figure SxA
------------------------
fig_SxA_original_validation_test_reliability.png/pdf

Supplementary Figure SxB
------------------------
fig_SxB_test_recalibration_reliability.png/pdf

Recommended manuscript interpretation
-------------------------------------
The original CatBoost should remain the primary model.
Platt/isotonic results should be reported as supplementary post-hoc
recalibration sensitivity analyses.
"""

    with open(
        os.path.join(
            args.outdir,
            "README_priority7_calibration.txt"
        ),
        "w",
        encoding="utf-8"
    ) as f:
        f.write(readme)

    # --------------------------------------------------------
    # Console report
    # --------------------------------------------------------
    print("\n" + "=" * 90)
    print("SUPPLEMENTARY TABLE Sx")
    print("=" * 90)
    print(table_sx.to_string(index=False))

    print("\n" + "=" * 90)
    print("SUPPLEMENTARY TABLE Sy")
    print("=" * 90)
    print(table_sy.to_string(index=False))

    print("\nPlatt calibration parameters:")
    print(f"  intercept = {platt_model.intercept_[0]:.6f}")
    print(f"  slope     = {platt_model.coef_[0][0]:.6f}")

    print("\nDone.")
    print("Outputs saved to:")
    print(args.outdir)


if __name__ == "__main__":
    main()
