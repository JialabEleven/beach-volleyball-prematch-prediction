#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reviewer robustness analysis for beach-volleyball prematch prediction.

Minimal sufficient revision package:
1) Audit that the modeling table is consistent with the previously used dataset.
2) Run expanding-window / rolling-origin validation with the retained identity-free CatBoost.
3) Preserve the original 2020-2022 held-out test block as the final window.
4) Perform COVID sensitivity analysis within the original 2020-2022 held-out test set.
5) Summarize annual schedule-sensitive features, especially days_since_last_match.

Default input and CatBoost settings are aligned with the prior final_catboost_v1 scripts.
"""

import os
import json
import hashlib
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, log_loss


# -----------------------------------------------------------------------------
# Fixed configuration inherited from the previous retained baseline
# -----------------------------------------------------------------------------
SEEDS = [41, 51, 61, 71, 81, 91, 101, 111, 121, 131]

FEATURE_COLUMNS = [
    "gender","circuit","country","bracket","round","month",
    "teamA_avg_age","teamB_avg_age","age_diff",
    "teamA_avg_hgt","teamB_avg_hgt","hgt_diff",
    "teamA_rank","teamB_rank","rank_diff",
    "teamA_is_qualifier","teamB_is_qualifier",
    "teamA_qualifier_num","teamB_qualifier_num","qualifier_num_diff",
    "one_side_qualifier","both_qualifier",
    "teamA_same_country","teamB_same_country",
    "teamA_is_new_pair","teamB_is_new_pair",
    "teamA_prior_matches","teamB_prior_matches","pair_prior_matches_diff",
    "teamA_prior_win_pct","teamB_prior_win_pct","pair_prior_win_pct_diff",
    "teamA_recent3_win_pct","teamB_recent3_win_pct","recent3_win_pct_diff",
    "teamA_recent5_win_pct","teamB_recent5_win_pct","recent5_win_pct_diff",
    "teamA_recent10_win_pct","teamB_recent10_win_pct","recent10_win_pct_diff",
    "teamA_recent20_win_pct","teamB_recent20_win_pct","recent20_win_pct_diff",
    "teamA_last30_match_count","teamB_last30_match_count","last30_match_count_diff",
    "teamA_last30_win_pct","teamB_last30_win_pct","last30_win_pct_diff",
    "teamA_last90_match_count","teamB_last90_match_count","last90_match_count_diff",
    "teamA_last90_win_pct","teamB_last90_win_pct","last90_win_pct_diff",
    "teamA_last180_match_count","teamB_last180_match_count","last180_match_count_diff",
    "teamA_last180_win_pct","teamB_last180_win_pct","last180_win_pct_diff",
    "teamA_days_since_last_match","teamB_days_since_last_match","days_since_last_match_diff",
    "teamA_avg_player_prior_matches","teamB_avg_player_prior_matches","avg_player_prior_matches_diff",
    "teamA_avg_player_prior_win_pct","teamB_avg_player_prior_win_pct","avg_player_prior_win_pct_diff",
    "teamA_avg_player_recent10_win_pct","teamB_avg_player_recent10_win_pct","avg_player_recent10_win_pct_diff",
    "teamA_avg_player_recent20_win_pct","teamB_avg_player_recent20_win_pct","avg_player_recent20_win_pct_diff",
    "teamA_avg_player_last90_match_count","teamB_avg_player_last90_match_count","avg_player_last90_match_count_diff",
    "teamA_avg_player_last90_win_pct","teamB_avg_player_last90_win_pct","avg_player_last90_win_pct_diff",
    "teamA_avg_player_days_since_last_match","teamB_avg_player_days_since_last_match","avg_player_days_since_last_match_diff",
    "teamA_both_players_new","teamB_both_players_new",
    "h2h_prior_matches","h2h_teamA_win_pct","h2h_recent3_teamA_win_pct",
    "teamA_elo","teamB_elo","elo_diff",
    "teamA_player1_elo","teamA_player2_elo","teamB_player1_elo","teamB_player2_elo",
    "teamA_avg_player_elo","teamB_avg_player_elo","avg_player_elo_diff",
    "teamA_blend_elo","teamB_blend_elo","blend_elo_diff"
]

IDENTITY_COLUMNS = [
    "teamA_key", "teamB_key",
    "teamA_player1", "teamA_player2",
    "teamB_player1", "teamB_player2",
    "team1_key", "team2_key",
    "team1_player1", "team1_player2",
    "team2_player1", "team2_player2",
    "teamA_name", "teamB_name", "team1_name", "team2_name",
    "player1", "player2", "winner_name", "loser_name"
]

# Previous manuscript / Table 1 reference values.
# Used only as a consistency check, not as a tuning target.
REFERENCE_TEST = {
    "auc": 0.8126,
    "accuracy": 0.7294,
    "f1": 0.7306,
    "logloss": 0.5253,
}

# Expanding-window rolling-origin design.
# W4 exactly preserves the original train/validation/test chronology.
WINDOWS = [
    {
        "window": "W1",
        "train_end": "2011-12-31",
        "valid_start": "2012-01-01",
        "valid_end": "2013-12-31",
        "test_start": "2014-01-01",
        "test_end": "2015-12-31",
    },
    {
        "window": "W2",
        "train_end": "2013-12-31",
        "valid_start": "2014-01-01",
        "valid_end": "2015-12-31",
        "test_start": "2016-01-01",
        "test_end": "2017-12-31",
    },
    {
        "window": "W3",
        "train_end": "2015-12-31",
        "valid_start": "2016-01-01",
        "valid_end": "2017-12-31",
        "test_start": "2018-01-01",
        "test_end": "2019-12-31",
    },
    {
        "window": "W4_original_final",
        "train_end": "2017-12-31",
        "valid_start": "2018-01-01",
        "valid_end": "2019-12-31",
        "test_start": "2020-01-01",
        "test_end": "2022-12-31",
    },
]


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def read_csv_fallback(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", low_memory=False)


def sha256_file(path: str, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def canonical_dataframe_fingerprint(df: pd.DataFrame) -> str:
    """
    Stable fingerprint of the modeling-relevant content.
    It is not sensitive to the source CSV's line endings / quoting style.
    """
    cols = ["date", "y"] + FEATURE_COLUMNS
    identity_present = [c for c in IDENTITY_COLUMNS if c in df.columns]
    cols += identity_present

    x = df[cols].copy()
    x["date"] = pd.to_datetime(x["date"], errors="raise").dt.strftime("%Y-%m-%d")

    # Keep the original row order after chronological stable sorting.
    sort_cols = ["date"]
    if "match_num" in df.columns:
        x["__match_num__"] = df["match_num"].values
        sort_cols.append("__match_num__")

    x = x.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    # pandas row hashes -> SHA256
    row_hash = pd.util.hash_pandas_object(x, index=False).to_numpy(dtype=np.uint64)
    return hashlib.sha256(row_hash.tobytes()).hexdigest()


def safe_date_str(x) -> str:
    if pd.isna(x):
        return "NA"
    return str(pd.Timestamp(x).date())


def chronological_sort(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise ValueError(f"Invalid/missing dates: {int(df['date'].isna().sum())}")

    sort_cols = ["date"]
    if "year" in df.columns:
        sort_cols.append("year")
    if "match_num" in df.columns:
        sort_cols.append("match_num")

    df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    return df, sort_cols


def split_window(df: pd.DataFrame, w: Dict[str, str]):
    train_end = pd.Timestamp(w["train_end"])
    valid_start = pd.Timestamp(w["valid_start"])
    valid_end = pd.Timestamp(w["valid_end"])
    test_start = pd.Timestamp(w["test_start"])
    test_end = pd.Timestamp(w["test_end"])

    train = df[df["date"] <= train_end].copy().reset_index(drop=True)
    valid = df[(df["date"] >= valid_start) & (df["date"] <= valid_end)].copy().reset_index(drop=True)
    test = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy().reset_index(drop=True)

    return train, valid, test


def fit_preprocessor(train: pd.DataFrame, features: List[str]):
    numeric, categorical = [], []
    medians, modes = {}, {}

    for c in features:
        if pd.api.types.is_numeric_dtype(train[c]):
            numeric.append(c)
            med = train[c].median()
            medians[c] = 0.0 if pd.isna(med) else float(med)
        else:
            categorical.append(c)
            m = train[c].mode(dropna=True)
            modes[c] = str(m.iloc[0]) if len(m) else "MISSING"

    return numeric, categorical, medians, modes


def transform(df: pd.DataFrame, features: List[str], numeric, categorical, medians, modes):
    X = df[features].copy()

    for c in numeric:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(medians[c])

    for c in categorical:
        X[c] = X[c].fillna(modes[c]).astype(str)

    return X


def evaluate(y_true, proba) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    proba = np.clip(np.asarray(proba, dtype=float), 1e-12, 1 - 1e-12)
    pred = (proba >= 0.5).astype(int)

    if len(np.unique(y_true)) < 2:
        auc_value = np.nan
    else:
        auc_value = float(roc_auc_score(y_true, proba))

    return {
        "auc": auc_value,
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "logloss": float(log_loss(y_true, proba, labels=[0, 1])),
    }


def train_one(train, valid, test, seed: int, args):
    features = FEATURE_COLUMNS
    numeric, categorical, medians, modes = fit_preprocessor(train, features)

    Xtr = transform(train, features, numeric, categorical, medians, modes)
    Xva = transform(valid, features, numeric, categorical, medians, modes)
    Xte = transform(test, features, numeric, categorical, medians, modes)

    ytr = train["y"].astype(int).to_numpy()
    yva = valid["y"].astype(int).to_numpy()
    yte = test["y"].astype(int).to_numpy()

    cat_idx = [Xtr.columns.get_loc(c) for c in categorical]

    model = CatBoostClassifier(
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        l2_leaf_reg=args.l2_leaf_reg,
        subsample=args.subsample,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=seed,
        has_time=True,
        allow_writing_files=False,
        verbose=False,
    )

    model.fit(
        Xtr,
        ytr,
        cat_features=cat_idx,
        eval_set=(Xva, yva),
        use_best_model=True,
        verbose=False,
    )

    p_valid = model.predict_proba(Xva)[:, 1]
    p_test = model.predict_proba(Xte)[:, 1]

    return {
        "model": model,
        "best_iteration": int(model.get_best_iteration()),
        "y_valid": yva,
        "p_valid": p_valid,
        "y_test": yte,
        "p_test": p_test,
    }


def format_mean_sd(mean_val, sd_val) -> str:
    return f"{mean_val:.4f} ± {sd_val:.4f}"


# -----------------------------------------------------------------------------
# 1) Dataset integrity audit
# -----------------------------------------------------------------------------
def audit_dataset(df: pd.DataFrame, input_path: str, outdir: str, reference_manifest: str = None):
    required = ["date", "y"] + FEATURE_COLUMNS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns:\n" + "\n".join(missing))

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="raise").astype(int)

    if df["date"].isna().any():
        raise ValueError("Dataset contains invalid dates.")

    target_values = sorted(df["y"].dropna().unique().tolist())
    if not set(target_values).issubset({0, 1}):
        raise ValueError(f"Target y is not binary: {target_values}")

    byte_sha = sha256_file(input_path)
    canonical_sha = canonical_dataframe_fingerprint(df)

    identity_present = [c for c in IDENTITY_COLUMNS if c in df.columns]

    checks = {
        "input_path": os.path.abspath(input_path),
        "file_size_bytes": int(os.path.getsize(input_path)),
        "sha256_file_bytes": byte_sha,
        "sha256_canonical_modeling_content": canonical_sha,
        "n_rows": int(len(df)),
        "n_columns_total": int(df.shape[1]),
        "n_required_identity_free_features": int(len(FEATURE_COLUMNS)),
        "date_min": safe_date_str(df["date"].min()),
        "date_max": safe_date_str(df["date"].max()),
        "year_min": int(df["date"].dt.year.min()),
        "year_max": int(df["date"].dt.year.max()),
        "target_values": target_values,
        "target_prevalence": float(df["y"].mean()),
        "identity_columns_present_in_raw_table": identity_present,
        "identity_columns_used_by_model": [],
        "missing_required_columns": missing,
        "duplicate_full_rows": int(df.duplicated().sum()),
    }

    # Known invariants from the previous manuscript/data pipeline.
    checks["check_n_rows_85509"] = bool(len(df) == 85509)
    checks["check_year_min_2000"] = bool(df["date"].dt.year.min() == 2000)
    checks["check_year_max_2022"] = bool(df["date"].dt.year.max() == 2022)
    checks["check_feature_count_104"] = bool(len(FEATURE_COLUMNS) == 104)
    checks["check_target_binary"] = bool(set(target_values).issubset({0, 1}))
    checks["check_required_columns_complete"] = bool(len(missing) == 0)

    structural_pass = all([
        checks["check_n_rows_85509"],
        checks["check_year_min_2000"],
        checks["check_year_max_2022"],
        checks["check_feature_count_104"],
        checks["check_target_binary"],
        checks["check_required_columns_complete"],
    ])
    checks["structural_consistency_pass"] = bool(structural_pass)

    # Optional exact comparison if a previously saved manifest exists.
    exact_reference_status = "NO_REFERENCE_MANIFEST"
    if reference_manifest and os.path.exists(reference_manifest):
        with open(reference_manifest, "r", encoding="utf-8") as f:
            ref = json.load(f)
        ref_sha = ref.get("sha256_canonical_modeling_content")
        if ref_sha:
            exact_reference_status = "MATCH" if ref_sha == canonical_sha else "MISMATCH"
    checks["exact_reference_fingerprint_status"] = exact_reference_status

    with open(os.path.join(outdir, "dataset_integrity_audit.json"), "w", encoding="utf-8") as f:
        json.dump(checks, f, indent=2, ensure_ascii=False)

    # Save a compact manifest that can be used for exact future comparison.
    manifest = {
        "input_path": os.path.abspath(input_path),
        "n_rows": int(len(df)),
        "n_columns_total": int(df.shape[1]),
        "date_min": safe_date_str(df["date"].min()),
        "date_max": safe_date_str(df["date"].max()),
        "sha256_file_bytes": byte_sha,
        "sha256_canonical_modeling_content": canonical_sha,
    }
    with open(os.path.join(outdir, "current_dataset_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Column-level schema record.
    schema = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "missing_n": [int(df[c].isna().sum()) for c in df.columns],
        "missing_pct": [float(df[c].isna().mean()) for c in df.columns],
        "is_model_feature": [c in FEATURE_COLUMNS for c in df.columns],
        "is_raw_identity_column": [c in IDENTITY_COLUMNS for c in df.columns],
    })
    schema.to_csv(os.path.join(outdir, "dataset_schema_audit.csv"), index=False)

    return checks


# -----------------------------------------------------------------------------
# 2) Rolling-origin validation
# -----------------------------------------------------------------------------
def run_rolling_origin(df: pd.DataFrame, outdir: str, args):
    all_rows = []
    window_prediction_store = {}

    for w in WINDOWS:
        train, valid, test = split_window(df, w)

        if min(len(train), len(valid), len(test)) == 0:
            raise RuntimeError(f"Empty split in {w['window']}: "
                               f"train={len(train)}, valid={len(valid)}, test={len(test)}")
        if train["date"].max() >= valid["date"].min():
            raise RuntimeError(f"Temporal overlap train/validation in {w['window']}")
        if valid["date"].max() >= test["date"].min():
            raise RuntimeError(f"Temporal overlap validation/test in {w['window']}")

        print("\n" + "=" * 90)
        print(f"{w['window']}")
        print(f"Train: <= {w['train_end']} | n={len(train):,}")
        print(f"Valid: {w['valid_start']} to {w['valid_end']} | n={len(valid):,}")
        print(f"Test : {w['test_start']} to {w['test_end']} | n={len(test):,}")
        print("=" * 90)

        test_probs = []
        valid_probs = []

        for seed in args.seeds:
            result = train_one(train, valid, test, seed, args)
            mv = evaluate(result["y_valid"], result["p_valid"])
            mt = evaluate(result["y_test"], result["p_test"])

            row = {
                "window": w["window"],
                "seed": seed,
                "train_end": w["train_end"],
                "valid_start": w["valid_start"],
                "valid_end": w["valid_end"],
                "test_start": w["test_start"],
                "test_end": w["test_end"],
                "n_train": len(train),
                "n_valid": len(valid),
                "n_test": len(test),
                "test_prevalence": float(test["y"].mean()),
                "best_iteration": result["best_iteration"],
                **{f"valid_{k}": v for k, v in mv.items()},
                **{f"test_{k}": v for k, v in mt.items()},
            }
            all_rows.append(row)
            valid_probs.append(result["p_valid"])
            test_probs.append(result["p_test"])

            print(
                f"seed={seed:3d} | test AUC={mt['auc']:.4f} | "
                f"Acc={mt['accuracy']:.4f} | F1={mt['f1']:.4f} | "
                f"LogLoss={mt['logloss']:.4f}"
            )

        mean_test_prob = np.mean(np.vstack(test_probs), axis=0)
        mean_valid_prob = np.mean(np.vstack(valid_probs), axis=0)

        pred_df = pd.DataFrame({
            "date": test["date"].dt.strftime("%Y-%m-%d"),
            "y": test["y"].to_numpy(),
            "probability_mean_seeds": mean_test_prob,
        })
        pred_path = os.path.join(outdir, f"predictions_{w['window']}.csv")
        pred_df.to_csv(pred_path, index=False)

        window_prediction_store[w["window"]] = {
            "test_df": test.copy(),
            "mean_test_prob": mean_test_prob,
            "mean_valid_prob": mean_valid_prob,
        }

    runs = pd.DataFrame(all_rows)
    runs.to_csv(os.path.join(outdir, "rolling_origin_metrics_all_seeds.csv"), index=False)

    # Mean ± SD across the same 10 seeds used in the manuscript.
    summary_rows = []
    for w in WINDOWS:
        g = runs[runs["window"] == w["window"]].copy()
        row = {
            "window": w["window"],
            "train_period": f"2000-{pd.Timestamp(w['train_end']).year}",
            "validation_period": f"{pd.Timestamp(w['valid_start']).year}-{pd.Timestamp(w['valid_end']).year}",
            "test_period": f"{pd.Timestamp(w['test_start']).year}-{pd.Timestamp(w['test_end']).year}",
            "n_train": int(g["n_train"].iloc[0]),
            "n_valid": int(g["n_valid"].iloc[0]),
            "n_test": int(g["n_test"].iloc[0]),
            "test_prevalence": float(g["test_prevalence"].iloc[0]),
            "n_seeds": int(len(g)),
            "best_iteration_mean": float(g["best_iteration"].mean()),
            "best_iteration_sd": float(g["best_iteration"].std(ddof=1)),
        }
        for metric in ["auc", "accuracy", "f1", "logloss"]:
            row[f"test_{metric}_mean"] = float(g[f"test_{metric}"].mean())
            row[f"test_{metric}_sd"] = float(g[f"test_{metric}"].std(ddof=1))
            row[f"test_{metric}_mean_sd"] = format_mean_sd(
                row[f"test_{metric}_mean"], row[f"test_{metric}_sd"]
            )
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(outdir, "rolling_origin_metrics_summary.csv"), index=False)

    supplementary = summary[[
        "window", "train_period", "validation_period", "test_period",
        "n_test", "test_auc_mean_sd", "test_accuracy_mean_sd",
        "test_f1_mean_sd", "test_logloss_mean_sd"
    ]].rename(columns={
        "window": "Window",
        "train_period": "Training period",
        "validation_period": "Validation period",
        "test_period": "Test period",
        "n_test": "Test N",
        "test_auc_mean_sd": "AUC",
        "test_accuracy_mean_sd": "Accuracy",
        "test_f1_mean_sd": "F1-score",
        "test_logloss_mean_sd": "Log-loss",
    })
    supplementary.to_csv(os.path.join(outdir, "supplementary_table_rolling_origin.csv"), index=False)

    # Figure: mean test AUC ± SD across temporal windows.
    x = np.arange(len(summary))
    plt.figure(figsize=(7.5, 5.0))
    plt.errorbar(
        x,
        summary["test_auc_mean"].to_numpy(),
        yerr=summary["test_auc_sd"].to_numpy(),
        marker="o",
        capsize=4,
    )
    plt.xticks(x, summary["test_period"].tolist())
    plt.xlabel("Held-out test period")
    plt.ylabel("AUC")
    plt.title("Rolling-origin temporal validation")
    plt.ylim(0.5, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig_rolling_origin_auc.png"), dpi=300)
    plt.savefig(os.path.join(outdir, "fig_rolling_origin_auc.pdf"))
    plt.close()

    return runs, summary, window_prediction_store


# -----------------------------------------------------------------------------
# 3) Reproduction consistency check for original final split
# -----------------------------------------------------------------------------
def reproduction_check(summary: pd.DataFrame, outdir: str, tolerance: float = 0.01):
    row = summary[summary["window"] == "W4_original_final"]
    if len(row) != 1:
        raise RuntimeError("W4_original_final summary not found.")
    r = row.iloc[0]

    rows = []
    all_within = True
    for metric, ref in REFERENCE_TEST.items():
        observed = float(r[f"test_{metric}_mean"])
        diff = observed - ref
        within = abs(diff) <= tolerance
        all_within = all_within and within
        rows.append({
            "metric": metric,
            "previous_reference": ref,
            "recomputed_W4_mean": observed,
            "difference": diff,
            "absolute_difference": abs(diff),
            "tolerance": tolerance,
            "within_tolerance": within,
        })

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(outdir, "previous_result_reproduction_check.csv"), index=False)

    status = {
        "reference_source": "previous manuscript Table 1",
        "comparison_window": "W4_original_final: train<=2017, valid=2018-2019, test=2020-2022",
        "tolerance_absolute": tolerance,
        "all_metrics_within_tolerance": bool(all_within),
        "interpretation": (
            "Strong consistency with the previously analyzed dataset/model pipeline"
            if all_within else
            "One or more metrics differ beyond tolerance; investigate dataset or pipeline changes"
        ),
    }
    with open(os.path.join(outdir, "previous_result_reproduction_status.json"), "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2, ensure_ascii=False)

    return out, status


# -----------------------------------------------------------------------------
# 4) COVID sensitivity analysis using ONLY W4 held-out predictions
# -----------------------------------------------------------------------------
def covid_sensitivity(window_prediction_store: Dict, outdir: str):
    key = "W4_original_final"
    if key not in window_prediction_store:
        raise RuntimeError("Original final window predictions unavailable.")

    test = window_prediction_store[key]["test_df"].copy().reset_index(drop=True)
    p = np.asarray(window_prediction_store[key]["mean_test_prob"], dtype=float)
    test["probability_mean_seeds"] = p

    periods = [
        ("2020 pandemic-disruption year", "2020-01-01", "2020-12-31"),
        ("2021-2022 later period", "2021-01-01", "2022-12-31"),
        ("2020-2022 original final test", "2020-01-01", "2022-12-31"),
    ]

    rows = []
    for label, start, end in periods:
        mask = (test["date"] >= pd.Timestamp(start)) & (test["date"] <= pd.Timestamp(end))
        sub = test.loc[mask].copy()
        if len(sub) == 0:
            continue
        m = evaluate(sub["y"].to_numpy(), sub["probability_mean_seeds"].to_numpy())
        rows.append({
            "period": label,
            "start": start,
            "end": end,
            "n_matches": len(sub),
            "teamA_win_prevalence": float(sub["y"].mean()),
            **m,
        })

    result = pd.DataFrame(rows)
    result.to_csv(os.path.join(outdir, "covid_sensitivity_metrics.csv"), index=False)

    test[["date", "y", "probability_mean_seeds"]].to_csv(
        os.path.join(outdir, "W4_final_test_predictions_mean_seeds.csv"), index=False
    )
    return result


# -----------------------------------------------------------------------------
# 5) Annual schedule-sensitive feature summaries
# -----------------------------------------------------------------------------
def pooled_year_summary(df: pd.DataFrame, col_a: str, col_b: str, feature_name: str):
    records = []
    years = sorted(df["date"].dt.year.dropna().unique())
    for year in years:
        d = df[df["date"].dt.year == year]
        values = pd.concat([
            pd.to_numeric(d[col_a], errors="coerce"),
            pd.to_numeric(d[col_b], errors="coerce"),
        ], ignore_index=True).dropna()
        if len(values) == 0:
            continue
        records.append({
            "year": int(year),
            "feature": feature_name,
            "n_side_observations": int(len(values)),
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)),
            "q25": float(values.quantile(0.25)),
            "median": float(values.median()),
            "q75": float(values.quantile(0.75)),
        })
    return records


def temporal_feature_summary(df: pd.DataFrame, outdir: str):
    specs = [
        (
            "teamA_days_since_last_match", "teamB_days_since_last_match",
            "team_days_since_last_match"
        ),
        (
            "teamA_avg_player_days_since_last_match", "teamB_avg_player_days_since_last_match",
            "avg_player_days_since_last_match"
        ),
        (
            "teamA_last90_match_count", "teamB_last90_match_count",
            "team_last90_match_count"
        ),
        (
            "teamA_recent10_win_pct", "teamB_recent10_win_pct",
            "team_recent10_win_pct"
        ),
    ]

    rows = []
    for col_a, col_b, name in specs:
        rows.extend(pooled_year_summary(df, col_a, col_b, name))

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(outdir, "annual_temporal_feature_summary.csv"), index=False)

    # Focused figure requested by reviewer concern: days_since_last_match.
    plot_df = summary[summary["feature"] == "team_days_since_last_match"].copy()
    if len(plot_df):
        x = plot_df["year"].to_numpy()
        y = plot_df["median"].to_numpy()
        lower = y - plot_df["q25"].to_numpy()
        upper = plot_df["q75"].to_numpy() - y

        plt.figure(figsize=(9.0, 5.0))
        plt.errorbar(x, y, yerr=np.vstack([lower, upper]), marker="o", capsize=3)
        plt.axvline(2020, linestyle="--", linewidth=1)
        plt.xlabel("Year")
        plt.ylabel("Days since last match, median (IQR)")
        plt.title("Annual distribution of days since last match")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "fig_days_since_last_match_by_year.png"), dpi=300)
        plt.savefig(os.path.join(outdir, "fig_days_since_last_match_by_year.pdf"))
        plt.close()

    return summary


# -----------------------------------------------------------------------------
# Optional raw archive audit (informational only)
# -----------------------------------------------------------------------------
def audit_raw_archive(raw_path: str, outdir: str):
    if not raw_path or not os.path.exists(raw_path):
        return None

    raw = read_csv_fallback(raw_path)
    info = {
        "raw_input_path": os.path.abspath(raw_path),
        "raw_file_size_bytes": int(os.path.getsize(raw_path)),
        "raw_sha256_file_bytes": sha256_file(raw_path),
        "raw_n_rows": int(len(raw)),
        "raw_n_columns": int(raw.shape[1]),
        "raw_columns": raw.columns.tolist(),
    }
    with open(os.path.join(outdir, "raw_archive_audit.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    return info


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Minimal rolling-origin + COVID sensitivity analysis for reviewer response"
    )
    parser.add_argument(
        "--input",
        default="./legacy_project/final_catboost_v1/data/prematch_features_v3.csv",
        help="Processed prematch modeling table",
    )
    parser.add_argument(
        "--raw-input",
        default="./legacy_project/beach-volleyball-master/data/full_archive/full_archive.csv",
        help="Original public archive; audited only if it exists",
    )
    parser.add_argument(
        "--outdir",
        default="./legacy_project/final_catboost_v1/rolling_origin_covid_output",
    )
    parser.add_argument(
        "--reference-manifest",
        default=None,
        help="Optional old manifest JSON for exact canonical fingerprint comparison",
    )
    parser.add_argument("--iterations", type=int, default=900)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--l2-leaf-reg", type=float, default=5.0)
    parser.add_argument("--subsample", type=float, default=0.85)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument(
        "--reproduction-tolerance",
        type=float,
        default=0.01,
        help="Absolute tolerance for comparison with previous Table 1 metrics",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 90)
    print("BEACH VOLLEYBALL: ROLLING-ORIGIN + COVID SENSITIVITY")
    print("=" * 90)
    print("Input :", args.input)
    print("Output:", args.outdir)
    print("Seeds :", args.seeds)
    print("Features:", len(FEATURE_COLUMNS), "identity-free prematch predictors")

    if len(FEATURE_COLUMNS) != 104:
        raise RuntimeError(f"Unexpected feature count: {len(FEATURE_COLUMNS)} (expected 104)")

    df = read_csv_fallback(args.input)

    # Audit before sorting/mutating.
    audit = audit_dataset(
        df,
        args.input,
        args.outdir,
        reference_manifest=args.reference_manifest,
    )

    print("\nDATASET AUDIT")
    print("Rows:", f"{audit['n_rows']:,}")
    print("Years:", audit["year_min"], "to", audit["year_max"])
    print("Structural consistency:", "PASS" if audit["structural_consistency_pass"] else "FAIL")
    print("Canonical fingerprint:", audit["sha256_canonical_modeling_content"])
    print("Old-manifest exact comparison:", audit["exact_reference_fingerprint_status"])

    if not audit["structural_consistency_pass"]:
        raise RuntimeError(
            "Dataset structural audit failed. Stop before model fitting to avoid analyzing a changed table."
        )

    audit_raw_archive(args.raw_input, args.outdir)

    df["y"] = pd.to_numeric(df["y"], errors="raise").astype(int)
    df, sort_cols = chronological_sort(df)

    runs, summary, prediction_store = run_rolling_origin(df, args.outdir, args)

    reproduction_df, reproduction_status = reproduction_check(
        summary,
        args.outdir,
        tolerance=args.reproduction_tolerance,
    )

    covid_df = covid_sensitivity(prediction_store, args.outdir)
    annual_df = temporal_feature_summary(df, args.outdir)

    config = {
        "input": args.input,
        "raw_input": args.raw_input,
        "outdir": args.outdir,
        "feature_count": len(FEATURE_COLUMNS),
        "identity_free": True,
        "seeds": args.seeds,
        "sort_columns": sort_cols,
        "windows": WINDOWS,
        "catboost": {
            "iterations": args.iterations,
            "depth": args.depth,
            "learning_rate": args.learning_rate,
            "l2_leaf_reg": args.l2_leaf_reg,
            "subsample": args.subsample,
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "has_time": True,
            "use_best_model": True,
        },
        "reproduction_reference": REFERENCE_TEST,
        "reproduction_tolerance": args.reproduction_tolerance,
    }
    with open(os.path.join(args.outdir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    readme = f"""Reviewer robustness analysis completed.

DATA CONSISTENCY
- Structural audit: {'PASS' if audit['structural_consistency_pass'] else 'FAIL'}
- Current rows: {audit['n_rows']}
- Current years: {audit['year_min']}-{audit['year_max']}
- Canonical fingerprint: {audit['sha256_canonical_modeling_content']}
- Exact old-manifest comparison: {audit['exact_reference_fingerprint_status']}
- Previous-result reproduction: {reproduction_status['interpretation']}

ROLLING-ORIGIN DESIGN
W1: train <=2011, validation 2012-2013, test 2014-2015
W2: train <=2013, validation 2014-2015, test 2016-2017
W3: train <=2015, validation 2016-2017, test 2018-2019
W4: train <=2017, validation 2018-2019, test 2020-2022 (original final split)

COVID SENSITIVITY
- Uses mean predictions from W4 only.
- Reports 2020 separately from 2021-2022.
- Does not retrain on pandemic-period outcomes.
- Annual temporal-feature summaries include days_since_last_match, last90_match_count, and recent10_win_pct.

PRIMARY FILES FOR REVISION
- supplementary_table_rolling_origin.csv
- fig_rolling_origin_auc.png / .pdf
- covid_sensitivity_metrics.csv
- annual_temporal_feature_summary.csv
- fig_days_since_last_match_by_year.png / .pdf
- dataset_integrity_audit.json
- previous_result_reproduction_check.csv
"""
    with open(os.path.join(args.outdir, "README_revision_analysis.txt"), "w", encoding="utf-8") as f:
        f.write(readme)

    print("\n" + "=" * 90)
    print("ROLLING-ORIGIN SUMMARY")
    print("=" * 90)
    cols = [
        "window", "train_period", "validation_period", "test_period", "n_test",
        "test_auc_mean_sd", "test_accuracy_mean_sd", "test_f1_mean_sd", "test_logloss_mean_sd"
    ]
    print(summary[cols].to_string(index=False))

    print("\n" + "=" * 90)
    print("PREVIOUS-RESULT REPRODUCTION CHECK")
    print("=" * 90)
    print(reproduction_df.to_string(index=False))
    print("Status:", reproduction_status["interpretation"])

    print("\n" + "=" * 90)
    print("COVID SENSITIVITY")
    print("=" * 90)
    print(covid_df.to_string(index=False))

    print("\nSaved all outputs to:", args.outdir)


if __name__ == "__main__":
    main()
