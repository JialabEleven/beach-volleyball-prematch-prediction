#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Priority 6: Global-model subgroup evaluation
============================================

Purpose
-------
Address Reviewer 2's request to clarify whether subgroup performance comes from:
    (a) one global model evaluated within prespecified subgroups, or
    (b) separate subgroup-specific models.

This script implements option (a):

    Full training set (<=2017)
        -> train ONE global CatBoost model per random seed
        -> predict ALL validation/test observations
        -> stratify predictions AFTER prediction by:
             Men / Women / AVP / FIVB
        -> calculate subgroup metrics

No subgroup-specific model retraining is performed.

The model specification, feature set, chronological split, missing-data handling,
decision threshold, and 10 random seeds are aligned with the retained baseline
CatBoost analysis used in the manuscript.

Primary outputs
---------------
priority6_global_subgroup_metrics_all_runs.csv
priority6_global_subgroup_metrics_summary_mean_sd.csv
priority6_global_subgroup_metrics_formatted.csv
priority6_global_subgroup_sample_sizes.csv
priority6_global_overall_metrics_all_runs.csv
priority6_global_overall_metrics_summary.csv
priority6_global_test_predictions_mean_10seeds.csv
priority6_global_valid_predictions_mean_10seeds.csv
priority6_old_vs_global_subgroup_auc.csv
fig_priority6_subgroup_test_auc.png/pdf
roc_<subgroup>_valid.png/pdf
roc_<subgroup>_test.png/pdf
priority6_run_config.json
README_priority6_global_subgroup.txt

Important interpretation
------------------------
The SD reported here reflects variability across the 10 model-training random seeds.
No bootstrap 95% CI is added in this Priority 6 analysis, to keep the subgroup
reporting consistent with the manuscript's existing Table 3 format.

This script intentionally does NOT perform significance tests between subgroups.
Therefore, manuscript wording should use terms such as "numerically higher" or
"somewhat higher", rather than "significantly higher", unless a separate formal
between-subgroup test is later conducted.
"""

import os
import json
import argparse
from typing import Tuple, Dict

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from catboost import CatBoostClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    log_loss,
    roc_curve,
    auc,
)


FINAL_SEEDS = [41, 51, 61, 71, 81, 91, 101, 111, 121, 131]

CATBOOST_PARAMS = {
    "iterations": 900,
    "depth": 6,
    "learning_rate": 0.03,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "l2_leaf_reg": 5,
    "subsample": 0.85,
    "allow_writing_files": False,
    "verbose": False,
}

DECISION_THRESHOLD = 0.5

FEATURE_COLUMNS = [
    "gender", "circuit", "country", "bracket", "round", "month",
    "teamA_avg_age", "teamB_avg_age", "age_diff",
    "teamA_avg_hgt", "teamB_avg_hgt", "hgt_diff",
    "teamA_rank", "teamB_rank", "rank_diff",
    "teamA_is_qualifier", "teamB_is_qualifier",
    "teamA_qualifier_num", "teamB_qualifier_num", "qualifier_num_diff",
    "one_side_qualifier", "both_qualifier",
    "teamA_same_country", "teamB_same_country",
    "teamA_is_new_pair", "teamB_is_new_pair",
    "teamA_prior_matches", "teamB_prior_matches", "pair_prior_matches_diff",
    "teamA_prior_win_pct", "teamB_prior_win_pct", "pair_prior_win_pct_diff",
    "teamA_recent3_win_pct", "teamB_recent3_win_pct", "recent3_win_pct_diff",
    "teamA_recent5_win_pct", "teamB_recent5_win_pct", "recent5_win_pct_diff",
    "teamA_recent10_win_pct", "teamB_recent10_win_pct", "recent10_win_pct_diff",
    "teamA_recent20_win_pct", "teamB_recent20_win_pct", "recent20_win_pct_diff",
    "teamA_last30_match_count", "teamB_last30_match_count", "last30_match_count_diff",
    "teamA_last30_win_pct", "teamB_last30_win_pct", "last30_win_pct_diff",
    "teamA_last90_match_count", "teamB_last90_match_count", "last90_match_count_diff",
    "teamA_last90_win_pct", "teamB_last90_win_pct", "last90_win_pct_diff",
    "teamA_last180_match_count", "teamB_last180_match_count", "last180_match_count_diff",
    "teamA_last180_win_pct", "teamB_last180_win_pct", "last180_win_pct_diff",
    "teamA_days_since_last_match", "teamB_days_since_last_match", "days_since_last_match_diff",
    "teamA_avg_player_prior_matches", "teamB_avg_player_prior_matches",
    "avg_player_prior_matches_diff",
    "teamA_avg_player_prior_win_pct", "teamB_avg_player_prior_win_pct",
    "avg_player_prior_win_pct_diff",
    "teamA_avg_player_recent10_win_pct", "teamB_avg_player_recent10_win_pct",
    "avg_player_recent10_win_pct_diff",
    "teamA_avg_player_recent20_win_pct", "teamB_avg_player_recent20_win_pct",
    "avg_player_recent20_win_pct_diff",
    "teamA_avg_player_last90_match_count", "teamB_avg_player_last90_match_count",
    "avg_player_last90_match_count_diff",
    "teamA_avg_player_last90_win_pct", "teamB_avg_player_last90_win_pct",
    "avg_player_last90_win_pct_diff",
    "teamA_avg_player_days_since_last_match", "teamB_avg_player_days_since_last_match",
    "avg_player_days_since_last_match_diff",
    "teamA_both_players_new", "teamB_both_players_new",
    "h2h_prior_matches", "h2h_teamA_win_pct", "h2h_recent3_teamA_win_pct",
    "teamA_elo", "teamB_elo", "elo_diff",
    "teamA_player1_elo", "teamA_player2_elo",
    "teamB_player1_elo", "teamB_player2_elo",
    "teamA_avg_player_elo", "teamB_avg_player_elo", "avg_player_elo_diff",
    "teamA_blend_elo", "teamB_blend_elo", "blend_elo_diff",
]

OLD_SUBGROUP_SPECIFIC_TEST_AUC = {
    "Men": 0.7989,
    "Women": 0.8211,
    "AVP": 0.8180,
    "FIVB": 0.8027,
}


def load_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", low_memory=False)


def detect_target_column(df: pd.DataFrame) -> str:
    candidates = ["y", "target", "label", "outcome", "result", "teamA_win", "team1_win", "win"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not detect target column. Tried: {candidates}")


def prepare_date_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        if out["date"].isna().any():
            raise ValueError(f"Invalid/missing dates: {int(out['date'].isna().sum())}")

        sort_cols = ["date"]
        if "match_num" in out.columns:
            sort_cols.append("match_num")
        out = out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    elif "year" in out.columns:
        out["year"] = pd.to_numeric(out["year"], errors="raise")
        out = out.sort_values(["year"], kind="mergesort").reset_index(drop=True)
    else:
        raise ValueError("Need either a 'date' or 'year' column.")

    return out


def split_by_time(
    df: pd.DataFrame,
    train_end: str = "2017-12-31",
    valid_end: str = "2019-12-31",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    if "date" in df.columns:
        train_end_ts = pd.Timestamp(train_end)
        valid_end_ts = pd.Timestamp(valid_end)

        train_df = df[df["date"] <= train_end_ts].copy()
        valid_df = df[(df["date"] > train_end_ts) & (df["date"] <= valid_end_ts)].copy()
        test_df = df[df["date"] > valid_end_ts].copy()
    else:
        train_df = df[df["year"] <= 2017].copy()
        valid_df = df[(df["year"] >= 2018) & (df["year"] <= 2019)].copy()
        test_df = df[df["year"] >= 2020].copy()

    if min(len(train_df), len(valid_df), len(test_df)) == 0:
        raise ValueError(
            f"Empty split detected: train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}"
        )

    return (
        train_df.reset_index(drop=True),
        valid_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def validate_required_columns(df: pd.DataFrame):
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Missing required feature columns:\n" + "\n".join(missing))


def make_xy(df: pd.DataFrame, target_col: str):
    X = df[FEATURE_COLUMNS].copy()
    y = pd.to_numeric(df[target_col], errors="raise").astype(int).to_numpy()

    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError(f"Target must be binary 0/1. Found: {np.unique(y)}")

    return X, y


def fill_missing(train_X, valid_X, test_X):
    train_X = train_X.copy()
    valid_X = valid_X.copy()
    test_X = test_X.copy()

    num_cols, cat_cols = [], []
    for c in train_X.columns:
        if pd.api.types.is_numeric_dtype(train_X[c]):
            num_cols.append(c)
        else:
            cat_cols.append(c)

    for c in num_cols:
        med = train_X[c].median()
        train_X[c] = train_X[c].fillna(med)
        valid_X[c] = valid_X[c].fillna(med)
        test_X[c] = test_X[c].fillna(med)

    for c in cat_cols:
        mode_series = train_X[c].mode(dropna=True)
        mode_val = mode_series.iloc[0] if not mode_series.empty else "missing"

        train_X[c] = train_X[c].fillna(mode_val).astype(str)
        valid_X[c] = valid_X[c].fillna(mode_val).astype(str)
        test_X[c] = test_X[c].fillna(mode_val).astype(str)

    cat_feature_indices = [train_X.columns.get_loc(c) for c in cat_cols]
    return train_X, valid_X, test_X, cat_feature_indices


def normalize_gender(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.upper()
    mapping = {
        "M": "M", "MEN": "M", "MALE": "M",
        "W": "W", "WOMEN": "W", "WOMAN": "W", "F": "W", "FEMALE": "W",
    }
    return s.map(mapping).fillna(s)


def normalize_circuit(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def subgroup_masks(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    gender = normalize_gender(df["gender"])
    circuit = normalize_circuit(df["circuit"])
    return {
        "Men": (gender == "M").to_numpy(),
        "Women": (gender == "W").to_numpy(),
        "AVP": (circuit == "AVP").to_numpy(),
        "FIVB": (circuit == "FIVB").to_numpy(),
    }


def evaluate_binary(y_true, proba, threshold=DECISION_THRESHOLD):
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)

    if len(y_true) == 0:
        raise ValueError("Cannot evaluate an empty subgroup.")
    if len(np.unique(y_true)) < 2:
        raise ValueError("AUC cannot be computed because subgroup contains only one class.")

    proba = np.clip(proba, 1e-12, 1 - 1e-12)
    pred = (proba >= threshold).astype(int)

    return {
        "auc": float(roc_auc_score(y_true, proba)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred)),
        "logloss": float(log_loss(y_true, proba)),
        "prevalence": float(np.mean(y_true)),
    }


def format_mean_sd(mean_val, sd_val, digits=4):
    return f"{mean_val:.{digits}f} ± {sd_val:.{digits}f}"


def save_roc(y_true, y_score, title, out_path_png, out_path_pdf):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path_png, dpi=300)
    plt.savefig(out_path_pdf)
    plt.close()


def plot_test_auc_summary(summary_df, outdir):
    order = ["Men", "Women", "AVP", "FIVB"]

    d = summary_df.copy()
    d = d[d["subgroup"].isin(order)].copy()
    d["subgroup"] = pd.Categorical(d["subgroup"], categories=order, ordered=True)
    d = d.sort_values("subgroup")

    labels = d["subgroup"].astype(str).tolist()
    means = d["test_auc_mean"].astype(float).to_numpy()
    sds = d["test_auc_sd"].astype(float).to_numpy()

    plt.figure(figsize=(7, 5))
    x = np.arange(len(labels))
    plt.bar(x, means, yerr=sds, capsize=4)
    plt.xticks(x, labels)
    plt.ylabel("Test AUC")
    plt.title("Global CatBoost test AUC across subgroups")
    plt.ylim(0.0, min(1.0, float(np.max(means + sds)) + 0.05))
    plt.tight_layout()

    plt.savefig(os.path.join(outdir, "fig_priority6_subgroup_test_auc.png"), dpi=300)
    plt.savefig(os.path.join(outdir, "fig_priority6_subgroup_test_auc.pdf"))
    plt.close()


def train_global_model_one_seed(X_train, y_train, X_valid, y_valid, X_test, seed):
    Xt, Xv, Xs, cat_feature_indices = fill_missing(X_train, X_valid, X_test)

    model = CatBoostClassifier(**CATBOOST_PARAMS, random_seed=seed)

    model.fit(
        Xt,
        y_train,
        cat_features=cat_feature_indices,
        eval_set=(Xv, y_valid),
        use_best_model=True,
        verbose=False,
    )

    valid_proba = model.predict_proba(Xv)[:, 1]
    test_proba = model.predict_proba(Xs)[:, 1]

    return valid_proba, test_proba


def summarize_subgroup_runs(runs_df):
    rows = []

    for subgroup, g in runs_df.groupby("subgroup", sort=False):
        row = {
            "subgroup": subgroup,
            "n_runs": len(g),
            "n_valid": int(g["n_valid"].iloc[0]),
            "n_test": int(g["n_test"].iloc[0]),
        }

        for split in ["valid", "test"]:
            for metric in ["auc", "accuracy", "f1", "logloss", "prevalence"]:
                col = f"{split}_{metric}"
                row[f"{col}_mean"] = float(g[col].mean())
                row[f"{col}_sd"] = float(g[col].std(ddof=1))

        rows.append(row)

    return pd.DataFrame(rows)


def make_formatted_subgroup_table(summary_df):
    rows = []
    for _, r in summary_df.iterrows():
        rows.append({
            "Subgroup": r["subgroup"],
            "Validation AUC": format_mean_sd(r["valid_auc_mean"], r["valid_auc_sd"]),
            "Validation Accuracy": format_mean_sd(r["valid_accuracy_mean"], r["valid_accuracy_sd"]),
            "Validation F1-score": format_mean_sd(r["valid_f1_mean"], r["valid_f1_sd"]),
            "Validation Log-loss": format_mean_sd(r["valid_logloss_mean"], r["valid_logloss_sd"]),
            "Test AUC": format_mean_sd(r["test_auc_mean"], r["test_auc_sd"]),
            "Test Accuracy": format_mean_sd(r["test_accuracy_mean"], r["test_accuracy_sd"]),
            "Test F1-score": format_mean_sd(r["test_f1_mean"], r["test_f1_sd"]),
            "Test Log-loss": format_mean_sd(r["test_logloss_mean"], r["test_logloss_sd"]),
        })
    return pd.DataFrame(rows)


def summarize_overall_runs(overall_runs_df):
    row = {"n_runs": len(overall_runs_df)}
    for split in ["valid", "test"]:
        for metric in ["auc", "accuracy", "f1", "logloss", "prevalence"]:
            col = f"{split}_{metric}"
            row[f"{col}_mean"] = float(overall_runs_df[col].mean())
            row[f"{col}_sd"] = float(overall_runs_df[col].std(ddof=1))
    return pd.DataFrame([row])


def main():
    parser = argparse.ArgumentParser(
        description="Priority 6: global CatBoost subgroup evaluation without subgroup-specific retraining."
    )

    parser.add_argument(
        "--input",
        type=str,
        default="/data2/zhangyiyi/beach_ai/final_catboost_v1/data/prematch_features_v3.csv",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="/data2/zhangyiyi/beach_ai/final_catboost_v1/priority6_global_subgroup_output",
    )
    parser.add_argument("--train-end", type=str, default="2017-12-31")
    parser.add_argument("--valid-end", type=str, default="2019-12-31")
    parser.add_argument("--seeds", type=int, nargs="+", default=FINAL_SEEDS)
    parser.add_argument("--min-subgroup-samples", type=int, default=100)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 90)
    print("PRIORITY 6: GLOBAL-MODEL SUBGROUP EVALUATION")
    print("=" * 90)
    print("Input :", args.input)
    print("Output:", args.outdir)
    print("Seeds :", args.seeds)
    print("Design: one global model per seed; subgroup stratification AFTER prediction.")

    df = load_csv(args.input)
    df = prepare_date_and_sort(df)
    validate_required_columns(df)
    target_col = detect_target_column(df)

    train_df, valid_df, test_df = split_by_time(
        df,
        train_end=args.train_end,
        valid_end=args.valid_end,
    )

    X_train, y_train = make_xy(train_df, target_col)
    X_valid, y_valid = make_xy(valid_df, target_col)
    X_test, y_test = make_xy(test_df, target_col)

    valid_masks = subgroup_masks(valid_df)
    test_masks = subgroup_masks(test_df)

    print("\nChronological split:")
    print(f"Train: {len(train_df):,}")
    print(f"Valid: {len(valid_df):,}")
    print(f"Test : {len(test_df):,}")

    sample_rows = []
    for subgroup in ["Men", "Women", "AVP", "FIVB"]:
        nv = int(valid_masks[subgroup].sum())
        nt = int(test_masks[subgroup].sum())

        sample_rows.append({
            "subgroup": subgroup,
            "n_valid": nv,
            "n_test": nt,
            "valid_prevalence": float(np.mean(y_valid[valid_masks[subgroup]])) if nv else np.nan,
            "test_prevalence": float(np.mean(y_test[test_masks[subgroup]])) if nt else np.nan,
        })

        if nv < args.min_subgroup_samples or nt < args.min_subgroup_samples:
            raise ValueError(
                f"{subgroup}: too few observations (valid={nv}, test={nt})."
            )
        if len(np.unique(y_valid[valid_masks[subgroup]])) < 2:
            raise ValueError(f"{subgroup}: validation subgroup contains only one class.")
        if len(np.unique(y_test[test_masks[subgroup]])) < 2:
            raise ValueError(f"{subgroup}: test subgroup contains only one class.")

    sample_df = pd.DataFrame(sample_rows)
    sample_df.to_csv(
        os.path.join(args.outdir, "priority6_global_subgroup_sample_sizes.csv"),
        index=False,
    )

    print("\nSubgroup sample sizes:")
    print(sample_df.to_string(index=False))

    subgroup_rows = []
    overall_rows = []
    all_valid_probs = []
    all_test_probs = []

    for i, seed in enumerate(args.seeds, start=1):
        print("\n" + "-" * 90)
        print(f"GLOBAL MODEL {i}/{len(args.seeds)} | seed={seed}")
        print("-" * 90)

        valid_proba, test_proba = train_global_model_one_seed(
            X_train, y_train, X_valid, y_valid, X_test, seed
        )

        all_valid_probs.append(valid_proba)
        all_test_probs.append(test_proba)

        valid_overall = evaluate_binary(y_valid, valid_proba)
        test_overall = evaluate_binary(y_test, test_proba)

        overall_rows.append({
            "seed": seed,
            **{f"valid_{k}": v for k, v in valid_overall.items()},
            **{f"test_{k}": v for k, v in test_overall.items()},
        })

        print(
            f"Overall | VALID AUC={valid_overall['auc']:.5f} | "
            f"TEST AUC={test_overall['auc']:.5f}"
        )

        for subgroup in ["Men", "Women", "AVP", "FIVB"]:
            mv = valid_masks[subgroup]
            mt = test_masks[subgroup]

            valid_metrics = evaluate_binary(y_valid[mv], valid_proba[mv])
            test_metrics = evaluate_binary(y_test[mt], test_proba[mt])

            subgroup_rows.append({
                "subgroup": subgroup,
                "seed": seed,
                "n_train_global": len(train_df),
                "n_valid": int(mv.sum()),
                "n_test": int(mt.sum()),
                "n_features": len(FEATURE_COLUMNS),
                **{f"valid_{k}": v for k, v in valid_metrics.items()},
                **{f"test_{k}": v for k, v in test_metrics.items()},
            })

            print(
                f"{subgroup:6s} | VALID AUC={valid_metrics['auc']:.5f} | "
                f"TEST AUC={test_metrics['auc']:.5f}"
            )

    subgroup_runs_df = pd.DataFrame(subgroup_rows)
    overall_runs_df = pd.DataFrame(overall_rows)

    subgroup_runs_df.to_csv(
        os.path.join(args.outdir, "priority6_global_subgroup_metrics_all_runs.csv"),
        index=False,
    )
    overall_runs_df.to_csv(
        os.path.join(args.outdir, "priority6_global_overall_metrics_all_runs.csv"),
        index=False,
    )

    subgroup_summary_df = summarize_subgroup_runs(subgroup_runs_df)
    subgroup_summary_df.to_csv(
        os.path.join(args.outdir, "priority6_global_subgroup_metrics_summary_mean_sd.csv"),
        index=False,
    )

    formatted_df = make_formatted_subgroup_table(subgroup_summary_df)
    formatted_df.to_csv(
        os.path.join(args.outdir, "priority6_global_subgroup_metrics_formatted.csv"),
        index=False,
    )

    overall_summary_df = summarize_overall_runs(overall_runs_df)
    overall_summary_df.to_csv(
        os.path.join(args.outdir, "priority6_global_overall_metrics_summary.csv"),
        index=False,
    )

    mean_valid_proba = np.mean(np.vstack(all_valid_probs), axis=0)
    mean_test_proba = np.mean(np.vstack(all_test_probs), axis=0)

    valid_pred_df = pd.DataFrame({
        "row_index_within_validation": np.arange(len(valid_df)),
        "y": y_valid,
        "probability_mean_across_seeds": mean_valid_proba,
        "gender": valid_df["gender"].astype(str).to_numpy(),
        "circuit": valid_df["circuit"].astype(str).to_numpy(),
    })
    test_pred_df = pd.DataFrame({
        "row_index_within_test": np.arange(len(test_df)),
        "y": y_test,
        "probability_mean_across_seeds": mean_test_proba,
        "gender": test_df["gender"].astype(str).to_numpy(),
        "circuit": test_df["circuit"].astype(str).to_numpy(),
    })

    if "date" in valid_df.columns:
        valid_pred_df.insert(1, "date", valid_df["date"].astype(str).to_numpy())
    if "date" in test_df.columns:
        test_pred_df.insert(1, "date", test_df["date"].astype(str).to_numpy())

    valid_pred_df.to_csv(
        os.path.join(args.outdir, "priority6_global_valid_predictions_mean_10seeds.csv"),
        index=False,
    )
    test_pred_df.to_csv(
        os.path.join(args.outdir, "priority6_global_test_predictions_mean_10seeds.csv"),
        index=False,
    )

    for subgroup in ["Men", "Women", "AVP", "FIVB"]:
        mv = valid_masks[subgroup]
        mt = test_masks[subgroup]
        tag = subgroup.lower()

        save_roc(
            y_valid[mv],
            mean_valid_proba[mv],
            title=f"Global CatBoost ROC - {subgroup} (VALID)",
            out_path_png=os.path.join(args.outdir, f"roc_{tag}_valid.png"),
            out_path_pdf=os.path.join(args.outdir, f"roc_{tag}_valid.pdf"),
        )
        save_roc(
            y_test[mt],
            mean_test_proba[mt],
            title=f"Global CatBoost ROC - {subgroup} (TEST)",
            out_path_png=os.path.join(args.outdir, f"roc_{tag}_test.png"),
            out_path_pdf=os.path.join(args.outdir, f"roc_{tag}_test.pdf"),
        )

    plot_test_auc_summary(subgroup_summary_df, args.outdir)

    comp_rows = []
    for subgroup in ["Men", "Women", "AVP", "FIVB"]:
        new_auc = float(
            subgroup_summary_df.loc[
                subgroup_summary_df["subgroup"] == subgroup,
                "test_auc_mean",
            ].iloc[0]
        )
        old_auc = OLD_SUBGROUP_SPECIFIC_TEST_AUC[subgroup]

        comp_rows.append({
            "subgroup": subgroup,
            "old_subgroup_specific_test_auc": old_auc,
            "new_global_model_test_auc": new_auc,
            "delta_global_minus_old": new_auc - old_auc,
        })

    comparison_df = pd.DataFrame(comp_rows)
    comparison_df.to_csv(
        os.path.join(args.outdir, "priority6_old_vs_global_subgroup_auc.csv"),
        index=False,
    )

    config = {
        "analysis": "Priority 6 global-model subgroup evaluation",
        "input": args.input,
        "outdir": args.outdir,
        "target_col": target_col,
        "train_end": args.train_end,
        "valid_end": args.valid_end,
        "seeds": args.seeds,
        "decision_threshold": DECISION_THRESHOLD,
        "feature_count": len(FEATURE_COLUMNS),
        "catboost_params": CATBOOST_PARAMS,
        "design": (
            "One global CatBoost model is trained on the complete training set "
            "for each seed. Validation/test predictions are generated for all "
            "observations first, then stratified by gender and circuit. "
            "No subgroup-specific retraining is performed."
        ),
        "uncertainty_reporting": "Mean ± SD across random seeds; no bootstrap CI in Priority 6.",
        "subgroup_significance_testing": False,
    }

    with open(
        os.path.join(args.outdir, "priority6_run_config.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    readme = f"""
Priority 6: Global-model subgroup evaluation
=============================================

Design
------
One global retained-baseline CatBoost model is trained per random seed.

Time split:
Train      : <= {args.train_end}
Validation : > {args.train_end} and <= {args.valid_end}
Test       : > {args.valid_end}

For each seed:
1. Train CatBoost once on the complete training set.
2. Predict the complete validation set.
3. Predict the complete test set.
4. Only after prediction, stratify observations into Men, Women, AVP, and FIVB.
5. Calculate AUC, Accuracy, F1-score, and Log-loss within each subgroup.

No subgroup-specific retraining is performed.

Primary files to inspect
------------------------
1. priority6_global_subgroup_metrics_formatted.csv
2. priority6_global_subgroup_metrics_summary_mean_sd.csv
3. priority6_global_overall_metrics_summary.csv
4. priority6_old_vs_global_subgroup_auc.csv
5. priority6_global_subgroup_sample_sizes.csv

Statistical reporting
---------------------
Mean ± SD across random seeds.
No bootstrap 95% CI is added in this Priority 6 analysis.
No formal between-subgroup significance test is performed.
"""

    with open(
        os.path.join(args.outdir, "README_priority6_global_subgroup.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(readme)

    print("\n" + "=" * 90)
    print("PRIORITY 6 SUBGROUP RESULTS")
    print("=" * 90)
    print(formatted_df.to_string(index=False))

    print("\n" + "=" * 90)
    print("OVERALL GLOBAL-MODEL REPRODUCTION CHECK")
    print("=" * 90)
    print(overall_summary_df.to_string(index=False))

    print("\n" + "=" * 90)
    print("OLD SUBGROUP-SPECIFIC VS NEW GLOBAL-MODEL TEST AUC")
    print("=" * 90)
    print(comparison_df.to_string(index=False))

    print("\nDone. Outputs saved to:")
    print(args.outdir)


if __name__ == "__main__":
    main()
