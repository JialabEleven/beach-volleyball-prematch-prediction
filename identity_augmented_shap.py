#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Priority 12: Identity-augmented CatBoost interpretation
=======================================================

Reviewer request
----------------
"For the identity analysis, show the importance/SHAP of the model with identity,
and soften the 'identity vs. transferable' dichotomy, Elo already encodes
identity-linked history, so the defensible claim is 'derived signals; raw identifiers.'"

Purpose
-------
Interpret the leak-proof, identity-augmented CatBoost model using:
1) CatBoost built-in feature importance averaged across 10 seeds.
2) Mean absolute SHAP values averaged across 10 seeds on the SAME prespecified
   held-out test subset.
3) A dedicated table reporting the six raw identity variables and their ranks.

Important design choices
------------------------
- Uses the same chronological split:
    Train      <= 2017-12-31
    Validation 2018-01-01 to 2019-12-31
    Test       >= 2020-01-01
- Uses the same identity-augmented feature set as the leak-proof identity analysis.
- Uses has_time=True.
- Uses the same 10 random seeds.
- No "best test seed" is selected for interpretation.
- Feature importance and mean |SHAP| are averaged across ALL 10 seeds.
- SHAP uses a fixed test subset (default n=2000) selected once before model fitting.
- Raw identities are treated as categorical features by CatBoost.

Default input
-------------
/data2/zhangyiyi/beach_ai/final_catboost_v1/data/prematch_features_v3.csv

Default output
--------------
/data2/zhangyiyi/beach_ai/final_catboost_v1/identity_augmented_explain_output
"""

import os
import json
import argparse
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, log_loss


SEEDS = [41, 51, 61, 71, 81, 91, 101, 111, 121, 131]

CATBOOST_PARAMS = {
    "iterations": 900,
    "depth": 6,
    "learning_rate": 0.03,
    "l2_leaf_reg": 5,
    "subsample": 0.85,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "has_time": True,
    "allow_writing_files": False,
    "verbose": False,
}

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
    "teamA_blend_elo","teamB_blend_elo","blend_elo_diff",
]

IDENTITY_COLUMNS = [
    "teamA_key","teamB_key",
    "teamA_player1","teamA_player2",
    "teamB_player1","teamB_player2",
]


def read_csv_fallback(path):
    try:
        return pd.read_csv(path, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", low_memory=False)


def prepare_and_split(df):
    df = df.copy()

    required = ["date", "y"] + FEATURE_COLUMNS + IDENTITY_COLUMNS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns:\n" + "\n".join(missing))

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise ValueError(f"Invalid dates: {int(df['date'].isna().sum())}")

    df["y"] = pd.to_numeric(df["y"], errors="raise").astype(int)

    sort_cols = ["date"]
    if "match_num" in df.columns:
        sort_cols.append("match_num")
    df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    train = df[df["date"] <= pd.Timestamp("2017-12-31")].copy().reset_index(drop=True)
    valid = df[
        (df["date"] >= pd.Timestamp("2018-01-01")) &
        (df["date"] <= pd.Timestamp("2019-12-31"))
    ].copy().reset_index(drop=True)
    test = df[df["date"] >= pd.Timestamp("2020-01-01")].copy().reset_index(drop=True)

    return df, train, valid, test


def fit_preprocessor(train, features):
    numeric, categorical = [], []
    medians, modes = {}, {}

    for c in features:
        if pd.api.types.is_numeric_dtype(train[c]):
            numeric.append(c)
            med = pd.to_numeric(train[c], errors="coerce").median()
            medians[c] = 0.0 if pd.isna(med) else float(med)
        else:
            categorical.append(c)
            m = train[c].mode(dropna=True)
            modes[c] = str(m.iloc[0]) if len(m) else "MISSING"

    return numeric, categorical, medians, modes


def transform(df, features, numeric, categorical, medians, modes):
    X = df[features].copy()

    for c in numeric:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(medians[c])

    for c in categorical:
        X[c] = X[c].fillna(modes[c]).astype(str)

    return X


def metric_dict(y, p):
    pred = (p >= 0.5).astype(int)
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    return {
        "auc": float(roc_auc_score(y, p)),
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        "logloss": float(log_loss(y, p)),
    }


def save_bar_plot(df, value_col, error_col, title, xlabel, path_base, topk):
    top = df.sort_values(value_col, ascending=False).head(topk).copy()
    top = top.iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(
        top["feature"],
        top[value_col],
        xerr=top[error_col] if error_col in top.columns else None,
        capsize=3,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Feature")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path_base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(path_base + ".pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Priority 12: identity-augmented CatBoost feature importance and SHAP"
    )
    parser.add_argument(
        "--input",
        default="/data2/zhangyiyi/beach_ai/final_catboost_v1/data/prematch_features_v3.csv"
    )
    parser.add_argument(
        "--outdir",
        default="/data2/zhangyiyi/beach_ai/final_catboost_v1/identity_augmented_explain_output"
    )
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--shap-sample-size", type=int, default=2000)
    parser.add_argument("--shap-sample-seed", type=int, default=20260909)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 92)
    print("PRIORITY 12: IDENTITY-AUGMENTED CATBOOST INTERPRETATION")
    print("=" * 92)
    print("Input:", args.input)
    print("Output:", args.outdir)
    print("Seeds:", args.seeds)

    df, train, valid, test = prepare_and_split(read_csv_fallback(args.input))
    features = FEATURE_COLUMNS + IDENTITY_COLUMNS

    print(f"Total rows       : {len(df):,}")
    print(f"Train            : {len(train):,}")
    print(f"Validation       : {len(valid):,}")
    print(f"Test             : {len(test):,}")
    print(f"Derived features : {len(FEATURE_COLUMNS)}")
    print(f"Raw identities   : {len(IDENTITY_COLUMNS)}")
    print(f"Total features   : {len(features)}")

    numeric, categorical, medians, modes = fit_preprocessor(train, features)

    Xtr = transform(train, features, numeric, categorical, medians, modes)
    Xva = transform(valid, features, numeric, categorical, medians, modes)
    Xte = transform(test, features, numeric, categorical, medians, modes)

    ytr = train["y"].to_numpy()
    yva = valid["y"].to_numpy()
    yte = test["y"].to_numpy()

    cat_idx = [Xtr.columns.get_loc(c) for c in categorical]

    # Fixed SHAP subset selected ONCE and reused for all seeds.
    rng = np.random.RandomState(args.shap_sample_seed)
    if len(Xte) > args.shap_sample_size:
        shap_positions = np.sort(
            rng.choice(len(Xte), size=args.shap_sample_size, replace=False)
        )
    else:
        shap_positions = np.arange(len(Xte))

    Xshap = Xte.iloc[shap_positions].copy()
    yshap = yte[shap_positions]
    shap_pool = Pool(Xshap, label=yshap, cat_features=cat_idx)

    pd.DataFrame({
        "test_position": shap_positions,
        "date": test.iloc[shap_positions]["date"].astype(str).to_numpy(),
        "y": yshap,
    }).to_csv(
        os.path.join(args.outdir, "fixed_shap_test_sample.csv"),
        index=False,
    )

    fi_per_seed = []
    shap_mean_abs_per_seed = []
    performance_rows = []
    shap_sum_matrix = np.zeros((len(Xshap), len(features)), dtype=float)

    for seed in args.seeds:
        print(f"\n===== seed {seed} =====")

        model = CatBoostClassifier(
            **CATBOOST_PARAMS,
            random_seed=seed,
        )

        model.fit(
            Xtr,
            ytr,
            cat_features=cat_idx,
            eval_set=(Xva, yva),
            use_best_model=True,
            verbose=False,
        )

        pva = model.predict_proba(Xva)[:, 1]
        pte = model.predict_proba(Xte)[:, 1]

        mv = metric_dict(yva, pva)
        mt = metric_dict(yte, pte)

        performance_rows.append({
            "seed": seed,
            "best_iteration": int(model.get_best_iteration()),
            **{f"validation_{k}": v for k, v in mv.items()},
            **{f"test_{k}": v for k, v in mt.items()},
        })

        print(
            f"Test AUC={mt['auc']:.4f} | "
            f"Accuracy={mt['accuracy']:.4f} | "
            f"F1={mt['f1']:.4f} | "
            f"LogLoss={mt['logloss']:.4f}"
        )

        # Built-in CatBoost importance
        fi = np.asarray(model.get_feature_importance(), dtype=float)
        fi_per_seed.append(fi)

        # Native CatBoost SHAP values; last column is expected value.
        shap_raw = np.asarray(
            model.get_feature_importance(shap_pool, type="ShapValues"),
            dtype=float
        )
        shap_values = shap_raw[:, :-1]

        shap_sum_matrix += shap_values
        shap_mean_abs_per_seed.append(
            np.mean(np.abs(shap_values), axis=0)
        )

    fi_arr = np.vstack(fi_per_seed)
    shap_abs_arr = np.vstack(shap_mean_abs_per_seed)
    shap_mean_matrix = shap_sum_matrix / len(args.seeds)

    perf = pd.DataFrame(performance_rows)
    perf.to_csv(
        os.path.join(args.outdir, "identity_interpretation_performance_all_runs.csv"),
        index=False,
    )

    perf_summary = pd.DataFrame([{
        "n_runs": len(perf),
        "test_auc_mean": perf["test_auc"].mean(),
        "test_auc_sd": perf["test_auc"].std(ddof=1),
        "test_accuracy_mean": perf["test_accuracy"].mean(),
        "test_accuracy_sd": perf["test_accuracy"].std(ddof=1),
        "test_f1_mean": perf["test_f1"].mean(),
        "test_f1_sd": perf["test_f1"].std(ddof=1),
        "test_logloss_mean": perf["test_logloss"].mean(),
        "test_logloss_sd": perf["test_logloss"].std(ddof=1),
    }])
    perf_summary.to_csv(
        os.path.join(args.outdir, "identity_interpretation_performance_summary.csv"),
        index=False,
    )

    summary = pd.DataFrame({
        "feature": features,
        "feature_class": [
            "Raw identifier" if f in IDENTITY_COLUMNS else "Derived prematch signal"
            for f in features
        ],
        "catboost_importance_mean": fi_arr.mean(axis=0),
        "catboost_importance_sd": fi_arr.std(axis=0, ddof=1),
        "mean_abs_shap_mean": shap_abs_arr.mean(axis=0),
        "mean_abs_shap_sd": shap_abs_arr.std(axis=0, ddof=1),
    })

    summary["catboost_importance_rank"] = (
        summary["catboost_importance_mean"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    summary["mean_abs_shap_rank"] = (
        summary["mean_abs_shap_mean"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    # Relative shares are useful for interpretation.
    total_fi = summary["catboost_importance_mean"].sum()
    total_shap = summary["mean_abs_shap_mean"].sum()
    summary["catboost_importance_share_pct"] = (
        100 * summary["catboost_importance_mean"] / total_fi
        if total_fi != 0 else np.nan
    )
    summary["mean_abs_shap_share_pct"] = (
        100 * summary["mean_abs_shap_mean"] / total_shap
        if total_shap != 0 else np.nan
    )

    summary = summary.sort_values(
        "mean_abs_shap_mean", ascending=False
    ).reset_index(drop=True)

    summary.to_csv(
        os.path.join(args.outdir, "identity_augmented_feature_importance_and_shap_all_features.csv"),
        index=False,
    )

    summary.head(args.topk).to_csv(
        os.path.join(args.outdir, "identity_augmented_top20_by_shap.csv"),
        index=False,
    )

    identity_only = summary[
        summary["feature"].isin(IDENTITY_COLUMNS)
    ].copy().sort_values("mean_abs_shap_rank")

    identity_only.to_csv(
        os.path.join(args.outdir, "raw_identity_feature_importance_and_shap.csv"),
        index=False,
    )

    group_summary = (
        summary.groupby("feature_class", as_index=False)
        .agg(
            n_features=("feature", "count"),
            total_catboost_importance=("catboost_importance_mean", "sum"),
            total_mean_abs_shap=("mean_abs_shap_mean", "sum"),
        )
    )
    group_summary["catboost_importance_share_pct"] = (
        100 * group_summary["total_catboost_importance"]
        / group_summary["total_catboost_importance"].sum()
    )
    group_summary["mean_abs_shap_share_pct"] = (
        100 * group_summary["total_mean_abs_shap"]
        / group_summary["total_mean_abs_shap"].sum()
    )
    group_summary.to_csv(
        os.path.join(args.outdir, "identity_vs_derived_attribution_share.csv"),
        index=False,
    )

    # Save averaged SHAP matrix for reproducibility / optional later plots.
    shap_matrix_df = pd.DataFrame(shap_mean_matrix, columns=features)
    shap_matrix_df.insert(0, "test_position", shap_positions)
    shap_matrix_df.to_csv(
        os.path.join(args.outdir, "mean_shap_values_across_10seeds_fixed_test_sample.csv"),
        index=False,
    )

    # Separate figures; no combined subplot.
    fi_plot_df = summary.sort_values(
        "catboost_importance_mean", ascending=False
    )
    save_bar_plot(
        fi_plot_df,
        "catboost_importance_mean",
        "catboost_importance_sd",
        f"Identity-augmented CatBoost feature importance (Top {args.topk})",
        "Mean CatBoost feature importance",
        os.path.join(args.outdir, f"fig_identity_augmented_feature_importance_top{args.topk}"),
        args.topk,
    )

    save_bar_plot(
        summary,
        "mean_abs_shap_mean",
        "mean_abs_shap_sd",
        f"Identity-augmented CatBoost mean absolute SHAP values (Top {args.topk})",
        "Mean |SHAP value|",
        os.path.join(args.outdir, f"fig_identity_augmented_shap_bar_top{args.topk}"),
        args.topk,
    )

    # Simple identity-only plot to make reviewer inspection easy.
    id_plot = identity_only.sort_values(
        "mean_abs_shap_mean", ascending=True
    )
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(
        id_plot["feature"],
        id_plot["mean_abs_shap_mean"],
        xerr=id_plot["mean_abs_shap_sd"],
        capsize=3,
    )
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_ylabel("Raw identity feature")
    ax.set_title("SHAP importance of raw identity variables")
    fig.tight_layout()
    fig.savefig(
        os.path.join(args.outdir, "fig_raw_identity_shap.png"),
        dpi=300, bbox_inches="tight"
    )
    fig.savefig(
        os.path.join(args.outdir, "fig_raw_identity_shap.pdf"),
        bbox_inches="tight"
    )
    plt.close(fig)

    config = {
        "input": args.input,
        "outdir": args.outdir,
        "train": "<= 2017-12-31",
        "validation": "2018-01-01 to 2019-12-31",
        "test": ">= 2020-01-01",
        "derived_feature_count": len(FEATURE_COLUMNS),
        "identity_feature_count": len(IDENTITY_COLUMNS),
        "total_feature_count": len(features),
        "identity_columns": IDENTITY_COLUMNS,
        "seeds": args.seeds,
        "shap_sample_size_requested": args.shap_sample_size,
        "shap_sample_size_used": int(len(Xshap)),
        "shap_sample_seed": args.shap_sample_seed,
        "catboost_parameters": CATBOOST_PARAMS,
        "interpretation_strategy":
            "Average CatBoost importance and mean absolute SHAP across all 10 seeds; no best-test-seed selection.",
    }
    with open(
        os.path.join(args.outdir, "run_config.json"),
        "w", encoding="utf-8"
    ) as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    readme = """
Priority 12: Identity-augmented CatBoost interpretation

Most important files to inspect:
1. raw_identity_feature_importance_and_shap.csv
2. identity_augmented_top20_by_shap.csv
3. identity_vs_derived_attribution_share.csv
4. identity_interpretation_performance_summary.csv
5. fig_identity_augmented_feature_importance_top20.png
6. fig_identity_augmented_shap_bar_top20.png
7. fig_raw_identity_shap.png

Interpretation:
- If raw identity variables rank low by both CatBoost importance and mean |SHAP|,
  the identity-augmented model relies mainly on derived prematch signals.
- If some raw identifiers rank high but out-of-time performance is worse, the
  defensible interpretation is that the model uses identity-linked patterns that
  do not improve temporal generalization.
- Avoid claiming that Elo is identity-free. Elo is a derived signal that encodes
  historical identity-linked performance; distinguish DERIVED SIGNALS from RAW
  IDENTIFIERS.
""".strip()

    with open(
        os.path.join(args.outdir, "README_priority12_identity_explain.txt"),
        "w", encoding="utf-8"
    ) as f:
        f.write(readme + "\n")

    print("\n" + "=" * 92)
    print("IDENTITY MODEL PERFORMANCE")
    print("=" * 92)
    print(perf_summary.to_string(index=False))

    print("\n" + "=" * 92)
    print("RAW IDENTITY FEATURES")
    print("=" * 92)
    print(
        identity_only[
            [
                "feature",
                "catboost_importance_mean",
                "catboost_importance_rank",
                "mean_abs_shap_mean",
                "mean_abs_shap_rank",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 92)
    print("ATTRIBUTION SHARE: RAW IDENTIFIERS VS DERIVED SIGNALS")
    print("=" * 92)
    print(group_summary.to_string(index=False))

    print("\nSaved to:", args.outdir)


if __name__ == "__main__":
    main()
