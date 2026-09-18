#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Priority 8: Feature-block ablation analysis
===========================================

Reviewer motivation
-------------------
"Feature-block ablations (Elo-only, ranking-only, no-Elo, etc.) to quantify
the marginal contribution of each family."

This script evaluates six prespecified feature configurations using the
retained baseline CatBoost model:

1. Full
2. Elo-only
3. Ranking-only
4. No-Elo
5. No-ranking
6. No-Elo-and-no-ranking

Design
------
- Chronological split:
    Train      <= 2017-12-31
    Validation 2018-01-01 to 2019-12-31
    Test       > 2019-12-31
- Same retained baseline CatBoost hyperparameters
- Same 10 random seeds
- Missing-data imputation fitted on training data only
- Decision threshold = 0.5
- Metrics:
    AUC
    Accuracy
    F1-score
    Log-loss
- Results summarized as mean ± SD across 10 seeds
- No bootstrap CI or DeLong testing in this ablation analysis

Outputs
-------
priority8_feature_ablation_all_runs.csv
priority8_feature_ablation_summary_mean_sd.csv
priority8_feature_ablation_summary_formatted.csv
priority8_feature_ablation_delta_vs_full.csv
priority8_feature_sets.csv
fig_priority8_feature_ablation_test_auc.png
fig_priority8_feature_ablation_test_auc.pdf
priority8_run_config.json
README_priority8_feature_ablation.txt
"""

import os
import json
import argparse
from typing import Dict, List, Tuple

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
)


# ============================================================
# GLOBAL SETTINGS
# ============================================================

FINAL_SEEDS = [41, 51, 61, 71, 81, 91, 101, 111, 121, 131]

DECISION_THRESHOLD = 0.5

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


# ============================================================
# RETAINED BASELINE FEATURE SET
# ============================================================

FULL_FEATURES = [
    # Context
    "gender", "circuit", "country", "bracket", "round", "month",

    # Age / height
    "teamA_avg_age", "teamB_avg_age", "age_diff",
    "teamA_avg_hgt", "teamB_avg_hgt", "hgt_diff",

    # Ranking
    "teamA_rank", "teamB_rank", "rank_diff",

    # Qualifier-related
    "teamA_is_qualifier", "teamB_is_qualifier",
    "teamA_qualifier_num", "teamB_qualifier_num", "qualifier_num_diff",
    "one_side_qualifier", "both_qualifier",
    "teamA_same_country", "teamB_same_country",

    # Pair/team history
    "teamA_is_new_pair", "teamB_is_new_pair",
    "teamA_prior_matches", "teamB_prior_matches", "pair_prior_matches_diff",
    "teamA_prior_win_pct", "teamB_prior_win_pct", "pair_prior_win_pct_diff",

    # Recent form
    "teamA_recent3_win_pct", "teamB_recent3_win_pct", "recent3_win_pct_diff",
    "teamA_recent5_win_pct", "teamB_recent5_win_pct", "recent5_win_pct_diff",
    "teamA_recent10_win_pct", "teamB_recent10_win_pct", "recent10_win_pct_diff",
    "teamA_recent20_win_pct", "teamB_recent20_win_pct", "recent20_win_pct_diff",

    # Recent activity
    "teamA_last30_match_count", "teamB_last30_match_count", "last30_match_count_diff",
    "teamA_last30_win_pct", "teamB_last30_win_pct", "last30_win_pct_diff",
    "teamA_last90_match_count", "teamB_last90_match_count", "last90_match_count_diff",
    "teamA_last90_win_pct", "teamB_last90_win_pct", "last90_win_pct_diff",
    "teamA_last180_match_count", "teamB_last180_match_count", "last180_match_count_diff",
    "teamA_last180_win_pct", "teamB_last180_win_pct", "last180_win_pct_diff",
    "teamA_days_since_last_match", "teamB_days_since_last_match", "days_since_last_match_diff",

    # Player summaries
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

    "teamA_avg_player_days_since_last_match",
    "teamB_avg_player_days_since_last_match",
    "avg_player_days_since_last_match_diff",

    "teamA_both_players_new", "teamB_both_players_new",

    # Head-to-head
    "h2h_prior_matches",
    "h2h_teamA_win_pct",
    "h2h_recent3_teamA_win_pct",

    # Elo
    "teamA_elo", "teamB_elo", "elo_diff",
    "teamA_player1_elo", "teamA_player2_elo",
    "teamB_player1_elo", "teamB_player2_elo",
    "teamA_avg_player_elo", "teamB_avg_player_elo", "avg_player_elo_diff",
    "teamA_blend_elo", "teamB_blend_elo", "blend_elo_diff",
]

ELO_FEATURES = [
    "teamA_elo", "teamB_elo", "elo_diff",
    "teamA_player1_elo", "teamA_player2_elo",
    "teamB_player1_elo", "teamB_player2_elo",
    "teamA_avg_player_elo", "teamB_avg_player_elo", "avg_player_elo_diff",
    "teamA_blend_elo", "teamB_blend_elo", "blend_elo_diff",
]

RANKING_FEATURES = [
    "teamA_rank",
    "teamB_rank",
    "rank_diff",
]


# ============================================================
# DATA UTILITIES
# ============================================================

def load_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", low_memory=False)


def detect_target_column(df: pd.DataFrame) -> str:
    candidates = [
        "y", "target", "label", "outcome", "result",
        "teamA_win", "team1_win", "win",
    ]
    for c in candidates:
        if c in df.columns:
            return c

    raise ValueError(
        f"Could not detect target column automatically. Tried: {candidates}"
    )


def prepare_date_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")

        if out["date"].isna().any():
            raise ValueError(
                f"Invalid or missing dates found: {int(out['date'].isna().sum())}"
            )

        sort_cols = ["date"]
        if "year" in out.columns:
            sort_cols.append("year")
        if "match_num" in out.columns:
            sort_cols.append("match_num")

        out = out.sort_values(
            sort_cols,
            kind="mergesort"
        ).reset_index(drop=True)

    elif "year" in out.columns:
        out["year"] = pd.to_numeric(
            out["year"],
            errors="raise"
        )
        out = out.sort_values(
            ["year"],
            kind="mergesort"
        ).reset_index(drop=True)

    else:
        raise ValueError(
            "Need either a 'date' or 'year' column for chronological splitting."
        )

    return out


def split_by_time(
    df: pd.DataFrame,
    train_end: str = "2017-12-31",
    valid_end: str = "2019-12-31",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    if "date" in df.columns:
        train_end_ts = pd.Timestamp(train_end)
        valid_end_ts = pd.Timestamp(valid_end)

        train_df = df[
            df["date"] <= train_end_ts
        ].copy()

        valid_df = df[
            (df["date"] > train_end_ts) &
            (df["date"] <= valid_end_ts)
        ].copy()

        test_df = df[
            df["date"] > valid_end_ts
        ].copy()

    else:
        train_df = df[
            df["year"] <= 2017
        ].copy()

        valid_df = df[
            (df["year"] >= 2018) &
            (df["year"] <= 2019)
        ].copy()

        test_df = df[
            df["year"] >= 2020
        ].copy()

    if min(
        len(train_df),
        len(valid_df),
        len(test_df)
    ) == 0:
        raise ValueError(
            "At least one chronological split is empty: "
            f"train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}"
        )

    return (
        train_df.reset_index(drop=True),
        valid_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def validate_features_exist(df: pd.DataFrame):
    missing = [
        c for c in FULL_FEATURES
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            "The following required baseline features are missing:\n"
            + "\n".join(missing)
        )


def make_xy(
    df: pd.DataFrame,
    target_col: str,
    feature_columns: List[str],
):
    X = df[
        feature_columns
    ].copy()

    y = pd.to_numeric(
        df[target_col],
        errors="raise"
    ).astype(int).to_numpy()

    if not set(
        np.unique(y)
    ).issubset({0, 1}):
        raise ValueError(
            f"Target must be binary 0/1. Found: {np.unique(y)}"
        )

    return X, y


def fill_missing(
    train_X: pd.DataFrame,
    valid_X: pd.DataFrame,
    test_X: pd.DataFrame,
):
    """
    Fit imputation using TRAINING DATA ONLY.

    Numeric:
        training median

    Categorical:
        training mode
    """
    train_X = train_X.copy()
    valid_X = valid_X.copy()
    test_X = test_X.copy()

    num_cols = []
    cat_cols = []

    for c in train_X.columns:
        if pd.api.types.is_numeric_dtype(
            train_X[c]
        ):
            num_cols.append(c)
        else:
            cat_cols.append(c)

    for c in num_cols:
        med = train_X[c].median()

        train_X[c] = train_X[c].fillna(
            med
        )
        valid_X[c] = valid_X[c].fillna(
            med
        )
        test_X[c] = test_X[c].fillna(
            med
        )

    for c in cat_cols:
        mode_series = train_X[c].mode(
            dropna=True
        )

        mode_val = (
            mode_series.iloc[0]
            if not mode_series.empty
            else "missing"
        )

        train_X[c] = train_X[c].fillna(
            mode_val
        ).astype(str)

        valid_X[c] = valid_X[c].fillna(
            mode_val
        ).astype(str)

        test_X[c] = test_X[c].fillna(
            mode_val
        ).astype(str)

    cat_feature_indices = [
        train_X.columns.get_loc(c)
        for c in cat_cols
    ]

    return (
        train_X,
        valid_X,
        test_X,
        cat_feature_indices,
    )


# ============================================================
# FEATURE CONFIGURATIONS
# ============================================================

def build_feature_sets() -> Dict[str, List[str]]:
    no_elo = [
        c for c in FULL_FEATURES
        if c not in ELO_FEATURES
    ]

    no_ranking = [
        c for c in FULL_FEATURES
        if c not in RANKING_FEATURES
    ]

    no_elo_no_ranking = [
        c for c in FULL_FEATURES
        if c not in ELO_FEATURES
        and c not in RANKING_FEATURES
    ]

    return {
        "Full": FULL_FEATURES.copy(),
        "Elo-only": ELO_FEATURES.copy(),
        "Ranking-only": RANKING_FEATURES.copy(),
        "No-Elo": no_elo,
        "No-ranking": no_ranking,
        "No-Elo-and-no-ranking": no_elo_no_ranking,
    }


# ============================================================
# METRICS
# ============================================================

def evaluate_binary(
    y_true,
    proba,
    threshold=DECISION_THRESHOLD,
):
    y_true = np.asarray(
        y_true
    )

    proba = np.asarray(
        proba,
        dtype=float
    )

    proba = np.clip(
        proba,
        1e-12,
        1 - 1e-12
    )

    pred = (
        proba >= threshold
    ).astype(int)

    return {
        "auc": float(
            roc_auc_score(
                y_true,
                proba
            )
        ),
        "accuracy": float(
            accuracy_score(
                y_true,
                pred
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                pred
            )
        ),
        "logloss": float(
            log_loss(
                y_true,
                proba
            )
        ),
    }


def format_mean_sd(
    mean_val,
    sd_val,
    digits=4,
):
    return (
        f"{mean_val:.{digits}f} "
        f"± {sd_val:.{digits}f}"
    )


# ============================================================
# MODEL FITTING
# ============================================================

def run_feature_configuration(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    feature_set_name: str,
    feature_columns: List[str],
    seeds: List[int],
):
    """
    Run one feature configuration across all seeds.
    """
    X_train, y_train = make_xy(
        train_df,
        target_col,
        feature_columns
    )

    X_valid, y_valid = make_xy(
        valid_df,
        target_col,
        feature_columns
    )

    X_test, y_test = make_xy(
        test_df,
        target_col,
        feature_columns
    )

    rows = []

    for i, seed in enumerate(
        seeds,
        start=1
    ):
        print(
            f"  seed {i}/{len(seeds)} = {seed}"
        )

        Xt, Xv, Xs, cat_feature_indices = fill_missing(
            X_train,
            X_valid,
            X_test
        )

        model = CatBoostClassifier(
            **CATBOOST_PARAMS,
            random_seed=seed,
        )

        fit_kwargs = {
            "X": Xt,
            "y": y_train,
            "eval_set": (Xv, y_valid),
            "use_best_model": True,
            "verbose": False,
        }

        if len(
            cat_feature_indices
        ) > 0:
            fit_kwargs[
                "cat_features"
            ] = cat_feature_indices

        model.fit(
            **fit_kwargs
        )

        valid_proba = (
            model.predict_proba(
                Xv
            )[:, 1]
        )

        test_proba = (
            model.predict_proba(
                Xs
            )[:, 1]
        )

        valid_metrics = evaluate_binary(
            y_valid,
            valid_proba
        )

        test_metrics = evaluate_binary(
            y_test,
            test_proba
        )

        best_iteration = (
            model.get_best_iteration()
        )

        rows.append({
            "feature_set": feature_set_name,
            "seed": seed,
            "n_features": len(
                feature_columns
            ),
            "n_train": len(
                train_df
            ),
            "n_valid": len(
                valid_df
            ),
            "n_test": len(
                test_df
            ),
            "best_iteration": best_iteration,
            **{
                f"valid_{k}": v
                for k, v in valid_metrics.items()
            },
            **{
                f"test_{k}": v
                for k, v in test_metrics.items()
            },
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# SUMMARIES
# ============================================================

def summarize_runs(
    runs_df: pd.DataFrame
):
    rows = []

    feature_order = [
        "Full",
        "Elo-only",
        "Ranking-only",
        "No-Elo",
        "No-ranking",
        "No-Elo-and-no-ranking",
    ]

    for feature_set in feature_order:
        g = runs_df[
            runs_df[
                "feature_set"
            ] == feature_set
        ].copy()

        if len(g) == 0:
            continue

        row = {
            "feature_set": feature_set,
            "n_features": int(
                g[
                    "n_features"
                ].iloc[0]
            ),
            "n_runs": len(
                g
            ),
        }

        for split in [
            "valid",
            "test",
        ]:
            for metric in [
                "auc",
                "accuracy",
                "f1",
                "logloss",
            ]:
                col = (
                    f"{split}_{metric}"
                )

                row[
                    f"{col}_mean"
                ] = float(
                    g[col].mean()
                )

                row[
                    f"{col}_sd"
                ] = float(
                    g[col].std(
                        ddof=1
                    )
                )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def make_formatted_table(
    summary_df: pd.DataFrame
):
    rows = []

    for _, r in summary_df.iterrows():
        rows.append({
            "Feature set": r[
                "feature_set"
            ],
            "N features": int(
                r[
                    "n_features"
                ]
            ),
            "Validation AUC": format_mean_sd(
                r[
                    "valid_auc_mean"
                ],
                r[
                    "valid_auc_sd"
                ]
            ),
            "Validation Accuracy": format_mean_sd(
                r[
                    "valid_accuracy_mean"
                ],
                r[
                    "valid_accuracy_sd"
                ]
            ),
            "Validation F1-score": format_mean_sd(
                r[
                    "valid_f1_mean"
                ],
                r[
                    "valid_f1_sd"
                ]
            ),
            "Validation Log-loss": format_mean_sd(
                r[
                    "valid_logloss_mean"
                ],
                r[
                    "valid_logloss_sd"
                ]
            ),
            "Test AUC": format_mean_sd(
                r[
                    "test_auc_mean"
                ],
                r[
                    "test_auc_sd"
                ]
            ),
            "Test Accuracy": format_mean_sd(
                r[
                    "test_accuracy_mean"
                ],
                r[
                    "test_accuracy_sd"
                ]
            ),
            "Test F1-score": format_mean_sd(
                r[
                    "test_f1_mean"
                ],
                r[
                    "test_f1_sd"
                ]
            ),
            "Test Log-loss": format_mean_sd(
                r[
                    "test_logloss_mean"
                ],
                r[
                    "test_logloss_sd"
                ]
            ),
        })

    return pd.DataFrame(
        rows
    )


def make_delta_table(
    summary_df: pd.DataFrame
):
    full_row = summary_df[
        summary_df[
            "feature_set"
        ] == "Full"
    ]

    if len(full_row) != 1:
        raise ValueError(
            "Could not uniquely identify the Full model."
        )

    full = full_row.iloc[0]

    rows = []

    for _, r in summary_df.iterrows():
        rows.append({
            "feature_set": r[
                "feature_set"
            ],
            "n_features": int(
                r[
                    "n_features"
                ]
            ),
            "test_auc_mean": r[
                "test_auc_mean"
            ],
            "delta_auc_vs_full": (
                r[
                    "test_auc_mean"
                ]
                - full[
                    "test_auc_mean"
                ]
            ),
            "test_accuracy_mean": r[
                "test_accuracy_mean"
            ],
            "delta_accuracy_vs_full": (
                r[
                    "test_accuracy_mean"
                ]
                - full[
                    "test_accuracy_mean"
                ]
            ),
            "test_f1_mean": r[
                "test_f1_mean"
            ],
            "delta_f1_vs_full": (
                r[
                    "test_f1_mean"
                ]
                - full[
                    "test_f1_mean"
                ]
            ),
            "test_logloss_mean": r[
                "test_logloss_mean"
            ],
            "delta_logloss_vs_full": (
                r[
                    "test_logloss_mean"
                ]
                - full[
                    "test_logloss_mean"
                ]
            ),
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# PLOT
# ============================================================

def plot_test_auc(
    summary_df: pd.DataFrame,
    outdir: str,
):
    order = [
        "Full",
        "Elo-only",
        "Ranking-only",
        "No-Elo",
        "No-ranking",
        "No-Elo-and-no-ranking",
    ]

    d = summary_df.copy()

    d[
        "feature_set"
    ] = pd.Categorical(
        d[
            "feature_set"
        ],
        categories=order,
        ordered=True
    )

    d = d.sort_values(
        "feature_set"
    ).reset_index(
        drop=True
    )

    labels = d[
        "feature_set"
    ].astype(
        str
    ).tolist()

    means = d[
        "test_auc_mean"
    ].astype(
        float
    ).to_numpy()

    sds = d[
        "test_auc_sd"
    ].astype(
        float
    ).to_numpy()

    plt.figure(
        figsize=(9, 5.5)
    )

    x = np.arange(
        len(labels)
    )

    plt.bar(
        x,
        means,
        yerr=sds,
        capsize=4
    )

    plt.xticks(
        x,
        labels,
        rotation=20,
        ha="right"
    )

    plt.ylabel(
        "Test AUC"
    )

    plt.title(
        "Feature-block ablation analysis"
    )

    ymin = max(
        0.0,
        float(
            np.min(
                means - sds
            )
        ) - 0.05
    )

    ymax = min(
        1.0,
        float(
            np.max(
                means + sds
            )
        ) + 0.03
    )

    plt.ylim(
        ymin,
        ymax
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            outdir,
            "fig_priority8_feature_ablation_test_auc.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.savefig(
        os.path.join(
            outdir,
            "fig_priority8_feature_ablation_test_auc.pdf"
        ),
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Priority 8 feature-block ablation analysis "
            "using the retained baseline CatBoost model."
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        default=(
            "/data2/zhangyiyi/beach_ai/"
            "final_catboost_v1/data/"
            "prematch_features_v3.csv"
        ),
    )

    parser.add_argument(
        "--outdir",
        type=str,
        default=(
            "/data2/zhangyiyi/beach_ai/"
            "final_catboost_v1/"
            "priority8_feature_ablation_output"
        ),
    )

    parser.add_argument(
        "--train-end",
        type=str,
        default="2017-12-31"
    )

    parser.add_argument(
        "--valid-end",
        type=str,
        default="2019-12-31"
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=FINAL_SEEDS
    )

    args = parser.parse_args()

    os.makedirs(
        args.outdir,
        exist_ok=True
    )

    print(
        "=" * 90
    )
    print(
        "PRIORITY 8: FEATURE-BLOCK ABLATION"
    )
    print(
        "=" * 90
    )

    print(
        f"Input : {args.input}"
    )
    print(
        f"Output: {args.outdir}"
    )
    print(
        f"Seeds : {args.seeds}"
    )

    df = load_csv(
        args.input
    )

    df = prepare_date_and_sort(
        df
    )

    validate_features_exist(
        df
    )

    target_col = detect_target_column(
        df
    )

    train_df, valid_df, test_df = split_by_time(
        df,
        train_end=args.train_end,
        valid_end=args.valid_end
    )

    print(
        "\nChronological split:"
    )
    print(
        f"Train: {len(train_df):,}"
    )
    print(
        f"Valid: {len(valid_df):,}"
    )
    print(
        f"Test : {len(test_df):,}"
    )

    feature_sets = build_feature_sets()

    # --------------------------------------------------------
    # Save exact feature definitions
    # --------------------------------------------------------
    feature_rows = []

    for set_name, columns in feature_sets.items():
        for c in columns:
            if c in ELO_FEATURES:
                family = "Elo"
            elif c in RANKING_FEATURES:
                family = "Ranking"
            else:
                family = "Other prematch"

            feature_rows.append({
                "feature_set": set_name,
                "feature": c,
                "feature_family": family,
            })

    pd.DataFrame(
        feature_rows
    ).to_csv(
        os.path.join(
            args.outdir,
            "priority8_feature_sets.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Run all six feature configurations
    # --------------------------------------------------------
    ordered_sets = [
        "Full",
        "Elo-only",
        "Ranking-only",
        "No-Elo",
        "No-ranking",
        "No-Elo-and-no-ranking",
    ]

    all_runs = []

    for set_name in ordered_sets:
        features = feature_sets[
            set_name
        ]

        print(
            "\n" + "=" * 90
        )
        print(
            f"Feature set: {set_name}"
        )
        print(
            f"N features : {len(features)}"
        )
        print(
            "=" * 90
        )

        runs = run_feature_configuration(
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,
            target_col=target_col,
            feature_set_name=set_name,
            feature_columns=features,
            seeds=args.seeds,
        )

        all_runs.append(
            runs
        )

    runs_df = pd.concat(
        all_runs,
        ignore_index=True
    )

    runs_df.to_csv(
        os.path.join(
            args.outdir,
            "priority8_feature_ablation_all_runs.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Summaries
    # --------------------------------------------------------
    summary_df = summarize_runs(
        runs_df
    )

    summary_df.to_csv(
        os.path.join(
            args.outdir,
            "priority8_feature_ablation_summary_mean_sd.csv"
        ),
        index=False
    )

    formatted_df = make_formatted_table(
        summary_df
    )

    formatted_df.to_csv(
        os.path.join(
            args.outdir,
            "priority8_feature_ablation_summary_formatted.csv"
        ),
        index=False
    )

    delta_df = make_delta_table(
        summary_df
    )

    delta_df.to_csv(
        os.path.join(
            args.outdir,
            "priority8_feature_ablation_delta_vs_full.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------
    plot_test_auc(
        summary_df,
        args.outdir
    )

    # --------------------------------------------------------
    # Run configuration
    # --------------------------------------------------------
    config = {
        "analysis": "Priority 8 feature-block ablation",
        "input": args.input,
        "outdir": args.outdir,
        "target_col": target_col,
        "train_end": args.train_end,
        "valid_end": args.valid_end,
        "seeds": args.seeds,
        "decision_threshold": DECISION_THRESHOLD,
        "catboost_params": CATBOOST_PARAMS,
        "full_feature_count": len(FULL_FEATURES),
        "elo_feature_count": len(ELO_FEATURES),
        "ranking_feature_count": len(RANKING_FEATURES),
        "feature_sets": feature_sets,
        "uncertainty_reporting": "Mean ± SD across random seeds",
        "bootstrap_ci": False,
        "delong_testing": False,
    }

    with open(
        os.path.join(
            args.outdir,
            "priority8_run_config.json"
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
Priority 8 Feature-block Ablation
=================================

Purpose
-------
Quantify the contribution of Elo and ranking feature families while keeping
the model, preprocessing, chronological split, and hyperparameters fixed.

Chronological split
-------------------
Train      <= {args.train_end}
Validation > {args.train_end} and <= {args.valid_end}
Test       > {args.valid_end}

Feature configurations
----------------------
Full:
    All {len(FULL_FEATURES)} retained baseline prematch features.

Elo-only:
    Only the {len(ELO_FEATURES)} Elo-derived variables.

Ranking-only:
    Only teamA_rank, teamB_rank, and rank_diff.

No-Elo:
    Full feature set with all Elo-derived variables removed.

No-ranking:
    Full feature set with the three ranking variables removed.

No-Elo-and-no-ranking:
    Full feature set with both Elo and ranking variables removed.

Metrics
-------
AUC
Accuracy
F1-score
Log-loss

Reporting
---------
Mean ± SD across 10 random seeds.

No bootstrap confidence intervals or DeLong tests are performed here because
this is a controlled feature-ablation sensitivity analysis.

Main files to inspect
---------------------
1. priority8_feature_ablation_summary_formatted.csv
2. priority8_feature_ablation_delta_vs_full.csv
3. fig_priority8_feature_ablation_test_auc.png
4. priority8_feature_sets.csv

Interpretation
--------------
Full vs Elo-only:
    How much predictive information is captured by Elo alone?

Full vs No-Elo:
    What is the marginal contribution of Elo to the complete model?

Full vs Ranking-only:
    How much predictive information is captured by ranking alone?

Full vs No-ranking:
    What is the marginal contribution of ranking within the full model?

No-Elo-and-no-ranking:
    How much discrimination remains in the remaining prematch information?
"""

    with open(
        os.path.join(
            args.outdir,
            "README_priority8_feature_ablation.txt"
        ),
        "w",
        encoding="utf-8"
    ) as f:
        f.write(
            readme
        )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------
    print(
        "\n" + "=" * 90
    )
    print(
        "FEATURE ABLATION SUMMARY"
    )
    print(
        "=" * 90
    )
    print(
        formatted_df.to_string(
            index=False
        )
    )

    print(
        "\n" + "=" * 90
    )
    print(
        "DELTA VS FULL"
    )
    print(
        "=" * 90
    )
    print(
        delta_df.to_string(
            index=False
        )
    )

    print(
        "\nDone. Outputs saved to:"
    )
    print(
        args.outdir
    )


if __name__ == "__main__":
    main()
