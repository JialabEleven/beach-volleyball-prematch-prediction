#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reviewer 2 - Trivial-favorite baseline analysis

Compares:
1) Elo favorite heuristic
2) Univariate Logistic Regression using avg_player_elo_diff only
3) Full CatBoost prematch model

Metrics:
- AUC
- Accuracy
- F1-score
- Log-loss

Evaluation:
- Train:      <= 2017-12-31
- Validation: 2018-01-01 to 2019-12-31
- Test:       > 2019-12-31
- Full CatBoost repeated across 10 random seeds
- Elo heuristic and Elo-only Logistic Regression are deterministic

The Elo favorite heuristic uses the standard Elo expected-score transform:
    P(Team A wins) = 1 / (1 + 10^(-EloDiff/400))
where EloDiff = avg_player_elo_diff = TeamA avg player Elo - TeamB avg player Elo.

Outputs:
- test_metrics_all_runs.csv
- validation_metrics_all_runs.csv
- comparison_summary_test.csv
- comparison_summary_validation.csv
- incremental_gain_test.csv
- run_config.json
"""

import os
import json
import argparse
import warnings

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, log_loss


warnings.filterwarnings("ignore")

SEEDS = [41, 51, 61, 71, 81, 91, 101, 111, 121, 131]
ELO_DIFF_COL = "avg_player_elo_diff"


# Exact baseline feature set used by the retained CatBoost model
FEATURE_COLUMNS = [
    "gender","circuit","country","bracket","round","month",

    "teamA_avg_age","teamB_avg_age","age_diff",
    "teamA_avg_hgt","teamB_avg_hgt","hgt_diff",

    "teamA_rank","teamB_rank","rank_diff",

    "teamA_is_qualifier","teamB_is_qualifier",
    "teamA_qualifier_num","teamB_qualifier_num",
    "qualifier_num_diff",

    "one_side_qualifier","both_qualifier",

    "teamA_same_country","teamB_same_country",

    "teamA_is_new_pair","teamB_is_new_pair",

    "teamA_prior_matches","teamB_prior_matches",
    "pair_prior_matches_diff",

    "teamA_prior_win_pct","teamB_prior_win_pct",
    "pair_prior_win_pct_diff",

    "teamA_recent3_win_pct","teamB_recent3_win_pct",
    "recent3_win_pct_diff",

    "teamA_recent5_win_pct","teamB_recent5_win_pct",
    "recent5_win_pct_diff",

    "teamA_recent10_win_pct","teamB_recent10_win_pct",
    "recent10_win_pct_diff",

    "teamA_recent20_win_pct","teamB_recent20_win_pct",
    "recent20_win_pct_diff",

    "teamA_last30_match_count","teamB_last30_match_count",
    "last30_match_count_diff",

    "teamA_last30_win_pct","teamB_last30_win_pct",
    "last30_win_pct_diff",

    "teamA_last90_match_count","teamB_last90_match_count",
    "last90_match_count_diff",

    "teamA_last90_win_pct","teamB_last90_win_pct",
    "last90_win_pct_diff",

    "teamA_last180_match_count","teamB_last180_match_count",
    "last180_match_count_diff",

    "teamA_last180_win_pct","teamB_last180_win_pct",
    "last180_win_pct_diff",

    "teamA_days_since_last_match",
    "teamB_days_since_last_match",
    "days_since_last_match_diff",

    "teamA_avg_player_prior_matches",
    "teamB_avg_player_prior_matches",
    "avg_player_prior_matches_diff",

    "teamA_avg_player_prior_win_pct",
    "teamB_avg_player_prior_win_pct",
    "avg_player_prior_win_pct_diff",

    "teamA_avg_player_recent10_win_pct",
    "teamB_avg_player_recent10_win_pct",
    "avg_player_recent10_win_pct_diff",

    "teamA_avg_player_recent20_win_pct",
    "teamB_avg_player_recent20_win_pct",
    "avg_player_recent20_win_pct_diff",

    "teamA_avg_player_last90_match_count",
    "teamB_avg_player_last90_match_count",
    "avg_player_last90_match_count_diff",

    "teamA_avg_player_last90_win_pct",
    "teamB_avg_player_last90_win_pct",
    "avg_player_last90_win_pct_diff",

    "teamA_avg_player_days_since_last_match",
    "teamB_avg_player_days_since_last_match",
    "avg_player_days_since_last_match_diff",

    "teamA_both_players_new",
    "teamB_both_players_new",

    "h2h_prior_matches",
    "h2h_teamA_win_pct",
    "h2h_recent3_teamA_win_pct",

    "teamA_elo","teamB_elo","elo_diff",

    "teamA_player1_elo","teamA_player2_elo",
    "teamB_player1_elo","teamB_player2_elo",

    "teamA_avg_player_elo",
    "teamB_avg_player_elo",
    "avg_player_elo_diff",

    "teamA_blend_elo",
    "teamB_blend_elo",
    "blend_elo_diff"
]


def time_split(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="raise")

    train = df[df["date"] <= "2017-12-31"].copy()
    valid = df[
        (df["date"] > "2017-12-31") &
        (df["date"] <= "2019-12-31")
    ].copy()
    test = df[df["date"] > "2019-12-31"].copy()

    return train, valid, test


def evaluate(y_true, prob):
    prob = np.asarray(prob, dtype=float)
    prob = np.clip(prob, 1e-12, 1 - 1e-12)
    pred = (prob >= 0.5).astype(int)

    return {
        "auc": roc_auc_score(y_true, prob),
        "accuracy": accuracy_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "logloss": log_loss(y_true, prob),
    }


def elo_expected_probability(elo_diff):
    """
    Standard Elo expected score.
    elo_diff = rating_A - rating_B
    """
    x = np.asarray(elo_diff, dtype=float)
    return 1.0 / (1.0 + np.power(10.0, -x / 400.0))


def prepare_elo_diff(train, valid, test):
    """
    Uses training-set median only if Elo-diff is missing.
    In the intended dataset, Elo-derived features should usually be available
    because ratings are initialized before online updating.
    """
    train_x = pd.to_numeric(train[ELO_DIFF_COL], errors="coerce")
    valid_x = pd.to_numeric(valid[ELO_DIFF_COL], errors="coerce")
    test_x = pd.to_numeric(test[ELO_DIFF_COL], errors="coerce")

    train_median = float(train_x.median())

    return (
        train_x.fillna(train_median).to_numpy(dtype=float),
        valid_x.fillna(train_median).to_numpy(dtype=float),
        test_x.fillna(train_median).to_numpy(dtype=float),
        train_median
    )


def fit_elo_only_logistic(train_x, train_y, valid_x, test_x):
    """
    Univariate logistic regression on avg_player_elo_diff.
    No test information is used during fitting.
    """
    model = LogisticRegression(
        solver="lbfgs",
        max_iter=5000,
        random_state=42
    )
    model.fit(train_x.reshape(-1, 1), train_y)

    valid_prob = model.predict_proba(valid_x.reshape(-1, 1))[:, 1]
    test_prob = model.predict_proba(test_x.reshape(-1, 1))[:, 1]

    return model, valid_prob, test_prob


def fit_catboost_preprocessor(train, valid, test):
    """
    Preprocessing aligned with the baseline CatBoost workflow:
    - Numeric: training-set median imputation
    - Categorical: training-set mode imputation, cast to string
    """
    X_train = train[FEATURE_COLUMNS].copy()
    X_valid = valid[FEATURE_COLUMNS].copy()
    X_test = test[FEATURE_COLUMNS].copy()

    cat_cols = []

    for c in FEATURE_COLUMNS:
        if pd.api.types.is_numeric_dtype(X_train[c]):
            median = X_train[c].median()
            X_train[c] = X_train[c].fillna(median)
            X_valid[c] = X_valid[c].fillna(median)
            X_test[c] = X_test[c].fillna(median)
        else:
            cat_cols.append(c)
            mode_vals = X_train[c].mode(dropna=True)
            fill_value = mode_vals.iloc[0] if len(mode_vals) else "MISSING"

            X_train[c] = X_train[c].fillna(fill_value).astype(str)
            X_valid[c] = X_valid[c].fillna(fill_value).astype(str)
            X_test[c] = X_test[c].fillna(fill_value).astype(str)

    cat_indices = [X_train.columns.get_loc(c) for c in cat_cols]

    return X_train, X_valid, X_test, cat_indices


def fit_full_catboost(train, valid, test, seed):
    X_train, X_valid, X_test, cat_indices = fit_catboost_preprocessor(
        train, valid, test
    )

    model = CatBoostClassifier(
        iterations=900,
        depth=6,
        learning_rate=0.03,
        l2_leaf_reg=5,
        subsample=0.85,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False
    )

    model.fit(
        X_train,
        train["y"].astype(int),
        cat_features=cat_indices,
        eval_set=(X_valid, valid["y"].astype(int)),
        use_best_model=True
    )

    valid_prob = model.predict_proba(X_valid)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]

    return valid_prob, test_prob


def make_summary(metrics_df):
    """
    Summarizes rows by model.
    Deterministic models have one row and therefore SD = 0.
    """
    rows = []

    for model_name, g in metrics_df.groupby("model", sort=False):
        row = {
            "model": model_name,
            "n_runs": len(g),
        }

        for metric in ["auc", "accuracy", "f1", "logloss"]:
            row[f"{metric}_mean"] = g[metric].mean()
            row[f"{metric}_sd"] = (
                g[metric].std(ddof=1) if len(g) > 1 else 0.0
            )

        rows.append(row)

    return pd.DataFrame(rows)


def formatted_summary(summary_df):
    out = summary_df[["model", "n_runs"]].copy()

    for metric in ["auc", "accuracy", "f1", "logloss"]:
        out[metric] = summary_df.apply(
            lambda r: f"{r[f'{metric}_mean']:.4f} ± {r[f'{metric}_sd']:.4f}",
            axis=1
        )

    return out


def incremental_gain(summary_test):
    """
    Positive values indicate better Full CatBoost performance.
    For log-loss, positive means baseline log-loss minus CatBoost log-loss.
    """
    full = summary_test[
        summary_test["model"] == "Full CatBoost"
    ].iloc[0]

    rows = []

    for baseline_name in [
        "Elo favorite heuristic",
        "Elo-only Logistic Regression"
    ]:
        base = summary_test[
            summary_test["model"] == baseline_name
        ].iloc[0]

        rows.append({
            "comparison": f"Full CatBoost vs {baseline_name}",
            "delta_auc": full["auc_mean"] - base["auc_mean"],
            "delta_accuracy": full["accuracy_mean"] - base["accuracy_mean"],
            "delta_f1": full["f1_mean"] - base["f1_mean"],
            # Positive value = lower/better CatBoost log-loss
            "logloss_reduction": base["logloss_mean"] - full["logloss_mean"],
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Reviewer 2 trivial-favorite baseline analysis"
    )

    parser.add_argument(
        "--input",
        default="/data2/zhangyiyi/beach_ai/final_catboost_v1/data/prematch_features_v3.csv",
        help="Path to prematch_features_v3.csv"
    )

    parser.add_argument(
        "--outdir",
        default="/data2/zhangyiyi/beach_ai/final_catboost_v1/trivial_favorite_baseline_output",
        help="Output directory"
    )

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 78)
    print("Reviewer 2 - Trivial-favorite baseline analysis")
    print("=" * 78)
    print("Input :", args.input)
    print("Output:", args.outdir)

    df = pd.read_csv(args.input)

    required_cols = set(
        ["date", "y", ELO_DIFF_COL] + FEATURE_COLUMNS
    )
    missing_cols = sorted(required_cols - set(df.columns))

    if missing_cols:
        raise ValueError(
            "Missing required columns:\n" + "\n".join(missing_cols)
        )

    df["y"] = pd.to_numeric(df["y"], errors="raise").astype(int)

    train, valid, test = time_split(df)

    print("\nDataset split:")
    print(f"  Train      : {len(train):,}")
    print(f"  Validation : {len(valid):,}")
    print(f"  Test       : {len(test):,}")

    print("\nOutcome prevalence:")
    print(f"  Train Team A win rate      : {train['y'].mean():.4f}")
    print(f"  Validation Team A win rate : {valid['y'].mean():.4f}")
    print(f"  Test Team A win rate       : {test['y'].mean():.4f}")

    print(f"\nMissing values in {ELO_DIFF_COL}:")
    print(f"  Train      : {train[ELO_DIFF_COL].isna().sum():,}")
    print(f"  Validation : {valid[ELO_DIFF_COL].isna().sum():,}")
    print(f"  Test       : {test[ELO_DIFF_COL].isna().sum():,}")

    train_elo, valid_elo, test_elo, elo_median = prepare_elo_diff(
        train, valid, test
    )

    print(f"  Training median used if needed: {elo_median:.6f}")

    validation_rows = []
    test_rows = []

    # ------------------------------------------------------------------
    # 1) Elo favorite heuristic
    # ------------------------------------------------------------------
    valid_prob_elo = elo_expected_probability(valid_elo)
    test_prob_elo = elo_expected_probability(test_elo)

    valid_metrics_elo = evaluate(valid["y"], valid_prob_elo)
    test_metrics_elo = evaluate(test["y"], test_prob_elo)

    validation_rows.append({
        "model": "Elo favorite heuristic",
        "seed": np.nan,
        **valid_metrics_elo
    })
    test_rows.append({
        "model": "Elo favorite heuristic",
        "seed": np.nan,
        **test_metrics_elo
    })

    print("\nElo favorite heuristic:")
    print("  VALID:", {k: round(v, 6) for k, v in valid_metrics_elo.items()})
    print("  TEST :", {k: round(v, 6) for k, v in test_metrics_elo.items()})

    # ------------------------------------------------------------------
    # 2) Univariate Logistic Regression on Elo difference
    # ------------------------------------------------------------------
    lr_model, valid_prob_lr, test_prob_lr = fit_elo_only_logistic(
        train_elo,
        train["y"].to_numpy(),
        valid_elo,
        test_elo
    )

    valid_metrics_lr = evaluate(valid["y"], valid_prob_lr)
    test_metrics_lr = evaluate(test["y"], test_prob_lr)

    validation_rows.append({
        "model": "Elo-only Logistic Regression",
        "seed": np.nan,
        **valid_metrics_lr
    })
    test_rows.append({
        "model": "Elo-only Logistic Regression",
        "seed": np.nan,
        **test_metrics_lr
    })

    print("\nElo-only Logistic Regression:")
    print("  coefficient :", float(lr_model.coef_[0][0]))
    print("  intercept   :", float(lr_model.intercept_[0]))
    print("  VALID:", {k: round(v, 6) for k, v in valid_metrics_lr.items()})
    print("  TEST :", {k: round(v, 6) for k, v in test_metrics_lr.items()})

    # ------------------------------------------------------------------
    # 3) Full CatBoost baseline - 10 seeds
    # ------------------------------------------------------------------
    print("\nFull CatBoost baseline:")

    for seed in SEEDS:
        valid_prob_cb, test_prob_cb = fit_full_catboost(
            train, valid, test, seed
        )

        valid_metrics_cb = evaluate(valid["y"], valid_prob_cb)
        test_metrics_cb = evaluate(test["y"], test_prob_cb)

        validation_rows.append({
            "model": "Full CatBoost",
            "seed": seed,
            **valid_metrics_cb
        })
        test_rows.append({
            "model": "Full CatBoost",
            "seed": seed,
            **test_metrics_cb
        })

        print(
            f"  Seed {seed:3d} | "
            f"VALID AUC={valid_metrics_cb['auc']:.6f} | "
            f"TEST AUC={test_metrics_cb['auc']:.6f}"
        )

    validation_df = pd.DataFrame(validation_rows)
    test_df = pd.DataFrame(test_rows)

    validation_df.to_csv(
        os.path.join(args.outdir, "validation_metrics_all_runs.csv"),
        index=False
    )
    test_df.to_csv(
        os.path.join(args.outdir, "test_metrics_all_runs.csv"),
        index=False
    )

    summary_valid = make_summary(validation_df)
    summary_test = make_summary(test_df)

    summary_valid.to_csv(
        os.path.join(args.outdir, "comparison_summary_validation.csv"),
        index=False
    )
    summary_test.to_csv(
        os.path.join(args.outdir, "comparison_summary_test.csv"),
        index=False
    )

    formatted_summary(summary_valid).to_csv(
        os.path.join(args.outdir, "comparison_summary_validation_formatted.csv"),
        index=False
    )
    formatted_summary(summary_test).to_csv(
        os.path.join(args.outdir, "comparison_summary_test_formatted.csv"),
        index=False
    )

    gains = incremental_gain(summary_test)
    gains.to_csv(
        os.path.join(args.outdir, "incremental_gain_test.csv"),
        index=False
    )

    config = {
        "input": args.input,
        "outdir": args.outdir,
        "target": "y",
        "elo_diff_feature": ELO_DIFF_COL,
        "elo_probability_formula": "1 / (1 + 10^(-avg_player_elo_diff / 400))",
        "train_period": "<= 2017-12-31",
        "validation_period": "2018-01-01 to 2019-12-31",
        "test_period": "> 2019-12-31",
        "catboost_seeds": SEEDS,
        "catboost_parameters": {
            "iterations": 900,
            "depth": 6,
            "learning_rate": 0.03,
            "l2_leaf_reg": 5,
            "subsample": 0.85,
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "use_best_model": True
        },
        "feature_count_full_catboost": len(FEATURE_COLUMNS),
        "missing_elo_diff": {
            "train": int(train[ELO_DIFF_COL].isna().sum()),
            "validation": int(valid[ELO_DIFF_COL].isna().sum()),
            "test": int(test[ELO_DIFF_COL].isna().sum())
        },
        "elo_diff_training_median": float(elo_median)
    }

    with open(
        os.path.join(args.outdir, "run_config.json"),
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 78)
    print("FINAL TEST SUMMARY")
    print("=" * 78)
    print(formatted_summary(summary_test).to_string(index=False))

    print("\nIncremental gain of Full CatBoost over simple baselines:")
    print(gains.to_string(index=False))

    print("\nSaved files:")
    for name in [
        "test_metrics_all_runs.csv",
        "validation_metrics_all_runs.csv",
        "comparison_summary_test.csv",
        "comparison_summary_test_formatted.csv",
        "comparison_summary_validation.csv",
        "comparison_summary_validation_formatted.csv",
        "incremental_gain_test.csv",
        "run_config.json",
    ]:
        print(" ", os.path.join(args.outdir, name))


if __name__ == "__main__":
    main()
