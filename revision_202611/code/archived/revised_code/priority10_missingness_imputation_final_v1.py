#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Priority 10: Missingness + imputation sensitivity analysis
==========================================================

Reviewer motivation
-------------------
Reviewer 1:
"How many values were missing overall and per feature? Which features had the
highest rates of missingness? ... Reporting missingness and optionally testing
sensitivity to imputation would strengthen the paper."

Reviewer 2:
"A per-feature missingness table and imputation-sensitivity analysis,
particularly for Elo/ranking, where median imputation may distort the dominant
predictor."

Minimal sufficient design
-------------------------
A. Missingness audit
   - Per-feature missing N / % for Overall, Train, Validation, Test
   - Dedicated table for Elo / ranking / qualifier-related predictors
   - Year-wise missingness for these key predictors

B. Imputation sensitivity using retained identity-free CatBoost
   1) Primary: numerical training median + categorical training mode
   2) Sensitivity: numeric NaN retained for CatBoost native handling + explicit
      categorical missing category
   3) Supplementary sensitivity: primary imputation + missingness indicators

C. Same chronological split, features, model settings and 10 seeds.
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
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, log_loss

SEEDS = [41, 51, 61, 71, 81, 91, 101, 111, 121, 131]
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
    "has_time": True,
}

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
    "teamA_avg_player_prior_matches", "teamB_avg_player_prior_matches", "avg_player_prior_matches_diff",
    "teamA_avg_player_prior_win_pct", "teamB_avg_player_prior_win_pct", "avg_player_prior_win_pct_diff",
    "teamA_avg_player_recent10_win_pct", "teamB_avg_player_recent10_win_pct", "avg_player_recent10_win_pct_diff",
    "teamA_avg_player_recent20_win_pct", "teamB_avg_player_recent20_win_pct", "avg_player_recent20_win_pct_diff",
    "teamA_avg_player_last90_match_count", "teamB_avg_player_last90_match_count", "avg_player_last90_match_count_diff",
    "teamA_avg_player_last90_win_pct", "teamB_avg_player_last90_win_pct", "avg_player_last90_win_pct_diff",
    "teamA_avg_player_days_since_last_match", "teamB_avg_player_days_since_last_match", "avg_player_days_since_last_match_diff",
    "teamA_both_players_new", "teamB_both_players_new",
    "h2h_prior_matches", "h2h_teamA_win_pct", "h2h_recent3_teamA_win_pct",
    "teamA_elo", "teamB_elo", "elo_diff",
    "teamA_player1_elo", "teamA_player2_elo", "teamB_player1_elo", "teamB_player2_elo",
    "teamA_avg_player_elo", "teamB_avg_player_elo", "avg_player_elo_diff",
    "teamA_blend_elo", "teamB_blend_elo", "blend_elo_diff",
]

REFERENCE_TEST = {
    "auc": 0.8126,
    "accuracy": 0.7294,
    "f1": 0.7306,
    "logloss": 0.5253,
}


def load_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", low_memory=False)


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    required = ["date", "y"] + FEATURE_COLUMNS
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError("Required columns are missing:\n" + "\n".join(missing))

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if out["date"].isna().any():
        raise ValueError(f"Invalid/missing date values: {int(out['date'].isna().sum())}")

    out["y"] = pd.to_numeric(out["y"], errors="raise").astype(int)
    if not set(out["y"].unique()).issubset({0, 1}):
        raise ValueError("Target y is not binary 0/1.")

    sort_cols = ["date"]
    if "match_num" in out.columns:
        sort_cols.append("match_num")
    out = out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    out["year_for_audit"] = out["date"].dt.year.astype(int)
    return out


def split_by_time(df: pd.DataFrame):
    train = df[df["date"] <= pd.Timestamp("2017-12-31")].copy().reset_index(drop=True)
    valid = df[(df["date"] >= pd.Timestamp("2018-01-01")) & (df["date"] <= pd.Timestamp("2019-12-31"))].copy().reset_index(drop=True)
    test = df[df["date"] >= pd.Timestamp("2020-01-01")].copy().reset_index(drop=True)
    return train, valid, test


def feature_block(feature: str) -> str:
    f = feature.lower()
    if "elo" in f:
        return "Elo"
    if "rank" in f:
        return "Ranking"
    if "qualifier" in f:
        return "Qualifier"
    if feature in {"gender", "circuit", "country", "bracket", "round", "month"}:
        return "Context"
    if "age" in f or "hgt" in f:
        return "Demographic/anthropometric"
    if feature.startswith("h2h_"):
        return "Head-to-head"
    return "Historical/recent activity"


def is_key_predictor(feature: str) -> bool:
    f = feature.lower()
    return ("elo" in f) or ("rank" in f) or ("qualifier" in f)


def missing_stats(df: pd.DataFrame, feature: str, prefix: str) -> Dict[str, float]:
    n = len(df)
    miss_n = int(df[feature].isna().sum())
    return {
        f"{prefix}_n": n,
        f"{prefix}_missing_n": miss_n,
        f"{prefix}_missing_pct": (100.0 * miss_n / n) if n else np.nan,
    }


def build_missingness_table(overall, train, valid, test):
    rows = []
    for f in FEATURE_COLUMNS:
        row = {"feature": f, "feature_block": feature_block(f), "dtype": str(overall[f].dtype)}
        row.update(missing_stats(overall, f, "overall"))
        row.update(missing_stats(train, f, "train"))
        row.update(missing_stats(valid, f, "validation"))
        row.update(missing_stats(test, f, "test"))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["overall_missing_pct", "train_missing_pct"], ascending=[False, False]
    ).reset_index(drop=True)


def build_key_missingness_by_year(df: pd.DataFrame) -> pd.DataFrame:
    key_features = [f for f in FEATURE_COLUMNS if is_key_predictor(f)]
    rows = []
    for year, g in df.groupby("year_for_audit", sort=True):
        for f in key_features:
            n = len(g)
            miss_n = int(g[f].isna().sum())
            rows.append({
                "year": int(year),
                "feature": f,
                "feature_block": feature_block(f),
                "n_matches": n,
                "missing_n": miss_n,
                "missing_pct": 100.0 * miss_n / n if n else np.nan,
            })
    return pd.DataFrame(rows)


def save_missingness_plot(missing_tab: pd.DataFrame, outdir: str, top_n: int = 20):
    top = missing_tab.head(top_n).iloc[::-1]
    plt.figure(figsize=(9, 7))
    plt.barh(top["feature"], top["overall_missing_pct"])
    plt.xlabel("Missing values (%)")
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} features by overall missingness")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig_missingness_top20.png"), dpi=300)
    plt.savefig(os.path.join(outdir, "fig_missingness_top20.pdf"))
    plt.close()


def infer_feature_types(train: pd.DataFrame, features: List[str]) -> Tuple[List[str], List[str]]:
    numeric, categorical = [], []
    for c in features:
        if pd.api.types.is_numeric_dtype(train[c]):
            numeric.append(c)
        else:
            categorical.append(c)
    return numeric, categorical


def fit_baseline_imputer(train: pd.DataFrame, features: List[str]):
    numeric, categorical = infer_feature_types(train, features)
    medians, modes = {}, {}
    for c in numeric:
        med = pd.to_numeric(train[c], errors="coerce").median()
        medians[c] = 0.0 if pd.isna(med) else float(med)
    for c in categorical:
        m = train[c].mode(dropna=True)
        modes[c] = str(m.iloc[0]) if len(m) else "__MISSING__"
    return numeric, categorical, medians, modes


def transform_baseline(df, features, numeric, categorical, medians, modes):
    X = df[features].copy()
    for c in numeric:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(medians[c])
    for c in categorical:
        X[c] = X[c].fillna(modes[c]).astype(str)
    return X


def transform_native_numeric_missing(df, features, numeric, categorical):
    X = df[features].copy()
    for c in numeric:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in categorical:
        X[c] = X[c].fillna("__MISSING__").astype(str)
    return X


def fit_missing_indicator_features(train: pd.DataFrame, features: List[str]) -> List[str]:
    return [c for c in features if train[c].isna().any()]


def transform_with_missing_indicators(df, features, numeric, categorical, medians, modes, indicator_features):
    X = transform_baseline(df, features, numeric, categorical, medians, modes)
    for c in indicator_features:
        X[f"{c}__missing"] = df[c].isna().astype(int).to_numpy()
    return X


def metrics(y_true, p):
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    pred = (p >= DECISION_THRESHOLD).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, p)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred)),
        "logloss": float(log_loss(y_true, p)),
    }


def prepare_strategy_data(strategy, train, valid, test):
    features = FEATURE_COLUMNS.copy()
    numeric, categorical, medians, modes = fit_baseline_imputer(train, features)

    if strategy == "baseline_median_mode":
        Xtr = transform_baseline(train, features, numeric, categorical, medians, modes)
        Xva = transform_baseline(valid, features, numeric, categorical, medians, modes)
        Xte = transform_baseline(test, features, numeric, categorical, medians, modes)

    elif strategy == "native_numeric_missing":
        Xtr = transform_native_numeric_missing(train, features, numeric, categorical)
        Xva = transform_native_numeric_missing(valid, features, numeric, categorical)
        Xte = transform_native_numeric_missing(test, features, numeric, categorical)

    elif strategy == "median_mode_plus_missing_indicators":
        indicator_features = fit_missing_indicator_features(train, features)
        Xtr = transform_with_missing_indicators(train, features, numeric, categorical, medians, modes, indicator_features)
        Xva = transform_with_missing_indicators(valid, features, numeric, categorical, medians, modes, indicator_features)
        Xte = transform_with_missing_indicators(test, features, numeric, categorical, medians, modes, indicator_features)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    cat_idx = [Xtr.columns.get_loc(c) for c in categorical]
    return Xtr, Xva, Xte, cat_idx


def run_strategy(strategy, train, valid, test, seeds):
    Xtr, Xva, Xte, cat_idx = prepare_strategy_data(strategy, train, valid, test)
    ytr = train["y"].astype(int).to_numpy()
    yva = valid["y"].astype(int).to_numpy()
    yte = test["y"].astype(int).to_numpy()

    rows, valid_probs, test_probs = [], [], []

    for seed in seeds:
        model = CatBoostClassifier(**CATBOOST_PARAMS, random_seed=seed)
        model.fit(Xtr, ytr, cat_features=cat_idx, eval_set=(Xva, yva), use_best_model=True, verbose=False)

        pva = model.predict_proba(Xva)[:, 1]
        pte = model.predict_proba(Xte)[:, 1]
        mva, mte = metrics(yva, pva), metrics(yte, pte)

        rows.append({
            "strategy": strategy,
            "seed": seed,
            "n_features_model": Xtr.shape[1],
            "best_iteration": int(model.get_best_iteration()),
            **{f"validation_{k}": v for k, v in mva.items()},
            **{f"test_{k}": v for k, v in mte.items()},
        })
        valid_probs.append(pva)
        test_probs.append(pte)

        print(f"[{strategy}] seed={seed} TEST AUC={mte['auc']:.4f} Acc={mte['accuracy']:.4f} F1={mte['f1']:.4f} LogLoss={mte['logloss']:.4f}")

    return pd.DataFrame(rows), yva, np.mean(np.vstack(valid_probs), axis=0), yte, np.mean(np.vstack(test_probs), axis=0)


def summarize_runs(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    order = ["baseline_median_mode", "native_numeric_missing", "median_mode_plus_missing_indicators"]
    for strategy in order:
        g = runs[runs["strategy"] == strategy]
        if len(g) == 0:
            continue
        row = {
            "strategy": strategy,
            "n_runs": len(g),
            "n_features_model": int(g["n_features_model"].iloc[0]),
            "best_iteration_mean": g["best_iteration"].mean(),
            "best_iteration_sd": g["best_iteration"].std(ddof=1),
        }
        for split in ["validation", "test"]:
            for metric in ["auc", "accuracy", "f1", "logloss"]:
                col = f"{split}_{metric}"
                row[f"{col}_mean"] = g[col].mean()
                row[f"{col}_sd"] = g[col].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def formatted_supplementary_table(summary: pd.DataFrame) -> pd.DataFrame:
    labels = {
        "baseline_median_mode": "Primary: training median / mode",
        "native_numeric_missing": "Sensitivity: native numeric missing handling",
        "median_mode_plus_missing_indicators": "Sensitivity: median / mode + missing indicators",
    }
    rows = []
    for _, r in summary.iterrows():
        rows.append({
            "Missing-data strategy": labels.get(r["strategy"], r["strategy"]),
            "Model features": int(r["n_features_model"]),
            "AUC": f"{r['test_auc_mean']:.4f} ± {r['test_auc_sd']:.4f}",
            "Accuracy": f"{r['test_accuracy_mean']:.4f} ± {r['test_accuracy_sd']:.4f}",
            "F1-score": f"{r['test_f1_mean']:.4f} ± {r['test_f1_sd']:.4f}",
            "Log-loss": f"{r['test_logloss_mean']:.4f} ± {r['test_logloss_sd']:.4f}",
        })
    return pd.DataFrame(rows)


def build_delta_vs_baseline(runs: pd.DataFrame) -> pd.DataFrame:
    base = runs[runs["strategy"] == "baseline_median_mode"].sort_values("seed").reset_index(drop=True)
    rows = []
    for strategy in ["native_numeric_missing", "median_mode_plus_missing_indicators"]:
        alt = runs[runs["strategy"] == strategy].sort_values("seed").reset_index(drop=True)
        if len(alt) == 0:
            continue
        if not np.array_equal(base["seed"].to_numpy(), alt["seed"].to_numpy()):
            raise RuntimeError(f"Seed alignment failed for {strategy}")
        row = {"comparison": f"{strategy} minus baseline_median_mode"}
        for split in ["validation", "test"]:
            for metric in ["auc", "accuracy", "f1", "logloss"]:
                col = f"{split}_{metric}"
                d = alt[col].to_numpy() - base[col].to_numpy()
                row[f"delta_{split}_{metric}_mean"] = float(d.mean())
                row[f"delta_{split}_{metric}_sd"] = float(d.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def save_imputation_auc_plot(summary: pd.DataFrame, outdir: str):
    labels = {
        "baseline_median_mode": "Median/mode",
        "native_numeric_missing": "Native numeric NaN",
        "median_mode_plus_missing_indicators": "Median/mode + indicators",
    }
    x = np.arange(len(summary))
    auc_mean = summary["test_auc_mean"].to_numpy()
    auc_sd = summary["test_auc_sd"].to_numpy()
    plt.figure(figsize=(8, 5.5))
    plt.errorbar(x, auc_mean, yerr=auc_sd, fmt="o", capsize=4, linewidth=1.5)
    plt.xticks(x, [labels.get(s, s) for s in summary["strategy"]], rotation=12, ha="right")
    plt.ylabel("Test AUC")
    plt.xlabel("Missing-data strategy")
    plt.title("Imputation sensitivity of the identity-free CatBoost model")
    lo = max(0.5, float(np.nanmin(auc_mean) - 0.02))
    hi = min(1.0, float(np.nanmax(auc_mean) + 0.02))
    if hi - lo < 0.04:
        center = float(np.nanmean(auc_mean))
        lo = max(0.5, center - 0.025)
        hi = min(1.0, center + 0.025)
    plt.ylim(lo, hi)
    for i, val in enumerate(auc_mean):
        plt.text(i, val + 0.002, f"{val:.4f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig_imputation_sensitivity_test_auc.png"), dpi=300)
    plt.savefig(os.path.join(outdir, "fig_imputation_sensitivity_test_auc.pdf"))
    plt.close()


def dataset_audit(df: pd.DataFrame) -> Dict:
    return {
        "n_rows": int(len(df)),
        "n_model_features": int(len(FEATURE_COLUMNS)),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "year_min": int(df["date"].dt.year.min()),
        "year_max": int(df["date"].dt.year.max()),
        "target_prevalence": float(df["y"].mean()),
        "expected_rows_85509": bool(len(df) == 85509),
        "expected_year_range_2000_2022": bool(df["date"].dt.year.min() == 2000 and df["date"].dt.year.max() == 2022),
        "all_required_features_present": True,
    }


def reproduction_check(summary: pd.DataFrame) -> pd.DataFrame:
    b = summary[summary["strategy"] == "baseline_median_mode"].iloc[0]
    rows = []
    for metric, ref in REFERENCE_TEST.items():
        observed = float(b[f"test_{metric}_mean"])
        rows.append({
            "metric": metric,
            "manuscript_reference": ref,
            "recomputed_baseline": observed,
            "difference": observed - ref,
            "absolute_difference": abs(observed - ref),
            "within_0.005": abs(observed - ref) <= 0.005,
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Priority 10: missingness and imputation sensitivity analysis")
    parser.add_argument("--input", default="./legacy_project/final_catboost_v1/data/prematch_features_v3.csv")
    parser.add_argument("--outdir", default="./legacy_project/final_catboost_v1/missingness_imputation_output")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--skip-third-strategy", action="store_true", help="Run only primary and native-missing sensitivity strategies.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 88)
    print("PRIORITY 10: MISSINGNESS + IMPUTATION SENSITIVITY")
    print("=" * 88)
    print("Input :", args.input)
    print("Output:", args.outdir)
    print("Seeds :", args.seeds)

    df = prepare_data(load_csv(args.input))
    train, valid, test = split_by_time(df)

    audit = dataset_audit(df)
    audit.update({
        "n_train": int(len(train)),
        "n_validation": int(len(valid)),
        "n_test": int(len(test)),
        "train_date_min": str(train["date"].min().date()),
        "train_date_max": str(train["date"].max().date()),
        "validation_date_min": str(valid["date"].min().date()),
        "validation_date_max": str(valid["date"].max().date()),
        "test_date_min": str(test["date"].min().date()),
        "test_date_max": str(test["date"].max().date()),
    })
    with open(os.path.join(args.outdir, "dataset_integrity_audit.json"), "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)

    print("\nDATASET AUDIT")
    print(f"Rows            : {len(df):,}")
    print(f"Model features  : {len(FEATURE_COLUMNS)}")
    print(f"Date range      : {audit['date_min']} to {audit['date_max']}")
    print(f"Train           : {len(train):,}")
    print(f"Validation      : {len(valid):,}")
    print(f"Test            : {len(test):,}")
    print(f"Team A win rate : {df['y'].mean():.4f}")

    missing_tab = build_missingness_table(df, train, valid, test)
    missing_tab.to_csv(os.path.join(args.outdir, "supplementary_table_missingness_all_features.csv"), index=False)
    missing_tab.head(20).to_csv(os.path.join(args.outdir, "supplementary_table_missingness_top20.csv"), index=False)

    key_tab = missing_tab[missing_tab["feature"].map(is_key_predictor)].copy()
    key_tab.to_csv(os.path.join(args.outdir, "key_predictor_missingness_elo_rank_qualifier.csv"), index=False)

    yearly_key = build_key_missingness_by_year(df)
    yearly_key.to_csv(os.path.join(args.outdir, "key_predictor_missingness_by_year.csv"), index=False)

    block_summary = missing_tab.groupby("feature_block", as_index=False).agg(
        n_features=("feature", "count"),
        mean_overall_missing_pct=("overall_missing_pct", "mean"),
        median_overall_missing_pct=("overall_missing_pct", "median"),
        max_overall_missing_pct=("overall_missing_pct", "max"),
    ).sort_values("mean_overall_missing_pct", ascending=False)
    block_summary.to_csv(os.path.join(args.outdir, "missingness_summary_by_feature_block.csv"), index=False)

    save_missingness_plot(missing_tab, args.outdir, top_n=20)

    print("\nTOP 20 FEATURES BY MISSINGNESS")
    print(missing_tab[["feature", "feature_block", "overall_missing_n", "overall_missing_pct", "train_missing_pct", "validation_missing_pct", "test_missing_pct"]].head(20).to_string(index=False))

    print("\nKEY ELO / RANKING / QUALIFIER MISSINGNESS")
    print(key_tab[["feature", "feature_block", "overall_missing_pct", "train_missing_pct", "validation_missing_pct", "test_missing_pct"]].to_string(index=False))

    strategies = ["baseline_median_mode", "native_numeric_missing"]
    if not args.skip_third_strategy:
        strategies.append("median_mode_plus_missing_indicators")

    all_runs = []
    prediction_store = {}

    for strategy in strategies:
        print("\n" + "=" * 88)
        print("RUNNING:", strategy)
        print("=" * 88)
        runs, yva, pva_mean, yte, pte_mean = run_strategy(strategy, train, valid, test, args.seeds)
        all_runs.append(runs)
        prediction_store[strategy] = {"y_validation": yva, "p_validation": pva_mean, "y_test": yte, "p_test": pte_mean}

    runs_all = pd.concat(all_runs, ignore_index=True)
    runs_all.to_csv(os.path.join(args.outdir, "imputation_sensitivity_metrics_all_runs.csv"), index=False)

    summary = summarize_runs(runs_all)
    summary.to_csv(os.path.join(args.outdir, "imputation_sensitivity_summary_mean_sd.csv"), index=False)

    formatted = formatted_supplementary_table(summary)
    formatted.to_csv(os.path.join(args.outdir, "supplementary_table_imputation_sensitivity.csv"), index=False)

    delta = build_delta_vs_baseline(runs_all)
    delta.to_csv(os.path.join(args.outdir, "imputation_sensitivity_delta_vs_baseline.csv"), index=False)

    save_imputation_auc_plot(summary, args.outdir)

    pred_df = pd.DataFrame({"date": test["date"].astype(str), "y": test["y"].to_numpy()})
    for strategy in strategies:
        pred_df[f"{strategy}_probability_mean_10seeds"] = prediction_store[strategy]["p_test"]
    pred_df.to_csv(os.path.join(args.outdir, "imputation_sensitivity_test_predictions_mean_10seeds.csv"), index=False)

    repro = reproduction_check(summary)
    repro.to_csv(os.path.join(args.outdir, "baseline_reproduction_check.csv"), index=False)

    config = {
        "input": args.input,
        "outdir": args.outdir,
        "n_rows": len(df),
        "n_features_primary": len(FEATURE_COLUMNS),
        "split": {"train": "<= 2017-12-31", "validation": "2018-01-01 to 2019-12-31", "test": ">= 2020-01-01"},
        "seeds": args.seeds,
        "decision_threshold": DECISION_THRESHOLD,
        "catboost_parameters": CATBOOST_PARAMS,
        "strategies": strategies,
        "strategy_descriptions": {
            "baseline_median_mode": "Numeric training median; categorical training mode.",
            "native_numeric_missing": "Numeric NaN retained for CatBoost native handling; categorical missing encoded as __MISSING__.",
            "median_mode_plus_missing_indicators": "Primary median/mode imputation plus binary indicators for features missing in training.",
        },
    }
    with open(os.path.join(args.outdir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    readme = f"""
Priority 10: Missingness + imputation sensitivity analysis

Input:
{args.input}

Main missingness outputs:
- supplementary_table_missingness_all_features.csv
- supplementary_table_missingness_top20.csv
- key_predictor_missingness_elo_rank_qualifier.csv
- key_predictor_missingness_by_year.csv
- missingness_summary_by_feature_block.csv
- fig_missingness_top20.png/pdf

Main imputation-sensitivity outputs:
- imputation_sensitivity_metrics_all_runs.csv
- imputation_sensitivity_summary_mean_sd.csv
- supplementary_table_imputation_sensitivity.csv
- imputation_sensitivity_delta_vs_baseline.csv
- fig_imputation_sensitivity_test_auc.png/pdf
- imputation_sensitivity_test_predictions_mean_10seeds.csv

Reproducibility:
- dataset_integrity_audit.json
- baseline_reproduction_check.csv

Interpretation:
If performance remains close between the primary median/mode strategy and the
native-numeric-missing strategy, the main conclusion is not materially dependent
on median imputation. The third strategy tests whether explicitly representing
missingness changes this conclusion.
""".strip()
    with open(os.path.join(args.outdir, "README_priority10_missingness_imputation.txt"), "w", encoding="utf-8") as f:
        f.write(readme + "\n")

    print("\n" + "=" * 88)
    print("FINAL IMPUTATION SENSITIVITY SUMMARY")
    print("=" * 88)
    print(formatted.to_string(index=False))

    print("\nDELTA VS PRIMARY BASELINE")
    print(delta.to_string(index=False))

    print("\nBASELINE REPRODUCTION CHECK")
    print(repro.to_string(index=False))

    print("\nSaved all outputs to:")
    print(args.outdir)


if __name__ == "__main__":
    main()
