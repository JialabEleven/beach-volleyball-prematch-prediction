#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Priority 5: standardized time-aware hyperparameter tuning

Purpose
-------
Address Reviewer 2's concern that model comparisons may be unbalanced if
some algorithms were tuned more extensively than others.

Design
------
- Hyperparameter selection uses ONLY matches dated <= 2017-12-31.
- Three expanding-window internal validation folds:
    1) train <= 2011, validate 2012-2013
    2) train <= 2013, validate 2014-2015
    3) train <= 2015, validate 2016-2017
- Selection criterion for every model: highest mean internal-validation AUC.
- External validation (2018-2019) and test (>2019) are never used for tuning.
- Logistic Regression: exhaustive search over C.
- Random Forest / XGBoost / CatBoost: reproducible random search with the
  same budget (default 30 configurations/model).
- After tuning, retrain each selected model on the complete <=2017 training
  set and evaluate on 2018-2019 validation and >2019 test across 10 seeds.

Main outputs
------------
hyperparameter_search_space.csv
tuning_fold_definition.csv
tuning_results_all_folds.csv
tuning_results_all_configurations.csv
selected_hyperparameters.csv
final_tuned_metrics_all_runs.csv
final_tuned_metrics_summary_mean_sd.csv
final_tuned_metrics_summary_formatted.csv
final_tuned_test_predictions_mean_10seeds.csv
comparison_original_vs_tuned_test.csv
ranking_original_vs_tuned.csv
run_config.json
README_priority5.txt
"""

import os
import json
import math
import argparse
import warnings
from itertools import product

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, log_loss
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

FINAL_SEEDS = [41, 51, 61, 71, 81, 91, 101, 111, 121, 131]
TUNING_SEED = 42

TUNING_FOLDS = [
    {"fold": 1, "train_end": "2011-12-31", "valid_start": "2012-01-01", "valid_end": "2013-12-31"},
    {"fold": 2, "train_end": "2013-12-31", "valid_start": "2014-01-01", "valid_end": "2015-12-31"},
    {"fold": 3, "train_end": "2015-12-31", "valid_start": "2016-01-01", "valid_end": "2017-12-31"},
]

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

ORIGINAL_TABLE1_TEST = {
    "Logistic Regression": {"auc": 0.7607, "accuracy": 0.6907, "f1": 0.6891, "logloss": 0.5809},
    "Random Forest":       {"auc": 0.7872, "accuracy": 0.7138, "f1": 0.7160, "logloss": 0.5588},
    "XGBoost":             {"auc": 0.8068, "accuracy": 0.7236, "f1": 0.7249, "logloss": 0.5320},
    "CatBoost":            {"auc": 0.8126, "accuracy": 0.7294, "f1": 0.7306, "logloss": 0.5253},
}

LR_SPACE = {"C": [0.01, 0.1, 1.0, 10.0, 100.0]}
RF_SPACE = {
    "n_estimators": [300, 500, 800],
    "max_depth": [None, 10, 20],
    "min_samples_leaf": [1, 2, 5],
    "max_features": ["sqrt", 0.5, 1.0],
}
XGB_SPACE = {
    "n_estimators": [400, 800, 1200],
    "max_depth": [3, 4, 6],
    "learning_rate": [0.02, 0.03, 0.05],
    "subsample": [0.80, 0.90, 1.00],
    "colsample_bytree": [0.80, 0.90, 1.00],
    "min_child_weight": [1, 3, 5],
    "reg_lambda": [1.0, 2.0, 5.0],
}
CAT_SPACE = {
    "iterations": [500, 900, 1300],
    "depth": [4, 6, 8],
    "learning_rate": [0.02, 0.03, 0.05],
    "l2_leaf_reg": [3.0, 5.0, 10.0],
    "subsample": [0.80, 0.90, 1.00],
}


def load_data(path):
    try:
        df = pd.read_csv(path, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1", low_memory=False)

    if "date" not in df or "y" not in df:
        raise ValueError("Input must contain 'date' and target column 'y'.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise ValueError(f"Invalid dates: {int(df['date'].isna().sum())}")

    df["y"] = pd.to_numeric(df["y"], errors="raise").astype(int)

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Missing required features:\n" + "\n".join(missing))

    sort_cols = ["date"] + (["match_num"] if "match_num" in df.columns else [])
    return df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)


def external_split(df, train_end="2017-12-31", valid_end="2019-12-31"):
    train_end, valid_end = pd.Timestamp(train_end), pd.Timestamp(valid_end)
    train = df[df["date"] <= train_end].copy().reset_index(drop=True)
    valid = df[(df["date"] > train_end) & (df["date"] <= valid_end)].copy().reset_index(drop=True)
    test = df[df["date"] > valid_end].copy().reset_index(drop=True)
    return train, valid, test


def make_fold(df, fold):
    tr_end = pd.Timestamp(fold["train_end"])
    va_start = pd.Timestamp(fold["valid_start"])
    va_end = pd.Timestamp(fold["valid_end"])
    tr = df[df["date"] <= tr_end].copy().reset_index(drop=True)
    va = df[(df["date"] >= va_start) & (df["date"] <= va_end)].copy().reset_index(drop=True)
    return tr, va


def make_xy(df):
    return df[FEATURE_COLUMNS].copy(), df["y"].astype(int).to_numpy()


def num_cat_cols(X):
    num = [c for c in X.columns if is_numeric_dtype(X[c])]
    cat = [c for c in X.columns if c not in num]
    return num, cat


def make_preprocessor(X):
    num, cat = num_cat_cols(X)
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", numeric, num), ("cat", categorical, cat)])


def prep_catboost(Xtr, Xva, Xte=None):
    num, cat = num_cat_cols(Xtr)
    a, b = Xtr.copy(), Xva.copy()
    c = Xte.copy() if Xte is not None else None
    med = a[num].median(numeric_only=True)
    a[num] = a[num].fillna(med); b[num] = b[num].fillna(med)
    if c is not None: c[num] = c[num].fillna(med)
    for col in cat:
        mode = a[col].mode(dropna=True)
        fill = mode.iloc[0] if len(mode) else "missing"
        a[col] = a[col].fillna(fill).astype(str)
        b[col] = b[col].fillna(fill).astype(str)
        if c is not None: c[col] = c[col].fillna(fill).astype(str)
    cat_idx = [a.columns.get_loc(x) for x in cat]
    return a, b, c, cat_idx


def metric_dict(y, p):
    p = np.clip(np.asarray(p, float), 1e-12, 1-1e-12)
    pred = (p >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y, p)),
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        "logloss": float(log_loss(y, p)),
    }


def build_model(name, Xtr, params, seed):
    if name == "Logistic Regression":
        return Pipeline([
            ("preprocessor", make_preprocessor(Xtr)),
            ("model", LogisticRegression(C=float(params["C"]), max_iter=1200, random_state=seed)),
        ])
    if name == "Random Forest":
        return Pipeline([
            ("preprocessor", make_preprocessor(Xtr)),
            ("model", RandomForestClassifier(
                n_estimators=int(params["n_estimators"]),
                max_depth=params["max_depth"],
                min_samples_leaf=int(params["min_samples_leaf"]),
                max_features=params["max_features"],
                random_state=seed, n_jobs=-1)),
        ])
    if name == "XGBoost":
        return Pipeline([
            ("preprocessor", make_preprocessor(Xtr)),
            ("model", XGBClassifier(
                n_estimators=int(params["n_estimators"]),
                max_depth=int(params["max_depth"]),
                learning_rate=float(params["learning_rate"]),
                subsample=float(params["subsample"]),
                colsample_bytree=float(params["colsample_bytree"]),
                min_child_weight=float(params["min_child_weight"]),
                reg_alpha=0.0,
                reg_lambda=float(params["reg_lambda"]),
                eval_metric="logloss", random_state=seed, n_jobs=-1)),
        ])
    raise ValueError(name)


def build_catboost(params, seed):
    return CatBoostClassifier(
        iterations=int(params["iterations"]),
        depth=int(params["depth"]),
        learning_rate=float(params["learning_rate"]),
        l2_leaf_reg=float(params["l2_leaf_reg"]),
        subsample=float(params["subsample"]),
        loss_function="Logloss", eval_metric="AUC",
        random_seed=seed, allow_writing_files=False,
        verbose=False, thread_count=-1, use_best_model=False,
    )


def config_list(space, n_search, seed, exhaustive=False):
    if exhaustive:
        keys = list(space)
        return [dict(zip(keys, vals)) for vals in product(*[space[k] for k in keys])]
    total = math.prod(len(v) for v in space.values())
    return list(ParameterSampler(space, n_iter=min(n_search, total), random_state=seed))


def eval_config_fold(name, params, tr_df, va_df, seed):
    Xtr, ytr = make_xy(tr_df); Xva, yva = make_xy(va_df)
    if name == "CatBoost":
        a, b, _, cat_idx = prep_catboost(Xtr, Xva)
        model = build_catboost(params, seed)
        model.fit(a, ytr, cat_features=cat_idx, verbose=False)
        p = model.predict_proba(b)[:, 1]
    else:
        model = build_model(name, Xtr, params, seed)
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xva)[:, 1]
    m = metric_dict(yva, p)
    return {**m, "n_train": len(tr_df), "n_valid": len(va_df)}


def tune_model(name, configs, train_era, seed):
    rows = []
    print("\n" + "="*88 + f"\nTUNING {name}: {len(configs)} configurations\n" + "="*88)
    for cid, params in enumerate(configs, 1):
        aucs = []
        print(f"\n{name} config {cid}/{len(configs)}: {params}")
        for fold in TUNING_FOLDS:
            tr, va = make_fold(train_era, fold)
            r = eval_config_fold(name, params, tr, va, seed)
            aucs.append(r["auc"])
            rows.append({
                "model": name, "config_id": cid,
                "params_json": json.dumps(params, sort_keys=True),
                "fold": fold["fold"], "train_end": fold["train_end"],
                "valid_start": fold["valid_start"], "valid_end": fold["valid_end"],
                **r,
            })
            print(f"  fold {fold['fold']}: AUC={r['auc']:.5f}")
        print(f"  mean internal AUC={np.mean(aucs):.6f}")

    detail = pd.DataFrame(rows)
    summary = (detail.groupby(["model", "config_id", "params_json"], as_index=False)
               .agg(mean_internal_auc=("auc", "mean"), sd_internal_auc=("auc", "std"),
                    mean_internal_accuracy=("accuracy", "mean"),
                    mean_internal_f1=("f1", "mean"),
                    mean_internal_logloss=("logloss", "mean"))
               .sort_values(["mean_internal_auc", "mean_internal_logloss"], ascending=[False, True])
               .reset_index(drop=True))
    summary["rank_by_auc"] = np.arange(1, len(summary)+1)
    best = summary.iloc[0]
    return detail, summary, json.loads(best["params_json"])


def final_eval_seed(name, params, tr_df, va_df, te_df, seed):
    Xtr, ytr = make_xy(tr_df); Xva, yva = make_xy(va_df); Xte, yte = make_xy(te_df)
    if name == "CatBoost":
        a, b, c, cat_idx = prep_catboost(Xtr, Xva, Xte)
        model = build_catboost(params, seed)
        model.fit(a, ytr, cat_features=cat_idx, verbose=False)
        pva = model.predict_proba(b)[:,1]; pte = model.predict_proba(c)[:,1]
    else:
        model = build_model(name, Xtr, params, seed)
        model.fit(Xtr, ytr)
        pva = model.predict_proba(Xva)[:,1]; pte = model.predict_proba(Xte)[:,1]
    return metric_dict(yva, pva), metric_dict(yte, pte), pte


def fmt(m, s): return f"{m:.4f} ± {s:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/data2/zhangyiyi/beach_ai/final_catboost_v1/data/prematch_features_v3.csv")
    ap.add_argument("--outdir", default="/data2/zhangyiyi/beach_ai/final_catboost_v1/hyperparameter_tuning_output")
    ap.add_argument("--train-end", default="2017-12-31")
    ap.add_argument("--valid-end", default="2019-12-31")
    ap.add_argument("--n-search", type=int, default=30)
    ap.add_argument("--tuning-seed", type=int, default=TUNING_SEED)
    ap.add_argument("--final-seeds", type=int, nargs="+", default=FINAL_SEEDS)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = load_data(args.input)
    train, valid, test = external_split(df, args.train_end, args.valid_end)
    print(f"Train={len(train):,}, Valid={len(valid):,}, Test={len(test):,}")

    # Fold audit
    fold_rows = []
    for fold in TUNING_FOLDS:
        tr, va = make_fold(train, fold)
        if len(tr)==0 or len(va)==0: raise RuntimeError(f"Empty tuning fold: {fold}")
        if tr["date"].max() >= va["date"].min(): raise RuntimeError(f"Temporal overlap in fold {fold['fold']}")
        fold_rows.append({**fold, "n_train": len(tr), "n_valid": len(va),
                          "train_prevalence": tr["y"].mean(), "valid_prevalence": va["y"].mean()})
    pd.DataFrame(fold_rows).to_csv(os.path.join(args.outdir, "tuning_fold_definition.csv"), index=False)

    spaces = {"Logistic Regression": LR_SPACE, "Random Forest": RF_SPACE, "XGBoost": XGB_SPACE, "CatBoost": CAT_SPACE}
    ss_rows = []
    for model, space in spaces.items():
        for hp, vals in space.items():
            ss_rows.append({"model": model, "hyperparameter": hp, "candidate_values": json.dumps(vals)})
    pd.DataFrame(ss_rows).to_csv(os.path.join(args.outdir, "hyperparameter_search_space.csv"), index=False)

    configs = {
        "Logistic Regression": config_list(LR_SPACE, args.n_search, args.tuning_seed, exhaustive=True),
        "Random Forest": config_list(RF_SPACE, args.n_search, args.tuning_seed+1),
        "XGBoost": config_list(XGB_SPACE, args.n_search, args.tuning_seed+2),
        "CatBoost": config_list(CAT_SPACE, args.n_search, args.tuning_seed+3),
    }

    details, summaries, selected = [], [], {}
    for name in ["Logistic Regression", "Random Forest", "XGBoost", "CatBoost"]:
        d, s, best = tune_model(name, configs[name], train, args.tuning_seed)
        details.append(d); summaries.append(s); selected[name] = best

    detail_df = pd.concat(details, ignore_index=True)
    summary_tune_df = pd.concat(summaries, ignore_index=True)
    detail_df.to_csv(os.path.join(args.outdir, "tuning_results_all_folds.csv"), index=False)
    summary_tune_df.to_csv(os.path.join(args.outdir, "tuning_results_all_configurations.csv"), index=False)

    selected_rows = []
    for name, p in selected.items():
        best = summary_tune_df[summary_tune_df["model"]==name].sort_values("mean_internal_auc", ascending=False).iloc[0]
        selected_rows.append({
            "model": name, "selected_params_json": json.dumps(p, sort_keys=True),
            "mean_internal_validation_auc": best["mean_internal_auc"],
            "sd_internal_validation_auc": best["sd_internal_auc"],
            "selection_criterion": "highest mean AUC across 3 expanding-window folds",
        })
    selected_df = pd.DataFrame(selected_rows)
    selected_df.to_csv(os.path.join(args.outdir, "selected_hyperparameters.csv"), index=False)
    print("\nSELECTED HYPERPARAMETERS\n", selected_df.to_string(index=False))

    # Final evaluation
    run_rows, prob_store = [], {k: [] for k in selected}
    for name, params in selected.items():
        print("\n" + "="*88 + f"\nFINAL {name}: {params}\n" + "="*88)
        for seed in args.final_seeds:
            vm, tm, pte = final_eval_seed(name, params, train, valid, test, seed)
            run_rows.append({"model": name, "seed": seed,
                             **{f"valid_{k}":v for k,v in vm.items()},
                             **{f"test_{k}":v for k,v in tm.items()}})
            prob_store[name].append(pte)
            print(f"seed={seed}: valid AUC={vm['auc']:.5f}, test AUC={tm['auc']:.5f}")

    runs = pd.DataFrame(run_rows)
    runs.to_csv(os.path.join(args.outdir, "final_tuned_metrics_all_runs.csv"), index=False)

    out_rows = []
    for name, g in runs.groupby("model", sort=False):
        r = {"model": name, "n_runs": len(g)}
        for split in ["valid", "test"]:
            for metric in ["auc", "accuracy", "f1", "logloss"]:
                col=f"{split}_{metric}"
                r[f"{col}_mean"] = g[col].mean(); r[f"{col}_sd"] = g[col].std(ddof=1)
        out_rows.append(r)
    final_summary = pd.DataFrame(out_rows)
    final_summary.to_csv(os.path.join(args.outdir, "final_tuned_metrics_summary_mean_sd.csv"), index=False)

    formatted = pd.DataFrame([{
        "Model": r["model"],
        "Validation AUC": fmt(r["valid_auc_mean"], r["valid_auc_sd"]),
        "Validation Accuracy": fmt(r["valid_accuracy_mean"], r["valid_accuracy_sd"]),
        "Validation F1-score": fmt(r["valid_f1_mean"], r["valid_f1_sd"]),
        "Validation Log-loss": fmt(r["valid_logloss_mean"], r["valid_logloss_sd"]),
        "Test AUC": fmt(r["test_auc_mean"], r["test_auc_sd"]),
        "Test Accuracy": fmt(r["test_accuracy_mean"], r["test_accuracy_sd"]),
        "Test F1-score": fmt(r["test_f1_mean"], r["test_f1_sd"]),
        "Test Log-loss": fmt(r["test_logloss_mean"], r["test_logloss_sd"]),
    } for _,r in final_summary.iterrows()])
    formatted.to_csv(os.path.join(args.outdir, "final_tuned_metrics_summary_formatted.csv"), index=False)

    pred_mean = pd.DataFrame({"date": test["date"].astype(str).to_numpy(), "y": test["y"].to_numpy()})
    for name, probs in prob_store.items():
        safe = name.lower().replace(" ", "_")
        pred_mean[f"{safe}_probability_mean_10seeds"] = np.mean(np.vstack(probs), axis=0)
    pred_mean.to_csv(os.path.join(args.outdir, "final_tuned_test_predictions_mean_10seeds.csv"), index=False)

    comp_rows = []
    for _,r in final_summary.iterrows():
        name=r["model"]; orig=ORIGINAL_TABLE1_TEST[name]
        comp_rows.append({
            "model": name,
            "original_test_auc": orig["auc"], "tuned_test_auc": r["test_auc_mean"],
            "delta_auc_tuned_minus_original": r["test_auc_mean"]-orig["auc"],
            "original_test_accuracy": orig["accuracy"], "tuned_test_accuracy": r["test_accuracy_mean"],
            "delta_accuracy_tuned_minus_original": r["test_accuracy_mean"]-orig["accuracy"],
            "original_test_f1": orig["f1"], "tuned_test_f1": r["test_f1_mean"],
            "delta_f1_tuned_minus_original": r["test_f1_mean"]-orig["f1"],
            "original_test_logloss": orig["logloss"], "tuned_test_logloss": r["test_logloss_mean"],
            "delta_logloss_tuned_minus_original": r["test_logloss_mean"]-orig["logloss"],
        })
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(os.path.join(args.outdir, "comparison_original_vs_tuned_test.csv"), index=False)

    orig_rank = pd.DataFrame([{"model":k,"original_test_auc":v["auc"]} for k,v in ORIGINAL_TABLE1_TEST.items()])
    orig_rank["original_auc_rank"] = orig_rank["original_test_auc"].rank(method="min", ascending=False).astype(int)
    tuned_rank = final_summary[["model","test_auc_mean"]].rename(columns={"test_auc_mean":"tuned_test_auc"})
    tuned_rank["tuned_auc_rank"] = tuned_rank["tuned_test_auc"].rank(method="min", ascending=False).astype(int)
    ranks = orig_rank.merge(tuned_rank,on="model")
    ranks["rank_changed"] = ranks["original_auc_rank"] != ranks["tuned_auc_rank"]
    ranks.sort_values("original_auc_rank").to_csv(os.path.join(args.outdir, "ranking_original_vs_tuned.csv"), index=False)

    cfg = {
        "input": args.input, "outdir": args.outdir,
        "external_split": {"train_end": args.train_end, "valid_end": args.valid_end},
        "tuning_folds": TUNING_FOLDS,
        "selection_metric": "mean internal validation AUC",
        "tuning_seed": args.tuning_seed,
        "random_search_budget_per_nonlinear_model": args.n_search,
        "final_seeds": args.final_seeds,
        "selected_hyperparameters": selected,
        "catboost_use_best_model": False,
        "note": "All hyperparameters were selected strictly within <=2017. External validation/test were not used for tuning."
    }
    with open(os.path.join(args.outdir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    with open(os.path.join(args.outdir, "README_priority5.txt"), "w", encoding="utf-8") as f:
        f.write("""Priority 5 standardized time-aware tuning\n\n"
                "Internal folds: <=2011 -> 2012-13; <=2013 -> 2014-15; <=2015 -> 2016-17.\n"
                "Selection metric: mean internal-validation AUC.\n"
                "External validation: 2018-2019. Test: >2019. Neither is used for tuning.\n\n"
                "Inspect first:\n"
                "1. selected_hyperparameters.csv\n"
                "2. final_tuned_metrics_summary_formatted.csv\n"
                "3. comparison_original_vs_tuned_test.csv\n"
                "4. ranking_original_vs_tuned.csv\n\n"
                "If ranking is unchanged and shifts are small, report as sensitivity analysis.\n"
                "If ranking changes materially, reconsider Table 1 and rerun downstream bootstrap/DeLong with tuned predictions.\n""")

    print("\nFINAL TUNED PERFORMANCE\n", formatted.to_string(index=False))
    print("\nORIGINAL VS TUNED\n", comp.to_string(index=False))
    print("\nSaved to:", args.outdir)


if __name__ == "__main__":
    main()
