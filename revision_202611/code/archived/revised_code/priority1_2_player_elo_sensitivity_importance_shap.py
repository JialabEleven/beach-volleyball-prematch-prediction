#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import importlib.util
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

TARGET_FEATURE = "avg_player_elo_diff"


def load_reference_module(path):
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Reference script not found: {path}")
    spec = importlib.util.spec_from_file_location("priority8_reference", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def mean_sd(x):
    x = np.asarray(x, dtype=float)
    return f"{x.mean():.4f} ± {x.std(ddof=1):.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="./legacy_project/final_catboost_v1/player_elo_sensitivity/prematch_features_opponent_average.csv",
    )
    parser.add_argument(
        "--outdir",
        default="./legacy_project/final_catboost_v1/player_elo_sensitivity/importance_shap_output",
    )
    parser.add_argument(
        "--reference-script",
        default="priority8_feature_block_ablation_final_v1.py",
    )
    parser.add_argument("--train-end", default="2017-12-31")
    parser.add_argument("--valid-end", default="2019-12-31")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--shap-sample-size", type=int, default=2000)
    parser.add_argument("--shap-sample-seed", type=int, default=20260909)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Reuse the exact retained-model definitions from Priority 8.
    ref = load_reference_module(args.reference_script)
    features = list(ref.FULL_FEATURES)
    params = dict(ref.CATBOOST_PARAMS)
    seeds = list(ref.FINAL_SEEDS)

    if len(features) != 104:
        raise RuntimeError(f"Expected 104 features, found {len(features)}")
    if TARGET_FEATURE not in features:
        raise RuntimeError(f"{TARGET_FEATURE} not found in FULL_FEATURES")

    print("=" * 90)
    print("PLAYER-ELO SENSITIVITY: FEATURE IMPORTANCE + SHAP")
    print("=" * 90)
    print("Input :", args.input)
    print("Output:", args.outdir)
    print("Seeds :", seeds)

    # Exact Priority-8 loading / sorting / split / preprocessing.
    df = ref.load_csv(args.input)
    df = ref.prepare_date_and_sort(df)
    ref.validate_features_exist(df)
    target_col = ref.detect_target_column(df)

    train_df, valid_df, test_df = ref.split_by_time(
        df,
        train_end=args.train_end,
        valid_end=args.valid_end,
    )

    X_train, y_train = ref.make_xy(train_df, target_col, features)
    X_valid, y_valid = ref.make_xy(valid_df, target_col, features)
    X_test, y_test = ref.make_xy(test_df, target_col, features)

    Xt, Xv, Xs, cat_idx = ref.fill_missing(X_train, X_valid, X_test)

    print(f"Train: {len(train_df):,}")
    print(f"Valid: {len(valid_df):,}")
    print(f"Test : {len(test_df):,}")

    # One fixed held-out test subset for SHAP, reused across all seeds.
    rng = np.random.RandomState(args.shap_sample_seed)
    if len(Xs) > args.shap_sample_size:
        shap_pos = np.sort(
            rng.choice(len(Xs), size=args.shap_sample_size, replace=False)
        )
    else:
        shap_pos = np.arange(len(Xs))

    X_shap = Xs.iloc[shap_pos].copy()
    y_shap = y_test[shap_pos]
    shap_pool = Pool(X_shap, label=y_shap, cat_features=cat_idx)

    fi_all = []
    shap_all = []
    perf_rows = []

    for i, seed in enumerate(seeds, 1):
        print(f"\n===== seed {i}/{len(seeds)} = {seed} =====")

        model = CatBoostClassifier(**params, random_seed=seed)

        fit_kwargs = dict(
            X=Xt,
            y=y_train,
            eval_set=(Xv, y_valid),
            use_best_model=True,
            verbose=False,
        )
        if cat_idx:
            fit_kwargs["cat_features"] = cat_idx

        model.fit(**fit_kwargs)

        p_valid = model.predict_proba(Xv)[:, 1]
        p_test = model.predict_proba(Xs)[:, 1]

        mv = ref.evaluate_binary(y_valid, p_valid)
        mt = ref.evaluate_binary(y_test, p_test)

        perf_rows.append({
            "seed": seed,
            "best_iteration": int(model.get_best_iteration()),
            **{f"valid_{k}": v for k, v in mv.items()},
            **{f"test_{k}": v for k, v in mt.items()},
        })

        print(
            f"Valid AUC={mv['auc']:.4f} | Test AUC={mt['auc']:.4f} | "
            f"Acc={mt['accuracy']:.4f} | F1={mt['f1']:.4f} | "
            f"LogLoss={mt['logloss']:.4f}"
        )

        # Built-in feature importance.
        fi_all.append(np.asarray(model.get_feature_importance(), dtype=float))

        # Native CatBoost SHAP; final column is expected value.
        shap_raw = np.asarray(
            model.get_feature_importance(shap_pool, type="ShapValues"),
            dtype=float,
        )
        shap_values = shap_raw[:, :-1]
        shap_all.append(np.mean(np.abs(shap_values), axis=0))

    perf = pd.DataFrame(perf_rows)
    perf.to_csv(os.path.join(args.outdir, "performance_all_runs.csv"), index=False)

    perf_summary = pd.DataFrame({
        "metric": ["valid_auc", "test_auc", "test_accuracy", "test_f1", "test_logloss"],
        "mean": [
            perf["valid_auc"].mean(),
            perf["test_auc"].mean(),
            perf["test_accuracy"].mean(),
            perf["test_f1"].mean(),
            perf["test_logloss"].mean(),
        ],
        "sd": [
            perf["valid_auc"].std(ddof=1),
            perf["test_auc"].std(ddof=1),
            perf["test_accuracy"].std(ddof=1),
            perf["test_f1"].std(ddof=1),
            perf["test_logloss"].std(ddof=1),
        ],
    })
    perf_summary.to_csv(
        os.path.join(args.outdir, "performance_summary.csv"), index=False
    )

    fi_arr = np.vstack(fi_all)
    shap_arr = np.vstack(shap_all)

    summary = pd.DataFrame({
        "feature": features,
        "catboost_importance_mean": fi_arr.mean(axis=0),
        "catboost_importance_sd": fi_arr.std(axis=0, ddof=1),
        "mean_abs_shap_mean": shap_arr.mean(axis=0),
        "mean_abs_shap_sd": shap_arr.std(axis=0, ddof=1),
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

    summary["catboost_importance_share_pct"] = (
        100 * summary["catboost_importance_mean"]
        / summary["catboost_importance_mean"].sum()
    )
    summary["mean_abs_shap_share_pct"] = (
        100 * summary["mean_abs_shap_mean"]
        / summary["mean_abs_shap_mean"].sum()
    )

    summary.sort_values("mean_abs_shap_rank").to_csv(
        os.path.join(args.outdir, "all_features_importance_shap_ranks.csv"),
        index=False,
    )

    top_fi = summary.sort_values(
        "catboost_importance_mean", ascending=False
    ).head(args.topk)
    top_shap = summary.sort_values(
        "mean_abs_shap_mean", ascending=False
    ).head(args.topk)

    top_fi.to_csv(
        os.path.join(args.outdir, f"top{args.topk}_by_catboost_importance.csv"),
        index=False,
    )
    top_shap.to_csv(
        os.path.join(args.outdir, f"top{args.topk}_by_mean_abs_shap.csv"),
        index=False,
    )

    target = summary.loc[summary["feature"] == TARGET_FEATURE].copy()
    target.to_csv(
        os.path.join(args.outdir, "avg_player_elo_diff_rank.csv"),
        index=False,
    )

    row = target.iloc[0]

    print("\n" + "=" * 90)
    print("PERFORMANCE CROSS-CHECK")
    print("=" * 90)
    print("Validation AUC:", mean_sd(perf["valid_auc"]))
    print("Test AUC      :", mean_sd(perf["test_auc"]))
    print("Test Accuracy :", mean_sd(perf["test_accuracy"]))
    print("Test F1       :", mean_sd(perf["test_f1"]))
    print("Test LogLoss  :", mean_sd(perf["test_logloss"]))

    print("\n" + "=" * 90)
    print("avg_player_elo_diff RANK")
    print("=" * 90)
    print(
        f"CatBoost importance: {row['catboost_importance_mean']:.6f} ± "
        f"{row['catboost_importance_sd']:.6f} | "
        f"rank {int(row['catboost_importance_rank'])}/{len(features)} | "
        f"share {row['catboost_importance_share_pct']:.2f}%"
    )
    print(
        f"Mean |SHAP|       : {row['mean_abs_shap_mean']:.6f} ± "
        f"{row['mean_abs_shap_sd']:.6f} | "
        f"rank {int(row['mean_abs_shap_rank'])}/{len(features)} | "
        f"share {row['mean_abs_shap_share_pct']:.2f}%"
    )

    print("\nTop 10 by CatBoost importance:")
    print(
        top_fi[["feature", "catboost_importance_mean", "catboost_importance_rank"]]
        .head(10)
        .to_string(index=False)
    )

    print("\nTop 10 by mean |SHAP|:")
    print(
        top_shap[["feature", "mean_abs_shap_mean", "mean_abs_shap_rank"]]
        .head(10)
        .to_string(index=False)
    )

    print("\nSaved to:", args.outdir)


if __name__ == "__main__":
    main()
