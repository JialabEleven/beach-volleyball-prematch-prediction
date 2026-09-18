#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, argparse
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, log_loss


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
    "teamA_key","teamB_key",
    "teamA_player1","teamA_player2",
    "teamB_player1","teamB_player2"
]


def read_csv_fallback(path):
    try:
        return pd.read_csv(path, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", low_memory=False)


def chronological_sort(df):
    df = df.copy()
    if "date" not in df.columns:
        raise ValueError("Missing required column: date")

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


def split_by_time(df, train_end="2017-12-31", valid_end="2019-12-31"):
    train_end = pd.Timestamp(train_end)
    valid_end = pd.Timestamp(valid_end)

    train = df[df["date"] <= train_end].copy().reset_index(drop=True)
    valid = df[(df["date"] > train_end) & (df["date"] <= valid_end)].copy().reset_index(drop=True)
    test = df[df["date"] > valid_end].copy().reset_index(drop=True)
    return train, valid, test


def chronology_audit(name, df):
    d = df["date"].to_numpy()
    violations = int(np.sum(d[1:] < d[:-1])) if len(d) > 1 else 0
    return {
        "split": name,
        "n_rows": len(df),
        "start_date": str(df["date"].min().date()),
        "end_date": str(df["date"].max().date()),
        "chronological_order_violations": violations,
        "is_chronological": violations == 0
    }


def build_feature_sets(df):
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Missing baseline features:\n" + "\n".join(missing))

    identity = [c for c in IDENTITY_COLUMNS if c in df.columns]
    if not identity:
        raise ValueError("No raw identity columns found.")

    baseline = FEATURE_COLUMNS.copy()
    full = baseline + identity
    return baseline, full, identity


def fit_preprocessor(train, features):
    numeric, categorical = [], []
    medians, modes = {}, {}

    for c in features:
        if pd.api.types.is_numeric_dtype(train[c]):
            numeric.append(c)
            med = train[c].median()
            medians[c] = 0.0 if pd.isna(med) else med
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


def metrics(y, p):
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    pred = (p >= 0.5).astype(int)
    return {
        "auc": roc_auc_score(y, p),
        "accuracy": accuracy_score(y, pred),
        "f1": f1_score(y, pred),
        "logloss": log_loss(y, p)
    }


def train_one(train, valid, test, features, target, seed, args):
    num, cat, med, mode = fit_preprocessor(train, features)

    Xtr = transform(train, features, num, cat, med, mode)
    Xva = transform(valid, features, num, cat, med, mode)
    Xte = transform(test, features, num, cat, med, mode)

    ytr = train[target].astype(int).to_numpy()
    yva = valid[target].astype(int).to_numpy()
    yte = test[target].astype(int).to_numpy()

    cat_idx = [Xtr.columns.get_loc(c) for c in cat]

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
        verbose=False
    )

    model.fit(
        Xtr, ytr,
        cat_features=cat_idx,
        eval_set=(Xva, yva),
        use_best_model=True,
        verbose=False
    )

    pva = model.predict_proba(Xva)[:, 1]
    pte = model.predict_proba(Xte)[:, 1]

    return {
        "model": model,
        "y_valid": yva,
        "p_valid": pva,
        "y_test": yte,
        "p_test": pte,
        "best_iteration": int(model.get_best_iteration())
    }


def summarize(df):
    rows = []
    for model_name, g in df.groupby("model", sort=False):
        row = {"model": model_name, "n_runs": len(g)}
        for split in ["valid", "test"]:
            for metric in ["auc", "accuracy", "f1", "logloss"]:
                col = f"{split}_{metric}"
                row[f"{col}_mean"] = g[col].mean()
                row[f"{col}_sd"] = g[col].std(ddof=1)
        row["best_iteration_mean"] = g["best_iteration"].mean()
        row["best_iteration_sd"] = g["best_iteration"].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def formatted_table(summary, split):
    display = {
        "CatBoost_baseline_has_time": "Leak-proof CatBoost baseline",
        "CatBoost_plus_identity_has_time": "Leak-proof CatBoost + identity"
    }
    rows = []
    for _, r in summary.iterrows():
        rows.append({
            "Model": display.get(r["model"], r["model"]),
            "AUC": f"{r[f'{split}_auc_mean']:.4f} ± {r[f'{split}_auc_sd']:.4f}",
            "Accuracy": f"{r[f'{split}_accuracy_mean']:.4f} ± {r[f'{split}_accuracy_sd']:.4f}",
            "F1-score": f"{r[f'{split}_f1_mean']:.4f} ± {r[f'{split}_f1_sd']:.4f}",
            "Log-loss": f"{r[f'{split}_logloss_mean']:.4f} ± {r[f'{split}_logloss_sd']:.4f}"
        })
    return pd.DataFrame(rows)


def delta_summary(runs):
    base = runs[runs["model"] == "CatBoost_baseline_has_time"].sort_values("seed").reset_index(drop=True)
    ident = runs[runs["model"] == "CatBoost_plus_identity_has_time"].sort_values("seed").reset_index(drop=True)

    if not np.array_equal(base["seed"].to_numpy(), ident["seed"].to_numpy()):
        raise RuntimeError("Seed alignment failed.")

    rows = []
    for split in ["valid", "test"]:
        row = {"split": split}
        for metric in ["auc", "accuracy", "f1", "logloss"]:
            d = ident[f"{split}_{metric}"].to_numpy() - base[f"{split}_{metric}"].to_numpy()
            row[f"identity_minus_baseline_{metric}_mean"] = d.mean()
            row[f"identity_minus_baseline_{metric}_sd"] = d.std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Leak-proof chronological CatBoost identity analysis with has_time=True"
    )
    parser.add_argument(
        "--input",
        default="/data2/zhangyiyi/beach_ai/final_catboost_v1/data/prematch_features_v3.csv"
    )
    parser.add_argument(
        "--outdir",
        default="/data2/zhangyiyi/beach_ai/final_catboost_v1/identity_leakproof_has_time_output"
    )
    parser.add_argument("--train-end", default="2017-12-31")
    parser.add_argument("--valid-end", default="2019-12-31")
    parser.add_argument("--iterations", type=int, default=900)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--l2-leaf-reg", type=float, default=5.0)
    parser.add_argument("--subsample", type=float, default=0.85)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = read_csv_fallback(args.input)
    if "y" not in df.columns:
        raise ValueError("Target column y not found.")
    df["y"] = pd.to_numeric(df["y"], errors="raise").astype(int)

    df, sort_cols = chronological_sort(df)
    train, valid, test = split_by_time(df, args.train_end, args.valid_end)

    audit = pd.DataFrame([
        chronology_audit("train", train),
        chronology_audit("validation", valid),
        chronology_audit("test", test)
    ])
    audit.to_csv(os.path.join(args.outdir, "chronological_order_audit.csv"), index=False)

    if not audit["is_chronological"].all():
        raise RuntimeError("Chronological audit failed.")

    baseline_features, full_features, identity_cols = build_feature_sets(df)

    with open(os.path.join(args.outdir, "raw_identity_columns_used.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(identity_cols))

    all_runs = []
    baseline_test_probs, identity_test_probs = [], []
    baseline_valid_probs, identity_valid_probs = [], []

    best_identity_auc = -np.inf
    best_identity_model = None
    best_identity_seed = None

    for seed in args.seeds:
        print(f"\n===== Seed {seed} =====")

        b = train_one(train, valid, test, baseline_features, "y", seed, args)
        i = train_one(train, valid, test, full_features, "y", seed, args)

        assert np.array_equal(b["y_valid"], i["y_valid"])
        assert np.array_equal(b["y_test"], i["y_test"])

        bv = metrics(b["y_valid"], b["p_valid"])
        bt = metrics(b["y_test"], b["p_test"])
        iv = metrics(i["y_valid"], i["p_valid"])
        it = metrics(i["y_test"], i["p_test"])

        all_runs.append({
            "seed": seed,
            "model": "CatBoost_baseline_has_time",
            "best_iteration": b["best_iteration"],
            **{f"valid_{k}": v for k, v in bv.items()},
            **{f"test_{k}": v for k, v in bt.items()}
        })

        all_runs.append({
            "seed": seed,
            "model": "CatBoost_plus_identity_has_time",
            "best_iteration": i["best_iteration"],
            **{f"valid_{k}": v for k, v in iv.items()},
            **{f"test_{k}": v for k, v in it.items()}
        })

        baseline_test_probs.append(b["p_test"])
        identity_test_probs.append(i["p_test"])
        baseline_valid_probs.append(b["p_valid"])
        identity_valid_probs.append(i["p_valid"])

        if it["auc"] > best_identity_auc:
            best_identity_auc = it["auc"]
            best_identity_model = i["model"]
            best_identity_seed = seed

        print("Baseline TEST :", {k: round(v, 4) for k, v in bt.items()})
        print("+Identity TEST:", {k: round(v, 4) for k, v in it.items()})

    runs = pd.DataFrame(all_runs)
    runs.to_csv(os.path.join(args.outdir, "leakproof_identity_metrics_all_runs.csv"), index=False)

    summary = summarize(runs)
    summary.to_csv(os.path.join(args.outdir, "leakproof_identity_summary_mean_sd.csv"), index=False)

    test_table = formatted_table(summary, "test")
    valid_table = formatted_table(summary, "valid")

    test_table.to_csv(
        os.path.join(args.outdir, "supplementary_table_identity_leakproof_test.csv"),
        index=False
    )
    valid_table.to_csv(
        os.path.join(args.outdir, "supplementary_table_identity_leakproof_validation.csv"),
        index=False
    )

    delta = delta_summary(runs)
    delta.to_csv(
        os.path.join(args.outdir, "identity_minus_baseline_delta_summary.csv"),
        index=False
    )

    pd.DataFrame({
        "date": test["date"].astype(str),
        "y": test["y"].to_numpy(),
        "baseline_has_time_probability_mean_10seeds":
            np.mean(np.vstack(baseline_test_probs), axis=0),
        "identity_has_time_probability_mean_10seeds":
            np.mean(np.vstack(identity_test_probs), axis=0)
    }).to_csv(
        os.path.join(args.outdir, "test_predictions_mean_10seeds.csv"),
        index=False
    )

    pd.DataFrame({
        "date": valid["date"].astype(str),
        "y": valid["y"].to_numpy(),
        "baseline_has_time_probability_mean_10seeds":
            np.mean(np.vstack(baseline_valid_probs), axis=0),
        "identity_has_time_probability_mean_10seeds":
            np.mean(np.vstack(identity_valid_probs), axis=0)
    }).to_csv(
        os.path.join(args.outdir, "validation_predictions_mean_10seeds.csv"),
        index=False
    )

    if best_identity_model is not None:
        pd.DataFrame({
            "feature": best_identity_model.feature_names_,
            "importance": best_identity_model.get_feature_importance()
        }).sort_values("importance", ascending=False).to_csv(
            os.path.join(args.outdir, "best_identity_model_feature_importance.csv"),
            index=False
        )

    config = {
        "input": args.input,
        "outdir": args.outdir,
        "chronological_sort_columns": sort_cols,
        "train_end": args.train_end,
        "valid_end": args.valid_end,
        "seeds": args.seeds,
        "has_time": True,
        "identity_columns": identity_cols,
        "baseline_feature_count": len(baseline_features),
        "identity_augmented_feature_count": len(full_features),
        "best_identity_seed_by_test_auc": best_identity_seed,
        "catboost_parameters": {
            "iterations": args.iterations,
            "depth": args.depth,
            "learning_rate": args.learning_rate,
            "l2_leaf_reg": args.l2_leaf_reg,
            "subsample": args.subsample,
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "has_time": True,
            "use_best_model": True
        }
    }

    with open(os.path.join(args.outdir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("\nFINAL TEST SUMMARY")
    print(test_table.to_string(index=False))
    print("\nVALIDATION SUMMARY")
    print(valid_table.to_string(index=False))
    print("\nIDENTITY - BASELINE DELTAS")
    print(delta.to_string(index=False))
    print("\nSaved to:", args.outdir)


if __name__ == "__main__":
    main()
