import os
import json
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, log_loss, roc_curve, auc

IDENTITY_COLUMNS = [
    "teamA_key", "teamB_key",
    "teamA_player1", "teamA_player2",
    "teamB_player1", "teamB_player2",
    "team1_key", "team2_key",
    "team1_player1", "team1_player2",
    "team2_player1", "team2_player2",
]

# Align subgroup analysis with the retained baseline CatBoost setting used in the main results.
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
    raise ValueError("Could not detect target column automatically.")


def split_by_time(df: pd.DataFrame, train_end: str = "2017-12-31", valid_end: str = "2019-12-31"):
    dfx = df.copy()
    if "date" in dfx.columns:
        dfx["date"] = pd.to_datetime(dfx["date"], errors="coerce")
        train_df = dfx[dfx["date"] <= train_end].copy()
        valid_df = dfx[(dfx["date"] > train_end) & (dfx["date"] <= valid_end)].copy()
        test_df = dfx[dfx["date"] > valid_end].copy()
    elif "year" in dfx.columns:
        train_df = dfx[dfx["year"] <= 2017].copy()
        valid_df = dfx[(dfx["year"] >= 2018) & (dfx["year"] <= 2019)].copy()
        test_df = dfx[dfx["year"] >= 2020].copy()
    else:
        raise ValueError("Need either a 'date' or 'year' column for time-aware split.")
    return train_df, valid_df, test_df


def make_xy(df: pd.DataFrame, target_col: str):
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    X = df[FEATURE_COLUMNS].copy()
    y = df[target_col].astype(int).values
    return X, y


def fill_missing(train_X: pd.DataFrame, valid_X: pd.DataFrame, test_X: pd.DataFrame):
    train_X = train_X.copy()
    valid_X = valid_X.copy()
    test_X = test_X.copy()

    cat_cols, num_cols = [], []
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


def evaluate_binary(y_true, proba):
    pred = (proba >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, proba)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred)),
        "logloss": float(log_loss(y_true, proba)),
    }


def save_roc(y_true, y_score, title, out_path_png, out_path_pdf):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", lw=1)
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


def save_importance(imp_df, outdir, tag, topk=20):
    imp_csv = os.path.join(outdir, f"feature_importance_{tag}.csv")
    imp_df.to_csv(imp_csv, index=False)

    top_imp = imp_df.head(topk).iloc[::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(top_imp["feature"], top_imp["importance_mean"])
    plt.xlabel("Mean importance")
    plt.ylabel("Feature")
    plt.title(f"Baseline CatBoost feature importance ({tag}, top {topk})")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"fig_feature_importance_{tag}_top{topk}.png"), dpi=300)
    plt.savefig(os.path.join(outdir, f"fig_feature_importance_{tag}_top{topk}.pdf"))
    plt.close()


def format_mean_sd(mean_val, sd_val, digits=4):
    return f"{mean_val:.{digits}f} ± {sd_val:.{digits}f}"


def run_one_group(df_all: pd.DataFrame, group_name: str, outdir: str, target_col: str, seeds: List[int], train_end: str, valid_end: str):
    train_df, valid_df, test_df = split_by_time(df_all, train_end=train_end, valid_end=valid_end)
    X_train, y_train = make_xy(train_df, target_col)
    X_valid, y_valid = make_xy(valid_df, target_col)
    X_test, y_test = make_xy(test_df, target_col)

    all_rows = []
    valid_probs = []
    test_probs = []
    importance_frames = []

    for seed in seeds:
        Xt, Xv, Xs, cat_feature_indices = fill_missing(X_train, X_valid, X_test)
        model = CatBoostClassifier(
            iterations=900,
            depth=6,
            learning_rate=0.03,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=seed,
            l2_leaf_reg=5,
            subsample=0.85,
            allow_writing_files=False,
            verbose=False,
        )
        model.fit(Xt, y_train, cat_features=cat_feature_indices, eval_set=(Xv, y_valid), use_best_model=True)

        valid_proba = model.predict_proba(Xv)[:, 1]
        test_proba = model.predict_proba(Xs)[:, 1]
        valid_metrics = evaluate_binary(y_valid, valid_proba)
        test_metrics = evaluate_binary(y_test, test_proba)

        all_rows.append({
            "group": group_name,
            "seed": seed,
            "n_total": len(df_all),
            "n_train": len(train_df),
            "n_valid": len(valid_df),
            "n_test": len(test_df),
            "n_features": len(FEATURE_COLUMNS),
            **{f"valid_{k}": v for k, v in valid_metrics.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()},
        })

        valid_probs.append(valid_proba)
        test_probs.append(test_proba)
        importance_frames.append(pd.DataFrame({
            "feature": FEATURE_COLUMNS,
            "importance": model.get_feature_importance()
        }))

    runs_df = pd.DataFrame(all_rows)

    # Summary mean / std
    metric_cols = [c for c in runs_df.columns if c.startswith("valid_") or c.startswith("test_")]
    summary = runs_df.groupby(["group"])[metric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = [
        col if isinstance(col, str)
        else col[0] if col[1] == ""
        else f"{col[0]}_{col[1]}"
        for col in summary.columns
    ]
    for c in summary.columns:
        if c != "group":
            summary[c] = summary[c].round(4)

    formatted = summary.copy()
    for split in ["valid", "test"]:
        for metric in ["auc", "accuracy", "f1", "logloss"]:
            formatted[f"{split}_{metric}_mean_sd"] = formatted.apply(
                lambda r: format_mean_sd(r[f"{split}_{metric}_mean"], r[f"{split}_{metric}_std"]), axis=1
            )

    # Mean ROC across seeds
    mean_valid = np.mean(np.vstack(valid_probs), axis=0)
    mean_test = np.mean(np.vstack(test_probs), axis=0)
    save_roc(
        y_valid, mean_valid,
        title=f"ROC Curve - {group_name} (VALID)",
        out_path_png=os.path.join(outdir, f"roc_{group_name.lower()}_valid.png"),
        out_path_pdf=os.path.join(outdir, f"roc_{group_name.lower()}_valid.pdf"),
    )
    save_roc(
        y_test, mean_test,
        title=f"ROC Curve - {group_name} (TEST)",
        out_path_png=os.path.join(outdir, f"roc_{group_name.lower()}_test.png"),
        out_path_pdf=os.path.join(outdir, f"roc_{group_name.lower()}_test.pdf"),
    )

    # Mean feature importance across seeds
    imp_long = pd.concat(importance_frames, ignore_index=True)
    imp_summary = (
        imp_long.groupby("feature")["importance"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "importance_mean", "std": "importance_std"})
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    imp_summary["importance_mean"] = imp_summary["importance_mean"].round(6)
    imp_summary["importance_std"] = imp_summary["importance_std"].round(6)
    save_importance(imp_summary, outdir, group_name.lower(), topk=20)

    return runs_df, summary, formatted[[
        "group",
        "valid_auc_mean_sd", "valid_accuracy_mean_sd", "valid_f1_mean_sd", "valid_logloss_mean_sd",
        "test_auc_mean_sd", "test_accuracy_mean_sd", "test_f1_mean_sd", "test_logloss_mean_sd",
    ]]




def plot_test_auc_summary(summary_df, outdir):
    """Create a single figure showing test AUC across all subgroups (mean ± SD)."""
    os.makedirs(outdir, exist_ok=True)

    df_plot = summary_df.copy()
    order = ["gender_M", "gender_W", "circuit_AVP", "circuit_FIVB"]
    label_map = {
        "gender_M": "Men",
        "gender_W": "Women",
        "circuit_AVP": "AVP",
        "circuit_FIVB": "FIVB",
    }
    df_plot = df_plot[df_plot["group"].isin(order)].copy()
    df_plot["group"] = pd.Categorical(df_plot["group"], categories=order, ordered=True)
    df_plot = df_plot.sort_values("group")

    labels = [label_map.get(g, g) for g in df_plot["group"].astype(str)]
    means = df_plot["test_auc_mean"].astype(float).values
    sds = df_plot["test_auc_std"].astype(float).values

    plt.figure(figsize=(7, 5))
    x = np.arange(len(labels))
    plt.bar(x, means, yerr=sds, capsize=4)
    plt.xticks(x, labels)
    plt.ylabel("Test AUC")
    plt.title("Test AUC across gender and competition circuits")
    plt.ylim(0.0, min(1.0, max(means + sds) + 0.05))
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig_subgroup_test_auc_summary.png"), dpi=300)
    plt.savefig(os.path.join(outdir, "fig_subgroup_test_auc_summary.pdf"))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Subgroup analysis for retained baseline CatBoost with repeated seeds")
    parser.add_argument("--input", type=str, default="./legacy_project/final_catboost_v1/data/prematch_features_v3.csv")
    parser.add_argument("--outdir", type=str, default="./legacy_project/final_catboost_v1/subgroup_outputs_with_sd")
    parser.add_argument("--min-samples", type=int, default=500)
    parser.add_argument("--train-end", type=str, default="2017-12-31")
    parser.add_argument("--valid-end", type=str, default="2019-12-31")
    parser.add_argument("--seeds", type=int, nargs="+", default=[41, 51, 61, 71, 81, 91, 101, 111, 121, 131])
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = load_csv(args.input)
    target_col = detect_target_column(df)

    notes = [
        "Subgroup analysis for the retained baseline CatBoost model",
        f"Input: {args.input}",
        f"Seeds: {args.seeds}",
        f"Time split: train <= {args.train_end}, valid <= {args.valid_end}, test > {args.valid_end}",
        f"Target column: {target_col}",
        f"Feature count: {len(FEATURE_COLUMNS)}",
    ]

    subgroup_defs = []
    if "gender" in df.columns:
        subgroup_defs.append(("gender_M", df[df["gender"].astype(str) == "M"].copy()))
        subgroup_defs.append(("gender_W", df[df["gender"].astype(str) == "W"].copy()))
    if "circuit" in df.columns:
        subgroup_defs.append(("circuit_AVP", df[df["circuit"].astype(str) == "AVP"].copy()))
        subgroup_defs.append(("circuit_FIVB", df[df["circuit"].astype(str) == "FIVB"].copy()))

    all_runs_list = []
    all_summary_list = []
    all_formatted_list = []

    for group_name, subdf in subgroup_defs:
        if len(subdf) < args.min_samples:
            notes.append(f"Skipped {group_name}: only {len(subdf)} rows (< {args.min_samples})")
            continue
        print(f"\n===== Running subgroup: {group_name} | n={len(subdf)} =====")
        runs_df, summary_df, formatted_df = run_one_group(
            subdf, group_name, args.outdir, target_col, args.seeds, args.train_end, args.valid_end
        )
        all_runs_list.append(runs_df)
        all_summary_list.append(summary_df)
        all_formatted_list.append(formatted_df)

    if not all_runs_list:
        raise ValueError("No subgroup satisfied the minimum sample requirement.")

    runs_all = pd.concat(all_runs_list, ignore_index=True)
    summary_all = pd.concat(all_summary_list, ignore_index=True)
    formatted_all = pd.concat(all_formatted_list, ignore_index=True)

    runs_all.to_csv(os.path.join(args.outdir, "subgroup_metrics_all_runs.csv"), index=False)
    summary_all.to_csv(os.path.join(args.outdir, "subgroup_metrics_summary_mean_sd.csv"), index=False)
    formatted_all.to_csv(os.path.join(args.outdir, "subgroup_metrics_summary_formatted.csv"), index=False)

    # Combined test AUC figure across all subgroups
    plot_test_auc_summary(summary_all, args.outdir)

    with open(os.path.join(args.outdir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "input": args.input,
            "outdir": args.outdir,
            "seeds": args.seeds,
            "train_end": args.train_end,
            "valid_end": args.valid_end,
            "min_samples": args.min_samples,
            "target_col": target_col,
            "feature_count": len(FEATURE_COLUMNS),
            "subgroups": [g for g, _ in subgroup_defs],
        }, f, indent=2, ensure_ascii=False)

    with open(os.path.join(args.outdir, "README_subgroup_analysis.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(notes))
        f.write("\n\nGenerated files:\n")
        f.write("- subgroup_metrics_all_runs.csv\n")
        f.write("- subgroup_metrics_summary_mean_sd.csv\n")
        f.write("- subgroup_metrics_summary_formatted.csv\n")
        f.write("- roc_*_valid/test.(png/pdf)\n")
        f.write("- fig_subgroup_test_auc_summary.(png/pdf)\n")
        f.write("- feature_importance_*.csv\n")
        f.write("- fig_feature_importance_*_top20.(png/pdf)\n")

    print(f"\nSaved subgroup metrics to: {os.path.join(args.outdir, 'subgroup_metrics_summary_formatted.csv')}")
    print(f"Saved all subgroup outputs to: {args.outdir}")


if __name__ == "__main__":
    main()
