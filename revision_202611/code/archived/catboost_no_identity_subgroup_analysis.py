import os
import argparse
import numpy as np
import pandas as pd
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


def load_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", low_memory=False)


def detect_target_column(df: pd.DataFrame) -> str:
    candidates = [
        "y", "target", "label", "outcome", "result",
        "teamA_win", "team1_win", "win"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError("Could not detect target column automatically.")


def detect_drop_columns(df: pd.DataFrame, target_col: str):
    drop_cols = [target_col]
    for c in [
        "match_id", "event_id", "tournament_id",
        "date", "match_date", "datetime",
        "winner_name", "loser_name"
    ]:
        if c in df.columns:
            drop_cols.append(c)
    for c in IDENTITY_COLUMNS:
        if c in df.columns:
            drop_cols.append(c)
    return list(dict.fromkeys(drop_cols))


def time_split(df: pd.DataFrame):
    dfx = df.copy()
    if "date" in dfx.columns:
        dfx["__sort_date__"] = pd.to_datetime(dfx["date"], errors="coerce")
        dfx = dfx.sort_values("__sort_date__").reset_index(drop=True)
    elif "match_date" in dfx.columns:
        dfx["__sort_date__"] = pd.to_datetime(dfx["match_date"], errors="coerce")
        dfx = dfx.sort_values("__sort_date__").reset_index(drop=True)
    elif "datetime" in dfx.columns:
        dfx["__sort_date__"] = pd.to_datetime(dfx["datetime"], errors="coerce")
        dfx = dfx.sort_values("__sort_date__").reset_index(drop=True)
    elif "year" in dfx.columns:
        dfx = dfx.sort_values("year").reset_index(drop=True)
    else:
        dfx = dfx.reset_index(drop=True)

    n = len(dfx)
    n_train = int(n * 0.70)
    n_valid = int(n * 0.15)

    train_df = dfx.iloc[:n_train].copy()
    valid_df = dfx.iloc[n_train:n_train + n_valid].copy()
    test_df = dfx.iloc[n_train + n_valid:].copy()
    return train_df, valid_df, test_df


def fill_missing(train_df, valid_df, test_df, feature_cols):
    train_df = train_df.copy()
    valid_df = valid_df.copy()
    test_df = test_df.copy()

    cat_cols, num_cols = [], []
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(train_df[c]):
            num_cols.append(c)
        else:
            cat_cols.append(c)

    for c in num_cols:
        med = train_df[c].median()
        train_df[c] = train_df[c].fillna(med)
        valid_df[c] = valid_df[c].fillna(med)
        test_df[c] = test_df[c].fillna(med)

    for c in cat_cols:
        if train_df[c].mode().shape[0] > 0:
            mode_val = train_df[c].mode().iloc[0]
        else:
            mode_val = "missing"
        train_df[c] = train_df[c].fillna(mode_val).astype(str)
        valid_df[c] = valid_df[c].fillna(mode_val).astype(str)
        test_df[c] = test_df[c].fillna(mode_val).astype(str)

    return train_df, valid_df, test_df, cat_cols, num_cols


def evaluate_binary(y_true, proba):
    pred = (proba >= 0.5).astype(int)
    return {
        "auc": roc_auc_score(y_true, proba),
        "accuracy": accuracy_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "logloss": log_loss(y_true, proba),
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
    plt.barh(top_imp["feature"], top_imp["importance"])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(f"CatBoost_no_identity Feature Importance ({tag}, Top {topk})")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"fig_feature_importance_{tag}_top{topk}.png"), dpi=300)
    plt.savefig(os.path.join(outdir, f"fig_feature_importance_{tag}_top{topk}.pdf"))
    plt.close()


def run_one_group(df_all: pd.DataFrame, group_name: str, outdir: str):
    target_col = detect_target_column(df_all)
    drop_cols = detect_drop_columns(df_all, target_col)
    feature_cols = [c for c in df_all.columns if c not in drop_cols]

    train_df, valid_df, test_df = time_split(df_all)

    train_df, valid_df, test_df, cat_cols, num_cols = fill_missing(
        train_df, valid_df, test_df, feature_cols
    )

    X_train = train_df[feature_cols]
    y_train = train_df[target_col].astype(int)

    X_valid = valid_df[feature_cols]
    y_valid = valid_df[target_col].astype(int)

    X_test = test_df[feature_cols]
    y_test = test_df[target_col].astype(int)

    cat_feature_indices = [feature_cols.index(c) for c in cat_cols]

    model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        verbose=100,
        random_seed=42
    )

    model.fit(
        X_train,
        y_train,
        cat_features=cat_feature_indices,
        eval_set=(X_valid, y_valid),
        use_best_model=True
    )

    valid_proba = model.predict_proba(X_valid)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]

    valid_metrics = evaluate_binary(y_valid, valid_proba)
    test_metrics = evaluate_binary(y_test, test_proba)

    # ROC
    save_roc(
        y_valid, valid_proba,
        title=f"ROC Curve - {group_name} (VALID)",
        out_path_png=os.path.join(outdir, f"roc_{group_name.lower()}_valid.png"),
        out_path_pdf=os.path.join(outdir, f"roc_{group_name.lower()}_valid.pdf"),
    )
    save_roc(
        y_test, test_proba,
        title=f"ROC Curve - {group_name} (TEST)",
        out_path_png=os.path.join(outdir, f"roc_{group_name.lower()}_test.png"),
        out_path_pdf=os.path.join(outdir, f"roc_{group_name.lower()}_test.pdf"),
    )

    # importance
    importances = model.get_feature_importance()
    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    save_importance(imp_df, outdir, group_name.lower(), topk=20)

    row = {
        "group": group_name,
        "n_total": len(df_all),
        "n_train": len(train_df),
        "n_valid": len(valid_df),
        "n_test": len(test_df),
        "n_features": len(feature_cols),
        "valid_auc": round(valid_metrics["auc"], 4),
        "valid_accuracy": round(valid_metrics["accuracy"], 4),
        "valid_f1": round(valid_metrics["f1"], 4),
        "valid_logloss": round(valid_metrics["logloss"], 4),
        "test_auc": round(test_metrics["auc"], 4),
        "test_accuracy": round(test_metrics["accuracy"], 4),
        "test_f1": round(test_metrics["f1"], 4),
        "test_logloss": round(test_metrics["logloss"], 4),
    }
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="./legacy_project/final_catboost_v1/data/prematch_features_v3.csv"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="./legacy_project/final_catboost_v1/subgroup_outputs"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=500
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = load_csv(args.input)

    notes = []
    notes.append("Subgroup analysis for CatBoost_no_identity")
    notes.append(f"Input: {args.input}")
    notes.append("Identity columns removed if present:")
    dropped_identity_cols = [c for c in IDENTITY_COLUMNS if c in df.columns]
    notes.extend(dropped_identity_cols)

    # groups
    subgroup_defs = []

    if "gender" in df.columns:
        subgroup_defs.append(("gender_M", df[df["gender"].astype(str) == "M"].copy()))
        subgroup_defs.append(("gender_W", df[df["gender"].astype(str) == "W"].copy()))

    if "circuit" in df.columns:
        subgroup_defs.append(("circuit_AVP", df[df["circuit"].astype(str) == "AVP"].copy()))
        subgroup_defs.append(("circuit_FIVB", df[df["circuit"].astype(str) == "FIVB"].copy()))

    rows = []
    for group_name, subdf in subgroup_defs:
        if len(subdf) < args.min_samples:
            notes.append(f"Skipped {group_name}: only {len(subdf)} rows (< {args.min_samples})")
            continue
        print(f"\n===== Running subgroup: {group_name} | n={len(subdf)} =====")
        row = run_one_group(subdf, group_name, args.outdir)
        rows.append(row)

    metrics_df = pd.DataFrame(rows)
    metrics_path = os.path.join(args.outdir, "subgroup_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    # summary note
    with open(os.path.join(args.outdir, "README_subgroup_analysis.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(notes))
        f.write("\n\nGenerated files:\n")
        f.write("- subgroup_metrics.csv\n")
        f.write("- roc_*_valid/test.(png/pdf)\n")
        f.write("- feature_importance_*_top20.(png/pdf)\n")
        f.write("- feature_importance_*.csv\n")

    print(f"\nSaved subgroup metrics to: {metrics_path}")
    print(f"Saved all subgroup outputs to: {args.outdir}")


if __name__ == "__main__":
    main()
