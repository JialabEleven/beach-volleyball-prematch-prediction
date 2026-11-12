import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, log_loss


def detect_target_column(df):
    candidates = [
        "target",
        "label",
        "y",
        "outcome",
        "result",
        "teamA_win",
        "team1_win",
        "win"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        "Could not detect target column automatically. "
        "Please check your feature file and modify the script if needed."
    )


def detect_drop_columns(df, target_col):
    drop_cols = [target_col]

    # 常见不应进入模型的列
    for c in [
        "match_id", "event_id", "tournament_id",
        "date", "match_date", "datetime",
        "teamA_name", "teamB_name",
        "team1_name", "team2_name",
        "winner_name", "loser_name"
    ]:
        if c in df.columns:
            drop_cols.append(c)

    return list(dict.fromkeys(drop_cols))


def train_valid_test_split_timeaware(df):
    """
    尽量和你原来的思路保持一致：
    1) 如果有日期列，就按日期排序
    2) 否则如果有 year，就按 year 排序
    3) 再做 70/15/15 切分
    """
    df = df.copy()

    if "date" in df.columns:
        df["__sort_date__"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("__sort_date__").reset_index(drop=True)
    elif "match_date" in df.columns:
        df["__sort_date__"] = pd.to_datetime(df["match_date"], errors="coerce")
        df = df.sort_values("__sort_date__").reset_index(drop=True)
    elif "datetime" in df.columns:
        df["__sort_date__"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.sort_values("__sort_date__").reset_index(drop=True)
    elif "year" in df.columns:
        df = df.sort_values("year").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    n = len(df)
    n_train = int(n * 0.70)
    n_valid = int(n * 0.15)

    train_df = df.iloc[:n_train].copy()
    valid_df = df.iloc[n_train:n_train + n_valid].copy()
    test_df = df.iloc[n_train + n_valid:].copy()

    return train_df, valid_df, test_df


def fill_missing(train_df, valid_df, test_df, feature_cols):
    train_df = train_df.copy()
    valid_df = valid_df.copy()
    test_df = test_df.copy()

    cat_cols = []
    num_cols = []

    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(train_df[c]):
            num_cols.append(c)
        else:
            cat_cols.append(c)

    # 数值列：中位数填补
    for c in num_cols:
        med = train_df[c].median()
        train_df[c] = train_df[c].fillna(med)
        valid_df[c] = valid_df[c].fillna(med)
        test_df[c] = test_df[c].fillna(med)

    # 类别列：众数填补 + 转字符串
    for c in cat_cols:
        if train_df[c].mode().shape[0] > 0:
            mode_val = train_df[c].mode().iloc[0]
        else:
            mode_val = "missing"
        train_df[c] = train_df[c].fillna(mode_val).astype(str)
        valid_df[c] = valid_df[c].fillna(mode_val).astype(str)
        test_df[c] = test_df[c].fillna(mode_val).astype(str)

    return train_df, valid_df, test_df, cat_cols, num_cols


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
        default="./legacy_project/final_catboost_v1/explain_outputs"
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=20
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=2000,
        help="Maximum number of test rows used for SHAP plotting."
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 读数据
    try:
        df = pd.read_csv(args.input, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(args.input, encoding="latin-1", low_memory=False)

    target_col = detect_target_column(df)
    drop_cols = detect_drop_columns(df, target_col)
    feature_cols = [c for c in df.columns if c not in drop_cols]

    print(f"Detected target column: {target_col}")
    print(f"Number of total features: {len(feature_cols)}")

    # 切分
    train_df, valid_df, test_df = train_valid_test_split_timeaware(df)

    # 缺失值处理 + 类别检测
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

    # 训练 CatBoost
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

    # 测试集预测
    test_pred_proba = model.predict_proba(X_test)[:, 1]
    test_pred = (test_pred_proba >= 0.5).astype(int)

    test_auc = roc_auc_score(y_test, test_pred_proba)
    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred)
    test_ll = log_loss(y_test, test_pred_proba)

    with open(os.path.join(args.outdir, "catboost_test_metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"Test AUC: {test_auc:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test F1: {test_f1:.4f}\n")
        f.write(f"Test LogLoss: {test_ll:.4f}\n")

    print(f"Test AUC = {test_auc:.4f}")
    print(f"Test Accuracy = {test_acc:.4f}")
    print(f"Test F1 = {test_f1:.4f}")
    print(f"Test LogLoss = {test_ll:.4f}")

    # ========= 1) CatBoost 原生特征重要性 =========
    importances = model.get_feature_importance()
    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    imp_df.to_csv(
        os.path.join(args.outdir, "catboost_feature_importance.csv"),
        index=False
    )

    top_imp = imp_df.head(args.topk).iloc[::-1]

    plt.figure(figsize=(8, 6))
    plt.barh(top_imp["feature"], top_imp["importance"])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(f"CatBoost Feature Importance (Top {args.topk})")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"fig_catboost_feature_importance_top{args.topk}.png"), dpi=300)
    plt.savefig(os.path.join(args.outdir, f"fig_catboost_feature_importance_top{args.topk}.pdf"))
    plt.close()

    # ========= 2) SHAP =========
    # 取测试集一部分做 SHAP，避免太慢
    if len(X_test) > args.sample_size:
        X_shap = X_test.sample(args.sample_size, random_state=42)
    else:
        X_shap = X_test.copy()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)

    # 如果返回 list（二分类老版本行为），取正类
    if isinstance(shap_values, list):
        shap_values_to_plot = shap_values[1]
    else:
        shap_values_to_plot = shap_values

    # SHAP bar 图
    plt.figure()
    shap.summary_plot(
        shap_values_to_plot,
        X_shap,
        plot_type="bar",
        max_display=args.topk,
        show=False
    )
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"fig_catboost_shap_bar_top{args.topk}.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(args.outdir, f"fig_catboost_shap_bar_top{args.topk}.pdf"), bbox_inches="tight")
    plt.close()

    # SHAP beeswarm 图
    plt.figure()
    shap.summary_plot(
        shap_values_to_plot,
        X_shap,
        max_display=args.topk,
        show=False
    )
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"fig_catboost_shap_beeswarm_top{args.topk}.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(args.outdir, f"fig_catboost_shap_beeswarm_top{args.topk}.pdf"), bbox_inches="tight")
    plt.close()

    print(f"All outputs saved to: {args.outdir}")


if __name__ == "__main__":
    main()
