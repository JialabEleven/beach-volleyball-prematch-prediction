
import argparse
from pathlib import Path
import os

import matplotlib.pyplot as plt
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, f1_score, log_loss, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception as e:
    HAS_XGBOOST = False
    XGBOOST_IMPORT_ERROR = repr(e)

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except Exception as e:
    HAS_CATBOOST = False
    CATBOOST_IMPORT_ERROR = repr(e)


FEATURE_COLUMNS = [
    # context
    "gender", "circuit", "country", "bracket", "round", "month",

    # static
    "teamA_avg_age", "teamB_avg_age", "age_diff",
    "teamA_avg_hgt", "teamB_avg_hgt", "hgt_diff",
    "teamA_rank", "teamB_rank", "rank_diff",
    "teamA_is_qualifier", "teamB_is_qualifier",
    "teamA_qualifier_num", "teamB_qualifier_num", "qualifier_num_diff",
    "one_side_qualifier", "both_qualifier",
    "teamA_same_country", "teamB_same_country",

    # pair/team history
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

    # player summary histories
    "teamA_avg_player_prior_matches", "teamB_avg_player_prior_matches", "avg_player_prior_matches_diff",
    "teamA_avg_player_prior_win_pct", "teamB_avg_player_prior_win_pct", "avg_player_prior_win_pct_diff",
    "teamA_avg_player_recent10_win_pct", "teamB_avg_player_recent10_win_pct", "avg_player_recent10_win_pct_diff",
    "teamA_avg_player_recent20_win_pct", "teamB_avg_player_recent20_win_pct", "avg_player_recent20_win_pct_diff",
    "teamA_avg_player_last90_match_count", "teamB_avg_player_last90_match_count", "avg_player_last90_match_count_diff",
    "teamA_avg_player_last90_win_pct", "teamB_avg_player_last90_win_pct", "avg_player_last90_win_pct_diff",
    "teamA_avg_player_days_since_last_match", "teamB_avg_player_days_since_last_match", "avg_player_days_since_last_match_diff",
    "teamA_both_players_new", "teamB_both_players_new",

    # h2h
    "h2h_prior_matches", "h2h_teamA_win_pct", "h2h_recent3_teamA_win_pct",

    # elo
    "teamA_elo", "teamB_elo", "elo_diff",
    "teamA_player1_elo", "teamA_player2_elo", "teamB_player1_elo", "teamB_player2_elo",
    "teamA_avg_player_elo", "teamB_avg_player_elo", "avg_player_elo_diff",
    "teamA_blend_elo", "teamB_blend_elo", "blend_elo_diff",
]


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values(["date", "match_num"]).reset_index(drop=True)


def split_by_time(df: pd.DataFrame, train_end: str = "2017-12-31", valid_end: str = "2019-12-31"):
    train = df[df["date"] <= train_end].copy()
    valid = df[(df["date"] > train_end) & (df["date"] <= valid_end)].copy()
    test = df[df["date"] > valid_end].copy()
    return train, valid, test


def make_xy(df: pd.DataFrame):
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    X = df[FEATURE_COLUMNS].copy()
    y = df["y"].astype(int).values
    return X, y


def get_num_cat_cols(X: pd.DataFrame):
    num_cols = [c for c in X.columns if is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]
    return num_cols, cat_cols


def make_preprocessor(X: pd.DataFrame):
    num_cols, cat_cols = get_num_cat_cols(X)

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )
    return preprocessor


def prepare_catboost_inputs(X_train: pd.DataFrame, X_valid: pd.DataFrame, X_test: pd.DataFrame):
    num_cols, cat_cols = get_num_cat_cols(X_train)

    X_train_cb = X_train.copy()
    X_valid_cb = X_valid.copy()
    X_test_cb = X_test.copy()

    medians = X_train_cb[num_cols].median(numeric_only=True)
    X_train_cb[num_cols] = X_train_cb[num_cols].fillna(medians)
    X_valid_cb[num_cols] = X_valid_cb[num_cols].fillna(medians)
    X_test_cb[num_cols] = X_test_cb[num_cols].fillna(medians)

    for col in cat_cols:
        mode_series = X_train_cb[col].mode(dropna=True)
        fill_value = mode_series.iloc[0] if not mode_series.empty else "missing"
        X_train_cb[col] = X_train_cb[col].fillna(fill_value).astype(str)
        X_valid_cb[col] = X_valid_cb[col].fillna(fill_value).astype(str)
        X_test_cb[col] = X_test_cb[col].fillna(fill_value).astype(str)

    cat_feature_indices = [X_train_cb.columns.get_loc(c) for c in cat_cols]
    return X_train_cb, X_valid_cb, X_test_cb, cat_feature_indices


def save_single_roc_curve(y_true, y_score, model_name: str, split_name: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name} ({split_name.upper()})")
    plt.legend(loc="lower right")
    plt.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")
    plt.savefig(os.path.join(out_dir, f"roc_{safe_name}_{split_name}.png"), dpi=300)
    plt.savefig(os.path.join(out_dir, f"roc_{safe_name}_{split_name}.pdf"))
    plt.close()


def save_combined_roc_curve(curve_dict, split_name: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(6, 6))
    for model_name, (y_true, y_score) in curve_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", lw=1, label="Random chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {split_name.upper()} set")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"roc_comparison_{split_name}.png"), dpi=300)
    plt.savefig(os.path.join(out_dir, f"roc_comparison_{split_name}.pdf"))
    plt.close()


def evaluate_predictions(name, y_valid, valid_prob, y_test, test_prob, roc_dir: str):
    outputs = {}
    for split_name, y, prob in [("valid", y_valid, valid_prob), ("test", y_test, test_prob)]:
        pred = (prob >= 0.5).astype(int)
        outputs[split_name] = {
            "auc": roc_auc_score(y, prob),
            "accuracy": accuracy_score(y, pred),
            "f1": f1_score(y, pred),
            "logloss": log_loss(y, prob),
            "y_true": y,
            "y_score": prob,
        }
        save_single_roc_curve(y, prob, name, split_name, roc_dir)

    print(f"\n=== {name} ===")
    for split_name, metrics in outputs.items():
        printable = {k: round(v, 4) for k, v in metrics.items() if k not in ["y_true", "y_score"]}
        print(split_name.upper(), printable)
    return outputs


def evaluate_sklearn_model(name, model, X_train, y_train, X_valid, y_valid, X_test, y_test, roc_dir: str):
    model.fit(X_train, y_train)
    valid_prob = model.predict_proba(X_valid)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]
    return evaluate_predictions(name, y_valid, valid_prob, y_test, test_prob, roc_dir)


def evaluate_catboost_model(name, model, X_train, y_train, X_valid, y_valid, X_test, y_test, cat_feature_indices, roc_dir: str):
    model.fit(
        X_train,
        y_train,
        cat_features=cat_feature_indices,
        eval_set=(X_valid, y_valid),
        verbose=False,
    )
    valid_prob = model.predict_proba(X_valid)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]
    return evaluate_predictions(name, y_valid, valid_prob, y_test, test_prob, roc_dir)


def save_metrics_csv(results_dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for model_name, split_dict in results_dict.items():
        row = {"model": model_name}
        for split_name, metrics in split_dict.items():
            row[f"{split_name}_auc"] = metrics["auc"]
            row[f"{split_name}_accuracy"] = metrics["accuracy"]
            row[f"{split_name}_f1"] = metrics["f1"]
            row[f"{split_name}_logloss"] = metrics["logloss"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "model_results.csv"), index=False)


def main():
    parser = argparse.ArgumentParser(description="Train baseline models with enriched dynamic features")
    parser.add_argument("--input", required=True, help="Path to prematch_features.csv")
    parser.add_argument("--outdir", default="./legacy_project/results", help="Directory to save metrics and ROC figures")
    args = parser.parse_args()

    results_dir = args.outdir
    roc_dir = os.path.join(results_dir, "roc_curves")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(roc_dir, exist_ok=True)

    df = load_data(args.input)
    train, valid, test = split_by_time(df)
    X_train, y_train = make_xy(train)
    X_valid, y_valid = make_xy(valid)
    X_test, y_test = make_xy(test)

    preprocessor = make_preprocessor(X_train)
    all_results = {}

    logreg = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1200)),
    ])
    all_results["Logistic Regression"] = evaluate_sklearn_model(
        "Logistic Regression", logreg, X_train, y_train, X_valid, y_valid, X_test, y_test, roc_dir
    )

    rf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )),
    ])
    all_results["Random Forest"] = evaluate_sklearn_model(
        "Random Forest", rf, X_train, y_train, X_valid, y_valid, X_test, y_test, roc_dir
    )

    if HAS_XGBOOST:
        xgb = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", XGBClassifier(
                n_estimators=800,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.85,
                colsample_bytree=0.85,
                min_child_weight=3,
                reg_alpha=0.0,
                reg_lambda=2.0,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            )),
        ])
        all_results["XGBoost"] = evaluate_sklearn_model(
            "XGBoost", xgb, X_train, y_train, X_valid, y_valid, X_test, y_test, roc_dir
        )
    else:
        print("\nXGBoost not installed or import failed. Skipping XGBoost model.")
        if 'XGBOOST_IMPORT_ERROR' in globals():
            print(f"XGBoost import error: {XGBOOST_IMPORT_ERROR}")

    if HAS_CATBOOST:
        X_train_cb, X_valid_cb, X_test_cb, cat_feature_indices = prepare_catboost_inputs(X_train, X_valid, X_test)
        cat = CatBoostClassifier(
            iterations=900,
            depth=6,
            learning_rate=0.03,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=42,
            l2_leaf_reg=5,
            subsample=0.85,
            allow_writing_files=False,
            verbose=False,
        )
        all_results["CatBoost"] = evaluate_catboost_model(
            "CatBoost", cat, X_train_cb, y_train, X_valid_cb, y_valid, X_test_cb, y_test, cat_feature_indices, roc_dir
        )
    else:
        print("\nCatBoost not installed or import failed. Skipping CatBoost model.")
        if 'CATBOOST_IMPORT_ERROR' in globals():
            print(f"CatBoost import error: {CATBOOST_IMPORT_ERROR}")

    save_metrics_csv(all_results, results_dir)

    valid_curve_dict = {}
    test_curve_dict = {}
    for model_name, split_dict in all_results.items():
        valid_curve_dict[model_name] = (split_dict["valid"]["y_true"], split_dict["valid"]["y_score"])
        test_curve_dict[model_name] = (split_dict["test"]["y_true"], split_dict["test"]["y_score"])

    save_combined_roc_curve(valid_curve_dict, "valid", roc_dir)
    save_combined_roc_curve(test_curve_dict, "test", roc_dir)

    print(f"\nSaved metrics CSV to: {os.path.join(results_dir, 'model_results.csv')}")
    print(f"Saved ROC curves to: {roc_dir}")


if __name__ == "__main__":
    main()
