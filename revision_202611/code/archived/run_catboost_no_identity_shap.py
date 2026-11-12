import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, log_loss

IDENTITY_CANDIDATES = [
    'teamA_key', 'teamB_key',
    'teamA_player1', 'teamA_player2', 'teamB_player1', 'teamB_player2',
    'teamA_player_1', 'teamA_player_2', 'teamB_player_1', 'teamB_player_2',
    'player1_name', 'player2_name', 'team1_name', 'team2_name',
    'winner_name', 'loser_name'
]


def detect_target_column(df: pd.DataFrame) -> str:
    candidates = ['target', 'label', 'y', 'outcome', 'result', 'teamA_win', 'team1_win', 'win']
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError('Could not detect target column automatically.')


def detect_drop_columns(df: pd.DataFrame, target_col: str):
    drop_cols = [target_col]
    for c in ['match_id', 'event_id', 'tournament_id', 'date', 'match_date', 'datetime']:
        if c in df.columns:
            drop_cols.append(c)
    return list(dict.fromkeys(drop_cols))


def train_valid_test_split_timeaware(df: pd.DataFrame):
    df = df.copy()
    if 'date' in df.columns:
        df['__sort_date__'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.sort_values('__sort_date__').reset_index(drop=True)
    elif 'match_date' in df.columns:
        df['__sort_date__'] = pd.to_datetime(df['match_date'], errors='coerce')
        df = df.sort_values('__sort_date__').reset_index(drop=True)
    elif 'datetime' in df.columns:
        df['__sort_date__'] = pd.to_datetime(df['datetime'], errors='coerce')
        df = df.sort_values('__sort_date__').reset_index(drop=True)
    elif 'year' in df.columns:
        df = df.sort_values('year').reset_index(drop=True)
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
        mode_val = train_df[c].mode().iloc[0] if train_df[c].mode().shape[0] > 0 else 'missing'
        train_df[c] = train_df[c].fillna(mode_val).astype(str)
        valid_df[c] = valid_df[c].fillna(mode_val).astype(str)
        test_df[c] = test_df[c].fillna(mode_val).astype(str)

    return train_df, valid_df, test_df, cat_cols, num_cols


def save_hbar(df_imp: pd.DataFrame, out_png: str, out_pdf: str, title: str, topk: int = 20):
    top_imp = df_imp.head(topk).iloc[::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(top_imp['feature'], top_imp['importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--topk', type=int, default=20)
    parser.add_argument('--sample-size', type=int, default=2000)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    try:
        df = pd.read_csv(args.input, encoding='utf-8', low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(args.input, encoding='latin-1', low_memory=False)

    target_col = detect_target_column(df)
    identity_cols = [c for c in IDENTITY_CANDIDATES if c in df.columns]
    base_drop_cols = detect_drop_columns(df, target_col)
    feature_cols = [c for c in df.columns if c not in base_drop_cols + identity_cols]

    with open(os.path.join(args.outdir, 'dropped_identity_columns.txt'), 'w', encoding='utf-8') as f:
        for c in identity_cols:
            f.write(c + '\n')

    with open(os.path.join(args.outdir, 'model_definition.txt'), 'w', encoding='utf-8') as f:
        f.write('Main model: CatBoost_no_identity\n')
        f.write(f'Target column: {target_col}\n')
        f.write(f'Number of features after dropping identity columns: {len(feature_cols)}\n')

    train_df, valid_df, test_df = train_valid_test_split_timeaware(df)
    train_df, valid_df, test_df, cat_cols, num_cols = fill_missing(train_df, valid_df, test_df, feature_cols)

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
        loss_function='Logloss',
        eval_metric='AUC',
        verbose=100,
        random_seed=42,
    )

    model.fit(
        X_train, y_train,
        cat_features=cat_feature_indices,
        eval_set=(X_valid, y_valid),
        use_best_model=True,
    )

    valid_proba = model.predict_proba(X_valid)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]
    valid_pred = (valid_proba >= 0.5).astype(int)
    test_pred = (test_proba >= 0.5).astype(int)

    metrics = pd.DataFrame([
        {
            'model': 'CatBoost_no_identity',
            'valid_auc': roc_auc_score(y_valid, valid_proba),
            'valid_accuracy': accuracy_score(y_valid, valid_pred),
            'valid_f1': f1_score(y_valid, valid_pred),
            'valid_logloss': log_loss(y_valid, valid_proba),
            'test_auc': roc_auc_score(y_test, test_proba),
            'test_accuracy': accuracy_score(y_test, test_pred),
            'test_f1': f1_score(y_test, test_pred),
            'test_logloss': log_loss(y_test, test_proba),
        }
    ])
    metrics.to_csv(os.path.join(args.outdir, 'catboost_no_identity_metrics.csv'), index=False)

    importances = model.get_feature_importance()
    imp_df = pd.DataFrame({'feature': feature_cols, 'importance': importances}).sort_values('importance', ascending=False).reset_index(drop=True)
    imp_df.to_csv(os.path.join(args.outdir, 'catboost_no_identity_feature_importance.csv'), index=False)
    save_hbar(
        imp_df,
        os.path.join(args.outdir, f'fig_catboost_no_identity_feature_importance_top{args.topk}.png'),
        os.path.join(args.outdir, f'fig_catboost_no_identity_feature_importance_top{args.topk}.pdf'),
        f'CatBoost_no_identity Feature Importance (Top {args.topk})',
        topk=args.topk,
    )

    if len(X_test) > args.sample_size:
        X_shap = X_test.sample(args.sample_size, random_state=42)
    else:
        X_shap = X_test.copy()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)
    if isinstance(shap_values, list):
        shap_values_to_plot = shap_values[1]
    else:
        shap_values_to_plot = shap_values

    plt.figure()
    shap.summary_plot(shap_values_to_plot, X_shap, plot_type='bar', max_display=args.topk, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f'fig_catboost_no_identity_shap_bar_top{args.topk}.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(args.outdir, f'fig_catboost_no_identity_shap_bar_top{args.topk}.pdf'), bbox_inches='tight')
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values_to_plot, X_shap, max_display=args.topk, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f'fig_catboost_no_identity_shap_beeswarm_top{args.topk}.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(args.outdir, f'fig_catboost_no_identity_shap_beeswarm_top{args.topk}.pdf'), bbox_inches='tight')
    plt.close()

    print('Done! Outputs saved to:', args.outdir)
    print(metrics.to_string(index=False))


if __name__ == '__main__':
    main()
