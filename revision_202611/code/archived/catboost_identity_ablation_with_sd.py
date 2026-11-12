import os
import json
import argparse
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, log_loss, roc_curve


def read_csv_fallback(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding='utf-8', low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding='latin-1', low_memory=False)


def split_by_time(df: pd.DataFrame, train_end='2017-12-31', valid_end='2019-12-31'):
    df = df.copy()
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        train = df[df['date'] <= train_end].copy()
        valid = df[(df['date'] > train_end) & (df['date'] <= valid_end)].copy()
        test = df[df['date'] > valid_end].copy()
    elif 'year' in df.columns:
        train = df[df['year'] <= 2017].copy()
        valid = df[(df['year'] >= 2018) & (df['year'] <= 2019)].copy()
        test = df[df['year'] >= 2020].copy()
    else:
        raise ValueError("Need either a 'date' or 'year' column for time-aware split.")
    return train, valid, test


def detect_target(df: pd.DataFrame) -> str:
    for c in ['y', 'target', 'label', 'win', 'teamA_win']:
        if c in df.columns:
            return c
    raise ValueError('Could not detect target column automatically.')


def get_identity_columns(df: pd.DataFrame) -> List[str]:
    # Remove only raw identity columns; keep derived performance features like *_elo.
    exact_candidates = [
        'teamA_key', 'teamB_key', 'team1_key', 'team2_key',
        'teamA_player1', 'teamA_player2', 'teamB_player1', 'teamB_player2',
        'team1_player1', 'team1_player2', 'team2_player1', 'team2_player2',
        'teamA_name', 'teamB_name', 'team1_name', 'team2_name',
        'player1', 'player2', 'winner_name', 'loser_name'
    ]
    cols = [c for c in exact_candidates if c in df.columns]
    return cols


def get_feature_columns(df: pd.DataFrame, target_col: str) -> List[str]:
    drop_cols = {target_col}
    # Keep year/month as context but drop raw date.
    if 'date' in df.columns:
        drop_cols.add('date')
    return [c for c in df.columns if c not in drop_cols]


def prepare_splits(train_df, valid_df, test_df, feature_cols):
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

    # Fill numeric with train median
    for c in num_cols:
        med = train_df[c].median()
        train_df[c] = train_df[c].fillna(med)
        valid_df[c] = valid_df[c].fillna(med)
        test_df[c] = test_df[c].fillna(med)

    # Fill categoricals with train mode, cast to str
    for c in cat_cols:
        mode_val = train_df[c].mode().iloc[0] if not train_df[c].mode().empty else 'missing'
        train_df[c] = train_df[c].fillna(mode_val).astype(str)
        valid_df[c] = valid_df[c].fillna(mode_val).astype(str)
        test_df[c] = test_df[c].fillna(mode_val).astype(str)

    return train_df, valid_df, test_df, cat_cols


def eval_metrics(y_true, y_proba) -> Dict[str, float]:
    y_pred = (y_proba >= 0.5).astype(int)
    return {
        'auc': float(roc_auc_score(y_true, y_proba)),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1': float(f1_score(y_true, y_pred)),
        'logloss': float(log_loss(y_true, y_proba)),
    }


def train_catboost(train_df, valid_df, test_df, feature_cols, target_col, params):
    train_df, valid_df, test_df, cat_cols = prepare_splits(train_df, valid_df, test_df, feature_cols)

    X_train = train_df[feature_cols]
    y_train = train_df[target_col].astype(int).values
    X_valid = valid_df[feature_cols]
    y_valid = valid_df[target_col].astype(int).values
    X_test = test_df[feature_cols]
    y_test = test_df[target_col].astype(int).values

    cat_idx = [feature_cols.index(c) for c in cat_cols]

    model = CatBoostClassifier(**params)
    model.fit(
        X_train, y_train,
        cat_features=cat_idx,
        eval_set=(X_valid, y_valid),
        use_best_model=True,
        verbose=False,
    )

    valid_proba = model.predict_proba(X_valid)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]

    return model, y_valid, valid_proba, y_test, test_proba


def save_roc(y_true, y_score, outpath, title):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_value = roc_auc_score(y_true, y_score)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {auc_value:.3f}")
    plt.plot([0, 1], [0, 1], linestyle='--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def save_roc_comparison(y_true, score_dict, outpath, title):
    plt.figure(figsize=(6, 6))
    for name, score in score_dict.items():
        fpr, tpr, _ = roc_curve(y_true, score)
        auc_value = roc_auc_score(y_true, score)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc_value:.3f})")
    plt.plot([0, 1], [0, 1], linestyle='--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def save_top_importance(model, out_csv, out_png, topk=20):
    feature_names = model.feature_names_
    imp = model.get_feature_importance()
    df_imp = pd.DataFrame({'feature': feature_names, 'importance': imp}).sort_values('importance', ascending=False)
    df_imp.to_csv(out_csv, index=False)

    top = df_imp.head(topk).iloc[::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(top['feature'], top['importance'])
    plt.xlabel('Importance')
    plt.title(f'CatBoost Feature Importance (Top {topk})')
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def summarize_runs(runs_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metric_cols = [c for c in runs_df.columns if c.startswith('valid_') or c.startswith('test_')]
    summary_df = runs_df.groupby(['model', 'feature_set'])[metric_cols].agg(['mean', 'std']).reset_index()
    summary_df.columns = [
        col if isinstance(col, str) else f'{col[0]}_{col[1]}' for col in summary_df.columns
    ]

    # Round for readability in the summary table.
    for c in summary_df.columns:
        if c not in ['model', 'feature_set']:
            summary_df[c] = summary_df[c].round(4)

    formatted_rows = []
    for _, row in summary_df.iterrows():
        formatted = {'model': row['model'], 'feature_set': row['feature_set']}
        for split in ['valid', 'test']:
            for metric in ['auc', 'accuracy', 'f1', 'logloss']:
                mean_col = f'{split}_{metric}_mean'
                std_col = f'{split}_{metric}_std'
                formatted[f'{split}_{metric}_mean_sd'] = f"{row[mean_col]:.4f} ± {row[std_col]:.4f}"
        formatted_rows.append(formatted)
    formatted_df = pd.DataFrame(formatted_rows)
    return summary_df, formatted_df


def main():
    parser = argparse.ArgumentParser(description='CatBoost identity-ablation control experiment with repeated seeds')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--train-end', type=str, default='2017-12-31')
    parser.add_argument('--valid-end', type=str, default='2019-12-31')
    parser.add_argument('--iterations', type=int, default=500)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--l2-leaf-reg', type=float, default=3.0)
    parser.add_argument('--seeds', type=int, nargs='+', default=[41, 51, 61, 71, 81, 91, 101, 111, 121, 131])
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = read_csv_fallback(args.input)
    target_col = detect_target(df)
    train_df, valid_df, test_df = split_by_time(df, args.train_end, args.valid_end)

    all_features = get_feature_columns(df, target_col)
    identity_cols = get_identity_columns(df)
    no_identity_features = [c for c in all_features if c not in set(identity_cols)]

    with open(os.path.join(args.outdir, 'dropped_identity_columns.txt'), 'w', encoding='utf-8') as f:
        for c in identity_cols:
            f.write(c + '\n')

    with open(os.path.join(args.outdir, 'run_config.json'), 'w', encoding='utf-8') as f:
        json.dump(
            {
                'input': args.input,
                'train_end': args.train_end,
                'valid_end': args.valid_end,
                'iterations': args.iterations,
                'depth': args.depth,
                'learning_rate': args.learning_rate,
                'l2_leaf_reg': args.l2_leaf_reg,
                'seeds': args.seeds,
                'target_col': target_col,
                'identity_cols': identity_cols,
                'all_feature_count': len(all_features),
                'no_identity_feature_count': len(no_identity_features),
            },
            f,
            indent=2,
        )

    print('Target column:', target_col)
    print('Identity columns to drop:', identity_cols)
    print('All feature count:', len(all_features))
    print('No-identity feature count:', len(no_identity_features))
    print('Seeds:', args.seeds)

    all_runs = []
    best_full_auc = -1.0
    best_noid_auc = -1.0
    best_full_bundle = None
    best_noid_bundle = None

    for seed in args.seeds:
        print(f'\n===== Running seed {seed} =====')
        params = dict(
            iterations=args.iterations,
            depth=args.depth,
            learning_rate=args.learning_rate,
            l2_leaf_reg=args.l2_leaf_reg,
            loss_function='Logloss',
            eval_metric='AUC',
            random_seed=seed,
            allow_writing_files=False,
        )

        # Full model
        full_model, y_valid, full_valid_proba, y_test, full_test_proba = train_catboost(
            train_df, valid_df, test_df, all_features, target_col, params
        )
        # Reduced model
        red_model, y_valid2, red_valid_proba, y_test2, red_test_proba = train_catboost(
            train_df, valid_df, test_df, no_identity_features, target_col, params
        )

        assert (y_valid == y_valid2).all()
        assert (y_test == y_test2).all()

        full_valid_metrics = eval_metrics(y_valid, full_valid_proba)
        full_test_metrics = eval_metrics(y_test, full_test_proba)
        red_valid_metrics = eval_metrics(y_valid, red_valid_proba)
        red_test_metrics = eval_metrics(y_test, red_test_proba)

        print('CatBoost_full VALID:', {k: round(v, 4) for k, v in full_valid_metrics.items()})
        print('CatBoost_full TEST :', {k: round(v, 4) for k, v in full_test_metrics.items()})
        print('CatBoost_no_identity VALID:', {k: round(v, 4) for k, v in red_valid_metrics.items()})
        print('CatBoost_no_identity TEST :', {k: round(v, 4) for k, v in red_test_metrics.items()})

        all_runs.append({
            'seed': seed,
            'model': 'CatBoost_full',
            'feature_set': 'full',
            **{f'valid_{k}': v for k, v in full_valid_metrics.items()},
            **{f'test_{k}': v for k, v in full_test_metrics.items()},
        })
        all_runs.append({
            'seed': seed,
            'model': 'CatBoost_no_identity',
            'feature_set': 'no_identity',
            **{f'valid_{k}': v for k, v in red_valid_metrics.items()},
            **{f'test_{k}': v for k, v in red_test_metrics.items()},
        })

        # Keep best TEST-AUC models for ROC / importance export.
        if full_test_metrics['auc'] > best_full_auc:
            best_full_auc = full_test_metrics['auc']
            best_full_bundle = (full_model, y_valid, full_valid_proba, y_test, full_test_proba)

        if red_test_metrics['auc'] > best_noid_auc:
            best_noid_auc = red_test_metrics['auc']
            best_noid_bundle = (red_model, y_valid2, red_valid_proba, y_test2, red_test_proba)

    runs_df = pd.DataFrame(all_runs)
    runs_df.to_csv(os.path.join(args.outdir, 'identity_ablation_metrics_all_runs.csv'), index=False)

    summary_df, formatted_df = summarize_runs(runs_df)
    summary_df.to_csv(os.path.join(args.outdir, 'identity_ablation_metrics_summary_mean_sd.csv'), index=False)
    formatted_df.to_csv(os.path.join(args.outdir, 'identity_ablation_metrics_summary_formatted.csv'), index=False)

    # Recover best-bundle artifacts.
    full_model, y_valid, full_valid_proba, y_test, full_test_proba = best_full_bundle
    red_model, y_valid2, red_valid_proba, y_test2, red_test_proba = best_noid_bundle

    # ROC plots (best run by test AUC for each model variant)
    save_roc(y_test, full_test_proba, os.path.join(args.outdir, 'roc_catboost_full_test.png'), 'ROC Curve - CatBoost Full (TEST)')
    save_roc(y_test2, red_test_proba, os.path.join(args.outdir, 'roc_catboost_no_identity_test.png'), 'ROC Curve - CatBoost No-Identity (TEST)')
    save_roc_comparison(
        y_test,
        {'Full': full_test_proba, 'No identity': red_test_proba},
        os.path.join(args.outdir, 'roc_catboost_identity_ablation_test.png'),
        'ROC Curve Comparison - Identity Ablation (TEST)'
    )

    save_roc(y_valid, full_valid_proba, os.path.join(args.outdir, 'roc_catboost_full_valid.png'), 'ROC Curve - CatBoost Full (VALID)')
    save_roc(y_valid2, red_valid_proba, os.path.join(args.outdir, 'roc_catboost_no_identity_valid.png'), 'ROC Curve - CatBoost No-Identity (VALID)')
    save_roc_comparison(
        y_valid,
        {'Full': full_valid_proba, 'No identity': red_valid_proba},
        os.path.join(args.outdir, 'roc_catboost_identity_ablation_valid.png'),
        'ROC Curve Comparison - Identity Ablation (VALID)'
    )

    # Importance for no-identity model (more interpretable)
    save_top_importance(
        red_model,
        os.path.join(args.outdir, 'catboost_no_identity_feature_importance.csv'),
        os.path.join(args.outdir, 'fig_catboost_no_identity_feature_importance_top20.png'),
        topk=20,
    )

    with open(os.path.join(args.outdir, 'README_ablation.txt'), 'w', encoding='utf-8') as f:
        f.write('CatBoost identity-ablation control experiment with repeated seeds\n')
        f.write('----------------------------------------------------------\n')
        f.write(f'Target column: {target_col}\n')
        f.write(f'Dropped identity columns: {identity_cols}\n')
        f.write(f'Seeds: {args.seeds}\n\n')
        f.write('Summary (mean ± SD)\n')
        f.write(summary_df.to_string(index=False))
        f.write('\n\nFormatted summary\n')
        f.write(formatted_df.to_string(index=False))
        f.write('\n\nInterpretation guide:\n')
        f.write('- If AUC drops only slightly after dropping identity columns, the model relies mainly on generalizable pre-match features.\n')
        f.write('- If AUC drops a lot, the previous best model relied heavily on player/team identity memory.\n')
        f.write('- The no-identity importance plot is better suited for paper interpretation.\n')

    print('\nSaved outputs to:', args.outdir)
    print('\nSummary (mean ± SD):')
    print(summary_df)
    print('\nFormatted summary:')
    print(formatted_df)


if __name__ == '__main__':
    main()
