#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Final Reviewer 2 Team A/B assignment symmetry analysis

Based on the original CatBoost baseline pipeline.

Design:
1. Reproduce original baseline (10 seeds)
2. Train swap-augmented model (10 seeds)
3. Evaluate prediction invariance:
       |P(Team A win)+P(swapped Team A win)-1|

Important:
- Keep original feature engineering
- Swap all A/B paired variables
- Reverse all *_diff variables automatically
- Reverse H2H Team-A win-rate variables
"""

import os
import argparse
import json
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, log_loss


SEEDS = [41,51,61,71,81,91,101,111,121,131]


# Complete baseline feature set
FEATURE_COLUMNS = [
    "gender","circuit","country","bracket","round","month",

    "teamA_avg_age","teamB_avg_age","age_diff",
    "teamA_avg_hgt","teamB_avg_hgt","hgt_diff",

    "teamA_rank","teamB_rank","rank_diff",

    "teamA_is_qualifier","teamB_is_qualifier",
    "teamA_qualifier_num","teamB_qualifier_num",
    "qualifier_num_diff",

    "one_side_qualifier","both_qualifier",

    "teamA_same_country","teamB_same_country",

    "teamA_is_new_pair","teamB_is_new_pair",

    "teamA_prior_matches","teamB_prior_matches",
    "pair_prior_matches_diff",

    "teamA_prior_win_pct","teamB_prior_win_pct",
    "pair_prior_win_pct_diff",

    "teamA_recent3_win_pct","teamB_recent3_win_pct",
    "recent3_win_pct_diff",

    "teamA_recent5_win_pct","teamB_recent5_win_pct",
    "recent5_win_pct_diff",

    "teamA_recent10_win_pct","teamB_recent10_win_pct",
    "recent10_win_pct_diff",

    "teamA_recent20_win_pct","teamB_recent20_win_pct",
    "recent20_win_pct_diff",

    "teamA_last30_match_count","teamB_last30_match_count",
    "last30_match_count_diff",

    "teamA_last30_win_pct","teamB_last30_win_pct",
    "last30_win_pct_diff",

    "teamA_last90_match_count","teamB_last90_match_count",
    "last90_match_count_diff",

    "teamA_last90_win_pct","teamB_last90_win_pct",
    "last90_win_pct_diff",

    "teamA_last180_match_count","teamB_last180_match_count",
    "last180_match_count_diff",

    "teamA_last180_win_pct","teamB_last180_win_pct",
    "last180_win_pct_diff",

    "teamA_days_since_last_match",
    "teamB_days_since_last_match",
    "days_since_last_match_diff",

    "teamA_avg_player_prior_matches",
    "teamB_avg_player_prior_matches",
    "avg_player_prior_matches_diff",

    "teamA_avg_player_prior_win_pct",
    "teamB_avg_player_prior_win_pct",
    "avg_player_prior_win_pct_diff",

    "teamA_avg_player_recent10_win_pct",
    "teamB_avg_player_recent10_win_pct",
    "avg_player_recent10_win_pct_diff",

    "teamA_avg_player_recent20_win_pct",
    "teamB_avg_player_recent20_win_pct",
    "avg_player_recent20_win_pct_diff",

    "teamA_avg_player_last90_match_count",
    "teamB_avg_player_last90_match_count",
    "avg_player_last90_match_count_diff",

    "teamA_avg_player_last90_win_pct",
    "teamB_avg_player_last90_win_pct",
    "avg_player_last90_win_pct_diff",

    "teamA_avg_player_days_since_last_match",
    "teamB_avg_player_days_since_last_match",
    "avg_player_days_since_last_match_diff",

    "teamA_both_players_new",
    "teamB_both_players_new",

    "h2h_prior_matches",
    "h2h_teamA_win_pct",
    "h2h_recent3_teamA_win_pct",

    "teamA_elo","teamB_elo","elo_diff",
    "teamA_player1_elo","teamA_player2_elo",
    "teamB_player1_elo","teamB_player2_elo",

    "teamA_avg_player_elo",
    "teamB_avg_player_elo",
    "avg_player_elo_diff",

    "teamA_blend_elo",
    "teamB_blend_elo",
    "blend_elo_diff"
]


def split_time(df):
    df=df.copy()
    df["date"]=pd.to_datetime(df["date"])

    return (
        df[df.date<="2017-12-31"],
        df[(df.date>"2017-12-31")&(df.date<="2019-12-31")],
        df[df.date>"2019-12-31"]
    )


def swap_assignment(df):

    out=df.copy()

    cols=list(df.columns)

    for c in cols:
        if c.startswith("teamA_"):
            b=c.replace("teamA_","teamB_")
            if b in cols:
                out[c]=df[b].values

        elif c.startswith("teamB_"):
            a=c.replace("teamB_","teamA_")
            if a in cols:
                out[c]=df[a].values


    # reverse difference features
    for c in cols:
        if c.endswith("_diff"):
            out[c]=-df[c].values


    # H2H direction
    if "h2h_teamA_win_pct" in cols:
        out["h2h_teamA_win_pct"]=1-df["h2h_teamA_win_pct"]

    if "h2h_recent3_teamA_win_pct" in cols:
        out["h2h_recent3_teamA_win_pct"]=1-df["h2h_recent3_teamA_win_pct"]


    # label inversion
    out["y"]=1-df["y"].values

    return out


def preprocess(train,valid,test):

    X1=train[FEATURE_COLUMNS].copy()
    X2=valid[FEATURE_COLUMNS].copy()
    X3=test[FEATURE_COLUMNS].copy()

    cat=[]

    for c in FEATURE_COLUMNS:

        if pd.api.types.is_numeric_dtype(X1[c]):

            med=X1[c].median()
            X1[c]=X1[c].fillna(med)
            X2[c]=X2[c].fillna(med)
            X3[c]=X3[c].fillna(med)

        else:

            cat.append(c)
            mode=X1[c].mode().iloc[0]

            X1[c]=X1[c].fillna(mode).astype(str)
            X2[c]=X2[c].fillna(mode).astype(str)
            X3[c]=X3[c].fillna(mode).astype(str)

    return X1,X2,X3,[X1.columns.get_loc(c) for c in cat]


def fit_model(train,valid,test,seed):

    X1,X2,X3,cat_idx=preprocess(train,valid,test)

    model=CatBoostClassifier(
        iterations=900,
        depth=6,
        learning_rate=0.03,
        l2_leaf_reg=5,
        subsample=0.85,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False
    )

    model.fit(
        X1,
        train.y,
        cat_features=cat_idx,
        eval_set=(X2,valid.y),
        use_best_model=True
    )

    return model.predict_proba(X3)[:,1]


def main():

    parser=argparse.ArgumentParser()
    parser.add_argument("--input",required=True)
    parser.add_argument("--outdir",default="./team_swap_final_output")

    args=parser.parse_args()

    os.makedirs(args.outdir,exist_ok=True)

    df=pd.read_csv(args.input)

    train,valid,test=split_time(df)

    train_swap=swap_assignment(train)
    train_aug=pd.concat([train,train_swap],ignore_index=True)

    baseline=[]
    swap=[]
    symmetry=[]

    for seed in SEEDS:

        p1=fit_model(train,valid,test,seed)

        p2=fit_model(train_aug,valid,test,seed)

        test_swap=swap_assignment(test)

        p3=fit_model(train_aug,valid,test_swap,seed)


        baseline.append({
            "seed":seed,
            "auc":roc_auc_score(test.y,p1),
            "accuracy":accuracy_score(test.y,p1>0.5),
            "f1":f1_score(test.y,p1>0.5),
            "logloss":log_loss(test.y,p1)
        })


        swap.append({
            "seed": seed,
            "auc": roc_auc_score(test.y, p2),
            "accuracy": accuracy_score(test.y, p2 >= 0.5),
            "f1": f1_score(test.y, p2 >= 0.5),
            "logloss": log_loss(test.y, p2)
        })


        symmetry.append({
            "seed":seed,
            "symmetry_error":np.mean(np.abs(p2+p3-1))
        })


    pd.DataFrame(baseline).to_csv(
        args.outdir+"/baseline_metrics_10seeds.csv",
        index=False
    )

    pd.DataFrame(swap).to_csv(
        args.outdir+"/swap_metrics_10seeds.csv",
        index=False
    )

    pd.DataFrame(symmetry).to_csv(
        args.outdir+"/symmetry_error_10seeds.csv",
        index=False
    )

    baseline_df = pd.DataFrame(baseline)
    swap_df = pd.DataFrame(swap)

    summary = pd.DataFrame([
        {
            "Model": "Original CatBoost baseline",
            "AUC_mean": baseline_df["auc"].mean(),
            "AUC_SD": baseline_df["auc"].std(),
            "Accuracy_mean": baseline_df["accuracy"].mean(),
            "Accuracy_SD": baseline_df["accuracy"].std(),
            "F1_mean": baseline_df["f1"].mean(),
            "F1_SD": baseline_df["f1"].std(),
            "LogLoss_mean": baseline_df["logloss"].mean(),
            "LogLoss_SD": baseline_df["logloss"].std()
        },
        {
            "Model": "Team A/B swap-augmented CatBoost",
            "AUC_mean": swap_df["auc"].mean(),
            "AUC_SD": swap_df["auc"].std(),
            "Accuracy_mean": swap_df["accuracy"].mean(),
            "Accuracy_SD": swap_df["accuracy"].std(),
            "F1_mean": swap_df["f1"].mean(),
            "F1_SD": swap_df["f1"].std(),
            "LogLoss_mean": swap_df["logloss"].mean(),
            "LogLoss_SD": swap_df["logloss"].std()
        }
    ])

    summary.to_csv(
        args.outdir+"/supplementary_table_team_swap_summary.csv",
        index=False
    )

    with open(args.outdir+"/run_config.json","w") as f:
        json.dump({
            "seeds":SEEDS,
            "feature_count":len(FEATURE_COLUMNS)
        },f,indent=2)


if __name__=="__main__":
    main()
