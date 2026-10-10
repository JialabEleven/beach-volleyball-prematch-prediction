
Priority 8 Feature-block Ablation
=================================

Purpose
-------
Quantify the contribution of Elo and ranking feature families while keeping
the model, preprocessing, chronological split, and hyperparameters fixed.

Chronological split
-------------------
Train      <= 2017-12-31
Validation > 2017-12-31 and <= 2019-12-31
Test       > 2019-12-31

Feature configurations
----------------------
Full:
    All 104 retained baseline prematch features.

Elo-only:
    Only the 13 Elo-derived variables.

Ranking-only:
    Only teamA_rank, teamB_rank, and rank_diff.

No-Elo:
    Full feature set with all Elo-derived variables removed.

No-ranking:
    Full feature set with the three ranking variables removed.

No-Elo-and-no-ranking:
    Full feature set with both Elo and ranking variables removed.

Metrics
-------
AUC
Accuracy
F1-score
Log-loss

Reporting
---------
Mean ± SD across 10 random seeds.

No bootstrap confidence intervals or DeLong tests are performed here because
this is a controlled feature-ablation sensitivity analysis.

Main files to inspect
---------------------
1. priority8_feature_ablation_summary_formatted.csv
2. priority8_feature_ablation_delta_vs_full.csv
3. fig_priority8_feature_ablation_test_auc.png
4. priority8_feature_sets.csv

Interpretation
--------------
Full vs Elo-only:
    How much predictive information is captured by Elo alone?

Full vs No-Elo:
    What is the marginal contribution of Elo to the complete model?

Full vs Ranking-only:
    How much predictive information is captured by ranking alone?

Full vs No-ranking:
    What is the marginal contribution of ranking within the full model?

No-Elo-and-no-ranking:
    How much discrimination remains in the remaining prematch information?
