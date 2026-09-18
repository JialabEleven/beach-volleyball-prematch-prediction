
Priority 6: Global-model subgroup evaluation
=============================================

Design
------
One global retained-baseline CatBoost model is trained per random seed.

Time split:
Train      : <= 2017-12-31
Validation : > 2017-12-31 and <= 2019-12-31
Test       : > 2019-12-31

For each seed:
1. Train CatBoost once on the complete training set.
2. Predict the complete validation set.
3. Predict the complete test set.
4. Only after prediction, stratify observations into Men, Women, AVP, and FIVB.
5. Calculate AUC, Accuracy, F1-score, and Log-loss within each subgroup.

No subgroup-specific retraining is performed.

Primary files to inspect
------------------------
1. priority6_global_subgroup_metrics_formatted.csv
2. priority6_global_subgroup_metrics_summary_mean_sd.csv
3. priority6_global_overall_metrics_summary.csv
4. priority6_old_vs_global_subgroup_auc.csv
5. priority6_global_subgroup_sample_sizes.csv

Statistical reporting
---------------------
Mean ± SD across random seeds.
No bootstrap 95% CI is added in this Priority 6 analysis.
No formal between-subgroup significance test is performed.
