Subgroup analysis for the retained baseline CatBoost model
Input: /data2/zhangyiyi/beach_ai/final_catboost_v1/data/prematch_features_v3.csv
Seeds: [41, 51, 61, 71, 81, 91, 101, 111, 121, 131]
Time split: train <= 2017-12-31, valid <= 2019-12-31, test > 2019-12-31
Target column: y
Feature count: 104

Generated files:
- subgroup_metrics_all_runs.csv
- subgroup_metrics_summary_mean_sd.csv
- subgroup_metrics_summary_formatted.csv
- roc_*_valid/test.(png/pdf)
- fig_subgroup_test_auc_summary.(png/pdf)
- feature_importance_*.csv
- fig_feature_importance_*_top20.(png/pdf)
