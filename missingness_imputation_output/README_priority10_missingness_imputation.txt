Priority 10: Missingness + imputation sensitivity analysis

Input:
/data2/zhangyiyi/beach_ai/final_catboost_v1/data/prematch_features_v3.csv

Main missingness outputs:
- supplementary_table_missingness_all_features.csv
- supplementary_table_missingness_top20.csv
- key_predictor_missingness_elo_rank_qualifier.csv
- key_predictor_missingness_by_year.csv
- missingness_summary_by_feature_block.csv
- fig_missingness_top20.png/pdf

Main imputation-sensitivity outputs:
- imputation_sensitivity_metrics_all_runs.csv
- imputation_sensitivity_summary_mean_sd.csv
- supplementary_table_imputation_sensitivity.csv
- imputation_sensitivity_delta_vs_baseline.csv
- fig_imputation_sensitivity_test_auc.png/pdf
- imputation_sensitivity_test_predictions_mean_10seeds.csv

Reproducibility:
- dataset_integrity_audit.json
- baseline_reproduction_check.csv

Interpretation:
If performance remains close between the primary median/mode strategy and the
native-numeric-missing strategy, the main conclusion is not materially dependent
on median imputation. The third strategy tests whether explicitly representing
missingness changes this conclusion.
