Reviewer robustness analysis completed.

DATA CONSISTENCY
- Structural audit: PASS
- Current rows: 85509
- Current years: 2000-2022
- Canonical fingerprint: b244068a639a0781878e75e8b3733d3227fef49dd9d42a23e8cf8bfea474a166
- Exact old-manifest comparison: NO_REFERENCE_MANIFEST
- Previous-result reproduction: Strong consistency with the previously analyzed dataset/model pipeline

ROLLING-ORIGIN DESIGN
W1: train <=2011, validation 2012-2013, test 2014-2015
W2: train <=2013, validation 2014-2015, test 2016-2017
W3: train <=2015, validation 2016-2017, test 2018-2019
W4: train <=2017, validation 2018-2019, test 2020-2022 (original final split)

COVID SENSITIVITY
- Uses mean predictions from W4 only.
- Reports 2020 separately from 2021-2022.
- Does not retrain on pandemic-period outcomes.
- Annual temporal-feature summaries include days_since_last_match, last90_match_count, and recent10_win_pct.

PRIMARY FILES FOR REVISION
- supplementary_table_rolling_origin.csv
- fig_rolling_origin_auc.png / .pdf
- covid_sensitivity_metrics.csv
- annual_temporal_feature_summary.csv
- fig_days_since_last_match_by_year.png / .pdf
- dataset_integrity_audit.json
- previous_result_reproduction_check.csv
