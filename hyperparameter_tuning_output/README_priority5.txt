Priority 5 standardized time-aware tuning

"
                "Internal folds: <=2011 -> 2012-13; <=2013 -> 2014-15; <=2015 -> 2016-17.
"
                "Selection metric: mean internal-validation AUC.
"
                "External validation: 2018-2019. Test: >2019. Neither is used for tuning.

"
                "Inspect first:
"
                "1. selected_hyperparameters.csv
"
                "2. final_tuned_metrics_summary_formatted.csv
"
                "3. comparison_original_vs_tuned_test.csv
"
                "4. ranking_original_vs_tuned.csv

"
                "If ranking is unchanged and shifts are small, report as sensitivity analysis.
"
                "If ranking changes materially, reconsider Table 1 and rerun downstream bootstrap/DeLong with tuned predictions.
