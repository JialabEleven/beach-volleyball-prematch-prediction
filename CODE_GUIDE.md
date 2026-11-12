# Code guide

## Scope and layout

The current reproducibility entry point is `revision_202611/reproduce.py`. Run commands from `revision_202611`, not from an older result folder. Root-level legacy scripts and outputs remain available for provenance. The current revision adds data provenance, chronological checks, a controlled primary-model reproduction, tournament freezing, and paired cluster-bootstrap sensitivity analyses.

This is a research reproduction workflow rather than a deployed match-prediction service. Full feature construction and model fitting may be lengthy; bootstrap-only reproduction uses supplied fixed predictions and avoids retraining.

## Environment

Python 3.11 is recommended. The controlled core analyses used Python 3.11.7 on Linux CPU. Exact package versions are in `requirements-core.txt`; historical reported environments and analysis-specific parameters are retained in `provenance/` and run manifests.

Linux setup:

```bash
git clone https://github.com/JialabEleven/beach-volleyball-prematch-prediction.git
cd beach-volleyball-prematch-prediction/revision_202611
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-core.txt
export PYTHONNOUSERSITE=1
python reproduce.py --help
```

For Windows, use `py -3.11 -m venv .venv` and the environment's `.venv\Scripts\python.exe` for subsequent Python commands. The complete workflow was validated primarily on Linux; native Windows equivalence is not asserted. Computation can be run on a Linux server through ordinary SSH.

`requirements-full-analysis.txt` adds the reported XGBoost version for the optional four-model comparison. Installing the core requirements alone does not establish a recovered historical SHAP environment. No fresh clean-environment installation or full historical interpretation refit is claimed by this documentation.

## Data acquisition and verification

```bash
python reproduce.py download --raw data/full_archive.csv
python reproduce.py verify --raw data/full_archive.csv --out outputs/cohort
python reproduce.py tests
```

Source repository: https://github.com/BigTimeStats/beach-volleyball

Fixed commit: `0fa423c5704de46a8cf0d9b8b542c694c0ae2276`.

Source file: `data/full_archive/full_archive.csv`.

Expected SHA256: `b1a4ce4b90cb76fef5e47b4037f3aa86d8f43e57990c185187df0466f4a14e3d`.

The fixed version was identified retrospectively; the original author download date is unknown. All 85,509 raw records were retained. Training/validation/test counts are 65,057/12,369/8,083, corresponding to source years through 2017, 2018–2019 and 2020–2022. The archive includes 719/3,158/4,206 records for 2020/2021/2022. It is not asserted to be exhaustive coverage of every real-world match.

Use the complete archive rather than concatenating yearly files. Preserve the supplied name-encoding and canonical team-assignment conventions; a new name-repair policy changes identities and histories. The downloader verifies the checksum and refuses a mismatching existing file. Source availability or API rate limits may affect downloading; an already downloaded matching file can be supplied using the same `--raw` path.

## Build original and frozen features

```bash
python reproduce.py build --raw data/full_archive.csv --out outputs/features
```

Outputs include `original_features.csv`, `frozen0_features.csv` and `frozen30_features.csv`, with source-row and source-defined-event identifiers. `--original-only` restricts construction to the original representation.

The original model has 104 derived predictors. Events combine circuit, tournament, country, year, source date and gender. Frozen histories admit strictly earlier source dates, exclude all same-source-date events and remain fixed within the target event. The buffered condition requires source date + 30 days < target source date. Current-record contextual predictors remain. The without-date scenario removes names containing `days_since`, `last30`, `last90` or `last180`, leaving 74 predictors; month and recent-N features remain.

Numeric medians and categorical modes are estimated from training data only. Fully missing numeric training columns retain NaN medians and are handled natively by CatBoost, including the buffered last30 win-rate columns.

## Refit the matched CatBoost scenarios

```bash
python reproduce.py train-r1 --features outputs/features/original_features.csv --scenario original --out outputs/original48 --threads 48
python reproduce.py train-r1 --features outputs/features/frozen0_features.csv --scenario frozen --out outputs/frozen48 --threads 48
python reproduce.py train-r1 --features outputs/features/frozen0_features.csv --scenario frozen_without_date_proxies --no-date-proxies --out outputs/without_date48 --threads 48
python reproduce.py train-r1 --features outputs/features/frozen30_features.csv --scenario buffer30 --out outputs/buffer48 --threads 48
```

Each command uses ten matched training runs, 900 maximum iterations, depth 6, learning rate 0.03, L2 regularization 5, subsample 0.85, Logloss loss and validation AUC for iteration selection. The CatBoost default `has_time=False` applies to these fits; do not transfer the `has_time=True` setting from the separate identity analysis. Changing threads can change numeric training paths and selected iterations; use 48 threads for comparison with the controlled reproduction.

Outputs include model files, run configuration, per-run metrics, saved validation/test predictions and mean/SD summaries. Consult `python reproduce.py train-r1 --help`. An optional `--smoke` run uses only a small subset and a short fit; it is an execution check, not a manuscript result.

Expected mean test AUC ± SD:

| Scenario | AUC |
|---|---|
| Original reproduced | 0.812574 ± 0.000691 |
| Frozen | 0.804558 ± 0.000559 |
| Frozen without date proxies | 0.805716 ± 0.001192 |
| Frozen with 30-day buffer | 0.799493 ± 0.000645 |

Training/test configurations were not selected by optimizing these test results. Full fitted models are not bundled in this compact update; the scripts save models when run.

For the optional full four-model comparison:

```bash
python -m pip install -r requirements-full-analysis.txt
python reproduce.py train-main --features outputs/features/original_features.csv --out outputs/main48 --threads 48
```

The verified exact primary-metric reproduction concerns CatBoost. It does not certify a fresh identical rerun of all other model families or every historical analysis.

## Reproduce S18 and S19 uncertainty without training

```bash
python code/reproduce_table_s18.py --out outputs/table_s18
python code/paired_cluster_bootstrap_48threads.py --out outputs/table_s19_bootstrap
```

S18 compares fixed archived CatBoost and XGBoost probabilities. S19 compares fixed updated original and frozen-history probabilities. Each uses 5,000 paired cluster resamples under 84 site-year groups and 99 groups additionally distinguishing source dates. The main definition pools genders and repeat stops sharing the same circuit/tournament/country/year identifier. These are source-based groups, not certified independent real-world tournaments.

Expected S18 CatBoost-minus-XGBoost AUC difference is 0.006125, with 95% intervals 0.004282–0.008070 and 0.004259–0.008123. Expected S19 original-minus-frozen difference is 0.008051, with intervals 0.005192–0.011053 and 0.005182–0.010975.

These are AUC differences of averaged probabilities, not differences of mean per-run AUCs. S18 uses a separately archived inference run; its settings must not be relabeled as the matched primary/temporal run. Exact numerical seeds remain in code and configuration records. Older principal-model predictions lack unique match IDs; metadata were aligned from archived loader order with complete date/label sequence checks, leaving residual identity-mapping uncertainty. Updated temporal predictions retain raw-row identifiers. Neither bootstrap includes uncertainty from retraining or fully accounts for players recurring across groups.

`code/paired_cluster_bootstrap.py` is retained as the legacy combined analysis and also emits superseded four-thread temporal comparisons. Use the two dedicated commands above for current S18/S19, not its legacy temporal rows.

## Supplementary tables and evidence

- `tables/submission_tables/`: standalone S16, S18 and S19 workbooks; S16 includes all 110 feature-level rows and its group summary. S16 is an original-representation interpretation analysis, not frozen-history SHAP.
- `tables/year_counts.csv`, `tables/year_circuit_gender_counts.csv`, `tables/cohort_flow.csv`: cohort and annual coverage.
- `tables/targeted_event_evidence.csv`: targeted date checks, including unresolved cases; not an exhaustive chronology audit.
- `provenance/dataset_manifest.json`: source version, checksums and acquisition limitations.
- `reference_results/main_table_reproduction/`: controlled primary-model metrics.
- `reference_results/r1_48threads/`: current temporal summaries, fixed predictions and bootstrap references.

Other original tables and figures map to the corresponding legacy result folders already present in the repository. Their environments and configurations differ by analysis; they were not all refitted under frozen histories.

## Validation and limitations

Run `python verify_package.py` inside the revision directory to check its manifest. The manifest records file integrity, not scientific validity or completion of all model fits. `VALIDATION_REPORT.md` describes earlier v2 tests; `UPLOAD_VALIDATION.md` records checks on this upload preparation.

Source dates are not certified match times; earlier event starts do not prove completion; earlier eligible Elo updates retain ordering uncertainty. A 30-day buffer is an assumption. Performance decline may reflect both removing questionable histories and excluding legitimate within-event information. SHAP is noncausal. No guarantee of exact cross-platform numeric equivalence or editor acceptance is made.

After uploading, use an actual Git commit or release tag to identify the code version cited in the manuscript. The upstream data commit is a different identifier. A tag is not a DOI; claim a DOI only after a real archive deposit has issued one.

# Historical analyses and complete repository map

## Which workflow should I choose

1. To inspect published results without fitting: use the authoritative result map below.
2. To reproduce the current raw-data provenance, primary CatBoost and frozen-history analysis: follow the current-revision workflow above.
3. To rerun an older supplementary experiment: use the root-level script and explicit input/output paths below. These are separate historical configurations, not frozen-history results.
4. To reproduce uncertainty without training: use the dedicated current S18/S19 scripts above, rather than rerunning every historical model.

The following root-level command examples assume the working directory is the **repository root** and that the revision environment has already been activated. Commands in the earlier current-revision sections instead assume the working directory is **revision_202611**. Keep these locations distinct.

## Data inputs for historical scripts

Most historical model scripts read the published processed table `data/prematch_features_v3.csv`, not raw full_archive.csv. It contains the original derived representation, metadata and raw identity fields; each experiment selects its intended predictor set. Do not point these scripts at raw data, a prediction CSV or an arbitrarily modified frozen table.

The current builder emits `revision_202611/outputs/features/original_features.csv`, plus frozen variants with explicit row/event metadata. Its portable current trainers select their feature list. When adapting any legacy loader to these newer files, verify target coding, row ordering, predictor selection and exclusion of metadata before claiming equivalence. The historical published processed input remains a useful fixed reference.

The public root scripts contain original absolute server paths as defaults. Always pass the explicit flags shown below. Creating a root directory with a similar name does not make those original defaults portable.

## Table and figure map

| Output | Historical script or current entry point | Retained results |
|---|---|---|
| Table 1; Fig. 2 | Current `revision_202611/reproduce.py train-main`; original training source retained in `code/archived/` | `baseline_with_sd_results_revise/`; controlled CatBoost evidence in `revision_202611/reference_results/main_table_reproduction/` |
| Table 2 | `catboost_identity_leakproof_has_time.py` | `identity_leakproof_has_time_output/` |
| Table 3; Fig. 6 | `global_model_subgroup_evaluation.py` | `global_subgroup_output/` |
| S1 | Current feature builder and the predictor list in `revision_202611/code/temporal/` | Complete manuscript feature dictionary supplied separately; predictor inventory and source construction records in the revision package |
| S2–S3 | `missingness_imputation.py` | `missingness_imputation_output/` |
| S4–S5 | `timeaware_hyperparameter_tuning.py` | `hyperparameter_tuning_output/` |
| S6 | `train_team_swap_symmetry.py` | Published swap results are retained with the original project outputs; new runs use the explicit output folder below |
| S7 | `trivial_favorite_baseline.py` | `trivial_favorite_baseline_output/` |
| S8–S9 | `bootstrap_delong.py` | `auc_uncertainty_delong_output/` |
| S10–S11; Fig. 3 | `calibration_analysis_final.py` | `calibration_output/` |
| S12; feature-ablation figure | `feature_block_ablation.py` | `feature_ablation_output/` |
| S13–S14 | `rolling_origin_covid_minimal.py` | `rolling_origin_covid_output/` |
| S15 | `pr_auc_confusion_threshold.py` | `pr_auc_confusion_output/` |
| S16 and continuation | `identity_augmented_shap.py` | `identity_augmented_explain_output/`; current complete workbook in `revision_202611/tables/submission_tables/` |
| S17 | Alternative Elo builder and interpretation scripts in `revision_202611/code/archived/revised_code/` | `player_elo_sensitivity/` |
| S18 | `revision_202611/code/reproduce_table_s18.py` | Current paired principal-model bootstrap references |
| S19 | Current `reproduce.py train-r1` and `paired_cluster_bootstrap_48threads.py` | `revision_202611/reference_results/r1_48threads/` |

`paper_outputs/`, `ablation_outputs/`, `identity_ablation_with_sd_results_aligned/`, `subgroup_outputs/` and `subgroup_outputs_with_sd/` include earlier exports or analysis variants. They are retained for traceability, not interchangeable sources for current manuscript numbers. In particular, older subgroup-specific training is different from the globally trained model stratified after prediction used for Table 3.

## Historical execution sequence and examples

### A Identity comparison and attribution

```bash
python catboost_identity_leakproof_has_time.py --input data/prematch_features_v3.csv --outdir revision_202611/outputs/legacy_identity
python identity_augmented_shap.py --input data/prematch_features_v3.csv --outdir revision_202611/outputs/legacy_identity_shap --shap-sample-size 2000
```

The first command compares 104 derived predictors against the same representation plus six identifiers under matched `has_time=True` processing. It saves per-run metrics, summaries and averaged test predictions. The second independently fits the identity-augmented interpretation models and produces all-feature importance/SHAP and group shares; it does not simply read the first command's fitted model. Expected outputs include `identity_augmented_feature_importance_and_shap_all_features.csv` and `identity_vs_derived_attribution_share.csv`.

The word `leakproof` in historical filenames is a legacy label. `has_time=True` preserves supplied record order in categorical processing; it does not verify true match order or remove every possible temporal contamination.

### B Global subgroup evaluation, then calibration

```bash
python global_model_subgroup_evaluation.py --input data/prematch_features_v3.csv --outdir revision_202611/outputs/legacy_global
python calibration_analysis_final.py --valid-predictions revision_202611/outputs/legacy_global/priority6_global_valid_predictions_mean_10seeds.csv --test-predictions revision_202611/outputs/legacy_global/priority6_global_test_predictions_mean_10seeds.csv --outdir revision_202611/outputs/legacy_calibration
```

The global script fits one model per run on the full training set, predicts all observations, then calculates gender/circuit strata. It does not train one model per subgroup. Calibration reads the two saved averaged-prediction files, fits recalibration on validation predictions only and evaluates the test predictions. File schemas must match the script; do not substitute an unrelated CSV merely because it contains probabilities. Calibration outputs include performance tables, binned curve data and figures.

### C Observation-level uncertainty, then PR/threshold performance

```bash
python bootstrap_delong.py --input data/prematch_features_v3.csv --identity-predictions revision_202611/outputs/legacy_identity/test_predictions_mean_10seeds.csv --outdir revision_202611/outputs/legacy_inference
python pr_auc_confusion_threshold.py --predictions revision_202611/outputs/legacy_inference/baseline_test_predictions_mean_10seeds.csv --prob-col catboost_probability_mean_10seeds --threshold 0.50 --outdir revision_202611/outputs/legacy_pr
```

The inference script fits models to obtain its own archived-run equivalent and computes observation-level bootstrap/DeLong results. It may skip the identity comparison if its prediction input is absent, so provide the file explicitly when reproducing S9. The PR script consumes averaged predictions and writes PR metrics, confusion counts and curve data. Its `--prob-col` argument identifies the probability column; it is not a model name. These uncertainty estimates are different from the current paired cluster bootstrap in S18/S19.

### D Missingness, tuning, symmetry, Elo benchmarks and ablation

```bash
python missingness_imputation.py --input data/prematch_features_v3.csv --outdir revision_202611/outputs/legacy_missingness
python timeaware_hyperparameter_tuning.py --input data/prematch_features_v3.csv --outdir revision_202611/outputs/legacy_tuning --n-search 30
python train_team_swap_symmetry.py --input data/prematch_features_v3.csv --outdir revision_202611/outputs/legacy_swap
python trivial_favorite_baseline.py --input data/prematch_features_v3.csv --outdir revision_202611/outputs/legacy_elo_benchmarks
python feature_block_ablation.py --input data/prematch_features_v3.csv --outdir revision_202611/outputs/legacy_ablation
```

These are independent fits, not postprocessing of the primary model. Tuning selects configurations inside three training-period folds. The missingness comparison has its own matched `has_time=True` baseline. Symmetry augments training representations and evaluates complementary predictions. Elo benchmarks compare the full model with simple avg_player_elo_diff baselines. Ablation evaluates six defined feature configurations. Their randomization/configuration records, reference outputs and preprocessing rules determine comparability; do not merge their AUCs into Table 1.

### E Rolling-origin and COVID-period analyses

```bash
python rolling_origin_covid_minimal.py --input data/prematch_features_v3.csv --raw-input revision_202611/data/full_archive.csv --outdir revision_202611/outputs/legacy_rolling
```

This evaluates expanding temporal windows and COVID-period test subsets under the original source-order representation. It is not the tournament-freezing analysis. Its raw input is audited if present; supply the verified path for the audit. The script's historical numeric tolerance is a diagnostic threshold, not evidence that differences within that threshold are exact reproductions.

### F Alternative individual-player Elo rule for S17

The public `player_elo_sensitivity/` folder contains results. The new revision also supplies the archived source needed to rebuild its alternative representation and interpretation:

```bash
python revision_202611/code/archived/revised_code/priority1_1_build_prematch_dataset_v3_features_sensitivity.py --input revision_202611/data/full_archive.csv --output revision_202611/outputs/features/opponent_average_features.csv --player-elo-update-mode opponent_team_average
python feature_block_ablation.py --input revision_202611/outputs/features/opponent_average_features.csv --outdir revision_202611/outputs/legacy_player_elo_models
python revision_202611/code/archived/revised_code/priority1_2_player_elo_sensitivity_importance_shap.py --input revision_202611/outputs/features/opponent_average_features.csv --outdir revision_202611/outputs/legacy_player_elo_shap --reference-script revision_202611/code/archived/revised_code/priority8_feature_block_ablation_final_v1.py --shap-sample-size 2000
```

Create `revision_202611/outputs/features` first if absent. The builder's `slot_pairing` mode is the original rule; `opponent_team_average` is the sensitivity. It uses raw input, whereas model/SHAP commands use the resulting feature table. The ablation command evaluates multiple feature sets; use its full-model row for the full-model performance comparison. The interpretation command imports the explicitly supplied reference script for exact predictor/model definitions. These commands document the supplied historical workflow; no fresh end-to-end S17 execution was performed for this guide.

### G Original 104-feature SHAP and paper exports

The revision directory retains `code/archived/run_catboost_no_identity_shap.py`, `run_catboost_shap_analysis.py` and `export_paper_tables_figures.py`. For the no-identity interpretation script, the sample flag is `--sample-size`, not the S16 script's `--shap-sample-size`:

```bash
python revision_202611/code/archived/run_catboost_no_identity_shap.py --input data/prematch_features_v3.csv --outdir revision_202611/outputs/legacy_no_identity_shap --topk 20 --sample-size 2000
```

For the primary 104-feature interpretation, consult the retained `explain_outputs/` and `no_identity_explain_outputs/` source configurations to identify the exact run used by the figure; this archived command is an additional historical workflow, not a certification that it regenerates the exact manuscript panel. `run_catboost_shap_analysis.py` also exposes input/output and sampling arguments; inspect its `--help` and model settings before selecting that alternative. The identity-augmented S16 command is not a replacement for the primary SHAP figure. `export_paper_tables_figures.py` has configured paths rather than the same standardized CLI: check its input/output constants before use. Paper exports assemble outputs; they do not establish that those outputs were regenerated from current inputs.

## Dependencies for older analyses

The core environment supports the current CatBoost/temporal/bootstrap workflow. Full four-model fitting additionally requires XGBoost; interpretation commands require their compatible SHAP environment. Historical reported versions are in `provenance/historical_versions_reported.json`. They are records of reported settings, not a promise that one universal environment reproduces every old experiment. Do not install an arbitrary newest SHAP version and label the resulting fit an exact reproduction without comparison. Historical source and reference results remain useful even when an environment must first be reconstructed.

## Common mistakes

- Running a raw-data builder on an already processed table, or a trainer on raw data.
- Treating `original_features.csv`, `frozen0_features.csv` and `frozen30_features.csv` as equivalent inputs.
- Using original absolute server defaults on a different computer.
- Taking the first similarly named summary file as the Table 1 source.
- Treating averaged-probability AUC as mean per-run AUC.
- Reusing identity-analysis `has_time=True` as if it were the current primary/frozen setting.
- Claiming subgroup/SHAP robustness under frozen histories without actually refitting those analyses.
- Deleting numerical seeds from code because they were omitted from manuscript prose.
- Running every legacy script and assuming it reproduces one single common baseline.
- Renaming a folder without updating its consumers' prediction-file arguments.

## Scope of this whole-repository guide

The twelve public root script argument parsers were inspected against the supplied repository snapshot; the alternative-Elo archived source was also inspected. Examples use those supported arguments. This is a source-level workflow check, not a fresh execution of every legacy command. Linux validation of the current six temporal tests and the S18/S19 reference draws remains as recorded in `UPLOAD_VALIDATION.md`. Complete historical training, SHAP refits, a fresh dependency install and native Windows reproduction were not newly tested for this guide.
