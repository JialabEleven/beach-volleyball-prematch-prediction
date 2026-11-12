# Beach volleyball prematch prediction

Code and reference results for interpretable modeling of professional AVP/FIVB beach-volleyball outcomes.

## Start here

Use **[revision_202611](revision_202611/README.md)** for the current revision. It provides a portable entry point, pinned source-data verification, source-order and frozen-history feature construction, controlled 48-thread CatBoost evaluation, and the current S18/S19 paired cluster-bootstrap analyses.

Read **[CODE_GUIDE.md](CODE_GUIDE.md)** for the complete new and historical workflows, installation, supported script arguments, input/output dependencies, table mapping and limitations.

The older root-level scripts and result folders are retained as historical analyses. They are not interchangeable configurations, and older single-run or four-thread results should not replace the manuscript's current primary and temporal-sensitivity estimates.

## Current result map

| Manuscript output | Authoritative reference |
|---|---|
| Table 1 | `baseline_with_sd_results_revise/model_results_all_runs.csv`; controlled CatBoost checks under `revision_202611/reference_results/main_table_reproduction/` |
| S16 and continuation | `revision_202611/tables/submission_tables/` |
| S18: CatBoost minus XGBoost cluster bootstrap | `revision_202611/code/reproduce_table_s18.py` |
| S19: matched frozen-history sensitivity | `revision_202611/reference_results/r1_48threads/`; bootstrap entry point `code/paired_cluster_bootstrap_48threads.py` |
| Data version and annual cohort | `revision_202611/provenance/` and `revision_202611/tables/` |

## Interpretation

The original main analysis is retained, with frozen-history analyses reported separately as sensitivities. Source dates commonly denote tournament starts, not actual match days. Match chronology and event-completion dates remain incompletely verified. Freezing excludes target-event updating, but does not certify that all earlier-source-date outcomes were available before competition. SHAP values are model attributions, not causal effects or intervention recommendations.

The controlled CatBoost main-table reproduction and three frozen scenarios were fitted on the authors' Linux server. This does not establish exact reproduction of every historical model or SHAP analysis. See the dated validation reports for the checks actually performed.

## Data and reuse

Third-party raw data should be retrieved from the pinned BigTimeStats source and verified by checksum; see `THIRD_PARTY_DATA_NOTICE.md` inside the revision directory. No third-party license is replaced by this repository. Reuse rights for the authors' code should be specified by a repository license selected by the authors.
