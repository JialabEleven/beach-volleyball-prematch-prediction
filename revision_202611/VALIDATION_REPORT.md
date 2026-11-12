# v2 validation report

The updated CLI help paths and controlled main-script argument parser passed. A 100-observation-per-partition, 8-iteration, one-seed smoke fit passed and its saved CatBoost training metadata confirms 48 threads. This smoke result is not a manuscript estimate.

The portable fixed-prediction bootstrap script reproduced all 5,000 draws for each of the 84- and 99-group definitions and the updated temporal comparison tables within 1e-12. The server results separately document 30 completed 48-thread frozen refits and exact reproduction of all archived per-seed CatBoost primary metrics. These full server fits preceded package assembly.

Full four-model training, SHAP/interpretation fits, a fresh clean-environment install/download and the complete raw-to-model chain were not rerun during v2 packaging. Earlier full feature reconstruction and temporal invariant checks are retained as v1 validation evidence, explicitly dated to that earlier package stage. Historical validation material is not evidence of running the newly packaged full training paths.

The retained primary CatBoost metrics come from seeds 42..132. Principal CatBoost-XGBoost inference predictions come from a separately documented 41..131 run. Old four-thread temporal results are preserved for traceability but superseded by reference_results/r1_48threads for the current sensitivity presentation. Source dates, actual chronology and event-completion limitations are unchanged. No public code repository or journal attachment was created automatically.
