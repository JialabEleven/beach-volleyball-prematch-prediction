# Package validation report

Validation was performed on the Linux server using ordinary SSH from the Windows session. Scripts were invoked from both the extracted package and an unrelated working directory. No Codex process was launched on the server.

Passed checks:
- Fixed full-archive SHA256, schema (85,509 rows × 65 raw columns), no exact duplicate full rows, required identifiers and dates, all annual counts and temporal split sizes.
- Full reconstruction of all 85,509 original feature rows; all archived feature columns agree at numerical tolerance 1e-10 with dtype/CSV-serialization differences allowed. No rows were dropped. The original feature CSV is not bundled.
- Six existing synthetic temporal invariant tests.
- Portable zero-lag freezing agrees with the existing implementation; 30-day release cutoff checks pass.
- Syntax checks of all supplied Python files.
- Portable CatBoost training/evaluation smoke check on 100 rows per partition, one seed and eight iterations; outputs explicitly labeled SMOKE_NOT_MANUSCRIPT_RESULTS.
- Standalone paired cluster bootstrap reproduces all 5,000 reference draws for each of the two cluster definitions and the resulting eight comparison rows at tolerance 1e-12.
- File-level manifest and ZIP CRC checks are performed at finalization; extracted-package integrity and CLI entry are checked.

Limits:
- All four historical model families and all full ten-seed training runs were not refitted during assembly. The historical Table 1 CatBoost AUC is not exactly reproduced by the separate R1 rerun; that discrepancy remains documented.
- A new clean dependency installation and a fresh network download were not tested here. Current known source bytes and public Git object metadata were used. Network provider access and API rate limits can change.
- The core requirements record the tested environment. Historical XGBoost/SHAP reported versions are supplied for provenance; a verified SHAP environment lock is unavailable. Matplotlib in the tested core is 3.8.0, distinct from the version reported for historical figures.
- Numeric feature agreement does not validate real-world match chronology. Source dates, match-number order, actual completion times and upstream exhaustive coverage remain limited as described in the chronology response.
- The original author download date could not be recovered. The 2022-09-29 public commit was matched retrospectively, not recorded as the historical download event.

Feature-building warnings about means of entirely missing age/height values reflect retained missing features, not deleted rows. The encoding policy and differences between full and yearly source files are documented in the data manifest.
