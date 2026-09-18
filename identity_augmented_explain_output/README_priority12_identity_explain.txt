Priority 12: Identity-augmented CatBoost interpretation

Most important files to inspect:
1. raw_identity_feature_importance_and_shap.csv
2. identity_augmented_top20_by_shap.csv
3. identity_vs_derived_attribution_share.csv
4. identity_interpretation_performance_summary.csv
5. fig_identity_augmented_feature_importance_top20.png
6. fig_identity_augmented_shap_bar_top20.png
7. fig_raw_identity_shap.png

Interpretation:
- If raw identity variables rank low by both CatBoost importance and mean |SHAP|,
  the identity-augmented model relies mainly on derived prematch signals.
- If some raw identifiers rank high but out-of-time performance is worse, the
  defensible interpretation is that the model uses identity-linked patterns that
  do not improve temporal generalization.
- Avoid claiming that Elo is identity-free. Elo is a derived signal that encodes
  historical identity-linked performance; distinguish DERIVED SIGNALS from RAW
  IDENTIFIERS.
