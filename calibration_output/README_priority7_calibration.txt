
Priority 7 Calibration Analysis
===============================

Validation observations: 12369
Test observations: 8083

Primary calibration assessment
------------------------------
The retained CatBoost model is evaluated without recalibration using:
- AUC
- Brier score
- ECE
- Log-loss
- Reliability curves

ECE definition
--------------
ECE uses 10 equally spaced probability bins over [0,1].
For each bin, the absolute difference between mean predicted probability and
observed event rate is weighted by the proportion of observations in that bin.

Recalibration sensitivity analysis
----------------------------------
Platt scaling and isotonic regression are fitted ONLY on validation predictions
and labels. The fitted calibrators are then applied to the untouched test set.

Supplementary Table Sx
----------------------
supplementary_table_Sx_original_calibration.csv

Supplementary Table Sy
----------------------
supplementary_table_Sy_recalibration_sensitivity.csv

Supplementary Figure SxA
------------------------
fig_SxA_original_validation_test_reliability.png/pdf

Supplementary Figure SxB
------------------------
fig_SxB_test_recalibration_reliability.png/pdf

Recommended manuscript interpretation
-------------------------------------
The original CatBoost should remain the primary model.
Platt/isotonic results should be reported as supplementary post-hoc
recalibration sensitivity analyses.
