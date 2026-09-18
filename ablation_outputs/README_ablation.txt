CatBoost identity-ablation control experiment
-------------------------------------------
Target column: y
Dropped identity columns: ['teamA_key', 'teamB_key', 'teamA_player1', 'teamA_player2', 'teamB_player1', 'teamB_player2']

               model feature_set  valid_auc  valid_accuracy  valid_f1  valid_logloss  test_auc  test_accuracy  test_f1  test_logloss
       CatBoost_full        full     0.8041          0.7220    0.7341         0.5338    0.7975         0.7210   0.7291        0.5447
CatBoost_no_identity no_identity     0.8227          0.7353    0.7422         0.5109    0.8118         0.7296   0.7311        0.5272

Interpretation guide:
- If AUC drops only slightly after dropping identity columns, the model relies mainly on generalizable pre-match features.
- If AUC drops a lot, the previous best model relied heavily on player/team identity memory.
- The no-identity importance plot is better suited for paper interpretation.
