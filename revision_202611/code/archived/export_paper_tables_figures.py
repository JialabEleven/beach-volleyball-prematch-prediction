import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = "./legacy_project/final_catboost_v1"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
ROC_DIR = os.path.join(RESULTS_DIR, "roc_curves")
OUT_DIR = os.path.join(BASE_DIR, "paper_outputs")

os.makedirs(OUT_DIR, exist_ok=True)

# ==========
# 1. 读取结果表
# ==========
results_csv = os.path.join(RESULTS_DIR, "model_results.csv")
df = pd.read_csv(results_csv)

# 统一保留 4 位小数，便于论文输出
df_round = df.copy()
num_cols = [c for c in df_round.columns if c != "model"]
for c in num_cols:
    df_round[c] = df_round[c].round(4)

# ==========
# 2. 输出表格
# ==========
# 完整结果表
full_csv = os.path.join(OUT_DIR, "table_model_results_full.csv")
df_round.to_csv(full_csv, index=False)

# TEST 主结果表（正文最常用）
test_cols = ["model", "test_auc", "test_accuracy", "test_f1", "test_logloss"]
df_test = df_round[test_cols].sort_values("test_auc", ascending=False).reset_index(drop=True)
test_csv = os.path.join(OUT_DIR, "table_test_performance_sorted.csv")
df_test.to_csv(test_csv, index=False)

# VALID 辅助表
valid_cols = ["model", "valid_auc", "valid_accuracy", "valid_f1", "valid_logloss"]
df_valid = df_round[valid_cols].sort_values("valid_auc", ascending=False).reset_index(drop=True)
valid_csv = os.path.join(OUT_DIR, "table_valid_performance_sorted.csv")
df_valid.to_csv(valid_csv, index=False)

# LaTeX 表（后面写论文可直接参考）
latex_path = os.path.join(OUT_DIR, "table_test_performance_sorted.tex")
with open(latex_path, "w", encoding="utf-8") as f:
    f.write(df_test.to_latex(index=False, float_format="%.4f"))

# ==========
# 3. 画图：TEST 集核心指标柱状图
# ==========
models = df_round["model"].tolist()
x = np.arange(len(models))
width = 0.25

plt.figure(figsize=(8, 5))
plt.bar(x - width, df_round["test_auc"], width, label="AUC")
plt.bar(x, df_round["test_accuracy"], width, label="Accuracy")
plt.bar(x + width, df_round["test_f1"], width, label="F1")
plt.xticks(x, models, rotation=15)
plt.ylabel("Score")
plt.ylim(0, 1)
plt.title("Test Set Performance Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_test_core_metrics.png"), dpi=300)
plt.savefig(os.path.join(OUT_DIR, "fig_test_core_metrics.pdf"))
plt.close()

# ==========
# 4. 画图：VALID 集核心指标柱状图
# ==========
plt.figure(figsize=(8, 5))
plt.bar(x - width, df_round["valid_auc"], width, label="AUC")
plt.bar(x, df_round["valid_accuracy"], width, label="Accuracy")
plt.bar(x + width, df_round["valid_f1"], width, label="F1")
plt.xticks(x, models, rotation=15)
plt.ylabel("Score")
plt.ylim(0, 1)
plt.title("Validation Set Performance Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_valid_core_metrics.png"), dpi=300)
plt.savefig(os.path.join(OUT_DIR, "fig_valid_core_metrics.pdf"))
plt.close()

# ==========
# 5. 画图：TEST 集 LogLoss 对比图
# ==========
plt.figure(figsize=(8, 5))
plt.bar(x, df_round["test_logloss"])
plt.xticks(x, models, rotation=15)
plt.ylabel("LogLoss")
plt.title("Test Set LogLoss Comparison (Lower is Better)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_test_logloss.png"), dpi=300)
plt.savefig(os.path.join(OUT_DIR, "fig_test_logloss.pdf"))
plt.close()

# ==========
# 6. 复制现有 ROC 图到论文目录，改成更直观的名字
# ==========
copy_map = {
    "roc_comparison_test.png": "fig_roc_comparison_test.png",
    "roc_comparison_test.pdf": "fig_roc_comparison_test.pdf",
    "roc_comparison_valid.png": "fig_roc_comparison_valid.png",
    "roc_comparison_valid.pdf": "fig_roc_comparison_valid.pdf",
    "roc_catboost_test.png": "fig_roc_catboost_test.png",
    "roc_catboost_test.pdf": "fig_roc_catboost_test.pdf",
    "roc_catboost_valid.png": "fig_roc_catboost_valid.png",
    "roc_catboost_valid.pdf": "fig_roc_catboost_valid.pdf",
}

for src_name, dst_name in copy_map.items():
    src = os.path.join(ROC_DIR, src_name)
    dst = os.path.join(OUT_DIR, dst_name)
    if os.path.exists(src):
        shutil.copy2(src, dst)

# ==========
# 7. 生成一个说明文件，告诉你哪些适合正文、哪些适合补充材料
# ==========
notes_path = os.path.join(OUT_DIR, "paper_output_notes.txt")
with open(notes_path, "w", encoding="utf-8") as f:
    f.write("Recommended main-text outputs:\n")
    f.write("1. table_test_performance_sorted.csv\n")
    f.write("2. fig_roc_comparison_test.png\n")
    f.write("3. fig_test_core_metrics.png\n")
    f.write("\nRecommended supplementary outputs:\n")
    f.write("1. table_model_results_full.csv\n")
    f.write("2. fig_roc_comparison_valid.png\n")
    f.write("3. fig_test_logloss.png\n")
    f.write("4. fig_catboost_test.png / fig_catboost_valid.png\n")

print("Done!")
print(f"Paper outputs saved to: {OUT_DIR}")
