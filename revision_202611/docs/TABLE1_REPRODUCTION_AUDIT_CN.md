# 主表与R1原流程重跑：复现差异核查报告

本次在Windows本地通过普通SSH执行；计算、模型、预测、日志均保存服务器。未覆盖原结果、未修改Word、未重跑四组全部实验。

## 1. 主表来源已确认

主Word Table 1的CatBoost测试结果0.8126±0.0007及验证结果0.8226±0.0004，与baseline_with_sd_results_revise/model_results_summary_mean_sd.csv对应。
其源逐种子文件为final_catboost_v1/baseline_with_sd_results_revise/model_results_all_runs.csv。run_config.json与逐种子表均为42、52、62、72、82、92、102、112、122、132。

重新汇总得到测试AUC均值0.812574423349155、样本SD 0.000691273124891。全部4个模型、验证/测试、4项指标的均值及SD在4位小数上与保存汇总一致：True。未发现本次核对的汇总算术错误。主Word的CatBoost行也与该汇总一致。

不要混用：baseline_with_sd_results是5种子早期版本；paper_outputs及README_VERSION.txt的0.8120是更早输出，不是当前Word主表；auc_uncertainty_delong_output的逐场预测使用41、51至131，不能当作主表42、52至132的原始预测。这一点从保存的列名、配置和代码可验证。

## 2. 输入、顺序与划分

本次从留存原CSV按原训练脚本读取/排序，并与R1重跑original_features.csv按R1代码读取/排序对比。104个特征顺序、分类列索引一致。训练65057、验证12369、测试8083；date/match_num/teamA_key/teamB_key/y行键及标签在各划分的对比结果详见逐特征表。

源CSV SHA256：33d7569603f40c0c9bfce86f4b48c5822e6456b5270d6fd03e576f991af8d5c4。

CSV中转后的文件字节哈希不同，训练/验证/测试分别有11703、1879、1270个预处理后数值单元出现浮点末位差别；转为float32后的差异均为0。分类值一致。不能把两个CSV称为逐字节或全部float64值完全相同。

关键控制实验：分别采用R1中转文件与原CSV/原预处理函数，在同一环境、种子92、4线程下重新训练，两者测试AUC均为0.811348290252899，验证AUC均为0.822772802368878。两者与R1保存预测的最大绝对差仅约1.1e-16。因此这些末位CSV差别没有解释该种子的AUC差异。

限制：历史主表运行时未保存输入校验和。我们核对的是当前留存的原文件与配置，不能据此独立证明历史运行时的文件从未变化。

## 3. 参数、环境与最佳迭代

原留存训练脚本与R1脚本共同显式设置：iterations=900，depth=6，learning_rate=0.03，l2_leaf_reg=5，subsample=0.85，loss_function=Logloss，eval_metric=AUC，random_seed为对应种子，allow_writing_files=False，verbose=False。

已发现的代码差别：原脚本未指定thread_count；R1固定thread_count=4。原fit未显式指定use_best_model；R1显式True。在本次复现实验中完整有效参数已保存，默认use_best_model是否一致以参数JSON为准。has_time并非本次基线显式设置；当前有效值为False。不能把其他身份敏感性实验的has_time=True写成主基线设置。

当前隔离环境：{"python": "3.11.7 (main, Dec 15 2023, 18:12:31) [GCC 11.2.0]", "numpy": "1.26.4", "pandas": "3.0.2", "sklearn": "1.5.2", "catboost": "1.2.10"}。这些核心版本与Word P151报告一致，但原主表没有对应的运行时环境快照或保存模型来独立认证。指定的原Python路径仍存在，当前其中无法导入catboost，不能用它恢复历史训练环境。

原主表逐种子表只保存指标，未保存best_iteration。R1保存了逐种子best_iteration。种子92的R1及本次4线程控制实验best_iteration=899（从0计数），tree_count=900。原主表种子92当时的最佳迭代轮数无法从该表恢复。

## 4. 最大差异种子92与原默认设置复验

原主表AUC：0.812647622126542；R1：0.811348290252899；差值R1−原表：−0.001299331873644。

进一步按原脚本省略thread_count和use_best_model，在当前同一环境、原CSV、原预处理、种子92下训练的结果：

```json
{
  "experiment": "original_defaults",
  "seed": 92,
  "best_iteration": 898,
  "tree_count": 899,
  "seconds": 38.379510164260864,
  "archived_table1_test_auc": 0.8126476221265424,
  "valid_auc": 0.8231177406113586,
  "valid_max_prediction_delta_vs_4threads": 0.15525360957214562,
  "test_auc": 0.8126476221265424,
  "test_max_prediction_delta_vs_4threads": 0.1938291113540147
}
```

完整模型、训练元数据、验证曲线及逐场预测保存在本目录，可以继续对比本次新训练的模型。原final_catboost_v1及R1原重跑目录未发现保存的.cbm模型；主表对应的42、52至132逐场预测亦未在已查目录中找到。因此，不能声称已经完成原历史模型与R1模型逐树/逐场直接对比；41、51至131的后续推断预测不能填补这一缺口。

## 5. 可用结论与边界

主表均值和SD有对应逐种子来源，重新汇总正确，使用相同10种子。当前留存输入、划分、特征及预处理的控制比较未显示导致种子92差异的输入变化。

### 线程差别的复验证据（以本节替代此前材料中的“原因尚未确定”）

默认线程全种子复验完成：True；已复验种子数：10。所有种子、验证/测试、AUC/accuracy/F1/logloss与原表逐种子指标的最大绝对差：0。

默认设置复验的测试AUC均值：0.812574423349155；SD：0.000691273124891。

新训练模型元数据证实默认设置实际thread_count=48；R1对照实际thread_count=4。种子92的get_all_params()有效模型参数完全一致，use_best_model在两种设置中都为True；差别是单独保存在system_options中的线程数。其他主要核心版本、输入和训练参数在本次控制实验中保持一致。

因此本次受控复验已将均值差异定位到执行线程配置不同，并且可以用原脚本默认设置重现原主表。不是已发现的数据筛选、特征集合或主表汇总错误。我们没有保存历史运行的实际线程元数据，故“原历史任务确实用了48线程”仍属于与留存脚本及精确复验一致的推断；本次默认运行确实用了48线程则有新模型元数据直接证明。

种子92的验证AUC曲线首次在从0计数的第6轮产生差异，不是仅最终汇总或选择最佳迭代时才出现。模型JSON的全局split_index不能跨模型直接比较；忽略该索引后，首次分裂描述差异位于第59棵树（从0计数，OnlineCtr的进一步语义还依赖模型元数据）。这里只将训练曲线差异作为定位依据，不把全局索引差异误判为首树结构差异。

本次证据支持：相同种子不够，还需固定线程配置。我们没有进一步审计CatBoost内部并行计算实现，因此不声称精确证明了内部数值归约或随机数流机制。

### 对论文及R1结果的处理

原主表可以继续保留，其汇总正确且已按原默认设置重现。R1四组在统一4线程条件下比较，匹配基线0.812198571659491也有对应重复实验支持。二者应分别标注，不把归档主表直接代入R1差值，也不需要仅为掩盖差异而删除匹配基线。当前没有覆盖原结果，也没有将冻结三组改为48线程。

如保留现有四组，可在方法/补充说明使用：

The archived primary analysis used the script's default threading configuration. The temporal sensitivity analyses fixed CatBoost to four threads for all scenarios. A controlled rerun using the original default configuration reproduced the archived ten-seed metrics, whereas the four-thread configuration reproduced the matched sensitivity baseline. Comparisons among temporal scenarios therefore use the separately matched four-thread baseline.

原R1回复和报告中“重跑差异原因尚未确定”的文字现已过时，应按本核查替换为线程配置及复验结果；时间顺序/赛事结束日限制本身并未因此消除。

## 文件

- catboost_seed_comparison.csv：10种子原主表/R1对照及差值。
- all_models_summary_recalculation.csv：4模型均值及SD复算。
- input_comparison_by_feature_split.csv：各划分/特征数值及float32差异、行键和标签核对。
- parameter_source_comparison.json：两份代码显式参数与fit调用。
- static_audit.json：数据/代码哈希及环境。
- controlled_experiments.csv、original_defaults_results.json：单种子控制实验。
- *_all_params.json、*_training_metadata.json、*.cbm、*_predictions.csv、*_validation_curve.csv：本次模型及预测证据。
- audit.log、default_threads.log：运行日志。

服务器目录：<PROJECT_ROOT>/table1_reproduction_audit_20261104
