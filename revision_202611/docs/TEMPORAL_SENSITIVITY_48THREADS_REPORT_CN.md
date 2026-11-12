# 48线程冻结敏感性分析：结果报告

已完成三组冻结场景各10种子训练，复用已精确复现原主表的48线程基线。新结果单独保存；原Word、原主表及既有4线程结果均未覆盖。

## 匹配条件

种子42、52、62、72、82、92、102、112、122、132。每个新模型显式thread_count=48；原基线在已完成复验中使用默认配置，保存元数据确认为48线程。共同参数：900次最大迭代、深度6、学习率0.03、L2=5、subsample=0.85、Logloss损失、AUC验证指标；有效use_best_model=True、has_time=False。最佳迭代按验证集AUC确定，各场景独立训练，未按测试结果选择设置。

训练65057、验证12369、测试8083场。训练期数值中位数/分类众数填补；分类转字符串。原/冻结/缓冲104特征，去日期场景74特征。移除名称包含days_since、last30、last90、last180的变量，保留month和近期N场。复用冻结特征及既有程序不变量审计，文件哈希和新模型参数见清单。三个场景并行，每个模型均48线程；系统负载会影响耗时。

## 逐种子性能均值与SD

SD描述种子训练差异，不是测试抽样置信区间。

| 场景 | 验证AUC | 测试AUC | Accuracy | F1 | Log-loss | Brier |
|---|---:|---:|---:|---:|---:|---:|
| 已复现原主表基线 | 0.822586 ± 0.000438 | 0.812574 ± 0.000691 | 0.729383 ± 0.002088 | 0.730637 ± 0.003592 | 0.525286 ± 0.000722 | 0.176914 ± 0.000292 |
| 赛事历史冻结 | 0.818230 ± 0.000690 | 0.804558 ± 0.000559 | 0.724644 ± 0.000944 | 0.726491 ± 0.001564 | 0.535095 ± 0.000685 | 0.180558 ± 0.000246 |
| 冻结并移除日期间隔/窗口变量 | 0.820746 ± 0.000440 | 0.805716 ± 0.001192 | 0.723518 ± 0.001264 | 0.725005 ± 0.002397 | 0.532954 ± 0.001391 | 0.179905 ± 0.000529 |
| 冻结并加入30天缓冲 | 0.816480 ± 0.000583 | 0.799493 ± 0.000645 | 0.718656 ± 0.001221 | 0.720663 ± 0.002776 | 0.540759 ± 0.000700 | 0.182862 ± 0.000273 |

## 配对赛事聚类bootstrap

以各场景10种子平均概率作为预测器，5000次配对整组重抽样，随机种子20261104。主定义circuit/tournament/country/year合并男女及同地重复站，共84组；加入源日期的敏感性定义99组。逐场通过raw_row身份对齐，避免用不唯一的日期/标签连接。以下AUC是平均概率预测器的AUC，与上表逐种子AUC均值不同。

| 比较：原基线减去 | AUC差 | 95%CI |
|---|---:|---:|
| 赛事历史冻结 | 0.008051 | [0.005192, 0.011053] |
| 冻结并移除日期间隔/窗口变量 | 0.006784 | [0.003916, 0.009733] |
| 冻结并加入30天缓冲 | 0.013037 | [0.008606, 0.017605] |

99组定义、各场景区间及全部重抽样保存于对应CSV。区间条件于固定预测，未完整处理跨赛事重复球员和训练不确定性。

缓冲组的teamA_last30_win_pct、teamB_last30_win_pct和last30_win_pct_diff在训练期全缺失，其中位数亦为NaN；沿用原处理规则保留数值NaN，由CatBoost原生处理。没有添加新的常数填补。执行中曾因进程初始化阻塞和过严的NaN预检查而调整启动/检查代码，已保存初次代码和日志；完成的种子从检查点复用，训练参数与特征未因此变动。

## 与旧4线程结果的关系

本报告使用统一48线程及已复现原主表基线，作为新版匹配时序敏感性结果。旧4线程结果保留，不能与新版任意拼接。线程改变已在原流程受控复验中解释主表与旧匹配基线差异；本次没有重算其他主分析模型、SHAP、校准或亚组。

## 时间规则及剩余限制

普通冻结纳入源赛事日期严格早于目标日期的赛事，同日期相互排除；目标赛事历史、交锋、Elo状态固定。缓冲要求source_date+30天<目标日期。当前记录背景变量保留。完整真实结束日期未核实，较早开始不保证已结束，30天是敏感性假设。此前合格历史Elo仍按确定性源记录顺序更新。

性能下降反映历史表示与可纳入信息变化，不能全部归因为移除泄漏，也不能据此证明真实逐场时序已验证。冻结可能排除合法的赛事内信息。原主分析及其SHAP/亚组结果仍受原构建假设限制。

## 论文及回复可用英文

Using a matched 48-thread CatBoost configuration and the same ten seeds, mean test AUC was 0.8126 for the reproduced original pipeline, 0.8046 for tournament-frozen histories, 0.8057 after additionally removing source-date interval/window predictors, and 0.7995 with a 30-day release buffer. All scenarios used the same temporal partitions, fixed model settings and validation-based iteration selection. These analyses quantify sensitivity to the historical-information representation but do not establish the absence of temporal leakage.

正文建议以本报告四组表及新bootstrap替换旧4线程敏感性数字；主表不变。所有描述应保留时间证据限制，最终补充表号和页码待作者编辑Word后填写。

## 保存文件

metrics_by_seed.csv、summary_mean_sd.csv、paired_seed_differences.csv；每个场景seed目录含模型、完整参数、训练元数据、验证学习曲线、验证/测试预测和指标；original_baseline含逐种子预测及原复验模型引用/哈希；paired_cluster_bootstrap.csv及全部draws、映射和平均预测。

环境和输入见run_manifest.json；最终核验见final_validation.json；全输出哈希见output_sha256.csv。

服务器目录：<PROJECT_ROOT>/R1_frozen_48threads_20261104
