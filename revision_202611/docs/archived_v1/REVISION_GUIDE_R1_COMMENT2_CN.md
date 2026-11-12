> Archived v1 submission draft. Its unresolved CatBoost reproduction statement is superseded by the v2 reproduction audit and 48-thread results. Update manuscript statements before submission.

# 审稿人1第二条意见：正文修改位置与英文替换稿

本清单仅针对数据来源与可复现性。P编号是当前Word主文档顶层段落编号，不是页码。原Word由作者自行修改；下列英文完成时及“accompanying package”在修改正文并上传ZIP作为投稿附件后使用。

## 1. Methods / Data source and prediction task，P103：替换数据来源段

This study used the complete public AVP/FIVB archive at data/full_archive/full_archive.csv in the BigTimeStats beach-volleyball repository. Retrospective file-object verification identified byte-equivalent versions of our complete archive and local 2022 yearly file at commit 0fa423c5704de46a8cf0d9b8b542c694c0ae2276, dated 29 September 2022. The complete archive contains 85,509 records and 65 columns, including men's and women's events. Its source-event dates range from 16 September 2000 to 23 September 2022. The original author download date was not retained and could not be recovered; the public commit date should not be interpreted as the download date. The fixed source link, checksums and file-object verification evidence are provided in the Supplementary Reproducibility Package.

在这一段后新增纳入流程：

All 85,509 complete-archive records were retained in the feature table. No complete-case deletion or deduplication was applied; the audit identified no exact duplicate full rows. The reconstruction procedure verifies required columns and identifiers and parses the source dates, then applies deterministic source-date/year/match-number ordering, name normalization, canonical outcome-independent Team A/Team B assignment, and feature generation. This ordering is a reproducible processing convention rather than externally verified match chronology. Missing feature values were retained during table construction and handled during model training as described below. The 2020, 2021 and 2022 cohorts contain 719, 3,158 and 4,206 matches, respectively. Annual counts by circuit and gender and the cohort-flow table are supplied with the supplementary package. Coverage refers to the identified public archive and is not asserted to be exhaustive for all AVP/FIVB matches.

注意：这里的“完整归档”表示上游文件名称和本地保留全体行，不表示覆盖全部现实比赛。不能把“85509→85509”写成上游没有筛选。

## 2. Methods，P104：保留A/B分配规则，修正结尾一句

保留目前按标准化队伍键排序的A/B定义。在结尾替换“Only information available before each match ... outcome-derived features were excluded”为：

Current-match outcome statistics and in-match measurements were excluded from the predictors. Historical outcomes were used to construct lagged performance, head-to-head and Elo features under the source-record timing assumptions described in the chronology audit.

这项修改与第一条意见共用，避免两份修改重复或相互矛盾。

## 3. Data preprocessing and time-aware split，P121–124附近：明确样本流与验证

建议在现有缺失值处理段之后补充：

The source-date-based partitions contained 65,057 training records through 2017, 12,369 validation records from 2018–2019, and 8,083 test records from 2020–2022. Dataset reconstruction checks the raw SHA256 checksum, schema, cohort size, annual counts, split sizes and required event/player identifiers. It does not silently remove records that fail required-field checks. Source-file checks and temporal program invariants do not establish actual match chronology; this limitation and the tournament-freezing sensitivity analyses are described separately.

全文已有相同划分时避免重复大段陈述；可把样本数集中到一个段落，并引用新增cohort-flow表。

## 4. Evaluation metrics and implementation，P151：新增复现包与环境边界

建议在原软件版本段后增加：

The accompanying Supplementary Reproducibility Package provides the feature-construction, source-date validation and model-evaluation scripts, together with dependency specifications and step-by-step execution instructions. Portable entry points support fixed-version raw-data acquisition, checksum verification, cohort reconstruction and the matched CatBoost sensitivity analyses. Archived model-comparison and interpretation scripts and result tables are also supplied. Machine-specific default path prefixes in the archived scripts were normalized for distribution, with changes recorded in the script manifest. The package validation report distinguishes tested reconstruction and evaluation paths from unexecuted historical analyses. Exact reproduction of the archived main-table AUC was not achieved in the matched rerun and is not asserted.

不要把旧环境的完整pip freeze直接当作可安装requirements：它包含大量本机file://路径。本包的requirements-core.txt记录本次实际验证环境；历史XGBoost/SHAP版本单独标为报告版本，不声称已在该环境验证。

## 5. 新增 Data and Code Availability 声明

建议放在期刊指定的数据/代码可用性位置；若没有指定，可在Methods之后、References之前新增：

The underlying match archive is publicly available from the BigTimeStats beach-volleyball repository at https://github.com/BigTimeStats/beach-volleyball/tree/0fa423c5704de46a8cf0d9b8b542c694c0ae2276. The analysis used data/full_archive/full_archive.csv; its SHA256 checksum and source-version verification are included in the Supplementary Reproducibility Package. This accompanying package provides the study's feature-construction and evaluation code, temporal checks, environment specifications, annual sample counts, reference outputs and reproduction instructions. The original download date is unavailable. Third-party raw data are retrieved from the fixed upstream source rather than redistributed in the package.

这份声明以把ZIP作为期刊补充附件上传为前提；目前尚未创建公共GitHub或Zenodo仓库，不写“code available at [虚构链接]”。如果期刊明确要求公共仓库而不接受附件，则作者后续发布此包，并用真实仓库/DOI替换代码获取方式即可。

## 6. BigTimeStats参考文献，P162：增加固定版本定位

可按期刊格式改为：

BigTimeStats (2022) AVP & FIVB Beach Volleyball Match Data and Statistics. Repository version: commit 0fa423c5704de46a8cf0d9b8b542c694c0ae2276 (29 September 2022). Available at: https://github.com/BigTimeStats/beach-volleyball/tree/0fa423c5704de46a8cf0d9b8b542c694c0ae2276.

如期刊要求访问日期，2026-11-04可用于本次核验日期；不能把它写成最初下载日期。不要沿用README旧的Data Last Updated作为2022记录覆盖的证据。

## 7. 补充材料应新增/上传的内容

1. Supplementary_Reproducibility_Package_v1.zip：代码、相对路径入口、环境、数据版本证据和验证报告。
2. tables/year_counts.csv：2000–2022全部年份的保留数量。
3. tables/year_circuit_gender_counts.csv：年份×巡回赛×性别及源日期范围。
4. tables/cohort_flow.csv：原始归档→特征表→训练/验证/测试样本数。
5. tables/feature_inventory_104.csv：104个预测变量名称、类型、缺失数量及算法位置；与正文S1定义一起使用。
6. provenance/dataset_manifest.json：固定版本、路径、哈希、下载日期未知、读取编码、来源限制。
7. tables/full_archive_vs_year_value_differences.csv：同一公共版本不同文件路径的35处姓名编码/国籍元数据差异；保留完整归档作为建模唯一来源。

新表编号按实际补充材料顺序确定。最终回复信需填入准确页码、行号和附件名。

## 8. 回复提交前的必要措辞检查

- 已证实：本地完整表/2022分年文件的字节内容与指定公共提交对应。
- 未恢复：作者最初下载日期、当时保存的提交记录、上游所有抓取与筛选细节。
- 不作推断：没有筛掉本地归档行不代表现实比赛全覆盖；文件对应不代表原始赛程时序正确。
- 数值边界：全量特征重建与历史主表AUC完全复现分别报告，不把两者混同。
- 投稿附件：上传ZIP后可称accompanying Supplementary Reproducibility Package；个人服务器目录不作为审稿人的访问链接。
