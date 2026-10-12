# T07｜固定推荐指标与评测集合（2026-10-06）

在任何新模型训练前，按照规范 §10.2 固定 ID 排名评测。valid NDCG@10 是配置选择指标；测试窗口只在冻结配置后报告。旧原版的模糊文本命中与新 MovieLens1M 指标不混算。

## Completed

- 实现稳定全集 Top-K：仅用 T06 训练候选 ID，排除用户训练期所有已评分候选；分数降序、ID 升序打破并列。缺分、NaN、padding、重复与目录外 ID 均显式失败。
- 实现二值 Recall@K、Hit@K、MRR@K、NDCG@K。缺失、空或非法预测在有相关目标用户的固定分母中记零并计入失败；无相关目标用户单独计数，不进入主均值。
- 从 T06 本地映射/训练历史与 valid/test 目标构建评测专用冻结 cohort。两个窗口共用 5351 个训练期 warm 用户和 3469 个候选物品；只在评测侧保存相关集合，供 T08/T09 使用，未向训练输入写入未来目标。

## Files Created / Modified

新增 `AgenticRec/src/agenticrec/evaluation/metrics.py`、`cohort.py`、包入口和 `AgenticRec/tests/unit/test_metrics.py`。两份逐用户 cohort 位于本地忽略目录 `data/eval_private/ml-1m-v1/`；公开汇总、SHA 和原生测试证据位于 `reproduction/native_runs/20261006_metrics/`。更新 TASKS、STATUS、README。

## Commands / Tests

```powershell
& AgenticRec/.venv-dev/Scripts/python.exe -m pytest AgenticRec/tests/unit/test_metrics.py -q
& AgenticRec/.venv-dev/Scripts/python.exe -m pytest AgenticRec/tests -q
& AgenticRec/.venv-dev-rebuild/Scripts/python.exe -m pytest AgenticRec/tests/unit/test_metrics.py -q
& AgenticRec/.venv-dev/Scripts/python.exe -m agenticrec.evaluation.cohort
```

cohort 冻结入口重复运行拒绝覆盖。测试包括完全命中、全错、短列表、空返回、非法/重复/已看物品、并列分、缺失预测、多相关目标与无相关用户。

## Verification Results

专项指标/评测集合 12 项通过，现代包完整 32 项通过，第二隔离环境专项 12 项通过。手算示例：R={1,3}，L=[2,1,3]，K=5。Recall=1，Hit=1，MRR=1/2；DCG=`1/log2(3)+1/log2(4)`，IDCG=`1+1/log2(3)`，NDCG≈0.6934264036，与实现一致。

| 冻结窗口 | 训练期 warm 用户 | 有可评相关目标的用户 | 无相关目标、主均值不计的用户 | warm 相关物品 |
| --- | ---: | ---: | ---: | ---: |
| valid | 5351 | 481 | 4870 | 12197 |
| test | 5351 | 954 | 4397 | 41499 |

评测集合 SHA 与 T06 的 data manifest、ID map 和相应 holdout 文件哈希绑定。[公开汇总](../reproduction/native_runs/20261006_metrics/cohort_summary.json)仅含聚合和哈希；逐用户数据不入 Git。这里的数字是覆盖与标签数量，模型 Recall/NDCG 等仍为 **NOT EVALUATED**。

## Findings

全局时间划分使 valid/test 能评价的 warm 用户只有训练期 warm 用户的一部分。后续报告须同时给出主均值的实际用户分母和 cold/无相关覆盖数，不能只挑命中用户。训练期低评分仍算已观察，候选过滤要排除；未来正反馈不作为训练负采样的排除依据。

## Blockers

T07 无新 blocker。T08/T09 的模型尚未训练或打分，无法判断是否超过 Popularity。旧资源权威 ID 与原包许可依旧 NOT VERIFIED，但不阻断已冻结的独立质量实验。

## Deviations from Spec

None。主指标和规则遵循规范 §6.3、§7.1、§10.2；未因模型结果选择新指标。旧 A1 正式 0/45 保留独立报告，不与本 protocol 的 ID 指标混合。

## Current Project State

T00/T01/T02/T03/T05/T06/T07 DONE；T04 原完整复现 BLOCKED；T08—T24 TODO。新模型收益 **NOT EVALUATED**，本轮新增 LLM 请求 0。

## Next Tasks

T08 按同一 T06 split、T07 cohort 跑 Random 和训练期 Popularity 基线，再实现/训练 BPR-MF、保存真实预测/成本与指标。仅用 valid NDCG@10 调参；测试不参与选择。之后 T09 训练和验证 LightGCN。
