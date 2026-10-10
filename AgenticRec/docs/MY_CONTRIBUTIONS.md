# 来源与个人贡献

按实际代码分为上游、个人实现和引用算法；不将现成算法或原有功能声称为原创。

| 类别 | 来源 / 边界 | 本项目实现与证据 |
|---|---|---|
| Upstream | Microsoft RecAI固定SHA、MIT；Plan First、ToolBox、Buffer、Map、memory/reflection/demo已存在 | 原checkout不改；兼容副本/可选patch独立保留；U1实际调用原组件。[UPSTREAM_DIFF](UPSTREAM_DIFF.md)、[走读](source_walkthrough.md)、[版本锁](../../reproduction/upstream_lock.json) |
| 自己实现 | 数据及实验工程 | `data.py`、`evaluation/cohort.py`、`metrics.py`：全局时间split、训练期ID/图/采样、固定全集与失败分母；`test_data_contract.py`、`test_no_leakage.py`、`test_metrics.py` |
| 引用算法，自己实现 | [BPR](https://arxiv.org/abs/1205.2618)、[LightGCN](https://arxiv.org/abs/2002.02126)已有方法，不称算法发明 | `models/bpr.py`、`models/lightgcn.py`和训练代码；梯度、小图传播、重载验证。[BPR](../../reports/rec_baselines/t08_20261006.md)、[LightGCN](../../reports/rec_baselines/t09_20261006.md)；未整体复制RecBole/参考仓库为自研 |
| 自己实现 | 推荐适配/召回/排序 | `adapters/model.py`已知用户与独立SessionSeedScorer；`retrieval/`、`ranking/`、`pipeline.py`多路召回、RRF、硬约束和证据。匿名seed pooling为本项目策略，不是原生LightGCN新用户推理。模型适配/固定流程/约束测试 |
| 自己实现 | Agent正确性与状态 | `agent/executor.py`有序重复步骤；`state.py`事件/provenance；`router.py`、`loop.py`路由及有界重规划。[先失败后通过](../../reports/plan_executor/t14_20261006.md)、[反馈](../../reports/preference_state/t15_20261006.md)、[故障隔离](../../reports/reliability/t17_20261007.md) |
| 自己实现 | 审计和对照实验 | `evaluation/`公开/隐藏输入、U1/F/A/O、消融、配对CI；runtime/adapters逐尝试计数、写前journal、余额停止、零请求恢复。[T19](../../reports/benchmark/t19_live_20261010.md)、[系统表](../reports/benchmark.md)、[负结果](../reports/ablations.md) |
| 自己实现 | T20交付 | `demo.py`离线推荐CLI；`t20_live_smoke.py`单次授权、持久intent和不重试；cohort恢复只接受公开摘要的精确SHA；`prepare_u1_compat.py`只准备U1源代码/导入settings。[T20验收](../../reports/release/t20_20261010.json)、新增发布回归 |

贡献有正确性、适配与工程价值，不预设全部质量指标改善。LightGCN低于BPR、无内容消融高于O、正式批次未触发重规划等负结果均公开。O/A请求下降仅在本合成协议成立，不是SOTA或线上CTR。

年份/反思的可选bugfix单独保存于`reproduction/patches/`，不替代原结果；未向微软提交或合并PR，不写官方采纳。GroupLens数据许可独立于MIT代码许可。
