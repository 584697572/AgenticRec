# 个人优化版本

本部分汇总截至 2026-10-12 的个人代码和研究；微软原始源码与兼容复现在 [original](../original/README.md)。所有已完成研究分支已合入默认分支，历史分支仍保留。

## 代码与实验

`AgenticRec/` 内包含可安装的 `agenticrec` Python 包、配置、脚本、测试和开发文档。已实现 BPR-MF / LightGCN、训练期 ID/图/负采样、已知用户与匿名评分、有序工具执行、硬约束、多轮反馈、规则路由以及可选融合。

`reports/` 集中保存真实模型结果、Agent 对照、消融、失败分析以及后续研究。`history/workspace/` 保存整理前的 README、STATUS、DECISIONS、TASKS 和最初规范；它们是历史记录，不再决定执行顺序。

| 已做优化 | 当前结论 | 证据 |
|---|---|---|
| 实际双版本同题评测 | 请求减少45.95%、总token减少85.17%；命中正向但未证明稳定质量/整任务改善 | [真实对比](reports/original_comparison_20261012.md) |
| Agent 请求路由 | 合成 benchmark 成功率同为81.33%，请求减少47.89% | [T19](reports/benchmark/t19_live_20261010.md) |
| BPR / LightGCN 强基线 | 排序质量接近，未证明 GCN 优于 BPR | [三 seed](reports/rec_baselines/frozen_model_validation_20261011.md) |
| RRF / 内容融合 / 学习门控 | RRF小样本外部结果正向；门控无可靠收益，内容消融有负结果 | [融合](reports/fusion_research_20261011.md) |
| LightGCN 计算实现 | 本机配对单轮计算37.67→5.36秒，不代表完整训练或在线加速 | [计算与采样](reports/rec_baselines/lightgcn_optimization_20261011.md) |
| 困难负采样 | GCN/BPR均未证明稳定质量收益 | [困难样本](reports/lightgcn_hard_negative_findings_20261011.md) |
| 最新显式低分监督 | 12次训练，完整valid有观测提升，audit不一致，NO VERIFIED GAIN | [低分监督](reports/lightgcn_explicit_low_findings_20261012.md) |

研究实现完整保留，但不因“最新”而默认启用。既有模型和实验不会被新研究覆盖。81.33%任务成功率、历史test约65% Hit@10、最新valid约47% Hit@10属于不同指标/人群，不能直接混比。

## 当前本地运行

从仓库根目录执行，均无需付费 API：

```powershell
python workspace.py prepare
& optimized/AgenticRec/.venv-t20/Scripts/python.exe -m agenticrec.cli doctor --offline
& optimized/AgenticRec/.venv-t20/Scripts/python.exe -m agenticrec.cli fixture
& optimized/AgenticRec/.venv-t20/Scripts/python.exe -m agenticrec.fusion_demo --request optimized/AgenticRec/configs/fixed_request.example.json --model bpr --profile validated-20261011 --mode rrf
```

第三条使用已验证的真实训练模型与显式选择的RRF；需要本地数据和权重。`fixture`仅为六部自造电影、toy scorer与FakeLLM的功能演示，不能拿来衡量推荐准确率。

```powershell
Push-Location optimized/AgenticRec
& .venv-t20/Scripts/python.exe -m pytest -q
Pop-Location
```

新 clone 的安装、数据与模型准备见 [运行指南](AgenticRec/docs/REPRODUCE.md)。Windows/CPU为当前验证范围；GPU/Linux及真实用户满意度没有由此次整理补齐。真实LLM预算配置与 `.env` 保持现有机制，本次整理新增请求0。

最新迁移验收见 [workspace_consolidation](reports/workspace_consolidation_20261012.md)，当前全局状态见 [STATUS](../STATUS.md)。
