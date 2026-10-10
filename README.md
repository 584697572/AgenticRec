# AgenticRec

基于 Microsoft RecAI / InteRecAgent 的电影推荐方法级复现与二次开发，包含 BPR-MF / LightGCN、硬约束、多轮偏好、有序计划、规则路由及真实对照实验。

T19 完成 24 个系统/消融/seed run。在150条冻结合成test上，O与A的Strict Success均为81.33%，O每seed请求从190降至99（减少47.89%）。失败、负结果与置信区间均保留。T20验证独立环境与发布材料；T04旧资源ID/许可及论文原条件仍BLOCKED。本项目不是微软官方实现，不声称论文全部复现或线上提升。

```powershell
git clone https://github.com/584697572/AgenticRec.git
cd AgenticRec
```

从 [开发包 README](AgenticRec/README.md) 安装并运行离线演示。完整步骤见 [复现指南](AgenticRec/docs/REPRODUCE.md)，改动见 [贡献清单](AgenticRec/docs/MY_CONTRIBUTIONS.md)，边界见 [已知限制](AgenticRec/docs/KNOWN_LIMITATIONS.md)，实验见 [T19报告](reports/benchmark/t19_live_20261010.md)。

| Upstream | My changes | Evidence |
|---|---|---|
| [InteRecAgent](https://github.com/microsoft/RecAI/tree/0959ecb05b0794748426e73e6efc1b6b35ec433d/InteRecAgent) 的planner、ToolBox、Buffer、Map | 固定源码、隔离兼容环境、U1模型适配 | [源码走读](AgenticRec/docs/source_walkthrough.md)、[差异](AgenticRec/docs/UPSTREAM_DIFF.md) |
| 预制资源与item_seq模型 | 无泄漏MovieLens协议、自行训练BPR-MF/LightGCN、匿名scorer | [真实实验](reports/benchmark/t19_live_20261010.json) |
| 计划覆盖及排除处理 | 有序执行、硬约束、反馈状态、路由与评测 | [测试](AgenticRec/tests)、[消融](AgenticRec/reports/ablations.md)、[失败分析](AgenticRec/reports/failure_analysis.md) |

`RecAI/`独立保存上游历史，不随本仓库分发。保留[MIT代码许可](LICENSE.txt)，数据和模型分别核验许可；数据、权重、环境、Key及私密trace被忽略。

[执行规范](AgenticRec_完整复现与优化执行规范.md)、[STATUS](STATUS.md)、[TASKS](TASKS.yaml)、[DECISIONS](DECISIONS.md)保留依赖、验收与决策。
