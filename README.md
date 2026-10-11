# AgenticRec

基于 Microsoft RecAI / InteRecAgent 的电影推荐方法级复现与二次开发，包含 BPR-MF / LightGCN、硬约束、多轮偏好、有序计划、规则路由及真实对照实验。

T19 完成 24 个系统/消融/seed run。在150条冻结合成test上，O与A的Strict Success均为81.33%，O每seed请求从190降至99（减少47.89%）。失败、负结果与置信区间均保留。T20验证独立环境与发布材料；T04旧资源ID/许可及论文原条件仍BLOCKED。本项目不是微软官方实现，不声称论文全部复现或线上提升。

2026-10-11完成[候选融合技术研究](reports/technical_upgrade_20261011.md)：补齐冻结BPR/GCN三seed、内容负收益单变量控制、公开信号学习门控、外部冻结评价及失败案例。学习门控和GCN对强BPR均为 **NO VERIFIED GAIN**。目标一致的RRF在外部51相关暖用户上NDCG@10从旧无内容的0.1510到0.2106，但开发审计未显示稳定优势，次要对照区间未做多重校正；不作跨域、论文或线上提升主张。最终完整295项回归及新锁环境35项专项通过；[发布字节补充验收](reports/distribution_integrity_20261011.md)验证Git导出与实际证据SHA一致，保留之前292项时点报告。

随后完成[LightGCN采样/聚合研究](reports/rec_baselines/lightgcn_optimization_20261011.md)：6条件×3seed/386epoch，准确率 **NO VERIFIED GAIN**；相同初始化与227更新的本机暖一轮训练计算37.67→5.36秒，未计加载/验证，仅一次配对，不称全训练/在线加速。311完整回归及既有独立锁环境16专项通过。新代码保留在实验分支，默认模型未替换。

新增推理为显式选择，旧默认流程不变；需先按报告重建数据与训练已验证的checkpoint：

```powershell
& AgenticRec/.venv-t20/Scripts/python.exe -m agenticrec.fusion_demo --request AgenticRec/configs/fixed_request.example.json --model bpr --profile validated-20261011 --mode rrf
```

强基线见[三seed结果](reports/rec_baselines/frozen_model_validation_20261011.md)，融合见[指标与消融](reports/fusion_research_20261011.md)、[真实失败案例](reports/fusion_failure_cases_20261011.md)，近期方法核查见[一手研究](reports/research_landscape_2026.md)。本轮新增付费请求0，旧证据与权重不覆盖。

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
