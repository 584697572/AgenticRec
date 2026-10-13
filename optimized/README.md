# 个人优化版本

本部分汇总截至 2026-10-13 的个人代码和研究；微软原始源码与兼容复现在 [original](../original/README.md)。所有已完成研究分支已合入默认分支，历史分支仍保留。

## 代码与实验

`AgenticRec/` 内包含可安装的 `agenticrec` Python 包、配置、脚本、测试和开发文档。已实现 BPR-MF / LightGCN、训练期 ID/图/负采样、已知用户与匿名评分、有序工具执行、硬约束、多轮反馈、规则路由以及可选融合。

`reports/` 集中保存真实模型结果、Agent 对照、消融、失败分析以及后续研究。`history/workspace/` 保存整理前的 README、STATUS、DECISIONS、TASKS 和最初规范；它们是历史记录，不再决定执行顺序。

| 已做优化 | 当前结论 | 证据 |
|---|---|---|
| 三版本全面同场对比 | 原配置/历史no_content/C1严格成功35%/31.33%/94.67%；180用户命中36.11%/32.22%/35%，未证明偏好质量提升 | [320题真实对照](reports/three_way_comparison_20261013.md) |
| 四组准确率优先路线 | C1/C2/C3严格成功95.83%、C4为95%；学习模块未证明增益，C1候选增加token43.68% | [完整比较与消融](reports/frontier_validation_20261012.md) |
| 实际双版本同题评测 | 请求减少45.95%、总token减少85.17%；命中正向但未证明稳定质量/整任务改善 | [真实对比](reports/original_comparison_20261012.md) |
| Agent 请求路由 | 合成 benchmark 成功率同为81.33%，请求减少47.89% | [T19](reports/benchmark/t19_live_20261010.md) |
| BPR / LightGCN 强基线 | 排序质量接近，未证明 GCN 优于 BPR | [三 seed](reports/rec_baselines/frozen_model_validation_20261011.md) |
| RRF / 内容融合 / 学习门控 | RRF小样本外部结果正向；门控无可靠收益，内容消融有负结果 | [融合](reports/fusion_research_20261011.md) |
| LightGCN 计算实现 | 本机配对单轮计算37.67→5.36秒，不代表完整训练或在线加速 | [计算与采样](reports/rec_baselines/lightgcn_optimization_20261011.md) |
| 困难负采样 | GCN/BPR均未证明稳定质量收益 | [困难样本](reports/lightgcn_hard_negative_findings_20261011.md) |
| 最新显式低分监督 | 12次训练，完整valid有观测提升，audit不一致，NO VERIFIED GAIN | [低分监督](reports/lightgcn_explicit_low_findings_20261012.md) |

研究实现完整保留，但不因“最新”而默认启用。既有模型和实验不会被新研究覆盖。81.33%任务成功率、历史test约65% Hit@10、最新valid约47% Hit@10属于不同指标/人群，不能直接混比。

最新三版本实验恢复真实历史T19 no_content模型seed42/推荐器/AgentLoop，独立新用户、同题400轮/系统、统一实际1024输出上限、冻结源码及资源。C1完整通过284/300任务，180位用户命中63人；其余两个版本分别通过105/94题、命中65/58人。C1对两个基线的严格成功配对区间为+54至+65.33、+58至+68.67百分点；四项推荐质量指标差值区间均跨零，不能称算法准确率提升。

相对原配置，C1请求减少41.93%、token减少78.86%；相对历史最佳配置，请求增加11.50%、token增加39.71%。16个继承年份解析失败、三轮通过率50%和拒绝解释0/20保留，没有按正式结果修代码或重跑。原版244项/历史配置198项身份未确认不一概视为虚构；独立来源对齐本身是系统贡献。匿名显式历史也不能衡量完整已知用户GCN效果。[预测前协议](reports/three_way_protocol_20261013.md)、[正式聚合](reports/three_way_comparison_20261013.json)、[独立评分和失败分项](reports/three_way_independent_audit_20261013.json)保留。历史86%来自不同任务定义，不能要求这次重复86%。

新入口为 `python -m agenticrec.evaluation.three_way_comparison`，依次prepare、calibrate、freeze、formal、report；只读核验脚本在 `AgenticRec/scripts/audit_three_way_comparison.py`。已完成目录拒绝覆盖，重复复核须换输出文件名；原始预测、标签和请求日志只留本地忽略目录。本轮没有新训练、下载或默认模型替换。

四组新实验入口是 `python -m agenticrec.evaluation.frontier_experiment`，顺序及范围见[预测前协议](reports/frontier_validation_protocol_20261012.md)。三seed小状态策略、语义GRU/联合精排、目录ID生成器确实训练，四组都进行了真实LLM正式运行；三seed组件对照及18组消融复用已记录解析，没有新增API。C1的Hit@10为45%，C2相同、C3为32.5%、C4为40%；与此前版本42.5%相比尚未证明更懂用户偏好。主要验证了来源目录、过滤及状态继承的作用，不能归因到LightGCN或完整Rank-GRPO/OpenOneRec。[评分/请求独立审计](reports/frontier_independent_audit_20261012.json)、[训练聚合](reports/frontier_training_20261012.json)保留核验依据。

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

新 clone 的安装、数据与模型准备见 [运行指南](AgenticRec/docs/REPRODUCE.md)。Windows/CPU为当前验证范围；GPU/Linux及真实用户满意度没有由此次整理补齐。密钥从被忽略的根 `.env` 读取；新路线按当前用户授权调用并监控真实余额，余额不足停止。本轮三版本校准及正式合计1,672次请求，观察余额窗口减少2.28元，其中正式窗口2.19元，逐版本账单成本未知；有限比较完成后已停止新增调用。

最新迁移验收见 [workspace_consolidation](reports/workspace_consolidation_20261012.md)，当前全局状态见 [STATUS](../STATUS.md)。
