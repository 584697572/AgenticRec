# 固定强基线三 seed 验证（2026-10-11）

只补齐已冻结配置的 seed 7/2026，复用 seed 42；没有新网格搜索或测试选参。

| 模型 | valid NDCG@10 mean ± std | 暴露过的 test NDCG@10 mean ± std | test Hit@10 | 参数量 |
|---|---:|---:|---:|---:|
| bpr | 0.145874 ± 0.002523 | 0.228919 ± 0.003741 | 0.650594 | 282304 |
| lightgcn | 0.147595 ± 0.000514 | 0.227695 ± 0.000859 | 0.650943 | 564608 |

GCN−BPR test NDCG差值 -0.001223，用户聚类95%区间 [-0.004936825306480019, 0.0025882318356280306]；**NO VERIFIED GAIN**。

8610 条用户预测四项指标精确重算，六 checkpoint 重载 valid 指标与训练记录完全一致。valid/test 主分母为481/954相关暖用户，无相关用户4870/4397另计。

配置：GCN一层64维、BPR32维；batch2048/lr0.001/reg0.0001/max32 epochs/patience8；seed7/42/2026。图、负采样和映射均仅使用train，不用未来target过滤。

所有逐seed指标、最佳/实际epoch、checkpoint SHA、活跃度层、覆盖度、热门占比、推理时间在机器报告中。训练计时来自并行开发运行，不能比较公平速度；推理计时包含整空间评分和Python排序，不能冒充在线QPS。

首次验证已经完成评分，但在报告路径转换时失败；原输出与错误日志保留。回归先复现相对路径错误后修复，第二次完整验证完成。执行时基于e1cb622，路径修复尚未提交；执行源码SHA及副本保留，不能声称当时工作树clean。

新的完整测试是历史 Previously Exposed Test。此处比旧弱配置提高不等于原创算法收益，也不是相对Microsoft原论文提升。

```powershell
foreach ($trial in @("bpr_l0_d32", "lightgcn_l1_d64")) {
  & AgenticRec/.venv-t20/Scripts/python.exe -m agenticrec.recommendation_search --plan AgenticRec/configs/frozen_model_validation_20261011_seed_7.json --trial $trial
  & AgenticRec/.venv-t20/Scripts/python.exe -m agenticrec.recommendation_search --plan AgenticRec/configs/frozen_model_validation_20261011_seed_2026.json --trial $trial
}
# 若运行目录已存在须在副本配置中使用新 run_id；禁止覆盖历史权重
& AgenticRec/.venv-t20/Scripts/python.exe -m agenticrec.frozen_model_validation --output artifacts/runs/<new_validation_id>
& AgenticRec/.venv-t20/Scripts/python.exe AgenticRec/scripts/report_frozen_validation.py
```
