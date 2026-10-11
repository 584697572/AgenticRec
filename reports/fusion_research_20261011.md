# 候选融合研究结果（2026-10-11）

真实结构化组件评测；没有新LLM调用。主学习门控结论：**NO VERIFIED GAIN**。

## 实验及分母

ML1M valid用户288/96/97分fit/select/audit；每用户三个训练期状态，先在用户内平均，不当三个独立用户。外部100K先冻结、门控随后冻结再评测；51相关暖用户、694无相关用户。历史ML1M/交互test是Previously Exposed Test。

| 方法 | 开发审计 NDCG@10 | 外部 NDCG@10 | 外部 Hit@10 |
|---|---:|---:|---:|
| popularity | 0.160231 | 0.141483 | 0.418301 |
| legacy_full | 0.106052 | 0.080556 | 0.238562 |
| legacy_no_content | 0.139546 | 0.151001 | 0.398693 |
| recall_only_off | 0.116390 | 0.089612 | 0.281046 |
| negative_penalty_off | 0.105863 | 0.082397 | 0.254902 |
| no_content | 0.149162 | 0.156309 | 0.398693 |
| matched_rrf_no_content | 0.166897 | 0.159976 | 0.411765 |
| fixed_best | 0.143437 | 0.176248 | 0.437908 |
| rrf | 0.156528 | 0.210644 | 0.535948 |
| conditional | 0.144982 | 0.173877 | 0.437908 |
| learned_42_full | 0.149162 | 0.156309 | 0.398693 |
| learned_42_no_quality | 0.149162 | 0.156309 | 0.398693 |
| learned_42_no_protection | 0.149162 | 0.156309 | 0.398693 |

主四项指标/三seed均值标准差、配对用户CI、状态/活跃度层及来源诊断均在机器报告。匹配RRF无内容为后补单变量诊断，不训练或按它选择参数；不能称另一次盲测。

## 验证与结论

全部 42024 个原始预测指标精确重算；480 个选定历史文件SHA不变。

- learned_mean_minus_no_content：NDCG差值 0.000000，95%用户聚类区间 [0.0, 0.0]。
- rrf_minus_legacy_no_content：NDCG差值 0.059643，95%用户聚类区间 [0.011644104460701129, 0.11266971147446116]。
- rrf_minus_matched_rrf_no_content：NDCG差值 0.050668，95%用户聚类区间 [0.014532487037237604, 0.08987205356898925]。
- rrf_minus_legacy_full：NDCG差值 0.130088，95%用户聚类区间 [0.07245800452520838, 0.18894112254238934]。
- negative_penalty_off_minus_legacy_full：NDCG差值 0.001841，95%用户聚类区间 [-0.004555208092605853, 0.008870745592600643]。
- recall_only_off_minus_legacy_full：NDCG差值 0.009056，95%用户聚类区间 [0.0016878136854353565, 0.018305112741962344]。

学习门控若退化为禁用内容或区间跨0，保留负结果，不默认上线。去源质量/候选保护消融即使无差异也完整保留，不声称每个组件有贡献。旧genre排序与一致融合排序的差异，不能冒称LightGCN或LLM规划提升。

## 工程及限制

新FusionPipeline是显式实验注入入口，旧CLI/模型默认保持稳定。门控10个可学习参数、每个候选四正则×三个seed、最多200步；原始候选/来源分数/重叠/保留率/最终标记可在本地diagnose查看。

指标不是满意度或CTR。外部用户少、元数据粗、未新测LLM解析；component latency只含预计算来源后的融合/排序，不能写在线QPS或纯算法加速。失败和计数修复均保留。

## 复现

```powershell
& AgenticRec/.venv-t20/Scripts/python.exe -m agenticrec.fusion_experiment develop --run-id <new_id>
& AgenticRec/.venv-t20/Scripts/python.exe -m agenticrec.fusion_experiment external --run-id <same_id>
& AgenticRec/.venv-t20/Scripts/python.exe AgenticRec/scripts/report_upgrade_research.py
```

训练/数据协议见optimization_proposal与fusion_protocol_20261011.json。正式run要求源码已提交，重复目录拒绝覆盖；报告脚本重建本轮固定run且只允许字节一致写回。
