# LightGCN 有限联合搜索（2026-10-10）

单seed开发搜索已执行；新模型test/三seed NOT EVALUATED。原正式模型未替换。

## Completed

- 完成预登记六个训练对照：1/2层×32/64维LightGCN、32/64维BPR；相同batch2048/lr.001/正则1e-4/max32/patience8。
- 仅valid NDCG@10选模型/epoch；独立重载checkpoint并从私密预测复算四项指标。

## Files Created / Modified

- 独立recommendation_search模块、搜索/冻结配置、专项测试、报告生成脚本、本报告和机器报告。
- STATUS、TASKS与ADR-037；旧模型、数据、训练数学实现、正式T19报告不变。

## Commands / Tests

```powershell
& AgenticRec/.venv-t20/Scripts/python.exe -m agenticrec.recommendation_search --plan AgenticRec/configs/recommendation_search_20261010.json --trial <registered_trial>
& AgenticRec/.venv-t20/Scripts/python.exe AgenticRec/scripts/report_recommendation_search.py
```

重复run/trial拒绝覆盖；私密预测、权重、训练曲线及stdout/stderr仅在ignored artifacts/runs/recommendation_search。

## Verification Results

专项 13 项；完整工程回归 266 项，失败 0，错误 0；六个checkpoint重载、原始曲线和预测复算通过；55个历史文件SHA不变。

| 模型 | 层数 | 维度 | 实际轮数 | 最优轮 | valid NDCG@10 | valid HitRate@10 |
|---|---:|---:|---:|---:|---:|---:|
| lightgcn | 1 | 32 | 32 | 27 | 0.147594 | 0.449064 |
| lightgcn | 1 | 64 | 21 | 13 | 0.148072 | 0.453222 |
| lightgcn | 2 | 32 | 32 | 30 | 0.144651 | 0.453222 |
| lightgcn | 2 | 64 | 29 | 21 | 0.146706 | 0.449064 |
| bpr | 0 | 32 | 32 | 29 | 0.148613 | 0.455301 |
| bpr | 0 | 64 | 32 | 29 | 0.148511 | 0.459459 |

所有模型valid主均值分母481人，无相关记录4870人另列，全集3469物品，历史已评分排除规则不变。

## Findings

- 选中LightGCN：lightgcn_l1_d64，最优epoch13；选中增强BPR：bpr_l0_d32，epoch29。
- 相对上一轮1层/32维/8轮，NDCG 0.131417 → 0.148072（相对+12.67%），HitRate 42.41% → 45.32%（+2.91个百分点）。
- 与上一轮比较，配对NDCG差值95%探索区间=[0.006609177782513942, 0.026788072603228078]；HitRate差值区间=[0.0, 0.058212058212058215]。
- 与增强BPR比较，配对NDCG差值=-0.000541，95%探索区间=[-0.008045737453982595, 0.00730970784245584]；HitRate差值=-0.21个百分点，区间=[-0.02702702702702703, 0.02079002079002079]。
- BPR32原前三轮验证曲线与旧运行完全相同：原patience2在第3轮停止，本轮第6轮恢复并继续改善。GCN32前八轮验证NDCG也精确一致。早停/训练预算确实是值得共同检查的缺口，但本实验同时改变上限与patience，不能单独归因。
- 不能将valid收益直接外推为test HitRate或真实用户喜欢概率。配对区间为探索性统计，未做多重比较校正。
- 公共维度和停止策略一致，GCN额外搜索层数；不声称两种模型调参总预算相等。

## Blockers

本轮：None。正式改进模型三seed/test仍NOT EVALUATED；T04旧资源ID/许可阻塞沿用。

## Deviations from Spec

用户明确要求T20后继续优化LightGCN，先LG-S01、T21暂待。遵守原选模指标及数据协议；历史test已暴露，后续开发不称新盲测。

## Current Project State

有限开发搜索已执行，基于valid NDCG选中的两模型配置已冻结到recommendation_search_selected_20261010.json。旧线上/演示模型未自动替换，API请求0、下载0；全部坏结果保留。

## Next Tasks

冻结选中配置再跑seed7/42/2026；按既往测试暴露事实报告正式全集指标和配对区间。若仅开发收益，保持实验性。
