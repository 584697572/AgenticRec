# 固定前100候选重排实验（2026-10-14）

**DONE / VERIFIED；NO VERIFIED QUALITY GAIN。**

同一MovieLens1M原180质量用户、3190确认目录与原未来目标，公开3/5/10部喜欢各60人。本轮只改变排序，不改变候选池、隐藏已看处理、数据跨度、原默认或微软源码。180人已用于历史诊断，明确是事后复测；不用本批答案训练或挑参数。参数只由原150 valid开发用户三seed平均Hit选择，先冻结再保存全部正式预测后评分。API0，下载0。

## 方法与真正执行内容

1. 基线为原冻结短序列epoch10/alpha.5，各seed原始前100。在完全相同池内对照线性边界残差、MLP边界残差、相同MLP概率质量目标。15特征明确加入原序列分数/排名，最终为原分数+有界修正；小幅修正上限由strength控制，不重新以热门分数作为基础。
2. 两个已封存train前缀，各三seed实训原短序列10轮，合计60epoch。重排训练分数仅来自目标以前的图与模型；CF/热门也只用相同前缀。避免直接让完整train模型给它自己学过的未来答案生成堆叠训练特征。完整train原冻结模型只在开发/正式推理使用，未用valid/test标签训练。
3. 三种残差各三seed20轮，共180epoch；边界目标比较最高正目标与第10高非正目标，辅助平方正则限制改动。它是Hit@10代理目标，并非精确指标；非正目标只是未观察到未来喜欢，不能称已证实不喜欢。候选无正目标统计但不造答案。
4. 原排序零修正＋三路线×两个epoch×三个strength，共19开发配置，8550条预测。各路线的开发最佳均复测，最终选择包含原排序，主要seed42固定；没有按正式最高seed或路线换结论。

开发选中：`{'epoch': 20, 'route': 'mlp_mass', 'strength': 0.3}`；开发平均指标 `{'hit': 0.23777777777777778, 'mrr': 0.08323985890652558, 'ndcg': 0.04750232653152353, 'recall': 0.022214132415935724}`。

## 原180人同场结果

| 排序版本 | 命中人数/180 | Hit@10 | Recall@10 | NDCG@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|
| 原C1 | 65 | 36.11% | 0.024753 | 0.080418 | 0.149645 |
| 此前混排 | 72 | 40.00% | 0.029605 | 0.083534 | 0.152463 |
| 热门 | 72 | 40.00% | 0.033576 | 0.092815 | 0.168913 |
| 原短序列 seed42 | 75 | 41.67% | 0.031989 | 0.087195 | 0.167288 |
| 原短序列 seed7 | 76 | 42.22% | 0.036020 | 0.093352 | 0.180957 |
| 原短序列 seed2026 | 80 | 44.44% | 0.043422 | 0.100255 | 0.184381 |
| 线性边界微调 seed42 | 75 | 41.67% | 0.034427 | 0.090126 | 0.166980 |
| 线性边界微调 seed7 | 77 | 42.78% | 0.039886 | 0.092666 | 0.173706 |
| 线性边界微调 seed2026 | 79 | 43.89% | 0.042745 | 0.098123 | 0.179347 |
| MLP边界微调 seed42 | 73 | 40.56% | 0.032783 | 0.091703 | 0.167063 |
| MLP边界微调 seed7 | 78 | 43.33% | 0.040240 | 0.094050 | 0.174189 |
| MLP边界微调 seed2026 | 81 | 45.00% | 0.043271 | 0.097776 | 0.181651 |
| MLP概率质量微调 seed42 | 70 | 38.89% | 0.031216 | 0.093417 | 0.167026 |
| MLP概率质量微调 seed7 | 76 | 42.22% | 0.033427 | 0.091343 | 0.155441 |
| MLP概率质量微调 seed2026 | 77 | 42.78% | 0.034244 | 0.094951 | 0.189830 |

| 开发最终候选seed42对照 | Hit差（百分点） | 95%配对区间（百分点） | 三次Holm p | 新命中/丢失 |
|---|---:|---:|---:|---:|
| 对原短序列42 | -2.78 | [-8.33, +2.78] | 1.000000 | 11/16 |
| 对此前混排 | -1.11 | [-8.33, +6.11] | 1.000000 | 21/23 |
| 对热门 | -1.11 | [-7.22, +5.00] | 1.000000 | 15/17 |

最终预定候选 `mlp_mass42` 命中70/180（38.89%）。相对原序列新增11人命中、丢失16人命中。主要统计以上述最终开发选择为准，未通过时不把点估计改善写成稳定增益。

各seed候选前100覆盖：2026: 146/180（81.11%）, 42: 145/180（80.56%）, 7: 147/180（81.67%）。所有重排版本用同一池；前100覆盖不因重排变化，不是前10已实现成功率。

## 训练候选与验证

| 前缀 | seed | 全查询 | 可训练查询 | 候选正目标覆盖 |
|---|---:|---:|---:|---:|
| 0 | 7 | 600 | 346 | 57.67% |
| 0 | 42 | 600 | 330 | 55.00% |
| 0 | 2026 | 600 | 321 | 53.50% |
| 1 | 7 | 747 | 424 | 56.76% |
| 1 | 42 | 747 | 442 | 59.17% |
| 1 | 2026 | 747 | 439 | 58.77% |

本机实际前缀训练92.26秒，残差训练9.85秒，特征构造81.62秒。不是Agent线上延迟或跨环境性能提升。

独立审计从raw train重建两个图/监督及12963前缀训练样本，核验5031特征/池预测重放、8550开发评分和推理、19配置选择、原180输入/目标、2700正式独立评分、2160新排序重放、540原基线及540旧控制、固定100候选和SHA、配对统计全部通过。

审计评分/原始时间边界/池排序/残差与概率质量公式及统计独立重算；神经网络结构/CF特征组件复用项目实现。不称模型算法的完全独立实现证明。**本轮严格Agent任务、LLM请求/token/线上延迟 NOT EVALUATED。**默认未替换。

[预注册协议](top100_rerank_protocol_20261014.md) · [正式聚合](top100_rerank_20261014_validation.json) · [独立审计](top100_rerank_20261014_independent_audit.json) · [开发选择](top100_rerank_20261014_selection.json) · [前缀训练](top100_rerank_20261014_prefix_training.json) · [残差训练](top100_rerank_20261014_residual_training.json) · [候选覆盖](top100_rerank_20261014_features.json)

原始用户/标签/候选/权重仅本地 `optimized/artifacts/comparison/top100_rerank_20261014/`。已完成目录禁止覆盖。

```powershell
$recPython = 'optimized/AgenticRec/.venv-t20/Scripts/python.exe'
$study = 'optimized/artifacts/comparison/top100_rerank_fresh'
& $recPython -m agenticrec.evaluation.top100_rerank prepare --directory $study
& $recPython -m agenticrec.evaluation.top100_rerank prefix --directory $study
& $recPython -m agenticrec.evaluation.top100_rerank cache_training --directory $study
& $recPython -m agenticrec.evaluation.top100_rerank residual --directory $study
& $recPython -m agenticrec.evaluation.top100_rerank freeze --directory $study
& $recPython -m agenticrec.evaluation.top100_rerank evaluate --directory $study
& $recPython optimized/AgenticRec/scripts/audit_top100_rerank.py --directory $study --output "$study/independent_audit.json"
```

本批结束后不按180人的结果追加参数搜索；后续继续须另设预定验证。

## 补充验证与结论

三seed描述性平均Hit：原序列42.78%、线性边界42.78%、MLP边界42.96%、MLP概率质量41.30%。边界MLP仅多0.19百分点，不据此称稳定提升；其主要seed42点估计反而少2人。开发胜者相对原排序的主要区间跨零，不能认定稳定提升或普遍下降，继续保留原默认。此结论不排除其他排序模型/特征有效，只说明本批预定方法未建立收益。

新增4项回归、相关17项一次通过，核验零修正完全保持原分数/排序、有界修正、前10边界梯度/多目标/错误输入、候选固定及平分确定性。沿用已有缺NumPy及Transformer mask警告，不影响本轮已验证运行。

[独立公式审计](top100_rerank_20261014_formula_audit.json) 在冻结训练程序后补充，只读已有权重和预测。另用显式线性/ReLU/tanh公式和独立列表排序重放10710条预测，不调用生产推理函数；用标量参考式验证2302真实训练组的两种损失，上界和全部预测一致。没有改已封存训练/选择/正式推理程序，没有新增调参或付费调用。

```powershell
& $recPython optimized/AgenticRec/scripts/audit_top100_formulas.py --directory $study --output "$study/formula_audit.json"
& $recPython -m pytest optimized/AgenticRec/tests/test_top100_rerank.py optimized/AgenticRec/tests/test_temporal_repair.py optimized/AgenticRec/tests/test_rank_upgrade.py --basetemp=optimized/artifacts/test_runs/top100_rerank_fresh -q
```
