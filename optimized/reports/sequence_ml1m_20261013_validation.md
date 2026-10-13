# 短序列在此前40%数据集与180用户上的同场复测

DONE / VERIFIED；**NO VERIFIED QUALITY GAIN**。

相同MovieLens1M、原180位用户、原确认目录3190部、公开3/5/10部喜欢各60人、原未来test目标。这批标签已经暴露，是事后复测，不是新盲测，也不是微软原论文效果。

| 排序版本 | 命中人数/180 | Hit@10 | Recall@10 | NDCG@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|
| 原C1等权 | 65 | 36.11% | 0.024753 | 0.080418 | 0.149645 |
| 此前混排 | 72 | 40.00% | 0.029605 | 0.083534 | 0.152463 |
| 标准ItemCF | 59 | 32.78% | 0.026466 | 0.070010 | 0.104848 |
| 只推荐热门 | 72 | 40.00% | 0.033576 | 0.092815 | 0.168913 |
| 短序列seed42（预定主要模型） | 75 | 41.67% | 0.031989 | 0.087195 | 0.167288 |
| 短序列seed7 | 76 | 42.22% | 0.036020 | 0.093352 | 0.180957 |
| 短序列seed2026 | 80 | 44.44% | 0.043422 | 0.100255 | 0.184381 |

| 主要短序列对照 | Hit差值（百分点） | 95%配对区间（百分点） | 三次Holm p |
|---|---:|---:|---:|
| 对此前混排 | +1.67 | [-5.00, +8.33] | 1.000000 |
| 对标准ItemCF | +8.89 | [+2.22, +15.56] | 0.049767 |
| 对只推荐热门 | +1.67 | [-5.56, +8.89] | 1.000000 |

三个seed全部保留；预定主要seed42，不事后挑最好seed。三seed平均Hit为42.78%，不是三次真实LLM运行。

主要模型相对旧混排净增3位命中用户、+1.67百分点；配对区间[-5.00,+8.33]跨零，未证明稳定提升。21位原本未命中的用户变为命中，18位反向变化。相对本批固定ItemCF，三次Holm p=.049767通过，但暴露队列结果不能外推全部ItemCF变体。主要模型Recall/NDCG/MRR均比混排点估计略高、仍低于热门，不能宣称全面超过热门。44.44%属于预先保留的另一个seed，不替代主模型结果。

| 公开历史长度 | 原混排命中/60 | 短序列seed42命中/60 |
|---|---:|---:|
| 3 | 25 | 20 |
| 5 | 23 | 31 |
| 10 | 24 | 24 |

## 实际训练与适用范围

复用之前短序列结构与另一数据开发选择的固定epoch10/alpha.5；电影ID不同，所以在1M原train中重新训练。模型保留完整3469连续ID，输出筛至原3190确认目录。固定32维/两头/一层/dropout.1、Adam lr.002/weight_decay1e-4、batch32、三个seed各10epoch。

5089训练用户、15267短历史样本；各用户训练正反馈80%位置切开、输入最后3/5/10、目标后20%，所有监督在train内。评测用户的训练期过去反馈可以参与学习，不能称全新用户隔离训练；推理不见用户ID/隐藏历史/未来标签。训练实测耗时127.46秒；不使用valid/test选模型或epoch，不重训旧对照。

此次只验证排序质量，原C1与混排各180份对照排名与过去真实LLM交付完全匹配。新模型没有新增LLM测试，不填严格任务成功率、调用延迟或token；API请求/新下载均0，默认不替换。

## 验证与失败记录

独立审计通过：15267训练样本从原始train重建、180用户公开历史/未来目标核对、1260份独立质量评分、540份新权重排名重放、720份历史对照逐项一致，其中360份与原真实LLM交付一致；冻结SHA、配对bootstrap/精确检验/Holm核验通过。

相关20项测试全部通过，保留既有缺NumPy警告。准备期遇到相对路径清单错误与完整训练ID维度错误；均在优化器训练/预测前修正，失败准备文件保留，新run2目录冻结后源码和协议不变。

[预测前协议](sequence_ml1m_protocol_20261013.md) · [完整聚合](sequence_ml1m_20261013_validation.json) · [独立审计](sequence_ml1m_20261013_independent_audit.json) · [训练曲线](sequence_ml1m_20261013_training.json)

私密证据目录 `optimized/artifacts/comparison/sequence_ml1m_20261013_run2/`，原数据/用户/标签/逐项排名/权重仍被Git忽略。

```powershell
$study = 'optimized/artifacts/comparison/sequence_ml1m_fresh'
$recPython = 'optimized/AgenticRec/.venv-t20/Scripts/python.exe'
& $recPython -m agenticrec.evaluation.sequence_ml1m prepare --directory $study
& $recPython -m agenticrec.evaluation.sequence_ml1m train --directory $study
& $recPython -m agenticrec.evaluation.sequence_ml1m evaluate --directory $study
& $recPython optimized/AgenticRec/scripts/audit_sequence_ml1m.py --directory $study --output "$study/independent_audit.json"
& $recPython optimized/AgenticRec/scripts/report_sequence_ml1m.py --directory $study
```

已完成目录和公开导出禁止覆盖；再次研究须使用新目录，不依据这批结果反复调参后称盲测。
