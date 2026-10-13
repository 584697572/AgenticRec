# 训练时间跨度、监督与候选排序修正结果

**DONE / VERIFIED；NO VERIFIED QUALITY GAIN。**

同一MovieLens1M原180质量用户、3190确认目录、公开3/5/10部喜欢各60人与原未来目标。这批已用于历史评测和诊断，也是本轮方向设计的依据，明确是事后复测，不是新的盲测。训练和配置选择只读取train/原150 valid开发用户，不用test标签拟合或挑参数。原默认未替换。准备阶段复制并封存原标签文件；模型训练和开发选择不消费正式标签，评分只在全部正式预测保存后执行。

## 真正完成的修正

1. 保留原15267条短时辅助任务，新增6687条真实30/90天任务，覆盖471用户；长期切点包含完整同时间戳批次并去重，不虚构半年以后反馈。长期例实际输入至首目标最少30.00天，中位62.13天。
2. 屏蔽切点之前所有正反馈参与softmax竞争；回归测试验证其梯度为0，目标冲突拒绝。masked_ordinal为单独mask对照，long_mixed按25%辅助/75%长期、组内用户等权采样。推理只见公开喜欢，未用隐藏历史过滤。
3. 两序列版本各三seed真实训练10轮，开发选择epoch5/10、alpha.25/.5/.75，共12配置。再独立训练10→32残差MLP排序器，特征图来自train35%/50%锚点以前、目标至少30天以后；训练候选为CF/内容/热门top100并集，不插入目标。loss优化正候选概率质量，不能称Hit@10精确可微目标。
4. 重排器三个seed各20轮，开发8配置。阶段A训练分数不进入重排训练特征；开发可比较并入阶段A top100。共20开发配置按三seed平均Hit、NDCG、预定顺序选，先冻结再一次生成正式预测。

阶段A选中：`{'alpha': 0.5, 'epoch': 5, 'route': 'long_mixed'}`。阶段B选中：`{'alpha': 1.0, 'epoch': 10, 'include_sequence': False}`。最终研究候选按开发数据选 `sequence`，主要seed42，其他seed全部保留。

## 原180用户同场准确率

| 版本 | 命中人数/180 | Hit@10 | Recall@10 | NDCG@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|
| 原C1 | 65 | 36.11% | 0.024753 | 0.080418 | 0.149645 |
| 此前混排 | 72 | 40.00% | 0.029605 | 0.083534 | 0.152463 |
| 只推荐热门 | 72 | 40.00% | 0.033576 | 0.092815 | 0.168913 |
| 标准ItemCF | 59 | 32.78% | 0.026466 | 0.070010 | 0.104848 |
| 原短序列seed42 | 75 | 41.67% | 0.031989 | 0.087195 | 0.167288 |
| 修正序列 seed42 | 75 | 41.67% | 0.030921 | 0.087916 | 0.177092 |
| 修正序列 seed7 | 76 | 42.22% | 0.040941 | 0.095843 | 0.179530 |
| 修正序列 seed2026 | 86 | 47.78% | 0.039098 | 0.092049 | 0.171409 |
| 候选重排 seed42 | 67 | 37.22% | 0.028379 | 0.083482 | 0.141323 |
| 候选重排 seed7 | 70 | 38.89% | 0.027567 | 0.080845 | 0.150774 |
| 候选重排 seed2026 | 69 | 38.33% | 0.028843 | 0.080739 | 0.140833 |

| 最终候选主要seed42对照 | Hit差值（百分点） | 95%配对区间（百分点） | 三次Holm p |
|---|---:|---:|---:|
| 对此前混排 | +1.67 | [-6.11, +9.44] | 1.000000 |
| 对热门 | +1.67 | [-5.56, +8.89] | 1.000000 |
| 对原短序列 | +0.00 | [-7.22, +7.22] | 1.000000 |

主要候选命中75/180（41.67%）。不按最高seed或哪个阶段的test成绩重新换候选。主质量结论以上述预定配对统计为准；其他模型/指标为描述性，不把参数修正本身当作效果提升。

修正序列三seed平均Hit为43.89%，原短序列三seed均值42.78%；这只是重复训练的描述性均值。本轮主要seed42对原短序列命中数相同，21位原未命中的变命中、21位反向变化，收益抵消；Recall点估计略降、NDCG/MRR略升，配对区间均跨零。重排主要版本37.22%，开发未选为最终候选，负结果完整保留。

## 训练和候选证据

阶段A六次真实训练含开发评分共328.46秒，阶段B三次训练共12.38秒（不含特征构造/开发）。时间是本机实测训练运行记录，不是Agent在线延迟或与原版公平耗时比较。

| 重排训练锚点 | 全部监督查询 | 候选含正目标、实际训练查询 | 图正边数 |
|---|---:|---:|---:|
| 0 | 600 | 398 | 162009 |
| 1 | 747 | 513 | 231442 |

候选无正目标的查询计入覆盖统计但不计算loss，不通过添加目标使训练显得可行。两个prefix图与其未来目标由原始train独立重建，future valid/test不进图。虽然mask修正及长期任务已实施，仍保留短时辅助；不能说时间分布完全等同180天的测试。

## 验证范围

独立从raw train重建15267辅助及6687长期样本，核验两个prefix训练图/未来目标、9000开发预测评分与20配置选择、原180公开历史与未来目标、1980正式评分、1080新权重预测重放、900历史对照、冻结SHA、bootstrap/精确检验/Holm均通过。

新增4项监督回归，相关33项一次通过；保留既有缺NumPy/mask警告。**本轮API0、新下载0，严格Agent任务/LLM资源消耗NOT EVALUATED。**新模型在研究代码中可复测，未切生产默认、不改原微软源码或旧评测条件。

[预测前协议](temporal_repair_protocol_20261013.md) · [正式聚合](temporal_repair_20261013_validation.json) · [独立审计](temporal_repair_20261013_independent_audit.json) · [序列训练/开发](temporal_repair_20261013_sequence_training.json) · [重排开发](temporal_repair_20261013_reranker_selection.json) · [重排训练](temporal_repair_20261013_reranker_training.json)

私密证据 `optimized/artifacts/comparison/temporal_repair_20261013/`；原数据、用户/标签、逐项预测和模型权重不公开。

```powershell
$recPython = 'optimized/AgenticRec/.venv-t20/Scripts/python.exe'
$study = 'optimized/artifacts/comparison/temporal_repair_fresh'
& $recPython -m agenticrec.evaluation.temporal_repair prepare --directory $study
& $recPython -m agenticrec.evaluation.temporal_repair sequence --directory $study
& $recPython -m agenticrec.evaluation.temporal_repair reranker --directory $study
& $recPython -m agenticrec.evaluation.temporal_repair freeze --directory $study
& $recPython -m agenticrec.evaluation.temporal_repair evaluate --directory $study
& $recPython optimized/AgenticRec/scripts/audit_temporal_repair.py --directory $study --output "$study/independent_audit.json"
```

已完成目录禁止覆盖；不继续依据本180人的结果追加参数搜索。
