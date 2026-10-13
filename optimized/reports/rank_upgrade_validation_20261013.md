# 四条推荐准确率优化路线：实际训练与同场结果

四条路线按授权顺序全部完成，**DONE / VERIFIED；NO VERIFIED GAIN**。短序列模型最终Hit@10点估计最高，但预定八次比较校正后，没有路线证明稳定超过热门榜或此前混排。保留研究配置，不替换默认模型。

## 最终同场结果

官方HetRec2011 MovieLens2k v2，7541部训练期电影；300位最终用户，公开最近3/5/10部喜欢，各100人。同一未来正反馈目标、全目录、前10名、相同公开已看排除。Hit@10表示前十部至少命中一部；Recall衡量命中目标的比例，NDCG/MRR考虑位置。

| 方法 | Hit@10 | Recall@10 | NDCG@10 | MRR@10 | 实际命中人数/300 |
|---|---:|---:|---:|---:|---|
| 热门榜 | 20.00% | 0.023770 | 0.043389 | 0.085366 | 60 |
| 标准完整邻居ItemCF | 25.33% | 0.031107 | 0.049268 | 0.084341 | 76 |
| 纯内容TF-IDF | 5.67% | 0.004971 | 0.008186 | 0.017735 | 17 |
| C1等权RRF方法适配 | 21.33% | 0.029035 | 0.046664 | 0.088015 | 64 |
| 此前1:.5:.25混排方法适配 | 24.67% | 0.031353 | 0.049164 | 0.090135 | 74 |
| ①热门残差学习排序 | 20.78% | 0.023866 | 0.045611 | 0.091369 | 57 / 65 / 65 |
| ②改进ItemCF | 23.00% | 0.028648 | 0.046334 | 0.085468 | 69 |
| ③短序列＋热门 | **25.89%** | 0.027674 | 0.045506 | **0.098937** | 76 / 79 / 78 |
| ④LightGCN匿名适配＋热门 | 20.11% | 0.025155 | 0.044847 | 0.087617 | 60 / 60 / 61 |

学习路线三个seed顺序为7、42、2026。主指标先在每位用户内平均seed，再平均300人；未选择最佳seed，未将900输出当900独立用户。改进ItemCF与五个简单对照为确定性方法。

| 路线 | 相对热门Hit差值（百分点） | 配对95%区间（百分点） | 八次Holm校正p | 相对此前混排差值（百分点） |
|---|---:|---|---:|---:|
| 学习排序 | +0.78 | [-3.44,+5.00] | 1.0000 | -3.89 |
| 改进ItemCF | +3.00 | [-1.33,+7.33] | 1.0000 | -1.67 |
| 短序列 | +5.89 | [+1.22,+10.67] | **0.1168** | +1.22 |
| LightGCN匿名适配 | +0.11 | [-0.56,+0.89] | 1.0000 | -4.56 |

短序列相对热门的未校正p=0.0146，但Holm后p=0.1168，未通过预定0.05阈值。不能只引用单项区间大于零而写“显著提升”。相对此前混排Hit区间[-4.00,+6.44]百分点，也未确认优势。相对标准ItemCF只高0.56百分点，是描述性比较，不另做事后显著性宣称。

短序列的Recall/NDCG点估计低于混排和标准ItemCF，不能说质量全面提升。其三档历史Hit为24.33%/24.67%/28.67%；各档用户不同，不能归因为增加历史长度的因果收益，也不用单档代替整体结论。

## 实际训练与选择

1. **学习排序**：协同、内容、热门各100的自然候选并集，十特征32宽MLP，三个seed各40轮。1242训练查询中1008个池内有正目标，234个没有，按协议跳过且记录，禁止插入未来目标。开发选择epoch10、残差alpha1。
2. **改进ItemCF**：36固定配置实际验证，涵盖邻居50/100/200、收缩10/50、IUF开关、三种热门混合。选择50邻居、收缩10、无IUF、alpha1，最终低于完整邻居ItemCF。
3. **短序列**：32维、一层两头因果Transformer，公开历史最多10部，未来多正反馈softmax；三个seed各30轮。选择epoch10、模型/热门标准化混合alpha0.5。这是SASRec式编码器任务适配，与原next-item设置不同。晚期损失继续下降但开发集未选中，不能以训练损失宣称推荐更好。
4. **LightGCN匿名适配**：三个seed各150次真实BPR更新，每次8192训练三元组、二层均匀LightGCN、32维，负例排除训练正反馈。初始损失约0.693，最终0.258/0.259/0.266，权重确实变化。冻结传播后的电影向量，训练公开喜欢均值残差适配器，各30轮。含未适配均值控制共12配置；选择适配epoch10、alpha0.25。用户向量只用于训练，匿名推理只读公开电影向量。本轮约等于热门，不代表LightGCN普遍无效。

66个共享开发配置、三个学习seed；27个排序/适配检查点及3个GCN电影向量文件留本地。四条全部开发完成才冻结，4500最终排名全部保存后才读目标评分；没有根据最终结果扩网格或改配置。

## 数据和实验边界

原1M用户和标签已多次历史暴露，此次新增[官方稳定HetRec2011资源](https://grouplens.org/datasets/hetrec-2011/)。旧zip仅返回256字节说明，官方目录有效文件为[MovieLens2k v2](https://files.grouplens.org/datasets/hetrec2011/hetrec2011-movielens-2k-v2.zip)。只下载18856822字节，SHA256 `6f03917faa2a2914a4f100a130f4fd40412785b0629950ef72e605406c91f52e`，全ZIP CRC通过。包内855598评分，毫秒时间戳、Latin-1元数据。非商业研究许可；原数据、用户、目标、排名和模型不公开分发。

只用官方ID/标题/年份/类型，不用IMDb/RT评分或未来标签。全局50/70/85%时间边界为2006-04-12、2007-03-31、2008-02-01，同时间戳归后期。排序特征前50%图、监督50—70%；正式协同图及模型训练信息止于70%。70—85%仅选配置，85%以后最终评分。414训练监督、180开发、300最终用户互异。

图可含评价用户过去训练反馈，不含其未来目标。公开最近3/5/10部来自决策时点之前，匿名推理不读真实ID/完整隐藏历史。公开输入含当时已发生的开发期反馈不等于按开发标签训练参数。目录仅前70%出现过正反馈的7541部电影。公开电影数据不声称现实用户与1M绝无重叠。

**25.89%不能与旧40%相减**：数据、时间窗口和人群变了。C1是新数据上的离线方法适配，微软原App本轮未运行，不称论文复现。自然语言、多轮条件、拒绝、严格Agent成功率、LLM延时和token未评测。

## 验证和资源

- 全套 **436 passed / 0 failed / 4 warnings，186.62秒**。新增9算法测试、6审计测试，涵盖独立CF/IUF数学、因果注意力、公开历史隔离、目标损失梯度、漏seed/伪造分数拒绝。首次18组件与旧基线测试通过；审计路径修正后6相关测试再次通过，不相加称442项。
- 独立审计从原ZIP重建855598评分、电影映射、两训练图、训练/开发/最终用户、公开历史和目标；重算4500评分、300热门、300标准ItemCF排名，全一致。
- 66配置跨seed选择、全部学习检查点与源码/输入/缓存/权重SHA通过；独立重算10000次bootstrap、10000次符号置换、八次Holm一致。
- 初次审计因Windows清单反斜杠与审计器正斜杠匹配失败；仅改审计器原生Path拼接。冻结训练器、标签、排名和指标不变。
- CPU环境Python3.11.14/Torch2.0.1+cpu/PyArrow21.0.0，训练4线程；API请求/费用 **0**，新增依赖/外部模型下载 **0**，只新增18.9MB官方数据。缓存/模型本地生成。旧Torch存储/注意力/CSR及NumPy缺失警告已记录，没有因此失败。
- 上游固定 `0959ecb05b0794748426e73e6efc1b6b35ec433d` 且干净，默认和旧结果保留，用户paper-notes未动。Blocker：None。

## 证据与再现

[预测前协议](rank_upgrade_protocol_20261013.md)、[最终聚合](rank_upgrade_validation_20261013.json)、[真实训练及全部开发配置](rank_upgrade_training_20261013.json)、[独立审计](rank_upgrade_independent_audit_20261013.json)、[测试记录](rank_upgrade_tests_20261013.json)。私密原始证据在 `optimized/artifacts/comparison/rank_upgrade_20261013/`。协议保留写入时NOT EVALUATED，后续状态由本报告记录。

根目录使用现有CPU环境；新机器先安装项目固定依赖，再下载上述v2到 `optimized/data/raw/hetrec2011-upgrade-20261013/hetrec2011-movielens-2k-v2.zip` 核对SHA。无需解压，不提交原包。使用全新输出目录：

```powershell
$study = 'optimized/artifacts/comparison/rank_upgrade_recheck'
$recPython = 'optimized/AgenticRec/.venv-t20/Scripts/python.exe'
& $recPython -m agenticrec.evaluation.rank_upgrade prepare --directory $study
& $recPython -m agenticrec.evaluation.rank_upgrade route1 --directory $study
& $recPython -m agenticrec.evaluation.rank_upgrade route2 --directory $study
& $recPython -m agenticrec.evaluation.rank_upgrade route3 --directory $study
& $recPython -m agenticrec.evaluation.rank_upgrade route4 --directory $study
& $recPython -m agenticrec.evaluation.rank_upgrade freeze --directory $study
& $recPython -m agenticrec.evaluation.rank_upgrade evaluate --directory $study
& $recPython -m agenticrec.evaluation.rank_upgrade report --directory $study
& $recPython optimized/AgenticRec/scripts/audit_rank_upgrade.py --directory $study --output "$study/independent_audit.json"
```

每步exit0才下一步。旧目录禁止覆盖；复查已经暴露的最终集合不再称新盲测，也不用其继续调参。

下一候选优先研究短序列的时间偏好与训练覆盖，保留ItemCF/热门/混排对照。本轮有限验证结束即停止，不自动新增搜索/LLM调用或切默认；实际Agent收益须另外固定新数据与系统同场协议。
