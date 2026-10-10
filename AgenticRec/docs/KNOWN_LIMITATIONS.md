# 已知限制与未验证事项

| 范围 | 事实 / 状态 | 影响 |
|---|---|---|
| 原版T04 | BLOCKED：旧checkpoint/矩阵行标签、原包许可和作者canonical协议未全证实；A1已有原App单/多轮及45条派生原评测，正式模糊hit保留0/45 | 不能写论文全部复现；U1是方法级资源/提示适配，不能称U0/A2 |
| 元数据 | 只有title/genres/标题year，没有完整剧情、片长、价格和情绪；评分>=4是建模假设 | 缺失字段澄清，不编造硬事实，不把评分当点击 |
| 评测外推 | 150条规则生成合成test、3seed，3,600为重复执行 | 不对应线上满意度/CTR/CVR，不当3,600独立样本 |
| U1/F解析 | F的ID字符串元素失败，U1计划契约失败 | O/F/U1差距不能单独归因于路由；正式test后未修提示重跑 |
| 负收益 | LightGCN三seed弱于BPR-MF；无内容86%高于O的81.33% | 未证明图模型/当前内容融合的质量增益，模块保留实验性 |
| 重规划 | T19真实触发0次，fixture可测一次重规划 | 工程能力通过，真实质量收益NOT EVALUATED |
| 计时/费用 | 串行预热，首次模型/worker加载不入逐请求延迟；顺序、缓存、时段有干扰；余额变化不是明细账单 | 不称线上QPS或纯路由因果加速，账单未查询保持null |
| 冷启动 | 独立SessionSeedScorer/内容/协同/热门；不学习新用户embedding | 不称LightGCN天然解决匿名用户，身份/历史必须授权 |
| 环境 | 已验Windows x64/Python3.11.14/CPU；legacy独立3.9.24 | Linux/macOS/GPU/云部署NOT VERIFIED；T20新的现代环境与源码快照重建，legacy解释器明确复用 |
| 冷导入 | T20首轮完整测试一项U1桥接失败约97秒；独立检查和后续完整测试通过，未确认根因，耗时与90秒超时吻合 | 保留首轮失败，不伪称代码bug已修，不增加重试/放宽超时 |
| 打包/CLI | 真实路径依赖editable仓库布局；部分功能是模块入口 | 非editable wheel下真实资源NOT VERIFIED；doctor只验基础、fixture使用toy scorer，实际命令见复现指南 |
| 发布许可 | [MovieLens条款](https://files.grouplens.org/datasets/movielens/ml-1m-README.txt)须引用、再分发/商业用途需许可 | 不发布原始/衍生数据、权重、Key、私密trace；代码MIT不覆盖数据 |

SASRec、学习式路由、LLM微调、多Agent、UI和上游PR未开展；可选任务须另行启用。T21在T20之后执行。
