# 公开技术研究核查（2026-10-11）

只使用作者论文/官方仓库；不是SOTA排行榜，也不是这些项目的本机复现。页面经本轮联网核查，GitHub动态HEAD未用作本项目依赖。论文年代、公开代码和实际可运行性分开；未下载权重/安装新框架。

## 方法与可用实现

| 对象 | 可核验思想及出处 | 对当前问题的启示和成本判断 |
|---|---|---|
| InteRecAgent，2023预印本 | [论文](https://arxiv.org/abs/2308.16505)、[固定上游](https://github.com/microsoft/RecAI/tree/0959ecb05b0794748426e73e6efc1b6b35ec433d/InteRecAgent)：推荐工具、记忆、示例规划、反思 | 是方法来源，不是2026新增能力；它不提供本项目独立训练/候选融合有效性的证据 |
| BPR-MF / LightGCN | [BPR](https://arxiv.org/abs/1205.2618)、[LightGCN](https://arxiv.org/abs/2002.02126)、[作者实现](https://github.com/gusye1234/LightGCN-PyTorch) | 成熟协同信号和强基线；实际项目未证明GCN优势。必须先补独立seeds，不能按年代替换 |
| SASRec（2018经典） | [作者代码](https://github.com/kang205/SASRec)：自注意力序列推荐 | 可小规模CPU训练；本项目已有timestamp，但需要新的因果序列目标及充分调优预算。不能称2025新模型；作为未来备选 |
| HSTU（2024基础）、ULTRA-HSTU（2026） | [HSTU原论文](https://arxiv.org/abs/2402.17152)、[官方代码](https://github.com/meta-recsys/generative-recommenders)、[ULTRA-HSTU论文](https://arxiv.org/abs/2602.16986)：长序列模型与系统协同设计，稀疏注意力/模型拓扑 | 学习长短期兴趣和计算预算意识；工业实现依赖GPU/专用算子，当前CPU+稀疏验证用户不能支持同规模验证。外部工业效果不移植为本项目收益 |
| Alibaba TorchEasyRec | [官方仓库](https://github.com/alibaba/TorchEasyRec)、[HSTU文档](https://github.com/alibaba/TorchEasyRec/blob/master/docs/source/models/dlrm_hstu.md)：匹配/排序、多兴趣/多任务、HSTU及SID量化、训练服务特征一致性 | 框架自身引用年份2024，当前能力经核查；不是独立新算法。当前导入整个TorchRec/动态embedding栈代价高；借鉴接口及校验，不替换锁定环境 |
| OpenOneRec（2025-12报告，2026公开模型） | [报告](https://arxiv.org/abs/2512.24762)、[官方仓库](https://github.com/Kuaishou-OneRec/OpenOneRec)：Itemic Tokens、协同与文本对齐、多阶段预训练/后训练及RecIF-Bench | 说明推荐与指令遵循要分别检验。公开1.7B/8B，标准训练约33B token；本机不重训，不下载。MovieLens粗元数据不足支撑同样基础模型主张 |
| OneSearch V1（2025）/V2（2026） | [V1](https://arxiv.org/abs/2509.03236)、[V2](https://arxiv.org/abs/2603.24422)、[作者代码/KuaiSearch入口](https://github.com/benchen4395/onesearch-family)：端到端生成搜索；V2研究潜在推理与自蒸馏 | 搜索query到商品不同于本项目匿名电影种子；学习召回/排序目标对齐。无商品查询/工业行为数据，不复现其训练或借用线上指标 |
| TAIRA（2025预印本，仓库标KDD2026） | [论文](https://arxiv.org/abs/2506.23485)、[作者代码](https://github.com/Alcein/TAIRA)：从成功与纠错轨迹提炼thought patterns、层次规划、多Agent | 思考模式需要真实决策空间与专家/执行轨迹；当前重规划0不足支持规划学习收益。多Agent不适合当前已证实瓶颈，本轮不加入 |
| τ-Rec（2026） | [作者仓库](https://github.com/nbharaths/tau-rec)、[论文DOI](https://doi.org/10.1145/3773078.3831847)：可程序核验约束、受控隐藏意图、多轮任务、重复一致性 | 借鉴公开/隐藏隔离、任务分层及失败保留；其丰富字段/策略与本目录不同，不把其分数拿来横比。不下载目录或发评测API |
| 自适应混合检索的邻近研究（2026） | [vstash](https://arxiv.org/abs/2604.15484)、[Exact Adaptive Hybrid Retrieval](https://arxiv.org/abs/2608.07152)：检索源分歧、动态融合及有限截断问题 | 属检索而非已证实电影推荐改进；说明可靠性与截断值得测。不能把本轮门控说成无先例原创，也不能引用其速度为本项目速度 |

## 三个实际候选比较

| 问题/候选 | 当前确有问题？经典替代及不足 | 数据/算力/强基线 | 可检验消融与决定 |
|---|---|---|---|
| 自适应内容/CF融合 | 有组合负结果及rank目标不一致；完全禁内容是最强简单替代，可能丢冷启动/语义互补，是否真的丢须测 | 现有训练图/metadata/valid；CPU小门控；no_content、固定权重、RRF直接比较；没有免费新LLM质量证据 | 分开recall/ranker/负反馈；gate vs固定gate、去源质量特征、去候选保护；优先实现但不预设胜出 |
| 轻量SASRec/动态兴趣 | 静态用户模型不吸收会话反馈，但未证明顺序是当前最大缺口；最近seed+itemCF更简单 | timestamp可用，CPU训练成本明显高；强BPR/GCN及最近历史控制必需 | 顺序打乱、长短窗口、状态融合；暂后置，不同时开训练课题 |
| 反馈决策/学习规划 | 当前实测replan0，parser混杂；可靠规则可能已经足够 | 标签和真实收益轨迹不足；F/A/规则router共享接口且成本全计 | 不用fixture证明真实质量；暂收敛到局部反馈及融合，不引入RL/多Agent |

生成式推荐、semantic ID/RQ-VAE、LLM增强与偏好对齐值得学习，当前缺少授权细粒度文本、足量跨域交互和GPU训练资源。HSTU/GRPO名称不能解决内容负收益的归因。采用可验证的轻量融合研究属于实际方法尝试，不等于超过以上工业项目或论文方法。
