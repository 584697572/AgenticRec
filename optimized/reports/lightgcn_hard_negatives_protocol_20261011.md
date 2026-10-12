# LG-HN01：混合困难负样本训练（预登记）

用户2026-10-11明确要求根据LG-D02诊断训练新模型。本次只改变未知负样本选择，不叠加低分监督、时间衰减、层权重或模型结构；不修改上游、旧模型或服务默认。

## 假设和边界

LG-D02中GCN随机负配对61.11%梯度较小；最高分池能增加训练难度，但未知高分也可能是真喜欢。H1：每次保留75%随机负样本、25%用8个train-unrated候选中当前模型最高分者，可能改善完整候选的排序。仅是有界假设，不预设成功，不把训练loss变化当质量。思路属于已有困难负采样方向，参照[MixGCF原论文](https://ericdongyx.github.io/papers/KDD21-Huang-et-al-MixGCF.pdf)，本实现没有其hop mixing，不宣称原创或论文复现。

## 固定数据和对照

- 沿用冻结MovieLens1M全局时间划分、5351用户、3469候选、462887正边、所有已评分train-seen过滤及原NDCG@10主指标。不能使用未来喜欢的物品来排除未知负样本。
- 同一原开发分组：fit+select共384人用于每epoch早停及seed42选择，97人仅在选择冻结和全部训练完成后报告。全部valid此前已暴露，因此只能称开发对照，不能独立盲测；所有test本轮不读取。
- GCN1层64维、均匀层权重；BPR32维，沿用此前强配置。lr0.001、reg0.0001、batch2048、max32epoch、patience8、两CPU线程。每epoch相同462887三元组、227次更新；每个模型分别比较uniform（0%困难）和mixed25（25%困难）。0%就是去掉困难采样的单变量消融。
- 四条件×seed7/42/2026，共12个真实训练，上限384epoch/87168次更新，最多2worker×2线程，API0/下载0/云资源0。不扩大比例/池大小/模型参数范围；无收益保留负结果。

## 公平计算和随机性

所有条件均从头训练，共享各seed的初始化规则、训练正边顺序、8候选池与混合随机硬掩码。正边shuffle、候选抽样、混合掩码三条独立随机流；这不同于旧训练器共享Python随机流，因此重训匹配uniform基线，不能冒充旧checkpoint原样复现。

只从train建立boolean已评分表，候选8个互异，无padding；Torch拒绝采样排除已评分与池内重复。不足8未知则明确失败，不偷看未来、不静默改变方案。所有条件都生成并评分8候选，uniform最终取池首元素；mixed25按固定随机掩码选最高分。选择使用detached分数、argmax平分按池中先出现者；最终仍仅一个负片参与原BPR数据loss与相同ego L2，离散选择不反传。GCN仅一次传播，稀疏backend沿用已验证缓存CSR，原文件不改。

## 执行与验收

先写独立损失/梯度、train-seen/padding/重复候选、混合比例/随机流、公平初始化、计划边界、checkpoint身份、拒绝覆盖与toy训练重载测试。正式训练前提交源码/协议/配置，启动时Git clean。

先四个seed42，按384人NDCG分别选GCN/BPR并冻结（并列按名称确定）；随后补全部四条件的7/2026，再进行开发审计。不能根据audit重选或扩网格。保存Git/source/plan/data/group/checkpoint/prediction SHA、每epoch loss/gradient/margin/容易题/实际困难比例、正边及候选流哈希、stdout/stderr/exit及训练时间。

最终报告保留四条件、所有seed、活跃度分层、热门集中度，主比较mixed25减uniform（GCN与BPR分别）；用户内先平均三seed，再paired bootstrap2000、seed42、探索性未多重校正95%区间。辅指标Recall/Hit/MRR不替代主NDCG。区间跨0称NO VERIFIED GAIN；即使开发区间排除0，也只能称开发证据，独立质量仍NOT EVALUATED，默认模型不自动替换。

冻结前912份旧报告/权重/历史诊断和研究元数据SHA，结束核对。正式结果从原始预测重算，权重重载及既有锁环境验证，全量回归通过后任务DONE。失败attempt保留；测试fixture不用于正式指标。
