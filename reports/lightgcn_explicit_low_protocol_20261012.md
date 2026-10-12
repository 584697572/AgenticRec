# LG-LF01：独立显式低分监督预登记

日期：2026-10-12。用户要求继续优化推荐模型，原版资源补证暂停；本轮不宣称微软论文复现或相对其指标提升。

## 假设与唯一变量

训练中 1/2 星反馈包含现有正例—未知排序目标缺失的偏好信号。保持未知负采样排除所有 train 已评分电影，另外为每个有低分的训练正例用户，均匀抽取一个该用户 train 中的 1/2 星电影。3 星不属于低分监督，未来评分不参与训练。

目标为 `mean softplus(s_unknown-s_positive) + lambda R_unknown + alpha mean(mask * softplus(s_low-s_positive))`。所有均值的分母为原 batch 大小，无低分的用户贡献零；原正则项及强度不变。低分不加入图、不进入未知采样、不作未来标签过滤。

唯一新增权重预先固定 alpha=0.25；alpha=0 为完整消融。此值是保守的附加监督系数，并非验证集搜索所得，不试其它系数、不叠加困难负样本。两种条件均计算低分分支，使用独立低分 RNG，保留相同初始化、正例顺序、未知候选及低分抽样。

## 对照与预算

配置：`AgenticRec/configs/lightgcn_explicit_low_20261012.json`。GCN 为 1 层/64 维/均匀层权重；强 BPR 为 32 维。四条件：gcn_uniform/gcn_low25/bpr_uniform/bpr_low25，seeds 7/42/2026，12 次训练。共同 lr=.001、lambda=.0001、batch2048、最多32轮、patience8；最多384 epoch、87168更新（每轮227）。2 个 CPU worker，每个2线程；API0、远程模型0、付费请求0。

数据沿用冻结 ML1M train/ID/候选图/valid，训练低分只能读取 train.parquet，并校验其 SHA。384 人开发选择/早停；先完成 seed42 四条件并按 NDCG@10 冻结，再补7/2026全部对照，最后97人开发审计和全valid汇总。Hit/Recall/MRR 为次要指标，不能根据 Hit 或 audit 改赢家。保留全部负结果及原始预测。

384/97 和 ML1M 原 test 均已暴露，当前结果只能称开发证据；本轮代码拒绝读取历史 test。按用户先平均三seed，再做2000次配对用户 bootstrap；区间未经多重检验校正，不把重复seed当独立用户样本。

## 未暴露评价登记与边界

不继续在97人 audit 上扩参数。后续确认采用固定发布的官方 HetRec 2011 MovieLens 2K，来源 https://grouplens.org/datasets/hetrec-2011/ ，先读取其 README 许可、验证包 CRC、固定 ZIP SHA 和本地不可覆盖快照，再准备 train/valid/test。尚未下载或查看其评分，状态 NOT EVALUATED。

预设新协议为用户内按时间升序（同时间按 item ID）80/10/10划分；train 正例阈值4、低分阈值2、暖用户至少5个train正例；ID/候选/图/已见历史均只来自train，全集候选Top10，冷启动和无相关用户独立报告。该协议与 ML1M 全局时间不同，只比较新数据内部同条件对照，不横比绝对值；同类MovieLens数据不能保证现实用户独立。

外部每种方法只保留本轮已登记的 alpha0/.25、已冻结结构/维度/超参数，3seed、相同早停规则，模型选择只用新valid，所有checkpoint冻结后一次性评test。最小新增下载只在开发证据支持推进时执行；不以可变 latest-small 作为正式研究结果依据，官方明确提示 latest 数据不适合报告研究结果。当前有限实验不包含外部训练，若没有清晰开发收益，不下载大包或继续旧audit网格。

## 验收

独立 dense 梯度/损失 oracle，空低分和padding、train映射/阈值/篡改、RNG配对、toy训练/重载/拒绝覆盖先测试；源码及配置在真实训练前提交，记录Git状态及源SHA。12checkpoint三环境身份字段检查、原始预测四指标重建、6配对流校验、旧文件/规范/权重SHA保护、完整工程回归及已有锁环境专项。第二环境复用checkout及资源，不冒称全新机器训练。

DONE只表示完成本有限研究验收。质量须单独记录，未暴露质量保持 NOT EVALUATED；开发CI包含0或下降记 NO VERIFIED GAIN。默认模型不替换，原规范/upstream/旧报告及权重不改。
