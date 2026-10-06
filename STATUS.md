# 工作区状态入口

最后更新：2026-10-06（Asia/Shanghai）。详细状态见 [AgenticRec/docs/STATUS.md](AgenticRec/docs/STATUS.md)，最新证据见 [T09 LightGCN 实测](reports/rec_baselines/t09_20261006.md)。

T06 DONE：官方稳定 MovieLens1M ZIP 的 MD5/CRC/SHA 验证通过；1,000,209 条评分按全局时间分成 800,164/100,024/100,021。训练期独立建立 5,351 个 warm 用户、3,469 个候选电影 ID、462,887 条正反馈图边；验证/测试冷启动与非正反馈保留计数。

T07 DONE：[固定指标协议](reports/t07_metrics_protocol_20261006.md)手算 Recall/Hit/MRR/NDCG 与稳定全候选排序通过，valid/test 评测用户与相关集已在本地冻结。valid/test 分别有481/954名 warm 用户拥有可评正反馈；空输出、失败及非法输出留在分母计零。

T08 DONE：[Random/Popularity/BPR-MF 实测](reports/rec_baselines/t08_20261006.md)完成 fixture、seed 42 全量训练、早停、checkpoint 重载和同协议 valid/test 指标。test NDCG@10 分别 0.014081/0.195795/0.192089；BPR-MF 本轮没有超过热门。T08 验收时两个独立环境各36项测试通过，指标从 checkpoint 重算一致；最终3 seeds 尚待后续配置确定。

T09 DONE：[LightGCN 训练与校验](reports/rec_baselines/t09_20261006.md)的稀疏/稠密小图传播误差≤1e-6、训练图与 holdout 零交叉、1/2/3 层 valid 选择和 seed 42 test 实测均通过。选中 3 层第7 epoch，test NDCG@10=0.194047；高于本轮 BPR-MF、低于 Popularity。模型与 ID/训练边 SHA、checkpoint 重算通过。下一任务 T10 模型适配与上游方法基线。

此前修复批次见[修复报告](reports/reproduction_repairs_20261005.md)：两项可选 upstream_compat 补丁及当时53项测试通过；原组件 fixed demo/反思完成 fixture 联调。该批未新增请求/下载，原版输入、预测与三个0/45保留。本批来源核查和路线更新见下文。

项目仓库：[584697572/AgenticRec](https://github.com/584697572/AgenticRec)，public；根项目 `origin` 指向该仓库，`master` 跟踪 `origin/master`。创建与首次推送已核验，见 [发布报告](reports/public_repository_20261005.md)。上游 `RecAI/` 保留独立远程与固定版本。

- DONE：T00、T01、T02、T03、T05。T03 完成资源审计/缺失记录/路线选择，未将旧模型语义标为通过；现代开发基础保持原验收。
- T03 审计结论：官方 MovieLens10M 元数据中 9883 个目录身份唯一对应、5 个歧义；source-backed 别名解析使旧 45 条诊断输入从 22 到 28 个可解析，另 6 个年份冲突、11 个该来源未收录标题。旧 checkpoint/矩阵语义与预制包独立许可仍 NOT VERIFIED，原数据和指标不变。见[目录核查报告](reports/catalog_identity_20261005.md)。
- T04 BLOCKED（其余验收前置）：真实单轮及两轮条件修改 VERIFIED；45 条 A1 派生输入的原 recbot/random/加权 popularity 评测链已完成，正式原模糊 hit 均为 0/45。新会话实际93次HTTP200、201944 token，高峰费率估算0.507208 CNY；账单未查询。canonical 子集、权威 ID/许可及原 demo/reflection 条件仍未验收。
- T06、T07、T08、T09 DONE；T10—T24 TODO。独立数据、指标协议、BPR-MF 与 LightGCN 单 seed 真实结果已固定；最终多 seed 与 Agent 改进仍 NOT EVALUATED。

当前路线：保留 A1 原功能执行证据，质量实验按 ADR-015 采用规范 B/upstream_rebuilt 的独立数据/模型路径。用户要求暂不追论文；作者 canonical 条件不再阻塞独立新实验。原规范和固定 upstream 未改，不把新的 raw_movie_id 套进旧 embedding。旧报告/证据保持历史快照，不重置额度。

此前真实运行批次的40+3项测试、原指标复算和22个旧证据检查保留。原映射21个target未知、4个年份不同、索引37截断且无Map，全部45个分母不变；年份剥离后的6个模糊诊断不替代正式结果。当前source-backed别名新增6个解析与该旧模糊命中不是同一指标。

Key仍按环境变量优先、根目录`.env`次之，不入Git/哈希清单。原额度仍已用93/120，本轮新 LLM 请求0。此前 MovieLens10M 元数据只用于旧目录核查；官方 MovieLens1M 是独立新实验的主数据集。当前优先 T10；后续模型须用同一冻结划分，不能事后换指标选有利结果。
