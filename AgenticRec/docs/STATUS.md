# 项目状态

最后更新：2026-10-06（Asia/Shanghai）。G0 资源审计/路线选择闭合，质量实验按规范独立重建路径推进。T06 数据/ID、T07 指标/评测集合、T08 基线/BPR-MF、T09 LightGCN、T10 U1 方法适配与 T11 多路候选/硬约束均已验收；下一批 T12 固定推荐流程。旧模型语义未获证明。

| 任务 | 状态 | 验证与剩余项 |
|---|---|---|
| T00 | DONE | 固定 RecAI 0959ecb05b0794748426e73e6efc1b6b35ec433d，原工作树 pristine |
| T01 | DONE | 原调用链、静态审计和原始失败证据保留 |
| T02 | DONE | Python 3.9.24，174 包离线重建及原资源/工具通过 |
| T03 | DONE（审计/路线） | 原资源及有界缺失记录完成；官方元数据身份核查与独立质量路径已验收，未将旧 checkpoint/矩阵语义/许可视为通过 |
| T04 | BLOCKED（其余条件） | 真实单轮、两轮年份修改和全部45条A1原评测执行链通过；正式三个分支 hit 均0/45；canonical 子集、ID/许可及原 demo/reflection 条件未验收 |
| T05 | DONE | Python 3.11.14，两个隔离环境各15项基础测试通过 |
| T06 | DONE | 官方 MD5/CRC/SHA 和编码检查；1000209 条评分按全局时间拆为 800164/100024/100021；5351 warm 用户、3469 训练候选物品、462887 训练图边；5 项数据/防泄漏测试通过 |
| T07 | DONE | Recall/Hit/MRR/NDCG 手算测试及稳定全集排序；本地 valid/test 固定 cohort，空/无效预测计零；12项专项测试与32项完整回归通过 |
| T08 | DONE | Random、Popularity、BPR-MF 同协议 seed 42 实测；valid/test NDCG@10 与 checkpoint 重算一致，36项测试在两个环境均通过；BPR-MF 未超过热门 |
| T09 | DONE | 小图 float32 稀疏/稠密误差≤1e-6；真实图零 holdout 边；1/2/3 层 valid 实验选中3层，test NDCG@10=0.194047；checkpoint/ID/训练边 SHA 验证通过 |
| T10 | DONE | U1 独立模型适配、已知/匿名路由、原 ToolBox/Buffer/Map 两条真实数据离线轨迹及源码对照通过；LLM 规划未评测 |
| T11 | DONE | 内容/协同/热门三路召回、RRF、硬约束前后校验、结构化失败与真实数据两路 smoke 通过；两环境各 55 项测试通过 |
| T12—T24 | TODO | 完整固定流程、最终多 seed 与 Agent 改进尚未验收 |

## 最新证据

- [T09 LightGCN 实测](../../reports/rec_baselines/t09_20261006.md)：作者论文与固定参考提交、训练期稀疏图、小图数值校验、层数比较及同协议 test 结果。机器结果见 [lightgcn_seed_42.json](../../reports/rec_baselines/lightgcn_seed_42.json)，私有 checkpoint 不入 Git。
- [T08 推荐基线实测](../../reports/rec_baselines/t08_20261006.md)：BPR-MF 小样本过拟合/重载验证、单 seed 全量训练、三模型同 cohort 评测完成。test NDCG@10 Random 0.014081、Popularity 0.195795、BPR-MF 0.192089；无提升宣称。机器结果见 [seed_42.json](../../reports/rec_baselines/seed_42.json)，私有 checkpoint/热门计数不入 Git。

- [MovieLens1M 数据协议](../../reports/t06_movielens_protocol_20261006.md)：官方稳定 ZIP 和条款本地保存、原始行不入 Git；训练期 ID/图/历史/候选与 holdout 分离，冷热分层和全部排除数记录在本地 `artifacts/data_manifest.json` 及公开摘要。
- [固定指标与评测 cohort](../../reports/t07_metrics_protocol_20261006.md)：valid/test 同用 5351 个训练期 warm 用户与 3469 个训练候选，分别481/954名用户有可评正反馈，其余用户显式记录无相关目标。
- [目录核查报告](../../reports/catalog_identity_20261005.md)：9883 个目录条目按官方元数据唯一对应，5 个歧义；12,189 条 source-backed 别名在读评测目标之前生成。45 条旧诊断输入有 28 个可解析、6 个年份冲突、11 个该来源无标题；不是新 benchmark 或质量指标。
- [路线决定](../../reproduction/quality_route_decision_20261005.json)：原包/源码历史未找到训练编号表，按 §5.5 为质量实验采用 B/upstream_rebuilt，原 A1 和三个 0/45 保留。64 项完整回归、范围/重复门禁通过；本轮 0 新 API 请求，共246462字节下载。

- [检查与修复报告](../../reports/reproduction_repairs_20261005.md)：可选年份绝对差与反思历史传递补丁验证；53 项完整回归通过。原版/修复版各在 zero/fixed demo 模式完成真实上游组件的 fixture 反思流程，0 真实请求。动态默认 all-mpnet-base-v2 未缓存，未下载；真实 LLM 下启用这些功能仍 NOT VERIFIED。
- [目录契约检查](../../reproduction/native_runs/20261005_repairs/catalog_contract_summary.json)：补丁改变 1 个旧映射，年份不一致由 4 变 3；严格标题/年份检查为 22 个唯一匹配、17 个缺标题、6 个缺对应年份。这是原目录可达性检查，不证明 checkpoint ID 对齐，也不是新的评测指标。
- [真实多轮与原评测报告](../../reports/a1_multiturn_eval_20261005.md)：93次HTTP200，实际输入184724/输出17220/总计201944 token；高峰估算0.507208CNY，账单null。
- [公开摘要](../../reproduction/native_runs/20261005_session/live_summary.json)：原保存对话、逐样本命中及原函数指标复算一致，分母全部45。第二轮原prompt带此前输入和回答，原memory为4条。
- [失败诊断](../../reproduction/native_runs/20261005_session/failure_diagnostics_summary.json)：21个最终target不可映射、4个映射年份不同、44个target带年份后缀；索引37被截断且无Map，仍在分母中。去掉年份的6个模糊命中是诊断，不是正式结果或模型提升。
- 完整离线联调94次mock、真实0请求；40项全量回归与3项原指标测试分别通过。未知参数退出2、0请求；真实证据Key扫描、重复门禁和22个旧证据的字节检查通过。

## 当前阻塞

1. checkpoint权威标题映射、矩阵行标签和原包独立再分发许可 NOT VERIFIED。目录ID在范围内、权重加载和有限分数不等于语义正确；[源码复核](../../reproduction/source_alignment_review_20261005.json)不能独立恢复训练映射。
2. 作者canonical等条件未对齐，继续限制原论文验收；用户已明确暂不追论文，它们不阻塞独立质量路径。该45条仍仅为A1派生兼容性/诊断输入。
3. 最终多 seed 与模型适配尚未实施。当前 T09 单 seed LightGCN 没有超过 Popularity，不能先看改进结果再调整评价指标；所有模型遵循同一冻结时间划分和 T07 指标。

## 下一任务

按 T10 模型适配与上游方法基线推进：已知用户 scorer 与匿名 SessionScorer 分离。旧权重和未知语义矩阵仅保留原功能对照，不借用来伪装可信质量模型；若以后获得权威旧映射，再独立复核。

## 本机配置与历史

Key由非空环境变量优先、根目录.env次之，仅本机保留。执行命令由执行端负责。旧连接1/1与单轮2/2记录不变；新额度已用93/120，现有任务已关闭，不能删除记录重跑。后续使用剩余额度需沿同一授权计数，不能另起台账重置120次。

[单轮报告](../../reports/live_original_app_20261005.md)、[Key配置](../../reports/local_key_setup_20261005.md)、[环境报告](../../reproduction/environment_dev.json)和旧报告保留历史快照。A1完整验收/A2尚未完成；B独立质量路径已选择，T08/T09 单 seed 结果已记录，最终多 seed 与 Agent 提升仍 NOT EVALUATED。

## T10 验收记录（2026-10-06）

T10 DONE：已知用户和匿名 seed scorer 分离，候选分数对齐、未知用户回退与 checkpoint SHA 拒绝测试通过；上游 `ToolBox`、`CandidateBuffer`、`MapTool` 三份源码逐字节核验后，在 legacy 子进程执行两条真实数据 U1 轨迹。Machine-readable 摘要：`../../reports/upstream_rebuilt/t10_20261006.json`；源码差异：`UPSTREAM_DIFF.md`。两个现代环境各 46 tests passed。当前 in-progress：无；下一任务 T11。Blocker：T04 原资源 ID/许可仍未证实；U1 LLM 规划、冷启动推荐质量、多 seed 尚未评测。

## T11 验收记录（2026-10-06）

T11 DONE：新增 `retrieval/` 和 `ranking/`，冻结 MovieLens1M 目录与 train-positive 边上三路召回、RRF 和双重硬过滤可用。专项 9 tests passed；两个独立 Python 3.11 环境各 55 tests passed；真实数据已知/匿名两路各 3 源、最终各 10 条且违规 0；跨进程哈希种子与真实轨迹 SHA 稳定。报告：`../../reports/candidate_fusion/t11_20261006.md`；机读摘要：`../../reports/candidate_fusion/t11_20261006.json`。当前 in-progress：无；下一任务 T12。T04 原资源问题保持 BLOCKED；T11 质量 NOT EVALUATED。
