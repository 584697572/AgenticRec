# 项目状态

最后更新：2026-10-07（Asia/Shanghai）。G0 资源审计/路线选择闭合，质量实验按规范独立重建路径推进。T06—T16 与 T18 的数据、指标、模型、U1 适配、候选/约束、固定流程、LLM runtime、有序执行器、多轮状态、规则路由和交互评测均已验收；下一批为 T17 故障注入、安全和会话隔离。旧模型语义未获证明。

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
| T12 | DONE | 结构化零 LLM 固定流程与 CLI；成功/匿名/排除/无解真实场景，证据字段、双环境 61 项测试及稳定轨迹 SHA 通过 |
| T13 | DONE | 严格 JSON schema、FakeLLM/live 门禁、逐尝试预算、401/429 分类、单次 timeout 与整轮 deadline；专项22项及两环境各88项完整回归通过，真实请求0 |
| T14 | DONE | 固定上游同名工具覆盖先红；新 PlanStep 列表保留重复/顺序/参数/trace，严格白名单与预校验；专项11项及两环境各99项通过 |
| T15 | DONE | 不可变 PreferenceState、事件化 patch、provenance、显式反馈优先、局部否定、原子冲突、幂等与 profile version；专项11项及两环境各110项通过 |
| T16 | DONE | DIRECT/PERSONALIZED/AGENT/CLARIFY、F/A/O、dev-only 阈值、最多一次重规划和四类 FakeLLM 轨迹；专项13项及两环境各136项通过，test 文件读取0 |
| T17 | TODO | 故障注入、安全和会话隔离尚未验收 |
| T18 | DONE | development/test 各150条、各40条多轮；公开/隐藏分离，用户/模板组隔离，完整分母评分器、16条人工核验和全空防刷通过 |
| T19—T24 | TODO | 系统基线、消融、多 seed、发布与最终审计尚未验收 |

## 最新证据

- [T16 规则路由与有限闭环](../../reports/agent_loop/t16_20261007.md)：结构化简单请求不进入 Agent；FakeLLM 成功、一次重规划、澄清和工具预算耗尽轨迹可重放。阈值仅由 development 公开输入选择，F/A/O 效果仍 NOT EVALUATED。

- [T18 交互评测冻结](../../reports/interactive_eval/t18_20261007.md)：development/test 各150条，八类场景与双层输入；test 隐藏目标不在公开对象中，全空输出的 Strict Success 为0，LLM simulator 默认关闭。

- [T15 多轮偏好状态](../../reports/preference_state/t15_20261006.md)：显式反馈不会被模型推断覆盖，单物品否定不扩展为类型否定，冲突 patch 原子拒绝，重复事件不重复写入；session 清理保留授权训练历史。

- [T14 有序计划执行器](../../reports/plan_executor/t14_20261006.md)：固定上游要求 `first-query → second-query` 实际只执行后者；新链路完整保留重复工具步骤，所有坏计划在工具运行前拒绝，上游源码未改。

- [T13 LLM/runtime](../../reports/llm_runtime/t13_20261006.md)：默认 FakeLLM 离线可运行，未知 transport 按 live fail-closed；每次真实尝试预记账，unknown usage/cost 保持 null，SDK 重试关闭并由 adapter 统一分类。

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
3. 最终多 seed 与 Agent 对照尚未实施。当前 T09 单 seed LightGCN 没有超过 Popularity，不能先看改进结果再调整评价指标；所有模型遵循同一冻结时间划分和 T07 指标。

## 下一任务

执行 T17：注入坏 JSON、timeout、429、无候选和 prompt injection，并验证同名用户、异步会话、反馈及缓存互不串扰。不得无限重试、越权执行工具、跨 session 泄漏或绕过硬约束。

## 本机配置与历史

Key由非空环境变量优先、根目录.env次之，仅本机保留。执行命令由执行端负责。旧连接1/1与单轮2/2记录不变；新额度已用93/120，现有任务已关闭，不能删除记录重跑。后续使用剩余额度需沿同一授权计数，不能另起台账重置120次。

[单轮报告](../../reports/live_original_app_20261005.md)、[Key配置](../../reports/local_key_setup_20261005.md)、[环境报告](../../reproduction/environment_dev.json)和旧报告保留历史快照。A1完整验收/A2尚未完成；B独立质量路径已选择，T08/T09 单 seed 结果已记录，最终多 seed 与 Agent 提升仍 NOT EVALUATED。

## T10 验收记录（2026-10-06）

T10 DONE：已知用户和匿名 seed scorer 分离，候选分数对齐、未知用户回退与 checkpoint SHA 拒绝测试通过；上游 `ToolBox`、`CandidateBuffer`、`MapTool` 三份源码逐字节核验后，在 legacy 子进程执行两条真实数据 U1 轨迹。Machine-readable 摘要：`../../reports/upstream_rebuilt/t10_20261006.json`；源码差异：`UPSTREAM_DIFF.md`。两个现代环境各 46 tests passed。当前 in-progress：无；下一任务 T11。Blocker：T04 原资源 ID/许可仍未证实；U1 LLM 规划、冷启动推荐质量、多 seed 尚未评测。

## T11 验收记录（2026-10-06）

T11 DONE：新增 `retrieval/` 和 `ranking/`，冻结 MovieLens1M 目录与 train-positive 边上三路召回、RRF 和双重硬过滤可用。专项 9 tests passed；两个独立 Python 3.11 环境各 55 tests passed；真实数据已知/匿名两路各 3 源、最终各 10 条且违规 0；跨进程哈希种子与真实轨迹 SHA 稳定。报告：`../../reports/candidate_fusion/t11_20261006.md`；机读摘要：`../../reports/candidate_fusion/t11_20261006.json`。当前 in-progress：无；下一任务 T12。T04 原资源问题保持 BLOCKED；T11 质量 NOT EVALUATED。

## T12 验收记录（2026-10-06）

T12 DONE：`pipeline.py` 与 CLI 完成严格结构化的固定推荐链，输出 ID、元数据、各路名次和个性化/RRF 分数。专项 6 tests、两环境各 61 tests 通过；真实四场景与无凭据 CLI 通过，规划/LLM/远程请求为0，连续轨迹 SHA 稳定。报告：`../../reports/fixed_pipeline/t12_20261006.md`；机读摘要：`../../reports/fixed_pipeline/t12_20261006.json`。下一任务 T13。T04 保持 BLOCKED，T12 质量 NOT EVALUATED。

## T13 验收记录（2026-10-06）

T13 DONE：`adapters/llm.py` 与 `runtime/` 完成 provider-neutral chat、严格 JSON object schema、FakeLLM/live 分离、请求/费用上界账本、timeout/deadline 和唯一外层重试。专项 22 tests、两环境各 88 tests、compileall 均通过；401不重试、429尝试逐次计数，unknown usage/cost 为 null+reason，未授权 live 不触达 transport。本轮真实请求0。报告：`../../reports/llm_runtime/t13_20261006.md`；机读摘要：`../../reports/llm_runtime/t13_20261006.json`。下一任务 T14。T04 保持 BLOCKED，供应商 live transport 与 Agent 质量 NOT EVALUATED。

## T14 验收记录（2026-10-06）

T14 DONE：固定上游 `ToolBox` 的重复调用保留要求重新实测 exit 1，仅执行 `second-query`；独立 `agenticrec.agent.executor` 使用严格 `PlanStep[]`，保留同工具重复调用、顺序、原参数、step_id 和 trace，并在执行前拒绝未知工具、重复 step_id 与坏参数。专项11 tests、两环境各99 tests、compileall 均通过；`RecAI/` 未修改，本轮真实请求0。报告：`../../reports/plan_executor/t14_20261006.md`；机读摘要：`../../reports/plan_executor/t14_20261006.json`。下一任务 T15。T04 保持 BLOCKED，Agent 效果 NOT EVALUATED。

## T15 验收记录（2026-10-06）

T15 DONE：新增 `agenticrec.agent.state`，实现不可变多轮状态、严格 `PreferencePatch`、逐字段来源/轮次/置信度证据、显式优先级、物品级局部否定、冲突原子澄清、事件幂等和缓存版本。专项11 tests、两环境各110 tests、compileall 均通过；人工确定性轨迹验证 session 清理只删除会话状态并保留授权训练历史。`RecAI/` 未修改，本轮真实请求0。报告：`../../reports/preference_state/t15_20261006.md`；机读摘要：`../../reports/preference_state/t15_20261006.json`。按依赖下一任务 T18，完成后再执行 T16。T04 保持 BLOCKED，推荐与 Agent 效果 NOT EVALUATED。

## T18 验收记录（2026-10-07）

T18 DONE：冻结 development/test 各150条交互 episode，各含75条结构化、75条文本和40条多轮，覆盖八类场景；公开输入与 evaluator-only target/反馈/故障分别序列化，用户组和模板组跨 split 无交集。评分器保留缺失/失败 attempt，统计约束、数量、成功/澄清、turn/tool、latency、token/request 和 fallback/abstain；未知 usage 保持 null。专项13 tests、两环境各123 tests、compileall 均通过；16条预选样本人工核验为 VERIFIED，全空输出在两套各150条上的 Strict Success 均为0。本轮下载0、真实请求0。报告：`../../reports/interactive_eval/t18_20261007.md`；机读摘要：`../../reports/interactive_eval/t18_20261007.json`。下一任务 T16。T04 保持 BLOCKED，系统效果 NOT EVALUATED。

## T16 验收记录（2026-10-07）

T16 DONE：新增规则 Router 和有限 AgentLoop，支持 DIRECT/PERSONALIZED/AGENT/CLARIFY 及 fixed_pipeline/always_agent/ours_router 三种可比模式。development 公开150条上的可见输入契约选择阈值0.75，路由计数75/7/26/42，test 文件读取0；该契约一致性不是系统质量指标。最终工具结果先校验，retryable/无效结果最多重规划一次，计划超出工具预算时在副作用前停止；重规划提示只含错误码和预算。FakeLLM 成功/重规划/澄清/预算耗尽轨迹通过；专项13 tests、Agent+执行器24 tests、两环境各136 tests、compileall 均通过。本轮下载0、真实请求0。报告：`../../reports/agent_loop/t16_20261007.md`；机读摘要：`../../reports/agent_loop/t16_20261007.json`。下一任务 T17。T04 保持 BLOCKED，F/A/O 效果 NOT EVALUATED。
