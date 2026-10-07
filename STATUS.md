# 工作区状态入口

最后更新：2026-10-07（Asia/Shanghai）。详细状态见 [AgenticRec/docs/STATUS.md](AgenticRec/docs/STATUS.md)，最新证据见 [T17 故障注入与会话隔离](reports/reliability/t17_20261007.md)。

T06 DONE：官方稳定 MovieLens1M ZIP 的 MD5/CRC/SHA 验证通过；1,000,209 条评分按全局时间分成 800,164/100,024/100,021。训练期独立建立 5,351 个 warm 用户、3,469 个候选电影 ID、462,887 条正反馈图边；验证/测试冷启动与非正反馈保留计数。

T07 DONE：[固定指标协议](reports/t07_metrics_protocol_20261006.md)手算 Recall/Hit/MRR/NDCG 与稳定全候选排序通过，valid/test 评测用户与相关集已在本地冻结。valid/test 分别有481/954名 warm 用户拥有可评正反馈；空输出、失败及非法输出留在分母计零。

T08 DONE：[Random/Popularity/BPR-MF 实测](reports/rec_baselines/t08_20261006.md)完成 fixture、seed 42 全量训练、早停、checkpoint 重载和同协议 valid/test 指标。test NDCG@10 分别 0.014081/0.195795/0.192089；BPR-MF 本轮没有超过热门。T08 验收时两个独立环境各36项测试通过，指标从 checkpoint 重算一致；最终3 seeds 尚待后续配置确定。

T09 DONE：[LightGCN 训练与校验](reports/rec_baselines/t09_20261006.md)的稀疏/稠密小图传播误差≤1e-6、训练图与 holdout 零交叉、1/2/3 层 valid 选择和 seed 42 test 实测均通过。选中 3 层第7 epoch，test NDCG@10=0.194047；高于本轮 BPR-MF、低于 Popularity。模型与 ID/训练边 SHA、checkpoint 重算通过。
T10 DONE：[U1 上游方法轨迹](reports/upstream_rebuilt/t10_20261006.md)使用冻结 LightGCN 和 MovieLens1M 候选，已知用户与匿名种子两路均经原 ToolBox/Buffer/Map 执行成功；两套独立环境各 46 项全量测试通过。计划为离线脚本，Agent LLM 规划 NOT EVALUATED；下一任务 T11 多路候选融合与硬约束。
T11 DONE：[多路召回与硬约束](reports/candidate_fusion/t11_20261006.md)完成内容/协同/热门候选、ID 去重、RRF、排序前后硬过滤；真实训练图的已知/匿名两路各 3 源、最终 10 个结果均无硬约束违规。两套独立环境各 55 项全量测试通过，跨进程哈希种子与真实轨迹 SHA 稳定。推荐质量 NOT EVALUATED；下一任务 T12 固定推荐流程。
T12 DONE：[固定推荐流程](reports/fixed_pipeline/t12_20261006.md)把结构化请求、召回、RRF、已知/匿名评分、最终硬校验和证据输出接成 CLI；成功、匿名、排除、无解四场景及无 Key CLI 均通过。两套独立环境各 61 项全量测试通过，连续真实轨迹 SHA 稳定；质量仍 NOT EVALUATED，下一任务 T13。
T13 DONE：[统一 LLM/runtime](reports/llm_runtime/t13_20261006.md)完成严格 JSON schema、默认关闭的 live 门禁、离线 FakeLLM、逐尝试预算、单次 timeout、整轮 deadline 与统一重试分类。专项 22 项、两套环境各 88 项完整回归通过；真实请求 0，供应商 live transport 与输出质量 NOT EVALUATED。下一任务 T14。
T14 DONE：[有序计划执行器](reports/plan_executor/t14_20261006.md)先以固定上游实测重复工具覆盖失败，再用严格 `PlanStep[]` 保留重复步骤、顺序、参数、step_id 与 trace；未知工具、重复 step_id 和坏参数在执行前拒绝。专项11项、两环境各99项完整回归通过；上游未改，真实请求0。下一任务 T15。
T15 DONE：[多轮偏好状态](reports/preference_state/t15_20261006.md)实现不可变 `PreferenceState`、事件化 `PreferencePatch`、逐字段 provenance、显式反馈优先、局部否定、冲突原子澄清、幂等事件与缓存版本。专项11项、两环境各110项完整回归通过；session 清理保留授权训练历史，上游未改，真实请求0。按依赖下一任务 T18。
T18 DONE：[交互评测冻结](reports/interactive_eval/t18_20261007.md)在 T16 调参前冻结 development/test 各150条、各40条多轮 episode，公开/隐藏字段分别序列化，用户组和模板组跨 split 隔离；完整分母评分器与全空输出防刷验证通过。专项13项、两环境各123项完整回归通过；真实请求0。下一任务 T16。
T16 DONE：[规则路由与有限闭环](reports/agent_loop/t16_20261007.md)实现 DIRECT/PERSONALIZED/AGENT/CLARIFY 与 F/A/O 三种模式；仅 development 公开输入选择0.75阈值，test 文件读取0。FakeLLM 成功、一次重规划、澄清、预算耗尽轨迹可重放；专项13项、两环境各136项完整回归通过，真实请求0。系统效果仍 NOT EVALUATED；下一任务 T17。
T17 DONE：[故障注入与会话隔离](reports/reliability/t17_20261007.md)覆盖坏 JSON、fixture timeout/429、无候选、prompt injection、恶意结果和同步工具超时；同名用户、异步 session、32个并发反馈及版本化缓存均无串扰。专项14项、两环境各150项完整回归通过；真实请求0。下一任务 T19。

此前修复批次见[修复报告](reports/reproduction_repairs_20261005.md)：两项可选 upstream_compat 补丁及当时53项测试通过；原组件 fixed demo/反思完成 fixture 联调。该批未新增请求/下载，原版输入、预测与三个0/45保留。本批来源核查和路线更新见下文。

项目仓库：[584697572/AgenticRec](https://github.com/584697572/AgenticRec)，public；根项目 `origin` 指向该仓库，`master` 跟踪 `origin/master`。创建与首次推送已核验，见 [发布报告](reports/public_repository_20261005.md)。上游 `RecAI/` 保留独立远程与固定版本。

- DONE：T00、T01、T02、T03、T05—T18。T03 完成资源审计/缺失记录/路线选择，未将旧模型语义标为通过；现代开发基础保持原验收。
- T03 审计结论：官方 MovieLens10M 元数据中 9883 个目录身份唯一对应、5 个歧义；source-backed 别名解析使旧 45 条诊断输入从 22 到 28 个可解析，另 6 个年份冲突、11 个该来源未收录标题。旧 checkpoint/矩阵语义与预制包独立许可仍 NOT VERIFIED，原数据和指标不变。见[目录核查报告](reports/catalog_identity_20261005.md)。
- T04 BLOCKED（其余验收前置）：真实单轮及两轮条件修改 VERIFIED；45 条 A1 派生输入的原 recbot/random/加权 popularity 评测链已完成，正式原模糊 hit 均为 0/45。新会话实际93次HTTP200、201944 token，高峰费率估算0.507208 CNY；账单未查询。canonical 子集、权威 ID/许可及原 demo/reflection 条件仍未验收。
- T06—T18 DONE；T19 IN PROGRESS；T20—T24 TODO。独立数据、指标协议、推荐模型、U1 上游方法、候选/约束、固定流程、LLM runtime、有序执行、多轮状态、规则路由、安全隔离和交互评测已固定；T19 推荐模型三 seed 已完成，Agent 效果仍 NOT EVALUATED。

当前路线：保留 A1 原功能执行证据，质量实验按 ADR-015 采用规范 B/upstream_rebuilt 的独立数据/模型路径。用户要求暂不追论文；作者 canonical 条件不再阻塞独立新实验。原规范和固定 upstream 未改，不把新的 raw_movie_id 套进旧 embedding。旧报告/证据保持历史快照，不重置额度。

此前真实运行批次的40+3项测试、原指标复算和22个旧证据检查保留。原映射21个target未知、4个年份不同、索引37截断且无Map，全部45个分母不变；年份剥离后的6个模糊诊断不替代正式结果。当前source-backed别名新增6个解析与该旧模糊命中不是同一指标。

Key仍按环境变量优先、根目录`.env`次之，不入Git/哈希清单。原额度仍已用93/120，本轮新 LLM 请求0。此前 MovieLens10M 元数据只用于旧目录核查；官方 MovieLens1M 是独立新实验的主数据集。T19 dry-run 已核验整批上界8,649次、剩余27次、缺口8,622次，live fail closed；三 seed 模型表已由原始预测重建。后续模型与 Agent 必须沿用冻结划分，不能事后换指标或 seed 选择有利结果。

## T10 阶段更新（2026-10-06）

- 当前阶段：G2，T10 已完成验收；T11 尚未开始。
- 已完成：冻结 BPR-MF/LightGCN 的身份与 checkpoint 校验；已知用户与匿名种子评分分流；原 ToolBox/CandidateBuffer/MapTool 进程隔离；真实 MovieLens1M 候选的已知用户、匿名种子两条 U1 离线轨迹；逐模块 `UPSTREAM_DIFF.md`。
- 最后验证：两套独立 Python 3.11 环境分别 `46 passed`（`--basetemp` 指向项目内）；U1 两条真实轨迹 `toolbox_success=true`、各映射 3 个、缓冲区重置、远程请求 0。公开 `reports/upstream_rebuilt/t10_20261006.json` 与本地忽略的私有轨迹 SHA 对应。初次全量测试的 2 个 pytest 临时目录错误由沙箱无权访问 `C:\Windows\Temp` 引起，项目内临时目录完整重跑通过。
- Blocker：T04 原 checkpoint/相似度矩阵 ID 语义及独立再分发许可仍 NOT VERIFIED，原论文条件仍 BLOCKED；不阻塞用户已选 U1 路线。U1 LLM 规划、冷启动质量和多 seed 效果仍 NOT EVALUATED。
- 下一任务：按规范 T11 实施多路候选融合与硬约束，沿用冻结 T06/T07 ID 与评测契约；不要把本轮离线计划当成 live Agent 结果。

## T11 阶段更新（2026-10-06）

- 当前阶段：G2；T11 已完成验收，T12 尚未开始。
- 已完成：冻结目录的内容 TF-IDF、授权用户模型 CF / 匿名 train-only item-item CF、训练期热门、RRF 与 ID 去重；显式排除、类型、年份和已看物品在召回前后严格校验；缺失字段、冲突、无解结构化返回。
- 最后验证：专项 9/9；两套现代环境全量各 55/55；真实训练图已知/匿名两路分别 703/709 可行候选、191/197 融合候选、各 10 个最终结果，硬约束违规 0。不同 Python 哈希种子与连续两次真实轨迹 SHA 稳定；公开 `reports/candidate_fusion/t11_20261006.json` 记录摘要和私有轨迹 SHA，完整 ID 轨迹在忽略目录。
- Blocker：T04 原资源 ID 语义和独立许可继续 NOT VERIFIED；T11 推荐质量 NOT EVALUATED，不把源数量或返回数量当作指标提升。
- 下一任务：T12 完整无规划固定推荐流程与 CLI 四场景，并在最终个性化重排后再次执行硬约束 gate；沿用相同候选/预算/字段进行后续公平对比。

## T12 阶段更新（2026-10-06）

- 当前阶段：G2；T12 已完成验收，T13 尚未开始。
- 已完成：严格结构化请求 schema；召回/RRF/用户或冷启动评分/最终硬校验的固定流程；推荐 ID、title/genres/year、来源名次、个性化分数与 RRF 分数证据；独立 CLI。
- 最后验证：专项 6/6；两个现代环境各 61/61；真实成功、匿名、排除、无解四场景通过，独立 CLI 在无 API 凭据环境返回 3 条结果且 LLM/远程请求均为0，连续两次私有轨迹 SHA 一致。
- Blocker：T04 原资源 ID/许可仍 NOT VERIFIED；T12 推荐质量 NOT EVALUATED。自由文本没有被伪装成零成本解析，需后续单独计费适配。
- 下一任务：T13 统一 LLM adapter、严格响应 schema、超时/重试和预算计数；默认真实调用继续关闭。

## T13 阶段更新（2026-10-06）

- 当前阶段：G3；T13 已完成验收，T14 尚未开始。
- 已完成：provider-neutral chat adapter、严格 JSON object schema、离线 FakeLLM 与 live 门禁分离、逐尝试请求/费用上界账本、unknown usage/cost 的 null+reason、单次 timeout、共享整轮 deadline 和唯一外层重试控制器。
- 最后验证：专项 22/22；两个现代环境各 88/88；compileall exit 0。401 只尝试一次，429 有界重试且两次均计数；未授权 live 在 transport 前失败；本轮远程请求0。
- Blocker：T04 原资源 ID/许可保持 BLOCKED；供应商特定 live transport、DeepSeek Agent 输出质量和实际账单在 T13 中 NOT EVALUATED。
- 下一任务：T14 先建立重复工具名的失败回归，再实现保留顺序、重复调用、step_id 和 trace 的 PlanStep 列表执行器。

## T14 阶段更新（2026-10-06）

- 当前阶段：G3；T14 已完成验收，T15 尚未开始。
- 已完成：固定上游重复工具覆盖再次实测；严格 `PlanStep[]` 解析、精确工具白名单、整计划预校验、有序重复执行、结构化停止 trace。`RecAI/` 保持原样。
- 最后验证：原上游保留要求 exit 1，观察仅 `second-query`；新执行器专项11/11，两个现代环境各99/99，compileall exit 0；本轮远程请求0。
- Blocker：T04 原资源 ID/许可保持 BLOCKED；T14 只证明执行正确性，推荐质量、任务成功率和成本收益 NOT EVALUATED。
- 下一任务：T15 实现带 provenance 的 PreferencePatch、显式覆盖、否定反馈、矛盾澄清、缓存版本与幂等更新。

## T15 阶段更新（2026-10-06）

- 当前阶段：G3；T15 已完成验收。T16 依赖尚未完成的 T18，因此下一可执行任务为 T18。
- 已完成：不可变多轮状态、事件化 patch、逐字段证据、显式/推断优先级、物品级局部否定、冲突原子拒绝、事件摘要幂等、profile version 和 session/训练历史隔离。
- 最后验证：专项11/11；两个现代环境各110/110；compileall exit 0；确定性人工轨迹覆盖 APPLIED/IDEMPOTENT/REJECTED/CLARIFY，远程请求0。修复前旧轮次硬字段回归测试稳定失败，修复后通过。
- Blocker：T04 原资源 ID/许可保持 BLOCKED；T15 只验证状态语义，推荐质量、任务成功率、延迟和成本均 NOT EVALUATED。
- 下一任务：按依赖先执行 T18，冻结公开/隐藏字段隔离的交互开发集/test 与完整分母评分器；在 T16 路由阈值调节之前固定 test。

## T18 阶段更新（2026-10-07）

- 当前阶段：G2/G3；T18 已完成验收，T16 的评测前置依赖已满足。
- 已完成：development/test 各150条，其中各75条结构化、75条文本、40条多轮；八类场景、用户/模板组隔离、公开/隐藏分离、manifest 哈希、16条人工核验子集和完整分母评分器。
- 最后验证：专项13/13；两个现代环境各123/123；compileall exit 0；全空 ABSTAIN 在两个 split 的 Fill@K 与 Strict Success 均为0；下载0、真实请求0。
- Blocker：T04 原资源 ID/许可保持 BLOCKED；T18 只冻结合成离线 benchmark，系统任务成功率、延迟、token 与 Agent 提升均 NOT EVALUATED。
- 下一任务：T16 规则路由与有限 Agent 闭环；阈值只可在 development 选择，test 私有目标不得进入 planner、工具历史或 demo 检索。

## T16 阶段更新（2026-10-07）

- 当前阶段：G3；T16 已完成验收，T17 尚未开始。
- 已完成：四路由、F/A/O 模式、development-only 阈值选择、严格工具 schema 提示、最终工具结果校验、最多一次重规划、planner/tool/replan/远程请求与共享预算账本统计；隐藏 evaluator 字段不能进入路由输入。
- 最后验证：专项13/13；Agent+执行器回归24/24；两个现代环境各136/136；compileall exit 0。150条 development 公开输入路由为 Agent 75、Clarify 7、Direct 26、Personalized 42；test 文件读取0、下载0、真实请求0。FakeLLM 四类轨迹的本地证据 SHA 稳定。
- Blocker：T04 原资源 ID/许可保持 BLOCKED；T16 只证明有限控制流和隔离契约，F/A/O 的任务成功率、延迟、token 成本及提升均 NOT EVALUATED。
- 下一任务：T17 故障注入、安全和会话隔离；覆盖坏 JSON、timeout、429、无候选、prompt injection、并发同名用户、异步 session、反馈和缓存串扰。

## T17 阶段更新（2026-10-07）

- 当前阶段：G3；T17 已完成验收，T19 的全部前置任务已满足。
- 已完成：本地坏 JSON、timeout、429、无候选、prompt injection 与恶意推荐结果注入；严格最终硬约束复核；有界同步工具执行器；session ID 唯一身份、同名显示名隔离、跨/同 session 异步反馈原子化、版本化深拷贝缓存。
- 最后验证：专项14/14（两环境）；T14/T16/T17 联合回归38/38；两个现代环境各150/150；compileall exit 0。单 session 32个并发 patch 全部保留；工具自身抛出的 TimeoutError 未误分类为等待超时；下载0、真实请求0。
- 同步 timeout 边界：调用方停止等待不等于运行中的 Python 线程被终止；有界池限制并发并记录 `work_may_continue`，超时工具不自动重试。无限阻塞或不可信工具仍需可终止子进程或协作取消。
- Blocker：T04 原资源 ID/许可保持 BLOCKED；T17 不证明分布式锁、跨进程缓存或在线安全，也不产生系统效果结论。
- 下一任务：T19 先执行 benchmark dry-run 和总请求数核验，再按既有授权额度运行 F/A/O/U1、规定消融、配对 CI、失败分类与三 seed 模型实验。

## T19 阶段更新（2026-10-07）

- 当前阶段：G4；T19 IN PROGRESS。推荐模型三 seed 子任务已完成，真实交互系统与消融仍被授权上限阻塞。
- 已完成：冻结 seed 7/42/2026 的 BPR-MF 与 LightGCN 运行；导出 2,862 行每用户 Top-10 原始预测，并由 evaluator-only test 目标独立重算 Random/Popularity/BPR-MF/LightGCN 两套统计。原始预测重算与全部三 seed 源报告逐指标一致。
- 模型结果：BPR-MF NDCG@10 为 0.203947 +/- 0.010365，LightGCN 为 0.195317 +/- 0.001138。LightGCN-BPR 配对用户 bootstrap 均值 -0.008631，95% CI [-0.012850, -0.004344]；这是负收益证据，不声称 LightGCN 改进。
- dry-run：150 条 frozen test 的 max-turn 合计 230，文本 turn 123；U1/F/A/O、五项消融和三个 Agent seed 的保守整批请求上界为 8,649。历史已使用 93/120，只剩 27，缺口 8,622；money budget 是否足够仍 NOT VERIFIED。dry-run 与本轮训练的远程请求均为 0。
- Blocker：T19 live 状态为 BLOCKED_AUTHORIZATION；U1/F/A/O、消融、交互任务成功率、约束满足率、延迟、token、工具次数及失败分类均 NOT EVALUATED。T04 原资源 ID/许可 blocker 不变。
- 验证：benchmark 专项两个环境各5/5、完整回归各155/155、两个环境 compileall exit 0；仅既有 PyTorch 可选 NumPy 与 TypedStorage warning。证据见 `reports/benchmark/t19_20261007.md`。
- Runner 进展：完成 public-only `RunSpec`、写前 `STARTED`/`COMPLETED`/`INTERRUPTED` 哈希链 journal、未决付费尝试恢复阻塞、run identity 校验、请求/金额整批门禁及 `EpisodeAttempt` 重建。真实150条 public test 离线 smoke 首跑/恢复 SHA 一致且请求0；24个正式 run 已冻结。联合专项两环境各12/12、完整回归各162/162、compileall通过。live transport、文本解析和正式 executor 仍未接入，T19保持 IN PROGRESS。
- 下一任务：先把 live runner 接到同一冻结 public episode 与 attempt writer；在任何请求前取得覆盖预注册批次的明确请求/金额授权，或按规范在看结果前预注册可承担的缩小协议。不得用现有 27 次零散试跑填正式表。
