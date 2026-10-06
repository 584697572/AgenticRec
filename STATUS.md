# 工作区状态入口

最后更新：2026-10-06（Asia/Shanghai）。详细状态见 [AgenticRec/docs/STATUS.md](AgenticRec/docs/STATUS.md)，最新证据见 [T15 多轮偏好状态](reports/preference_state/t15_20261006.md)。

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

此前修复批次见[修复报告](reports/reproduction_repairs_20261005.md)：两项可选 upstream_compat 补丁及当时53项测试通过；原组件 fixed demo/反思完成 fixture 联调。该批未新增请求/下载，原版输入、预测与三个0/45保留。本批来源核查和路线更新见下文。

项目仓库：[584697572/AgenticRec](https://github.com/584697572/AgenticRec)，public；根项目 `origin` 指向该仓库，`master` 跟踪 `origin/master`。创建与首次推送已核验，见 [发布报告](reports/public_repository_20261005.md)。上游 `RecAI/` 保留独立远程与固定版本。

- DONE：T00、T01、T02、T03、T05—T14。T03 完成资源审计/缺失记录/路线选择，未将旧模型语义标为通过；现代开发基础保持原验收。
- T03 审计结论：官方 MovieLens10M 元数据中 9883 个目录身份唯一对应、5 个歧义；source-backed 别名解析使旧 45 条诊断输入从 22 到 28 个可解析，另 6 个年份冲突、11 个该来源未收录标题。旧 checkpoint/矩阵语义与预制包独立许可仍 NOT VERIFIED，原数据和指标不变。见[目录核查报告](reports/catalog_identity_20261005.md)。
- T04 BLOCKED（其余验收前置）：真实单轮及两轮条件修改 VERIFIED；45 条 A1 派生输入的原 recbot/random/加权 popularity 评测链已完成，正式原模糊 hit 均为 0/45。新会话实际93次HTTP200、201944 token，高峰费率估算0.507208 CNY；账单未查询。canonical 子集、权威 ID/许可及原 demo/reflection 条件仍未验收。
- T06—T15 DONE；T16—T24 TODO。独立数据、指标协议、推荐模型、U1 上游方法、候选/约束、固定流程、LLM runtime、有序执行与多轮状态已固定；最终多 seed 与 Agent 效果仍 NOT EVALUATED。

当前路线：保留 A1 原功能执行证据，质量实验按 ADR-015 采用规范 B/upstream_rebuilt 的独立数据/模型路径。用户要求暂不追论文；作者 canonical 条件不再阻塞独立新实验。原规范和固定 upstream 未改，不把新的 raw_movie_id 套进旧 embedding。旧报告/证据保持历史快照，不重置额度。

此前真实运行批次的40+3项测试、原指标复算和22个旧证据检查保留。原映射21个target未知、4个年份不同、索引37截断且无Map，全部45个分母不变；年份剥离后的6个模糊诊断不替代正式结果。当前source-backed别名新增6个解析与该旧模糊命中不是同一指标。

Key仍按环境变量优先、根目录`.env`次之，不入Git/哈希清单。原额度仍已用93/120，本轮新 LLM 请求0。此前 MovieLens10M 元数据只用于旧目录核查；官方 MovieLens1M 是独立新实验的主数据集。当前按依赖优先 T18：先冻结交互评测集和评分器，再执行依赖它的 T16；后续模型须用同一冻结划分，不能事后换指标选有利结果。

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
