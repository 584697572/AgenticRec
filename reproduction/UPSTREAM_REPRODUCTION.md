2026-10-05 当前：[目录核查](../reports/catalog_identity_20261005.md)确认9883个source metadata身份、保留5个歧义；来源别名让45条旧诊断输入28个可解析，未重评或覆盖旧指标。旧模型/矩阵映射仍NOT VERIFIED；A1功能证据保留，质量路径按ADR-015独立重建。T03审计已闭合，T04原完整验收仍BLOCKED，下面是此前批次快照。

此前修复：[修复报告](../reports/reproduction_repairs_20261005.md)验证可选年份/反思历史补丁及原组件fixture流程。最新完整回归64项通过，未接入或重跑旧真实入口。

# 原版复现阶段报告（2026-10-04更新）

2026-10-05 最新：原 App 真实单轮及两轮年份修改已验证；原 recbot、random、加权 popularity 各完整执行45条A1派生输入，正式模糊hit均0/45。实际93次HTTP200、201944 token，高峰估算0.507208CNY；账单未查询。21个target不可映射、4个映射年份不同、索引37截断且无Map，均保留。当前任务已关闭，不能删除记录重跑；T04其余条件仍BLOCKED，A2未验证。详见[多轮与评测报告](../reports/a1_multiturn_eval_20261005.md)。以下未运行/等待Key等文字为历史快照。

## 此前历史记录

2026-10-05 最新：用户选择根目录 `.env` 持久配置，执行端已自动读取并完成原 App 真实单轮 SUCCESS/VERIFIED（2次HTTP200、3880 token、目录约束全部通过）。当前单轮2/2额度已用；多轮/原评测及资源语义限制仍未验收，T04整体BLOCKED。见 [真实单轮报告](../reports/live_original_app_20261005.md)。下文尚未运行/等待Key的描述均为此前历史阶段。

2026-10-05 更新：已准备并离线验证受限原 App 单轮执行器；用户已允许请求，具体单轮额度为 2 次/512 输出 token/1 CNY。执行端没有用户终端的 Key，真实 App 仍 NOT VERIFIED，原评测仍 NOT EVALUATED。ReDial 原始 test 1,342 条已按固定 UniCRS 版本取得，原 notebook 整数年份兼容回归先失败后通过，产出 45 条可重放 A1 输入；不声称与作者 canonical 子集一致。T05 独立现代环境/基础测试已验收，其依赖为 T00/T01。完整结果见 [2026-10-05 报告](../reports/progress_20261005.md)，下文原始阶段记录继续保留。

原版**离线工具链和受控应用运行已通过**：固定InteRecAgent、原电影目录/矩阵、原SASRec权重、原gte-base，没有方法级重建或新增模型。原app.py真实启动HTTP200，两轮回调使用mock LLM响应。2026-10-04原OpenAICall真实DeepSeek连接通过，HTTP200、OK、usage=12+1=13，唯一授权尝试已消耗。**原app真实单轮/多轮NOT VERIFIED；原评测NOT EVALUATED；完整A1尚未验收，A2 not_run。** 连接结果详见[本轮报告](../reports/live_connection_20261004.md)与[native_runs/20261004](native_runs/20261004/)；其余2026-10-01历史运行证据保持原样。

## Completed

- T02：隔离Python3.9.24/Torch1.13.1+cpu环境；pip check、完整native导入、原SDK mock通过。第二个环境从缓存离线重建，174个包版本逐项一致。
- T03已完成的检查：5个完整原movie成员CRC/SHA、schema/ID范围、完整矩阵有限值、原模型兼容加载。T03任务仍IN_PROGRESS，因为语义/来源尚未全部核验。
- 原工具前置：Query、BufferStore、SQLSearch、SimilarItem、三种排序schema、Map通过；原SASRec的36个参数key和值完全与checkpoint相同，并对9888目录项实际predict，shape(1,9888)，全部有限。
- 原app启动与两轮原回调受控检查通过；保留原记忆/工具执行链。LLM边界使用scripted fixture或真实SDK+MockTransport，真实API请求0。
- app.py和one_turn_eval.py的help通过；8项离线回归通过。原始失败、输出、配置及hash保留。
- 独立兼容patch通过原固定upstream的git apply --check，未实际修改原仓库。
- 原始native输出按字节保留，包括Windows CRLF及原logger空白；源码/文档执行diff --check时排除该原始证据目录，另外核验其暂存字节与实际输出一致。

## Files Created / Modified

- 新增环境输入/lock、environment_legacy.json、legacy_compat.patch、UPSTREAM_DIFF.md、本报告和reproduction/README.md。
- 新增资源下载/续取、隔离运行、native import/SDK/工具/app/权重验证、重建比对和证据整理脚本；新增test_matrix_resume.py。
- 原始成功输出与配置复制到`native_runs/20261001/`；详细安装/网络失败日志保留于本地忽略目录`logs/a1_20261001/`。
- 更新STATUS.md、AgenticRec/docs/STATUS.md、TASKS.yaml、DECISIONS.md、resource_manifest/route_decision、spec_issues及验证脚本；旧资源manifest保留为before_native_a1历史快照。
- `.gitignore`排除解释器、模型和缓存。固定RecAI和最高执行规范均未修改。

## Commands / Tests

以下在D:\AgenticRec执行，完整命令、时间和真实退出码保存在native_runs；没有用PowerShell最后一条命令掩盖子进程失败。

```powershell
python reproduction/scripts/validate_complete_resources.py
python reproduction/scripts/run_isolated.py -- reproduction/.venv-legacy/Scripts/python.exe -m pip check
python reproduction/scripts/run_isolated.py -- reproduction/.venv-legacy/Scripts/python.exe reproduction/scripts/native_legacy_probe.py
python reproduction/scripts/run_isolated.py -- reproduction/.venv-legacy/Scripts/python.exe reproduction/scripts/native_sdk_mock.py
python reproduction/scripts/run_isolated.py -- reproduction/.venv-legacy/Scripts/python.exe reproduction/scripts/native_checkpoint_contract.py
python reproduction/scripts/run_isolated.py -- reproduction/.venv-legacy/Scripts/python.exe reproduction/scripts/native_tool_smoke.py --agent-replay --app-startup
python reproduction/scripts/verify_legacy_rebuild.py
python -m unittest discover -s reproduction/tests -v
```

重建时全部版本固定，uv仅离线读取两个已使用官方索引的cache；初始多索引选择失败及错误URL cache定位尝试都保留。
原eval.py必须按规范设置PYTHONPATH；第一次漏设退出1，补齐后help退出0。没有修改eval源码。

## Verification Results

| 检查 | 结果 | 范围 |
|---|---|---|
| 原5个movie成员 | PASS | 完整CRC/SHA，来源为固定README公开ZIP |
| 环境重建/依赖 | PASS | 174包版本一致，pip check，实际模块导入 |
| 原模型参数/推理 | PASS | 36个key和值一致；9888项有限分数 |
| 原完整工具链 | PASS | 真实目录、gte-base、矩阵、checkpoint |
| 原app启动/回调 | PASS（mock LLM） | loopback HTTP200；2轮；服务验证后关闭 |
| 原SDK | PASS（mock HTTP） | 真实SDK序列化及响应解析；不是连接证明 |
| 离线回归 | PASS | 8项 |
| checkpoint标题语义/矩阵ID顺序 | NOT VERIFIED | 没有权威mapping；运行成功不能替代语义核验 |
| 原wrapper真实LLM单次连接 | PASS / VERIFIED（2026-10-04） | DeepSeek，HTTP200/OK，13 token；本地原始记录已核对 |
| 原app真实单轮/多轮 | not_run / NOT VERIFIED | 本次连接授权1/1已消耗，后续调用额度另需确认 |
| 原数据/原指标评测 | not_run / NOT EVALUATED | 原eval数据未获得 |
| 论文设置A2 | not_run | 没有原模型/数据/完整实验条件 |

原安装失败：greenlet3.2.5和chroma-hnswlib0.7.2缺MSVC；native模型加载失败：Hub旧接口/Accelerate新接口不匹配。已按失败证据固定兼容包。两个后续断言失败来自测试契约（typo匹配假设、零维数组类型），没有伪称上游bug修复。

## Findings

- 原矩阵9889×9889 float64；目录9888项、ID1..9888；原模型embedding36255×32允许对子集查表。36个原权重完全匹配，排除了静默丢参数后使用随机初始化的情况。
- 原模型保持training_mode=True，原Buffer少末尾ID，重复tool计划仍会覆盖。它们作为baseline已知问题保留，按后续任务先回归再修，不混入兼容patch。
- Toy Stroy真实模糊输出Toys；精确Toy Story查到ID1062。原相似召回本次493项，原偏好排序和Map返回真实标题；这些是smoke证据，不是质量指标。
- 原app两轮受控SQL/热门排序均返回Pulp Fiction、Forrest Gump、Toy Story；计划和摘要响应是HTTP fixture，不能当LLM会自行正确规划的证明。
- 原SDK接口核验使用[OpenAI Docs](https://developers.openai.com/api/reference/cli/resources/chat/subresources/completions)；未迁移API或全局降SDK。
- 只读取movie成员，未下载全ZIP、其他domain或重复模型格式。gte-base仅所需9文件；矩阵TLS中断后按分块cache续取，失败传输的部分字节未完整计量，不虚报总网络量。依赖下载总量也未完整计量。
- checkpoint和模型加载在不继承真实凭证的隔离子进程内，先核验来源/hash和受限weights_only读取；模型资源不入Git，不再分发。

## Blockers

1. 原wrapper单次真实连接已通过（deepseek-flash，128输出上限，1 CNY预算，1/1尝试已消耗）；原app单轮/多轮和其新增调用额度尚未完成。通用模板仍默认禁用，不重置已消耗记录。
2. 原评测JSONL不在固定仓库/原资源包内；preprocess_redial.ipynb依赖外部UniCRS文件和原有选择过程。不能自造两条样例冒充原评测。
3. 权威训练mapping、矩阵ID语义、预制包独立授权仍NOT VERIFIED。notebook保存映射冲突是风险线索，不能直接证明checkpoint标签；未开展公平推荐质量评测。

## Deviations from Spec

按照规范§5.4处理实际兼容错误：CPU构建、固定传递依赖、zero-demo排除无法构建的可选Chroma；均记录ADR-006与独立diff。dynamic demo暂未验证。目录采用ADR-001的工作区并列布局。
受控原app启动使用本地LLM fixture并复用已经由原构造器初始化的同一Gallery，减少重复CPU编码；仅是测试适配器，明确区分真实LLM原版运行。没有更改总体路线、训练新模型或修推荐/Agent行为。

## Current Project State

T00/T01/T02 DONE，T03 IN_PROGRESS，T04 BLOCKED。选择A1原资源兼容功能路线；B未启用。原版离线前置通过，但完整原版复现和正式实验尚未验收。所有正式指标null / NOT EVALUATED。

## Next Tasks

完成T03映射/来源核查。单次连接已完成，按§5.3下一步原app单轮→多轮→校验原eval数据与原评测，后续请求另需明确额度。单轮方案见[可审阅计划](../reports/live_app_single_turn_plan.md)。保留每次真实输出与失败，不提前进入模型训练、Agent增强、新UI或A2指标宣称。
