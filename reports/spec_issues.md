# 执行中发现的规范与上游假设限制

## 2026-10-05：T03 审计完成不等于旧语义已获证明

用户已要求暂不追论文，按目录核验→旧映射搜索→独立 ID/模型路径推进。T03 的验收是记录资源状态、缺失依据、许可和 A/B 决定；规范 §5.5 明确提供重建路径。继续无限等待作者 canonical/旧映射再启动 T06，属于此前调度过严。现在以独立质量路线闭合 T03 审计，不把 NOT VERIFIED 的旧 checkpoint/matrix 语义改成通过；T04 的原版完整验收仍阻塞。决定和可比性影响见 ADR-015，原规范无需更改。

## 2026-10-05：原反思递归与年份误选可最小修复

原反思递归漏传评测提供的 chat_history，原同名电影映射的日期差未取绝对值。两处原行为已先复现断言失败，独立 opt-in AST 补丁及真实上游组件离线对照通过。不能把源码可修复误写成原 demo 资源缺失：固定/seed 示例本来已存在；缺的是作者 canonical 条件核验、动态默认模型缓存及真实功能恢复证据。补丁和严格目录契约的语义影响见 ADR-014 与 `reports/reproduction_repairs_20261005.md`；不需要修改执行规范。

## 2026-10-05：A1 派生评测输入的 target 可达性与表示

- 原要求：§5.3/T04 保留原输入语义、原模糊文本指标和真实结果；缺 canonical 作者条件只能记 A1。
- 实际发现：全部 45 条原 notebook 派生输入已真实执行，三个原分支正式 hit 均为 0/45。原筛选要求对话至少有两个可映射正反馈，但 process_conv 独立选择最后首次出现的正反馈；21 个最终 target 无法映射当前目录，4 个映射对应不同年份。44 个 target 带年份后缀，原 Map 输出只有标题。
- 不能采用的解释：该 45 条不能直接当作全部 target 可达、作者 canonical 等价或 ID ground truth 已对齐的 benchmark。HTTP200/脚本退出0不能证明推荐质量或每条推荐成功。
- 最小处理：原始输入、全部样本、原指标与正式零结果继续保留；增加独立 coverage/失败诊断摘要。任何去掉年份/修正 target/按覆盖分层的后续实验须单独命名、记录规模与所有排除原因；不能覆盖旧数据或将事后筛选当成冻结 test。
- 可比性：不影响本轮原执行链证据。改变评分目标表示或样本集会改变质量评测条件，不能与本轮正式原指标或论文数字直接比较。后续算法/Agent 比较仍按 T06 起的独立冻结协议执行。
- 证据：`reproduction/native_runs/20261005_session/failure_diagnostics_summary.json`、`reports/a1_multiturn_eval_20261005.md`。

原规范保持不变；以上是事实记录与后续验收建议，没有调整总体项目路线。

## 2026-10-10：T19 执行实现修正（无需修改规范）

§10.6 明确要求“去掉内容只留协同”；先前实现的 no_collaborative 不是该条件。已补 no_content，额外条件仍明确保留。旧 8,649 请求上界误计反馈到达和结构化直接路由，已按实际允许的响应/规划上限修正为 5,202。该修正发生在真实 T19 test 请求之前，测试集、训练模型、开发集路由阈值不变。详见 DECISIONS.md ADR-033；不因此标记 T19 完成。

## Prior historical records (superseded by current status where noted)

# 执行规范勘误建议

2026-10-05：规范未改。实际原 ReDial 预处理对当前原目录的 int64 年份做 Timestamp 相减，42 个 title/year 映射失败；回归先失败后通过，采用局部输入 dtype 适配，详见 ADR-009。上游还以带符号日期差选重名电影，并以 set 顺序选择同轮 target；本轮保留这些行为，固定 hash seed=42，输出45条 A1 输入而非保证50条。建议后续规范明确“固定预处理实现和 seed、记录派生输入与作者 canonical 子集差异”；目前不改原规范，也不升级成 A2。T05 任务卡依赖 T00/T01，之前等待 T04 是执行调度过严，不是规范缺陷。

2026-10-01最新核验：隔离Python3.9和完整原资源现已取得，原工具、真实权重和原app离线回调通过；A1功能路线继续。原eval.py第一次help失败是本项目wrapper未设置规范已有PYTHONPATH，补齐后通过，不是上游需修改。资源ID语义和真实评测仍NOT VERIFIED。下述旧的B决定和“无3.9”描述是历史快照，ADR-005/ADR-006及最新STATUS取代它们；规范无需为这些执行错误修改。

复核更正：ADR-005 撤回依据维度不同冻结 B 的决定。规范不要求目录覆盖所有 embedding，较大模型可接受目录子集；上轮严格维度断言只证明行数不同。固定 notebook 保存输出与目录出现具体标题映射冲突，训练映射和原权重执行继续核查，T03 IN_PROGRESS。无需修改规范，修正的是本项目对证据的解释。

T03 新增事实（2026-10-01）：原包共享 ID 假设未获验证。完整电影目录 ID=1..9888，矩阵头部 (9889,9889)，checkpoint metadata 与物品 embedding 行数=36255；清单无 ID 映射文件。不是 Google 链接失效，也不能断言所有模型分数错误。要求提供权威共同映射、训练来源和资源许可；取得之前 A1 BLOCKED，采用规范 B 方法级重建，详见 ADR-004。以下为首轮历史建议，规范本体不修改。

规范文件未修改；SHA256仍为`d88f02c84855234673514519486e74e73c61ffd4e2ec75594020fdf9b8ebb677`。

1. §4结构示例写`LICENSE`，固定源码实际为`LICENSE.txt`。保留真实文件名，版本锁按实际文件取hash；不影响实验。
2. [S23]所指`eval/eval_single_turn.sh`存在，路径无需修正；源码shell含占位目录以及`--demo_dir`与Python的`--demo_dir_or_file`差异。建议把该具体差异补到规范§5.3，强化已有“实际参数以Python为准”的要求；Python使用parse_known_args，后续wrapper须报未知参数。
3. §16提到执行包README_START_HERE.md、AGENTS.md、TASKS.yaml及example config；本工作区起初只有完整规范，clone也无AGENTS.md。按附录生成TASKS.yaml用于调度，不伪称收到完整执行包；不创建推测性的AGENTS覆盖规则。
4. README资源链接在当前网络可读取HTML；Google下载入口返回474M文件确认页。Issue #110不是当前资源失效的充分证据，不自动选择B。需要完整包、授权、ID和模型兼容核验后完成T03。
5. 当前只有Python3.13.7可见，3.9/3.11未发现；后续环境阶段需要合法取得隔离解释器或记录替代方案，不能把规范建议版本当已建立环境。

所有运行/实验指标保持NOT EVALUATED。以上仅修正路径与事实记录，不修改任务依赖或研究路线。

## T20接口和发布重建核查（2026-10-10）

规范§12.1为目标接口；当前实际CLI有doctor/fixture/train/recommend/benchmark，data/cohort/episodes通过模块，模型评测在train内，系统报告通过generate_system_reports。data prepare/validate、eval-rec、chat、eval-agent、report统一CLI名尚未实现。建议将等价实际入口列为当前兼容范围，或另立接口补齐任务；不修改本规范、不改变实验协议。实际命令与限制已在REPRODUCE逐项公开，未将不存在的目标命令写为验收通过。

新checkout已有公开cohort_summary但无私密labels，旧freeze会阻断重建；T20按公开SHA精确恢复后16个数据文件完全一致。原始规范SHA保持不变。历史seed报告在全新实验副本归档保留后训练，原证据不覆盖；U1另用源码/settings准备，不下载A1资源。
