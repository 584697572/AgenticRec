# 原版复现阻塞检查与隔离修复（2026-10-05）

本轮在 T03/T04 范围检查可修复源码与缺失事实，继续 A1。没有新模型训练、新评测集或新付费请求；既有真实评测正式零结果原样保留。本报告从验证摘要和原始测试日志生成。

## Completed

- 两项原行为错误已用自造 fixture 先产生断言失败，再以独立、显式可选的最小 AST 补丁修复：同名电影年份选择，反思递归外部历史传递。
- 原 DemoSelector(fixed)、Critic、Agent 与真实 SDK MockTransport 对照通过；原版/补丁各运行 zero/fixed 两种设置。每组 planner → critic(No) → planner → critic(Yes)，原版第二次历史丢失而补丁保留。fixture gallery/buffer/tools/回答并非真实 App 推荐。
- 对保存的 45 条输入检查补丁影响与严格目录可达性。新增 guard 拒绝不满足契约的整批新目录内评测，不删除失败样本。

## Files Created / Modified

新增 upstream_bugfixes、check_repair_impact、probe_original_features、render_repairs_report 脚本及 test_upstream_bugfixes。精确差异和 opt-in 方式见 [补丁说明](../reproduction/patches/README.md)。原始命令/退出码/native stdout/stderr 与公开摘要在 reproduction/native_runs/20261005_repairs/；含 target 的逐行记录仅在忽略的 reproduction/logs/repair_check_20261005/。更新状态、决策、规范建议和 README，未改原规范、RecAI/ 或既有 compat 工作副本。

## Commands / Tests

```powershell
& .\reproduction\.venv-legacy\Scripts\python.exe -m unittest discover -s reproduction/tests -v
& .\reproduction\.venv-legacy\Scripts\python.exe reproduction/scripts/probe_original_features.py
& .\reproduction\.venv-legacy\Scripts\python.exe reproduction/scripts/check_repair_impact.py
```

后两条保留已有产物，重复运行退出 2 且 0 请求。另执行 py_compile、重复门禁、旧证据哈希和 Git 检查。最初 fixture 缺工具 name/desc 的错误另存；修正 fixture 后、修复源码前的有效回归是 3 个断言失败，不将测试构造错误当上游 bug 证据。初次报告的 PowerShell 管道中文编码损坏已在发布前改为 UTF-8 源文件生成，不修改实验记录。

## Verification Results

- 全量 53 项测试 PASS，退出 0；新增补丁/目录契约测试共 10 项。
- 4 组原组件 fixture 流程 PASS，16 次 SDK mock、0 真实请求，usage=null、usage_is_fixture=true。固定示例取上游 placeholder 文件前 3 个（文件共 14 个），作者 canonical demo 等价性 NOT VERIFIED。
- 重复门禁与编译 PASS，已有 26 个公开修复产物字节不变；431 个此前私有证据及正式真实评测产物哈希不变。
- 新真实请求 0，下载 0 字节；原授权仍已用 93/120，旧真实 usage/费用估算和台账不变。
- 新推荐质量指标为 null，算法收益 NOT EVALUATED。原三个分支 0/45 不是补丁的结果，不能推断补丁收益。

## Findings

| 对当前 45 条保存输入的映射检查 | 原行为 | 仅年份补丁 |
| --- | ---: | ---: |
| 未映射 target 数 | 21 | 21 |
| 已映射但年份不一致 | 4 | 3 |

仅 1 个映射改变；另外三个年份不一致不能靠绝对差修复，原近似标题分支仍未校验年份。严格契约为 22 个唯一标题/年份匹配、17 个缺标题、6 个缺对应年份。不补造条目、选另一个更容易命中的 target 或生成筛选后的 test。

原 movie_map 的 signed 日期差误选较早版本；原反思 [递归调用](https://github.com/microsoft/RecAI/blob/0959ecb05b0794748426e73e6efc1b6b35ec433d/InteRecAgent/llm4crs/agent_plan_first_openai.py#L463) 未传 chat_history，在一轮评测中丢掉外部历史。已核验 planner 和 Critic 实际 mock prompt，未更换原 prompt 或指标。

[上游示例实现](https://github.com/microsoft/RecAI/blob/0959ecb05b0794748426e73e6efc1b6b35ec433d/InteRecAgent/llm4crs/demo/base.py)及 fixed/seed 文件本来可用，缺的是原实验条件对齐。动态默认 sentence-transformers/all-mpnet-base-v2 不在隔离缓存中，未下载或换 embedding 冒充原设置；作者 paper demo 的具体文件和来源未获证明。

已有资源 archive 的 15 个成员仅有三域的目录表/columns/settings/矩阵/checkpoint，无训练 ID map 或独立许可证。固定 notebook 以过滤后 ratings 的 encounter order 建新 ID，缺原 raw release/过滤顺序不能从 embedding 数量恢复标题。目录 exact match 不等于 checkpoint/矩阵语义证明。

## Blockers

权威 checkpoint 行标题映射、矩阵行语义、资源独立许可、作者 canonical 子集与 demo 条件仍未解决。真实 LLM 下开启 demo/reflection 尚 NOT VERIFIED。当前输入不能声明为全部 target 可达的目录内 benchmark，缺失事实不能由代码编造。

## Deviations from Spec

无路线变更；重大决定 ADR-014。两项补丁为显式 opt-in upstream_compat 行为变化，未来须单独记录条件和结果；已有原版真实入口、输入、预测和正式指标不变。未提前训练、改 Agent 架构、改 prompt、改指标或筛掉样本。

## Current Project State

两处源码错误已经验证修复；完整资源/原实验验收仍未通过。T03 IN_PROGRESS、T04 BLOCKED、T06—T24 TODO。本轮不是完整论文复现或算法改进收益。

## Next Tasks

继续 T03 权威映射/许可与 T04 原配置核验。已查看的 45 条只能作为明确限定的兼容性/失败诊断输入，不能事后筛出 22 条称冻结 test。前置闭合后按 T06 数据 → T07 指标 → T08 BPR-MF → T09 LightGCN 推进。启用补丁的真实运行须沿同一授权剩余额度计数，不能删除标记或另起 120 次额度。
