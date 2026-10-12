# 原wrapper真实 DeepSeek 连接验证（2026-10-04）

## Completed

用户按明确授权在自己的PowerShell执行单次真实连接，原OpenAICall及现有SDK取得SUCCESS、HTTP200、回复OK。已核对工作区原始结果、额度记录、固定源码SHA和真实usage，保留可追溯证据；不把连接通过当完整原app复现。

## Files Created / Modified

新增受限连接脚本、5项SDK/HTTP离线测试、DeepSeek禁用模板、本报告、原app单轮计划和native_runs/20261004证据；更新LIVE_LLM_GUIDE、README、UPSTREAM_REPRODUCTION、两个STATUS、TASKS、DECISIONS。profile、授权记录、Key环境变量和完整本地日志不入Git；Key未写入任何项目文件。最高规范、固定upstream和2026-10-01原始证据未改。

## Commands / Tests

用户实际执行：`reproduction/.venv-legacy/Scripts/python.exe reproduction/scripts/live_llm_connection.py`。已有验证：同一legacy解释器全量unittest discover，16项通过。额度验证通过mock transport断言已消耗记录拒绝、无新增HTTP、旧结果/ledger字节保持一致；规范SHA、upstream pristine、source/docs diff --check和证据哈希检查。

## Verification Results

| 项目 | 实际结果 |
|---|---|
| 原wrapper真实连接 | SUCCESS / VERIFIED；HTTP200，reply=OK |
| 模型和模式 | deepseek-flash；thinking disabled |
| 真实HTTP请求 | 1；唯一授权尝试1/1已消耗 |
| 实际usage | prompt=12，completion=1，total=13；cache hit=0，miss=12 |
| 单次elapsed | 2.1585443秒；连接脚本实测，不是benchmark latency |
| 实际费用 | null，供应商账单未查询 |
| 保守费用估算 | 0.000032 CNY，按当次公开高峰缓存未命中输入/输出单价计算；不是实际扣费 |
| 原源码 | SHA256=60e7d6c6fbca1aaf334efb5106edaffccbf34cba7dcb5e983d5980ff94251b57，未修改 |
| 离线验证 | 全16项通过，其中5项新增真实SDK+mock HTTP门禁测试 |
| 原app真实单轮/多轮 | NOT VERIFIED |
| 原评测/正式质量指标 | NOT EVALUATED |

原始时间started_at_utc=2026-10-03T16:16:16.139890+00:00，即北京时间2026-10-04 00:16:16.139890，未改写原时间字段。

## Findings

DeepSeek实际接受原OpenAICall的Chat Completions调用；测试层显式关闭thinking、SDK/HTTP重试和重定向，原外层仅1次。真实返回验证了单次连接的非思考选项，不能推导整套Agent会规划成功。价格依据为结果记录中的2026-10-04[DeepSeek人民币价格快照来源](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/)；12×2/1,000,000+1×8/1,000,000=0.000032 CNY。客户端不查询账单或额外model-list以免产生授权范围外请求。

## Blockers

原app真实单轮/多轮和对应新增额度尚未授权执行；原eval数据缺失；权威checkpoint标题mapping、矩阵ID顺序及预制资源独立授权仍NOT VERIFIED。原wrapper连接不再是blocker。

## Deviations from Spec

None（本轮按§3.3/§5.3）。测试层服务/模式/重试配置已公开记录ADR-007；这是A1兼容功能连接，不能当A2论文原模型设置复现。没有新增模型、训练、Agent增强或UI开发。

## Current Project State

原资源离线工具链与mock原app通过；原wrapper真实连接已VERIFIED，授权额度已消耗。T00/T01/T02 DONE、T03 IN_PROGRESS、T04 BLOCKED、T05—T24 TODO；A1完整验收未完成，A2 not_run。此次连接的usage/耗时保留为连接smoke证据，不回填正式benchmark指标。

## Next Tasks

按规范§5.3执行原app单轮，再多轮，最后补齐原eval数据并运行原评测；T03语义/来源核验继续。具体待授权单轮范围见[live_app_single_turn_plan.md](live_app_single_turn_plan.md)：固定查询、最多2次HTTP、每次最多512输出token、新增1 CNY预算，不重试。不删除本次额度记录，也不默认延用本次授权。
