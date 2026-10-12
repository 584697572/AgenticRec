# T19 System Executor 阶段证据

状态：`CORE IMPLEMENTED / FROZEN FIXTURE PASS / LIVE NOT EVALUATED`

本批将 public episode、逐轮反馈、统一 LLM 账本和可恢复 journal 接成一个明确的 system executor 核心。它不产生系统效果结论，也没有进行远程调用。

## 执行契约

- F：结构化输入直接 fixed flow；文本输入先做一次独立计费解析，后续显式反馈由状态更新进入 fixed flow。
- A：同一 public 输入始终进入 AgentLoop。
- O 与 router 消融：结构化简单请求走 fixed；文本在首次 planner 调用内解释并规划。
- `no_replanning` 要求 AgentLoop 的 `max_replans=0`；`no_explicit_preference_state` 可见当前反馈，但不把 patch 持久化为 episode preference state。
- U1：必须提供共享同一账本的 upstream turn adapter；真实 adapter 尚未完成。
- parser/planner/upstream 的 live requests 和 token 从同一 `BudgetLedger` 增量生成 `EpisodeAttempt`；fixed-only route 为0。

## 多轮与隔离

`FeedbackSchedule` 从 evaluator-only episode 中只保存 `turn/kind/patch`。对象和 journal 不保存接受集合、期望状态、规范硬约束、完整 evaluator events 或 fault。反馈回调只在首轮系统结果之后执行。

多轮 run 必须在 `RunSpec` 中绑定 schedule SHA256；该值进入 run identity、header 与报告。缺 SHA 时在 journal 创建前停止，更换 schedule 后不能恢复原 journal。轮数按实际进度记录：首轮1、收到或校验反馈2、最终响应完成3；缺失反馈为 `MISSING_FEEDBACK`，非法反馈为 `INVALID_FEEDBACK`。

新增字段会改变 RunSpec identity。旧的 runner-only smoke journal 保留为历史快照，新版脚本写入独立的 `runner_smoke_o_seed7_feedback_bound.jsonl`，SHA256 为 `eac792dfa7fd9d4d6f9d6003919e149c8ed6100d1bbba82e5ba30428b6de1f7c`；不会把旧 journal 当成新版可恢复状态。

## 冻结数据机械联调

命令：

```powershell
& .\AgenticRec\.venv-dev\Scripts\python.exe .\AgenticRec\scripts\t19_system_executor_smoke.py --journal .\artifacts\runs\t19\system_executor_fixture_f_20261007.jsonl
```

结果：

- frozen public test episodes：150；
- FakeLLM text parser calls：75；
- fixture fixed pipeline calls：190；
- 完成三轮 episode：40；
- 完整 journal 恢复后的 executor calls：0；
- remote API requests：0；
- evaluator-only 字段进入 journal：0；
- feedback SHA256：`4293ade99a073c08dceff4cdd7dd37d1ca4a64544bc8cf576dbac7c4d083fa55`；
- journal SHA256：`98cf2378a60001a8ec7d5f76533acc0c9e50eb434b3871bea69e0020fde590fb`；
- metrics：`NOT EVALUATED`。

另有模拟 live transport 集成测试，确认一次 F text 解析在 runner 中记录 request=1、input=20、output=5 token；它是本地 transport fixture，不是供应商调用。

## 验证

- executor/runner 联合专项：两个 modern 环境各 `22 passed`；
- 完整测试：两个环境各 `212 passed, 2 warnings`；
- `compileall`：两个环境均 exit 0；
- 下载：0；
- 真实请求：0。

两个 warning 是既有 PyTorch optional NumPy/TypedStorage 提示。

## 未完成

- A/O 使用真实 fixed recommendation tool 的工厂与 prompt 契约；
- U1 upstream-rebuilt 的真实 planner/ToolBox turn adapter；
- evaluator-only `ranking_timeout` 的隔离故障注入；
- DeepSeek 输出质量、正式 U1/F/A/O、消融和系统指标。

下一批先实现并以 fixture 验证上述三个 adapter，再重新核对整批请求上界。现有27次额度仍不得用于零散正式结果。
