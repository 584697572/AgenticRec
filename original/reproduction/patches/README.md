# 可选上游正确性修复

来源为固定 `0959ecb05b0794748426e73e6efc1b6b35ec433d`。两份 patch 展示一行行为差异；`upstream_bugfixes.py` 对对应函数 AST 做有计数校验的最小修改并执行。它不写入 `RecAI/` 或已有 `.runtime/InteRecAgent-compat/`，不改变既有原版 CLI 默认行为。

- `redial_absolute_year.patch`：同名电影按绝对日期差找最近年份，修复带符号差导致总偏向更早版本的问题。保留原模糊标题匹配；它仍不能给出权威训练 ID。下游实验可达性必须另外检查。
- `reflection_external_history.patch`：反思递归继续传递 `chat_history`，使一轮评测提供的外部历史不会退化为本地 memory。保留原 Critic、反思限次、prompt 和 memory 行为。

这两项标记为 `upstream_compat` 行为修复，与 `upstream_pristine` 分开；旧 45 条输入、真实预测和正式 0/45 指标不更新、不重算成新指标。未来启用补丁的实验必须另存配置与结果，不能冒充未修改上游或作者 A2。

无需 Key 的验证命令：

```powershell
& .\reproduction\.venv-legacy\Scripts\python.exe -m unittest discover -s reproduction/tests -p test_upstream_bugfixes.py -v
& .\reproduction\.venv-legacy\Scripts\python.exe reproduction/scripts/probe_original_features.py
& .\reproduction\.venv-legacy\Scripts\python.exe reproduction/scripts/check_repair_impact.py
```

后两个脚本保留已有产物、不覆盖同名输出。探针执行真实上游 DemoSelector、Critic、Agent、SDK 的 fixture 流程；不是原 App 的真实 LLM 推荐结果。固定示例取上游 placeholder 文件前 3 个；作者 canonical demo 等价性未核实。动态模式默认 `sentence-transformers/all-mpnet-base-v2` 不在隔离缓存中，本轮未下载，也未替换为另一 embedding 后称为原版。

新实验可显式使用 `corrected_movie_map()`；Agent 可在局部上下文以 `patch.object(CRSAgentPlanFirstOpenAI, "run", corrected_agent_run(agent_module.__dict__))` 启用历史修复。`require_reachable_targets()` 接收候选实验输入和目录，在任一 target 无唯一标题/年份匹配时拒绝整批输入，不会筛掉失败样本。这个检查不注入 target 到 Agent，也不验证 checkpoint 的行语义。当前 45 条仍可用于已经记录的原版兼容性执行，但不能直接声明为全部目标可达的目录内 benchmark。
