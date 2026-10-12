# T19 工程收口与30元预算核算（2026-10-10）

T19 **IN PROGRESS**。本轮零下载、零真实 API 请求。推荐模型三seed及已保存负收益不变；真实 U1/F/A/O、消融效果、配对CI、失败分层数量仍 **NOT EVALUATED**。完整live报告重建 **NOT VERIFIED**。

## 执行边界

U1 使用真实冻结 LightGCN 候选和同一推荐流程，计费 planner 生成完整 FixedRequest 与原 tool_name/input 计划；后续是实际原版 ToolBox/CandidateBuffer/MapTool。离线集成用明确的 FakeLLM，返回3条合法候选并实测3次工具调用。保持原同名工具覆盖行为，不调用原 CRSAgent 自由文本总结。U1 子进程启动包含在延迟中；不能据此归因 planner 效率。原 RecAI 仍为0959ecb05b0794748426e73e6efc1b6b35ec433d且工作树干净。

补齐规范协同-only的no_content；保留额外no_collaborative（仅禁协同召回、仍用训练模型最终排序）。三seed、150条冻结test、反馈、目标和开发集阈值均不变。fault仅在执行层做确定性本地ranking失败与同约束fallback，不向远端制造失败，不把fault标志或目标送入prompt。

## 修复与验证

- 修复前回归失败：文本首轮执行后丢失偏好状态、直接fixed工具未计数、非思考参数缺失、响应次数上界错误、失败parser/planner未进入trace、仅请求充足时误判READY。
- 两个独立现代环境完整回归分别 **239 passed**。唯一warning为既有PyTorch可选NumPy和TypedStorage；不影响结果。
- 专项真实U1/消融/fallback与旧bridge联合 **5 passed**，不访问API。初次沙箱内legacy子进程90秒超时；相同原版和新适配集成在沙箱外通过，未修改原源码绕过失败。
- 逐请求intent/terminal审计与EpisodeAttempt请求/token独立复核；拒绝fixture、缺请求、重复请求、未决intent和token不一致；失败usage保持null，已知部分token不冒充全额成本。
- 固定流程F仅单次文本parser，planner计数0；失败解析仍计parser invocation。U1失败规划保留planner invocation，fallback计入工具总数。
- `compileall`与`git diff --check`通过；规范SHA256为d88f02c84855234673514519486e74e73c61ffd4e2ec75594020fdf9b8ebb677，未修改。

## 预算核算

用户最新选择：**保持总预算30元，先完成工程并重新核算**。`reproduction/t19_budget_20261010.json`为公开的工程planning profile，allow_paid_api=false。6,000只是提议的规划请求上限，并不把该选择解释成完整批次授权。历史A1的93/120记录保留，未清零。

27个独立run（always_agent复用A），最大5,202请求：U1 570、F 225、A 1,140、O及四个路由/推荐消融各594、no_replanning 297。反馈到达不算模型响应；结构化O不规划；多轮最多两次系统响应。旧8,649上界是本项目执行错误，已在真实test前修正。

根据[DeepSeek官方价格](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/)，2026-10-10核验峰值输入/输出分别2/8 CNY每百万token。输入按8,320 token、输出按1,024 token保守规划，单次预留0.024832元，全批129.176064元，较30元差99.176064元。只算全部请求打满输出上限也为42.614784元；仅压缩输入上界无法覆盖峰值最坏批次。**这不是实际或预测账单；实际调用数/输出长度可能更少，目前未验证能在30元内跑完。**

## 命令与证据

工作目录AgenticRec：

```powershell
& .\.venv-dev\Scripts\python.exe -m pytest tests -q -p no:cacheprovider --basetemp=.tmp_t19_final_dev
& .\.venv-dev-rebuild\Scripts\python.exe -m pytest tests -q -p no:cacheprovider --basetemp=.tmp_t19_final_rebuild
& .\.venv-dev\Scripts\python.exe scripts/t19_fixture_check.py
& .\.venv-dev\Scripts\python.exe -m agenticrec.cli benchmark --config configs/benchmark.yaml --dry-run
```

机械联调证据留在忽略目录`artifacts/runs/t19/fixture_audit_20261010_v3/`，只用于工程验收，不能填正式效果表；聚合计数与SHA另存本报告相邻JSON。正式批次需要原始provider调用、每episode attempt/trace、source/config/模型checkpoint/data/feedback/fault SHA，并完成全部27run后才能生成正式系统表。

## 剩余任务

在30元范围审查实际token费用结算及保守停止策略，先使用development验证成本与现代连接；任何无法证明完整运行的方案要保留预算停止和失败分母，不能删难例、只取单seed或降低单方预算。随后按同条件正式执行U1/F/A/O及规定消融，独立重算原始记录、配对CI和失败分层；通过后才能标T19 DONE并进入T20。T04原资源ID/许可障碍保留。
