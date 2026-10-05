# T03 路线：A1 原资源兼容功能，完整验收仍待前置

2026-10-05 最新：原 App 真实单轮及两轮年份修改已验证；原 recbot、random、加权 popularity 各完整执行45条A1派生输入，正式模糊hit均0/45。实际93次HTTP200、201944 token，高峰估算0.507208CNY；账单未查询。21个target不可映射、4个映射年份不同、索引37截断且无Map，均保留。当前任务已关闭，不能删除记录重跑；T04其余条件仍BLOCKED，A2未验证。详见[多轮与评测报告](../reports/a1_multiturn_eval_20261005.md)。以下未运行/等待Key等文字为历史快照。

## 此前历史记录

2026-10-05 最新：用户选择根目录 `.env` 持久配置，执行端已自动读取并完成原 App 真实单轮 SUCCESS/VERIFIED（2次HTTP200、3880 token、目录约束全部通过）。当前单轮2/2额度已用；多轮/原评测及资源语义限制仍未验收，T04整体BLOCKED。见 [真实单轮报告](../reports/live_original_app_20261005.md)。下文尚未运行/等待Key的描述均为此前历史阶段。

2026-10-05 最新：A1 原资源兼容功能路线继续，T03 IN_PROGRESS、T04 BLOCKED；没有转 B。原工具/原 App 离线及单次真实连接通过，原 App 真实对话和原评测未验收。新调用授权已具备，等待用户 Key 终端运行；raw ReDial test 已取得，45 条 notebook 派生 A1 输入已准备，canonical 子集等价性、权威模型 ID 映射/矩阵语义及资源独立许可仍 NOT VERIFIED。T05 按其 T00/T01 依赖已完成；后续训练仍遵循 T03/T06/T07 前置。以下旧日期是历史调查快照，不能当当前阻塞状态。

2026-10-01最新结论：继续A1，不启用B。T02 DONE；完整原movie资源、真实原SASRec加载/推理及原工具链已通过，原app.py的loopback启动和两轮mock-LLM回调也通过。权威ID语义、预制包来源/独立授权未全部闭合，T03 IN_PROGRESS；真实LLM授权和原eval数据缺失，T04 BLOCKED。完整A1仍NOT VERIFIED，A2 not_run，正式指标NOT EVALUATED。

证据：`native_runs/20261001/`、`environment_legacy.json`、`resource_manifest.json`和`UPSTREAM_REPRODUCTION.md`。兼容diff仅requirements；固定upstream原Python源码不变。以下是ADR-005重开调查时的历史快照，其中“完整权重未下载/未安装”描述已被本轮真实执行取代。

2026-10-01，T03 **IN_PROGRESS**；候选路线 **A1**，最终路线 **NOT VERIFIED**。ADR-005 撤回上轮冻结 B 的决定。A1/U0 原版运行尚未验收；U0/A2 `not_run`，所有指标 **NOT EVALUATED**。

## 已确认事实

- 固定 RecAI 源码完整保留于 `RecAI/`，HEAD `0959ecb05b0794748426e73e6efc1b6b35ec433d`，无 upstream patch。
- Google 原资源 ZIP 可用，497403134 bytes、15 条目；之前两次小范围读取471747 bytes。原电影表9888项，ID=1..9888，矩阵头部(9889,9889)，checkpoint 静态配置和 embedding 行数36255。完整 ZIP/矩阵/权重未下载。
- 本轮从 [官方 PyPI 固定发布](https://pypi.org/project/unirec/0.0.1a4/)取得 UniRec wheel121607 bytes，核验发布 SHA256；包含 metadata 的新增响应正文136478 bytes。未安装依赖。版本仅是 upstream requirements 允许的最低版本，不能宣称是权重训练版本。
- UniRec `load_model_freely` 根据 checkpoint config 建模；`predict/forward_item_emb` 和 SASRec 序列路径按输入整数直接查 embedding，无电影标签 remap。原 RecModelTool 直接把目录 ID 用作 `item_seq/item_id`。
- 执行原函数体的 CPU fixture 能对9888项目录子集产生合法形状，原排名工具把 Toy Story 的目录 ID1062传入。这里只验证索引与调用路径，encoder 和权重是明确自造 fixture，不能当作原模型运行或推荐质量结果。

## 上轮结论修正

要求词表总行数完全相等过严。目录子集只要 ID 在模型范围内即可被查表；历史 `--require-alignment` 退出1只是行数比较，不足以证明原版不可用。因此撤回最终 B，按规范 A1 优先调查，不开始 B 数据重建。

## 剩余语义风险

| 电影 | 固定 notebook 保存的映射 ID | 实际原包目录 ID | 目录旧 ID 对应电影 |
|---|---:|---:|---|
| Toy Story | 17 | 1062 | Hot Shots! Part Deux |
| Jumanji | 105 | 1023 | Beautiful Thing |
| Grumpier Old Men | 232 | 1325 | Big Heat, The |

notebook 缓存 users/items=298073/36254，checkpoint 配置加 padding=298074/36255，提示二者可能来自同一映射规模。**这是由保存输出及配置作出的推断，notebook 未执行，输出可能陈旧；不能当作 checkpoint 的权威训练行标签。** 原包没有独立 mapping 文件，ID 语义仍 NOT VERIFIED。

## 当前缺项与下一步

1. 先建立规范 T02 的隔离 legacy 环境，保存真实 import/兼容性日志；不把现有 Python3.13 的 fixture 当作 legacy 复现。
2. 取得 checkpoint 权威训练映射/来源；必要时仅下载电影 checkpoint 做安全加载和原工具冒烟。下载权重本身不能提供缺失电影标签，当前不为此额外下载。
3. 只有 A1 前置验收后才进行 T04；按规范资源不可用/无法建立有效基线的事实重新评估 B，记录最小替代及可比性。原系统即使能输出分数，也不自动代表 ID 语义正确。

原包独立数据/权重许可仍 NOT VERIFIED；本地子集与 wheel 不提交或再分发，代码 MIT 不替代数据许可。矩阵完整 CRC/数值、checkpoint 完整 CRC/权重加载/实际库版本均 NOT VERIFIED。无真实 LLM 调用，allow_paid_api=false，api_request_cap=0。

历史事实及纠正见 `reports/t03_resource_audit.md`（旧快照）、`reports/t03_id_recheck.md`（本轮）、`reproduction/id_mapping_recheck.json`、`DECISIONS.md` 和两个分开的证据索引。
