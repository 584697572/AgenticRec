# 项目状态

最后更新：2026-10-05（Asia/Shanghai）。当前 G1 原执行链已跑通，G0 的资源权威语义和原实验条件尚未闭合；保持 A1。

| 任务 | 状态 | 验证与剩余项 |
|---|---|---|
| T00 | DONE | 固定 RecAI 0959ecb05b0794748426e73e6efc1b6b35ec433d，原工作树 pristine |
| T01 | DONE | 原调用链、静态审计和原始失败证据保留 |
| T02 | DONE | Python 3.9.24，174 包离线重建及原资源/工具通过 |
| T03 | IN_PROGRESS | 原资源加载通过；权威训练标题映射、矩阵语义、预制包独立许可未闭合；最终 target 覆盖限制已计算 |
| T04 | BLOCKED（其余条件） | 真实单轮、两轮年份修改和全部45条A1原评测执行链通过；正式三个分支 hit 均0/45；canonical 子集、ID/许可及原 demo/reflection 条件未验收 |
| T05 | DONE | Python 3.11.14，两个隔离环境各15项基础测试通过 |
| T06—T24 | TODO | T06仍依赖T03/T05；新模型训练、完整benchmark和算法收益未完成 |

## 最新证据

- [检查与修复报告](../../reports/reproduction_repairs_20261005.md)：可选年份绝对差与反思历史传递补丁验证；53 项完整回归通过。原版/修复版各在 zero/fixed demo 模式完成真实上游组件的 fixture 反思流程，0 真实请求。动态默认 all-mpnet-base-v2 未缓存，未下载；真实 LLM 下启用这些功能仍 NOT VERIFIED。
- [目录契约检查](../../reproduction/native_runs/20261005_repairs/catalog_contract_summary.json)：补丁改变 1 个旧映射，年份不一致由 4 变 3；严格标题/年份检查为 22 个唯一匹配、17 个缺标题、6 个缺对应年份。这是原目录可达性检查，不证明 checkpoint ID 对齐，也不是新的评测指标。
- [真实多轮与原评测报告](../../reports/a1_multiturn_eval_20261005.md)：93次HTTP200，实际输入184724/输出17220/总计201944 token；高峰估算0.507208CNY，账单null。
- [公开摘要](../../reproduction/native_runs/20261005_session/live_summary.json)：原保存对话、逐样本命中及原函数指标复算一致，分母全部45。第二轮原prompt带此前输入和回答，原memory为4条。
- [失败诊断](../../reproduction/native_runs/20261005_session/failure_diagnostics_summary.json)：21个最终target不可映射、4个映射年份不同、44个target带年份后缀；索引37被截断且无Map，仍在分母中。去掉年份的6个模糊命中是诊断，不是正式结果或模型提升。
- 完整离线联调94次mock、真实0请求；40项全量回归与3项原指标测试分别通过。未知参数退出2、0请求；真实证据Key扫描、重复门禁和22个旧证据的字节检查通过。

## 当前阻塞

1. checkpoint权威标题映射、矩阵行标签和原包独立再分发许可 NOT VERIFIED。目录ID在范围内、权重加载和有限分数不等于语义正确；[源码复核](../../reproduction/source_alignment_review_20261005.json)不能独立恢复训练映射。
2. 作者canonical子集、hash seed、模型/提示及demo/reflection等原实验条件未对齐。该45条是A1派生输入，有覆盖与表示限制，不是论文A2。
3. 尚未完成T06/T07数据和指标前置，不能先训练新模型再补协议。

## 下一任务

继续T03来源/ID/许可核验及T04原配置恢复，保留当前零结果，不事后删除坏样本或修改正式阈值。前置闭合后按T06数据→T07指标→T08 BPR-MF→T09 LightGCN推进。

## 本机配置与历史

Key由非空环境变量优先、根目录.env次之，仅本机保留。执行命令由执行端负责。旧连接1/1与单轮2/2记录不变；新额度已用93/120，现有任务已关闭，不能删除记录重跑。后续使用剩余额度需沿同一授权计数，不能另起台账重置120次。

[单轮报告](../../reports/live_original_app_20261005.md)、[Key配置](../../reports/local_key_setup_20261005.md)、[环境报告](../../reproduction/environment_dev.json)和旧原资源报告保留历史快照身份。A1完整验收/A2尚未完成，B未启用；所有新模型收益为NOT EVALUATED。
