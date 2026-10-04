# 项目状态

最后更新：2026-10-05（Asia/Shanghai）。当前 G0 的资源语义核验尚未闭合，G1 的原版真实运行与现代基础建设并行推进。

| 任务 | 状态 | 验证与剩余项 |
|---|---|---|
| T00 | DONE | 固定 RecAI 0959ecb05b0794748426e73e6efc1b6b35ec433d，原仓库 pristine |
| T01 | DONE | 源码走读及原始失败证据保留 |
| T02 | DONE | Python 3.9.24，174 包离线重建、原资源加载/工具验证通过 |
| T03 | IN_PROGRESS | 原资源完整性/加载通过；checkpoint 标题映射、矩阵语义和预制资源许可未全部核验 |
| T04 | BLOCKED | 真实连接已验证；受限原 App 单轮准备就绪，等待用户 Key 终端执行；多轮/原评测未执行 |
| T05 | DONE | Python 3.11.14，7 锁定依赖及项目离线重建一致；两环境各 15 测试、doctor/fixture/依赖检查通过 |
| T06—T24 | TODO | T06 依赖 T03/T05；尚未训练新模型或修改 Agent 算法 |

## 本轮完成的证据

- 原 App 完整离线单轮成功：真实资源、原 app.py 和原 Filter/Ranking/Map 执行，三个电影 ID 满足 Comedy、年份≥1990；2 次 mock HTTP，真实请求 0。没有把这个结果写为 live。
- 用户已授权继续请求。本次保守采用已提出的具体单轮额度：2 次 HTTP 尝试、512 输出 token/次、1 CNY、deepseek-flash、thinking disabled、零重试。新额度与 2026-10-04 已用完的连接额度分开。
- 真实 UniCRS/ReDial test 固定 commit 和 Git blob/SHA，8,751,948 bytes、1,342 条对话。42 个整数年份/日期异常先保存原失败，再通过独立 dtype 适配运行原 notebook 函数；筛选得到 45 条 A1 输入。未下载 train/valid/DBpedia，未把它宣称为作者原 50 条文件。
- T05 不依赖 T04；按规范任务卡完成现代基础，没有提前实现推荐训练、复杂 Agent 或 UI。

详细结果、命令、错误与差异见 [本轮阶段报告](../../reports/progress_20261005.md)。环境见 [environment_dev.json](../../reproduction/environment_dev.json)。完整私有日志在 reproduction/logs/；公开摘要和哈希在 reproduction/native_runs/20261005/ 与 reproduction/progress_evidence_20261005.json。

## 当前阻塞

1. Key 已由用户配置在自己的 PowerShell，但不会自动进入本执行端。授权已具备，尚缺该终端执行真实原 App 单轮后的结果。命令见 [LIVE_LLM_GUIDE](../../reproduction/LIVE_LLM_GUIDE.md)。不要发送 Key。
2. checkpoint 权威标题映射、矩阵行标签及原包独立再分发许可 NOT VERIFIED；目录 ID 在模型范围内和分数有限不能替代语义验证。Issue #110 公开评论本轮复查为空。
3. canonical 原评测子集、作者 hash seed/设置未取得。A1 输入已可重复准备，但原模糊文本 hit 尚未运行；不等同 ID Hit@K 或论文 A2。

## 下一任务

受限真实原 App 单轮 → 核验 trace/usage → 多轮 → 原评测，保持规范 §5.3 顺序。继续 T03 的来源/语义核验；闭合后执行 T06 数据与时间划分、T07 指标单测，再 T08/T09 算法基线与 LightGCN。没有真实训练/评测的指标保持 NOT EVALUATED。

## 历史事实

2026-10-01：5 个原 movie 资源完整 CRC/SHA、原 SASRec 36 参数逐项一致、原 app loopback HTTP200 和两轮 mock 回调通过；原文件未改。2026-10-04：用户完成单次真实 DeepSeek 连接，HTTP200/OK，prompt/completion/total=12/1/13，2.1585443 秒（单次连接耗时）；实际账单未查询，费用 null。旧 evidence index 只验证当时快照，不用于新文档当前哈希。

路线 A1 保留，B 未启用；完整 A1、A2 和推荐/Agent 改进尚未完成。
