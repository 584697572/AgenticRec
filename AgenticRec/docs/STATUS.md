# 项目状态

最后核验：2026-10-01（Asia/Shanghai）。当前阶段：G0 的源码与资源审计完成；现代开发环境 T05 待开始，G1 未完成。

最高实施规范 `AgenticRec_完整复现与优化执行规范.md` 已完整读取，文件 SHA256 未变。任务依赖按规范附录保留于根目录 `TASKS.yaml`；规范不修改。

| 任务 | 状态 | 验证 / 证据 | 限制 |
|---|---|---|---|
| T00 锁定工作区、源码与出处 | DONE | 固定 SHA 0959ecb05b0794748426e73e6efc1b6b35ec433d，upstream pristine；upstream_lock.json | sibling 布局见 ADR-001 |
| T01 源码走读与静态审计 | DONE | 四项隔离源码检查通过；重复工具保留断言真实失败；源码路径/符号核验 | 审计不代表 A1；未修 bug |
| T03 核验资源并决定 A/B | DONE | Range 检查 0；资源审计 0；共享 ID 维度要求 1；route_decision、manifest、t03_evidence_index | A1 BLOCKED；B 已选，NOT EVALUATED |
| T02 legacy 环境 | TODO（conditional） | Python 3.9/Conda 未发现；未安装 | NOT VERIFIED；不阻塞规范 B 的独立 dev 工作 |
| T04 原资源运行 / 原评测 | BLOCKED | 原资源共同 ID 不能验收；legacy/来源/授权未就绪 | U0/A2 not_run，不能宣称原版复现 |
| T05 现代环境与协议 / 离线测试 | TODO | 尚无开发包或成功 lock | 下一批首先执行 |
| T06—T24 | TODO | 后续任务依赖未满足 | 所有推荐/Agent 指标 NOT EVALUATED |

## 本轮核验

- 读取器离线测试 3 项通过；公开原包 497,403,134 bytes、15 条目；两次响应正文合计 471,747 bytes（约 461 KiB），未下载完整 ZIP。
- 完整 settings、columns 和电影 Feather 已验证成员 CRC 与 SHA。目录 9,888 个唯一连续 ID=1..9888，各列无 null，257 个重复标题；无 ID 0 行。
- 矩阵仅头部 `(9889,9889)`；checkpoint 仅前缀，静态 metadata `n_items=36255`、embedding `(36255,32)`。没有 unpickle/torch.load；完整权重、矩阵语义和运行兼容性 NOT VERIFIED。
- 原包没有 ID 映射/授权文件。更多 embedding 行不能自动证明所有分数错误，但共享 ID 语义缺证据；保留真实退出 1，采用规范 B（ADR-004）。
- 原 CandidateBuffer 初始化对真实目录遗漏 ID 9888；保留未修复。upstream 无 patch。
- `verify_g0.py` 核对未改规范、固定源码、25 项依赖、实际审计和下载子集 hash；本轮产物/hash 在 t03_evidence_index，G0 index 保留为历史快照。

## Blocker

A1 所需共同 ID 映射、训练来源、独立资源授权、完整权重校验与 legacy 兼容性尚未验证。3.9/3.11 环境未建立；本机现有 Python 3.13.7、pyarrow 24.0.0 可做当前审计。GPU GTX1650Ti 4096 MiB；Torch/CUDA 训练兼容性 NOT VERIFIED。

GroupLens 的早期 urllib 证书错误保留，未禁用 TLS；官方数据条款已通过 Web 阅读，本轮无 MovieLens 下载。真实 LLM 默认关闭，`allow_paid_api=false`、`api_request_cap=0`，无真实调用或指标实验。

## 下一任务

先 T05：隔离现代开发环境、新包与 CLI、配置/错误 schema、doctor --offline、FakeLLM 和预定义工具 fixture；完成相应测试与真实 lock。再 T06：官方 MovieLens 1M 小规模数据获取、授权/hash、全局时间划分、训练期映射与过滤、warm/cold 分层、ID 契约、防泄漏测试。

不得提前训练新模型、修复后期 Agent、引入 UI 或其他架构；未验证写 NOT VERIFIED，未实验写 NOT EVALUATED。原资源 U0 和未来方法级重建 U1 明确区分。
