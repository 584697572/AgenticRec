# T03 小范围资源核验与路线决定

## Completed

T03 DONE：原链接可达性、15 项 ZIP 清单、所需 settings、真实表 schema/ID、矩阵头部、checkpoint 静态 metadata 及授权缺口已记录，路线 B 已决定。DONE 仅表示资源调查和路线决定验收；不表示原资源可运行。

## Files Created / Modified

新增 `reproduction/scripts/inspect_remote_resources.py`、`validate_resource_subset.py`、`reproduction/tests/test_remote_zip.py`、`reproduction/t03_evidence_index.json`、`resource_contract_summary.json`、`archive_inventory.json` 和本报告。两个 summary/清单保存实际核验事实以便 Git 内审阅，完整原始日志及下载子集留在忽略目录。更新 manifest、route_decision、verify_g0、STATUS、TASKS、DECISIONS、源码审计及 spec_issues；不提交预制数据/权重。

## Commands / Tests

```text
python -m unittest discover -s reproduction/tests -v
python reproduction/scripts/inspect_remote_resources.py --byte-cap 2097152
python reproduction/scripts/inspect_remote_resources.py --details --byte-cap 2097152
python reproduction/scripts/validate_resource_subset.py
python reproduction/scripts/validate_resource_subset.py --require-alignment
python reproduction/scripts/verify_g0.py
git diff --check
```

证据及每条命令的真实 stdout、stderr、exitcode 见 `reproduction/logs/t03_20261001/`，文件 SHA256 见本轮 evidence index。前一轮 G0 index 为历史快照，不能当作本轮更新文档的当前 hash。

## Verification Results

- Range 读取器 3 项离线测试通过；两个网络检查均退出 0，总响应正文 471,747 bytes，约 461 KiB；包含重复目录及下载确认页，不是网络总协议流量。
- settings、columns、Feather 完整成员 CRC/SHA 校验通过。目录 9,888 行；ID 唯一连续 1..9888，无 ID 0；各列无 null，tags 为 list<string>；标题有 257 个重复条目。
- 矩阵头部 `(9889,9889)`、float64；理论文件长度与 ZIP 清单一致；完整矩阵值和 CRC **NOT VERIFIED**。
- checkpoint 只读取前缀，`pickletools.genops` 静态分析完整 metadata 指令；配置 n_items=36255，物品 embedding 为 `(36255,32)`。没有 unpickle、torch.load 或权重执行。训练来源、完整 CRC、运行兼容性 **NOT VERIFIED**。
- 一般审计退出 0；共享 ID 维度要求真实断言失败，退出 1；这是保留的 blocker 证据，不是测试全绿。
- 原 CandidateBuffer 初始化在真实目录长度下遗漏合法 ID 9888；原源码未修改。
- 规范 SHA、固定 upstream HEAD、源码 pristine、任务依赖及资源证据由 verify_g0 验证。提交前 diff 检查记录于日志。

## Findings

原包 497,403,134 bytes，Google 服务确实支持 Range；不因历史 Issue #110 宣称链接失效。包内没有 ID 映射、授权说明、评测文件或训练 split。更多 embedding 行本身不足以证明所有分数错误，但不能确认 ID 语义一致，不能用猜测映射补齐原版基线。既有 pyarrow 24.0.0 可读真实 Feather，本轮无依赖安装。

官方 [MovieLens 10M 条款](https://files.grouplens.org/datasets/movielens/ml-10m-README.html)要求研究使用注明来源，重分发另行许可，商业用途事先许可；该条款不能推定预制 checkpoint/整个包的许可。原子集仅保留本地，未重新分发。

## Blockers

A1 原资源共享 ID 契约无法验证；checkpoint 训练映射/来源、预制包独立授权均缺少证据。T04 BLOCKED，U0/A2 `not_run`。legacy 3.9 和现代 3.11 环境尚未建立。真实 LLM 默认关闭，未取得预算；离线 B 工作可独立推进。

## Deviations from Spec

未引入新路线；采用规范已有 B fallback。原资源实际情况与共同 ID 假设不一致，记录 ADR-004。仅下载核验所需子集，完整矩阵/checkpoint 的校验明确 NOT VERIFIED。沿用 ADR-001 的 sibling 工作区布局，upstream 保持原样。

## Current Project State

G0 的 T00/T01/T03 已完成；T02 未执行，不能宣称 legacy 复现成功。G1 尚未完成，T05 TODO，B NOT EVALUATED。所有推荐/Agent 指标仍 null，没有模型训练、真实 API、付费调用或 UI。

## Next Tasks

依规范先 T05：隔离现代环境、新包、配置与错误 schema、doctor --offline、FakeLLM/工具 fixture；验证通过后 T06：少量官方 MovieLens 1M 下载、授权和 hash、时间划分、warm/cold 分层、ID 契约和防泄漏测试。T07 后的训练与复杂 Agent 改造不提前启动。
