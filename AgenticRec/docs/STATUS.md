# 项目状态

最后核验：2026-10-01（Asia/Shanghai）。当前阶段：G0 资源契约复核，恢复 A1 优先调查；G1 未完成。

最高实施规范 `AgenticRec_完整复现与优化执行规范.md` 已完整读取，SHA256 未变。25 项任务依赖保留于根 `TASKS.yaml`。ADR-005 撤回 ADR-004 冻结 B 的决定，旧报告/索引仅作历史快照。

| 任务 | 状态 | 验证 / 证据 | 限制 |
|---|---|---|---|
| T00 锁定工作区、源码与出处 | DONE | 固定 SHA0959ecb05b0794748426e73e6efc1b6b35ec433d、upstream_lock，工作树 pristine | sibling布局 ADR-001 |
| T01 源码走读与静态审计 | DONE | 原4项审计通过；重复工具覆盖的真实失败保留 | 未修上游算法 |
| T03 核验资源并决定 A/B | IN_PROGRESS（重开） | 6项离线fixture/源码测试通过；id_mapping_recheck、manifest、route_decision | A1候选；最终路线NOT VERIFIED |
| T02 legacy环境 | TODO（conditional） | 无Python3.9/Conda、未安装 | 下一批先建立隔离环境 |
| T04 原资源工具/Agent运行 | BLOCKED | 真实模型、完整资源和legacy前置未就绪 | U0/A2 not_run；不能用fixture代替 |
| T05—T24 | TODO | 尚未启动 | 不提前数据重建或训练 |

## 当前证据

- 原电影表9888项、ID=1..9888；矩阵头部(9889,9889)；checkpoint metadata/embedding为36255行。原小JSON/表完整CRC与SHA已核验，完整矩阵/权重未下载；语义映射NOT VERIFIED。
- 新复核证明原UniRec直接按ID索引，较大词表允许目录子集；原工具把Toy Story的目录ID1062直接传入模型。严格行数相等只是诊断，不是运行条件。
- 固定notebook的保存映射为Toy Story17、Jumanji105、Grumpier Old Men232；真实目录分别1062、1023、1325。notebook人数/物品数加padding与checkpoint一致，提示不同映射来源；陈旧输出不能证明checkpoint权威标签，继续调查。
- 本轮从官方PyPI下载固定UniRec源码wheel121607bytes，加metadata新增响应正文136478bytes；仅静态读取，未安装。训练库版本NOT VERIFIED。
- 新增3项原函数体CPU fixture与旧3项Range测试共6项通过。一次失败是测试命名空间漏np，已补齐且保留日志。没有原权重推理或实验指标。
- 原CandidateBuffer遗漏合法ID9888、重复工具覆盖均未修。原源码pristine；规范未改。新证据索引`t03_recheck_evidence_index.json`对应本轮，旧index对应历史commit，不能当作当前文档hash。

## Blocker

权威checkpoint训练映射、完整资源校验、预制包独立授权和真实legacy兼容性待核验。Python3.9/3.11环境尚未建立；当前Python3.13.7仅做审计/CPU fixture。新增包未安装，未变更系统CUDA。

真实LLM关闭：allow_paid_api=false、api_request_cap=0；所有运行指标null / NOT EVALUATED。官方数据条款不等于预制包授权；资源子集/wheel不入Git或再分发。

## 下一任务

依规范先T02隔离legacy环境及实际import/兼容性日志，同时继续T03取得权重映射与来源。必要时仅下载电影checkpoint安全核验；完整资源与环境前置成立后T04无LLM工具冒烟。原版即使能打分，也要独立报告ID语义是否正确。

最终A/B路线暂不冻结；不提前启动MovieLens重建、模型训练、复杂Agent改造或UI。当前复核报告：[t03_id_recheck.md](../../reports/t03_id_recheck.md)。
