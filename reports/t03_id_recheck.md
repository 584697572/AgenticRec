# 原资源 ID 契约复核

## Completed

完成用户要求的上游预处理、资源映射、UniRec 加载及打分路径检查。纠正“维度不相等就不能运行”的判断，撤回冻结 B，恢复规范 A1 优先调查。T03 重新 IN_PROGRESS，原版运行不标完成。

## Files Created / Modified

新增 `reproduction/scripts/fetch_unirec_audit.py`、`recheck_id_contract.py`、`reproduction/tests/test_id_lookup.py`、`id_mapping_recheck.json`、`unirec_source_manifest.json`、`t03_recheck_evidence_index.json` 及本报告。更新 STATUS、TASKS、DECISIONS、manifest、route_decision、源码审计、verify_g0 和旧维度诊断的说明。没有修改规范、RecAI 或上游算法；旧报告/索引保留为对应 commit 的历史快照。

## Commands / Tests

```text
python reproduction/scripts/fetch_unirec_audit.py
python reproduction/scripts/recheck_id_contract.py
python -m unittest discover -s reproduction/tests -v
python reproduction/scripts/verify_g0.py
git diff --check
```

命令 stdout/stderr/exitcode 保存于 `reproduction/logs/t03_recheck_20261001/`。固定 notebook 仅解析源码和保存输出，未执行预处理，也没有下载原交互数据。

## Verification Results

- 新增下载响应正文136478 bytes（约133 KiB）：其中固定 UniRec wheel121607 bytes，其余为官方 metadata；wheel SHA256 与官方记录一致。未安装包，没有下载大矩阵或完整 checkpoint。
- 6 项离线测试通过：3 项 Range 测试及3项原函数体 CPU fixture。目录全部9888项可索引36255行 embedding；ID36255越界被拒绝；原 RecModelTool.run 将 Toy Story 目录ID1062直接作为序列输入，无额外映射。
- 模型 weights/encoder 是明确自造确定性 fixture，仅验证索引、输入流和原函数体，不代表完整 SASRec 推理、原权重加载或推荐质量。没有计算 Recall/NDCG/延迟/token 等指标。
- 首次 test run 有1项失败：隔离原 CandidateBuffer 处理 numpy 输出时测试 harness 漏提供 np，触发 NameError 被原工具吞掉。补齐测试命名空间后6项通过；保留首轮失败日志，未把它误报成上游 bug。
- 规范 SHA、固定 pristine upstream、任务依赖、真实资源与新复核记录由 verify_g0 验证；提交前 diff 检查和本轮文件 hash 单独保存。

## Findings

| 链路 | 实际代码 / 证据 | 结论 |
|---|---|---|
| movies.ipynb cell7/8/15 | 按交互出现顺序生成 i+1，目录采用相同 map | 若同批生成，物品与交互应共用映射 |
| notebook 保存输出 | users/items=298073/36254，checkpoint配置=298074/36255 | 与padding后规模吻合；来源相关性是推断 |
| 三项标题对照 | Toy Story 17→目录1062；Jumanji105→1023；Grumpier Old Men232→1325 | 保存的映射与原包表不同；不能用陈旧输出证明实际权重标签 |
| 目录原ID对应 | 17=Hot Shots! Part Deux，105=Beautiful Thing，232=Big Heat, The | 未确认权重映射时有语义风险 |
| RecModelTool.run | 标题→目录整数，直接送item_seq/item_id | 没有目录到模型的转换 |
| UniRec predict/forward_item_emb | 直接embedding lookup候选ID | 子集目录无需覆盖完整词表；严格行数相等不是运行条件 |
| UniRec load_model_freely | checkpoint config构建模型、load_state_dict(strict=False) | 不接收电影目录，不能自动修正标签映射 |
| 模型推理模式 | loader和原RecModelTool初始化均未调用eval | 静态风险；实际dropout影响未验证，不在本轮修复 |

UniRec代码来自 [官方固定0.0.1a4发布](https://pypi.org/project/unirec/0.0.1a4/)，满足上游依赖下界；**checkpoint 实际训练库版本 NOT VERIFIED**，没有静默升级或把该候选视作成功 legacy lock。符号行号和源 hash 见 `id_mapping_recheck.json`。

## Blockers

checkpoint 权威电影 ID 映射/训练来源缺证据；保存 notebook 输出不能替代它。完整矩阵/checkpoint 校验、真实模型加载、Python3.9 legacy环境和独立资源授权仍 NOT VERIFIED。T04 not_run，真实 LLM 默认关闭。

## Deviations from Spec

纠正本项目先前偏早转B的判断，恢复规范§0 A1优先；见ADR-005。规范未改。采用其允许的现有解释器 CPU fixture 和静态源码核验，明确不冒充 legacy 运行。未训练、未重建数据、未改 prompt/评测。

## Current Project State

T00/T01 DONE；T03 IN_PROGRESS（重新打开路线决定）；T02 TODO；T04 BLOCKED；T05及后续未启动。A/B最终路线NOT VERIFIED，所有指标NOT EVALUATED。累计两个资源调查轮的响应正文608225 bytes（约594 KiB，不含早期网页probe与浏览工具流量）。

## Next Tasks

先按T02建立隔离legacy环境并记录真实import/兼容性证据；继续T03调查权重映射与来源。必要时只取电影checkpoint安全检查；具备资源/环境前置后T04无LLM工具冒烟。只有取得相应阻塞证据后才重新决定B，不先启动MovieLens重建。
