# 目录身份、年份、别名与旧模型映射核查（2026-10-05）

本轮执行用户认可的顺序前两步：先核查目录身份，再寻找旧模型映射。暂不追论文条件；原 A1 证据保留，质量实验按规范 §5.5 的独立重建路径准备。数字由公开摘要和 native 测试日志生成，不填写未执行的推荐指标。

## Completed

- 固定原目录/原数据哈希，以 [GroupLens 官方 MovieLens10M](https://grouplens.org/datasets/movielens/10m/) movies.dat 核查标题、年份和类型。匹配失败或歧义明确保留。
- 完整来源 metadata 身份/别名先冻结，之后才读取保存的45条输入做事后诊断；不根据命中结果设计别名或筛 test。
- 检查原包15成员、完整本地源码历史70个 distinct InteRecAgent 路径及两版 movies notebook，未取得旧训练编号表。搜索不证明全网不存在。
- T03 审计/路线选择通过；旧模型/矩阵语义仍 NOT VERIFIED。ADR-015 选择独立质量路径，解除 T06 的审计依赖，未实施新训练。

## Files Created / Modified

新增 catalog_identity、fetch_ml10m_metadata、freeze_catalog_identity、windows_metadata_range、报告生成脚本及11个 identity 测试。更新资源 manifest、route_decision、STATUS、TASKS、DECISIONS、README 与 spec_issues。元数据、源文件、身份/别名和 target 逐行结果仅在忽略目录，本报告/公开摘要不再分发 MovieLens 数据。

公开证据：reproduction/native_runs/20261005_identity/。本地4个冻结文件：data/metadata_private/catalog_identity_20261005/ 下的 catalog_to_ml10m.jsonl、source_aliases.jsonl、saved_target_diagnostics.json、manifest.json；这不是 T06 的训练期 ID map。

## Commands / Tests

```powershell
& .\reproduction\.venv-legacy\Scripts\python.exe reproduction/scripts/fetch_ml10m_metadata.py
& .\reproduction\.venv-legacy\Scripts\python.exe reproduction/scripts/freeze_catalog_identity.py
& .\reproduction\.venv-legacy\Scripts\python.exe -m unittest discover -s reproduction/tests -v
```

获取/冻结脚本重复运行退出2，不覆盖或重新下载。先保存缺实现的测试失败，后新增11项及完整64项通过。另完成 compile、范围/路径/域名门禁和22字节原生suffix入口冒烟；原始命令、stdout/stderr和退出码保留。

## Verification Results

| 检查 | 实测结果 |
| --- | ---: |
| 原目录条目 | 9888 |
| 官方元数据条目 | 10681 |
| 标题/年份/类型唯一身份匹配 | 9883 |
| 身份歧义 | 5 |
| 来源可追踪别名记录 | 12189 |
| 原直接匹配保持 | 22 |
| 保存输入的可解析目标 | 28 |
| 保存输入的年份冲突 | 6 |
| 保存输入在该来源缺标题 | 11 |

上述是元数据/解析检查，不是 Recall、ID Hit 或算法提升。原输入45条、目录编号/年份、权重、矩阵和三个正式0/45指标不改。5个歧义未随意挑一个ID，6个年份冲突未覆盖为目录年份，11个来源缺标题不代表现实中不存在该电影。

完整64项回归退出0；已有431个私有运行证据和旧正式产物字节保持。CRC/SHA通过。提取元数据和条款共246440字节，加suffix冒烟22字节，本批共246462字节。压缩包尾部含其它成员压缩片段；仅完整抽取movies.dat/README，没有抽取ratings或tags、完整ZIP或新模型。新增LLM请求0。

## Findings

仅接受来源中的明确a.k.a.别名和英文冠词位置差异，配合NFKC/大小写/空白规范化；无模糊匹配、生成中文译名或强制年份修正。新增6个source-backed解析保留现有目录ID，不插入电影。源数据的标题也可能有人工录入错误，结论限定为该版本官方元数据的一致性。

目录→官方raw_movie_id匹配不能恢复checkpoint→电影或矩阵→电影映射。原预处理按过滤评分的encounter order编号，导出时丢弃原raw ID；本地两版均不保存item_id_map。原checkpoint metadata的label也不能单独确定原评分版本/过滤顺序，本轮没有加载旧pickle到含Key的进程。

Python HTTPS证书链校验失败；Windows默认证书链验证成功，所有正文均按206和Content-Range校验，不关闭TLS、不回退整包下载。[官方条款](https://files.grouplens.org/datasets/movielens/ml-10m-README.html)许可本地研究并限制再分发；它不证明第三方原模型包的独立许可。

## Blockers

旧checkpoint训练编号、矩阵生成语义、原包独立许可仍未获证明；5个目录身份歧义与6个target年份冲突保留，未伪装修复。原论文canonical条件不再阻塞独立新实验。新质量实验尚缺T06/T07，不是现在就能开始模型训练。

## Deviations from Spec

按用户授权及规范§5.5采用独立B/upstream_rebuilt质量路径，A1历史功能证据保留；属于已提供的fallback，非新设计。当前ML10M只用于旧目录来源核查，主benchmark仍按T06固定官方稳定MovieLens1M，未换为latest或事后筛出的28条。重大决定ADR-015记录影响；原规范/upstream未改。

## Current Project State

T00/T01/T02/T03/T05 DONE；T03 DONE指完成审计/缺失记录和路线决定，不表示旧语义通过。T04原完整验收BLOCKED；T06 TODO且T03/T05依赖满足，T07—T24 TODO。训练、补丁收益和新推荐指标均NOT EVALUATED。

## Next Tasks

T06：取得官方稳定MovieLens1M，固定全局时间窗口，原ID↔模型ID只由训练候选产生，记录冷启动/排除覆盖，所有已评分物品不得当未知负样本。通过数据不变量后T07手算指标，再T08 BPR-MF/T09 LightGCN。保留上游planner/toolbox方法对照，不把新编号套到旧权重或把新数据结果当论文原版。
