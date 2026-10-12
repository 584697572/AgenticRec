# 原版证据集中补齐（2026-10-12）

本轮完成一次有限、只读的证据核查，未训练、未调用LLM、未下载数据包或权重、未联系作者。三项原版缺口没有全部关闭：旧ID语义及原包独立许可仍NOT VERIFIED，作者电影评测文件仍缺，T04/A2仍BLOCKED。完成的是调查与可核验材料整理，不是论文复现验收。

## Completed

- 当前官方HEAD、完整文件树、README及MIT许可取得并核验；HEAD仍是固定0959ecb05b0794748426e73e6efc1b6b35ec433d，未升级或fetch上游。
- 本地95个可用commit中，检查InteRecAgent的101个可达blob、98个文本blob及5个相关notebook版本。git log记录70个历史路径；rev-list对象记录69个路径，不将这两个不同口径混用。
- 检索全部状态的InteRecAgent/mapping/license Issue，返回9/1/0项；读#110及#75/#107/#106相关评论。mapping检索唯一结果#72属于其他子项目，没有作为电影映射证据。
- 阅读并保存arXiv v3论文、官方MovieLens10M条款；记录原包15成员、旧checkpoint静态字符串/标志。未执行notebook或反序列化权重。
- 整理[机器摘要](upstream_evidence_20261012.json)和[作者材料请求草稿](upstream_material_request_20261012.md)，草稿 **NOT SENT**。

## Files Created / Modified

新增两份只读采集/离线核验脚本、报告、摘要和请求草稿；更新STATUS、TASKS、DECISIONS和spec_issues。原始HTTP响应、头、时间、SHA、历史扫描及测试日志保存于忽略的artifacts/runs/upstream_evidence_20261012，不公开完整数据/权重或原始评测记录。

## Commands / Tests

```powershell
& reproduction/scripts/evidence_public_20261012.ps1
& reproduction/scripts/evidence_public_20261012.ps1 -Supplement -Output artifacts/runs/upstream_evidence_20261012/public_supplement
& reproduction/scripts/evidence_public_20261012.ps1 -Focused -Output artifacts/runs/upstream_evidence_20261012/public_focused
& reproduction/scripts/evidence_public_20261012.ps1 -Discussions -Output artifacts/runs/upstream_evidence_20261012/public_discussions
& AgenticRec/.venv-t20/Scripts/python.exe reproduction/scripts/audit_upstream_evidence_20261012.py --freeze
& AgenticRec/.venv-t20/Scripts/python.exe reproduction/scripts/audit_upstream_evidence_20261012.py
& reproduction/.venv-legacy/Scripts/python.exe -m unittest discover -s reproduction/tests -v
```

所有输出目录拒绝覆盖；freeze在修改既有状态文件前执行。原始下载已完成，以上不是提示在现有目录重复下载；离线audit可重跑，摘要只接受字节一致重建。四段采集合计共享18请求/4MiB正文预算，不重试、不跟随重定向、不使用身份验证或Key；仅指定官方HTTPS域名，TLS验证未关闭。

## Verification Results

64项既有复现测试通过，退出0；本轮没有重跑335项现代算法全量测试，也不把上轮335当本轮执行。HTTP18次、正文1,170,907 bytes（约1.12MiB），模型/数据包下载0、付费LLM请求0；这是受控采集正文，不是全协议流量。浏览工具另用于同一公开来源的文字核对，不计入该HTTP字节数。

首个commit详情响应触及131072字节上限，closed issue清单触及524288上限，均保留前缀并排除完整性结论；改用小commit列表和精确Issue搜索，未放宽网络边界。版本探测v4返回404保留；abstract实际只列v1/v2/v3，本报告仅采用v3。历史两份ReDial notebook约2.2MB，初始本地1MiB文本限额使audit失败；保留audit_first.log，将本地单blob限额改为3MiB后通过，网络预算不变。辅助默认GBK文本读取曾失败，改显式UTF-8后完成，不把辅助失败算正式评测。

离线核验逐一检查18响应的文件长度/SHA、累计预算、搜索完整标志/页数、官方tree非truncated、README/LICENSE与固定Git blob逐字节相同，以及692份既有跟踪文件/五原资源/12新训练权重的冻结SHA。可变状态入口及spec_issues明确不在保护清单；它们的变更由Git diff审阅。原规范、upstream固定HEAD及clean状态保持；所有推荐数字、旧报告和默认模型不改。

## Findings

### 1. 编号：仍缺权威字典

checkpoint静态metadata确认SASRec、36255 item行；目录9888电影、相似度9889×9889（含padding）。维度不同不单独证明错误，但现有源码直接把目录ID传给embedding/矩阵，没有可核验共同映射。

两个movies notebook版本按过滤后unique出现顺序创建item_id_map/user_id_map，然后导出新ID；没有找到持久化的旧raw_movie_id↔checkpoint行表。原包15成员也没有映射、split或评测清单。当前完整官方树和已检查历史范围未找到补充映射；这不是“全网不存在”的证明。

论文表中的用户/item数与旧checkpoint metadata相同，是来源相容线索，不是行标签证明。论文链接标MovieLens10M，但表A1为27,042,493交互；官方10M只有10,000,054评分，当前notebook实际读取CSV且有ml-latest注释。原始release/过滤后来源尚未唯一确定，不据此下载多个latest版本猜映射。[论文v3 §4.1/附录A](https://arxiv.org/html/2308.16505v3)、[官方10M说明](https://files.grouplens.org/datasets/movielens/ml-10m-README.html)

另一个边界：ItemCF notebook当前路径配置为Steam，输出float16；本地movie矩阵是float64。因此它提供算法参考，不能当本地movie矩阵的生成日志。不同电影标题/目录的source-backed别名仍不能恢复embedding行。

### 2. 许可：代码已明确，外部包仍未明确

官方代码MIT已核实；README仅说明志愿者准备外部包并提供链接，没有取得该包的单独许可说明。[MIT](https://github.com/microsoft/RecAI/blob/0959ecb05b0794748426e73e6efc1b6b35ec433d/LICENSE.txt)、[资源说明](https://github.com/microsoft/RecAI/blob/0959ecb05b0794748426e73e6efc1b6b35ec433d/InteRecAgent/README.md)

MovieLens10M条款对研究、署名、再分发和商业用途分别有条件；它不单独说明志愿者checkpoint或矩阵的许可。必须确认准确原数据release及其条款，并由有权发布者说明外部资源适用范围。本轮没有得出“禁止使用”或“允许全部用途”的法律结论。[官方条款](https://files.grouplens.org/datasets/movielens/ml-10m-README.html)

精确license Issue检索0结果仅表示该检索未返回材料；不是不存在授权的证明。项目继续只公开代码、摘要和重建步骤，不公开原数据包/旧权重。

### 3. 评测：补齐协议事实，未补齐实际电影样本

以下均来自arXiv **2308.16505v3（2024-01-30）**；没有将其自动等同2025 TOIS修订版。论文事实与本项目新协议分开：[§4.1/§4.3/附录A](https://arxiv.org/html/2308.16505v3)

| 条件 | 论文v3明确内容 | 当前边界 |
|---|---|---|
| LLM | GPT-4；列出gpt-4-0613，GPT-3.5为0613 | 本项目A1是DeepSeek，历史模型可调用性未测试 |
| 推荐工具 | SQL、ItemCF、无位置嵌入SASRec | checkpoint静态pickle中use_pos_emb/use_position_emb后均为BININT1 0；支持该配置线索，不证明ID语义 |
| 数据 | 5-core、leave-one-out，工具用train+valid训练 | 本项目质量主线是ML1M全局时间train-only，不能改它来冒称原条件 |
| 示例/召回 | 动态示例3条，硬候选1000，软召回前5% | 已有配置可恢复部分设置，但实际资源/示例集仍需确认 |
| 单轮样本 | 初采500，策略过滤后MovieLens表A1为496 | 原496条文件、sample IDs、生成日志未取得 |
| 单轮任务 | 全空间Recall@5；排序NDCG@20，19负例+1正例 | 旧45条ReDial输入及模糊hit不是同一协议 |
| 用户模拟 | 普通1000、lifelong100实例 | 对应原文件/随机名单/模拟输出未取得 |

此前“canonical 50条”的说法应修正：50是preprocess_redial.ipynb cell14的源码截断，A1最终45条；不能称论文电影canonical子集。原45条结果不删除、不重算为496条，也不修改上游指标。

发现作者相关回复：#75贡献者XuHwang在2024-10-27分享 **Steam** simulator/one-turn数据的RecDrive链接；本轮记录链接但不访问/下载该跨领域数据。#107由提问者称存在脚本后关闭，没有附movie评测文件；#106贡献者说明Steam元数据来自Kaggle和SASRec资源，不能将它套到movie。#110截至本轮API核验仍open/0评论。[#75作者回复](https://github.com/microsoft/RecAI/issues/75#issuecomment-2440073884)、[#107](https://github.com/microsoft/RecAI/issues/107)、[#106回复](https://github.com/microsoft/RecAI/issues/106#issuecomment-3759787039)

### 4. 新发现的生成风险：不据此判断论文实验

固定movies.ipynb cell37先令history排除最后一项，紧接着又以末10项覆盖history，并把同一列表最后项设为target。依该代码片段，非空列表目标会出现在history中。cell24当前N=900，而部分缓存输出展示500行；cell35抽样没有本cell显式seed。它们提示notebook代码/缓存不能替代实际生成日志。[固定notebook](https://github.com/microsoft/RecAI/blob/0959ecb05b0794748426e73e6efc1b6b35ec433d/InteRecAgent/preprocess/movies.ipynb)

本轮仅静态记录，**未证明作者正式评测实际使用该段**，不能宣称论文结果泄漏；也不执行或修复该段来生成新的“原canonical数据”。安全修复/生成研究若开展，须独立回归和协议，不覆盖原证据。

## Blockers

旧checkpoint训练映射/准确数据release、movie矩阵行标签/生成配置、外部包独立许可、论文movie canonical输入/候选及当时模型/示例运行记录仍缺。T04/A2 **BLOCKED**；本轮搜索任务DONE不代表这些缺口DONE。

## Deviations from Spec

None：按§1/§5/T03/T04执行有界补证，原B/upstream_rebuilt继续，A1保留。新增历史生成风险和旧“canonical 50”表述修正记ADR-045及spec_issues，不改原规范。未更换数据/模型/指标或升级上游。

## Current Project State

独立算法/Agent系统与最近真实训练证据保持；新增EA01有界证据调查完成。三项原版缺口仍开放，论文A2未验收。本轮只增加来源与条件认知，不产生任何准确率、延迟、token或成本提升指标。

## Next Tasks

下一动作是获得有权发布者的回复及准确文件身份，按[请求草稿](upstream_material_request_20261012.md)核对映射、许可、movie评测输入/19负候选/生成日志。当前没有发送授权，不发布Issue、不发邮件。若没有新材料，停止扩大检索/下载；独立研究可继续，先明确未暴露评价，再另行登记显式低分监督。T21及可选T22—T24原状态不改变。
