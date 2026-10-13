# 冻结短序列接入真实Agent：四版本同场结果

**DONE / VERIFIED**。配置、源码、输入和权重正式前冻结；默认版本未替换。

本批短序列相对混排的Hit点估计-5.00百分点，实际请求+1.66%、总token+2.92%。未证明稳定推荐收益，不能称准确率或资源全面优化。

本轮把短序列排序接入实际C1文本解析、证据验证、action0多轮状态合并、目录硬过滤和结构化标题交付；其余三种排序使用同一流程，分别真实调用LLM。微软原App本轮未运行。

原414监督/180开发/300最终用户全部排除，新的150正式用户与9校准用户也不同。正式120质量用户按公开3/5/10喜欢各40人，30反馈题、24硬条件、8不可满足；182题/版、222轮/版，共728系统任务/888轮。目标是85%时点后的新正反馈。旧25.89%、旧40%与本批不能直接相减。

## 实际质量与任务完成

| 版本 | Hit@10 | 命中/120 | Recall@10 | NDCG@10 | MRR@10 | 严格完成/174 | 严格完成率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 短序列seed42 | 30.83% | 37 | 0.016572 | 0.060871 | 0.128515 | 168 | 96.55% |
| 现有1:.5:.25混排 | 35.83% | 43 | 0.032241 | 0.086289 | 0.173843 | 169 | 97.13% |
| 标准ItemCF | 35.83% | 43 | 0.027500 | 0.077329 | 0.137335 | 170 | 97.70% |
| 热门榜 | 29.17% | 35 | 0.034151 | 0.080777 | 0.143839 | 169 | 97.13% |

Hit表示前十部至少命中一部，不能替代严格按条件完成。解析或交付失败仍保留在120人的分母；严格分母174可满足题，8拒绝另报。

| 短序列对照 | Hit差值（百分点） | 配对95%区间（百分点） | 三次Holm p |
|---|---:|---|---:|
| 相对现有1:.5:.25混排 | -5.00 | [-13.33,+3.33] | 0.9404 |
| 相对标准ItemCF | -5.00 | [-12.50,+2.50] | 0.9404 |
| 相对热门榜 | +1.67 | [-6.67,+10.00] | 0.9404 |

严格完成相对混排差值-0.57百分点，95%区间[-1.72,+0.00]；预定-2百分点非劣：建立。
相对现有混排的稳定质量收益：NO VERIFIED GAIN。其他指标和失败均公开，不因单个seed或单档较高就替换默认。

## 实际消耗

| 版本 | 请求 | 输入token | 输出token | 语义修复请求 | turn p50秒 | turn p95秒 | 排序p95秒 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 短序列seed42 | 245 | 168475 | 40369 | 23 | 1.359 | 2.429 | 0.0878 |
| 现有1:.5:.25混排 | 241 | 163146 | 39768 | 19 | 1.466 | 2.220 | 0.1609 |
| 标准ItemCF | 240 | 163376 | 39082 | 18 | 1.386 | 2.050 | 0.0482 |
| 热门榜 | 242 | 164625 | 39555 | 20 | 1.385 | 2.501 | 0.0930 |

校准93＋正式968＝1061次真实生成请求。正式余额窗口减少0.86元，校准开始至正式结束观察减少0.95元；独立账单未知，不把窗口差当精确实验账单。
DeepSeek deepseek-flash、temperature0、非思考、实际1024输出上限、零网络重试；最多三次语义修复单独计入请求/token。不同上一回答会改变后续解析上下文，不能预设各版请求必然相同。串行轮换执行顺序，排序预热；延时是本机/本批实测，网络和厂商缓存有影响，不声称工业QPS或生产加速。没有新下载或训练。

## 条件、多轮和拒绝

| 版本 | 条件通过/24 | 多轮整题通过/30 | 结构正确拒绝/8 | 有解释拒绝/8 |
|---|---:|---:|---:|---:|
| 短序列seed42 | 24 | 25 | 8 | 0 |
| 现有1:.5:.25混排 | 24 | 26 | 8 | 0 |
| 标准ItemCF | 24 | 27 | 8 | 0 |
| 热门榜 | 24 | 26 | 8 | 0 |

原C1年份证据校验等语义失败保留，没有根据正式结果修规则、放宽条件或重跑失败。空候选的验证结构可算正确拒绝；空回答不算有解释。二/三轮操作覆盖加类型、组合类型、切换类型、清空、缩年份、换批及改数量，实际六种操作各5题。

## 同用户直接公开输入的组件验证

以下720份输出先保存排名再评分；它们直接使用公开电影列表，没有经过LLM解析。除主seed42外，另外两个seed仅离线组件验证，不能冒称三次独立在线运行。

| 组件 | 命中/120 | Hit@10 | Recall@10 | NDCG@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|
| sequence:42 | 37 | 30.83% | 0.016572 | 0.060871 | 0.128515 |
| sequence:7 | 41 | 34.17% | 0.022192 | 0.070900 | 0.139478 |
| sequence:2026 | 43 | 35.83% | 0.026910 | 0.068637 | 0.140919 |
| mixed:None | 43 | 35.83% | 0.032241 | 0.086289 | 0.173843 |
| itemcf:None | 43 | 35.83% | 0.027500 | 0.077329 | 0.137335 |
| popular:None | 35 | 29.17% | 0.034151 | 0.080777 | 0.143839 |

实际LLM质量输出与同用户直接输入排名一致数量（各120）：{'sequence': 119, 'mixed': 119, 'itemcf': 119, 'popular': 119}。不一致按真实输出计分，不用纯排序替换失败或不同的实际交付。

## 验证与证据

- 校准48系统任务/80轮、93次HTTP200均有usage；独立重放78成功执行、2语义失败保留。
- 正式888轮的条件/质量/身份/数量/前批排除与整题状态独立重算，868次成功执行从冻结权重重放；20次解析失败保留。逐请求意图/响应/用量全对账。
- 原包重新检查时间边界、电影身份、150正式/9校准用户与所有旧监督/评价用户隔离；源码/输入/旧模型SHA、720组件评分、bootstrap/置换/Holm、严格非劣均通过独立复核。
- 443项完整测试一次通过，另9项审计测试通过，共452项已验证；不称单次452全套。22项相关组件测试也通过。四排序钩子与旧300用户逐项重现共1200排名，不使用旧标签调参。
- 全部数据、权重、原始排名、标签、Key和余额/请求日志留忽略目录，只公开聚合。原上游和默认未改，paper-notes未动。有限批次结束即停止调用。Blocker：None。

[预测前协议](sequence_agent_protocol_20261013.md) · [全部聚合/统计](sequence_agent_20261013_validation.json) · [独立审计](sequence_agent_20261013_independent_audit.json) · [测试/等价记录](sequence_agent_20261013_tests.json) · [原校准审计](sequence_agent_20261013_calibration_audit.json)

原校准审计记录当时审计脚本SHA；后来增加纯组件审计，不改变已冻结运行逻辑或评分。最终审计SHA是公开当前脚本。私密原始证据目录 `optimized/artifacts/comparison/sequence_agent_20261013/`，已完成目录禁止覆盖。协议保留创建时NOT EVALUATED，本报告为最终状态。

## 再现入口

根目录已有数据/权重环境，先使用全新目录；prepare只取已确认来源资源，无训练和下载。

```powershell
$study = 'optimized/artifacts/comparison/sequence_agent_recheck'
$recPython = 'optimized/AgenticRec/.venv-t20/Scripts/python.exe'
& $recPython -m agenticrec.evaluation.sequence_agent prepare --directory $study
& $recPython -m agenticrec.evaluation.sequence_agent regression --directory $study
& $recPython -m agenticrec.evaluation.sequence_agent calibration --directory $study
& $recPython optimized/AgenticRec/scripts/audit_sequence_agent.py --directory $study --output "$study/calibration_audit.json" --calibration-only
& $recPython -m agenticrec.evaluation.sequence_agent freeze --directory $study
& $recPython -m agenticrec.evaluation.sequence_agent offline --directory $study
& $recPython -m agenticrec.evaluation.sequence_agent formal --directory $study
& $recPython -m agenticrec.evaluation.sequence_agent report --directory $study
& $recPython optimized/AgenticRec/scripts/audit_sequence_agent.py --directory $study --output "$study/independent_audit.json"
```

每步exit0再执行下一步。calibration/formal为真实付费调用，历史完成批次不要重复。该集合现已暴露，再跑只是复查，不称新盲测；新增研究需要另一份预先固定的评价方案。
