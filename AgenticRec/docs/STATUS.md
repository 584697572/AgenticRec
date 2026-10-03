# 项目状态

## 2026-10-04 单次真实连接通过

用户在已注入Key的PowerShell执行受限脚本，原OpenAICall真实连接SUCCESS/HTTP200，返回OK。实际prompt_tokens=12、completion_tokens=1、total_tokens=13，单次elapsed=2.1585443秒（非benchmark）；供应商账单未查询，actual_cost_cny=null，按当次公开高峰率估算上界0.000032 CNY。原始结果、额度记录及原源码SHA已在本地核对，唯一授权尝试已消耗，重复调用门禁验证不再发送HTTP且不改证据。新增5项SDK/HTTP离线测试、全16项离线测试通过；fixture不算真实连接。Key不进入项目文件；原app真实单轮/多轮仍NOT VERIFIED，原评测NOT EVALUATED，T04整体BLOCKED。报告见reports/live_connection_20261004.md，原始证据见reproduction/native_runs/20261004/。

## 2026-10-02 真实LLM配置指导

用户请求开始指导真实LLM运行。已提供LIVE_LLM_GUIDE、默认禁用的公开参数模板和不调用网络的preflight。配置/budget门禁3项测试通过；provider/model/Base URL/请求及token限额/金额预算等待用户答复，真实请求0，连接NOT VERIFIED。T04保持BLOCKED，其他资源/原评测前置未改变。当前操作先同终端注入Key→公开profile→preflight，再在授权齐备后准备单次连接；不把CONFIG_READY当连接成功。
同日更新：用户选择DeepSeek。已核对官方地址https://api.deepseek.com与建议模型deepseek-flash，并提供禁用的专用模板；不再要求用户自行寻找地址/model。首次拟议1次HTTP尝试、输出最多128 token、1 CNY预算，尚未授权；Key须在用户终端输入。真实连接仍NOT VERIFIED，T04仍BLOCKED。
以下为已验证的原资源离线运行快照；旧a1_evidence_index对应commit74c4fb5及当时文件内容，不用于新文档的当前hash。

最后验证：2026-10-01（Asia/Shanghai）。阶段G0收尾/A1原版运行前置检查；G1完整复现尚未验收。
规范完整读取且SHA不变；固定upstream保持pristine。本轮已跑通原资源离线工具链及原app的受控回调；真实LLM与评测仍有外部前置。

| 任务 | 状态 | 最新验证 | 剩余项 |
|---|---|---|---|
| T00 固定源码/出处 | DONE | HEAD=0959ecb05b0794748426e73e6efc1b6b35ec433d，原仓库未改 | 无 |
| T01 源码走读/风险复核 | DONE | 源码hash与原失败证据保留 | 后续按依赖修复 |
| T02 隔离legacy | DONE | Python3.9.24、pip check、原模块import、SDK mock、174包版本离线重建一致 | dynamic demo未验证；CPU不是GPU性能 |
| T03 资源契约/路线 | IN_PROGRESS | 5完整电影成员CRC/SHA、矩阵有限值、36项原模型参数完全一致 | 权威映射、矩阵ID语义、预制包独立授权 |
| T04 原资源复现 | BLOCKED | 真实工具前置/原app mock通过；原SDK单次DeepSeek真实连接HTTP200 | 原app单轮/多轮及新增调用授权、原eval数据及T03剩余前置 |
| T05—T24 | TODO | 未启动 | 不跳阶段训练、增强Agent或制作新UI |

## 已验证证据

- 原电影目录9888项、ID1..9888；完整矩阵9889×9889 float64，所有数值有限。仅下载原movie成员，未下载完整ZIP及其他domain；TLS中断后按CRC/SHA保护的分块缓存续取，没有禁用证书校验。
- gte-base固定revision c078288308d8dee004ab72c6191778064285ec0c，原SentenceTransformers真实编码标题和类别。Toy Stroy实际匹配Toys，不伪称纠错成功。
- 原UniRec加载原SASRec checkpoint；36个state key及参数逐项完全相同，无missing/unexpected。对9888目录项预测shape(1,9888)，分数有限；大词表不阻碍运行，但不能证明标题语义正确。
- 原Query、BufferStore、SQLSearch、SimilarItem、popularity/similarity/preference排序、Map均通过。Toy Story目录ID1062；本次相似召回493项；这是smoke输出，不是Recall指标。
- 原app.py实际启动loopback Gradio3.40.1，HTTP200；两轮回调走原SDK/工具/记忆，LLM响应为HTTP fixture。usage字段不是实际token用量，正式runtime_metrics全为null。
- app.py和one_turn_eval.py的help通过；后者按规范设置PYTHONPATH。原评测未运行；仓库/资源ZIP不含原评测jsonl，notebook依赖外部UniCRS输入及原有选择步骤。
- 8项离线回归通过；原始失败日志未覆盖。运行证据见reproduction/native_runs/20261001。
- 原Python文件逐个hash一致；只在独立副本requirements固定兼容包。原末尾ID遗漏、重复tool覆盖、训练模式等问题保持未修。

## Blocker

1. 2026-10-04单次DeepSeek真实连接已通过，1/1尝试已使用；原app单轮/多轮及其新增调用授权尚未完成。原app/批量评测不在本次连接授权范围内；通用模板保持默认禁用，不重置已消耗额度。
2. 原eval数据缺失；不把自造context/target或notebook保存输出当已校验原评测集。
3. checkpoint权威标题映射与矩阵ID语义NOT VERIFIED。notebook保存Toy Story→17、原目录→1062，保存输出不是checkpoint行标签。预制包再分发许可NOT VERIFIED；用户授权本地下载/检查，未再分发资源。

## 下一任务

继续T03取得权威mapping/provenance及授权信息；单次连接已完成，按规范§5.3下一步原app单轮，再多轮，最后校验原eval数据并运行原指标。每一步新增调用另按明确额度授权，不复用本次1/1已消耗记录。T04完整验收前不标DONE，A2和正式指标仍not_run / NOT EVALUATED。

路线A1（原资源兼容功能）；B未启用。旧报告/index保留历史快照身份，最新证据索引为reproduction/a1_evidence_index.json。
报告：[UPSTREAM_REPRODUCTION.md](../../reproduction/UPSTREAM_REPRODUCTION.md)。
