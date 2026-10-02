# 项目状态

## 2026-10-02 真实LLM配置指导

用户请求开始指导真实LLM运行。已提供LIVE_LLM_GUIDE、默认禁用的公开参数模板和不调用网络的preflight。配置/budget门禁3项测试通过；provider/model/Base URL/请求及token限额/金额预算等待用户答复，真实请求0，连接NOT VERIFIED。T04保持BLOCKED，其他资源/原评测前置未改变。当前操作先同终端注入Key→公开profile→preflight，再在授权齐备后准备单次连接；不把CONFIG_READY当连接成功。
以下为已验证的原资源离线运行快照；旧a1_evidence_index对应commit74c4fb5及当时文件内容，不用于新文档的当前hash。

最后验证：2026-10-01（Asia/Shanghai）。阶段G0收尾/A1原版运行前置检查；G1完整复现尚未验收。
规范完整读取且SHA不变；固定upstream保持pristine。本轮已跑通原资源离线工具链及原app的受控回调；真实LLM与评测仍有外部前置。

| 任务 | 状态 | 最新验证 | 剩余项 |
|---|---|---|---|
| T00 固定源码/出处 | DONE | HEAD=0959ecb05b0794748426e73e6efc1b6b35ec433d，原仓库未改 | 无 |
| T01 源码走读/风险复核 | DONE | 源码hash与原失败证据保留 | 后续按依赖修复 |
| T02 隔离legacy | DONE | Python3.9.24、pip check、原模块import、SDK mock、174包版本离线重建一致 | dynamic demo未验证；CPU不是GPU性能 |
| T03 资源契约/路线 | IN_PROGRESS | 5完整电影成员CRC/SHA、矩阵有限值、36项原模型参数完全一致 | 权威映射、矩阵ID语义、预制包独立授权 |
| T04 原资源复现 | BLOCKED | 真实工具前置通过；原app HTTP200，两轮原回调+mock LLM通过 | 真实provider/model/预算、原eval数据及T03剩余前置 |
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

1. 规范§3.3/§5.3要求真实API调用先明确provider、model ID、请求/token限额、预算金额及单位、安全配置密钥。默认allow_paid_api=false、api_request_cap=0；真实请求0。
2. 原eval数据缺失；不把自造context/target或notebook保存输出当已校验原评测集。
3. checkpoint权威标题映射与矩阵ID语义NOT VERIFIED。notebook保存Toy Story→17、原目录→1062，保存输出不是checkpoint行标签。预制包再分发许可NOT VERIFIED；用户授权本地下载/检查，未再分发资源。

## 下一任务

继续T03取得权威mapping/provenance及授权信息；补齐真实LLM配置与预算后，按规范§5.3先单次连接，再原app单轮/多轮，最后校验原eval数据并运行原指标。T04完整验收前不标DONE，A2和正式指标仍not_run / NOT EVALUATED。

路线A1（原资源兼容功能）；B未启用。旧报告/index保留历史快照身份，最新证据索引为reproduction/a1_evidence_index.json。
报告：[UPSTREAM_REPRODUCTION.md](../../reproduction/UPSTREAM_REPRODUCTION.md)。
