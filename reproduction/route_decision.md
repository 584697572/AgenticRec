# T03 路线决定：B 方法级重建

2026-10-01；T03 资源调查及路线决定 **DONE**。原资源运行 A1 **BLOCKED**；A2/U0 `not_run`；B **NOT EVALUATED**。这不是原系统运行验收。

## 原要求与实测

按规范先调查 A1，核验资源共享 ID 契约。README 的 GoogleDrive 原包可访问；正常提交公开下载确认表单后，服务支持 HTTP 206 Range。原 ZIP 共 497,403,134 bytes、15 个条目。两次有上限的读取累计收到 471,747 bytes（约 461 KiB，含确认页和重复目录），完整 ZIP 未下载。目录清单及响应记录见 `reproduction/logs/t03_20261001/`，提取子集在忽略目录 `data/raw/upstream_audit/`。

| 资源 | 真实核验 | 限制 |
|---|---|---|
| movie/settings.json、columns.json | 完整成员 CRC、SHA256；所需键和指向成员存在 | 未安装到 upstream/resources |
| movies.ftr | 完整 CRC、SHA256；9,888 行，ID 唯一连续 1..9888；无 padding 行，各列无 null；257 个重复标题 | Feather，不是 CSV；无电影年份/价格补造 |
| movie_sim.npy | 仅头部：float64、shape=(9889,9889)，包含 ID 0 的位置；按头部计算大小与目录一致 | 矩阵值、完整 CRC、行 ID 语义 NOT VERIFIED |
| SASRec-SASRec.pth | 仅 64 KiB 前缀；静态解析完整 metadata pickle 的指令，配置 n_items=36255、embedding shape=(36255,32)、dataset=ml-10m | 未反序列化、未加载权重；完整 CRC、训练来源和 UniRec 兼容性 NOT VERIFIED |

**目录/矩阵 9,889 个位置与 checkpoint 36,255 行不一致，原包没有映射文件。** 更多 embedding 行本身不能证明每个分数错误，但不足以验证同一 ID 指向同一电影；前提契约无法验收。`validate_resource_subset.py --require-alignment` 实际退出 1。不得裁剪 embedding、猜测映射或用补造数据掩盖。

另执行原 CandidateBuffer.__init__，以真实目录长度验证：候选仅 9,887 项，漏掉合法 ID 9888。原方法未修改；修复留在规范的后续回归阶段。

## 决定与可比性

采用规范 B 路线，后续完成 T05 再推进 T06 的 MovieLens 1M 数据契约；保留固定 upstream 原始代码，重建 U1 upstream_rebuilt 的方法与工具对照。停止下载原包约 385.6 MB 的压缩矩阵和约 13.8 MB 的压缩 checkpoint：现有证据已足以确定当前无法核验资源对齐，完整下载不能补出缺失映射。重大决定见 ADR-004。

不能宣称复现论文数字，也不能把未来 U1 当 U0。恢复 A1 须取得可信共同 ID 映射、完整资源校验和 legacy 运行证据，本轮不尝试推测修复。

## 授权与记录

原包清单无 LICENSE/README/terms；预制资源及 checkpoint 的独立许可、训练来源 **NOT VERIFIED**。仅本地检查子集，不提交或重新分发。代码 MIT 不等于数据 MIT。官方 [MovieLens 10M 条款](https://files.grouplens.org/datasets/movielens/ml-10m-README.html)要求研究使用注明来源，再分发另获许可，商业使用事先取得许可；不能据此推定整个上游预制包获得授权。B 路线将单独按 [MovieLens 1M 条款](https://files.grouplens.org/datasets/movielens/ml-1m-README.txt)记录数据来源；本轮未下载 MovieLens。

原 Google/RecDrive 页面与 Issue #110 调查保留为上一轮历史，不将第三方失效报告当作本轮本机结论。实际 blocker 是未验证的共享 ID 契约及资源来源，不是 Google 链接失效。

本轮没有依赖安装、模型训练、真实 LLM、付费 API 或指标实验。`allow_paid_api=false`、`api_request_cap=0`，运行指标全部 null / NOT EVALUATED。
