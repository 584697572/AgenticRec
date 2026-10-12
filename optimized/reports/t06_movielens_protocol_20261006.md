# T06｜官方 MovieLens1M 数据与训练期 ID 协议（2026-10-06）

本报告只记录真实数据准备与合同验证。规范 §6.3 的时间窗、正反馈、warm 用户与候选全集在训练前冻结；后续模型共享这份协议。valid NDCG@10 用于模型选择，T07 尚需验证实现。不存在可报告的算法提升。

## Completed

- 仅从 [GroupLens 官方稳定 MovieLens1M](https://grouplens.org/datasets/movielens/1m/) 取得 ZIP、README 与校验文件；官方 MD5、全部 ZIP CRC 与 SHA256 核对通过。许可允许符合条款的研究使用，未经单独许可不得再分发数据，故原行/处理数据只在 Git 忽略目录。
- 检查 3883 个电影、6040 个匿名用户和 1000209 条评分的字段/引用。电影标题按实际可解码的 ISO-8859-1 处理，评分与用户文件按 ASCII；未使用性别、邮编等人口属性，也未生成剧情、片长等缺失字段。
- 按全局 timestamp 约80/10/10划分；同时间戳整体进入后一个窗口。rating≥4为正反馈，训练期至少5个正反馈的用户进入 warm 训练；候选电影为训练期至少一次正反馈的电影。全训练期的低评分仍记为已观察，不会作为未知负样本。
- 冻结 Parquet、训练期双向 ID 映射、时间/过滤/覆盖清单及 SHA。验证和测试的每条交互保留，记录 warm/cold 用户与电影分层；训练图、候选、热门统计和 ID 不读取 holdout。旧资源继续独立保存。

## Files Created / Modified

新增 `AgenticRec/src/agenticrec/data.py`、`AgenticRec/configs/data.yaml`、2 个合同/防泄漏测试文件；更新 `AgenticRec/pyproject.toml` 与开发依赖锁，增加 `pyarrow==21.0.0`。本地忽略目录新增 `data/raw/ml-1m/`、`data/processed/ml-1m-v1/`、`artifacts/data_manifest.json`；公开仅保留汇总哈希于 `reproduction/native_runs/20261006_data/`。同步更新 STATUS、TASKS、README、DECISIONS。

## Commands / Tests

```powershell
& AgenticRec/.venv-dev/Scripts/python.exe -m agenticrec.data --config AgenticRec/configs/data.yaml
& AgenticRec/.venv-dev/Scripts/python.exe -m pytest AgenticRec/tests/unit/test_data_contract.py AgenticRec/tests/regression/test_no_leakage.py -q
& AgenticRec/.venv-dev/Scripts/python.exe -m pytest AgenticRec/tests -q
& reproduction/.runtime/uv.exe pip check --python AgenticRec/.venv-dev/Scripts/python.exe
```

第二隔离环境用已缓存依赖离线安装 `requirements.dev.lock.txt` 与项目包，运行同一数据合同/防泄漏测试和 `uv pip check`。原 ZIP 仅首次取得，重复数据准备拒绝覆盖已冻结产物。

## Verification Results

| 项目 | 实测 |
| --- | ---: |
| 官方 ZIP 大小 / MD5 | 5917549 字节 / `c4d9eecfca2ab87c1945afe126590906` |
| 原始评分 / 电影 / 用户 | 1000209 / 3883 / 6040 |
| train / valid / test 评分 | 800164 / 100024 / 100021 |
| warm 用户 / 训练候选电影 | 5351 / 3469 |
| 训练正反馈图边 | 462887 |
| valid 正反馈：warm / 冷启动 | 12197 / 45615 |
| test 正反馈：warm / 冷启动 | 41499 / 12953 |

训练末时间为 `975768719`，验证起始 `975768738`、验证末 `978133367`、测试起始 `978133414`（Unix 秒），满足严格的跨窗口先后关系。详细分层、排除、种子、原文件及全部十个处理文件的 SHA 见[公开数据清单](../reproduction/native_runs/20261006_data/data_manifest.json)；本地原行不在 Git。两个环境的数据合同与防泄漏测试各5项通过；完整新包20项测试通过；依赖检查通过。尚无 Recall/NDCG/模型训练结果，标记 **NOT EVALUATED**。

## Findings

全局时间窗造成 valid 正反馈中的冷启动占比较高，不能只报告 warm 指标而隐去覆盖率。主 warm 排序与 cold 层应分别报告；所有模型共用训练候选全集、已评分过滤、相同时间边界及并列处理规则。低于4星是明确已评分，不等于未知；训练负采样不得偷看未来评分。

首次本地 MD5 校验脚本误把官方 `MD5 (ml-1m.zip) = ...` 的首字段当作哈希而退出；下载文件未损坏。按官方格式重新解析，MD5 与所有 ZIP CRC 均通过，没有再次下载 ZIP。

## Blockers

T06 无新 blocker。旧权重/矩阵语义与原包独立许可仍 NOT VERIFIED，T04 完整原版验收继续 BLOCKED。T07 指标、T08/T09 模型和任何效果提升尚未执行。

## Deviations from Spec

None。本轮使用规范指定稳定 MovieLens1M、全局时间窗及训练期映射。独立质量路线 B 是先前按规范 §5.5 和用户授权记录的 ADR-015；不以旧 A1 的0/45文本命中衡量新模型提升。

## Current Project State

T00/T01/T02/T03/T05/T06 DONE；T03 是资源审计/路线验收，T04 原完整复现 BLOCKED；T07—T24 TODO。新模型收益 **NOT EVALUATED**。本轮新增 LLM 请求 0。

## Next Tasks

T07 按规范先用可手算 fixture 验证 Recall@K、NDCG@K 等固定指标和空结果分母，再用 valid NDCG@10 进行模型选择；T08 在同一数据协议上跑 Random、Popularity、BPR-MF，T09 跑 LightGCN。测试集只用于冻结配置后的最终报告，不因结果不佳更换主指标。
