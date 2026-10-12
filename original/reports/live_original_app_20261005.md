# 原 App 真实 DeepSeek 单轮验证（2026-10-05）

## Completed

用户将 Key 保存到本机 `.env` 后，执行端直接运行既定命令。固定 InteRecAgent 的原 app 入口、原回调及原 SQL/Ranking/Map 工具完成真实单轮，返回 SUCCESS / VERIFIED。不是 mock 或原 wrapper 的简单 OK 连接检查。

## Files Created / Modified

真实原始证据保存在忽略目录 `reproduction/logs/live_app_single_turn_20261005/`，含请求/响应、实际 usage、工具轨迹、原 worker 日志、参数与原资源哈希。公开摘要见 `reproduction/native_runs/20261005_live/original_app_live_summary.json`；本轮索引见 `reproduction/local_key_evidence_20261005.json`。`.env` 不进入 Git、公开日志或哈希清单。执行源码 commit `240f53d`，原 upstream 未改。

## Commands / Tests

执行端实际运行：

```powershell
& .\reproduction\.venv-legacy\Scripts\python.exe .\reproduction\scripts\live_app_single_turn.py
```

此前 32 项离线测试通过。真实结果核验：两次 HTTP 200、原工具三步执行、目录约束通过、usage 自洽、没有实际 Key 混入运行证据。再次调用的门禁用禁止资源/HTTP 初始化的 mock 核验，返回 BLOCKED、新增请求0、原结果/额度记录逐字节不变。

## Verification Results

| 项目 | 实际结果 |
|---|---|
| 请求/响应模型 | deepseek-flash / deepseek-flash，两次一致 |
| 请求设置 | thinking disabled；最多512输出token/次；SDK/HTTP无重试 |
| 真实请求 | 2/2，HTTP状态200、200，额度已消耗 |
| 实际usage | 输入3655、输出225、总计3880 token；usage_is_fixture=false |
| 单轮原回调耗时 | 2.8178234秒；一次功能冒烟，不是benchmark |
| 含初始化总耗时 | 304.3412709秒；原模型/目录本地加载和编码 |
| 实际账单 | null，未查询供应商账单 |
| 高峰费率估算 | 0.00911 CNY；不是实际扣费 |
| 目录/输出校验 | 3个不同有效ID、Comedy、年份≥1990、回答含映射标题、Filter/Ranking/Map执行，全部通过 |
| UI | ui_served=false；受控CLI驱动原app回调 |
| 完整A1/A2、原评测 | 尚未完成 / NOT EVALUATED |

实际输出为 Forrest Gump（1994）、Babe（1995）、Pretty Woman（1990），对应当前原目录 ID 7/78/84。条件是“since 1990”，包含1990年；不能改写成严格大于1990。估算按当日已核对的 [DeepSeek人民币高峰费率](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/)：3655×2/1e6 + 225×8/1e6 = 0.00911；缓存折扣及真实结算未查询。

## Findings

`.env` 自动读取解决了不同 PowerShell 进程不共享 Key 的问题。真实 LLM 生成计划和摘要，原工具实际返回目录项；没有注入手写计划或以 LLM 自评替代目录检查。目录校验只覆盖这一固定场景，不证明任意输出无幻觉或新模型带来质量提升。

## Blockers

Key 配置/命令代执行/真实单轮不再阻塞。多轮、原评测与 demo/reflection 原设置恢复未验收；原 checkpoint 权威标题映射、矩阵行标签语义、预制包独立许可及 canonical 作者评测子集等限制仍存在。当前单轮2/2额度已消耗，不删除或复用其记录。

## Deviations from Spec

继续 A1。DeepSeek非思考、CPU、zero-demo、关闭reflection/shortening、原CLI回调及API边界限额的既有差异均保留并公开；不声称A2论文设置。规范、upstream源码、原推荐算法和prompt模板未改；`.env`持久化由用户明确选择。

## Current Project State

原 App 真实单轮 VERIFIED；T04整体仍 BLOCKED，不能把一个场景成功当完整复现。T00/T01/T02/T05 DONE，T03 IN_PROGRESS，T06—T24 TODO。所有推荐算法收益与benchmark指标 NOT EVALUATED。

## Next Tasks

按 §5.3 准备并验收有独立计数的真实多轮，再原评测；继续资源语义/来源核验。T03闭合后按T06数据→T07指标→T08 BPR-MF→T09 LightGCN推进。命令执行由执行端负责，Key保持本机持久配置。
