# 工作区状态入口

最后更新：2026-10-05（Asia/Shanghai）。详细状态见 [AgenticRec/docs/STATUS.md](AgenticRec/docs/STATUS.md)，本轮证据见 [阶段报告](reports/progress_20261005.md)。

- DONE：T00、T01、T02、T05。现代 Python 3.11 开发环境及离线重建已验收，两个环境各 15 项基础测试通过。
- T03 IN_PROGRESS：原资源可运行，但 checkpoint/矩阵的权威 ID 语义及预制包独立许可仍 NOT VERIFIED。真实 ReDial raw test 已取得，按原 notebook 加 dtype 适配生成 45 条 A1 输入；作者 canonical 子集等价性 NOT VERIFIED。
- T04 BLOCKED：原工具、原 app mock 和真实连接已通过。新 App 单轮额度已授权并配置，执行端不持有用户 Key；在用户已有 Key 的终端执行 reproduction/scripts/live_app_single_turn.py。真实单轮/多轮 NOT VERIFIED，原评测 NOT EVALUATED。
- T06—T24 TODO，按 TASKS.yaml 的依赖推进；训练与算法收益 NOT EVALUATED。

保持 A1 路线，未转 B。原规范和固定 upstream 未改；旧报告/证据继续保留历史快照身份。不要重复运行已用完额度的连接测试或删除额度记录。
