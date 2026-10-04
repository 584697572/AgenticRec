# 工作区状态入口

最后更新：2026-10-05（Asia/Shanghai）。详细状态见 [AgenticRec/docs/STATUS.md](AgenticRec/docs/STATUS.md)，最新证据见 [真实单轮报告](reports/live_original_app_20261005.md)和[Key配置报告](reports/local_key_setup_20261005.md)。

- DONE：T00、T01、T02、T05。现代 Python 3.11 开发环境及离线重建已验收，两个环境各 15 项基础测试通过。
- T03 IN_PROGRESS：原资源可运行，但 checkpoint/矩阵的权威 ID 语义及预制包独立许可仍 NOT VERIFIED。真实 ReDial raw test 已取得，按原 notebook 加 dtype 适配生成 45 条 A1 输入；作者 canonical 子集等价性 NOT VERIFIED。
- T04 BLOCKED：原工具、原 app mock 和真实连接已通过。本机 .env 已配置；执行端完成真实单轮 SUCCESS/VERIFIED，2次HTTP200、3880 token。多轮 NOT VERIFIED，原评测 NOT EVALUATED。
- T06—T24 TODO，按 TASKS.yaml 的依赖推进；训练与算法收益 NOT EVALUATED。

保持 A1 路线，未转 B。原规范和固定 upstream 未改；旧报告/证据继续保留历史快照身份。不要重复运行已用完额度的连接测试或删除额度记录。

2026-10-05 Key 配置更新：环境变量优先，其次根目录 `.env`；9 项新测试、全 32 项离线回归通过，Git 忽略已核验。详见 [配置报告](reports/local_key_setup_20261005.md)。

真实单轮已验证，详见 [真实 App 报告](reports/live_original_app_20261005.md)。2/2单轮额度已使用，重复执行门禁通过，未改写原证据。
