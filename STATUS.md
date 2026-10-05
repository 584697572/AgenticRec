# 工作区状态入口

最后更新：2026-10-05（Asia/Shanghai）。详细状态见 [AgenticRec/docs/STATUS.md](AgenticRec/docs/STATUS.md)，最新证据见[真实多轮与原评测报告](reports/a1_multiturn_eval_20261005.md)。

最新检查与修复见[修复报告](reports/reproduction_repairs_20261005.md)：两项可选 `upstream_compat` 行为补丁修正同名电影年份选择和反思丢失外部历史；53 项全量回归通过。真实上游 fixed demo/Critic/反思在 fixture 与 SDK MockTransport 上通过，未作新的真实调用或下载。45 条原输入仅 22 条满足唯一标题/年份目录契约；新增预检拒绝不满足契约的整批新实验，不删除样本。原版输入、预测及三个 0/45 指标原样保留，T03/T04 状态不变。

项目仓库：[584697572/AgenticRec](https://github.com/584697572/AgenticRec)，public；根项目 `origin` 指向该仓库，`master` 跟踪 `origin/master`。创建与首次推送已核验，见 [发布报告](reports/public_repository_20261005.md)。上游 `RecAI/` 保留独立远程与固定版本。

- DONE：T00、T01、T02、T05。现代 Python 3.11 开发环境及离线重建已验收，两个环境各 15 项基础测试通过。
- T03 IN_PROGRESS：原资源可运行，但 checkpoint/矩阵的权威 ID 语义及预制包独立许可仍 NOT VERIFIED。真实 ReDial raw test 已取得，按原 notebook 加 dtype 适配生成 45 条 A1 输入；作者 canonical 子集等价性 NOT VERIFIED。
- T04 BLOCKED（其余验收前置）：真实单轮及两轮条件修改 VERIFIED；45 条 A1 派生输入的原 recbot/random/加权 popularity 评测链已完成，正式原模糊 hit 均为 0/45。新会话实际93次HTTP200、201944 token，高峰费率估算0.507208 CNY；账单未查询。canonical 子集、权威 ID/许可及原 demo/reflection 条件仍未验收。
- T06—T24 TODO，按 TASKS.yaml 的依赖推进；训练与算法收益 NOT EVALUATED。

保持 A1 路线，未转 B。原规范和固定 upstream 未改；旧报告/证据继续保留历史快照身份。不要重复运行已用完额度的连接测试或删除额度记录。

本轮40项全量离线回归及3项原指标测试通过；原指标复算、重复执行门禁、Key扫描和22个旧证据字节检查通过。21个target无法映射目录、4个映射年份不同、索引37规划截断且无Map，全部保留在45个分母中。去掉年份的6个诊断模糊命中不替代正式零结果。

Key仍按环境变量优先、根目录`.env`次之；不进入Git或证据哈希清单。旧单轮2/2额度保持原样；新额度已用93/120，现有会话已关闭，不能删除记录重跑。下一步继续T03来源/语义核验和T04原配置恢复，再按T06/T07前置推进模型。
