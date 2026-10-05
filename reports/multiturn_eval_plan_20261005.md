# T04 多轮与原评测执行范围

2026-10-05：用户明确选择新增 120 次请求、每次最多 1024 输出 token、总预算 3 CNY；DeepSeek deepseek-flash、thinking disabled、零重试。授权记录和启用 profile 仅保存在本机忽略目录，公开 example 仍默认禁用。

1. 校验原资源、兼容副本 Python 源码和冻结评测输入 SHA。
2. 无真实请求的完整联调：两轮原 App 回调，原 recbot、random 和 popularity 三分支各处理全部 45 条输入。
3. 联调验收后记录源文件/提交并执行相同真实命令。两轮场景为先推荐 1990 年及之后的 3 部喜剧，再改为 1980 年之前的 3 部喜剧；原 memory 需传递此前输入和回答。
4. 原 recbot 评测固定 45 条全部入分母；random 为原无放回抽样，popularity 为原按 visited_num 加权抽样，均单 seed 42。指标保留原 hit_judge 严格 `partial_ratio > 80` 和原 evaluator，不写成 ID Hit@K/NDCG。
5. 核验原保存的 conversations、原函数指标、逐样本命中与失败、HTTP 响应/usage；公开摘要不含真实对话、target、私密 trace 或 Key。

本轮不更改原算法、prompt、数据或 eval 源码。复用经过原构造函数初始化的同一 Gallery，只为避免重复编码 9,888 个标题；原评测创建独立 bot 和 CandidateBuffer，不沿用多轮会话。模型加载进程不注入真实 Key，父进程经 stdio 转发 SDK 请求。

原作者 canonical 子集、checkpoint 权威标题对应、矩阵行语义和原包独立许可仍 NOT VERIFIED。45 条输入是固定原 notebook 派生的 A1 输入，不能宣称论文 A2。zero-demo、reflection/shortening 关闭；恢复原实验设置另作独立验证，当前未验收。

门禁先保存失败后实现；当前全 40 项离线回归通过，证据为 `reproduction/logs/continuation_20261005/session_tests_after.*`。端到端联调与真实多轮/原评测在本文建立时仍 NOT VERIFIED / NOT EVALUATED，完成后另附结果报告。旧单轮与连接额度及 22 个历史证据文件已建立字节快照，不清空或复用。

本轮源码入口：`reproduction/scripts/live_reproduction.py`。真实执行需要已授权的本机 profile；全新 checkout 缺 profile 会以 0 请求拒绝。调用发送前独占、fsync 记录请求槽，失败/超时占额度，未知 usage 或未解决的旧请求阻止继续发送。金额使用已核验高峰输入 2 / 输出 8 CNY 每百万 token 的保守估算及逐请求规划检查，真实账单未查询，不宣称供应商账单硬限额。
