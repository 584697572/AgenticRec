# 工作区状态入口

唯一详细状态见 [AgenticRec/docs/STATUS.md](AgenticRec/docs/STATUS.md)。

T00/T01/T02 DONE；T03 IN_PROGRESS；T04 BLOCKED。保留A1原资源兼容功能路线。
原资源、真实SASRec权重、完整工具链以及原app.py的本地启动/两轮mock-LLM回调已验证通过。
真实LLM对话NOT VERIFIED，原评测NOT EVALUATED；缺provider/model/预算授权、原eval数据及权威共享ID语义证据。
规范和固定upstream均未改；仅独立运行副本的requirements有兼容patch。T05—T24未启动。
最新报告见[reproduction/UPSTREAM_REPRODUCTION.md](reproduction/UPSTREAM_REPRODUCTION.md)。

2026-10-02开始真实LLM接入指导，当前步骤为本机安全配置与零请求preflight；见[LIVE_LLM_GUIDE.md](reproduction/LIVE_LLM_GUIDE.md)。服务/model及预算等待用户答复，真实连接仍NOT VERIFIED。
