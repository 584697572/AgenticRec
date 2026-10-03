# 工作区状态入口

唯一详细状态见 [AgenticRec/docs/STATUS.md](AgenticRec/docs/STATUS.md)。

T00/T01/T02 DONE；T03 IN_PROGRESS；T04 BLOCKED。保留A1原资源兼容功能路线。
原资源、真实SASRec权重、完整工具链以及原app.py的本地启动/两轮mock-LLM回调已验证通过。
真实LLM单次连接VERIFIED（DeepSeek HTTP200，13 token）；原app真实对话NOT VERIFIED，原评测NOT EVALUATED。后续调用额度、原eval数据及权威共享ID语义证据仍待处理。
规范和固定upstream均未改；仅独立运行副本的requirements有兼容patch。T05—T24未启动。
最新报告见[reproduction/UPSTREAM_REPRODUCTION.md](reproduction/UPSTREAM_REPRODUCTION.md)。

2026-10-02开始真实LLM接入指导，当前步骤为本机安全配置与零请求preflight；见[LIVE_LLM_GUIDE.md](reproduction/LIVE_LLM_GUIDE.md)。用户已确定DeepSeek；官方地址和建议模型deepseek-flash已填入禁用模板。Key由用户本机输入，请求/token/金额预算仍待授权，真实连接NOT VERIFIED。最新接入报告见[DeepSeek配置报告](reports/deepseek_setup_20261002.md)。

2026-10-04更新：用户在自己的终端完成受限原wrapper真实连接，SUCCESS、HTTP200、reply=OK，prompt/completion/total=12/1/13，单次耗时2.1585443秒（非benchmark）。本地原始结果/额度记录已核对，已消耗唯一授权尝试；16项离线测试通过。实际扣费未查询，费用null，估算高峰上界0.000032 CNY。原app/原评测不在这一次授权内。报告见[真实连接验证](reports/live_connection_20261004.md)。
