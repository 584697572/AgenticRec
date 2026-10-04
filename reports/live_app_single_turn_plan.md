# 下一步：原 app 单轮真实运行（尚未执行）

2026-10-05 状态更新：用户已允许请求和输出，采用下方拟议的 2 次尝试/512 输出 token/1 CNY 额度，已实现并离线验证。运行命令 `& .\reproduction\.venv-legacy\Scripts\python.exe .\reproduction\scripts\live_app_single_turn.py`；在用户 Key 所在终端执行。此次受控 CLI 调用原 app 回调，不启动 UI 服务。真实单轮仍 NOT VERIFIED；下文“待授权”仅指 2026-10-04 历史计划。

2026-10-04。原wrapper真实连接已通过；严格按规范§5.3进入原app单轮，然后才是多轮和原评测。本次1/1连接额度已消耗，以下是待授权的具体范围，当前真实单轮NOT VERIFIED。

- 使用已有原电影目录/矩阵/SASRec/gte-base和legacy环境，不下载新资源、不改原源码、算法或prompt模板。
- 运行独立兼容副本的原app.py，movie、chat、plan_first=1、langchain=0、demo_mode=zero、reflection=0、shorten=0、reply_style=concise。模型沿用deepseek-flash非思考模式。
- 自动只提交一句：`Recommend three comedy movies released since 1990.`，不开放随意发送请求的UI；测试完成关闭本地服务。
- 拟议新额度：**最多2次HTTP尝试，每次输出最多512 token，新增预算1 CNY，不自动重试**。原源码与已保存mock轨迹显示，正常工具成功路径有一次规划和一次最终摘要；直接答复或工具失败可能少于2次，任何额外调用必须由测试层拒绝。
- 输出token上限须在调用边界统一控制：原plan_and_exe与_summarize_recommendation均调用OpenAICall.call，不能只改app CLI参数就宣称所有调用受限。该限制作为独立测试层公开，不覆盖原baseline源码。
- 记录原模型生成的真实计划、真实工具执行轨迹、实际推荐标题/数量与目录类型/年份约束、真实SDK usage、每次HTTP状态及失败；不注入手写计划或伪造摘要，不把LLM声称“符合”当工具约束校验。
- 验收需确认原app入口、单轮回调、原工具链、输出和目录约束；仅直接LLM答复、未调用工具时不能算推荐工具链成功。失败保留，不以增加额度或重复调用掩盖。
- 2次上限由单轮测试层持久记录控制，成功/失败/超时均不重置。本次授权不含第二轮、原eval、批量benchmark，也不代表供应商账单硬限额；实际扣费未取得时仍为null。

当前没有发起本计划的真实请求，也未生成单轮运行结果。授权后实施受限单轮测试层并先离线验证，再指导用户在已有Key的同一终端执行。原共享ID语义和原评测数据问题独立保留，单轮通过也不能标T04 DONE。
