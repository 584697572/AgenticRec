# T19 Provider 与文本解析阶段证据

状态：`IMPLEMENTED / FIXTURE VERIFIED / LIVE NOT VERIFIED`

本批为 T19 可恢复 runner 补齐 modern DeepSeek/OpenAI-compatible transport 与 F 自由文本解析入口。没有运行正式 benchmark，没有填写系统指标，也没有消耗远程请求。

## 实现边界

- `OpenAICompatibleTransport` 使用 Python 标准库 HTTPS，不新增依赖；只由 allowlist 工厂映射到 `https://api.deepseek.com`。
- 请求固定 `response_format={"type":"json_object"}`、`sdk_max_retries=0`；T13 `ChatAdapter` 继续是唯一重试控制者。
- 禁止非 HTTPS、URL 凭据、query/fragment 与自动重定向；响应最大 2 MiB。
- provider usage 进入统一账本；供应商账单未查询时费用是 `null`，原因为 `provider_bill_not_queried`。
- Key 按 `OPENAI_API_KEY` 环境变量优先、根目录 `.env` 次之读取；不扩展其他变量、不修改环境、模块导入时不读文件、错误不回显内容。
- F 文本解析返回完整且精确的 `FixedRequest`；公开 `user_id` 和 `history_authorized` 不可被模型修改，消息以 JSON envelope 作为数据。
- A/O 保留首次 planner 调用内解释并规划的冻结口径，避免额外 parser 调用改变已预注册的8,649次上界。结构化 F 仍为0 LLM 调用。

## 验证

- 两个 modern 环境的 provider/解析专项：各 `35 passed`。
- 两个 modern 环境的完整测试：各 `197 passed, 2 warnings`；warning 为既有 PyTorch optional NumPy/TypedStorage 提示。
- 两个环境 `compileall`：exit 0。
- `git diff --check`：通过。
- 依赖下载：0。
- 远程 API 请求：0。

## 未完成

- modern transport 的真实 DeepSeek 网络连接：`NOT VERIFIED`。
- DeepSeek 对冻结文本 episode 的抽取准确率：`NOT EVALUATED`。
- U1/F/A/O 正式 system executor 到 runner 的连接：未完成。
- T19 系统与消融指标：`NOT EVALUATED`。

下一步实现正式 executor，在离线 fixture 中核对每个系统的实际 request/tool/turn 计数和 `EpisodeAttempt` 转换；覆盖整批的额度与金额授权满足前不得启动 live batch。
