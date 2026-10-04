# 2026-10-05：已授权的原 App 单轮

## 保存一次 Key，交由执行端运行

用户已明确选择项目根目录 `.env` 持久保存。文件位置是 `D:\AgenticRec\.env`，已创建空字段，Git 忽略已核验。用本机编辑器填入并保存：

```dotenv
OPENAI_API_KEY=你的DeepSeek实际Key
```

这里使用 `OPENAI_API_KEY` 是为了兼容原接口，填写的是 DeepSeek Key。无需添加 `Bearer`；不要将真实值贴到聊天。`.env` 是本机明文文件，Git 忽略不是加密。

preflight、connection、原 App 单轮三个入口现已支持：非空进程环境变量优先，否则读取固定项目根目录 `.env`。支持 UTF-8/BOM、单双引号、注释；只读取这一个字段，不执行插值/命令，不从文件修改 provider、API 地址或预算。SDK JSON/profile 仍不含 Key。加载器无导入副作用，worker 与 `--offline` 路径不主动读取 `.env`；这不是 OS 文件访问沙箱。

保存后通知执行端即可，由执行端运行和核验，不再要求用户手动执行测试命令。已授权的当前单轮仍为 2 次 HTTP 尝试、512 输出 token/次、1 CNY；保存 Key 不会清空已用额度。缺失、语法错误或重复 Key 赋值以 0 请求拒绝，并只输出不含内容的错误。

本次 9 项新增离线测试、全 32 项通过；用户已保存 Key，零请求检查 CONFIG_READY，真实 App 已由执行端完成：SUCCESS/VERIFIED、2次HTTP200、3880 token，原工具与目录校验全部通过。详情见 [本机 Key 配置报告](../reports/local_key_setup_20261005.md)。以下命令保留作可复现入口，执行责任已由用户交给执行端。

连接测试已通过，无需重复。用户已允许继续请求；本轮采用此前提出的具体额度：DeepSeek `deepseek-flash`、非思考、最多 2 次 HTTP 尝试、每次最多 512 输出 token、1 CNY，无重试。当前工作区的专用授权 profile 已写入忽略目录，不包含 Key。

可复现入口如下；执行端现在直接读取已保存的 `.env` 并负责运行：

```powershell
& .\reproduction\.venv-legacy\Scripts\python.exe .\reproduction\scripts\live_app_single_turn.py
```

本次命令已成功执行，2/2额度已使用，不要重复运行或删除额度记录。详情见 [真实单轮报告](../reports/live_original_app_20261005.md)。

运行过程先校验原资源 SHA 和 Python 源码，再初始化原 app，CPU 本地编码约需 5 分钟。原 app 的过滤/排序/映射及回调实际运行；此次以受控 CLI 调用，不开放交互 UI。原算法和 prompt 模板不改，测试层仅限制 API 和观察 Map ID。加载资源的子进程只有占位 Key，真实 Key 只在负责 HTTPS 的父进程内存中。

末尾 `SUCCESS` 还要求真实目录验证通过：3 个不同 ID、Comedy、年份≥1990、原工具实际执行、回答包含映射标题。它只验收这一个场景，不代表多轮、严格文本无幻觉或原评测通过。返回 `BLOCKED` / `FAILED` 时保留完整目录，把末尾 JSON 发回即可；不要删除额度文件或盲目重试。失败和超时也占用尝试。

专用证据在 `reproduction/logs/live_app_single_turn_20261005/`。计划/摘要原始响应、实际 usage、工具 trace 和错误均保存在忽略目录，终端仅显示可分享摘要。账单未查询则 actual_cost_cny=null；客户端未实现供应商账单硬限额。全新 checkout 没有本机授权文件时会以 0 请求拒绝，不会自动开启付费。

2026-10-05 离线联调 SUCCESS 后已完成真实单轮 SUCCESS。离线证据仍单独保留，不能混用usage或耗时。以下保留早期配置指导，当前步骤以上述单轮命令为准。

## 历史配置指导（以下授权状态为当时快照）

# 真实LLM运行：第一步，本机配置检查

2026-10-04最新状态：单次真实连接VERIFIED，已消耗1/1授权尝试；原app真实对话NOT VERIFIED，原评测NOT EVALUATED。2026-10-02配置准备说明保留在后文。
下一步遵循规范§5.3：单次连接→原app单轮→多轮；原评测另有数据与资源语义前置。

## 2026-10-04：已授权首次 DeepSeek 连接

执行结果已确认：用户在自己的终端运行成功，HTTP200、OK、usage=12输入+1输出=13，elapsed=2.1585443秒（非benchmark）；结果及额度记录已核对并保留到native_runs/20261004。该连接命令已经执行，下面为复现说明，**不要再次调用或删除额度记录**。原app下一步范围见[单轮计划](../reports/live_app_single_turn_plan.md)；本次授权不覆盖它。

用户已明确授权：一次连接、输出最多128 token、1元人民币。已在忽略目录准备 `reproduction/.runtime/live_llm.json` 和授权记录；通用/DeepSeek example 仍默认禁用，不把模板改成付费默认值。本次授权只适用于连接检查，不授权启动原UI或批量评测。

在用户已注入Key的同一个PowerShell窗口，从工作区根目录运行，无需激活环境、安装依赖或下载资源：

```powershell
& .\reproduction\.venv-legacy\Scripts\python.exe .\reproduction\scripts\live_llm_connection.py
```

脚本先检查本地profile和当前进程Key存在性，直接加载固定upstream的原 `llm4crs/utils/open_ai.py`，校验SHA256；不导入原app/图库。通过测试层的SDK工厂配置 `max_retries=0`、原wrapper的 `retry_limits=1`、HTTP transport的 `retries=0`，关闭重定向，只允许一次对官方 `/chat/completions` 的固定短提示POST。按[DeepSeek Chat Completions接口](https://api-docs.deepseek.com/api/create-chat-completion/)明确追加 `thinking={"type":"disabled"}`，保留原wrapper构造消息和响应解析的行为；原文件未修改。这是A1兼容服务连接，不是原模型/论文设置复现。

在HTTP transport发送前用独占创建和fsync预留 `reproduction/.runtime/live_llm_connection_20261004.request.json`，跨进程共用这份记录。成功、失败、DNS/TLS错误、超时都不自动返还请求额度；再次执行将BLOCKED，保留旧证据。不要删除记录来重试。脚本没有额外model-list、余额或计费API调用。

结果写入忽略目录 `reproduction/logs/live_llm_20261004/connection_result.json`。用户可将终端JSON反馈给Codex，不含Key/请求headers/供应商错误原文。原wrapper输出的traceback在测试层抑制，保留安全的error_type和HTTP状态；这不是删除原始实验失败，受限请求的失败同样写入本地结果且保留额度。

- `SUCCESS`：原wrapper取得非空响应，实际HTTP200和合法usage已校验。仅代表连接路径通过；原app仍NOT VERIFIED。
- `FAILED`：一次尝试已消耗，不重试；贴回结果检查。没有HTTP响应时无法确认服务端是否收到请求，remote_api_requests为null，而http_attempts仍为1。
- `BLOCKED`：本次没有新HTTP尝试；看reason。若是configuration_or_key_missing，在同一窗口重新注入Key；若是authorized_attempt_already_reserved，不要再次调用或重置额度。

官方[人民币价格](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/)在2026-10-04核对：Flash高峰百万token缓存未命中输入2元、输出8元。用1024输入token的保守规划额度和128输出上限估算0.003072元；1024是规划假设，**不是实际输入测量或tokenizer证明的硬上限**。实际usage返回后另算保守高峰费用上界；供应商账单未查询时actual_cost_cny为null，不冒称0元或精确扣费。请求和输出上限由测试层约束，供应商账单硬限额未实现；1元仅授权本次固定短提示检查。未下载DeepSeek tokenizer或额外模型。

配置预算后，示例profile仍会BLOCKED；后续零请求检查使用默认本地profile：`python reproduction/scripts/live_llm_preflight.py`。CONFIG_READY不等于连接成功；真实结果由上述受限脚本产生。

## 先提供不含密钥的信息

### 已确定使用 DeepSeek（2026-10-02）

用户已确定厂商为 DeepSeek。无需自行查找技术参数：官方 OpenAI 兼容地址为 `https://api.deepseek.com`，本次建议模型为 `deepseek-flash`。这与项目 `api_type=open_ai` 兼容；这里的变量名 OPENAI_API_KEY 可以装载 DeepSeek 的 Key，并不要求购买 OpenAI 的服务。参数依据：[DeepSeek 官方入门](https://api-docs.deepseek.com/)与[模型/价格](https://api-docs.deepseek.com/quick_start/pricing/)。服务模型用于 A1 兼容功能验证，不代表论文原模型/设置的 A2 复现。

已准备 `reproduction/live_llm.deepseek.example.json`；模型、地址和币种已填好，保持调用禁用、请求限额为 0。首次建议只授权 **1 次 HTTP 尝试、最多输出 128 token、金额预算 1 元人民币**，不自动重试。该金额是拟议预算，尚未获得用户授权，也不是供应商账单硬限额；真实费用需按返回 usage 和当时价格核算，不能填假费用。

首先在下节同一 PowerShell 终端安全输入 Key，再执行以下零请求检查（无需安装或下载）：

```powershell
python reproduction/scripts/live_llm_preflight.py --profile reproduction/live_llm.deepseek.example.json
```

此时预期 `key_present=true`、`api_requests=0`、`status=BLOCKED`，缺项为付费授权、请求限额及金额预算；这不是 Key 无效，也没有连接厂商。只反馈上述字段，不发送 Key。等用户授权后，再创建本地 profile 并进行单次连接测试。

DeepSeek 当前默认开启 thinking，且支持非思考模式。首次简短连接测试拟显式选择非思考模式，避免低输出上限消耗在推理内容上；此调用选项尚未经过真实接口验证，不能宣称原 app 已兼容或跑通。现有 preflight 不验证该选项。

以下通用参数说明适用于其他厂商；DeepSeek 的地址和模型已在上述模板提供，不需要用户重复查找。

回复provider、model ID、API Base URL（Azure另需API version）、总请求上限、每次输出token上限、预算金额与单位。
依据规范§3.3，在这些信息确定并明确授权前，保持allow_paid_api=false、api_request_cap=0。
现有原SDK使用Chat Completions；不能把仅支持其他API的模型名直接代入。确定服务后再核对其官方接口与模型能力。

## 在自己的PowerShell终端配置Key

密钥只注入当前终端，不写命令历史中的字面量或任何项目文件。
[OpenAI官方认证文档](https://developers.openai.com/api/reference/overview)要求将Key作为秘密，由环境变量或密钥管理服务加载。
项目沿用OPENAI_API_KEY变量；其他兼容供应商也使用该变量名传其自己的Key。

```powershell
Set-Location D:\AgenticRec
$llmSecureKey = Read-Host '输入本次服务的 API Key（不回显）' -AsSecureString
$llmKeyPointer = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($llmSecureKey)
try {
    $env:OPENAI_API_KEY = [System.Runtime.InteropServices.Marshal]::PtrToStringBSTR($llmKeyPointer)
} finally {
    [System.Runtime.InteropServices.Marshal]::ZeroFreeBSTR($llmKeyPointer)
    $llmSecureKey.Dispose()
}
```

这里只对你运行命令的PowerShell及其子进程有效，不会自动进入Codex的其他终端。
后续连接检查应在同一终端执行；只反馈不含密钥的结果。

## 配置公开参数并做零请求检查

```powershell
if (-not (Test-Path -LiteralPath reproduction/.runtime/live_llm.json)) {
    Copy-Item -LiteralPath reproduction/live_llm.example.json -Destination reproduction/.runtime/live_llm.json
}
notepad reproduction/.runtime/live_llm.json
```

JSON只填写公开连接参数及已授权的预算，不含API Key。模板默认禁止调用；在已明确授权后才改allow_paid_api和请求额度。
运行：

```powershell
python reproduction/scripts/live_llm_preflight.py
```

- CONFIG_READY：参数完整、当前终端存在Key；不证明Key有效、网络可用、模型权限或真实连接成功。
- BLOCKED：按missing_or_invalid补齐。预期返回码2；检查始终0 API请求。
- 脚本只读取Key是否存在，不打印Key、Base URL或错误配置的原内容；不导入SDK、加载checkpoint或访问网络。

模板及检查用于远程HTTPS接口；若准备使用本地HTTP模型或其他计费方式，先说明实际服务，不猜测参数。

## 配置完成后

回复provider/model/限额/金额预算，以及preflight输出。随后准备**最多1次HTTP尝试**的连接测试，使用原OpenAICall主调用路径，关闭SDK自动重试并统一计算请求消耗。
[OpenAI官方重试说明](https://developers.openai.com/api/docs/guides/rate-limits)提醒SDK和外层重试叠加会增加请求。
原app默认还有规划、摘要等调用，首次测试不直接开放不受限的UI；单次连接成功后再按已授权剩余额度运行原app单轮和多轮。
费用无法核实时记null及原因；不能把声明的金额预算当成已经实现了供应商账单硬限额。

结束当前终端使用后，可清除本次注入的环境变量：

```powershell
Remove-Item Env:OPENAI_API_KEY
```

本阶段没有升级SDK、改prompt、改推荐算法、下载额外模型或运行真实评测。
