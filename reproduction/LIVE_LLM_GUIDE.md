# 真实LLM运行：第一步，本机配置检查

2026-10-02。当前仅准备配置，真实连接NOT VERIFIED，真实API请求0。
下一步遵循规范§5.3：单次连接→原app单轮→多轮；原评测另有数据与资源语义前置。

## 先提供不含密钥的信息

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
