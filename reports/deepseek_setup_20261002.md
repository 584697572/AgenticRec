# DeepSeek 配置指导（2026-10-02）

## Completed

用户已选择 DeepSeek；核对官方 OpenAI 兼容地址与当前建议模型，提供专用禁用模板及本机 Key 输入指导。未连接真实 API。

## Files Created / Modified

新增 reproduction/live_llm.deepseek.example.json、本报告；更新 reproduction/LIVE_LLM_GUIDE.md、STATUS.md、AgenticRec/docs/STATUS.md。

## Commands / Tests

`python -m unittest discover -s reproduction/tests -p test_live_llm_preflight.py -v`；专用模板 preflight CLI 及 subprocess 退出码检查；以 Key 存在布尔 fixture 检查模板仍拒绝；`git diff --check`、规范 SHA256 与 pristine upstream 检查。

## Verification Results

3 项已有门禁测试通过。实际 CLI 输出 BLOCKED、key_present=false、api_requests=0；子进程退出码确认为 2。仅将 Key 存在性设为 true 的离线 fixture 仍拒绝，缺项准确为 allow_paid_api、api_request_cap、money_budget；该 fixture 不是有效 Key 或真实连接证据。diff --check 通过；规范 SHA256 保持 d88f02c84855234673514519486e74e73c61ffd4e2ec75594020fdf9b8ebb677，upstream 无修改。

## Findings

[DeepSeek 官方入门](https://api-docs.deepseek.com/)给出 OpenAI 兼容地址 https://api.deepseek.com 和模型名 deepseek-flash；[模型说明](https://api-docs.deepseek.com/quick_start/pricing/)标明默认 thinking，并支持非思考模式。拟在首次短连接检查显式选非思考模式；尚未验证该选项与原 app 的真实兼容性。两次尝试打开 thinking_mode 指南均遇官方页面抓取超时，使用成功访问的官方入门/模型页核对模式能力；未据此修改上游。

## Blockers

用户须在自己的 PowerShell 输入 Key，并授权请求/token/金额限额。拟议首次 1 次 HTTP 尝试、最多输出 128 token、预算 1 CNY，尚未授权；模板保持 allow_paid_api=false、api_request_cap=0、money_budget=null。原评测数据及资源语义/来源前置仍未解决。

## Deviations from Spec

None。不改变规范、上游源码、依赖或算法；只准备服务配置和指导。更换服务/模型属于 A1 功能验证，不宣称 A2 论文设置复现。

## Current Project State

T04 BLOCKED；真实连接 NOT VERIFIED；原评测 NOT EVALUATED。本轮真实 API 请求 0，无实验结果或实际费用数据。

## Next Tasks

同终端安全输入 Key → 授权首次请求/token/金额限额 → 本地 profile 的零请求检查 → 受限单次真实连接 → 原 app 单轮/多轮 → 补齐并运行原评测。遵守规范§3.3/§5.3。
