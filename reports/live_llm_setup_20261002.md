# 真实LLM接入准备（2026-10-02）

## Completed

核对规范§3.3/§5.3和原SDK读取变量；准备本机密钥注入指导、默认禁用模板及零请求preflight。真实连接尚未执行。

## Files Created / Modified

新增reproduction/LIVE_LLM_GUIDE.md、live_llm.example.json、live_setup_evidence.json、scripts/live_llm_preflight.py及tests/test_live_llm_preflight.py；更新STATUS入口、详细STATUS和reproduction/README.md。

## Commands / Tests

`python -m unittest discover -s reproduction/tests -v`；3项新preflight测试另在legacy Python3.9执行；preflight默认/模板拒绝以及未知CLI参数拒绝；源码语法和source/docs diff --check。

## Verification Results

全11项离线测试通过；3项新增门禁测试在3.9通过。未配置/未授权模板按预期退出2，未知参数退出2；日志见reproduction/logs/live_setup_20261002，产物hash见live_setup_evidence.json。真实API请求0，live connection NOT VERIFIED。

## Findings

原主链使用OPENAI_API_KEY、OPENAI_API_BASE和OPENAI_API_TYPE；已有SDK2.48.0，无需安装新SDK。原外层和SDK默认均有重试，实际连接阶段需统一控制，不能只传外层retry_limits=1就宣称只有1次HTTP尝试。用户终端注入的Key不会自动进入Codex其他终端。

## Blockers

provider/model/API Base URL及明确请求、token、金额预算等待用户答复；Key由用户本机安全配置。原T03语义/来源及T04评测数据阻塞仍保留。

## Deviations from Spec

None。没有修改上游源码、SDK版本或最高规范，没有发起付费请求；配置检查不是连接成功或实验结果。

## Current Project State

真实运行准备阶段；T04保持BLOCKED。preflight的CONFIG_READY只代表参数完整和Key存在，不证明有效性或授权模型可用。

## Next Tasks

用户给出公开服务/模型与预算信息→核对供应商接口→同终端安全注入Key及填profile→零请求preflight→按授权最多1次HTTP连接尝试→原app单轮/多轮。正式连接结果出来前不继续评测。
