# 本机 .env 持久配置（2026-10-05）

## Completed

按用户明确偏好创建根目录 `.env` 空 Key 字段，用户随后已在本机填写；连接检查、原 wrapper 连接脚本与原 App 单轮入口支持显式读取。运行命令与结果检查由执行端承担，用户只需在本机保存 Key。

## Files Created / Modified

新增 `reproduction/scripts/local_api_key.py`、`reproduction/tests/test_local_api_key.py`；修改三个 live/preflight 入口和指导/状态/决策文档。`.env` 只在本机、被 Git 忽略，未读取其内容作为工具输出或提交；未将 Key 写入 JSON profile。未提交的旧 PowerShell 输入窗口尝试存入本地忽略目录，已退出主实现。

## Commands / Tests

`reproduction/.venv-legacy/Scripts/python.exe -m unittest discover -s reproduction/tests -v`；`git check-ignore -v -- .env`；`git ls-files -- .env`；新 profile 的零请求 preflight。日志位于 `reproduction/logs/dotenv_20261005/`。

## Verification Results

先保存旧行为不支持文件读取的失败回归：9 项测试中含 10 个 subtest/assertion 失败、1 个缺入口符号错误；实现后完整 32 项测试通过，其中新增 9 项。覆盖环境变量优先、UTF-8/BOM/引号/注释、缺失/空值、重复/坏语法、解码错误不泄露、禁止插值/环境改写、真实入口接线（执行函数为 mock）、offline 不读 Key、worker 导入无读文件副作用。所有测试远程请求 0。Git ignore 生效，`git ls-files -- .env` 为空。

## Findings

普通及提权运行都缺少原用户 PowerShell 的进程环境变量，实际原 App 命令曾以退出 2 / BLOCKED / 0 请求拒绝。根目录 `.env` 是跨执行进程可用的持久配置。新实现只在 live 父进程入口显式读取，没有全局 `load_dotenv()` 导入副作用；Key 不自动注入资源 worker。文件使用明文存储，Git 忽略不等同加密或 OS 沙箱隔离。

## Blockers

用户已在本机保存 Key，最新零请求检查 CONFIG_READY（key_present=true、api_requests=0），执行端已完成真实单轮 SUCCESS/VERIFIED，2次HTTP200，usage=3655+225=3880；详见 [真实运行报告](live_original_app_20261005.md)。原资源权威 ID 语义、原评测设置等 T03/T04 限制仍独立存在。

## Deviations from Spec

总体路线、上游与算法未变。用户本轮明确选择 `.env`，取代此前“仅当前终端、不落文件”的操作约定；ADR-010 记录这一偏好变化。文件不进 Git，不变更预算或重置额度。

## Current Project State

T00/T01/T02/T05 DONE；T03 IN_PROGRESS；T04 BLOCKED；T06—T24 TODO。配置能力已完成；真实原 App 单轮 VERIFIED，多轮 NOT VERIFIED，评测/算法收益 NOT EVALUATED。之前 `progress_evidence_20261005.json` 对应 commit 82bbf4e 的历史快照，不能拿其旧脚本 hash 校验本次修改。

## Next Tasks

已由执行端完成并核验原 App 单轮，接下来按规范准备多轮、原评测与算法前置。现有单轮额度 2 次尝试、512 输出 token/次、1 CNY；不再让用户代跑命令。
