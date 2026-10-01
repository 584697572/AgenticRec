# 项目状态

最后核验：2026-10-01（Asia/Shanghai）。当前阶段：**G0首批审计；G0尚未整体验收**。

最高实施规范：工作区`AgenticRec_完整复现与优化执行规范.md`，完整读取1240行，未修改；任务依赖按附录导出至工作区`TASKS.yaml`。

| 任务 | 状态 | 验证命令 / 退出码 / 日志 | 产物与hash |
|---|---|---|---|
| T00 锁定工作区、源码与出处 | DONE | `git -C RecAI rev-parse HEAD`、`git -C RecAI status --short`；0；`reproduction/logs/g0/git_verification.json`；audit_v2的pin测试通过 | upstream_lock.json；SHA和文件hash在版本锁及g0_evidence_index.json |
| T01 逐文件走读与静态风险复核 | DONE（仅审计） | `python reproduction/scripts/audit_upstream.py`；0；`audit_v2.stdout/stderr.log`、audit_result.json。四项检查通过；源码路径/符号人工对照 | source_walkthrough.md、static_audit.md、control_flow_trace.json、duplicate_observed.json；hash见g0_evidence_index.json |
| T03 核验资源并决定A/B | IN_PROGRESS | 公开资源小范围probe已执行，程序退出0仅代表完成记录；不代表资源验证通过。详见resource_manifest.json与resource_network/download日志 | manifest及route_decision.md；授权、包清单、ID/矩阵/model检查NOT VERIFIED；最终路线未冻结 |
| T02 legacy环境 | TODO | 未安装；Python3.9/Conda未发现 | 没有成功lock或兼容patch，不伪造 |
| T04 原资源运行/评测 | TODO | 前置资源、环境、预算未就绪，not_run | A1/A2 NOT VERIFIED；U0 NOT EVALUATED |
| T05—T24 | TODO | 本轮未执行后续实现、训练与实验；可选任务需满足条件 | 所有模型/系统指标NOT EVALUATED |

## 尚缺证据 / 约束

- 原资源包尚未下载。Google入口可访问但返回474M病毒扫描确认HTML；RecDrive为SPA；没有证明失效，也没有确认下载授权与内容。T03未完成，不能标DONE。
- 本机只发现Python3.13.7；py launcher无注册Python；没有Conda。3.9/3.11环境尚未建立；GPU为GTX1650Ti 4096MiB，Torch/CUDA兼容性NOT VERIFIED。
- Python urllib访问GroupLens时证书链验证失败；保留错误，不关闭TLS；官方条款通过Web工具读到。后续数据下载需正常TLS链路径。
- 真实LLM默认关闭，allow_paid_api=false、api_request_cap=0，未请求预算或读取真实key；live not_run。
- 原重复工具调用保留要求失败；已观察覆盖，但未修复。T14仍TODO，不以审计通过掩盖bug。

## 下一任务

1. 继续T03：在原资源可达事实基础上核验包授权、清单、settings和ID契约，明确能否走A1；证据不足不提前激活B。
2. 工作区布局已记录ADR-001；按T02/T05分别建立legacy/dev隔离环境，确认3.9/3.11来源和真实依赖组合。T02阻塞时按规范允许继续独立dev。
3. A路径前置成立再T04无LLM工具冒烟；B路径决定成立并完成T05后再T06数据契约。不得跳到训练。

## 最后验证

- 上游HEAD固定`0959ecb05b0794748426e73e6efc1b6b35ec433d`；分支work/agenticrec；上游跟踪文件diff为空。
- 原执行规范hash不变；初始用户文件未覆盖。
- 审计测试4项通过；重复保留要求1项真实断言失败（退出1），未修复。
- 自造fixture只证明隔离控制流：原run→plan_and_exe→ToolBox→候选变化→原Map→总结→原DialogueMemory。
- 无真实数据实验、无模型训练、无依赖安装、无真实API、无push/PR。
