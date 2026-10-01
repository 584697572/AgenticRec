# G0首批执行报告（2026-10-01）

## Completed

- 完整读取1240行执行规范，保留原SHA256；核查空工作区及不存在的执行包文件。
- T00 DONE：clone真实上游，固定0959ecb05b0794748426e73e6efc1b6b35ec433d，创建本地work/agenticrec；源码工作树干净；保留LICENSE.txt。
- T01 DONE（审计）：逐文件走读buffer/query/sql/memory/critic/demo/eval与主控制流，复核F01—F10；保存真实隔离控制流fixture与重复工具覆盖的失败证据。
- T03只完成访问调查及缺失登记，IN_PROGRESS；最终A/B路线尚未核验，不能算DONE。

## Files Created / Modified

新增STATUS.md、DECISIONS.md、TASKS.yaml、.gitignore、LICENSE.txt；AgenticRec/docs下STATUS/source_walkthrough/static_audit；reproduction下upstream_lock/environment_inventory/resource_manifest/route_decision、三个审计脚本；reports/spec_issues.md及本报告。证据hash见reproduction/g0_evidence_index.json。

RecAI是新clone；上游跟踪源码未修改。完整执行规范未修改。原始运行日志仅本地保留reproduction/logs/g0并被Git忽略。

## Commands / Tests

- git clone https://github.com/microsoft/RecAI.git RecAI（退出0）。
- git -c safe.directory=D:/AgenticRec/RecAI switch --create work/agenticrec 0959ecb05b0794748426e73e6efc1b6b35ec433d（退出0，本地权限升级）。
- git -C RecAI rev-parse HEAD / status --short / diff --exit-code 指定SHA -- InteRecAgent LICENSE.txt（均退出0；git_verification.json）。
- python reproduction/scripts/probe_resources.py --run-label network / download（均退出0；HTTP/传输失败独立记录，程序成功不等于资源可用）。
- python reproduction/scripts/audit_upstream.py --require-duplicates（退出1，未修复的真实断言失败；duplicate_requirement_v2日志）。
- python reproduction/scripts/audit_upstream.py（退出0，audit_v2日志）。
- python reproduction/scripts/verify_g0.py（退出0，deliverables日志）。
- git diff --cached --check全量首检退出2，原规范已有Markdown双空格换行触发；许可证原CRLF也被默认检查视为行尾空白。原文件均未改，新增产物单独检查，命令级cr-at-eol允许原CRLF；.gitattributes防止原规范/许可证checkout换行归一化。

## Verification Results

- 审计测试实际运行4项，failures=0、errors=0；全部通过。
- 重复步骤保留要求实际运行1项，failures=1、errors=0；失败。实际只有second-query，first-query丢失；未修复。
- 交付校验通过：原规范hash、固定原始源码hash、许可证、25任务依赖和无环检查、源码符号、资源状态与指标null。
- 初跑审计harness常量解析错误已修复，旧失败日志保留；v2证明的是上游字典覆盖，不能把harness异常当upstream问题。
- A1/A2 NOT VERIFIED，所有推荐/系统指标NOT EVALUATED；未安装依赖、未训练、未运行原app/UI/真实评测、未调用真实API。

## Findings

上游原工具依靠共享CandidateBuffer；排序模型输入是item_seq，不能直接换BPR-MF/LightGCN checkpoint。Map取前N并重置候选；反思、历史压缩、demo选择原版已有。Google预览可达，公开下载入口返回474M确认页；RecDrive仅SPA，不能据此说包可得或链接失效。原settings缺失已实测。

## Blockers

- T03原资源包/数据授权/ID与checkpoint契约尚未验证，A/B最终决定未冻结。
- 当前可见Python3.13.7，legacy3.9/dev3.11未建立；只有4GiB GPU硬件清单，torch/CUDA兼容性未验证。
- GroupLens的Python urllib访问证书链失败；TLS未关闭，官方条款通过Web读取；后续下载需正常证书链路径。
- live默认禁止且预算为0，后续真实LLM验收未获授权。

## Deviations from Spec

§4推荐布局因RecAI目录写入被沙箱拒绝，采用工作区根目录并列AgenticRec/reproduction，RecAI保留独立原历史；ADR-001记录原要求、错误、替代及无实验语义影响。缺失TASKS.yaml从附录生成，不重拟路线。T03不强行DONE，不以资源页面可达推定资源完整。

## Current Project State

G0首批审计与版本锚点已建立；T00/T01 DONE，T03 IN_PROGRESS，G0尚未整体验收。可重放隔离控制流测试；原资源复现与论文实验均未完成。没有remote push/PR。

## Next Tasks

按原依赖先完成T03包授权、资源清单与契约、A1/B最终决定；再T02/T05隔离环境。A前置成立后T04无LLM工具冒烟；B决定成立且T05完成后才T06。先数据契约和指标fixture，再训练；本轮停止在首批范围。
