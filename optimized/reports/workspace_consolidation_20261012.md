# 两部分目录整合验收（2026-10-12）

**本地整合与验证 DONE；GitHub发布 PENDING。** 本次是文件与发布管理变更，不是新的推荐质量实验。

## 最终结构与归属

仅两处公开实体目录：`original/` 为微软固定源码子模块、兼容复现脚本/环境、原版报告及本地原资源；`optimized/` 为截至4919c35的个人实现、独立模型实验、历史记录和本地数据/权重。

固定子模块commit：`0959ecb05b0794748426e73e6efc1b6b35ec433d`。原上游工作树始终干净。研究以fast-forward整合到master，旧分支和Git历史保留；历史规范不再约束后续执行。

根README、两个版本README和当前运行指南重新整理；被替换的旧导航、状态、决策、TASKS及规范保留原字节。旧路径通过被忽略的本地junction/hardlink兼容，不重复下载数据/模型。两个密钥辅助模块改为直接定位根 `.env`，旧模块也归档；真实Key内容没有读取、修改或输出，避免编辑器原子保存造成过时硬链接。

15个旧 `.tmp_t19*` 目录转入 `optimized/artifacts/archived_test_workspaces/`，无删除或ACL修改。环境和正式实验权重不覆盖。

## 验证

| 检查 | 真实结果 |
|---|---|
| 优化版完整pytest | 350通过，3个已有Torch警告；131.15秒测试时间 |
| 原版unittest | 64通过 |
| 原App离线运行 | SUCCESS，原资源和44份原Python文件校验后运行过滤/排序/Map；2次本地mock transport，远程请求0 |
| BPR与LightGCN已验证profile入口 | 两者exit0；请求使用匿名结构化示例，加载真实权重，远程请求0 |
| doctor / fixture | exit0，fixture明确为toy数据/FakeLLM |
| 原始输入完整性 | 695份原Git跟踪输入在新位置或档案中SHA一致 |
| 数据及checkpoint完整性 | 363份二进制文件SHA和大小一致 |
| 上游 | 固定commit、工作树干净、gitlink正确 |
| 全新公开源码快照 | 导出718个公开文件，prepare及fixture通过，5个当前导航文档无断链；复用已有环境，不含原子模块checkout/数据/权重 |
| Git暂存字节与私密排除 | 全部实际暂存文件逐项按Git blob核对；公开目录仅original/optimized，数据/权重/Key/环境/私密日志不入索引 |
| git diff --cached --check | 通过；MIT许可按实际CRLF字节保留并声明cr-at-eol |

原始日志和真实退出码位于本地忽略的 `optimized/artifacts/verification/two_part_20261012/`；迁移库存见 [layout_20261012.json](../history/layout_20261012.json)。早期检查、失败尝试和两个源码导出目录均保留，不覆盖已有科学实验。

## 已处理问题

最初PowerShell执行策略阻止直接运行迁移脚本，使用单次进程参数运行自有脚本，没有改系统策略。首次pytest因系统临时目录不可访问未进入测试，改用工作区内新basetemp。日志包装器的相对可执行文件定位修正后完整运行；最终新增4项目录/Key定位回归均包含在350项中。

旧路径ignore规则迁移时先在暂存中发现runtime/cache/log路径，补齐规则并从索引移除，最终逐项检查后才提交；没有发布私密文件。部分旧测试目录属于不同沙箱账户，直接Move失败；在相应可用权限上下文完成同盘原子归档，不改变ACL、不删除内容。失败的备用父目录归档在首项失败并回滚，空临时目录清理后没有遗留。

## 效果与限制

新增训练0、付费LLM请求0、模型/数据下载0。模型数学、默认模型、采样配置、原资源和历史指标不变；密钥文件定位仅是迁移适配。工程验收不产生新的准确率或线上效果结论。

原资源ID、独立许可及canonical论文条件仍未完成验证。GCN相对强BPR及最新低分监督仍NO VERIFIED GAIN。此次源码快照验证不是Linux/GPU、全新依赖安装或原论文完整复现。

## 发布

本地验收已完成。提交并推送master后核对远端SHA，随后记录GitHub发布结果；不使用force push，不删除历史分支。
