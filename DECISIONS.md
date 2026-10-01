# 重大决策

## ADR-004：原资源共享 ID 契约无法核验，采用规范 B 路线

- 日期：2026-10-01；状态：采用；替代 ADR-002 的待定路线。
- 原要求：A1 优先；T03 核验目录、矩阵和 checkpoint 共享 ID、来源、授权与兼容性；真实阻塞时记录原因，使用 B 方法级重建。
- 发现：原 Google 包可读，ZIP 497,403,134 bytes、15 条目。真实电影目录 9,888 行且 ID=1..9888；矩阵头部 shape=(9889,9889)；checkpoint 静态配置 n_items=36255、item_embedding.weight shape=(36255,32)。清单无 ID 映射及独立授权文件。对齐要求真实断言失败，退出 1。
- 原方案当前不能验收的原因：更多 embedding 行本身不证明所有推荐错误，但无法证实同一 ID 的语义；裁剪或猜测映射会改变 baseline。预制资源许可、完整权重 CRC 和运行兼容性仍 NOT VERIFIED。
- 替代：A1 BLOCKED；T03 的调查和路线决定 DONE；进入规范已有 B 路线，后续 T05 → T06，以独立 MovieLens 1M 数据契约构建 U1。T04 原资源运行保持 not_run，其他阶段未提前执行。
- 下载最小化：两次 bounded Range 合计 471,747 bytes（含确认页及重复目录），完整表和小 JSON 验证 CRC，矩阵/checkpoint 仅前缀。没有完整 ZIP、矩阵或权重下载，没有反序列化和模型加载。原子集保留于本地忽略目录，不再分发。
- 实验影响：B 只支持方法级重建；不能等同论文数字，不能冒称 U0/A2。未来 U1 对照必须明确新数据、映射、配置与固定源版本。所有性能指标 NOT EVALUATED。
- 证据：`reproduction/resource_manifest.json`、`route_decision.md`、`t03_evidence_index.json` 及 `reproduction/logs/t03_20261001/` 的原始日志；upstream 无行为 patch。

## ADR-001：上游与新增产物采用工作区并列布局

- 日期：2026-10-01；状态：采用。
- 原要求：§4.1推荐在RecAI内新增AgenticRec/reproduction；保留上游历史与许可证。
- 发现：clone成功，但当前沙箱拒绝在新RecAI仓库中创建普通子目录（New-Item PermissionDenied），同时.git写操作需权限升级。工作区根目录允许普通产物写入。
- 最小替代：`RecAI/`保留原Git仓库与固定HEAD；`AgenticRec/`、`reproduction/`、`reports/`及状态文件放在用户指定工作区根目录。后续源码引用统一`RecAI/InteRecAgent/...`。根项目Git忽略RecAI，upstream_lock保存准确URL/SHA与文件hash。
- 影响：只有路径变化，不改prompt、模型、数据或评测。所有upstream跟踪文件保持原样；没有将Windows ACL放宽或修改全局Git配置。
- 分支：用一次命令级`safe.directory`创建`work/agenticrec`；权限升级用户与clone用户不同，故需该临时Git配置。没有push。

## ADR-002：保留A1优先；当前不以HTML访问替代资源核验

本条为首轮历史决定，最终路线已由 ADR-004 的实际资源证据更新。

- 状态：A1调查中；A/B最终路线未冻结（NOT VERIFIED）。
- 原要求：先尝试A1，真实资源/历史模型不可得时记录证据转B；T03校验授权、settings、ID、矩阵和checkpoint。
- 发现：Google资源预览可达；公开下载入口返回病毒扫描确认页，文件显示474M。RecDrive只取得SPA页面。资源包、settings、表、矩阵和checkpoint尚未取得。没有原资源失效的充分证据。
- 决定：T03保留IN_PROGRESS，下一步核验公开原包清单及数据许可；必要时下载到本地忽略目录，不自动加载checkpoint。B仍是规范fallback，当前不激活也不重建数据。
- 影响：U0/A1/A2均not_run；没有论文数字比较。若后续确实阻塞，再记录B转换原因与可比性变化。
- 合规：MovieLens官方条款已通过Web读取；研究使用须引用，不重分发、不用于商业用途；本轮未下载或发布MovieLens。预制包许可证仍NOT VERIFIED，代码MIT不用于替代数据授权。

## ADR-003：仅用隔离原源码建立G0控制流证据

- 原要求：指定解释器/网络不足时先静态审计和CPU fixture；先保存bug失败证据，再按依赖修复。
- 发现：没有3.9/3.11/Conda；只见3.13.7。真实资源未就绪，原app有import与模型初始化副作用。
- 决定：用标准库AST加载固定源码选定class并延迟类型注解，原方法体不改；使用明确fixture替代LLM、Gallery、过滤和排序。保留真实原Map/Buffer/Agent控制流；不导入原app、不安装旧依赖。
- 影响：只能验证控制流和F04重复覆盖，不属于原资源功能复现；SQL、推荐分数、embedding与真实LLM全部NOT VERIFIED/NOT EVALUATED。不是后续FakeLLM产品实现，T05仍TODO。
