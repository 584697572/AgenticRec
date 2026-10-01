# 重大决策

## ADR-001：上游与新增产物采用工作区并列布局

- 日期：2026-10-01；状态：采用。
- 原要求：§4.1推荐在RecAI内新增AgenticRec/reproduction；保留上游历史与许可证。
- 发现：clone成功，但当前沙箱拒绝在新RecAI仓库中创建普通子目录（New-Item PermissionDenied），同时.git写操作需权限升级。工作区根目录允许普通产物写入。
- 最小替代：`RecAI/`保留原Git仓库与固定HEAD；`AgenticRec/`、`reproduction/`、`reports/`及状态文件放在用户指定工作区根目录。后续源码引用统一`RecAI/InteRecAgent/...`。根项目Git忽略RecAI，upstream_lock保存准确URL/SHA与文件hash。
- 影响：只有路径变化，不改prompt、模型、数据或评测。所有upstream跟踪文件保持原样；没有将Windows ACL放宽或修改全局Git配置。
- 分支：用一次命令级`safe.directory`创建`work/agenticrec`；权限升级用户与clone用户不同，故需该临时Git配置。没有push。

## ADR-002：保留A1优先；当前不以HTML访问替代资源核验

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
