# 重大决策

## ADR-006：A1 采用工作区内 Python3.9、CPU构建与 zero-demo 兼容环境

- 最终验证（2026-10-01）：采用，T02 DONE。174个实际包版本在第二个离线环境逐项一致；pip check、完整native导入、原SDK本地HTTP mock通过。发生HfFolder/cached_download和Bert调用split_torch_state_dict_into_shards兼容错误后，固定Hub0.16.4、Transformers4.33.2、Tokenizers0.13.3、Accelerate0.23.0；没有全局降级SDK。
- 实际运行：5个完整电影成员CRC/SHA通过；原Gallery以真实gte-base编码9888标题；原SASRec的36个state key和数值全部与checkpoint一致，并对9888目录项产生有限分数。原查询、过滤、相似召回及三个排序schema通过。原app.py在loopback返回HTTP200，两轮回调通过，但LLM HTTP响应明确是fixture。正式性能指标均NOT EVALUATED。
- 当前路线：保留A1原资源兼容功能路线，原始Python源码完全未改，仅副本requirements发生兼容差异，diff已导出。T03仍IN_PROGRESS（ID语义/来源/独立授权待核验），T04仍BLOCKED（真实LLM授权与原eval数据缺失）。没有转B、没有提前训练或Agent增强。以下记录保留安装过程；最终事实以上述native证据为准。

- 原要求：README Python3.9 + 原 requirements；隔离 legacy/dev，不改系统CUDA，先关闭demo/reflection/shortening缩小复现面。
- 实测：系统无3.9/Conda。下载固定uv0.9.2及托管CPython3.9.24到工作区，禁止系统命令和注册表写入；创建独立venv。原requirements解析成功，安装先因greenlet需要C++编译器失败；固定兼容wheel2.0.2后又因chroma-hnswlib0.7.2缺Windows编译环境失败，原始日志保留。
- 最小替代：PyTorch1.13.1+cpu处于原版允许范围，减少下载并不改系统驱动；UniRec固定0.0.1a4。zero-demo入口不实例化Chroma，仅从本地兼容依赖清单排除chromadb固定项；其他原版固定包保留。动态demo暂NOT VERIFIED，不安装系统级编译工具，不偷偷升级Chroma算法。成功依赖组合须实测后锁定，不能将当前input文件视作成功lock。
- 源码隔离：原RecAI保持pristine，在忽略目录复制固定InteRecAgent；最小兼容改动先保留失败测试、再改copy，最终导出独立patch。原表/矩阵/权重来自README原包，已下载小文件复用，不创建替代矩阵。
- 资源网络：完整电影checkpoint CRC/SHA通过，使用当前已安装Torch的weights_only读取真实参数，明确不是legacy模型运行。矩阵下载曾TLS中断，保留失败证据并使用经测试的分块缓存/有限重试，TLS始终校验。
- 比较影响：CPU环境与zero-demo/固定依赖仅用于A1兼容功能验证，不能当A2论文设置或GPU性能结果。共同ID语义风险独立标注，不能因工具输出就宣称准确率提升。
- 真实LLM：规范§3.3默认预算0，已异步请求提供商/model/限额与安全密钥配置；收到前继续全部离线工作，不擅自付费。

## ADR-005：撤回冻结 B，恢复 A1 优先核查

- 日期：2026-10-01；状态：采用；取代 ADR-004 的最终路线决定，旧条目保留历史。
- 原要求：规范 §0 默认先尝试 A1；资源或历史模型无法获得时才按证据使用 B。§5.2 要求核验 ID 语义，不要求候选表覆盖模型的全部 embedding。
- 复核发现：UniRec 0.0.1a4 的 predict/forward_item_emb 直接用候选 ID 查 embedding；只要 ID 有效，较大模型允许目录子集。执行原函数体的 CPU fixture 成功对目录 9888 项打分，原 RecModelTool.run 把 Toy Story 的目录 ID 1062 原样传入。没有加载真实 checkpoint，fixture 分数不是推荐实验。
- 上轮错误：把 9889 与 36255 不相等的严格维度断言当作转 B 的充分依据，结论过早。保留原退出 1 的历史证据，明确其只证明行数不相等。
- 新证据：固定 upstream notebook 的保存输出映射 Toy Story→17、Jumanji→105、Grumpier Old Men→232；真实目录分别为1062、1023、1325，且目录 ID17 是 Hot Shots! Part Deux。notebook 保存的 users/items=298073/36254 与 checkpoint 配置加 padding 后相同。这提示资源映射来源可能不同，但保存输出可能陈旧，仍不能等同 checkpoint 的权威行标签。
- 决定：T03 重新 IN_PROGRESS；route_candidate=A1，route_final=NOT VERIFIED，B 冻结撤回。先核查原权重加载、可信训练映射和 legacy 环境，再判断原版可用性；不推进 MovieLens 重建或模型训练。T04 仍 BLOCKED，因为完整资源/环境/语义验证尚未就绪。
- 下载：本轮新增 136478 bytes 响应正文，包含官方 PyPI 固定 wheel 121607 bytes 和 metadata；SHA 校验通过，仅检查代码、未安装。没有下载完整电影 checkpoint 或大矩阵，因为权重数值本身不能给出缺失的电影标签。
- 可比性：本轮无模型或评测变更；U0/A1 运行 NOT VERIFIED，所有指标 NOT EVALUATED。后续原版运行若能执行，仍需区分“工具可运行”与“共享 ID 语义正确”。
- 证据：`reproduction/id_mapping_recheck.json`、`unirec_source_manifest.json`、`reports/t03_id_recheck.md`、原始日志与本轮 hash index。固定 RecAI 未修改；UniRec 版本只是 requirements 允许的最低版本，不宣称 checkpoint 训练时使用该版本。

## ADR-004：原资源共享 ID 契约无法核验，采用规范 B 路线

历史记录：本条的最终路线决定已由 ADR-005 撤回；维度不同不构成不能执行的充分证据。

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
