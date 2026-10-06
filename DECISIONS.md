# 重大决策

## ADR-007：授权的单次 DeepSeek 连接由独立测试层限制

- 日期：2026-10-04；状态：采用；单次真实连接VERIFIED，原app仍NOT VERIFIED。
- 原要求：规范§3.3/§5.3，在真实工具冒烟后先单次LLM连接；明确provider/model/请求/token/金额限额，统一重试控制，不把Key写入文件、不改原baseline。
- 发现：用户已授权一次连接、输出128 token、1 CNY，并在自己的PowerShell配置Key；Codex执行终端没有Key。原OpenAICall默认外层重试5次、SDK还有默认重试，且未显式选择DeepSeek非思考模式。
- 最小实现：单独脚本直接加载固定upstream且校验原文件SHA，不改任何原方法体；测试层OpenAI工厂设置SDK max_retries=0、原wrapper retry_limits=1、HTTP retries=0、不跟随重定向，仅对固定短提示额外配置thinking disabled。独占、fsync的固定额度记录在发送前创建，跨进程重复调用、失败和超时均不自动返还额度。不进行额外余额/model-list API查询。
- 依据：[DeepSeek官方Chat Completions](https://api-docs.deepseek.com/api/create-chat-completion/)及[人民币价格](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/)，2026-10-04核对。固定提示的1024输入token规划额度不是tokenizer认证上限；输出限制128。费用上界估计按公开高峰率，实际扣费未查询则null；未实现供应商账单硬限额。
- 可比性：这是A1兼容功能的单次原SDK连接证据，改变服务模型和非思考选项须公开，不能算A2论文设置或完整原app复现。受限脚本不授权原UI/单轮/多轮/评测的新请求。原算法、prompt模板和源码保持原样，无新训练或Agent增强。
- 验证：用户同终端完成真实HTTP200/OK，actual usage=12输入+1输出=13，唯一尝试已消耗；原始结果/额度已核对。16项离线测试通过，包括真实SDK+mock HTTP成功、500、超时、重定向及额度/配置门禁。fixture usage明确标注，不写入正式实验指标；已消耗真实额度的门禁验证不新增调用且保留原结果。原app单轮/多轮需另行明确限额，未传输Key。

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

## ADR-008：已授权的受限原 App 单轮与独立 T05

- 日期：2026-10-05；采用。用户明确允许请求及输出，执行此前已提出的单轮具体范围：DeepSeek deepseek-flash、thinking disabled、2 次 HTTP 尝试、512 输出 token/次、1 CNY、零重试。旧的 1 次连接授权不重置；后续批量实验尚无具体可执行配额配置。
- 原要求：§5.3 单次连接后先原 App 单轮；§5.2 不向加载未获完全信任 checkpoint 的进程提供真实 Key；保留原算法/prompt、按预算记录每次尝试。
- 发现：Key 在用户 PowerShell，执行端不继承；原 App 初始化加载 checkpoint。原外层/SDK 重试会放大请求，摘要 token 配置也需在统一边界控制。
- 实现：无真实 Key 的原 App 子进程继续调用原 SDK，通过 stdio 交给父进程限额发送 HTTPS；父进程在发送前原子登记尝试、fsync、禁止重试/重定向，不重置失败额度。原源码及原提示不变，原 UI launch 在本次 batch harness 中被抑制。代码不把这个进程隔离称作 OS 安全沙箱或供应商账单硬限额。
- 验证：完整原资源/原 app 离线单轮 SUCCESS；原 Filter/Ranking/Map 和目录约束通过；2 次 mock HTTP、真实 0。门禁最初遗漏原 stop 参数，保留失败，回归先失败再修复。真实 App 当前 NOT VERIFIED。
- T05：任务卡依赖 T00/T01，不依赖 T04，因此现代包/环境独立验收符合原规范。Python 3.11.14 + 7 锁定依赖，不污染 legacy；第二环境离线重建及15测试通过。T06 仍依赖 T03/T05，未越过数据/指标前置训练。
- 可比性：只是 A1 兼容运行准备和 T05 基础，DeepSeek/zero-demo/CPU 等差异仍在；不能声明 A2、算法提升或完整 Agent 可靠性完成。

## ADR-009：原 ReDial 预处理仅适配整数年份，冻结派生输入

- 日期：2026-10-05；采用。原要求：使用真实原始对话和原指标，输入/选择过程可追溯；类型不兼容时最小适配，先失败回归再修改。
- 发现：固定 UniCRS f7d8f95104d6eba5371cfa845b243057a632079e 仍有 test_data_dbpedia_raw.jsonl；8,751,948 bytes/1,342条，Git blob 与 SHA 校验通过。原 notebook movie_map 在42个 title/year 上因 int64年份减Timestamp失败。作者canonical50条文件和原hash seed不在固定资源中。
- 最小适配：保持原 notebook 函数体，将局部 DataFrame 年份转为当年1月1日 Timestamp，只为兼容比较类型，不新增真实月日事实；原表和 app 输入不变。测试确认等价于原函数的datetime输入，并保留其带符号差最小值选更早电影的原行为。
- 最小下载：映射由各title/year及固定catalog独立决定，故仅下载test，不需要train/valid映射来确定test保留条件。严格保留原test顺序、至少2个已映射正反馈、前50个再process_conv的规则，得到45条（5条按原逻辑丢弃）；不新增样本。原set逻辑有5个并列target对话，固定PYTHONHASHSEED=42。再次独立进程执行，输出SHA256完全一致。
- 可比性：允许命名“A1原notebook派生评测输入”，canonical作者子集等价性NOT VERIFIED、原论文A2 NOT VERIFIED。不能把新 seed 或整数年代表日期隐瞒成原作者设置；此次无LLM评测、所有指标NOT EVALUATED。后续如修正重名/并列策略，必须作为单独方法变更比较，不能覆盖此次输入。
- 合规：ReDial官方数据声明CC BY4.0并要求归属引用；UniCRS代码MIT独立记录；两者都不能授予MovieLens预制包/权重许可。数据与target保留本地忽略目录，不推送或再分发。

## ADR-010：按用户明确偏好使用根目录 .env 持久保存 Key

- 日期：2026-10-05；采用。用户明确要求 .env 持久化并把命令执行交给执行端，因此更新此前仅当前 PowerShell 内存配置的操作约定；不是根据第三方内容自行改变密钥存储方式。
- 发现：普通与提权环境的 Process/User/Machine 均未配置 OPENAI_API_KEY，实际调用安全拒绝、0 请求。旧 PowerShell 弹窗辅助脚本因系统执行策略未能运行；没有弹出窗口、没有改策略，未提交脚本已保存于忽略目录。用户随后明确选择 .env，不继续界面方案。
- 实现：根目录 .env 已被既有 .gitignore 覆盖；仅创建空字段，不写入任何真实值。三入口显式读取，非空环境变量优先；只解析 OPENAI_API_KEY，拒绝重复/坏语法、错误不回显，支持UTF-8/BOM及引号注释；不执行变量插值、不修改其他环境配置、不在 import 时读文件。
- 验证：先保留失败回归，再全32项离线通过。offline路径不读密钥，worker导入不读.env；文件未跟踪；用户保存后preflight key_present=true、CONFIG_READY、请求0；执行端已接手受限真实单轮。读取文件的明文密钥会进入live父进程内存；worker不主动读取或注入，此约定不宣称OS文件沙箱隔离。
- 影响：原版源码、资源、prompt、模型与预算均不变；已消耗额度不重置。只解决安全本机配置和命令代执行，不表示真实App/评测已通过。.env 明文仅保留本机，不纳入公开证据、哈希清单或 Git。

- ADR-010 验证补充：执行端自动读取用户已保存的 Key，真实原 App 单轮 SUCCESS/VERIFIED；2次HTTP200、3655输入+225输出=3880 token，真实Filter/Ranking/Map及目录约束通过。2/2额度已消耗，重复执行拒绝、无新增请求且证据字节不变；实际Key未出现于运行日志。多轮和原评测仍未验收。

## ADR-011：按用户授权创建个人 public 仓库

- 日期：2026-10-05；采用。用户明确要求在自己的账号下新建 public 仓库；本次创建和推送已有项目属于该授权范围，取代规范对未授权推送的默认禁止。
- 发现：根项目没有远程；GitHub 连接器与本机 Git 凭据分别确认账号 `584697572`，该账号下 `AgenticRec` 名称可用。根仓库已有 12 个提交、250 个跟踪文件，独立上游的远程仍为 Microsoft RecAI。
- 决定：创建 `https://github.com/584697572/AgenticRec`，根项目 `origin` 使用不含凭据的 HTTPS URL，保留 `master` 分支和现有提交历史。上游 remote、固定 SHA、执行规范与数据保持原样；根首页明确区分上游、个人改动、证据和未完成项。
- 验证：GitHub 创建返回 HTTP 201；首次推送成功。匿名 API 返回 public、owner 正确、默认分支 master，远程提交与本地发布前提交 `9ab928d5c0fcfe58679d6c9147381a30f248911d` 一致。检查全部可达历史中的 251 个唯一 blob，未发现已配置的实际 Key 或被排除的资源路径；一处 URL 凭据格式命中经人工核对为自造测试 fixture。`.env` 和私密日志仍被忽略。
- 影响：只改变项目托管位置和公开文档，不改变实验设置、任务依赖或验收状态。公开单轮摘要不等于完成 T04 或算法评测。证据见 `reproduction/publication_20261005.json`；该证据明确锚定首次推送，不伪称包含随后新增的发布文档提交。

## ADR-012：独立多轮与原评测额度，保留 A1 条件

- 日期：2026-10-05；采用。用户明确选择新增 120 次请求、每次最多 1024 输出 token、总预算 3 CNY，沿用 DeepSeek deepseek-flash 非思考、零重试。既有连接与单轮额度不重置。
- 执行：原 App 两轮后，按固定 45 条 A1 输入调用未修改的 `one_turn_eval.py` main、原 RecBotWrapper、原 hit_judge 和原 evaluator；随机与加权热门分支离线运行。只将 context 传入原 Agent，target 保留 evaluator；预检 45 条 target 均不在可见 context 中。
- 兼容边界：API 转发、输出上限与无重试仍在测试层；资源工作进程不注入真实 Key；复用真实 Gallery 避免重复编码，不替代目录/模型，eval 创建独立原 bot/buffer。seed 42、zero-demo、关闭 reflection/shortening 均明确记录，原作者设置未恢复。
- 预算：120 个持久请求槽，发送前独占创建并 fsync；超时和失败不归还。未知 usage/未解决的旧请求停止后续付费。每次先作保守字节规划与高峰费率检查，再按实际 usage 更新估算；实际账单为 null，这不是供应商计费硬限额。
- 可比性：只能报告该 45 条派生输入和该模型/设置下的 A1 原模糊文本 hit；不能据此宣称论文 A2 或新算法收益。权威 ID 对应、矩阵语义、资源独立许可和 canonical 子集限制保留。本文建立时多轮/评测结果未验收，完成后保存独立报告。

- ADR-012 验证补充：40+3项离线测试、完整94次mock联调、真实原 App两轮及三个原评测分支45条全部执行完成。实际93次HTTP200、184724输入+17220输出=201944 token，高峰估算0.507208 CNY、账单null。正式三个原指标均0/45；复算一致，22个旧证据不变，重复执行无新增请求，实际Key未出现在证据。结果见 `reports/a1_multiturn_eval_20261005.md`。

## ADR-013：保留原零结果，分开记录目录/表示/生成失败

- 原要求：不删坏样本、不更换原指标、不伪造实验结果；评测错误或源码与假设不一致时记录原因和影响。
- 实际发现：原筛选只保证对话至少两个正反馈可映射，原 process_conv 选择的最终 target 并不受此条件约束。45条中21个最终target不可映射、4个映射年份不同、44个target带年后缀；样本37规划finish_reason=length且无Map。HTTP200不代表每个推荐成功。
- 决定：原输入、全部分母、原 hit_judge 和正式 0/45 结果不变；增加独立离线诊断。剥离年份后的6个模糊命中是另一种目标表示，不能替代正式结果或宣称推荐模型提升；它们也没有记录Map ID的对应证明。
- 替代/后续：若后续修正映射/评分表示，必须另建协议、声明覆盖和排除数量、冻结独立评测，不能把当前已查看结果的样本事后筛成更好test。原论文canonical输入和模型ID仍待权威来源核验；不为此下载无明确用途的大包。
- 可比性：本轮可证明原执行链在A1条件下工作，但不支持论文质量复现或算法收益。未改变总体路线，未直接修改规范；建议见 `reports/spec_issues.md`，真实诊断由 `audit_a1_eval_failures.py` 从保存产物计算。

## ADR-014：隔离修复年份选择与反思历史丢失，保留原版评测

- 日期：2026-10-05；用户要求检查是否可修复。规范 §5.4 要求最小补丁、先回归、不得偷换模型/prompt/指标；原版对照必须保留。
- 实际发现：原 movie_map 对同名电影取 signed date_diff 的 idxmin，2007 年版本可误选 1957 年版本；原 Agent 反思递归未传 chat_history，原评测外部对话在第二次规划和 Critic 中丢失。正确 fixture 的先行回归产生 3 个真实断言失败；更早一次 fixture 缺工具 name/desc 的错误另存，不作为历史丢失证据。
- 原要求与阻碍：不能把改过映射/输入的结果算作原版，不能把缺资源语义伪装成运行成功；原筛选只保留有两个可映射正反馈的对话，并不保证最终 target 可达。
- 最小替代：在单独 `upstream_bugfixes.py` 中严格定位并执行 AST 补丁，日期比较加 abs，递归显式传 chat_history。两份 patch 展示精确差异；RecAI/、既有 compat 工作副本和真实 CLI 默认不改。新增 target_catalog_contract/require_reachable_targets 对拟开展的目录内新评测拒绝整批坏输入，不静默筛样本，不修改原 45 条。
- 验证：53 项完整回归通过；原 DemoSelector(fixed)、Critic、Agent 和 SDK MockTransport 完成四组原版/补丁×zero/fixed 对照，原版第二次规划/critic 无外部历史，补丁均保留，反思限次和 memory 保持。fixture 使用自造 gallery/buffer/tools/响应，不能当真实 App 推荐结果。固定示例来自上游已存在的 placeholder 文件；作者 canonical demo 等价性未证明。动态默认模型未缓存，避免额外下载；动态与真实功能恢复仍 NOT VERIFIED。
- 真实输入影响：45 条中补丁只改变 1 个映射，年份不一致从 4 到 3；严格目录契约有 22 个唯一标题/年份匹配、17 个缺标题、6 个缺年份。21 个原映射未知仍是未知，没有数据补造。训练 ID/矩阵语义和包许可没有新证明，不能用目录 exact match 替代它们。
- 可比性：补丁是 upstream_compat 的行为变化；后续启用须独立记录和评测，不能归为 upstream_pristine/A2。旧真实预测、全部 45 分母和三个 0/45 指标保持不变。本轮新请求/下载为 0，120 次授权仍已用 93 次，无台账重置、无路线变更。证据见 `reports/reproduction_repairs_20261005.md`。

## ADR-015：冻结目录身份映射，质量实验采用规范中的独立重建路径

- 日期：2026-10-05；用户明确要求暂不追论文条件，并授权依次核查目录/年份/别名、寻找旧训练映射、找不到则按规范建立统一 ID 和独立模型。本决定改变当前验收重点，不修改原规范或已有原版结果。
- 原要求：§5.2 要求目录、相似度和模型 ID 语义可核验；§5.5 在历史资源无法满足契约时允许 MovieLens 重建，保留上游 planner/toolbox，使用明确标注的适配器；T06 固定 MovieLens 1M，并禁止 holdout 参与训练映射。
- 新事实：仅获取官方 ML10M 元数据及条款，压缩包范围响应共 246440 字节，另 22 字节验证下载入口。9888 个原目录条目中 9883 个按标题/年份/类型唯一对应该版本官方记录，5 个歧义。12,189 条完整源元数据推导的别名在读取评测 target 之前冻结；45 条事后诊断为 28 个可解析、6 个年份冲突、11 个该来源未收录的标题。新增 6 个别名匹配不是模型命中指标或新冻结 test。
- 旧映射搜索：已检查原包 15 成员、固定源码与完整本地历史的 70 个 distinct InteRecAgent 路径、两版 movies notebook；它们都不保存 item_id_map。源码由过滤评分的 encounter order 编号，随后只导出新 ID 目录，缺确切原始评分版本/顺序不能恢复旧行号。官方目录身份匹配不等于 checkpoint/matrix 行标签；未凭行数不同宣称模型不可运行或所有分数错误。
- 决定：原 A1 功能运行、旧资源及正式三个 0/45 保留；推荐质量对照采用规范 B/upstream_rebuilt 的独立数据/模型路径，后续按 T06→T07→T08→T09，不强制重训旧 SASRec。旧资源的语义与独立许可继续 NOT VERIFIED，不把新 raw_movie_id 填进旧 embedding 行。
- T03 验收解释：完成的是资源审计、可追溯缺失记录和 A/B 路线决定，不是所有旧资源通过语义验收。因此 T03 可记 DONE 并解除 T06 的审计依赖；T04 原完整验收仍 BLOCKED，作者条件不再阻塞独立新实验。未实施或验收 T06/T07/训练。
- 替代与影响：若后续取得可核验的旧训练映射，可另开原资源对齐检查；本轮有界搜索不是“全网不存在”的证明。新实验不能与旧 45 条正式指标直接混算或宣称论文复现。元数据/映射和真实 target 仅本地，公开摘要不再分发 MovieLens 数据；GroupLens 元数据条款不替第三方原模型包授权。
- 环境处理：Python 证书链校验失败后使用 Windows 默认信任链验证 HTTPS，没有关闭 TLS。范围读取和 CRC/SHA 校验、请求/字节上限及重复门禁均保留；没有新付费调用。本轮测试和状态证据见 `reports/catalog_identity_20261005.md`。

## ADR-016：固定独立质量实验协议，不在看到结果后改指标

- 日期：2026-10-06。用户要求先搁置旧模型 ID 与原资源许可问题，转向优化；提出若必要可换评价指标判断提升。规范 §6.3、§7.1 已规定全局时间划分、统一全集候选、valid NDCG@10 调参与测试隔离。选择先完成 T06 数据和 T07 指标，再运行全部基线/改进模型；不按测试结果事后更换主指标。
- 官方稳定 MovieLens1M ZIP 的 MD5/CRC/SHA 验证通过；仅完整下载这一个 5,917,549 字节的官方数据包、README 与 MD5，并为现代环境固定 PyArrow 21.0.0。MovieLens 原文件、映射和逐行数据留在忽略目录，不公开再分发。电影标题按该包实际编码 iso-8859-1 解码；用户人口属性未作为模型特征。
- 全局 timestamp 切分在约80/10/10分位，边界上相同时间戳整体进入后窗。rating≥4 正反馈、训练期至少5个正反馈构成 warm 用户、候选电影来自训练期至少一次正反馈。验证/测试新用户和新电影保留为冷启动层；训练图、ID、历史、热门与将来的负采样只看 train。源 SHA、配置 SHA、每个数据文件 SHA 与边界/排除数保存在本地 manifest；公开仅汇总与哈希。
- 影响：旧 A1 三个0/45 是另一种文本命中和数据来源，不与新 MovieLens1M NDCG/Recall 混算。旧 checkpoint/矩阵 ID 与包许可仍 NOT VERIFIED、T04 保留 BLOCKED；新模型效果需 T07→T08/T09 后真实运行，当前 NOT EVALUATED。决策无新增 LLM 请求。证据见 `reports/t06_movielens_protocol_20261006.md`。

## ADR-017：T08 固定 CPU BPR 环境与单 seed 基线结果

- 日期：2026-10-06。规范 T08 要求先 fixture 过拟合/重载，再单 seed 全量，且只用 valid NDCG@10 选择模型；§7.1 最终配置仍需 3 seeds。当前设备无需 CUDA，为遵守用户尽量少下载资源的要求，选官方 `torch==2.0.1+cpu`，其 wheel 下载约166 MiB，依赖不需额外 218 MiB MKL 包；两套 Python3.11 环境可从同一缓存离线重建。未安装可选 NumPy 桥接，张量计算/测试通过，警告留在报告。
- 训练只读 T06 train、train_positive 与映射；负采样排除全部训练期已评分候选，未来标签不参与。Random 和 Popularity 与 BPR 同用 T07 cohort、候选和过滤；valid 选择 best epoch 1，之后首次读取 test。模型 checkpoint 与训练热门计数只保留本地，公开汇总值与 SHA。未修改上游、旧 A1 指标或冻结协议。
- 实际结果：seed 42 test NDCG@10 Random 0.014081、Popularity 0.195795、BPR-MF 0.192089。BPR-MF 未超过热门，不做提升宣称；不能据此事后换主指标。两环境各36项测试、checkpoint 重算通过。此依赖选择仅改变现代独立实验环境，不改变数据和模型比较口径。证据见 `reports/rec_baselines/t08_20261006.md` 与机器结果 JSON；T09 和最终3 seeds 待后续任务。

## ADR-018：T09 固定 LightGCN 图来源、层数选择与结果边界

- 日期：2026-10-06。规范 §7.1/T09 要求训练期二部图、`D^(-1/2)AD^(-1/2)` 稀疏传播、包含第0层的聚合、小图数值校验、层数实验与仅 valid 选择。参考 [LightGCN 论文](https://arxiv.org/abs/2002.02126)及作者 PyTorch 仓库固定提交 `947ca2b3b1d2d3545b114145710cb06c4e57b3d2` 的 model.py；只核验源码与版本，不克隆参考仓库。自行实现 T06 模型 ID 的对称 COO 图与均匀层聚合。
- 以已校验的 `train_positive.parquet` 462887 条边建图，训练期已评分排除规则沿用 T08；验证/测试交互只在独立审计和 evaluator 中读取。按预设配置比较 1/2/3 层，seed42、dim32、lr1e-3、epoch上限8；valid NDCG@10 选中3层第7 epoch 后才读 test。未以旧 checkpoint 的未知编号作新模型 ID。小图误差≤1e-6、真实图与 holdout 零交叉、checkpoint/ID/训练边哈希和重算均通过。
- 真实 test NDCG@10 LightGCN 0.194047，BPR-MF 0.192089，Popularity 0.195795；单 seed 结果不能宣称超过最强简单基线或统计显著提升。参考原论文数据、划分和旧 A1 命中不是本次指标。T09 不改变 T06/T07 协议，唯一新增的是独立 LightGCN 实现与训练产物；旧 T04 ID/许可仍 NOT VERIFIED，最终3 seeds 和 T10 适配待后续。证据见 `reports/rec_baselines/t09_20261006.md`。
- checkpoint 审计补充：首轮 checkpoint 只有配置哈希与维度，不满足 §7.3“包含模型配置”的完整要求；旧结果、三份模型和对应源码已移入本地忽略归档。补充完整配置入 checkpoint、加载时比对后用原 seed/超参数重跑。三层 valid 指标与 test 聚合结果逐项相同，无基于 test 的调参；正式 checkpoint SHA 更新，详见 T09 报告。
