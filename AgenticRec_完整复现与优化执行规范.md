# AgenticRec：完整复现、改进优化与验收执行规范

**项目二｜基于 Microsoft RecAI / InteRecAgent 的约束感知交互式推荐智能体**

文档版本：1.0 ｜ 核验日期：2026-10-01 ｜ 面向对象：编码 Agent 与项目负责人

> 本文件是实施规范，不是完成报告。上游事实、静态审计发现、待验证事项和新功能设计分别标记；所有实验结果目前均为未运行。优先阅读本文件与 AGENTS.md，再按 TASKS.yaml 的依赖执行。不得把“安装成功”“示例跑通”“方法重建”写成“完整复现论文指标”。

## 0. 项目执行摘要

### 0.1 最终交付是什么

在保留上游出处、原始快照和基线的前提下，交付一个能处理自然语言需求、利用真实推荐模型召回和排序、保持多轮偏好、严格执行可验证约束的推荐 Agent。重点不是再做一个 RAG，而是补齐推荐算法训练、用户建模、推荐评测和成熟代码二开能力。

**核心主线：上游核验 → 最小运行复现 → 无泄漏数据与推荐基线 → 上游工具适配 → 约束与偏好建模 → 有预算的计划执行与路由 → 公平实验 → 可复现发布。**

最终必须能回答：上游如何工作；自己修改了什么；为什么需要修改；收益来自推荐模型还是 Agent；哪些请求不需要 Agent；结果是否遵守用户约束；实验是否泄漏；失败时系统如何退回可解释的结果。

### 0.2 只承诺这一版范围

| 级别 | 内容 | 完成标准 |
|---|---|---|
| P0：投递版必做 | 单一电影领域；BPR-MF、LightGCN；上游计划链适配；多路召回；显式约束与偏好状态；规则路由；有限重规划；离线评测；CLI；测试与报告 | 冷启动、正常推荐、反馈修改、无解与故障场景有可复现证据 |
| P1：核心后增强 | SASRec；更丰富且有授权的元数据；从验证集学习路由；小型接口服务 | 必须提出独立假设和实验，不得阻塞 P0 |
| P2：本版不做 | LLM 微调 / RL；大规模用户模拟；多 Agent；MCP 大集成；互联网爬虫；K8s；平台级 UI | 不因热门名词扩展范围 |

BPR 是训练目标/方法，本文的具体基线名称为 **BPR-MF**。LightGCN 也可用 BPR loss 训练；两者不是互斥技术。SASRec 不列为强制，避免同时维护图推荐和序列推荐两条复杂训练线。[S12][S13]

### 0.3 对“复现”的四种命名

| 标记 | 含义 | 允许的表述 |
|---|---|---|
| A0 | 仅阅读代码与论文 | 静态审计完成 |
| A1 | 原资源、指定源码下跑通工具和 Agent，模型或依赖存在已记录差异 | 上游功能复现 / 兼容性复现 |
| A2 | 原数据、划分、模型、提示、指标和运行条件均可对齐，并有重复实验 | 指定实验的论文复现；必须注明具体实验，不泛称全部复现 |
| B | 原资源缺失，改用公开 MovieLens 数据重建并保留/适配原方法 | InteRecAgent 方法级复现及二次开发；不能与论文数字直接比较 |

**默认策略：先尝试 A1；资源或历史模型不可获得时，保留阻塞证据并走 B，不无限等待，也不伪造资源。** B 可以形成完整简历项目，但必须准确命名。

## 1. 上游信息与事实边界

### 1.1 权威入口与固定版本

- 上游仓库：https://github.com/microsoft/RecAI
- 主子项目：https://github.com/microsoft/RecAI/tree/main/InteRecAgent
- 本次核验快照：`0959ecb05b0794748426e73e6efc1b6b35ec433d`
- 固定版本入口：https://github.com/microsoft/RecAI/tree/0959ecb05b0794748426e73e6efc1b6b35ec433d/InteRecAgent
- 论文：https://arxiv.org/abs/2308.16505
- 上游给出的 TOIS 引用：https://doi.org/10.1145/3731446
- 已报告的数据问题：https://github.com/microsoft/RecAI/issues/110

以上 SHA 是本次读取 `main` 得到的基线，不代表 Agent 执行时仍是最新版本。不得自动追随新 `main` 覆盖基线；需要升级时单独建立 ADR 和对照实验。[S01][S02][S03][S22]

RecAI 是包含多个研究子项目的仓库；本项目只深入 InteRecAgent。上游是有论文与实现依据的研究系统，**不因组织名称、Star 或仓库有 tests 目录就认定为生产就绪**。本文件未验证全部数据下载、依赖安装、训练和端到端运行。

本次核验通过 GitHub 连接器读取公开源码；本地容器尝试 clone 时 DNS 解析失败。因此以下是源码审计，不是已在本机成功运行的复现报告。执行 Agent 必须在自己的运行环境重新建立证据。

### 1.2 已核实的上游能力

上游将推荐工具划分为 query、retrieval、ranking，使用预先准备的物品表、相似度资源和推荐模型。InteRecAgent 自身不负责训练/更新这些模型。已有 Plan First、可选反思、示例选择和历史压缩；这些不能冒充本项目新增创新。[S02]

| 已存在的文件 / 符号 | 本项目需要读懂什么 |
|---|---|
| `InteRecAgent/app.py` | 配置、物品库、CandidateBuffer、六类工具、Agent 的装配 |
| `llm4crs/environ_variables.py` | 导入时即读取领域 settings；资源路径如何解析 |
| `llm4crs/corups/base.py` / `BaseGallery` | 物品表、SQL、标题与 ID 转换、模糊匹配；`corups` 是上游真实拼写 |
| `llm4crs/buffer/base.py` | 候选集、相似度和轨迹如何在工具间传递；执行时补读完整文件 |
| `llm4crs/query/query_tool.py` | 信息查询；执行时补读 |
| `llm4crs/retrieval/sql_tool.py` | 候选过滤；执行时补读 |
| `llm4crs/retrieval/itemcf_tool.py` | 按种子物品读取预计算相似度，不是现成的语义检索工具 |
| `llm4crs/ranking/reco_model_tool.py` | UniRec 模型加载；popularity / similarity / preference 分支 |
| `llm4crs/agent_plan_first_openai.py` | Plan → ToolBox → 工具 → 回答 → 可选 Critic 的控制流 |
| `llm4crs/utils/open_ai.py` | SDK、超时、重试、token 统计 |
| `eval/one_turn_eval.py` | 输入 `context` 与 `target`；文本模糊命中评测 |
| `eval/user_simulator.py` | 多轮仿真入口；执行时补读，不假设其已满足新评测要求 |

这些路径来自实际源码与目录读取。[S04]—[S10]。`agent_plan_first_openai.py` 即使走非 LangChain 调用路径，仍导入 LangChain 的 `Tool`；**不能把 `--langchain 0` 理解成彻底无 LangChain 依赖**。[S06]

### 1.3 原执行链必须画明白

```text
app.py 装配
  → BaseGallery + CandidateBuffer
  → LookUp / HardFilter / SoftFilter / Ranking / Map 等工具
  → CRSAgentPlanFirstOpenAI.init_agent()
  → run({"input": ...})
  → plan_and_exe()
  → 解析模型文本中的计划
  → ToolBox.run()
  → 工具修改 CandidateBuffer
  → Map / LookUp 产生可供回答的内容
  → 总结与可选 Critic
  → 对话记忆更新
```

新 Agent 不能只读 README 就重写一套不相关系统。必须在 `docs/source_walkthrough.md` 中给出上述链路的实际文件、方法、输入、输出和一条执行轨迹。[S04][S06]

## 2. 已发现的风险与可形成个人贡献的切入点

### 2.1 静态审计清单

| 编号 | 观察到的事实 / 风险 | 必须完成的动作 |
|---|---|---|
| F01 | Issue #110 的提交者报告预制数据资源下载有问题；不是本次对两个链接的实测结论 | 小范围检查访问结果、授权和资源完整性；失败转 B 路径 |
| F02 | requirements 包含 LangChain 0.0.312、Gradio 3.40.1、PyTorch 1.12.1–1.13.1，同时 OpenAI SDK 只设下界 | 分离 legacy / dev 环境；锁定实际成功组合，不全局升级 |
| F03 | 主 SDK 包装使用 `OpenAI` 客户端，但旧评测基线仍出现 `openai.ChatCompletion.create` | 针对使用到的分支写兼容补丁和 mock 测试；禁止整包降到旧 SDK 破坏主链 |
| F04 | ToolBox 将工具计划列表转换为以工具名为键的字典 | 编写重复工具调用回归测试；改为有序 `PlanStep[]`，保留重复步骤 |
| F05 | 排序工具向模型传 `item_seq`、`item_seq_len`、`item_id` | BPR-MF/LightGCN 需要独立用户 ID 评分适配器；不能直接换 checkpoint |
| F06 | 部分 unwanted 处理是降低分数，不是先从候选中移除 | 验证候选不足时是否仍输出被排除物品；新链路先硬过滤、后排序 |
| F07 | app.py 在模块级创建共享 bot 与 CandidateBuffer | 并发使用前验证状态串扰；新服务按 session 隔离可变状态 |
| F08 | SimilarItemTool 直接用物品 ID 下标访问相似度矩阵，并排除第 0 列 | ID 映射、padding、矩阵形状必须联合校验 |
| F09 | 一轮评测以文本模糊匹配计算 hit；其 popularity 基线是加权抽样 | 保留为 `upstream_text_hit` 与 `upstream_weighted_pop`；不能改名成 NDCG 或确定性 Top-K 热门 |
| F10 | Gallery CSV 使用 `names=columns`；初始化会加载 gte-base | 验证表头、dtype 和本地模型缓存；不能把表头误当数据 |

来源：F01 [S03]；F02 [S05]；F03 [S07][S10]；F04 [S06]；F05/F06 [S08]；F07 [S04]；F08 [S09]；F09 [S10]；F10 [S11]。除了可由 Python 字典语义直接推导的 F04，风险是否在目标环境触发，均需测试确定。

### 2.2 只保留三个主要贡献主题

**贡献 A：推荐模型适配与可复现实验。** 自己训练 BPR-MF、LightGCN，实现一致的 ID、数据划分和评分接口；区分已知用户与冷启动；建立完整训练到推理链。

**贡献 B：约束感知、多轮偏好驱动的推荐。** 硬条件由确定性规则保证；显式反馈高于历史推断；未知字段不瞎补；冲突和无解时可澄清。

**贡献 C：保留重复动作的计划执行与成本路由。** 修复实际执行语义；简单请求走固定流程，复杂请求才启用有限规划；用同模型、同工具池、同评测集验证收益与成本。

这些是待实现设计，不是已经证明的创新。训练现成算法本身不是算法原创；修复重复调用首先是正确性贡献；路由是否改善效率需实验，不预设一定有效。

## 3. Agent 的权限、行为与停止规则

### 3.1 允许与禁止

允许：读取公开源码；在用户指定工作区 clone；创建本地分支、隔离环境、源码、测试和文档；在获授权的数据上做本地实验。

默认禁止：自动付费调用模型；租云 GPU；上传数据/权重；推送仓库；发 PR；删除用户文件；修改系统 Python/CUDA；关闭 TLS 验证；绕过网站登录/访问控制；把 API key、原始用户偏好、私有路径写进公开日志。

如果需要受限操作，将对应任务标记为 BLOCKED，写明所缺授权；可继续不依赖它的离线任务，不反复尝试。网络、数据或历史模型不可获得时，不得生成假的成功日志。

### 3.2 单次任务执行协议

读取 `STATUS.md` 和 `TASKS.yaml` → 选择依赖已满足的任务 → 先写/确认验收测试 → 最小实现 → 运行测试 → 保存原始日志与产物哈希 → 更新状态 → 记录 diff 与原理说明。

任务状态只有 `TODO / IN_PROGRESS / BLOCKED / DONE / SKIPPED`。DONE 必须附测试命令、退出码、日志位置、产物与 SHA；条件不满足的原资源复现记 BLOCKED 或说明性 SKIPPED，绝不能记 DONE。

每完成一个阶段，输出：完成内容、证据、未完成内容、已知风险、下一项依赖。不得仅凭模型自评或“应该可以”通过验收。代码须符合所在仓库已有规则；本文件不能覆盖更高优先级指令。

### 3.3 预算边界

默认 `allow_paid_api=false`、`api_request_cap=0`。先用 FakeLLM 完成单元与系统测试。用户明确授权后，单独配置提供商、model ID、总请求数、token 上限和预算单位；缺失时拒绝启动批量真实模型评测。

每一轮的路由解析、规划、输出修复、最终解释和重试都计入 LLM 消耗。SDK 自动重试与外层重试只能有一个统一控制者，避免预算乘法膨胀。费用未知时记录 `null` 和原因，不写 0。[S16][S17]

## 4. 仓库组织与版本隔离

### 4.1 推荐组织

```text
RecAI/                              # 保留 fork / clone 的上游历史
├── InteRecAgent/                    # 上游主子项目；仅最小兼容补丁
├── AgenticRec/                      # 本项目新增；以下路径均待实现
│   ├── pyproject.toml
│   ├── src/agenticrec/
│   │   ├── cli.py
│   │   ├── config.py
│   │   ├── data/                    # 数据、映射、split、manifest
│   │   ├── models/                  # popularity、BPR-MF、LightGCN
│   │   ├── adapters/                # upstream / scorer / LLM 适配
│   │   ├── retrieval/               # CF、content、fusion
│   │   ├── ranking/                 # rerank、硬约束
│   │   ├── agent/                   # plan、executor、router、state
│   │   ├── tools/                   # 有 schema 的推荐工具
│   │   ├── evaluation/              # 指标、协议、实验汇总
│   │   └── runtime/                 # deadline、trace、session
│   ├── configs/
│   ├── tests/{unit,integration,regression,fixtures}/
│   ├── scripts/
│   ├── docs/
│   └── reports/
├── reproduction/                    # upstream lock、compat diff、基线报告
└── LICENSE                          # 上游许可证必须保留
```

`data/`、`artifacts/`、模型缓存和私密 trace 不入 Git。公开仓库只保留合法的脚本、配置、摘要报告、自造 fixture 和下载说明。原始 MovieLens 及衍生文件、权重的公开发布先核验授权，不能把代码的 MIT License 套到数据上。[S01][S14]

### 4.2 已存在的 Git 命令：在干净工作区执行

```bash
# 不要在已有项目中重复 clone / 覆盖；先检查 pwd、目录和 git status。
git clone https://github.com/microsoft/RecAI.git RecAI
cd RecAI
git switch --create work/agenticrec 0959ecb05b0794748426e73e6efc1b6b35ec433d
git rev-parse HEAD
git status --short
```

需要保留完全原始工作树时，可创建本地 detached worktree；存在同名目录就停止，不覆盖。无需现在 fork 到远端或 push。

```bash
git worktree add --detach ../RecAI-pristine \
  0959ecb05b0794748426e73e6efc1b6b35ec433d
```

保存 `reproduction/upstream_lock.json`，至少包含 URL、commit、核验日期、读取文件列表、许可证和本地补丁 SHA。开发提交用小步、可回滚方式；不修改基线数据来让新方法显得更好。

## 5. 第一阶段：上游复现与兼容性修复

### 5.1 环境策略

**legacy 环境：** 上游 README 建议 Python 3.9；按其 requirements 建立隔离环境。此环境只用于复现，不对外公开服务。`torch` 与 CUDA、NumPy、huggingface-hub、sentence-transformers、OpenAI、UniRec 的实际兼容组合均待验证。[S02][S05]

**dev 环境：** 默认选择已有的 Python 3.11 解释器，使用独立环境；PyTorch、Pydantic、NumPy、pandas、pytest、sentence-transformers 等版本在首次成功安装与测试后锁定。不把本文件中的版本建议当作已测试 lockfile。

机器没有指定解释器、没有网络或没有 GPU时：记录约束；先做静态审计和 CPU fixture 测试。不得为跑通 legacy 自动更换用户系统 CUDA。

```bash
# 上游文档的起点命令，不保证在执行机器上一遍安装成功。
conda create -n interecagent-legacy python=3.9 -y
conda activate interecagent-legacy
cd RecAI/InteRecAgent
python -m pip install -r requirements.txt
python -m pip check
```

路径要按实际 cwd 调整。Agent 应把安装 stdout/stderr、OS、Python、包版本写入 environment report；不得导出包含密钥的完整环境变量。第一次成功后保存精确锁定文件，再重建新环境验证。

### 5.2 资源契约：不能只下载一个 CSV

`llm4crs/environ_variables.py` 在导入时读取 `resources/<DOMAIN>/settings.json`。缺失它时，甚至 `app.py --help` 也可能在参数处理前失败。先校验资源，再调用入口。[S04][S18]

settings 中需要的真实键为：

```json
{
  "GAME_INFO_FILE": "<物品表的相对路径>",
  "TABLE_COL_DESC_FILE": "<列说明 JSON 的相对路径>",
  "MODEL_CKPT_FILE": "<可由上游 UniRec 加载的 checkpoint>",
  "ITEM_SIM_FILE": "<预计算相似度 npy>",
  "USE_COLS": ["<实际列名>"],
  "CATEGORICAL_COLS": ["<实际分类列名>"]
}
```

上述只是键结构，不是可直接运行的资源文件。Agent 必须验证：文件存在、哈希、来源、格式、物品唯一 ID、模型的 item 数量、padding 约定、相似度矩阵行列顺序、必要列和类别值。

物品表至少有 `id` 和 `title`；热门排序会用到 `visited_num`。重建时 `visited_num` 只能来自训练期数据，不得统计验证/测试交互。CSV 读法包含 `names=columns`，需核实是否带表头；优先用显式 schema 的 parquet 并声明所需依赖。[S08][S11]

未获信任的 checkpoint 不加载到含密钥的环境；需先确认来源。不能用随机张量冒充预训练权重，更不能把 RecBole / 自写 PyTorch 权重改扩展名后交给 UniRec。

### 5.3 A 路径：原资源可得

先检查 README 的原资源入口和 `settings.json`，记录实际可访问性。Issue 报告不等于每个网络环境均无法访问，不绕过访问控制。[S02][S03]

依次执行：资源验证 → 无 LLM 工具冒烟 → 单次 LLM 连接检查 → CLI/原 UI 单轮 → 多轮 → 原评测。LLM 调用必须已有预算授权。

原入口模板如下；环境变量值由用户安全配置，不把真实 key 写进脚本：

```bash
# 在 RecAI/InteRecAgent 目录；资源与环境就绪后。
export DOMAIN=movie
export OPENAI_API_TYPE=open_ai
# OPENAI_API_KEY、OPENAI_API_BASE、LLM_MODEL 由安全环境注入。
: "${OPENAI_API_KEY:?需要已授权的密钥}"
: "${LLM_MODEL:?需要可用的模型标识}"
python app.py --engine "$LLM_MODEL" --bot_type chat \
  --plan_first 1 --langchain 0 --demo_mode zero \
  --enable_reflection 0 --enable_shorten 0
```

这里的 `LLM_MODEL` 是本地脚本变量，用来传 `--engine`，不是上游自行读取的配置。主调用链读取 `OPENAI_API_TYPE`；不要混淆 README 某处的 `API_TYPE`。[S04][S06][S07]

先关闭 demo、reflection、shortening 缩小问题面，再按原实验设置逐项恢复。换模型、关闭这些功能之后只是运行复现，不是论文复现。

原一轮评测最小输入是 `{"context": "对话文本", "target": "目标标题"}`，使用原数据时保持其语义。新造两条样例只能算 smoke test。[S10]

```bash
# 同一目录；EVAL_PATH 必须指向已校验的原评测数据。
: "${EVAL_PATH:?需要明确评测文件}"
export AGENT_ENGINE="$LLM_MODEL"
PYTHONPATH="$PWD" python eval/one_turn_eval.py \
  --agent recbot --data "$EVAL_PATH" --bot_type chat \
  --plan_first 1 --langchain 0 --demo_mode zero \
  --enable_reflection 0 --num_rec 5 \
  --save upstream_eval_output.jsonl
```

官方示例 shell 含占位路径；必须核对当前源码的实际参数。`one_turn_eval.py` 使用 `parse_known_args()`，未知参数可能被忽略，因此新增 wrapper 必须对拼错的参数直接报错。原评测 hit 是模糊文本命中，不得与新 ID-based Hit@K 混用。[S10]

### 5.4 必须采用最小补丁，而不是换个系统

按实际触发情况处理 SDK 的新旧接口、缺失依赖、路径和资源 dtype。每个补丁写：原始错误、最小复现、变更原因、测试、对基线语义的影响。

**兼容补丁不能同时偷偷加入新的推荐模型、改 prompt 或换评测指标。** 在报告中区分 `upstream_pristine`、`upstream_compat`、`upstream_rebuilt`、`ours`。

上游 Plan、Buffer、Tool 的抽象应通过适配器实际进入 baseline。不能最后只是复制一个微软 README，而代码变成完全独立的新项目。

### 5.5 B 路径：资源或历史模型缺失

登记阻塞原因，转入 MovieLens 重建。重建物品表、相似度和训练管线；保留原 planner / toolbox 的方法基线，换成明确标注的模型适配器。**不强制为了兼容 checkpoint 再训练一套历史模型**：先做可解释的 scorer 接口和方法级对照。

必要时将上游依赖放入独立进程，通过 JSON 请求/响应桥接。不能让旧 LangChain 的依赖冲突污染训练环境。实现桥接前先测最简单的受控脚本适配，避免过早做微服务。

阶段结束输出：`UPSTREAM_REPRODUCTION.md`、`UPSTREAM_DIFF.md`、环境报告、资源 manifest、运行/阻塞日志、源码讲解。未运行字段保留 `not_run`。

## 6. 第二阶段：数据、ID 与无泄漏实验协议

### 6.1 数据集固定为 MovieLens 1M

P0 使用官方稳定数据集，不使用会滚动更新的 latest 数据作为主 benchmark。MovieLens 1M 提供 ratings、movies、users；电影字段含 ID、标题、genres，标题中可解析年份。它**没有完整剧情、片长、价格、情绪标签**，不能直接支持“轻松、不压抑、两小时以内”这些硬事实筛选。[S14][S15]

官方入口：https://grouplens.org/datasets/movielens/1m/

P0 的硬条件只做：类型包含/排除、年份范围、明确排除的物品、已观看过滤、Top-K。不使用性别、邮编等人口属性；对话身份仅用匿名 ID。中文类型可用人工可审查的枚举映射，不擅自生成中文片名和剧情。

未知属性返回 `UNKNOWN_ATTRIBUTE`；询问用户是否接受仅按已知条件推荐。只有获得新数据来源和授权，并冻结字段 provenance 后，才开启片长/剧情等增强。

### 6.2 数据准备产物

`items.parquet`：`raw_item_id, item_id, title, genres, year, metadata_source`。

`interactions.parquet`：`raw_user_id, raw_item_id, rating, timestamp`，不覆盖原时间。

`id_map.json`：原 ID 与模型 ID 双向映射、padding 规则、schema 版本、哈希。

`split_manifest.json`：全局时间边界、过滤规则、用户/物品/交互数量、排除原因、SHA、随机种子。

`train/valid/test`：显式分开保存；Agent 服务只接触被授权的训练期历史和当前会话。评测 target 独立于 serving 输入。

### 6.3 默认实验协议：先简单、固定、可解释

这是本项目设计，不是上游论文的默认划分。所有模型必须使用同一协议。

1. 对原始交互按 timestamp 全局排序，选择约 80% / 10% / 10% 的时间窗口；相同时间戳不得切到两侧，将实际边界和比例写进 manifest。
2. `rating >= 4` 定义为正反馈，作为本项目的建模假设；不宣称它就是点击/购买。其余已观察评分不当作未知物品进行负采样。
3. 用户至少有 5 个训练期正反馈才进入 warm-user 模型训练。物品候选全集固定为训练期至少有一次正反馈的物品；映射只依赖训练数据。
4. 验证/测试中的新用户、新物品不悄悄丢弃：主表单独记录排除数量和覆盖率，冷启动另外分层报告。
5. 训练图、共现相似度、热门统计、用户 embedding、采样分布只能来自训练窗口。不得读验证/测试建立图或修复映射。
6. 本版不做 train+valid 重训；测试时也只使用训练期历史，便于各系统完全对齐。需要滚动推荐时建立另一个明确协议。
7. 主评测采用固定全集排序，剔除该用户训练期所有已评分物品，再评估测试正反馈。验证也剔除训练期已评分物品。
8. 排序按 score 降序、item_id 升序打破并列；随机模型固定 seed。候选不足如实返回，不复制物品填满。

### 6.4 必测的数据不变量

`max(train.timestamp) < min(valid.timestamp) <= max(valid.timestamp) < min(test.timestamp)`；相同时间边界按 manifest 调整。

训练与验证/测试交互键无交叉；图边全部能追溯到 train；负样本不属于该用户任一训练期已评分物品；padding ID 不出现在结果；模型维度、相似度矩阵、物品映射哈希一致；变更 split 后旧 checkpoint 必须拒绝加载。

未来正样本不能被用来排除训练负采样，否则训练获得了未来知识。离线隐式反馈负样本可能包含未来喜欢的物品，这是限制，需要说明，而不是偷看测试后“修正”。

## 7. 第三阶段：推荐算法与统一适配器

### 7.1 必做模型

| 模型 | 实现要求 | 需要保存的证据 |
|---|---|---|
| Random | 固定 seed、不放回、只选合法候选 | 可重放输出；不是主优化对象 |
| Popularity | 训练期正交互计数，稳定排序 | 训练统计快照；不要混同上游加权抽样 |
| BPR-MF | PyTorch 用户/物品 embedding，配对排序损失 | 配置、训练曲线、best checkpoint、指标、梯度测试 |
| LightGCN | 训练期二部图、归一化传播、层聚合、BPR 训练 | 稀疏图来源、传播验证、层数实验、模型与 ID 哈希 |

BPR-MF 评分 `s(u,i)=p_u·q_i`；目标可实现为 `mean(softplus(s(u,j)-s(u,i))) + λ·regularization`。明确 λ 的归一化约定，不把不同 batch 下的正则混为一谈。[S12]

LightGCN 采用用户—物品二部图传播：`E(l+1)=D^(-1/2) A D^(-1/2) E(l)`，聚合包括第 0 层；本版默认均匀平均。使用稀疏计算；不要构建百万级稠密图。模型细节以论文及核验后的参考实现为准。[S13]

初始搜索范围是执行建议，不是性能承诺：embedding 维度 32/64，learning rate 1e-3，LightGCN 层数 1/2/3；先 fixture，再单 seed 的 ML1M，再对最终配置跑 3 seeds。仅用 valid NDCG@10 选择配置和 early stopping；测试不参与调参。

### 7.2 不要求手写一切，但必须真正验证

BPR-MF 建议自行实现，以便掌握 loss、采样和优化。LightGCN 可以参考作者实现或 RecBole，但必须标明来源、固定版本，并在小图上验证传播、评分和 top-k。RecBole 是可选交叉验证工具，不增加另一套独立数据划分。[S19][S20]

可学习产物：写出一个 batch 的 user/item 维度、正负样本、loss、梯度流向；画出一个小图的两层传播结果；说明图推荐为何不能直接解决匿名新用户。

### 7.3 核心接口契约：先写协议再接工具

以下为待实现接口，不是上游现成 API：

```python
class Recommender:
    def recommend(self, user_id, candidate_ids, k): ...
    def score(self, user_id, candidate_ids): ...

class SessionScorer:
    def score_from_seeds(self, liked_ids, disliked_ids, candidate_ids): ...
```

`score` 的输出顺序必须与输入 `candidate_ids` 对齐，返回有限实数；未知用户不能偷偷映射到 0 号用户。冷启动使用 content / item-item / popularity 或明确的 seed pooling 策略，并在 trace 标注 `profile_source=cold_start`。

上游 `prefer` 标题列表并不天然等于训练用户 ID。适配器应先解析显式身份和历史授权，再决定：已知用户调用 BPR/LightGCN；匿名用户调用独立 SessionScorer。历史偏好 pooling 是本项目策略，不宣称是原生 LightGCN 推理。[S08]

模型 checkpoint 包含 state_dict、模型配置、id_map_hash、split_hash、训练 seed、代码 SHA、依赖版本；加载时全部校验。无可用模型时回退热门并明确记录，不能继续输出随机 embedding 的结果。

## 8. 第四阶段：召回、约束与偏好状态

### 8.1 固定推荐流程先于 Agent

```text
结构化请求 + 匿名用户身份 / 训练期历史
  → 候选全集与硬条件
  → CF 召回 + 内容召回 + 必要的热门回退
  → 基于 ID 的去重与融合
  → 个性化评分 / 冷启动评分
  → 再次校验硬约束
  → Top-K ID + 可核验的解释依据
```

第一版内容特征仅使用 title、genres、year。可先使用 TF-IDF，后接冻结的 embedding 模型；语义库记录 model revision 与输入模板。仅有这些元数据时不得宣传能理解完整剧情。其作用与 LocalDoc 的知识检索区分：这里要验证内容信号与协同信号如何互补。

初始每路召回 100、候选并集最多 200、输出 5 或 10，均为待调参数。使用 RRF 作为无需跨模型原分数校准的基线：`score(i)=Σ 1/(60+rank_r(i))`，未出现在某路中该路贡献为 0。不要直接相加不同量纲的向量相似度和 BPR 分数。

P0 排序可用用户模型分数与融合排名的固定规则；是否增加 LLM reranker 不是必需。所有可比较系统应使用相同候选空间、预算和字段，避免给新方法更多数据。

### 8.2 显式约束是执行规则，不是 prompt 愿望

`Constraints` 至少包含 `include_genres, exclude_genres, year_min, year_max, excluded_item_ids, exclude_seen, k`。类型条件采用枚举和参数化查询，不执行任意 LLM SQL。

硬过滤必须前置，并在最终输出再校验。候选不足时返回更少结果并解释；无解时返回 `NO_FEASIBLE_ITEMS` 或澄清请求。**不得为了填满 Top-K 自动放宽年份、不喜欢的物品或排除类型。**

同一请求同时要求包含和排除同一类型，属于显式冲突；需要澄清，不用“最新条件覆盖”悄悄解决。同一字段的后续明确修改可以覆盖旧值，但必须留下事件记录。

### 8.3 偏好状态的数据结构

```text
PreferenceState
  session_id / user_id(optional) / version
  hard_constraints
  liked_item_ids / disliked_item_ids
  soft_preferences: [{value, source_turn, confidence}]
  history_item_ids: 仅授权训练期历史
  evidence: 每个状态字段对应哪条用户输入
  unresolved_fields / contradictions
```

优先级：本轮明确硬条件 > 本轮明确偏好 > 历史明确偏好 > 模型推断。明确“不喜欢某片”不自动推断为“不喜欢该片全部类型”。显式新要求与历史不同是正常更新，不视为用户错误。

对每次反馈使用事件化更新 `PreferencePatch`；校验通过后一次性提交 state。工具重试不能重复追加反馈。会话与长期用户历史分开存放；清空 session 不删除训练数据。

### 8.4 必须做好的五类例子

1. 只有类型/年份：走确定性过滤与排序，不做多轮规划。
2. 已知用户请求个性化：调自己的模型，记录候选和最终评分。
3. “类似这个片，但排除恐怖”：识别种子，内容/协同召回，再执行硬排除。
4. 第二轮改成另一年份范围：修改状态并重新推荐，不沿用过期缓存。
5. 问片长、剧情情绪或无解条件：说明字段缺失或约束冲突，不靠 LLM 编造。

## 9. 第五阶段：计划执行、路由与最小 Harness

### 9.1 计划格式

新计划必须是有序列表，不得转为以 tool_name 为唯一键的字典。模型先输出结构化 `Plan`；不支持原生结构化输出的提供商走文本 JSON + 严格校验，最多一次有预算的修复。

```json
{
  "plan_version": "1",
  "steps": [
    {"step_id": "s1", "tool": "content_recall", "args": {"query": "genre:Comedy"}},
    {"step_id": "s2", "tool": "content_recall", "args": {"query": "genre:Adventure"}},
    {"step_id": "s3", "tool": "rank_candidates", "args": {"k": 5}}
  ]
}
```

这是协议示例，不是真实推荐结果。两个 content_recall 必须都执行。step_id 唯一；tool 必须严格匹配注册表；schema 拒绝未知参数、越界 k、空 query、任意文件路径、SQL 或 Python 执行。

### 9.2 推荐工具契约

| 工具 | 输入重点 | 输出 / 状态语义 |
|---|---|---|
| `get_user_history` | 已授权身份，截止时间 | 允许读取的历史；不含评测 target |
| `lookup_items` | 明确 item_ids | 物品元数据及来源 |
| `content_recall` | query、合法条件、top_k | 候选 ID、排名、来源 |
| `collaborative_recall` | user_id、top_k | 模型候选；冷启动返回结构化状态 |
| `filter_candidates` | candidate_version、constraints | 过滤后的新版本，不原地污染历史版本 |
| `rank_candidates` | candidate_version、profile_version、k | 对齐 ID 的评分与排序 |
| `ask_clarification` | 冲突/缺失字段 | 面向用户的澄清，终止本轮工具循环 |

工具统一返回 `ToolResult(ok, data, error_code, retryable, elapsed_ms, provenance)`。LLM 看少量摘要；完整候选、模型得分与日志存结构化 trace，不能把整个目录塞进 prompt。

### 9.3 路由而不是所有请求都调用 Agent

路由输出枚举：`DIRECT / PERSONALIZED / AGENT / CLARIFY`。

第一版用可审计规则：已有结构化条件且无歧义 → DIRECT；身份与偏好明确 → PERSONALIZED；需要组合种子/反馈/缺失信息检查 → AGENT；冲突、缺少字段或不支持要求 → CLARIFY。

自由文本即使走固定流程，也可能需要一次意图抽取 LLM 调用；不能把它的成本从“非 Agent baseline”里扣掉。只有规则可解析或直接提交结构化请求，才可以宣称 0 LLM。

复杂路径采用有限状态机：`PARSE → PLAN → EXECUTE → VALIDATE → RESPOND`，允许 `VALIDATE → REPLAN` 最多一次。验证依据是字段存在、硬条件、候选数量和工具结果，不是只让 LLM 说“我觉得足够”。

### 9.4 执行预算与错误分类

起始配置建议：单轮最多 6 次工具调用、4 次 LLM 请求（含解析/解释/修复）、1 次重规划。超时阈值在目标硬件实测后冻结；不预设“复杂问题 8 秒完成”。

配置要同时包含每次请求 timeout 和整轮 deadline。同步 CPU 计算放独立 worker；给 Future 设置 timeout 不代表底层工作已停止。验证取消、并发限流和资源回收后，才能称有可靠超时控制。

401/403/参数错误：不重试，报告配置问题；429/部分 5xx/连接失败：在总 deadline 内按统一策略有限重试；无候选：不是网络故障，不重复请求同一工具；模型不可用：固定推荐流程降级。SDK 自带重试须显式配置，计数可审计。[S16][S17]

缓存键包括数据/模型版本、user_id、profile_version、constraints hash、query、top_k。不得仅按 query 缓存跨用户推荐；反馈更新后旧缓存失效。

### 9.5 安全与会话隔离

检索元数据和模型输出都视为数据，不是执行指令。对标题中嵌入“忽略规则”等文本保持原样/转义，不执行它。工具白名单、参数化 SQL、只读数据目录、禁止任意 shell。

为每个 session 单独创建 state 与候选存储；无状态模型权重可共享。对两用户交替调用、并发反馈、超时后重试、清空历史写回归测试。不要因为本机 CLI 只有一个用户就宣称已解决多租户。

## 10. 第六阶段：评测、对照与统计

### 10.1 必须分成两套任务，不能混成一张表

**推荐模型评测：** 给定训练期用户历史，预测测试窗口的正反馈。比较 Random、Popularity、BPR-MF、LightGCN。指标来自固定全集排序，不调用 LLM。

**交互推荐评测：** 给定可见需求、反馈与元数据，在若干轮内输出合法推荐或正确澄清。比较固定流程、上游方法基线、全部走 Agent、新路由 Agent。评测 target 和隐藏偏好只在 evaluator，不能出现在 planner、工具返回、例子检索或用户历史中。

不拿上游模糊文本 hit、MovieLens 全集 NDCG、LLM 用户模拟成功率直接比较高低。这三种数字回答不同问题。[S10]

### 10.2 模型指标定义

每用户固定相关集合 R，输出唯一且合法的排名列表 L：

`Recall@K = |L[:K] ∩ R| / |R|`；`Hit@K = 1[至少命中一个]`；`MRR@K = 第一个相关结果排名的倒数，无命中为 0`。

二值 NDCG：`DCG@K = Σ rel(rank)/log2(rank+1)`，用同一用户理想排序的 IDCG 归一。无相关物品的用户不进入主均值，必须报告数量；不得在不同模型之间变动用户集合。

模型有 0 个候选也计入失败/零分；不能从平均延迟与成功率分母删除。先用手工可算的 fixture 验证完全命中、全错、短列表、并列、多个相关项；再跑真实数据。

### 10.3 交互评测集

建议 P0 建立 150 条 development episode（调试/配置），150 条冻结 test episode；其中至少各有 30 条跨轮反馈样例。数量可因预算缩减，但必须预先记录实际规模，不能为追求好结果删 bad case。

按 user / scenario family 分组隔离开发与测试；不能把同一句模板换个电影名就当完全独立的新任务。类型包含：显式筛选、个性化、种子相似、排除/反馈、约束冲突、未知属性、冷启动、工具错误。

评测分两层：结构化输入的组件评测（排除语言解析干扰）；真实文本端到端评测（所有系统包括相同的解析成本）。先用规则用户反馈与人工核验样本，LLM simulator 为可选扩展。

若使用 simulator，其可见隐藏目标与 Agent 可见输入必须是两个对象；不能共享对话状态。不要让同一个 LLM 同时生成题目、做推荐、再作为唯一裁判。模拟器结果不等于真实用户满意度。

### 10.4 指标不能被“全拒答”刷高

| 指标 | 明确定义与防刷规则 |
|---|---|
| Item Validity | 输出中存在于当前合法目录的 ID 比例；无效 ID 计错，不能事后静默删掉 |
| Constraint Precision | 输出物品中满足全部硬条件的比例；可行推荐任务输出空列表记 0 |
| Fill@K | 合法输出数 / 请求 K；最多为 1；候选不足必须同时报告 |
| Strict Success | 可行任务：合法、全部硬条件满足、达到预设数量要求，且命中隐藏接受集合/满足预注册目标；不可行任务：正确澄清或报告无解 |
| Preference Update Accuracy | 反馈后的结构化状态与已标注 patch 是否一致 |
| Turns / Tool Calls | 所有尝试和失败均计入；达到上限不算成功 |
| Latency | p50/p95、端到端墙钟；含解析、解释、重试和失败；注明并发与硬件 |
| Tokens / Requests | 输入输出、修复、解释、重试分别记录；未知 usage 不伪造 |
| Fallback / Abstain Rate | 降级、澄清、无解分开，和成功率共同报告 |

严格成功的隐藏接受集合不能简单等于“目标电影名已出现在输入里”；基于规则构造的任务必须标注为合成 benchmark，不能宣传线上效果。

### 10.5 必须保留的系统对照

`U0 upstream_compat`：原资源可用时按原协议报告；若不可用标 not_run。

`U1 upstream_rebuilt`：原 planner/tool 设计配上重建资源，在本项目协议下运行；所有改动公开。不能将其命名为论文原版结果。

`F fixed_pipeline`：与新系统相同的推荐工具、模型、约束处理和数据，不做动态规划。这个强基线不可省略。

`A always_agent`：与新系统相同 LLM、工具池和预算，所有任务进入 Agent。

`O ours_router`：相同条件下的路由系统。仅 O 改成更强的 LLM 不算路由提升。

`LLM-only` 可作为诊断，但其没有同等工具信息，不能作为唯一主对照。上游基线缺失时，仍可完成 F/A/O 对照，但必须明确研究结论的范围。

### 10.6 预注册的两个核心假设

H1：确定性硬约束和显式反馈状态能减少非法推荐，且不会通过大量空输出虚增指标。联看 Constraint Precision、Fill@K、Strict Success、Abstain。H1 对照应是去掉约束/偏好模块的消融或相应上游分支；F 本身已含同样的硬约束，不能把共有能力包装成 O 对 F 的新增收益。

H2：规则路由在任务成功率基本保持的情况下减少请求和延迟。以开发集选阈值，冻结后测试；建议将允许的成功率下降界设为 2 个百分点，但它是项目决策阈值，不是已证明的结果。只有配对差值置信区间的下界高于 -0.02，才可按此界讨论非劣；只看点估计接近不足以证明。

消融至少包含：去掉用户模型只留内容；去掉内容只留协同；去掉显式偏好状态；全部走 Agent；去掉重规划。硬过滤的移除实验仅在离线沙盒，不能把不安全版本作为默认服务。

每个模型最终配置跑 3 个 seeds；Agent 使用固定测试集，按 seed/运行次序做配对比较。报告均值、标准差和按用户/episode 配对 bootstrap 的 95% 区间。样本过少时写“证据不足”，不写“显著领先”。失败与负收益必须保留。

## 11. 工程测试、故障注入与回归清单

以下测试路径是 Agent 需要创建的目标，不是上游已存在测试：

| 测试组 | 最小验收内容 |
|---|---|
| `test_data_contract.py` | 时间边界、ID 往返、缺失列、原始文件 hash、schema |
| `test_no_leakage.py` | 图/热门/负采样只读 train；target 不进入 serving 输入 |
| `test_metrics.py` | 手算 Recall/Hit/MRR/NDCG；空、短、重复、无效结果 |
| `test_bpr.py` | 评分形状、梯度非零、小样本可过拟合、checkpoint 重载一致 |
| `test_lightgcn.py` | 小图稀疏/稠密传播一致；padding 隔离；训练图边来源 |
| `test_model_adapter.py` | user-id 与 item-seq 接口分离、冷启动、ID/model hash 拒载 |
| `test_constraints.py` | 排除项绝不因候选不足返回；冲突澄清；unknown 属性 |
| `test_plan_execution.py` | 同工具重复调用不丢失；保持顺序；非法参数/未知工具失败 |
| `test_preference_state.py` | 多轮覆盖、证据归属、否定反馈、幂等更新、缓存失效 |
| `test_runtime.py` | 401不重试、429有界重试、总deadline、预算停止、统计含失败 |
| `test_session_isolation.py` | 两用户交叉与并发请求不串状态、不串缓存 |
| `test_end_to_end.py` | FakeLLM 下成功、无解、故障、重规划、澄清的固定轨迹 |

先 fail 再 fix 的测试要保存原始证据，尤其是重复工具计划与排除物品两项。测试通过数必须从测试结果生成，不在报告中手填。覆盖率可作为辅助，不设随意的“90% 就可靠”门槛。

故障注入使用本地假工具/假服务，不对外部服务制造真实流量。超时、异常、无候选和模型返回坏 JSON 必须是可重复 fixture。真实模型测试单独标记并要求显式预算。

## 12. 统一命令契约与可复现产物

### 12.1 新 CLI 的目标接口

**以下是新增待实现的命令，不是下载上游后就能运行。** Agent 应先实现 `pyproject.toml` 和 CLI，并使未知参数报错；完成后再在文档中标成可用。

```bash
# 均在 RecAI/AgenticRec 中；先在独立 dev 环境安装本项目。
python -m pip install -e '.[dev]'
python -m agenticrec.cli doctor --offline
python -m agenticrec.cli data prepare --config configs/data.yaml
python -m agenticrec.cli data validate --config configs/data.yaml
python -m agenticrec.cli train --config configs/bpr.yaml --seed 42
python -m agenticrec.cli train --config configs/lightgcn.yaml --seed 42
python -m agenticrec.cli eval-rec --config configs/eval_rec.yaml
python -m agenticrec.cli chat --config configs/agent.yaml --llm fake
python -m agenticrec.cli eval-agent --config configs/eval_agent.yaml --llm fake
python -m agenticrec.cli benchmark --config configs/benchmark.yaml --dry-run
python -m agenticrec.cli report --run-dir artifacts/runs/<run_id>
python -m pytest tests/unit tests/integration tests/regression
```

`data prepare` 不得默认发布数据；`--dry-run` 不得触发 LLM 请求、训练或大文件下载。`doctor --offline` 只检查，不打印密钥；检查缺失资源、权限、CUDA/CPU 和预算，并区分 WARN/BLOCKED。

CLI 最终必须支持 fixture 模式，用自造小目录和 FakeLLM 一条命令跑通，不要求用户先下载真实 MovieLens、模型或配置密钥。

### 12.2 每次实验输出

```text
artifacts/runs/<run_id>/
  config.resolved.yaml
  environment.json
  provenance.json             # code/data/model/prompt/lock hash
  dataset_summary.json
  metrics.json
  predictions.jsonl           # 仅本地；按许可决定是否可公开
  traces.jsonl                # 脱敏；工具/预算/路由与解释证据
  stdout.log / stderr.log
  report.md
```

`metrics.json` 的 not_run / failed / completed 明确区分；没有运行的指标为 null。保存失败数、样本数、过滤数、种子与测量环境。公开报告由原始产物生成，不能手动编辑提升比例。

性能测试分别报告冷启动与预热后、单请求与有限并发；把模型加载时间、训练时间与在线延迟分开。单机离线吞吐不写成线上 QPS，离线 NDCG 不写成 CTR/CVR 提升。

## 13. 分阶段验收与任务调度

### G0：审计与路线选择

已锁定 SHA、读懂源码、记录资源可得性和依赖风险；明确 A1/A2/B，不允许“资源缺失但完整复现”。输出 T00—T03 的证据；T02 环境若阻塞可不阻塞 B 的现代开发。

### G1：原方法基线 + 数据基础

可用原资源时完成 A1 冒烟与原评测；否则 B 方法基线有可追踪上游来源。MovieLens 数据契约、split、ID、指标 fixture 通过；不能先训模型后补数据协议。

### G2：推荐系统独立成立

Random、Popularity、BPR-MF、LightGCN 在同一协议下有真实结果；冷启动路径、候选、硬约束和固定流程可离线使用。此时即使没有 LLM，也不是空壳。

### G3：Agent 完整可测

重复计划步骤正确、偏好更新可追踪、简单与复杂请求可区分、重规划与预算有界、session 隔离；FakeLLM 的所有关键轨迹通过。真实 LLM 未获授权则明确尚未完成 live 验收。

### G4：可投递发布

冻结评测、配对对照、消融和失败分析完成；真实 LLM 与真实数据结果具备或明确标注限制；独立环境按 README 可重建；个人改动清单、技术报告和演示可用。

**不要求所有指标同时提升。** 但至少一个主贡献要有可验证价值：正确性 bug 修复、约束可靠性、模型适配能力或成功率—成本取舍。没有收益的模块应删除/保持实验性，而不是增加话术。

任务详细依赖、输入、产出与验收见本文件附录任务卡和 `TASKS.yaml`；两者由同一任务表生成。全局完成度不能按“生成了多少代码文件”计算。

## 14. 故障与阻塞处置表

| 阻塞 | 正确处理 | 禁止处理 |
|---|---|---|
| GitHub/模型下载不可达 | 保存错误；可继续已获得源码的审计或离线 fixture；标记缺失 | 宣称已 clone / 已下载；关闭 TLS 校验 |
| 原资源不可用 | 报告 A 路径阻塞，走 B | 自造资源后仍标论文复现 |
| 旧包解析失败 | 隔离环境、逐项定位、保存兼容 diff；新环境独立推进 | 在用户全局环境随意降包 |
| 无 GPU | fixture/小批 CPU 验证；测内存；缩小 batch | 宣称完成全量 GPU 训练 |
| 只有匿名请求 | cold-start scorer / 内容 / 热门 | 随机给用户一个已有训练 ID |
| 查询需要缺失字段 | UNKNOWN_ATTRIBUTE + 澄清 | LLM 生成片长/价格当真值 |
| 新方法比固定流程差 | 保存负结果，分析路由/解析/召回/排序瓶颈 | 删除难题、只换新方法模型或偷调测试 |
| 无法完成真实 LLM 评测 | 保留 FakeLLM 工程验收，live 标 not_run | 把 fixture 结果当端到端效果 |
| PR 未合并 | 写“已提交（链接）”或仅本地贡献 | 写“贡献被官方采纳” |

## 15. 面试、简历与原创边界

### 15.1 README 必须明确三栏

**Upstream：** 微软 InteRecAgent 的框架、原有功能、原论文和许可证。

**My changes：** 具体新增/修改文件、推荐训练链、适配器、重复计划修复、约束和偏好状态、路由与评测。

**Evidence：** 每一项贡献对应实验/测试/故障案例；未完成条目不出现在 Features 的已完成列表。

项目名建议：`AgenticRec — Constraint-Aware Interactive Recommendation on InteRecAgent`。不声称官方微软项目、生产级推荐平台或 SOTA。

### 15.2 必须能够脱离 Agent 讲清楚的内容

| 面试问题 | 应对应的项目证据 |
|---|---|
| 原版做了什么？你的改动是什么？ | 固定上游 SHA、UPSTREAM_DIFF、逐模块来源 |
| 为什么不是另一个向量检索 Demo？ | 模型训练、图结构、ID、负采样、评测集 |
| LightGCN 如何处理匿名用户？ | 明确其限制及 cold-start adapter，不混接口 |
| 推荐实验如何避免泄漏？ | 全局时间 split、图边/热门 provenance、测试断言 |
| 为什么需要 Agent？ | F/A/O 配对实验与复杂场景轨迹 |
| 多轮反馈如何生效？ | PreferencePatch、state version、候选差异 |
| 约束为何不能只放 prompt？ | 过滤测试、候选不足与无解案例 |
| 如何证明优化有效？ | 消融、置信区间、负结果、成本与延迟 |
| 如何保证基线公平？ | 相同数据、模型、工具、预算与解析成本 |
| 有线上业务指标吗？ | 如实区分离线实验、模拟交互与真实线上数据 |

项目不能包办大厂所有笔试/面试；算法题、机器学习基础、Python、数据库与并发原理仍需独立准备。两个项目的分工是：LocalDoc 讲检索与 Harness；本项目重点讲推荐模型、用户/候选建模与二开实验。

### 15.3 简历模板：只有验收后才能填

“基于 Microsoft InteRecAgent 完成〔功能复现/方法级重建，按事实选择〕，构建 MovieLens 1M 的时间划分、ID 映射和 BPR-MF/LightGCN 训练—推理链，完成统一全集推荐评测。”

“设计约束感知的多路推荐工具与可追踪偏好状态，修复重复工具计划覆盖问题，支持明确排除、反馈更新、冷启动和无解澄清；对应〔实际测试/指标〕。”

“在固定 LLM、工具与数据条件下比较固定流程、全部 Agent 与路由系统，以〔实际样本量〕开展消融；在〔具体任务〕取得〔真实效果与成本变化〕，公开复现脚本与失败分析。”

占位内容不是简历事实；不填虚构百分比，不把原上游能力全部写成自研。

## 16. 首次交给 Agent 的执行方式

先将本执行包放入目标工作区的 `execution_pack/`，不要覆盖现有仓库指令。让 Agent 先读 `README_START_HERE.md`、`AGENTS.md`、本规范、`TASKS.yaml`。

默认第一批为 **T00、T01、T03**：核对本地状态、固定源码、画执行链、检查资源契约；同时可以做不安装依赖的兼容性分析。确认工作区和路线后，再完成 legacy/dev 环境及训练任务。

第一批报告只需要：实际仓库 SHA、源码流程、资源状态、选 A/B 的原因、要改的最小文件集和下一任务。不允许第一轮就生成几千行新系统并声称完成。

后续用 `prompts/CONTINUE_AGENT.txt` 恢复上下文；任务完成与否以真实产物和状态文件为准，而不是上一轮 Agent 的口头结论。

## 17. 资料与核验索引

本文件中 `[Sxx]` 为来源编号。源代码事实来自固定 SHA，设计与验收阈值是本项目规范，不是来源原文结论。外部资料核验日期均为 2026-10-01；执行时若变化，记录差异，不静默替换基线。

**[S01] Microsoft RecAI：项目与许可证**  
https://github.com/microsoft/RecAI  
核验用途：官方仓库；MIT 仅适用于对应代码，不等于数据授权。

**[S02] InteRecAgent README**  
https://github.com/microsoft/RecAI/blob/0959ecb05b0794748426e73e6efc1b6b35ec433d/InteRecAgent/README.md  
核验用途：工具分层、环境与资源、可选功能、论文引用。

**[S03] Issue #110：资源下载问题**  
https://github.com/microsoft/RecAI/issues/110  
核验用途：2026-04-27 的用户报告；不是本执行环境的访问测试。

**[S04] app.py**  
https://github.com/microsoft/RecAI/blob/0959ecb05b0794748426e73e6efc1b6b35ec433d/InteRecAgent/app.py  
核验用途：模块装配、全局 bot/buffer、真实参数。

**[S05] requirements.txt**  
https://github.com/microsoft/RecAI/blob/0959ecb05b0794748426e73e6efc1b6b35ec433d/InteRecAgent/requirements.txt  
核验用途：固定包与范围约束，尚未证明兼容。

**[S06] Plan-first Agent / ToolBox**  
https://github.com/microsoft/RecAI/blob/0959ecb05b0794748426e73e6efc1b6b35ec433d/InteRecAgent/llm4crs/agent_plan_first_openai.py  
核验用途：控制流、字典化计划、示例/记忆/反思。

**[S07] OpenAICall SDK 包装**  
https://github.com/microsoft/RecAI/blob/0959ecb05b0794748426e73e6efc1b6b35ec433d/InteRecAgent/llm4crs/utils/open_ai.py  
核验用途：客户端调用、重试与统计。

**[S08] RecModelTool**  
https://github.com/microsoft/RecAI/blob/0959ecb05b0794748426e73e6efc1b6b35ec433d/InteRecAgent/llm4crs/ranking/reco_model_tool.py  
核验用途：UniRec、序列输入、排除与排序分支。

**[S09] SimilarItemTool**  
https://github.com/microsoft/RecAI/blob/0959ecb05b0794748426e73e6efc1b6b35ec433d/InteRecAgent/llm4crs/retrieval/itemcf_tool.py  
核验用途：预计算相似度与 ID 下标约定。

**[S10] One-turn evaluation**  
https://github.com/microsoft/RecAI/blob/0959ecb05b0794748426e73e6efc1b6b35ec433d/InteRecAgent/eval/one_turn_eval.py  
核验用途：context/target、模糊 hit、加权热门抽样及旧 API 分支。

**[S11] BaseGallery**  
https://github.com/microsoft/RecAI/blob/0959ecb05b0794748426e73e6efc1b6b35ec433d/InteRecAgent/llm4crs/corups/base.py  
核验用途：数据加载、CSV names、模糊匹配模型。

**[S12] BPR: Bayesian Personalized Ranking from Implicit Feedback**  
https://arxiv.org/abs/1205.2618  
核验用途：推荐配对目标的原始论文。

**[S13] LightGCN 论文**  
https://arxiv.org/abs/2002.02126  
核验用途：图传播与层聚合方法。

**[S14] MovieLens 1M README 与数据条款**  
https://files.grouplens.org/datasets/movielens/ml-1m-README.txt  
核验用途：字段、编码、使用/再分发约束。

**[S15] MovieLens 1M 官方下载页**  
https://grouplens.org/datasets/movielens/1m/  
核验用途：稳定数据集入口。

**[S16] 官方 OpenAI Python SDK**  
https://github.com/openai/openai-python  
核验用途：SDK 重试、timeout 与使用说明；以运行时锁定版本为准。

**[S17] OpenAI 官方错误说明**  
https://developers.openai.com/api/docs/guides/error-codes  
核验用途：错误分类参考，不要求本项目使用特定供应商。

**[S18] 领域资源配置**  
https://github.com/microsoft/RecAI/blob/0959ecb05b0794748426e73e6efc1b6b35ec433d/InteRecAgent/llm4crs/environ_variables.py  
核验用途：导入即加载 settings.json，六个资源/列配置键。

**[S19] LightGCN PyTorch 参考实现**  
https://github.com/gusye1234/LightGCN-PyTorch  
核验用途：可选模型实现参照；执行时固定其 commit。

**[S20] RecBole**  
https://github.com/RUCAIBox/RecBole  
核验用途：可选模型/指标交叉验证；不得另起不一致的数据协议。

**[S21] UniRec**  
https://github.com/microsoft/UniRec  
核验用途：原模型工具依赖及可选训练/导出参考。

**[S22] RecAI main 快照元数据**  
https://api.github.com/repos/microsoft/RecAI/branches/main  
核验用途：本次读取取得 SHA；分支引用会变化，正文基线固定。

**[S23] 上游 eval 示例 shell**  
https://github.com/microsoft/RecAI/blob/0959ecb05b0794748426e73e6efc1b6b35ec433d/InteRecAgent/eval/eval_single_turn.sh  
核验用途：包含占位路径；实际参数以 Python 文件为准。

**[S24] InteRecAgent 论文**  
https://arxiv.org/abs/2308.16505  
核验用途：论文入口；复现实验需执行 Agent 完整阅读对应协议。

## 附录 A. Agent 任务卡（与 TASKS.yaml 一致）

任务依赖优先于编号；T18须先于T16完成测试冻结。T02/T04为条件复现任务，阻塞不等于完成；T22—T24为可选。所有任务初始状态为 TODO。

### T00｜锁定工作区、源码与出处

阶段：G0；类型：core；依赖：无。

输入：用户指定工作区；EXECUTION_SPEC.md 第1/3/4章。

执行：检查 cwd、已有文件、git status；不覆盖脏工作区。 取得指定上游 SHA，记录实际来源、许可证和允许的操作；若网络不可用，标记阻塞，不伪造 clone。

交付：`reproduction/upstream_lock.json`；`AgenticRec/docs/STATUS.md`。

验收：源码 SHA 与规范一致，或有单独记录的升级决定。 已有用户改动未被覆盖；无 push、无密钥入库。

测试/检查：`git rev-parse HEAD`；`git status --short`。

本人必须讲清：能区分 upstream、fork、新模块与兼容补丁。

### T01｜逐文件走读与静态风险复核

阶段：G0；类型：core；依赖：T00。

输入：固定版本 InteRecAgent；规范第1/2章。

执行：补读 buffer/query/sql/memory/critic/eval 文件。 从 app.py 到候选、模型输入和输出画出调用链。 逐条确认 F01—F10；问题先分类为事实、静态推断或运行待验证。

交付：`AgenticRec/docs/source_walkthrough.md`；`AgenticRec/docs/static_audit.md`。

验收：每个模块有输入/输出与 state 变化说明。 原有 reflection、memory、demo selector 不被写成新增。

测试/检查：`人工/Agent 对照源码检查每个路径和符号；保留行号或函数名`。

本人必须讲清：不看 README 能复述一次请求的调用顺序。

### T02｜隔离 legacy 环境与最小兼容补丁

阶段：G0；类型：conditional；依赖：T00, T01。

条件：原环境可获得；失败可阻塞 A 路径但不能假装完成，也不阻塞独立 B 路径。

输入：上游 requirements；本机解释器/包与权限。

执行：仅在隔离环境尝试上游安装并保存错误；记录具体兼容包。 只修已触发的 SDK/依赖问题，所有补丁独立留 diff。 不得因主链新版 OpenAI 客户端而直接全局降级 SDK。

交付：`reproduction/environment_legacy.json`；`reproduction/legacy_compat.patch`；`reproduction/requirements.legacy.lock.txt`。

验收：环境可重建；或存在具体 BLOCKED 日志和 B 路径决定。 兼容补丁没有混入推荐增强。

测试/检查：`python -m pip check`；`仅运行已验证不会自动调 API 的 import/adapter mock 测试`。

本人必须讲清：能解释 old/new SDK 接口为何不能靠盲目降包解决。

### T03｜核验资源并决定 A/B 路径

阶段：G0；类型：core；依赖：T00, T01。

输入：README 资源入口；settings.json 契约；Issue #110。

执行：检查当前网络可访问性、下载授权、文件清单和 checkpoint 来源。 校验键、ID、矩阵、表头、模型兼容；没有资源则登记缺失。 决定 A1/A2/B；选择 B 不声称已复现论文。

交付：`reproduction/resource_manifest.json`；`reproduction/route_decision.md`。

验收：访问状态与报告来源区分清楚。 A/B 路线、原因、缺失资源和数据合规记录齐全。

测试/检查：`静态验证资源清单；下载/加载前按权限门禁执行`。

本人必须讲清：为什么 CSV、相似度矩阵和 checkpoint 必须共享 ID 映射？

### T04｜原资源下的工具、Agent 与原评测复现

阶段：G1；类型：conditional；依赖：T02, T03。

条件：仅 A 路径且原资源/授权齐备；否则 BLOCKED/SKIPPED 并写原因。

输入：已验证原资源；legacy 环境；明确模型与预算授权。

执行：先无 LLM 工具冒烟，再单轮、多轮。 按原指标运行原数据，保存模型/提示/参数差异。 恢复需要的示例和反思设置；缺原论文条件只能记 A1。

交付：`reproduction/UPSTREAM_REPRODUCTION.md`；`reproduction/native_runs/`。

验收：原始输出、配置和数据哈希可追溯；未运行项为 not_run。 原文模糊命中不标为 ID NDCG。

测试/检查：`规范5.3的原 app.py 与 one_turn_eval.py 命令（前置完成后）`。

本人必须讲清：区分运行复现、方法复现和论文实验复现。

### T05｜现代开发环境、协议与离线测试骨架

阶段：G1；类型：core；依赖：T00, T01。

输入：规范第4/7/9/12章；configs/experiment_contract.example.yaml。

执行：新增 AgenticRec 包、CLI、锁文件和基础 schema。 实现 doctor --offline、FakeLLM、预算门禁和 fixture；不依赖旧包启动新训练。

交付：`AgenticRec/pyproject.toml`；`AgenticRec/src/agenticrec/`；`AgenticRec/tests/`；`AgenticRec/requirements.dev.lock.txt`。

验收：无 key、无网络也能加载新包和运行 fixture。 未知配置/参数报错；默认 0 付费请求。

测试/检查：`python -m agenticrec.cli doctor --offline`；`python -m pytest tests/unit/test_config.py`。

本人必须讲清：解释依赖隔离和配置 schema 的职责。

### T06｜MovieLens 数据、时间划分和 ID 契约

阶段：G1；类型：core；依赖：T03, T05。

输入：官方 MovieLens 1M 与条款；规范第6章。

执行：记录授权与下载哈希，检查编码和字段，不重分发原文件。 按全局时间拆分与训练期过滤，保存 warm/cold 分层。 在数据阶段保留 dev/test 用户与场景分组，防止后续评测泄漏。

交付：`data/processed/`；`artifacts/data_manifest.json`；`AgenticRec/configs/data.yaml`。

验收：训练图、热门、映射不读取 holdout。 所有排除数量、ID 和时间边界有记录。 缺失剧情/片长不生成假值。

测试/检查：`python -m pytest tests/unit/test_data_contract.py tests/regression/test_no_leakage.py`。

本人必须讲清：说清楚“未观察”与“明确不喜欢”不同。

### T07｜先验证指标，再跑大实验

阶段：G1；类型：core；依赖：T05, T06。

输入：冻结 split；手算 fixture；规范第10章。

执行：实现 ID-based Recall/Hit/MRR/NDCG 和稳定 top-k。 覆盖空、短、重复、并列、无效 ID、多相关项；定义 cohort 与分母。

交付：`AgenticRec/src/agenticrec/evaluation/metrics.py`；`AgenticRec/tests/unit/test_metrics.py`。

验收：手算结果与实现一致。 失败和空返回不会从系统评测分母消失。

测试/检查：`python -m pytest tests/unit/test_metrics.py`。

本人必须讲清：能手算一个用户的 NDCG@5。

### T08｜Random、Popularity 与 BPR-MF

阶段：G2；类型：core；依赖：T06, T07。

输入：train/valid 数据；模型配置与指标。

执行：训练统计生成热门，自己实现 BPR-MF 与采样。 先小样本过拟合与重载一致，再单 seed 全量；仅 valid 选模型。

交付：`AgenticRec/src/agenticrec/models/bpr.py`；`AgenticRec/configs/bpr.yaml`；`artifacts/models/bpr/`；`reports/rec_baselines/`。

验收：有真实训练日志，梯度和输出维度正确。 negative sampler 不读取未来，也不采训练期已评分物品。 是否超过热门如实报告，不伪造提升。

测试/检查：`python -m pytest tests/unit/test_bpr.py`；`python -m agenticrec.cli train --config configs/bpr.yaml --seed 42`。

本人必须讲清：解释 BPR loss 与 user/item embedding 的更新。

### T09｜LightGCN 训练和小图校验

阶段：G2；类型：core；依赖：T08。

输入：训练期正反馈图；BPR-MF 数据协议；论文与参考代码。

执行：实现/适配稀疏归一化传播、0层在内聚合和 BPR 训练。 对小图验证稀疏/稠密结果，记录参考代码与版本。

交付：`AgenticRec/src/agenticrec/models/lightgcn.py`；`AgenticRec/configs/lightgcn.yaml`；`artifacts/models/lightgcn/`。

验收：传播数值校验通过（float32建议误差≤1e-5，按设备记录）。 图无 holdout 边；checkpoint 与 ID/hash 一致。

测试/检查：`python -m pytest tests/unit/test_lightgcn.py`；`python -m agenticrec.cli train --config configs/lightgcn.yaml --seed 42`。

本人必须讲清：能解释层数、过平滑风险与为何不靠用户文本生成训练 ID。

### T10｜模型适配与上游方法基线

阶段：G2；类型：core；依赖：T01, T05, T06, T08, T09。

输入：上游 Plan/Tool/Buffer；独立模型与资源；A/B 决策。

执行：实现已知用户 scorer 与匿名 seed scorer 的分离接口。 接入原工具/计划方法；legacy 可用时可进程隔离，不可用时做最小移植并逐项标注。 形成 U1 upstream_rebuilt，不能冒称 U0。

交付：`AgenticRec/src/agenticrec/adapters/`；`AgenticRec/docs/UPSTREAM_DIFF.md`；`reports/upstream_rebuilt/`。

验收：相同 candidate_ids 的评分对齐；未知用户有明确 fallback。 能展示至少一条源自上游方法的端到端轨迹。 上游来源与改动一一对应。

测试/检查：`python -m pytest tests/integration/test_model_adapter.py tests/integration/test_upstream_bridge.py`。

本人必须讲清：解释 item_seq 与 user_id 两类模型的区别。

### T11｜多路候选融合与硬约束

阶段：G2；类型：core；依赖：T10。

输入：模型 scorer；物品元数据；约束 schema。

执行：实现内容/协同/热门候选、ID 去重、RRF 基线和排序前后硬校验。 再现候选不足时低分屏蔽仍可能返回 unwanted 的情形，采用真正过滤。

交付：`AgenticRec/src/agenticrec/retrieval/`；`AgenticRec/src/agenticrec/ranking/`。

验收：排除物品在任意 k 和候选不足场景均不返回。 无解/缺失字段结构化返回；不放宽硬条件。

测试/检查：`python -m pytest tests/unit/test_constraints.py tests/unit/test_fusion.py`。

本人必须讲清：解释召回上限、RRF 与模型原分数不可直接相加。

### T12｜完整固定推荐流程

阶段：G2；类型：core；依赖：T11。

输入：候选与约束工具；用户/冷启动 scorer。

执行：实现无规划固定流程，并使 CLI 可单独推荐。 输出推荐 ID、元数据证据、各路来源及评分，建立强 F baseline。

交付：`AgenticRec/src/agenticrec/pipeline.py`；`reports/fixed_pipeline/`。

验收：无 LLM 结构化请求可工作；自然语言解析单独计费。 成功、匿名、排除、无解四场景可复现。

测试/检查：`python -m pytest tests/integration/test_fixed_pipeline.py`。

本人必须讲清：说明哪些问题已经不需要 Agent。

### T13｜统一 LLM 适配、schema 与预算

阶段：G3；类型：core；依赖：T05, T12。

输入：FakeLLM；供应商配置（允许为空）；规范第3/9章。

执行：实现 chat adapter 与严格响应校验；缺少字段不得默默接受。 统一超时、SDK 重试与预算计数；默认真实调用关闭。

交付：`AgenticRec/src/agenticrec/adapters/llm.py`；`AgenticRec/src/agenticrec/runtime/`。

验收：每次尝试计数；401不重试；unknown usage 不记0。 没有授权的 live 测试被明确阻止而非偷偷运行。

测试/检查：`python -m pytest tests/unit/test_llm_adapter.py tests/unit/test_budget.py`。

本人必须讲清：说明每请求 timeout 与整轮 deadline 的区别。

### T14｜有序计划执行器与上游问题回归

阶段：G3；类型：core；依赖：T10, T13。

输入：F04 源码与独立最小例子；工具 schema。

执行：先写包含重复工具名的失败测试，再实现 PlanStep 列表。 执行严格白名单，保留步骤顺序、重复调用、step_id 和 trace。

交付：`AgenticRec/src/agenticrec/agent/executor.py`；`AgenticRec/tests/regression/test_plan_execution.py`。

验收：同工具两次调用两次执行，原参数均保留。 未知工具、重复 step_id 与坏参数拒绝执行。

测试/检查：`python -m pytest tests/regression/test_plan_execution.py`。

本人必须讲清：为什么 dict 保序仍不能保留重复 key？

### T15｜多轮偏好状态与证据更新

阶段：G3；类型：core；依赖：T12, T13。

输入：约束与用户状态协议；人工标注 feedback fixture。

执行：实现 PreferencePatch 与带 provenance 的状态。 处理显式覆盖、否定反馈、矛盾澄清、缓存版本与幂等。

交付：`AgenticRec/src/agenticrec/agent/state.py`；`AgenticRec/tests/unit/test_preference_state.py`。

验收：明确反馈优先，模型推断不能盖过用户硬条件。 一次否定不被错误扩展为全类型否定。

测试/检查：`python -m pytest tests/unit/test_preference_state.py`。

本人必须讲清：区别 session memory 与训练历史。

### T16｜规则路由与有限 Agent 闭环

阶段：G3；类型：core；依赖：T14, T15, T18。

输入：强固定流程；已冻结评测协议（不读 test目标）。

执行：实现 DIRECT/PERSONALIZED/AGENT/CLARIFY。 只在必要时规划；工具结果校验后最多一次重规划。 路由阈值只在 dev 集选择，解析/解释成本纳入统计。

交付：`AgenticRec/src/agenticrec/agent/router.py`；`AgenticRec/src/agenticrec/agent/loop.py`。

验收：FakeLLM 的成功/重规划/澄清/预算耗尽轨迹可重放。 不把所有请求都走 Agent 作为唯一方案。

测试/检查：`python -m pytest tests/integration/test_end_to_end.py`。

本人必须讲清：解释为何要比较 fixed_pipeline 与 always_agent。

### T17｜故障注入、安全和会话隔离

阶段：G3；类型：core；依赖：T16。

输入：Agent/runtime；并发和恶意元数据 fixture。

执行：校验同名用户请求/异步会话/反馈/缓存不串扰。 注入坏 JSON、超时、429、无候选、prompt injection，测试降级与停止。

交付：`AgenticRec/tests/regression/`；`reports/reliability/`。

验收：没有越权工具执行、跨 session 状态泄漏。 失败不会挂死、无限重试或绕过硬约束。

测试/检查：`python -m pytest tests/regression/test_runtime.py tests/regression/test_session_isolation.py`。

本人必须讲清：同步任务 timeout 后为何还可能继续占资源？

### T18｜冻结交互评测集与评分器

阶段：G2/G3；类型：core；依赖：T07, T12, T15。

输入：data split 与分组名单；规范第10章。

执行：建立开发集和冻结 test；公开/隐藏字段分开序列化。 实现约束、数量、成功/澄清、token/latency 的完整分母。 保留至少一个人工核验子集，默认不启 LLM simulator。

交付：`data/eval_private/`；`artifacts/eval_manifest.json`；`AgenticRec/src/agenticrec/evaluation/episodes.py`。

验收：test target 不进入可见 prompt、demo 检索或工具历史。 全空输出不能刷高可行任务成功率。 在 T16 调参前冻结 test。

测试/检查：`python -m pytest tests/regression/test_eval_leakage.py tests/unit/test_episode_metrics.py`。

本人必须讲清：为什么模拟用户满意度不是线上指标？

### T19｜真实基线、消融与配对统计

阶段：G4；类型：core；依赖：T17, T18, T10。

输入：冻结模型/LLM/数据/工具/预算；用户授权的真实调用预算。

执行：跑模型3 seeds；系统 U1/F/A/O 使用同 LLM与工具，U0视资源可得性。 做规定消融、配对CI与失败分类；输出两套评测表。 无授权时仅 fake 工程结果，live 状态不可完成。

交付：`artifacts/runs/`；`AgenticRec/reports/benchmark.md`；`AgenticRec/reports/ablations.md`；`AgenticRec/reports/failure_analysis.md`。

验收：表格可由原始预测重新生成；无手填指标。 含负收益、失败与分层覆盖率；各方法条件一致。 置信度不够时不声称显著提升。

测试/检查：`python -m agenticrec.cli benchmark --config configs/benchmark.yaml --dry-run`；`预算授权后运行同一命令去掉 --dry-run；先核验总请求数`。

本人必须讲清：指出收益来自模型、约束还是路由，并展示证据。

### T20｜独立环境复现与发布材料

阶段：G4；类型：core；依赖：T19。

输入：代码/lockfile/数据准备脚本/实验报告。

执行：在新环境按 README 运行 fixture 与已授权真实流程。 补 CLI demo、来源、许可、改动与局限；不发布受限数据、key与私密 trace。

交付：`AgenticRec/README.md`；`AgenticRec/docs/REPRODUCE.md`；`AgenticRec/docs/MY_CONTRIBUTIONS.md`；`AgenticRec/docs/KNOWN_LIMITATIONS.md`。

验收：第三方按文档知道哪些命令存在、数据如何合法取得。 不宣称保证面试、生产级或论文全部复现。

测试/检查：`python -m pytest tests/unit tests/integration tests/regression`；`干净环境完整重建日志`。

本人必须讲清：能按上游/自己/引用实现三类说明原创边界。

### T21｜技术报告与简历事实核对

阶段：G4；类型：core；依赖：T20。

输入：真实报告与贡献清单。

执行：准备5分钟项目叙述和逐模块深挖问答。 逐条把简历表述映射到代码、测试、指标，删除未完成承诺。

交付：`AgenticRec/docs/INTERVIEW_GUIDE.md`；`AgenticRec/docs/RESUME_FACTS.md`。

验收：每一条可在仓库/报告中核验。 原版功能与新贡献分开；离线不写线上指标。

测试/检查：`贡献-证据矩阵逐条审核`。

本人必须讲清：不依赖 Agent 解释任意一个关键模块。

### T22｜可选：序列推荐 SASRec

阶段：P1；类型：optional；依赖：T20。

条件：P0完成且项目负责人明确启用。

输入：明确序列建模假设与额外资源预算。

执行：建立按时间的序列输入和因果 mask；共享既有 split。 仅比较有足够历史的用户分层，报告额外成本。

交付：`可选 SASRec adapter、配置与实验`。

验收：不修改测试窗口提升结果；真实贡献明确。

测试/检查：`序列因果性/未来信息隔离测试`；`与现有模型统一协议对照`。

本人必须讲清：解释序列推荐与图协同的适用差异。

### T23｜可选：学习式成本路由

阶段：P1；类型：optional；依赖：T20, T18。

条件：P0完成，数据足够且明确授权。

输入：足够的独立训练/dev 路由样本；预注册非劣阈值。

执行：学习小型路由器，不训练大模型。 与规则路由对照，测试成本含路由本身。

交付：`可选 router model 与实验`。

验收：复杂度带来可测收益才保留；否则维持规则版本。

测试/检查：`路由训练/测试隔离`；`配对成功率与成本比较`。

本人必须讲清：解释成功率-成本边界及样本偏差。

### T24｜可选：向上游贡献最小修复

阶段：P1；类型：optional；依赖：T21。

条件：只有明确授权才能执行远端动作。

输入：独立bug证据；上游贡献规则；显式远端提交授权。

执行：将可独立复现的正确性/文档修复拆为最小 diff。 先形成本地 patch 与测试；只有明确授权才提交远端 PR。

交付：`上游补丁和测试；若已授权则附真实 PR链接`。

验收：不得把未提交/未合并写成被微软采纳。 按实际状态陈述贡献。

测试/检查：`原始失败与修复后通过对照`。

本人必须讲清：解释bug修复边界，不夹带整个新项目。
