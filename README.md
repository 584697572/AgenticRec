# AgenticRec

基于 Microsoft RecAI / InteRecAgent 的固定版本复现与推荐算法研究项目。当前已验证原 App 真实单轮、多轮及 45 条 A1 派生输入的原评测执行链，算法改进与论文条件对齐仍在实施中。

当前优先级是可信 ID 与独立质量实验。[目录核查](reports/catalog_identity_20261005.md)以官方元数据确认 9883 个唯一目录身份、保留 5 个歧义。旧模型行标签仍未知，原 A1 证据保留。独立质量路线按规范 B/upstream_rebuilt 推进；T06 的[MovieLens1M 数据协议](reports/t06_movielens_protocol_20261006.md)已冻结，官方 1,000,209 条评分全局时间拆分和训练期 ID 通过检查。下一任务 T07 指标手算与实现；新模型尚未训练，效果 NOT EVALUATED。

## 当前进度

| 内容 | 已验证范围 | 尚未完成 |
| --- | --- | --- |
| 原版复现 | 原资源、工具、真实单轮和两轮条件修改通过；原 recbot、random、加权 popularity 各完成 45 条评测，正式模糊 hit 均为 0/45 | 权威 ID 语义、目录/target 覆盖、canonical 作者子集及 demo/reflection 原实验条件 |
| 开发基础 | 独立 Python 3.11.14 环境、配置校验、评分协议、FakeLLM、离线 CLI；两个环境各 15 项基础测试通过 | 推荐训练、完整 Agent 与 benchmark |
| 运行配置 | 本地 `.env`、持久请求槽、金额检查与私密 trace 分离；原版侧64项回归及新包20项测试通过，旧原指标记录保留 | 算法收益与完整 benchmark 均为 NOT EVALUATED |

本轮原执行链已跑通，正式零结果及目录覆盖、输出截断、指标表示问题均保留在[多轮与评测报告](reports/a1_multiturn_eval_20261005.md)。当前保留 A1 路线；论文 A2 尚未验证，详见 [STATUS.md](STATUS.md) 和 [任务依赖](TASKS.yaml)。

后续[检查与修复](reports/reproduction_repairs_20261005.md)验证了两项独立行为补丁：纠正同名电影年份选择、保留反思重跑的外部历史。53 项完整回归通过；固定示例与反思仅完成离线原组件联调。原版真实结果不改写；补丁未接入既有真实运行入口，收益尚未评测。

## 来源、个人改动与证据

| Upstream | My changes | Evidence |
| --- | --- | --- |
| Microsoft InteRecAgent 的 planner、工具、候选 buffer、原推荐模型与提示 | 固定版本审计、隔离环境、资源完整性检查和兼容运行脚本 | [上游版本锚点](reproduction/upstream_lock.json)、[原版复现记录](reproduction/UPSTREAM_REPRODUCTION.md) |
| 原 App 与原评测执行链 | 受限调用入口、本地 Key 配置、无真实 Key 的模型工作进程、原指标复算和失败诊断 | [多轮与原评测报告](reports/a1_multiturn_eval_20261005.md)、[真实单轮报告](reports/live_original_app_20261005.md) |
| 原 ReDial notebook 预处理 | 整数年份兼容适配、固定派生输入及重复运行验证 | [决策记录](DECISIONS.md)、[阶段报告](reports/progress_20261005.md) |
| 原同名电影映射及反思流程 | 显式可选的两行正确性修复、严格目录可达性预检 | [补丁说明](reproduction/patches/README.md)、[修复报告](reports/reproduction_repairs_20261005.md) |
| 原目录与训练编号假设 | 官方元数据身份/别名交叉核查、完整来源冻结、旧映射有界搜索和独立质量路线 | [目录核查](reports/catalog_identity_20261005.md)、[路线决定](reproduction/quality_route_decision_20261005.json) |

上游仓库：[microsoft/RecAI](https://github.com/microsoft/RecAI)，固定 commit：`0959ecb05b0794748426e73e6efc1b6b35ec433d`。本项目不是微软官方项目；上游代码许可保留于 [LICENSE.txt](LICENSE.txt)，数据和模型的许可分别核验。

## 开始使用

```powershell
git clone https://github.com/584697572/AgenticRec.git
cd AgenticRec
```

先阅读 [执行规范](AgenticRec_完整复现与优化执行规范.md) 和 [当前状态](STATUS.md)。现代开发环境见 [开发包说明](AgenticRec/README.md)，原版 legacy 环境、资源获取及兼容步骤见 [复现说明](reproduction/README.md)。上游 `RecAI/` 是单独取得并固定版本的 checkout，不随本仓库一起提交。

在已按开发包说明建立环境的工作区，可执行以下离线检查；无需 API Key：

```powershell
& .\AgenticRec\.venv-dev\Scripts\python.exe -m agenticrec.cli doctor --offline --config AgenticRec/configs/experiment_contract.example.yaml
& .\AgenticRec\.venv-dev\Scripts\python.exe -m agenticrec.cli fixture
& .\AgenticRec\.venv-dev\Scripts\python.exe -m pytest AgenticRec/tests/unit/test_config.py -q
```

真实 LLM 配置和调用范围见 [LIVE_LLM_GUIDE.md](reproduction/LIVE_LLM_GUIDE.md)。Key 只保存在本机根目录 `.env`，该文件被 Git 忽略；已有运行额度和原始证据不可删除后重用。

## 仓库内容

- `AgenticRec/`：本项目开发包、配置、基础测试和详细状态。
- `reproduction/`：版本锁、兼容脚本、回归测试及可公开的验证摘要。
- `reports/`：阶段报告、失败记录与限制说明。
- `STATUS.md`、`TASKS.yaml`、`DECISIONS.md`：进度、依赖与重大决策。

数据、权重、模型缓存、虚拟环境、真实 Key 和私密 trace 不入 Git。公开摘要保留验证范围；未执行的实验不填写指标或提升比例。
