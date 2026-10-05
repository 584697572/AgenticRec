# AgenticRec

基于 Microsoft RecAI / InteRecAgent 的固定版本复现与推荐算法研究项目。当前已验证原 App 的一次真实 LLM 单轮调用，算法改进与完整评测仍在实施中。

## 当前进度

| 内容 | 已验证范围 | 尚未完成 |
| --- | --- | --- |
| 原版复现 | 原资源加载、原工具与 App 离线联调；DeepSeek 真实单轮通过，Filter → Ranking → Map 返回 3 个满足本轮目录约束的结果 | 多轮验证、原评测、checkpoint 与目录 ID 的权威语义对应 |
| 开发基础 | 独立 Python 3.11.14 环境、配置校验、评分协议、FakeLLM、离线 CLI；两个环境各 15 项基础测试通过 | 推荐训练、完整 Agent 与 benchmark |
| 运行配置 | 本地 `.env` 读取、受限真实调用、日志与原始证据分离；32 项离线回归通过 | 批量实验与算法收益均为 NOT EVALUATED |

一次真实单轮通过不等于完整论文复现。当前保留 A1 原资源功能复现路线；资源许可与语义限制、多轮及评测前置详见 [STATUS.md](STATUS.md) 和 [任务依赖](TASKS.yaml)。

## 来源、个人改动与证据

| Upstream | My changes | Evidence |
| --- | --- | --- |
| Microsoft InteRecAgent 的 planner、工具、候选 buffer、原推荐模型与提示 | 固定版本审计、隔离环境、资源完整性检查和兼容运行脚本 | [上游版本锚点](reproduction/upstream_lock.json)、[原版复现记录](reproduction/UPSTREAM_REPRODUCTION.md) |
| 原 App 的工具执行链 | 受限调用入口、本地 Key 配置、无真实 Key 的模型工作进程及离线回归 | [真实单轮报告](reports/live_original_app_20261005.md)、[Key 配置报告](reports/local_key_setup_20261005.md) |
| 原 ReDial notebook 预处理 | 整数年份兼容适配、固定派生输入及重复运行验证 | [决策记录](DECISIONS.md)、[阶段报告](reports/progress_20261005.md) |

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
