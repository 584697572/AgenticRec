# AgenticRec

电影推荐 Agent 项目，只分为两个部分：**微软原版的尽可能接近复现**和**个人优化版本**。

| 部分 | 内容 | 入口 |
|---|---|---|
| `original/` 微软原版复现 | 固定 RecAI / InteRecAgent 源码、隔离旧环境、兼容补丁、原资源和原 App 的运行证据 | [原版说明](original/README.md) |
| `optimized/` 个人优化 | BPR / LightGCN、硬约束、多轮反馈、有序工具执行、Agent 路由、融合及全部优化实验 | [优化版说明](optimized/README.md) |

本项目不是微软官方项目。原版主要功能流程已运行；旧资源 ID 对应、独立许可及论文评测条件仍未完全验证。个人优化使用独立的数据映射和训练模型，不能混称微软原版。

## 目录与获取

```text
AgenticRec/
├── original/
│   ├── RecAI/             # 微软源码子模块，固定 commit
│   ├── reproduction/      # 旧环境、兼容副本、运行脚本及测试
│   ├── reports/           # 原版复现报告
│   └── data/              # 原预制资源，仅本地，不公开分发
├── optimized/
│   ├── AgenticRec/        # 个人实现：src、configs、scripts、tests、docs
│   ├── reports/           # 推荐模型、Agent、消融和失败分析
│   ├── history/           # 历史状态、决策、原规范及迁移记录
│   ├── data/              # 独立实验数据，仅本地
│   └── artifacts/         # 权重、私密预测与实验原始输出，仅本地
├── workspace.py           # 创建本地兼容入口；不下载、不请求 LLM
├── STATUS.md
└── DECISIONS.md
```

```powershell
git clone https://github.com/584697572/AgenticRec.git
cd AgenticRec
git submodule update --init original/RecAI
python workspace.py prepare
python workspace.py status
```

子模块固定于 `0959ecb05b0794748426e73e6efc1b6b35ec433d`，不会自动升级。新 clone 不包含环境、数据、模型或 Key；按两个部分的说明安装和准备。已经整理好的本地工作区无需重新下载。

`workspace.py prepare` 创建被 Git 忽略的兼容链接，让历史脚本和记录里的旧路径继续可用。Windows 下旧目录入口隐藏于文件浏览器，两个实体目录保持清楚；兼容链接不复制模型或数据。每次新建工作副本后运行一次，重复执行不覆盖已有目录。原始记录中的旧命令属于历史时点，当前入口以两个 README 为准；查看旧布局分支请使用独立工作副本。

## 当前真实效果

- **Agent 效率：**冻结合成交互 benchmark 中，路由版与每次调用 Agent 的版本任务成功率均为 **81.33%**，每 seed 请求从 **190 降至 99，减少 47.89%**。不是线上用户效果。
- **推荐准确率：**固定三 seed 的历史测试中，BPR NDCG@10 为 **0.228919**，LightGCN 为 **0.227695**，差异区间跨零；尚未证明 LightGCN 优于强 BPR。
- **最新低分监督研究：**完成 12 次真实训练；LightGCN 完整验证 Hit@10 观测从 **44.98% 到 47.05%**，但审计表现不一致、区间跨零，结论仍为 **NO VERIFIED GAIN**，没有替换默认模型。
- **工程验证：**整理前最新记录为 346 项完整测试通过。整理后的测试与文件完整性结果单独记录在 [迁移验收](optimized/reports/workspace_consolidation_20261012.md)。

详见 [Agent 对照](optimized/reports/benchmark/t19_live_20261010.md)、[模型对照](optimized/reports/rec_baselines/frozen_model_validation_20261011.md)、[最新研究](optimized/reports/lightgcn_explicit_low_findings_20261012.md)、[原版证据](original/reports/upstream_evidence_20261012.md)。实验负结果和历史分支保留，没有把未验证方法自动切为默认方案。

Key 保留在被忽略的根目录 `.env`，不公开数据、权重、环境或私密 trace。[MIT](LICENSE.txt) 适用于代码；原资源及 MovieLens 数据的条款须分别判断。

最初执行规范现在仅作为历史资料，**不再约束后续执行或恢复顺序**。当前状态看 [STATUS](STATUS.md)，整理决策看 [DECISIONS](DECISIONS.md)。
