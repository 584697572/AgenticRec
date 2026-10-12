# 微软原版：尽可能接近的复现版本

本部分保留 Microsoft RecAI / InteRecAgent 的原始源码、原预制电影资源的本地副本、旧依赖环境和运行证据。个人训练的 BPR / LightGCN 与优化 Agent 在 [另一部分](../optimized/README.md)。

## 源码与修改边界

- `RecAI/` 为固定子模块，commit `0959ecb05b0794748426e73e6efc1b6b35ec433d`，原工作树保持干净。
- `reproduction/.runtime/InteRecAgent-compat/` 为原代码的兼容运行副本，数据和环境均留本地。
- `reproduction/legacy_compat.patch` 为环境兼容差异；[差异说明](reproduction/UPSTREAM_DIFF.md)记录适用范围。
- `reproduction/patches/` 的年份、反思历史修复单独保留，不能把修复后的行为称为未修改上游。
- 原版仍使用原 SASRec checkpoint / 相似度矩阵；不以个人训练的 LightGCN 替换后宣称原版复现。

## 已验证与剩余缺口

原电影目录、矩阵和 checkpoint 可加载，原工具链、应用入口及受控多轮运行已有证据；真实 DeepSeek 单轮、两轮以及 45 条派生 A1 输入的执行记录保留。

**论文级完整复现未完成：**旧模型和矩阵的电影 ID 权威对应关系未证实；预制资源独立许可未核实；作者 canonical 电影评测文件与模型条件未完整对齐。45 条派生输入不是论文 canonical 评测，运行成功不等于推荐质量或论文指标复现。

最新核查见 [原版证据补齐](reports/upstream_evidence_20261012.md)、[真实运行](reports/live_original_app_20261005.md)、[多轮与原评测诊断](reports/a1_multiturn_eval_20261005.md)。旧报告保留原日期和字节，不将历史“未运行”文字改写成当前状态。

## 运行入口

从仓库根目录执行。现有本地环境和资源已迁移，不必重新安装或下载。

```powershell
python workspace.py prepare
git -C original/RecAI rev-parse HEAD
git -C original/RecAI status --short
& original/reproduction/.venv-legacy/Scripts/python.exe -m unittest discover -s original/reproduction/tests -v
```

新 clone 按 [旧环境及资源指南](reproduction/README.md)安装；先在仓库根目录运行 `workspace.py prepare`，该指南中的旧路径将通过兼容入口有效。正式运行前核对数据和 patch，不重复使用已消耗的历史请求额度或覆盖原实验目录。

真实 LLM 指导见 [LIVE_LLM_GUIDE](reproduction/LIVE_LLM_GUIDE.md)。Key 仍保存在仓库根目录 `.env`，两个版本直接定位并读取同一份根文件；导入模块不会读取或输出 Key。此次仅调整辅助脚本的Key文件定位，没有修改微软原始源码或原App的算法。

上游 [官方源码](https://github.com/microsoft/RecAI/tree/0959ecb05b0794748426e73e6efc1b6b35ec433d/InteRecAgent) 使用 MIT 代码许可；本部分不公开分发许可尚未核实的原数据包或 checkpoint。
