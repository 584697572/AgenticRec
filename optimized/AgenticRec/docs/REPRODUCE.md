# 当前运行与重建指南（Windows / CPU）

本指南适用于两部分目录布局。原版使用 [original](../../../original/README.md) 的旧环境；个人实现使用 Python3.11.14 / Torch2.0.1+cpu / PyArrow21.0.0。原T20完整历史配方按字节保存在 [旧指南](../../history/navigation_before_20261012/AgenticRec/docs/REPRODUCE.md)，不覆盖旧实验或其日志。

## 1. 已有本地工作区

从仓库根目录执行；不下载、不请求LLM：

```powershell
python workspace.py prepare
python workspace.py status
& optimized/AgenticRec/.venv-t20/Scripts/python.exe -m agenticrec.cli doctor --offline
& optimized/AgenticRec/.venv-t20/Scripts/python.exe -m agenticrec.cli fixture
& optimized/AgenticRec/.venv-t20/Scripts/python.exe -m agenticrec.fusion_demo --request optimized/AgenticRec/configs/fixed_request.example.json --model bpr --profile validated-20261011 --mode rrf
Push-Location optimized/AgenticRec
& .venv-t20/Scripts/python.exe -m pytest -q
Pop-Location
```

RRF为显式选择的融合方案；可以改为 `--mode no_content` 作同模型对照。模型和历史质量结论不会因为此演示而改变。`fixture`使用六部自造电影和FakeLLM；真实模型入口需要已有数据、权重。

## 2. 全新 checkout 的环境

先取得固定子模块并创建本地兼容入口。安装阶段需要官方依赖下载，已有完整缓存可离线；不要覆盖已有环境。

```powershell
git submodule update --init original/RecAI
python workspace.py prepare
python original/reproduction/scripts/bootstrap_uv.py
$env:UV_PYTHON_INSTALL_DIR = "$PWD/original/reproduction/.runtime/python"
$env:UV_CACHE_DIR = "$PWD/original/reproduction/.uv-cache"
& original/reproduction/.runtime/uv.exe python install 3.11.14 --no-bin --no-registry --native-tls
& original/reproduction/.runtime/uv.exe venv optimized/AgenticRec/.venv-dev --python 3.11.14 --offline
& original/reproduction/.runtime/uv.exe pip install --python optimized/AgenticRec/.venv-dev/Scripts/python.exe --native-tls --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cpu -r optimized/AgenticRec/requirements.dev.lock.txt
& original/reproduction/.runtime/uv.exe pip install --python optimized/AgenticRec/.venv-dev/Scripts/python.exe --offline --no-build-isolation --no-deps -e optimized/AgenticRec
& original/reproduction/.runtime/uv.exe pip check --python optimized/AgenticRec/.venv-dev/Scripts/python.exe
& optimized/AgenticRec/.venv-dev/Scripts/python.exe -m agenticrec.cli fixture
```

本地 `.venv-t20` 是现有环境名，新安装示例采用 `.venv-dev`；后续命令相应替换。依赖版本来自lock，CPU wheel和PyPI使用官方索引。新clone没有数据或checkpoint。

## 3. 数据、模型与完整实验

MovieLens1M 数据使用独立 GroupLens 条款，不公开分发。已有 `optimized/data/` 和 `optimized/artifacts/` 先校验复用，不能重复生成到同一冻结目录。新工作副本的完整下载、数据冻结、cohort重建和三seed原始训练步骤见 [归档配方](../../history/navigation_before_20261012/AgenticRec/docs/REPRODUCE.md)。运行 `workspace.py prepare` 后，该配方的旧路径为有效兼容入口，其物理输出归属于本布局。

现有的固定强基线训练命令见 [三seed报告](../../reports/rec_baselines/frozen_model_validation_20261011.md)，最新低分监督协议见 [研究协议](../../reports/lightgcn_explicit_low_protocol_20261012.md)。这些方法可能要求冻结run_id且拒绝覆盖：完整重建应在新的工作副本中执行，不能直接重跑到旧权重目录。历史test已暴露，不能称新盲测。

真实LLM配置保持现有机制，Key在根目录 `.env`，被Git忽略。结构化推荐不调用LLM；自由文本/规划入口另计真实请求。不要删除历史调用账本来解锁重复运行。此次目录迁移本身不创建真实API请求。

目录迁移验收见 [报告](../../reports/workspace_consolidation_20261012.md)。旧源码、原记录、数据和权重逐项核对；本地隐藏兼容链接不进入GitHub，重新clone后运行 `workspace.py prepare` 即可建立。
