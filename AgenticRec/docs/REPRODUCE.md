# 复现指南（Windows / CPU）

从仓库根目录运行，除明确的`Push-Location`。锁定Python3.11.14、PyTorch2.0.1+cpu、PyArrow21.0.0，不改系统Python/CUDA。T20命令、原始日志和SHA见[验收](../../reports/release/t20_20261010.json)。独立现代环境与无资源源码快照已验；legacy解释器复用范围单独记录，不声称另一台空白机器全量重建。

## 1. 环境与零资源演示

按[README快速开始](../README.md#离线快速开始)初始化项目内uv/Python和`.venv-dev`。初始bootstrap固定uv0.9.2并校验SHA，需要一份已有Python。环境存在时另取名称，不覆盖。CPU官方索引的同名包可能遮蔽PyPI中的锁定版本，安装使用`unsafe-best-match`，仅使用这两个官方索引及精确lock，不加未知索引。完整缓存可加`--offline`。

```powershell
& AgenticRec/.venv-dev/Scripts/python.exe -m agenticrec.cli doctor --offline
& AgenticRec/.venv-dev/Scripts/python.exe -m agenticrec.cli fixture
& AgenticRec/.venv-dev/Scripts/python.exe -m pytest AgenticRec/tests/unit/test_config.py AgenticRec/tests/integration/test_release_demo.py -q
```

fixture不读数据、权重、Key，不联网，返回PASS/is_fixture=true/remote_api_requests=0；正常、匿名、排除、反馈、无解、重复工具和A/O路由均用自造目录与toy scorer，质量NOT EVALUATED。doctor仅验基础环境/配置，不验模型或GPU。

## 2. 官方数据与冻结

先阅读[MovieLens1M条款](https://files.grouplens.org/datasets/movielens/ml-1m-README.txt)：研究使用须引用，未经单独许可不得再分发，商业用途须另获许可。[官方稳定1M下载页](https://grouplens.org/datasets/movielens/1m/)提供约6MB ZIP、README、MD5，不使用latest，不下载其它领域或额外模型。以下只用于全新数据目录；已有文件校验复用，不覆盖。

```powershell
if (Test-Path data/raw/ml-1m) { throw "原始目录已存在，请检查而非覆盖" }
New-Item -ItemType Directory data/raw/ml-1m | Out-Null
Invoke-WebRequest https://files.grouplens.org/datasets/movielens/ml-1m.zip -OutFile data/raw/ml-1m/ml-1m.zip
Invoke-WebRequest https://files.grouplens.org/datasets/movielens/ml-1m-README.txt -OutFile data/raw/ml-1m/ml-1m-README.txt
Invoke-WebRequest https://files.grouplens.org/datasets/movielens/ml-1m.zip.md5 -OutFile data/raw/ml-1m/ml-1m.zip.md5
& AgenticRec/.venv-dev/Scripts/python.exe -m agenticrec.data --config AgenticRec/configs/data.yaml
& AgenticRec/.venv-dev/Scripts/python.exe -m agenticrec.evaluation.cohort
& AgenticRec/.venv-dev/Scripts/python.exe -m agenticrec.evaluation.episodes
```

无需解压ZIP。data校验固定SHA、官方MD5、CRC、schema与时间/ID契约。三个入口分别生成`data/processed/ml-1m-v1`及`artifacts/data_manifest.json`、私密cohort、交互集及`artifacts/eval_manifest.json`。`reproduction/native_runs/20261006_metrics`是固定协议路径，不是当前运行日期。已有私密输出拒绝覆盖；随Git发布的cohort摘要可用于恢复未发布标签，但新标签与摘要SHA必须完全一致，摘要原文保留。

T06评分1,000,209条，时间split 800,164/100,024/100,021；warm用户5,351，训练候选电影3,469，模型ID从1开始、0为padding。图/热门/ID只来自train，rating>=4为正反馈；采样排除所有训练期已评分物品，不偷看未来。目标只进evaluator；不能将原MovieLens ID直接当模型ID。

## 3. 从零训练与真实推荐

先冻结数据与cohort，再训练；仅valid选配置/早停，不train+valid重训，不因已看test调参。训练器拒绝覆盖已有权重或报告。**全新checkout且没有模型权重**时，先归档保留Git发布的历史seed报告；不要在原实验工作区执行此归档。

```powershell
if (Test-Path artifacts/models) { throw "已有模型，请另建全新工作副本" }
if (Test-Path artifacts/reference_rec_reports) { throw "归档已存在，不重复移动" }
New-Item -ItemType Directory artifacts/reference_rec_reports | Out-Null
foreach ($seed in @(7, 42, 2026)) {
    foreach ($name in @("seed_$seed.json", "lightgcn_seed_$seed.json")) {
        $sourceReport = Join-Path reports/rec_baselines $name
        if (Test-Path -LiteralPath $sourceReport) {
            Move-Item -LiteralPath $sourceReport -Destination (Join-Path artifacts/reference_rec_reports $name)
        }
    }
}
& AgenticRec/.venv-dev/Scripts/python.exe -m agenticrec.cli train --config AgenticRec/configs/bpr.yaml --seed 42
& AgenticRec/.venv-dev/Scripts/python.exe -m agenticrec.cli train --config AgenticRec/configs/lightgcn.yaml --seed 42
& AgenticRec/.venv-dev/Scripts/python.exe -m agenticrec.cli recommend --request AgenticRec/configs/fixed_request.example.json --model lightgcn --seed 42
```

训练保存真实曲线、checkpoint、身份hash与valid/test模型指标至`artifacts/models`及`reports/rec_baselines`；最终三seed为7/42/2026。LightGCN搜索1/2/3层仅按valid选，不预设优于BPR。T20快照按当前冻结配置从零训练seed42，作为复现校验，不改写T19三seed结论。

recommend严格解析结构化JSON，输出ID、元数据、来源、评分与约束证据，0LLM；示例匿名Comedy、排除Horror、1990–2000、K=5。匿名走独立scorer，不随机分配已有用户embedding。真实自由文本解析须LLM；当前没有交互chat CLI。

## 4. U1前置与完整测试

新包fixture不需要legacy。完整集成测试含原ToolBox/Buffer/Map与真实冻结模型，先准备上游/legacy；已有上游不重复clone，只核对SHA：

```powershell
git clone https://github.com/microsoft/RecAI.git RecAI
git -C RecAI checkout --detach 0959ecb05b0794748426e73e6efc1b6b35ec433d
& reproduction/.runtime/uv.exe python install 3.9.24 --no-bin --no-registry --native-tls
& reproduction/.runtime/uv.exe venv reproduction/.venv-legacy --python 3.9.24 --seed --native-tls
& reproduction/.venv-legacy/Scripts/python.exe -m pip install -r reproduction/requirements.legacy.lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
python reproduction/scripts/prepare_u1_compat.py
```

U1准备脚本核对固定原组件SHA，复制上游源码，仅写导入需要的settings，旧资源路径明确标UNUSED_BY_U1；实际Gallery/scorer由新包提供，不伪称这些旧资源存在。已有A1兼容副本不覆盖。旧`prepare_compat.py`需原资源/gte缓存，不用于全新U1；无需为U1下载原包、矩阵、checkpoint或gte-base。legacy包版本不影响独立现代训练环境。资源与模型前置满足后：

```powershell
& AgenticRec/.venv-dev/Scripts/python.exe -m pytest AgenticRec/tests/integration/test_upstream_bridge.py -q
Push-Location AgenticRec
& .venv-dev/Scripts/python.exe -m pytest tests/unit tests/integration tests/regression -q
Pop-Location
```

桥接预检验证源码、无Key子进程和候选状态。首次legacy导入受90秒超时约束，T20首轮一项失败已保留；独立检查与后续完整测试通过，根因未确认，未增加重试或放宽超时。

## 5. 一次真实LLM工程验收（默认关闭）

先完成真实数据/LightGCN seed42。只读取结构化公开请求，不读取test target，不修改T19。默认profile关闭付费，下面0请求、不读Key：

```powershell
& AgenticRec/.venv-dev/Scripts/python.exe AgenticRec/scripts/t20_live_smoke.py --profile AgenticRec/configs/release_smoke.example.json --run-id my_release_smoke
```

明确授权后，把示例副本保存至被忽略的`artifacts/runs/t20/my_authorized_profile.json`，只将allow_paid_api改true，保持1请求、最多1024输出token、非思考、零重试。0.024832CNY是单请求保守预留而非实际消费。根目录`.env`仅存`OPENAI_API_KEY=<本机值>`，不提交、不复制至发布快照；非空环境变量Key优先。

```powershell
& AgenticRec/.venv-dev/Scripts/python.exe AgenticRec/scripts/t20_live_smoke.py --profile artifacts/runs/t20/my_authorized_profile.json --run-id my_release_smoke --live
```

run-id单次使用：启动前持久intent、发送前写journal，已有目录拒绝且新增请求0。不确定交付不自动更换run-id重试，先检查证据；余额不足停止，不充值。固定真实LightGCN与真实planner经严格Plan/推荐工具输出五部满足约束的电影。usage真实，账单未查询记null；此smoke不是新质量实验。

## 6. T19复核与原报告重建

T19执行源码commit `ad4d493`，汇总commit `6bf79dd`。T20增加demo/文档与cohort精确恢复，不改算法、提示、模型、测试集。严格复核原版本需独立工作副本固定原commit，不能把后续源码SHA当原实验。

数据/cohort/models/episodes齐备时零请求核算：

```powershell
& AgenticRec/.venv-dev/Scripts/python.exe -m agenticrec.cli benchmark --config AgenticRec/configs/benchmark.yaml --dry-run
```

该命令不发LLM、不训练、不下载，不等于正式实验完成。公开authorization文件只是本项目历史授权快照，不是第三方账户授权；完整批次须本人明确授权、重新核算，才可去掉`--dry-run`。不可无意重跑2,850次请求或重置旧journal；默认批次路径`artifacts/runs/t19/live_20261010`应保留。

私密预测/trace不公开。持有原产物时可0请求重建三张表：

```powershell
& AgenticRec/.venv-dev/Scripts/python.exe -c "from agenticrec.evaluation.system_reports import generate_system_reports; generate_system_reports('AgenticRec/configs/benchmark.yaml', 'artifacts/runs/t19/live_20261010')"
```

函数校验配置/manifest/预测/providerjournal一致，按原路径写报告；缺原始证据不得手填。模型重算用`evaluation.model_benchmark`的`export_predictions`/`score_predictions`/`write_scored_summary`。历史`export_t19_model_predictions.py`会写partial报告，只用于独立初始实验，不覆盖完成的T19报告。

## 7. 实际命令与尚未实现的统一入口

| 实际入口 | 范围 |
|---|---|
| `agenticrec.cli doctor --offline` / `fixture` | 基础检查 / 自造离线推荐及Agent演示 |
| `agenticrec.cli train --config ... --seed ...` | BPR-MF/LightGCN训练及模型评测 |
| `agenticrec.cli recommend --request ...` | 结构化0LLM真实资源推荐 |
| `agenticrec.cli benchmark --config ... --dry-run` | 零请求核算；live另有授权前置 |
| `python -m agenticrec.data --config ...` | 官方源校验及时间/ID准备 |
| `python -m agenticrec.evaluation.cohort` / `.episodes` | 私密cohort精确恢复/冻结交互集 |
| `scripts/t20_live_smoke.py` | 默认关闭的单次真实工程检查 |
| `evaluation.system_reports.generate_system_reports` | 原始产物验证并重建表格 |

规范§12.1目标CLI名`data prepare/validate`、`eval-rec`、`chat`、`eval-agent`、`report`尚未实现；功能通过上表真实入口提供，不把目标命令写成已验。未知参数退出2。这一接口差异记录于[spec_issues](../../reports/spec_issues.md)，原规范不改。
