# AgenticRec 开发包

已验收 T05 与 T06：独立 Python 3.11.14、严格配置、评分协议、FakeLLM，以及官方稳定 MovieLens1M 的全局时间划分和训练期 ID 契约。训练、真实 SDK、推荐流水线和效果 benchmark 尚未实现；不能从数据合同测试推导推荐质量。

T06 数据输入从官方 [MovieLens1M](https://grouplens.org/datasets/movielens/1m/) 下载并按官方 MD5 核对，原 ZIP/条款保留在本地忽略目录 `../data/raw/ml-1m/`。在工作区根目录运行 `& AgenticRec/.venv-dev/Scripts/python.exe -m agenticrec.data --config AgenticRec/configs/data.yaml`；首次产出 `../data/processed/ml-1m-v1/` 与 `../artifacts/data_manifest.json`，重复运行拒绝覆盖。用 `& AgenticRec/.venv-dev/Scripts/python.exe -m pytest AgenticRec/tests/unit/test_data_contract.py AgenticRec/tests/regression/test_no_leakage.py -q` 核验。详细规则和计数见[数据协议报告](../reports/t06_movielens_protocol_20261006.md)。

从工作区根目录运行已存在的命令：

```powershell
& AgenticRec/.venv-dev/Scripts/python.exe -m agenticrec.cli doctor --offline --config AgenticRec/configs/experiment_contract.example.yaml
& AgenticRec/.venv-dev/Scripts/python.exe -m agenticrec.cli fixture
& AgenticRec/.venv-dev/Scripts/python.exe -m pytest AgenticRec/tests/unit/test_config.py -q
```

配置文件采用 YAML 1.2 的 JSON 子集；未知字段、未知 CLI 参数、布尔型数量、NaN 预算均报错。默认付费请求为 0。FakeLLM 不联网，usage 为 null，带 fixture 标记。AttemptBudget 只是内存中的尝试计数原语，不是已完成的 T13 SDK、持久预算或账单硬限额。

隔离环境重建命令（首次需要取得公开 Python/依赖，后续可用缓存）：

```powershell
$env:UV_PYTHON_INSTALL_DIR = "$PWD/reproduction/.runtime/python"
$env:UV_CACHE_DIR = "$PWD/reproduction/.uv-cache"
& reproduction/.runtime/uv.exe python install 3.11.14 --no-bin --no-registry --native-tls
& reproduction/.runtime/uv.exe venv AgenticRec/.venv-dev-rebuild --python 3.11.14 --offline
& reproduction/.runtime/uv.exe pip install --python AgenticRec/.venv-dev-rebuild/Scripts/python.exe -r AgenticRec/requirements.dev.lock.txt --offline
& reproduction/.runtime/uv.exe pip install --python AgenticRec/.venv-dev-rebuild/Scripts/python.exe --no-build-isolation --no-deps -e AgenticRec --offline
& reproduction/.runtime/uv.exe pip check --python AgenticRec/.venv-dev-rebuild/Scripts/python.exe
```

上述重建路径应为空；不要覆盖正在使用的环境。首次没有缓存时，依赖安装需去掉 `--offline`，环境建立后保持版本锁不变。T05 的首次重建验证了7个锁定依赖和两个环境各15项基础测试；T06 增加 PyArrow 21.0.0 后已用缓存对第二环境离线安装并运行5项数据合同/防泄漏测试。历史环境记录见 [环境报告](../reproduction/environment_dev.json)，新增验证见[数据协议报告](../reports/t06_movielens_protocol_20261006.md)。

上游独立保存在 `../RecAI/`；legacy 环境在 `../reproduction/.venv-legacy/`。开发包不依赖 legacy 启动，不更改原推荐算法。完整任务、原资源限制和后续依赖见 [状态](docs/STATUS.md)。
