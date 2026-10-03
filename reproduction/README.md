# 原版运行检查

2026-10-02真实LLM接入指导见[LIVE_LLM_GUIDE.md](LIVE_LLM_GUIDE.md)：先本机安全配置和零请求preflight，再按授权额度做单次真实连接。CONFIG_READY不等于连接验证通过。

2026-10-04：原OpenAICall经现有SDK真实连接DeepSeek通过，HTTP200、OK、usage=12+1=13；原app真实单轮/多轮尚未验证。唯一授权尝试已消耗；不要删除额度记录来重复运行。见[连接报告](../reports/live_connection_20261004.md)和[native_runs/20261004](native_runs/20261004/)。16项离线回归通过；SDK/HTTP新增测试使用现有legacy解释器运行：`reproduction/.venv-legacy/Scripts/python.exe -m unittest discover -s reproduction/tests -v`。

规范和固定upstream不变；先读`UPSTREAM_REPRODUCTION.md`确认已验证范围及未完成事项。
数据、checkpoint、解释器、模型cache及原始日志都留在本地忽略目录。资源获取脚本不会下载其他两个domain或完整ZIP；不得以fixture结果填写正式指标。

## 已建立环境

工作区工具`reproduction/.runtime/uv.exe`版本0.9.2，官方wheel及SHA记录于本地`logs/a1_20261001/uv_bootstrap.json`。
Python3.9.24和环境均在项目目录。新环境可用pip按固定版本安装，CPUwheel来自官方PyTorch索引：

```powershell
python reproduction/scripts/bootstrap_uv.py
$env:UV_PYTHON_INSTALL_DIR = "$PWD/reproduction/.runtime/python"
$env:UV_CACHE_DIR = "$PWD/reproduction/.uv-cache"
& reproduction/.runtime/uv.exe python install 3.9 --no-bin --no-registry --native-tls
& reproduction/.runtime/uv.exe venv reproduction/.venv-legacy --python 3.9 --seed --native-tls
& reproduction/.venv-legacy/Scripts/python.exe -m pip install -r reproduction/requirements.legacy.lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

本轮另外建立`.venv-legacy-rebuild`，通过锁定清单、`--offline`和已缓存的两个官方索引重建，版本逐项一致；没有重复下载大包。uv的多索引策略只在该离线、全部版本已固定的缓存验证中使用，失败的默认策略及URL缓存定位尝试也保留日志。

## 无真实API检查

在工作区根目录执行；`run_isolated.py`创建不继承真实密钥的子进程，模型cache指向项目，模型下载默认关闭。

```powershell
python reproduction/scripts/validate_complete_resources.py
python reproduction/scripts/prepare_compat.py
python reproduction/scripts/record_compat_diff.py
python reproduction/scripts/run_isolated.py -- reproduction/.venv-legacy/Scripts/python.exe -m pip check
python reproduction/scripts/run_isolated.py -- reproduction/.venv-legacy/Scripts/python.exe reproduction/scripts/native_legacy_probe.py
python reproduction/scripts/run_isolated.py -- reproduction/.venv-legacy/Scripts/python.exe reproduction/scripts/native_sdk_mock.py
python reproduction/scripts/run_isolated.py -- reproduction/.venv-legacy/Scripts/python.exe reproduction/scripts/native_tool_smoke.py --agent-replay --app-startup
```

首次/每次完整Gallery初始化会真实编码9888个标题，CPU需等待；没有用随机embedding替换。使用`run_logged.py --name <新名称> -- <命令>`可保存原始stdout/stderr/真实退出码，不覆盖旧验证。
全量小型离线回归：`python -m unittest discover -s reproduction/tests -v`。

## 真实LLM和评测前置

规范§3.3/§5.3规定默认`allow_paid_api=false`、`api_request_cap=0`。
必须由用户明确提供/授权provider、model ID、请求上限、token上限、金额预算及单位，并在本机安全配置密钥；不要写入Git或聊天。
原app与原eval调用模板按规范§5.3执行。首次真实连接授权已消耗1/1；原app/评测新请求另需额度授权。原eval数据文件仍未获得，不把自造smoke样例当原数据。
完整资源可加载不证明checkpoint共享ID语义正确；正式评测还需权威训练映射、来源与资源授权核验。
