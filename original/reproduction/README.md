2026-10-05 当前：[目录核查](../reports/catalog_identity_20261005.md)以官方元数据冻结9883个唯一身份和来源别名、保留5个歧义。最新64项测试通过；有界搜索未取得旧训练映射，ADR-015保留A1功能证据并为质量实验选择规范独立重建路径，T03审计/路线完成，下一批T06。以下修复/真实运行记录是前批快照。

此前修复：[修复报告](../reports/reproduction_repairs_20261005.md)验证独立年份/反思历史补丁，53项测试及原组件fixture对照通过，原真实45条和0/45保留。

官方元数据获取脚本只抽取 movies.dat/README，Windows原生TLS验证与1MiB字节上限；重复获取/冻结拒绝覆盖。目录身份模块不分配旧checkpoint行标签。本地来源和映射不再分发，旧资源许可限制仍保留。

# 原版运行检查

2026-10-05 最新：原 App 真实单轮及两轮年份修改已验证；原 recbot、random、加权 popularity 各完整执行45条A1派生输入，正式模糊hit均0/45。实际93次HTTP200、201944 token，高峰估算0.507208CNY；账单未查询。21个target不可映射、4个映射年份不同、索引37截断且无Map，均保留。当前任务已关闭，不能删除记录重跑；T04其余条件仍BLOCKED，A2未验证。详见[多轮与评测报告](../reports/a1_multiturn_eval_20261005.md)。以下未运行/等待Key等文字为历史快照。

## 此前历史记录

2026-10-05 最新：用户选择根目录 `.env` 持久配置，执行端已自动读取并完成原 App 真实单轮 SUCCESS/VERIFIED（2次HTTP200、3880 token、目录约束全部通过）。当前单轮2/2额度已用；多轮/原评测及资源语义限制仍未验收，T04整体BLOCKED。见 [真实单轮报告](../reports/live_original_app_20261005.md)。下文尚未运行/等待Key的描述均为此前历史阶段。

2026-10-05 当前入口：[受限原 App 单轮](LIVE_LLM_GUIDE.md)，新授权已具备，最多 2 次尝试/每次 512 输出 token/1 CNY；在用户已有 Key 的终端运行 `reproduction/.venv-legacy/Scripts/python.exe reproduction/scripts/live_app_single_turn.py`。本轮 offline 联调通过，live 仍 NOT VERIFIED。T05 已完成，见 [现代开发包](../AgenticRec/README.md)。ReDial raw test 已取得，45 条 A1 派生评测输入已准备且重复运行哈希一致；canonical 作者子集仍 NOT VERIFIED。其余下文日期保留为历史记录。

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
