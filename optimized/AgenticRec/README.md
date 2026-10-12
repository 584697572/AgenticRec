# AgenticRec 个人实现包

这是 `optimized/` 中的可安装 Python 包。当前效果、模型选择和实验边界见 [优化版总览](../README.md)，原版在 [original](../../original/README.md)。

- `src/agenticrec/`：推荐模型、召回/融合、约束、状态、执行器、路由、LLM和评价。
- `configs/`：运行配置、固定强基线和保留的研究配置。
- `scripts/`：训练编排、实验重算、验证及分析。
- `tests/`：单元、回归和集成测试。
- `reports/`：[Agent benchmark](reports/benchmark.md)、[消融](reports/ablations.md)、[失败分析](reports/failure_analysis.md)。更完整的模型研究位于 [上一级reports](../reports)。

从仓库根目录执行已有本地环境：

```powershell
python workspace.py prepare
& optimized/AgenticRec/.venv-t20/Scripts/python.exe -m agenticrec.cli doctor --offline
& optimized/AgenticRec/.venv-t20/Scripts/python.exe -m agenticrec.cli fixture
& optimized/AgenticRec/.venv-t20/Scripts/python.exe -m agenticrec.fusion_demo --request optimized/AgenticRec/configs/fixed_request.example.json --model bpr --profile validated-20261011 --mode rrf
```

新环境与数据/训练流程见 [REPRODUCE](docs/REPRODUCE.md)。[贡献清单](docs/MY_CONTRIBUTIONS.md)、[上游差异](docs/UPSTREAM_DIFF.md)、[限制](docs/KNOWN_LIMITATIONS.md)保留技术细节；其中日期和旧路径属于当时的记录，旧路径由本地兼容入口支持，当前全局状态以 [STATUS](../../STATUS.md)为准。

工程测试不等于质量提升；fixture不使用真实训练模型或真实LLM。最新低分监督仍NO VERIFIED GAIN，不自动替换默认模型。代码采用 [MIT](../../LICENSE.txt)，数据与原资源独立判断许可。
