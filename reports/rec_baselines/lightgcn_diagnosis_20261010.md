# LightGCN 诊断：层数与训练配置的匹配（2026-10-10）

本轮诊断 DONE；正式改进模型的三seed/test验收 NOT EVALUATED。当前最有证据、最值得先改的是传播层数与训练配置的匹配。不能将单seed的开发收益包装为正式测试或线上提升。

## Completed

- 审计图构造、padding/ID、传播、BPR损失、ego正则、训练采样和全集排序。独立稠密小图损失及所有参数梯度误差检查通过（阈值1e-7）。
- 在固定train/valid、seed42、32维、lr0.001、lambda1e-4下完成6个预登记/分阶段登记训练对照；未进行新test评测。
- 复算原1层/3层验证曲线与最终指标，保护52个历史基线文件。

## Files Created / Modified

- 三份计划：`AgenticRec/configs/lightgcn_diagnostic*_20261010.json`。
- 独立训练模块：`AgenticRec/src/agenticrec/lightgcn_diagnostic.py`。
- 聚合重建：`AgenticRec/scripts/report_lightgcn_diagnosis.py`。
- 独立数值/标签访问保护测试：`AgenticRec/tests/unit/test_lightgcn_diagnostic.py`。
- 本报告、[机器报告](lightgcn_diagnosis_20261010.json)、STATUS、TASKS和ADR-036。原训练器、正式config、模型和T19表格不变。

## Commands / Tests

从仓库根目录运行（每个plan只能首次创建自己的run，重复执行拒绝覆盖）：

```powershell
& AgenticRec/.venv-t20/Scripts/python.exe -m agenticrec.lightgcn_diagnostic --plan AgenticRec/configs/lightgcn_diagnostic_20261010.json
& AgenticRec/.venv-t20/Scripts/python.exe -m agenticrec.lightgcn_diagnostic --plan AgenticRec/configs/lightgcn_diagnostic_layer1_20261010.json
& AgenticRec/.venv-t20/Scripts/python.exe -m agenticrec.lightgcn_diagnostic --plan AgenticRec/configs/lightgcn_diagnostic_patience_20261010.json
& AgenticRec/.venv-t20/Scripts/python.exe AgenticRec/scripts/report_lightgcn_diagnosis.py
```

在`AgenticRec/`执行专项：`python -m pytest tests/unit/test_lightgcn_diagnostic.py tests/unit/test_lightgcn.py tests/unit/test_bpr.py tests/unit/test_metrics.py -q -k "not real_graph"`。真实图的holdout交叉测试沿用T20已验证证据且其数据SHA未变，本轮不读新test标签。

## Verification Results

25项专项通过，1项真实holdout审计因本轮仅验证集范围而主动不选。compileall与git diff检查通过；本轮未重跑完整258项，不能把T20的253项历史验收当作当前全量测试。

| 配置 | 实际训练轮数 | 最优轮 | valid NDCG@10 |
|---|---:|---:|---:|
| 原配置：3层 / batch32768 / 8轮 / patience2 | 8 | 7 | 0.121864516 |
| 3层 / batch32768 / 最多32轮 / patience8 | 25 | 17 | 0.122388154 |
| 3层 / batch2048 / 最多8轮 / patience2 | 4 | 2 | 0.122004474 |
| 1层 / batch32768 / 最多8轮 / patience2 | 8 | 8 | 0.118816000 |
| 1层 / batch2048 / 最多8轮 / patience2 | 8 | 8 | 0.131416648 |
| 3层 / batch2048 / 最多8轮 / patience8 | 8 | 8 | 0.122805226 |

相同8轮、batch2048、1,816次参数更新的实际预算下，1层比3层高 **7.01%**；探索性配对用户差值=0.008611422，95%区间=[0.002158970, 0.015717173]。两者最优均在第8轮，1层patience2未触发，3层patience8也完成8轮。相对原3层大batch配置的验证增幅为7.84%。这些百分比是NDCG相对变化，不是成功率百分点。

- 延长3层训练的差值区间包含0；仅缩小3层batch的差值区间也包含0。
- 放宽3层小batch早停的差值区间包含0，不能将过早停止单独认定为最大性能根因。
- 原1层/3层valid曲线和最终指标精确复现；训练loss严格位相等检查最初失败，最大差异6.33e-9，在随后声明的1e-7数值阈值内。保留初次检查，未修改模型指标。
- 首次pytest因basetemp父目录尚未创建产生3个setup错误（20项通过），创建父目录后通过。路径保护的相对路径/bytes先2项红，再修复为绿；不涉及真实test目标。
- 保留缺少可选NumPy桥接与TypedStorage弃用警告，未修改固定依赖版本。

## Findings

1. **最优先：层数与训练配置联选。** 原batch32768下1层不如3层；换batch2048后，1层优于同样训练满8轮的3层。原搜索只在固定短预算下比较层数，结论不能直接沿用到新的更新频率。
2. **热门集中与表示趋同是观察到的现象。** 同预算3层/1层的Top10位置中，训练热门前100的占比分别为99.00%/94.07%；不同电影数量131/219。抽取固定2000对不同训练用户，表示余弦均值为0.9463/0.7085；整个图表示在sqrt(degree)主模式上的能量占比89.98%/63.31%。支持个性化表示不足的解释，但不证明过平滑是唯一原因；也不把更分散本身等同于更好。
3. **单纯增加训练量不是已验证的主要解法。** 大batch每轮15次更新、小batch227次的差异确实存在，但3层配置单独延长训练或减batch没有明确验证收益。第一假设被真实对照削弱，不能只凭训练步数差距下结论。
4. **未发现明显的公式/ID错误。** 对照固定[作者模型源码](https://github.com/gusye1234/LightGCN-PyTorch/blob/947ca2b3b1d2d3545b114145710cb06c4e57b3d2/code/model.py)的传播、均匀聚合和ego正则；小图独立数值与梯度通过。官方[参数默认值](https://github.com/gusye1234/LightGCN-PyTorch/blob/947ca2b3b1d2d3545b114145710cb06c4e57b3d2/code/parse.py)不是本数据集性能保证，未照搬1000轮或换切分。
5. **采样/维度仍可检查，但尚无效果证据。** 官方Python采样先均匀选用户，再选该用户正样本；本项目BPR和LightGCN均按训练正边遍历，top10%训练用户占35.43%的训练边。它是共享建模选择，不是已证实的LightGCN相对下降原因。32/64维联合搜索尚未完成。

## Blockers

本轮诊断：None。完整改进结果：三seed和新模型的正式评测 NOT EVALUATED。原T04旧资源ID/许可/论文条件继续BLOCKED，未用于本次独立模型。

## Deviations from Spec

用户在T20后明确要求优先诊断LightGCN，因此T21暂待，另立LG-D01；未修改原规范或T00—T24验收历史。探索实验继续遵守valid-only选择，但历史test已被看过，不能称新的独立确认性实验。第二、三阶段根据前阶段valid结果追加并事先登记，保留全部对照；没有改test窗口、删除坏结果或选test最优。

## Current Project State

LG-D01诊断完成，T19原负结果与52个历史基线文件完全保留；原推荐/演示链仍加载原LightGCN checkpoint，未自动替换成诊断模型。原模型数据/config/训练数学代码不变；真实LLM请求0，新数据/权重/依赖下载0。前两组执行源码commit a3f4f12，最后早停对照commit 27e4095；源码差异只涉及诊断文件访问保护，基础训练数学代码SHA相同。Python文件保护不覆盖native I/O/OS边界，实际训练加载器只访问明确train/valid路径。

## Next Tasks

1. 围绕层数1/2/3与batch/早停做有限valid-only联合搜索；验证32/64维，并按可比选择规则同时检查BPR，避免只给LightGCN增加调参预算。
2. 冻结最终配置再跑7/42/2026三个seed；保留旧结果，并按预先记录协议做全集指标、配对统计与冷启动/约束链回归。复用历史test时明确后续开发与既往测试暴露，不包装为新的盲测。
3. 只有上述验收通过后才另存/接入新模型；之后继续T21技术报告与简历事实核对。
