# LG-LF01：显式低分监督真实训练（2026-10-12）

单变量：训练低分1/2星的独立排序监督，权重0/.25；原未知采样、图、正则不变。强BPR同步训练，alpha0是移除低分监督的消融。

12训练×三seed（7/42/2026），实际330 epoch / 74910更新。
384人开发选择/早停，seed42按NDCG冻结，97人最后开发审计；全valid有481名相关暖用户，其余4870无相关不进Hit分母。两组早停epoch允许不同，共同上限32、patience8。

**已有valid已经暴露，本轮没有读取旧test；这不是新盲测。独立推荐质量 NOT EVALUATED，不声称微软论文或线上业务提升。**

| 方法 | 选择NDCG | 审计NDCG ± seed标准差 | 审计Hit@10 | 全valid NDCG | 全valid Hit@10 |
|---|---:|---:|---:|---:|---:|
| gcn_uniform | 0.137638 | 0.179833 ± 0.004825 | 45.70% | 0.146147 | 44.98% |
| gcn_low25 | 0.140796 | 0.177880 ± 0.008942 | 46.39% | 0.148275 | 47.05% |
| bpr_uniform | 0.135902 | 0.177236 ± 0.001923 | 46.39% | 0.144237 | 45.11% |
| bpr_low25 | 0.135221 | 0.178616 ± 0.004596 | 43.99% | 0.143972 | 45.46% |

seed42冻结选择：{"bpr": "bpr_uniform", "lightgcn": "gcn_low25"}；不按审计改赢家。

## 配对差异与消融

在用户内平均三seed后2000次user bootstrap；95%区间未经多重校正，仅为开发探索。
- selection bpr_low25_minus_bpr_uniform：差-0.000681，CI[-0.010632, +0.008473]。
- selection gcn_low25_minus_bpr_uniform：差+0.004895，CI[-0.005033, +0.014372]。
- selection gcn_low25_minus_gcn_uniform：差+0.003158，CI[-0.006246, +0.012515]。
- audit bpr_low25_minus_bpr_uniform：差+0.001380，CI[-0.013886, +0.016392]。
- audit gcn_low25_minus_bpr_uniform：差+0.000643，CI[-0.015022, +0.016357]。
- audit gcn_low25_minus_gcn_uniform：差-0.001953，CI[-0.016092, +0.011326]。

## 训练机制诊断（事后，仅train）

使用前轮已固定的正例—低分配对，对best checkpoint计算低分得分不低于正例的用户宏平均比例；这只说明拟合训练偏好，不代表未见电影推荐提升，也不用于改alpha或重新选模型。

| 方法 | train偏好倒置比例（三seed平均） |
|---|---:|
| gcn_uniform | 22.47% |
| gcn_low25 | 12.63% |
| bpr_uniform | 22.19% |
| bpr_low25 | 11.02% |

## 结论与验证

GCN NO VERIFIED GAIN；BPR NO VERIFIED GAIN。独立质量 NOT EVALUATED，默认模型不替换。
完整工程346项、已有锁环境11专项通过；两个环境各12checkpoint的原始预测/指标完全一致。
重算46176项用户指标并验证聚合/区间/分层；6组初始化与采样哈希一致，820份旧文件/权重SHA不变。
训练commits ['483a036bffdca83c62408444bce81f28ce4fcc7b']，所有训练启动Git clean。源码SHA 71e8c6b0ff947aa32fe6069e552f91538b029f4dbb9aaad2950af9ea0fa55505；plan SHA db8e4767586117f248985c209163f8bb0859354e7e7f704e7d51cc434e7bb428。
六组uniform最佳epoch与最终选择指标均与前轮一致；best权重最大差1.788e-6，不声称bit-identical重放。BPR seed42的非最佳第32轮NDCG有0.000166差异，原因未作因果验证，不用于选择新参数。
两个环境复用当前checkout和冻结数据，不是全新机器重训；CPU并发时间只记录成本，不是隔离速度基准。
初始缺模块失败及系统tmp无权限错误保留，工作区tmp测试通过；全套测试可使用历史评测fixture，本轮训练/审计/报告不读test。API0、数据包/权重下载0；另只读归档官方HetRec README 8105bytes，SHA及非商业许可记录保留。

交付时先回归复现新目录缺少初始化，再修补运行目录与历史保护ledger自动创建；未重训或修改已完成算法。PowerShell将Torch stderr警告转成NativeCommandError的日志保留；直接捕获两个环境Python真实退出码均0、验证JSON完全一致。

## 复现

在已完成LG-D02/LG-HN01、拥有冻结ML1M数据/开发组/训练样本和已有环境的工作区运行；新checkout先按项目README重建前置数据与环境。初始化自动创建目录和保护ledger，现有结果仅允许校验后续跑，不覆盖。

```powershell
& .\AgenticRec\.venv-t20\Scripts\python.exe .\AgenticRec\scripts\run_explicit_low_study.py
& .\AgenticRec\.venv-t20\Scripts\python.exe .\AgenticRec\scripts\verify_explicit_low_study.py --tag rerun
```

未暴露HetRec2011确认协议已预登记；本次不因开发观测无限扩alpha或重复旧audit，不以该数据包尚未取得冒称外部结论。
