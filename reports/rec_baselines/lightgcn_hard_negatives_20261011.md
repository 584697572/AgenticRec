# LG-HN01：混合困难负样本真实训练（2026-10-11）

仅改变未知负样本选择：uniform与25%最高分池+75%随机；GCN/BPR均等预算。原上游/数据划分/指标/服务默认不改，低分和时间监督未加入。

12次真实训练，实际316epoch/71732次更新；四条件×seed7/42/2026。每epoch462887训练三元组，227次更新，最多32epoch/patience8，batch2048/lr0.001/reg0.0001。GCN1层64维、BPR32维沿用各自强配置，维度不同不宣称纯结构效率。

384人用于早停与seed42选择，冻结后补全其他seed，97人最后做开发审计。此前valid已暴露，这不是独立测试。全量valid为5351暖用户，其中481人有合格目标、4870人无目标单列；无目标不作为Hit分母。

## 所有条件（主NDCG@10）

| 方法 | 选择组NDCG均值 | 审计NDCG均值±标准差 | 审计Hit@10 |
|---|---:|---:|---:|
| gcn_uniform | 0.137638 | 0.179833 ± 0.004825 | 45.70% |
| gcn_mixed25 | 0.135620 | 0.184476 ± 0.000798 | 47.77% |
| bpr_uniform | 0.135902 | 0.177236 ± 0.001923 | 46.39% |
| bpr_mixed25 | 0.137987 | 0.186348 ± 0.004722 | 48.45% |

seed42冻结选择：{"bpr": "bpr_uniform", "lightgcn": "gcn_uniform"}。不根据审计更换赢家，不扩大比例。

## 配对差异与消融

用户内先平均三个seed，再bootstrap2000/seed42；探索性95%区间未多重校正。uniform是移除困难选择的单变量消融。

- bpr_mixed25_minus_bpr_uniform：NDCG差+0.009112，区间[-0.000616, +0.020008]。
- gcn_mixed25_minus_gcn_uniform：NDCG差+0.004643，区间[-0.002620, +0.012401]。
- gcn_uniform_minus_bpr_uniform：NDCG差+0.002597，区间[-0.006662, +0.012149]。

结论：GCN NO VERIFIED GAIN；BPR NO VERIFIED GAIN。独立推荐质量NOT EVALUATED，不能写微软论文或线上提升。默认模型没有替换。

## 训练信号及公平性

每seed的初始化、正边顺序、8候选池与随机掩码逐epoch核验哈希一致，所有条件均评分候选池。最终仅一个负片参与原BPR数据loss和ego正则；选择detach不反传。候选互异、排除所有train评分与padding，不查看未来喜欢标签。

Torch候选流与独立正边shuffle不同于旧共用Python随机流，因此重新训练匹配uniform基线；本次对照不是旧checkpoint原样重放。训练并行时间仅记录开销，不作为孤立性能实验。

| 方法 | 最佳epoch处容易题比例（seed均值） | 实际困难选择比例 |
|---|---:|---:|
| gcn_uniform | 63.19% | 0.00% |
| gcn_mixed25 | 40.46% | 24.98% |
| bpr_uniform | 66.81% | 0.00% |
| bpr_mixed25 | 47.75% | 24.96% |

训练信号强弱不是验证精度；未知高分电影可能是真喜欢，困难负例风险尚不能用未来标签修正。所有条件、活动度分层和热门集中度保留在JSON。

## 验证

46176项原始用户指标/均值/区间/分层/热门集中度重算；12个checkpoint重载与选择指标一致；训练commits ['c10f337aa51822eefae918709d92839cfbbd9a15']，dirty runs []。完整335项及既有锁环境10专项通过，锁环境12个真实模型重载指标完全一致，912历史文件SHA不变。既有环境/共享数据，不是全新checkout重训。

完整工程测试可能使用历史评测fixture；本轮训练/审计/报告没有打开test标签。API0、下载0，未改旧权重。

## 复现

```powershell
& .\AgenticRec\.venv-t20\Scripts\python.exe AgenticRec/scripts/run_hard_negative_study.py
& .\AgenticRec\.venv-t20\Scripts\python.exe AgenticRec/scripts/report_hard_negative_study.py
```

需要原冻结数据及开发分组。每个run拒绝覆盖；runner只恢复身份和SHA完全一致的成功结果，部分失败需要另行记录处理。报告仅首次创建或完全一致重建。后续显式低分监督需单独登记，不能据本次audit继续调参。
