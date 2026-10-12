# LightGCN采样与层聚合优化（2026-10-11）

有界开发研究完成。结论：**NO VERIFIED GAIN**。不改变原服务默认，不打开历史test，不称微软论文/线上提升。

## 实验与选择

固定MovieLens1M训练图与全集候选；384个开发用户选择/early stop，97个开发用户在冻结后审计。此前valid已用过，audit不是独立盲测。6条件×3seed；每epoch462887三元组，batch2048，lr0.001，L2=0.0001，至多32轮/patience8。GCN1层64维，BPR32维沿用上轮强配置。

seed42开发选择GCN=gcn_edge_uniform，BPR=bpr_edge；另两seed及audit后不重新挑赢家。

| 方法 | 选择集NDCG@10（3seed均值） | 审计NDCG@10（均值±std） | 审计Hit@10 |
|---|---:|---:|---:|
| gcn_edge_uniform | 0.139709 | 0.178813±0.003069 | 46.05% |
| gcn_user_uniform | 0.137873 | 0.172841±0.003891 | 47.08% |
| gcn_edge_residual | 0.136964 | 0.179920±0.001899 | 46.74% |
| gcn_user_residual | 0.133184 | 0.176563±0.006804 | 47.77% |
| bpr_edge | 0.136462 | 0.173068±0.010467 | 46.74% |
| bpr_user | 0.133950 | 0.168417±0.005094 | 44.67% |

## 消融与不确定性

edge/user是边遍历/均匀用户正片采样；uniform/residual是ego/邻居0.5/0.5与0.8/0.2。单变量控制保持训练更新上限相同。按用户先平均seed再bootstrap；下面是探索性、未多重校正的95%区间，不作为普遍显著提升。

- gcn_edge_residual_minus_gcn_edge_uniform：NDCG差值0.001107，区间[-0.004251, 0.006324]。
- gcn_edge_uniform_minus_bpr_edge：NDCG差值0.005745，区间[-0.003597, 0.016709]。
- gcn_edge_uniform_minus_gcn_edge_uniform：NDCG差值0.000000，区间[0.000000, 0.000000]。
- gcn_user_residual_minus_gcn_edge_residual：NDCG差值-0.003357，区间[-0.020510, 0.013888]。
- gcn_user_residual_minus_gcn_user_uniform：NDCG差值0.003722，区间[-0.008648, 0.014744]。
- gcn_user_uniform_minus_gcn_edge_uniform：NDCG差值-0.005972，区间[-0.018603, 0.006434]。

## 验证、成本与限制

真实训练18次、实际386epoch；18权重重载，69264原始用户指标重算，480历史文件及31历史权重SHA保持。完整回归311项通过。新增API0、下载0、云费用0；本地CPU墙钟不是线上latency。

前置测试首次因系统临时目录写权限发生5个setup error；指定工作区新basetemp后通过。缺可选NumPy、TypedStorage弃用及实验CSR beta警告保留。原始失败日志不删。

原生CSR更慢，直接缓存转置只小幅变化；固定CSR的sum内核配合显式固定图反向通过独立损失/梯度和真实图检查。kernel报告只测相同batch计算，不将比例冒称全训练/在线加速；GPU/Linux未验证。

串行相同初始权重、样本、227次Adam更新的一轮训练循环：原COO 37.67秒，缓存CSR sum 5.36秒，观测7.02倍/耗时减少85.76%。包含采样、张量、loss/backward/optimizer，不含数据/模型准备、valid及冷启动；仅一次固定顺序配对，不是完整训练或线上加速。最大参数差异8.94e-08，最大batch loss差异5.96e-08。

采样干预检查：最活跃约10%用户占原边35.43%，均匀用户采样占10.04%；样本数量相同且全来自train。改变分布真实生效，不代表准确率必然提高。

既有独立锁定环境16项新专项通过，真实残差GCN seed7重载和384人选择指标精确一致；共享checkout/数据/权重，未新建环境或再次从零重训。

本轮未跑新的真实LLM、外部或历史test，正式独立质量提升仍NOT EVALUATED。三seed复用相同用户，不等于三倍独立样本。发现负结果后不扩大搜索或自动接入。原T04阻塞保持。

## 复现

```powershell
& AgenticRec/.venv-t20/Scripts/python.exe -m agenticrec.lightgcn_optimization train --plan AgenticRec/configs/lightgcn_optimization_20261011.json --trial <registered_name> --seed 42
# 完成全部6个seed42后freeze；再跑每条件seed7和2026，最后audit。已有run拒绝覆盖。
& AgenticRec/.venv-t20/Scripts/python.exe -m agenticrec.lightgcn_optimization freeze --plan AgenticRec/configs/lightgcn_optimization_20261011.json
& AgenticRec/.venv-t20/Scripts/python.exe -m agenticrec.lightgcn_optimization audit --plan AgenticRec/configs/lightgcn_optimization_20261011.json
& AgenticRec/.venv-t20/Scripts/python.exe AgenticRec/scripts/report_lightgcn_optimization.py
```

层加权依据[原论文](https://arxiv.org/abs/2002.02126)，均匀用户采样依据[固定作者代码](https://github.com/gusye1234/LightGCN-PyTorch/blob/947ca2b3b1d2d3545b114145710cb06c4e57b3d2/code/utils.py)。属于经典方法的本项目验证，不称算法首创。
