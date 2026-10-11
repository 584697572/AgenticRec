# 融合失败与互补案例（2026-10-11）

从正式原始预测选出的6项对照、4个不同查询，仅作解释；不是代表性频率，不用于选参，不当6个独立用户。case_id可由私密行的split/user/state复算；不发布原用户、目标电影列表或数据包。

| Case | 数据 | 类别 | 状态 | 方法A NDCG | 方法B NDCG | 门控内容权重 |
|---|---|---|---|---:|---:|---:|
| e33dcf460912 | develop | rrf_gain | genre_constraint | rrf 0.630930 | legacy_no_content 0.000000 | 0 |
| 2b32e7924b56 | develop | rrf_failure | genre_constraint | rrf 0.173428 | legacy_no_content 0.790839 | 0 |
| e33dcf460912 | develop | gate_misses_content_benefit | genre_constraint | fixed_best 0.333333 | learned_42_full 0.000000 | 0 |
| ad47843d0422 | external | rrf_gain | genre_constraint | rrf 1.000000 | legacy_no_content 0.000000 | 0 |
| b2dd65270c04 | external | rrf_failure | genre_constraint | rrf 0.000000 | legacy_no_content 1.000000 | 0 |
| ad47843d0422 | external | gate_misses_content_benefit | genre_constraint | fixed_best 0.630930 | learned_42_full 0.000000 | 0 |

RRF能找回内容互补，也有真实退步用户。门控禁用内容会错过固定融合能命中的目标；该例支持“有漏选”现象，不证明类别不平衡是唯一因果。来源分数、名次、重叠、截断及最终输出全部保留于本地source_records.jsonl。
