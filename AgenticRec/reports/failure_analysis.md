# T19 Failure Analysis (IN PROGRESS)

Recommendation-model finding: LightGCN underperforms BPR-MF on NDCG@10 for seeds 7 and 2026, and is higher only for seed 42. The three-seed mean is therefore a negative gain; it is retained rather than omitted.

Interactive failure categories are frozen as invalid item, constraint violation, short fill, wrong status, preference-update mismatch, timeout/429/tool error, fallback, abstention, and missing attempt. Counts are **NOT EVALUATED** because the paid U1/F/A/O and ablation attempts have not run.

Attribution to recommendation model, hard constraints, or routing is **NOT EVALUATED**. It will require paired episode traces under identical conditions; no causal claim is made from the model-only table.
