# T19 Benchmark (IN PROGRESS)

Generated from frozen per-user predictions for seeds 7, 42, and 2026. Values are mean +/- sample standard deviation across seeds.

| Method | Recall@10 | NDCG@10 | MRR@10 | HitRate@10 |
|---|---:|---:|---:|---:|
| Random | 0.002942 +/- 0.000706 | 0.014783 +/- 0.001392 | 0.039712 +/- 0.004042 | 0.120894 +/- 0.013106 |
| Popularity | 0.053370 +/- 0.000000 | 0.195795 +/- 0.000000 | 0.340002 +/- 0.000000 | 0.617400 +/- 0.000000 |
| BPR-MF | 0.055484 +/- 0.003434 | 0.203947 +/- 0.010365 | 0.353455 +/- 0.019045 | 0.623690 +/- 0.011338 |
| LightGCN | 0.054671 +/- 0.000524 | 0.195317 +/- 0.001138 | 0.340176 +/- 0.005948 | 0.618449 +/- 0.003779 |

Candidate protocol: full 3,469-item mapped MovieLens1M universe with train-seen filtering; 5,351 total users, 954 users with test relevance, and 4,397 reported separately without relevance.

LightGCN minus BPR-MF NDCG@10 by seed: [-0.012814969581793373, 0.001957421260007697, -0.015034257741914081]. The mean is negative; the paired interval below tests that loss. No LightGCN improvement claim is made.

Paired user bootstrap for LightGCN minus BPR-MF NDCG@10: mean -0.008631, 95% CI [-0.012850, -0.004344] over 954 users, averaging the three pre-registered seeds per user. The table was regenerated from frozen per-user Top-10 predictions; the artifact records their SHA256 and evaluator denominator.

## Interactive systems

| System | Strict Success | Constraint Precision | Fill@K | Requests | Latency p50/p95 |
|---|---:|---:|---:|---:|---:|
| U1 | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED |
| F | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED |
| A | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED |
| O | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED |

The paid run is blocked before request 1: the audited ceiling is 8,649 new requests, 27 remain authorized, and the shortfall is 8,622.

Machine-readable local evidence: `artifacts/runs/t19/model_summary.json` and `artifacts/runs/t19/dry_run_plan.json`.
