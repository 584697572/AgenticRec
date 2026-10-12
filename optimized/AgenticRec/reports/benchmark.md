# T19 Benchmark (COMPLETED)

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

Real DeepSeek deepseek-flash, non-thinking, 1024-token cap, zero network retries; frozen 150-episode synthetic test, three seeds. Mean +/- sample SD. All failures remain in the denominator.

| System | Strict Success | Constraint Precision | Fill@K | Requests/run | Tools/run | Latency p50/p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| U1 | 0.6244 +/- 0.0102 | 0.7089 +/- 0.0102 | 0.5311 +/- 0.0139 | 190.0000 +/- 0.0000 | 337.3333 +/- 4.9329 | 1359.5/3088.3 |
| F | 0.6867 +/- 0.0067 | 0.8000 +/- 0.0000 | 0.6000 +/- 0.0000 | 75.0000 +/- 0.0000 | 146.0000 +/- 0.0000 | 380.3/1224.0 |
| A | 0.8133 +/- 0.0067 | 0.9600 +/- 0.0000 | 0.7600 +/- 0.0000 | 190.0000 +/- 0.0000 | 171.0000 +/- 0.0000 | 1122.2/2322.7 |
| O | 0.8133 +/- 0.0067 | 0.9600 +/- 0.0000 | 0.7600 +/- 0.0000 | 99.0000 +/- 0.0000 | 163.3333 +/- 0.5774 | 393.3/2161.3 |

U0: NOT RUN; original resource semantics/license remain unverified. U1 is a rebuilt Plan First JSON adapter with common frozen recommendation pipeline followed by original ToolBox/Buffer/Map; no paper prompt or result claim.

O minus A paired episode bootstrap: {"constraint_precision":{"bootstrap_samples":10000,"bootstrap_seed":42,"ci_lower":0.0,"ci_upper":0.0,"confidence":0.95,"mean_difference":0.0,"paired_count":150},"fill_at_k":{"bootstrap_samples":10000,"bootstrap_seed":42,"ci_lower":0.0,"ci_upper":0.0,"confidence":0.95,"mean_difference":0.0,"paired_count":150},"latency_ms":{"bootstrap_samples":10000,"bootstrap_seed":42,"ci_lower":-778.6618333333532,"ci_upper":-551.087500000021,"confidence":0.95,"mean_difference":-662.1755555555542,"paired_count":150},"request_count":{"bootstrap_samples":10000,"bootstrap_seed":42,"ci_lower":-0.72,"ci_upper":-0.5,"confidence":0.95,"mean_difference":-0.6066666666666667,"paired_count":150},"strict_success":{"bootstrap_samples":10000,"bootstrap_seed":42,"ci_lower":0.0,"ci_upper":0.0,"confidence":0.95,"mean_difference":0.0,"paired_count":150}}

H2 noninferiority at -0.02: supported

Provider bill: NOT QUERIED. Peak-rate token estimate: 4.997432 CNY over 2850 request attempts; 0 episodes have unknown usage. Conservative planning reservation: 70.771200 CNY. Unknown usage remains null; no partial token sum is presented as a total cost ceiling.

Latency includes parsing, tools and failures; serial episodes, prewarmed model and U1 worker. One-time model/legacy-worker warmup is excluded; each U1 request still creates fresh candidate state. This synthetic offline benchmark does not measure online satisfaction or CTR.
