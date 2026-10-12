# T19 Ablations (COMPLETED)

Same frozen episodes, three seeds, LLM, caps and hard constraints. always_agent reuses A.

| Condition | Strict Success | Constraint Precision | Fill@K | Requests/run |
|---|---:|---:|---:|---:|
| O | 0.8133 +/- 0.0067 | 0.9600 +/- 0.0000 | 0.7600 +/- 0.0000 | 99.0000 +/- 0.0000 |
| no_user_model | 0.8333 +/- 0.0000 | 0.9600 +/- 0.0000 | 0.7600 +/- 0.0000 | 99.0000 +/- 0.0000 |
| no_content | 0.8600 +/- 0.0067 | 0.9600 +/- 0.0000 | 0.7600 +/- 0.0000 | 99.0000 +/- 0.0000 |
| no_explicit_preference_state | 0.7911 +/- 0.0139 | 0.9422 +/- 0.0038 | 0.7422 +/- 0.0038 | 99.0000 +/- 0.0000 |
| always_agent | 0.8133 +/- 0.0067 | 0.9600 +/- 0.0000 | 0.7600 +/- 0.0000 | 190.0000 +/- 0.0000 |
| no_replanning | 0.8156 +/- 0.0038 | 0.9600 +/- 0.0000 | 0.7600 +/- 0.0000 | 99.0000 +/- 0.0000 |

no_user_model removes trained-user and collaborative scoring/recall, retaining content plus train-popularity fallback. no_content removes TF-IDF/genre seed scoring, retaining collaborative recall/rank plus the same fallback. Additional exploratory no_collaborative: NOT RUN; deferred before test execution to prioritize required experiments with existing funds.

no_explicit_preference_state exposes current feedback but does not commit its patch. no_replanning permits one planner attempt. Deterministic ranking_timeout is a local execution fixture with a content fallback; no failures are deliberately sent to the provider.

Paired confidence intervals are generated in the local summary.json; no gain is asserted when intervals are inconclusive.
