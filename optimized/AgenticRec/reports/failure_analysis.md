# T19 Failure Analysis (COMPLETED)

Categories may overlap; counts are over all 450 attempts per condition, including unsuccessful ones. Examples omit raw preferences/titles.

Recommendation model finding remains negative: three-seed LightGCN NDCG@10 underperforms BPR-MF; it is retained.

## U1

Categories: {"acceptance_miss":38,"fallback_mismatch":16,"parse_or_plan_failure":131,"short_fill":121,"wrong_status":131}

Coverage: {"family:cold_start":{"episodes":30,"strict_successes":21},"family:constraint_conflict":{"episodes":45,"strict_successes":37},"family:exclusion_feedback":{"episodes":120,"strict_successes":66},"family:explicit_filter":{"episodes":60,"strict_successes":56},"family:personalized":{"episodes":60,"strict_successes":15},"family:seed_similar":{"episodes":60,"strict_successes":33},"family:tool_error":{"episodes":30,"strict_successes":10},"family:unknown_attribute":{"episodes":45,"strict_successes":43},"layer:structured":{"episodes":225,"strict_successes":183},"layer:text":{"episodes":225,"strict_successes":98}}

Examples: [{"categories":["wrong_status","parse_or_plan_failure"],"episode_id":"test-002","family":"constraint_conflict","layer":"text","seed":7,"status":"INVALID_PLAN"},{"categories":["short_fill","wrong_status","parse_or_plan_failure"],"episode_id":"test-004","family":"seed_similar","layer":"text","seed":7,"status":"INVALID_PLAN"},{"categories":["short_fill","wrong_status","parse_or_plan_failure"],"episode_id":"test-008","family":"personalized","layer":"text","seed":7,"status":"INVALID_PLAN"},{"categories":["short_fill","wrong_status","parse_or_plan_failure"],"episode_id":"test-010","family":"personalized","layer":"text","seed":7,"status":"INVALID_PLAN"},{"categories":["short_fill","wrong_status","parse_or_plan_failure"],"episode_id":"test-018","family":"exclusion_feedback","layer":"text","seed":7,"status":"INVALID_PLAN"}]

## F

Categories: {"acceptance_miss":51,"parse_or_plan_failure":90,"preference_update_mismatch":72,"short_fill":90,"wrong_status":90}

Coverage: {"family:cold_start":{"episodes":30,"strict_successes":30},"family:constraint_conflict":{"episodes":45,"strict_successes":45},"family:exclusion_feedback":{"episodes":120,"strict_successes":39},"family:explicit_filter":{"episodes":60,"strict_successes":60},"family:personalized":{"episodes":60,"strict_successes":27},"family:seed_similar":{"episodes":60,"strict_successes":42},"family:tool_error":{"episodes":30,"strict_successes":21},"family:unknown_attribute":{"episodes":45,"strict_successes":45},"layer:structured":{"episodes":225,"strict_successes":194},"layer:text":{"episodes":225,"strict_successes":115}}

Examples: [{"categories":["short_fill","wrong_status","parse_or_plan_failure"],"episode_id":"test-004","family":"seed_similar","layer":"text","seed":7,"status":"INVALID_PARSE"},{"categories":["acceptance_miss"],"episode_id":"test-010","family":"personalized","layer":"text","seed":7,"status":"OK"},{"categories":["short_fill","wrong_status","preference_update_mismatch","parse_or_plan_failure"],"episode_id":"test-014","family":"exclusion_feedback","layer":"text","seed":7,"status":"INVALID_PARSE"},{"categories":["short_fill","wrong_status","preference_update_mismatch","parse_or_plan_failure"],"episode_id":"test-018","family":"exclusion_feedback","layer":"text","seed":7,"status":"INVALID_PARSE"},{"categories":["short_fill","wrong_status","preference_update_mismatch","parse_or_plan_failure"],"episode_id":"test-030","family":"exclusion_feedback","layer":"text","seed":7,"status":"INVALID_PARSE"}]

## A

Categories: {"acceptance_miss":66,"parse_or_plan_failure":18,"short_fill":18,"wrong_status":18}

Coverage: {"family:cold_start":{"episodes":30,"strict_successes":30},"family:constraint_conflict":{"episodes":45,"strict_successes":45},"family:exclusion_feedback":{"episodes":120,"strict_successes":96},"family:explicit_filter":{"episodes":60,"strict_successes":60},"family:personalized":{"episodes":60,"strict_successes":27},"family:seed_similar":{"episodes":60,"strict_successes":42},"family:tool_error":{"episodes":30,"strict_successes":21},"family:unknown_attribute":{"episodes":45,"strict_successes":45},"layer:structured":{"episodes":225,"strict_successes":194},"layer:text":{"episodes":225,"strict_successes":172}}

Examples: [{"categories":["short_fill","wrong_status","parse_or_plan_failure"],"episode_id":"test-004","family":"seed_similar","layer":"text","seed":7,"status":"INVALID_PLAN"},{"categories":["acceptance_miss"],"episode_id":"test-010","family":"personalized","layer":"text","seed":7,"status":"OK"},{"categories":["acceptance_miss"],"episode_id":"test-030","family":"exclusion_feedback","layer":"text","seed":7,"status":"OK"},{"categories":["acceptance_miss"],"episode_id":"test-033","family":"personalized","layer":"structured","seed":7,"status":"OK"},{"categories":["short_fill","wrong_status","parse_or_plan_failure"],"episode_id":"test-034","family":"seed_similar","layer":"text","seed":7,"status":"INVALID_PLAN"}]

## O

Categories: {"acceptance_miss":66,"parse_or_plan_failure":18,"short_fill":18,"wrong_status":18}

Coverage: {"family:cold_start":{"episodes":30,"strict_successes":30},"family:constraint_conflict":{"episodes":45,"strict_successes":45},"family:exclusion_feedback":{"episodes":120,"strict_successes":96},"family:explicit_filter":{"episodes":60,"strict_successes":60},"family:personalized":{"episodes":60,"strict_successes":27},"family:seed_similar":{"episodes":60,"strict_successes":42},"family:tool_error":{"episodes":30,"strict_successes":21},"family:unknown_attribute":{"episodes":45,"strict_successes":45},"layer:structured":{"episodes":225,"strict_successes":194},"layer:text":{"episodes":225,"strict_successes":172}}

Examples: [{"categories":["short_fill","wrong_status","parse_or_plan_failure"],"episode_id":"test-004","family":"seed_similar","layer":"text","seed":7,"status":"INVALID_PLAN"},{"categories":["acceptance_miss"],"episode_id":"test-010","family":"personalized","layer":"text","seed":7,"status":"OK"},{"categories":["acceptance_miss"],"episode_id":"test-030","family":"exclusion_feedback","layer":"text","seed":7,"status":"OK"},{"categories":["acceptance_miss"],"episode_id":"test-033","family":"personalized","layer":"structured","seed":7,"status":"OK"},{"categories":["short_fill","wrong_status","parse_or_plan_failure"],"episode_id":"test-034","family":"seed_similar","layer":"text","seed":7,"status":"INVALID_PLAN"}]

## no_user_model

Categories: {"acceptance_miss":57,"parse_or_plan_failure":18,"short_fill":18,"wrong_status":18}

Coverage: {"family:cold_start":{"episodes":30,"strict_successes":30},"family:constraint_conflict":{"episodes":45,"strict_successes":45},"family:exclusion_feedback":{"episodes":120,"strict_successes":102},"family:explicit_filter":{"episodes":60,"strict_successes":60},"family:personalized":{"episodes":60,"strict_successes":33},"family:seed_similar":{"episodes":60,"strict_successes":39},"family:tool_error":{"episodes":30,"strict_successes":21},"family:unknown_attribute":{"episodes":45,"strict_successes":45},"layer:structured":{"episodes":225,"strict_successes":198},"layer:text":{"episodes":225,"strict_successes":177}}

Examples: [{"categories":["short_fill","wrong_status","parse_or_plan_failure"],"episode_id":"test-004","family":"seed_similar","layer":"text","seed":7,"status":"INVALID_PLAN"},{"categories":["acceptance_miss"],"episode_id":"test-010","family":"personalized","layer":"text","seed":7,"status":"OK"},{"categories":["acceptance_miss"],"episode_id":"test-033","family":"personalized","layer":"structured","seed":7,"status":"OK"},{"categories":["short_fill","wrong_status","parse_or_plan_failure"],"episode_id":"test-034","family":"seed_similar","layer":"text","seed":7,"status":"INVALID_PLAN"},{"categories":["acceptance_miss"],"episode_id":"test-039","family":"seed_similar","layer":"structured","seed":7,"status":"OK"}]

## no_explicit_preference_state

Categories: {"acceptance_miss":56,"constraint_violation":15,"parse_or_plan_failure":23,"preference_update_mismatch":120,"short_fill":38,"wrong_status":23}

Coverage: {"family:cold_start":{"episodes":30,"strict_successes":30},"family:constraint_conflict":{"episodes":45,"strict_successes":45},"family:exclusion_feedback":{"episodes":120,"strict_successes":86},"family:explicit_filter":{"episodes":60,"strict_successes":60},"family:personalized":{"episodes":60,"strict_successes":27},"family:seed_similar":{"episodes":60,"strict_successes":42},"family:tool_error":{"episodes":30,"strict_successes":21},"family:unknown_attribute":{"episodes":45,"strict_successes":45},"layer:structured":{"episodes":225,"strict_successes":188},"layer:text":{"episodes":225,"strict_successes":168}}

Examples: [{"categories":["short_fill","wrong_status","parse_or_plan_failure"],"episode_id":"test-004","family":"seed_similar","layer":"text","seed":7,"status":"INVALID_PLAN"},{"categories":["acceptance_miss"],"episode_id":"test-010","family":"personalized","layer":"text","seed":7,"status":"OK"},{"categories":["preference_update_mismatch"],"episode_id":"test-014","family":"exclusion_feedback","layer":"text","seed":7,"status":"OK"},{"categories":["preference_update_mismatch"],"episode_id":"test-018","family":"exclusion_feedback","layer":"text","seed":7,"status":"OK"},{"categories":["preference_update_mismatch"],"episode_id":"test-027","family":"exclusion_feedback","layer":"structured","seed":7,"status":"OK"}]

## no_replanning

Categories: {"acceptance_miss":65,"parse_or_plan_failure":18,"short_fill":18,"wrong_status":18}

Coverage: {"family:cold_start":{"episodes":30,"strict_successes":30},"family:constraint_conflict":{"episodes":45,"strict_successes":45},"family:exclusion_feedback":{"episodes":120,"strict_successes":97},"family:explicit_filter":{"episodes":60,"strict_successes":60},"family:personalized":{"episodes":60,"strict_successes":27},"family:seed_similar":{"episodes":60,"strict_successes":42},"family:tool_error":{"episodes":30,"strict_successes":21},"family:unknown_attribute":{"episodes":45,"strict_successes":45},"layer:structured":{"episodes":225,"strict_successes":194},"layer:text":{"episodes":225,"strict_successes":173}}

Examples: [{"categories":["short_fill","wrong_status","parse_or_plan_failure"],"episode_id":"test-004","family":"seed_similar","layer":"text","seed":7,"status":"INVALID_PLAN"},{"categories":["acceptance_miss"],"episode_id":"test-010","family":"personalized","layer":"text","seed":7,"status":"OK"},{"categories":["acceptance_miss"],"episode_id":"test-030","family":"exclusion_feedback","layer":"text","seed":7,"status":"OK"},{"categories":["acceptance_miss"],"episode_id":"test-033","family":"personalized","layer":"structured","seed":7,"status":"OK"},{"categories":["short_fill","wrong_status","parse_or_plan_failure"],"episode_id":"test-034","family":"seed_similar","layer":"text","seed":7,"status":"INVALID_PLAN"}]

## no_content

Categories: {"acceptance_miss":45,"parse_or_plan_failure":18,"short_fill":18,"wrong_status":18}

Coverage: {"family:cold_start":{"episodes":30,"strict_successes":30},"family:constraint_conflict":{"episodes":45,"strict_successes":45},"family:exclusion_feedback":{"episodes":120,"strict_successes":117},"family:explicit_filter":{"episodes":60,"strict_successes":60},"family:personalized":{"episodes":60,"strict_successes":27},"family:seed_similar":{"episodes":60,"strict_successes":42},"family:tool_error":{"episodes":30,"strict_successes":21},"family:unknown_attribute":{"episodes":45,"strict_successes":45},"layer:structured":{"episodes":225,"strict_successes":203},"layer:text":{"episodes":225,"strict_successes":184}}

Examples: [{"categories":["short_fill","wrong_status","parse_or_plan_failure"],"episode_id":"test-004","family":"seed_similar","layer":"text","seed":7,"status":"INVALID_PLAN"},{"categories":["acceptance_miss"],"episode_id":"test-010","family":"personalized","layer":"text","seed":7,"status":"OK"},{"categories":["acceptance_miss"],"episode_id":"test-033","family":"personalized","layer":"structured","seed":7,"status":"OK"},{"categories":["short_fill","wrong_status","parse_or_plan_failure"],"episode_id":"test-034","family":"seed_similar","layer":"text","seed":7,"status":"INVALID_PLAN"},{"categories":["acceptance_miss"],"episode_id":"test-043","family":"tool_error","layer":"structured","seed":7,"status":"OK"}]

These are observed failures, not proven causal attribution. Parser/plan failures, acceptance misses, constraint violations and state mismatches are separated to guide analysis. Paired ablations supply limited attribution; results do not establish online gains.
