"""Create private real-data U1 traces and a public, title-free evidence summary."""
from collections import Counter
import json
from pathlib import Path

import pyarrow.parquet as pq

from agenticrec.adapters.model import KnownUserScorer, RoutingScorer, SessionSeedScorer
from agenticrec.adapters.upstream_bridge import UPSTREAM_SOURCES, run_upstream_plan
from agenticrec.data import digest
from agenticrec.training import ROOT, load_train_contract


def main():
    manifest, n_users, n_items, _seen, edges = load_train_contract()
    scorer = KnownUserScorer.from_artifacts("lightgcn", seed=42)
    session = SessionSeedScorer.from_frozen_train()
    router = RoutingScorer(scorer, session)
    table = pq.read_table(ROOT / "data/processed/ml-1m-v1/items.parquet",
                          columns=["item_id", "title"])
    titles = {int(i): title for i, title in zip(table["item_id"].to_pylist(),
                                                table["title"].to_pylist()) if i is not None}
    if set(titles) != set(range(1, n_items + 1)):
        raise ValueError("Frozen title IDs do not match train-only candidates")
    candidates = list(range(1, n_items + 1))
    seed_ids = [i for i, _ in sorted(Counter(i for _, i in edges).items(),
                                     key=lambda entry: (-entry[1], entry[0]))[:2]]
    routes = {
        "known_user": router.rank(candidates, 10, user_id=1, history_authorized=True),
        "anonymous_seed": router.rank(candidates, 10, liked_ids=seed_ids),
    }
    traces = {}
    for name, route in routes.items():
        traces[name] = {"model_candidate_ids": route.candidate_ids,
                        "aligned_model_scores": route.scores,
                        "ranked_ids": route.ranked_ids,
                        "seed_ids": seed_ids if name == "anonymous_seed" else [],
                        "upstream_execution": run_upstream_plan(route, titles, top_k=3)}
    private_path = ROOT / "artifacts/upstream_rebuilt/t10_trace_20261006.json"
    private_path.parent.mkdir(parents=True, exist_ok=True)
    private_path.write_text(json.dumps(traces, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    baseline = json.loads((ROOT / "reports/rec_baselines/lightgcn_seed_42.json").read_text(encoding="utf-8"))
    public = {
        "status": "PASS", "task": "T10", "variant": "U1_upstream_rebuilt",
        "upstream_commit": "0959ecb05b0794748426e73e6efc1b6b35ec433d",
        "upstream_source_sha256": UPSTREAM_SOURCES,
        "data_manifest_sha256": digest(ROOT / "artifacts/data_manifest.json"),
        "id_map_sha256": manifest["id_map_sha256"],
        "model": "lightgcn_seed_42_selected_layer_3",
        "checkpoint_sha256": baseline["trials"][str(baseline["selection"]["selected_layers"])]["checkpoint_sha256"],
        "candidate_count": n_items, "known_user_count": n_users,
        "private_trace_sha256": digest(private_path),
        "private_trace_path": private_path.relative_to(ROOT).as_posix(),
        "plan_source": "scripted_offline_fixture_not_llm",
        "remote_api_requests": 0,
        "traces": {name: {
            "profile_source": trace["upstream_execution"]["profile_source"],
            "fallback_reason": trace["upstream_execution"]["fallback_reason"],
            "toolbox_success": trace["upstream_execution"]["toolbox_success"],
            "selected_count": len(trace["upstream_execution"]["selected_ids"]),
            "mapped_count": len(trace["upstream_execution"]["mapped_ids"]),
            "buffer_reset": trace["upstream_execution"]["buffer_reset"],
            "credential_env_present": trace["upstream_execution"]["credential_env_present"],
        } for name, trace in traces.items()},
        "original_app": "NOT VERIFIED",
        "paper_reproduction": "NOT VERIFIED",
        "agent_llm_planning": "NOT EVALUATED",
    }
    report = ROOT / "reports/upstream_rebuilt/t10_20261006.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(public, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(json.dumps(public, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
