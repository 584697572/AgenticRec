"""Run four real T12 structured scenarios and a standalone CLI invocation."""
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys

from agenticrec.data import digest
from agenticrec.pipeline import FixedRecommendationPipeline, FixedRequest
from agenticrec.training import ROOT, load_train_contract


def make_request(*, user_id=None, authorized=False, liked=(), constraints=None):
    return FixedRequest.parse({
        "schema_version": 1, "user_id": user_id, "history_authorized": authorized,
        "liked_item_ids": list(liked), "disliked_item_ids": [], "seen_item_ids": [],
        "constraints": constraints or {"k": 5},
    })


def validate_response(name, response):
    if response.llm_requests != 0 or response.remote_api_requests != 0 or response.planner != "not_invoked":
        raise AssertionError(name + " unexpectedly invoked a planner or remote model")
    for row in response.recommendations:
        if not row.title or not row.genres or row.rrf_score <= 0 or not row.source_ranks:
            raise AssertionError(name + " omitted required metadata/source/score evidence")


def main():
    manifest, n_users, n_items, _seen, edges = load_train_contract()
    pipeline = FixedRecommendationPipeline.from_frozen("lightgcn", 42)
    popular_seed = sorted(Counter(item for _user, item in edges).items(),
                          key=lambda pair: (-pair[1], pair[0]))[0][0]
    constrained = {"include_genres": ["Comedy"], "exclude_genres": ["Horror"],
                   "year_min": 1990, "year_max": 2000, "excluded_item_ids": [],
                   "exclude_seen": True, "k": 5,
                   "required_fields": ["title", "genres", "year"]}
    responses = {
        "success_known_user": pipeline.recommend(
            make_request(user_id=1, authorized=True, constraints=constrained)),
        "anonymous_seed": pipeline.recommend(
            make_request(liked=(popular_seed,), constraints={"k": 5})),
    }
    base = pipeline.recommend(make_request(constraints={"k": 5}))
    if base.status != "OK" or not base.recommendations:
        raise AssertionError("Could not establish exclusion scenario baseline")
    excluded_id = base.recommendations[0].item_id
    responses["explicit_exclusion"] = pipeline.recommend(make_request(
        constraints={"excluded_item_ids": [excluded_id], "k": 10}))
    responses["no_feasible"] = pipeline.recommend(make_request(
        constraints={"year_min": 1800, "year_max": 1850, "k": 5}))
    for name, response in responses.items():
        validate_response(name, response)
    if (responses["success_known_user"].status != "OK"
            or responses["success_known_user"].profile_source != "known_user"
            or responses["anonymous_seed"].status != "OK"
            or responses["anonymous_seed"].profile_source != "cold_start"
            or responses["explicit_exclusion"].status != "OK"
            or excluded_id in [row.item_id for row in responses["explicit_exclusion"].recommendations]
            or responses["no_feasible"].status != "NO_FEASIBLE_ITEMS"
            or responses["no_feasible"].recommendations):
        raise AssertionError("A required T12 scenario violated its contract")

    private_dir = ROOT / "artifacts/fixed_pipeline"
    private_dir.mkdir(parents=True, exist_ok=True)
    cli_request = private_dir / "cli_request_20261006.json"
    cli_request.write_text(json.dumps({
        "schema_version": 1, "user_id": None, "history_authorized": False,
        "liked_item_ids": [], "disliked_item_ids": [], "seen_item_ids": [],
        "constraints": {"include_genres": ["Comedy"], "exclude_genres": ["Horror"],
                        "year_min": 1990, "year_max": 2000, "k": 3}
    }, sort_keys=True) + "\n", encoding="utf-8")
    env = {key: value for key, value in os.environ.items()
           if not key.endswith(("_API_KEY", "_ACCESS_TOKEN", "_SECRET"))}
    completed = subprocess.run(
        [sys.executable, "-m", "agenticrec.cli", "recommend", "--request", str(cli_request)],
        cwd=ROOT / "AgenticRec", env=env, text=True, capture_output=True,
        timeout=120, check=False)
    if completed.returncode != 0:
        raise RuntimeError("Standalone fixed CLI failed: " + completed.stderr[-500:])
    cli_response = json.loads(completed.stdout)
    if (cli_response["status"] != "OK" or cli_response["llm_requests"] != 0
            or cli_response["remote_api_requests"] != 0
            or len(cli_response["recommendations"]) != 3):
        raise AssertionError("Standalone CLI response violated fixed-flow contract")

    trace = {"dataset": "ml-1m-v1", "model": "lightgcn_seed_42_selected",
             "excluded_item_id": excluded_id, "popular_seed_id": popular_seed,
             "responses": {name: response.to_dict() for name, response in responses.items()},
             "cli_response": cli_response}
    trace_path = private_dir / "t12_trace_20261006.json"
    trace_path.write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    public = {
        "status": "PASS", "task": "T12", "variant": "F_fixed_pipeline",
        "dataset": "ml-1m-v1", "candidate_count": n_items, "warm_user_count": n_users,
        "id_map_sha256": manifest["id_map_sha256"],
        "data_manifest_sha256": digest(ROOT / "artifacts/data_manifest.json"),
        "private_trace_path": trace_path.relative_to(ROOT).as_posix(),
        "private_trace_sha256": digest(trace_path),
        "input_mode": "structured_json", "planner": "not_invoked",
        "llm_requests": 0, "remote_api_requests": 0,
        "cli": {"status": cli_response["status"],
                "recommendation_count": len(cli_response["recommendations"])},
        "scenarios": {name: {"status": response.status,
                            "profile_source": response.profile_source,
                            "recommendation_count": len(response.recommendations),
                            "shortfall": response.hard_constraint_shortfall,
                            "fused_count": response.candidate_counts["fused"],
                            "metadata_evidence_complete": (
                                all(row.title and row.genres and row.source_ranks
                                    for row in response.recommendations)
                                if response.recommendations else None)}
                      for name, response in responses.items()},
        "holdout_labels_loaded": False,
        "natural_language_parsing": "NOT INVOKED; requires separately metered parser",
        "recommendation_quality": "NOT EVALUATED",
    }
    report = ROOT / "reports/fixed_pipeline/t12_20261006.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(public, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(public, sort_keys=True))


if __name__ == "__main__":
    main()
