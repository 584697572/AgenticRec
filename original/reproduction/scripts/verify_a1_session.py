"""Recompute original fuzzy hits from private artifacts and emit a safe summary."""
import argparse
import ast
import hashlib
import json
import math
import re
from pathlib import Path
from unittest.mock import patch

import live_reproduction as app
from live_app_http import write_new
from local_api_key import read_api_key
from reproduction_session_http import AUTHORIZED_PROFILE, DATA_SHA256, valid_usage

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(directory, offline=False, scan_key=False):
    directory = Path(directory).resolve()
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "SUCCESS" and result["worker_exit_code"] == 0
    assert result["usage_is_fixture"] is offline
    assert result["budget"] == AUTHORIZED_PROFILE
    assert result["multiturn"]["status"] == "SUCCESS"
    assert result["multiturn"]["previous_input_and_answer_visible_on_second_turn"]
    assert len(result["multiturn"]["memory"]) == 4
    assert all(t["validation"]["passed"] for t in result["multiturn"]["turns"])
    assert result["real_key_in_worker"] is False and result["ui_served"] is False
    assert result["evaluation_data_sha256"] == DATA_SHA256
    for name, sha in result["provenance"]["source_sha256"].items():
        assert digest(ROOT / name) == sha, "source bytes changed after execution"
    data = app.frozen_data()
    # Execute the exact original metric body, without importing the model-loaded evaluator.
    from rapidfuzz import fuzz
    source = ROOT / "RecAI/InteRecAgent/eval/one_turn_eval.py"
    node = next(n for n in ast.parse(source.read_text(encoding="utf-8")).body
                if isinstance(n, ast.FunctionDef) and n.name == "hit_judge")
    namespace = {"re": re, "fuzz": fuzz, "Tuple": tuple}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    judge = namespace["hit_judge"]
    evaluations = {}
    for method, evaluation in result["evaluation"].items():
        saved_path = directory / ("original_%s_conversations.jsonl" % method)
        saved = [json.loads(line) for line in saved_path.read_bytes().splitlines()]
        assert len(saved) == len(evaluation["observations"]) == evaluation["samples"] == len(data) == 45
        assert saved == evaluation["conversations"]
        hits = []
        for i, (record, expected, observation) in enumerate(zip(saved, data, evaluation["observations"])):
            assert record["context"] == expected["context"] and record["target"] == expected["target"]
            assert record["answer"] == observation["answer"] and observation["index"] == i
            hit = bool(judge(record["answer"], record["target"]))
            assert hit is observation["hit"]
            hits.append(hit)
        assert sum(hits) == evaluation["hits"]
        assert evaluation["metrics_from_original_function"] == {"hit": sum(hits) / 45}
        observations = evaluation["observations"]
        latencies = sorted(r["elapsed_seconds"] for r in observations)
        evaluations[method] = {"metric_name": "upstream_text_hit" if method != "popularity" else "upstream_weighted_pop_text_hit",
            "samples": 45, "hits": sum(hits), "upstream_text_hit": sum(hits) / 45,
            "error_reply_count": sum(r["error_reply"] for r in observations),
            "no_map_count": sum(not r["mapped_ids"] for r in observations) if method == "recbot" else None,
            "duplicate_mapped_id_count": sum(len(set(r["mapped_ids"])) != len(r["mapped_ids"]) for r in observations) if method == "recbot" else None,
            "conversation_output_sha256": digest(saved_path),
            "sample_elapsed_seconds_p50": latencies[len(latencies) // 2],
            "sample_elapsed_seconds_p95": latencies[math.ceil(0.95 * len(latencies)) - 1],
            "sample_diagnostics": [{"index": r["index"], "hit": r["hit"],
                "error_reply": r["error_reply"], "mapped_count": len(r["mapped_ids"]),
                "tool_calls": len(r["tool_trace"])} for r in observations]}
    assert set(evaluations) == {"recbot", "random", "popularity"}
    calls = result["calls"]
    assert len(calls) == result["http_attempts"] <= 120
    assert len(calls) == len(list(directory.glob("request_*.json"))) == len(list(directory.glob("response_*.json")))
    totals = {k: 0 for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
    for call in calls:
        number = call["attempt"]
        reservation = json.loads((directory / ("request_%04d.json" % number)).read_text(encoding="utf-8"))
        saved = json.loads((directory / ("response_%04d.json" % number)).read_text(encoding="utf-8"))
        assert saved == call and saved["usage_is_fixture"] is offline
        payload = json.dumps(call["request"], ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
        assert reservation["request_body_sha256"] == hashlib.sha256(payload).hexdigest()
        assert call["http_status"] == 200 and call["error_type"] is None and call["provider_response_valid"]
        assert call["request"]["model"] == call["response"]["model"] == "deepseek-flash"
        assert call["request"]["thinking"] == {"type": "disabled"}
        assert valid_usage(call["response"]["usage"], call["request"]["max_tokens"])
        for k in totals:
            totals[k] += call["response"]["usage"][k]
    if offline:
        assert result["remote_api_requests"] == 0 and len(calls) == 94
        assert result["usage"] is None and result["estimated_cost_cny_peak_ceiling"] is None
    else:
        assert result["remote_api_requests"] == len(calls) and result["usage"] == totals
        assert result["original_evaluation"] == "A1 COMPLETED"
        estimate = (totals["prompt_tokens"] * 2 + totals["completion_tokens"] * 8) / 1000000
        assert abs(estimate - result["estimated_cost_cny_peak_ceiling"]) < 1e-10 and estimate <= 3
    assert result["actual_cost_cny"] is None and result["provider_billing_hard_cap_enforced"] is False
    assert result["paper_A2"] == result["shared_checkpoint_id_semantics"] == result["canonical_author_subset_equivalence"] == "NOT VERIFIED"
    before = {p.name: digest(p) for p in directory.iterdir() if p.is_file()}
    with patch.object(app, "resource_checks", side_effect=AssertionError("unexpected resource check")), \
         patch.object(app, "frozen_data", side_effect=AssertionError("unexpected input read")), \
         patch.object(app, "SessionHTTP", side_effect=AssertionError("unexpected HTTP")):
        blocked = app.execute(directory, "unused-fixture-placeholder")
    assert blocked["status"] == "BLOCKED" and blocked["new_http_attempts"] == 0
    assert before == {p.name: digest(p) for p in directory.iterdir() if p.is_file()}
    prior = json.loads((ROOT / "reproduction/.runtime/prior_live_evidence_snapshot_20261005.json").read_text(encoding="utf-8"))
    assert all(digest(ROOT / p) == sha for p, sha in prior.items())
    if scan_key:
        key = read_api_key(environ={}).encode()
        assert key
        for path in directory.iterdir():
            if path.is_file():
                assert key not in path.read_bytes(), "configured key found in private evidence"
    summary = {"status": "VERIFIED", "scope": "offline_fixture_only" if offline else "A1_original_recbot_evaluation",
        "raw_result_sha256": digest(directory / "result.json"), "evaluation_data_sha256": DATA_SHA256,
        "multiturn_verified": not offline, "multiturn_validation_passed": True,
        "multiturn_memory_entries": 4, "previous_input_and_answer_visible_on_second_turn": True,
        "evaluation": evaluations, "http_attempts": len(calls), "remote_api_requests": result["remote_api_requests"],
        "usage": result["usage"], "usage_is_fixture": offline,
        "actual_cost_cny": None, "actual_cost_reason": "provider_bill_not_queried",
        "estimated_cost_cny_peak_ceiling": result["estimated_cost_cny_peak_ceiling"],
        "price_source": result["price_source"], "price_checked_date": result["price_checked_date"],
        "provider_billing_hard_cap_enforced": False, "budget": AUTHORIZED_PROFILE,
        "seed": 42, "num_rec": 5, "demo_mode": "zero", "reflection": 0, "shortening": 0,
        "sdk_max_retries": 0, "upstream_retry_limits": 1, "http_transport_retries": 0,
        "repeat_gate_verified": True, "prior_evidence_files_unchanged": len(prior),
        "actual_key_scan_performed": scan_key, "source_sha256": result["provenance"]["source_sha256"],
        "total_elapsed_including_load_seconds": result["total_elapsed_including_load_seconds"],
        "canonical_author_subset_equivalence": "NOT VERIFIED", "shared_checkpoint_id_semantics": "NOT VERIFIED",
        "paper_A2": "NOT VERIFIED", "new_algorithm_evaluation": "NOT EVALUATED"}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--scan-local-key", action="store_true")
    parser.add_argument("--publish", type=Path)
    args = parser.parse_args()
    summary = verify(args.directory, args.offline, args.scan_local_key)
    if args.publish:
        output = args.publish.resolve()
        if not output.is_relative_to(ROOT / "reproduction/native_runs"):
            parser.error("public summary must be inside reproduction/native_runs")
        output.parent.mkdir(parents=True, exist_ok=True)
        write_new(output, summary)
    print(json.dumps({k: summary[k] for k in ("status", "scope", "http_attempts", "remote_api_requests", "usage",
        "estimated_cost_cny_peak_ceiling", "repeat_gate_verified", "prior_evidence_files_unchanged")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
