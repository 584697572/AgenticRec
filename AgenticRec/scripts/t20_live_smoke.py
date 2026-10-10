"""One explicitly authorized live plan against frozen real recommendation resources.

Default mode never reads a key or sends a request. A run ID is single-use;
uncertain delivery is never retried by restarting the script.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

from agenticrec.config import parse_config

ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", args.run_id):
        parser.error("run-id must be a safe single directory name")
    profile = json.loads(args.profile.read_text(encoding="utf-8-sig"))
    if set(profile) != {"schema_version", "scope", "llm"} or profile["scope"] != "t20_release_smoke":
        parser.error("profile must have the t20_release_smoke scope")
    config = parse_config({"schema_version": profile["schema_version"], "llm": profile["llm"]}).llm
    if not args.live:
        print(json.dumps({"status": "NOT RUN", "remote_api_requests": 0,
                          "live_authorized": False, "scope": profile["scope"]}))
        return 0
    if (not config.allow_paid_api or config.api_request_cap != 1
            or config.max_retries != 0 or config.max_output_tokens > 1024):
        parser.error("live smoke requires explicit authorization, one request, no retries, <=1024 output tokens")
    directory = ROOT / "artifacts/runs/t20" / args.run_id
    if directory.exists():
        # Do not read/rewrite its evidence or allocate a second request.
        print(json.dumps({"status": "ALREADY_STARTED", "new_remote_api_requests": 0}))
        return 3
    from agenticrec.evaluation.cost_limits import PER_REQUEST_COST_CNY
    if config.money_budget < PER_REQUEST_COST_CNY:
        parser.error("profile budget does not cover the conservative request reservation")
    import torch
    from agenticrec.adapters.account_balance import BalanceClient, BalanceGuard, GuardedTransport
    from agenticrec.adapters.providers import build_live_chat_adapter
    from agenticrec.agent.loop import AgentLimits
    from agenticrec.agent.router import RoutingRequest
    from agenticrec.evaluation.live_benchmark import AuditedTransport, append_json, write_json
    from agenticrec.evaluation.system_factory import build_benchmark_agent_loop
    from agenticrec.pipeline import FixedRecommendationPipeline, FixedRequest
    from agenticrec.runtime.secrets import read_api_key

    torch.set_num_threads(1)
    pipeline = FixedRecommendationPipeline.from_frozen("lightgcn", 42)
    request = FixedRequest.from_json_file(ROOT / "AgenticRec/configs/fixed_request.example.json")
    key = read_api_key()
    if not key:
        parser.error("OPENAI_API_KEY missing; contents are not logged")
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "intent.json", {"scope": profile["scope"], "request_cap": 1,
               "profile_sha256": hashlib.sha256(args.profile.read_bytes()).hexdigest()}, immutable=True)
    guard = BalanceGuard(BalanceClient(key), observer=lambda row: append_json(directory / "balance.jsonl", row))
    initial = guard.refresh()
    adapter = build_live_chat_adapter(config, api_key=key, seed=42)
    audit = AuditedTransport(adapter.transport, directory / "llm_attempts.jsonl")
    audit.current_episode = "release-structured-example"
    adapter.transport = GuardedTransport(audit, guard)
    loop = build_benchmark_agent_loop("A", fixed_pipeline=pipeline, planner=adapter,
                                      planned_cost_ceiling=PER_REQUEST_COST_CNY)
    loop.limits = AgentLimits(max_planner_calls=1, max_tool_calls=4, max_replans=0)
    result = loop.run(RoutingRequest.from_fixed("release-example", request))
    write_json(directory / "trace.json", result.to_dict())
    write_json(directory / "ledger.json", adapter.ledger.snapshot())
    final = guard.refresh()
    recommendations = (result.response or {}).get("recommendations", [])
    valid = (result.status == "OK" and result.tool_calls >= 1 and len(recommendations) == 5
             and all("Comedy" in row["genres"] and "Horror" not in row["genres"]
                     and 1990 <= row["year"] <= 2000 for row in recommendations))
    report = {"status": "VERIFIED" if valid else "NOT VERIFIED", "scope": "T20_live_tool_smoke_not_quality",
              "model": config.model_id, "nonthinking": True, "max_output_tokens": config.max_output_tokens,
              "remote_generation_requests": guard.requests, "planner_status": result.status,
              "tool_calls": result.tool_calls, "recommendation_count": len(recommendations),
              "constraints_verified": valid, "ledger": adapter.ledger.snapshot(),
              "initial_balance": initial, "final_balance": final,
              "actual_cost_cny": None, "actual_cost_reason": "provider_bill_not_queried"}
    write_json(directory / "report.json", report)
    print(json.dumps(report, ensure_ascii=True, sort_keys=True))
    return 0 if valid else 3


if __name__ == "__main__":
    raise SystemExit(main())
