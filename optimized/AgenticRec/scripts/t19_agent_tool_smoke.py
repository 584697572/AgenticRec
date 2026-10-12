"""Offline A/O smoke using the real frozen LightGCN recommendation pipeline."""

import hashlib
import json

from agenticrec.adapters.llm import ChatAdapter
from agenticrec.agent.router import RoutingRequest
from agenticrec.config import LLMBudget
from agenticrec.evaluation.system_factory import build_benchmark_agent_loop
from agenticrec.pipeline import FixedRecommendationPipeline, FixedRequest
from agenticrec.testing import FakeLLM


def _payload(*, user_id=None):
    return {
        "schema_version": 1,
        "user_id": user_id,
        "history_authorized": False,
        "liked_item_ids": [],
        "disliked_item_ids": [],
        "seen_item_ids": [],
        "constraints": {
            "include_genres": ["Action"],
            "exclude_genres": [],
            "year_min": None,
            "year_max": None,
            "excluded_item_ids": [],
            "exclude_seen": True,
            "k": 3,
            "required_fields": [],
        },
    }


def _plan(payload):
    return json.dumps({
        "plan": [{
            "step_id": "recommend-1",
            "tool_name": "recommend",
            "arguments": {"request": payload},
        }]
    }, sort_keys=True, separators=(",", ":"))


def _adapter(*responses):
    fake = FakeLLM(list(responses))
    return ChatAdapter(fake, LLMBudget()), fake


def _response_sha256(response):
    encoded = json.dumps(
        response, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main():
    pipeline = FixedRecommendationPipeline.from_frozen("lightgcn", 42)
    request = FixedRequest.parse(_payload())
    routing = RoutingRequest.from_fixed("t19-agent-tool-smoke", request)

    planner_a, fake_a = _adapter(_plan(_payload()))
    report_a = build_benchmark_agent_loop(
        "A", fixed_pipeline=pipeline, planner=planner_a,
        planned_cost_ceiling=0,
    ).run(routing)

    planner_o, fake_o = _adapter()
    report_o = build_benchmark_agent_loop(
        "O", fixed_pipeline=pipeline, planner=planner_o,
        planned_cost_ceiling=0,
    ).run(routing)

    changed = _payload(user_id=1)
    planner_rejected, _fake_rejected = _adapter(_plan(changed))
    rejected = build_benchmark_agent_loop(
        "A", fixed_pipeline=pipeline, planner=planner_rejected,
        planned_cost_ceiling=0,
    ).run(routing)

    recommendations = report_a.response["recommendations"]
    result = {
        "status": "PASS",
        "scope": "real_fixed_pipeline_execution_not_system_quality",
        "model": "lightgcn",
        "seed": 42,
        "A": {
            "status": report_a.status,
            "planner_calls": report_a.planner_calls,
            "tool_calls": report_a.tool_calls,
            "remote_api_requests": report_a.remote_api_requests,
            "recommendation_count": len(recommendations),
            "action_constraint_satisfied": all(
                "Action" in row["genres"] for row in recommendations
            ),
            "response_sha256": _response_sha256(report_a.response),
            "fixture_planner_calls": len(fake_a.calls),
        },
        "O": {
            "status": report_o.status,
            "planner_calls": report_o.planner_calls,
            "tool_calls": report_o.tool_calls,
            "remote_api_requests": report_o.remote_api_requests,
            "response_sha256": _response_sha256(report_o.response),
            "fixture_planner_calls": len(fake_o.calls),
        },
        "changed_identity_plan": {
            "status": rejected.status,
            "tool_calls": rejected.tool_calls,
        },
        "metrics": "NOT EVALUATED",
    }
    if not (
        report_a.status == "OK"
        and report_a.planner_calls == 1
        and report_a.tool_calls == 1
        and len(recommendations) == 3
        and result["A"]["action_constraint_satisfied"]
        and report_o.status == "OK"
        and report_o.planner_calls == 0
        and report_o.response == report_a.response
        and rejected.status == "INVALID_PLAN"
        and rejected.tool_calls == 0
    ):
        raise RuntimeError("agent recommendation tool smoke failed")
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
