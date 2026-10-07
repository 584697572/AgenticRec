"""T16 rule routing and bounded agent-loop acceptance trajectories."""
import json

import pytest

from agenticrec.adapters.llm import ChatAdapter
from agenticrec.agent.executor import PlanExecutor, ToolDefinition
from agenticrec.agent.loop import AgentLimits, AgentLoop
from agenticrec.agent.router import (
    CalibrationExample,
    Route,
    Router,
    RouterPolicy,
    RoutingRequest,
    SystemMode,
    select_agent_threshold,
)
from agenticrec.config import LLMBudget
from agenticrec.pipeline import FixedRequest
from agenticrec.protocols import ToolResult
from agenticrec.ranking.constraints import Constraints
from agenticrec.testing import FakeLLM


class FixedReply:
    def __init__(self, status="OK", recommendations=None):
        self.status = status
        self.recommendations = recommendations or [{"item_id": 1}]

    def to_dict(self):
        return {"status": self.status, "recommendations": self.recommendations}


class RecordingPipeline:
    def __init__(self):
        self.calls = []

    def recommend(self, request):
        self.calls.append(request)
        return FixedReply()


def fixed_request(*, personalized=False):
    return FixedRequest(
        constraints=Constraints(include_genres=("Action",), k=1),
        user_id=7 if personalized else None,
        history_authorized=personalized,
    )


def planner(*responses):
    transport = FakeLLM(responses)
    return ChatAdapter(transport, LLMBudget()), transport


def plan(tool_name="recommend", arguments=None, step_id="s1"):
    return json.dumps({
        "plan": [{
            "step_id": step_id,
            "tool_name": tool_name,
            "arguments": arguments or {"query": "action"},
        }]
    })


def agent_request(**overrides):
    values = {
        "request_id": "episode-1",
        "public_input": {"mode": "text", "message": "Recommend an action movie"},
        "agent_need_score": 0.9,
    }
    values.update(overrides)
    return RoutingRequest(**values)


def build_loop(responses, invoke, *, limits=AgentLimits(), mode=SystemMode.OURS_ROUTER):
    adapter, transport = planner(*responses)
    executor = PlanExecutor([
        ToolDefinition("recommend", {"query": str}, invoke),
    ])
    pipeline = RecordingPipeline()
    loop = AgentLoop(
        router=Router(RouterPolicy(system_mode=mode, agent_threshold=0.75)),
        fixed_pipeline=pipeline,
        planner=adapter,
        executor=executor,
        limits=limits,
        planned_cost_ceiling=0,
    )
    return loop, pipeline, transport


def test_ours_router_keeps_simple_and_personalized_requests_out_of_agent():
    adapter, transport = planner()
    pipeline = RecordingPipeline()
    executor = PlanExecutor([
        ToolDefinition("recommend", {"query": str}, lambda _: ToolResult(True, data={})),
    ])
    loop = AgentLoop(
        Router(RouterPolicy(agent_threshold=0.75)), pipeline, adapter, executor,
        planned_cost_ceiling=0,
    )

    direct = loop.run(RoutingRequest.from_fixed("direct", fixed_request()))
    personalized = loop.run(
        RoutingRequest.from_fixed("personalized", fixed_request(personalized=True))
    )

    assert direct.route is Route.DIRECT and direct.status == "OK"
    assert personalized.route is Route.PERSONALIZED and personalized.status == "OK"
    assert len(pipeline.calls) == 2
    assert transport.calls == []
    assert direct.planner_calls == personalized.planner_calls == 0


def test_unresolved_or_low_confidence_input_clarifies_without_side_effects():
    invoked = []
    loop, pipeline, transport = build_loop(
        [], lambda args: invoked.append(args) or ToolResult(True, data={})
    )

    result = loop.run(agent_request(agent_need_score=0.4, unresolved_fields=("year",)))

    assert result.route is Route.CLARIFY
    assert result.status == "CLARIFY"
    assert result.planner_calls == result.tool_calls == result.replans == 0
    assert pipeline.calls == [] and transport.calls == [] and invoked == []


def test_fake_llm_success_trajectory_is_replayable_and_fully_accounted():
    invoked = []

    def invoke(arguments):
        invoked.append(dict(arguments))
        return ToolResult(
            True,
            data={"status": "OK", "recommendations": [{"item_id": 3}]},
            provenance="fixture",
        )

    loop, _, transport = build_loop([plan()], invoke)
    result = loop.run(agent_request())

    assert result.route is Route.AGENT and result.status == "OK"
    assert result.response["recommendations"] == [{"item_id": 3}]
    assert result.planner_calls == 1 and result.tool_calls == 1
    assert result.replans == 0 and result.remote_api_requests == 0
    assert result.budget["fixture_attempts"] == 1
    assert invoked == [{"query": "action"}]
    replay = result.to_dict()
    assert replay["plans"][0]["plan"][0]["tool_name"] == "recommend"
    assert replay["executions"][0]["traces"][0]["result"]["provenance"] == "fixture"
    assert '"name":"recommend"' in transport.calls[0][0][1]
    assert "Recommend an action movie" in transport.calls[0][1][1]


def test_retryable_tool_failure_replans_once_then_succeeds():
    calls = []

    def invoke(arguments):
        calls.append(dict(arguments))
        if len(calls) == 1:
            return ToolResult(False, error_code="TEMPORARY", retryable=True,
                              provenance="fixture")
        return ToolResult(True, data={"status": "OK", "recommendations": []},
                          provenance="fixture")

    loop, _, transport = build_loop(
        [plan(arguments={"query": "first"}),
         plan(arguments={"query": "second"}, step_id="s2")],
        invoke,
    )
    result = loop.run(agent_request())

    assert result.status == "OK"
    assert result.planner_calls == 2 and result.tool_calls == 2
    assert result.replans == 1 and len(transport.calls) == 2
    assert calls == [{"query": "first"}, {"query": "second"}]
    assert "TEMPORARY" in transport.calls[1][1][1]


def test_second_retryable_failure_stops_without_a_third_plan():
    calls = []

    def invoke(arguments):
        calls.append(dict(arguments))
        return ToolResult(False, error_code="TEMPORARY", retryable=True,
                          provenance="fixture")

    loop, _, transport = build_loop(
        [plan(arguments={"query": "first"}),
         plan(arguments={"query": "second"}, step_id="s2")],
        invoke,
    )
    result = loop.run(agent_request())

    assert result.status == "TOOL_FAILED"
    assert result.planner_calls == 2 and result.tool_calls == 2
    assert result.replans == 1 and len(transport.calls) == 2
    assert result.error_code == "TEMPORARY"


def test_invalid_final_tool_result_is_validated_then_replanned_once():
    calls = []

    def invoke(arguments):
        calls.append(dict(arguments))
        if len(calls) == 1:
            return ToolResult(True, data={"unexpected": "shape"}, provenance="fixture")
        return ToolResult(True, data={"status": "OK", "recommendations": []},
                          provenance="fixture")

    loop, _, _ = build_loop(
        [plan(arguments={"query": "bad-shape"}),
         plan(arguments={"query": "valid-shape"}, step_id="s2")],
        invoke,
    )
    result = loop.run(agent_request())

    assert result.status == "OK"
    assert result.planner_calls == 2 and result.tool_calls == 2
    assert result.replans == 1
    assert calls == [{"query": "bad-shape"}, {"query": "valid-shape"}]


def test_nonretryable_tool_failure_never_replans():
    def invoke(_):
        return ToolResult(False, error_code="BAD_INPUT", retryable=False,
                          provenance="fixture")

    loop, _, transport = build_loop([plan()], invoke)
    result = loop.run(agent_request())

    assert result.status == "TOOL_FAILED" and result.error_code == "BAD_INPUT"
    assert result.planner_calls == 1 and result.tool_calls == 1
    assert result.replans == 0 and len(transport.calls) == 1


def test_tool_budget_exhaustion_fails_before_any_tool_call():
    invoked = []
    oversized = json.dumps({"plan": [
        {"step_id": "s1", "tool_name": "recommend", "arguments": {"query": "a"}},
        {"step_id": "s2", "tool_name": "recommend", "arguments": {"query": "b"}},
    ]})
    loop, _, transport = build_loop(
        [oversized],
        lambda args: invoked.append(args) or ToolResult(True, data={}),
        limits=AgentLimits(max_planner_calls=2, max_tool_calls=1, max_replans=1),
    )
    result = loop.run(agent_request())

    assert result.status == "BUDGET_EXHAUSTED"
    assert result.error_code == "tool_call_budget_exhausted"
    assert result.planner_calls == 1 and result.tool_calls == 0
    assert result.replans == 0 and len(transport.calls) == 1 and invoked == []


def test_invalid_plan_is_rejected_before_tool_side_effects():
    invoked = []
    loop, _, _ = build_loop(
        [plan(tool_name="unregistered")],
        lambda args: invoked.append(args) or ToolResult(True, data={}),
    )
    result = loop.run(agent_request())

    assert result.status == "INVALID_PLAN"
    assert result.error_code == "plan_validation_failed"
    assert result.tool_calls == 0 and result.replans == 0 and invoked == []


def test_fixed_always_agent_and_ours_are_distinct_comparable_modes():
    request = RoutingRequest.from_fixed("same-request", fixed_request())

    fixed = Router(RouterPolicy(system_mode=SystemMode.FIXED_PIPELINE)).decide(request)
    always = Router(RouterPolicy(system_mode=SystemMode.ALWAYS_AGENT)).decide(request)
    ours = Router(RouterPolicy(system_mode=SystemMode.OURS_ROUTER)).decide(request)

    assert fixed.route is Route.DIRECT
    assert always.route is Route.AGENT
    assert ours.route is Route.DIRECT


def test_structured_constraint_conflict_routes_to_clarification():
    conflict = FixedRequest(
        constraints=Constraints(
            include_genres=("Action",), exclude_genres=("action",), k=1
        )
    )
    decision = Router(RouterPolicy()).decide(
        RoutingRequest.from_fixed("conflict", conflict)
    )

    assert decision.route is Route.CLARIFY
    assert decision.reason == "conflict:included_and_excluded_genre_overlap"


def test_threshold_selection_accepts_development_only_and_is_deterministic():
    examples = [
        CalibrationExample(0.1, False),
        CalibrationExample(0.4, False),
        CalibrationExample(0.8, True),
        CalibrationExample(0.9, True),
    ]
    selected = select_agent_threshold(
        examples, candidates=(0.25, 0.5, 0.75), split="development"
    )
    assert selected.threshold == 0.75
    assert selected.correct == 4 and selected.total == 4
    with pytest.raises(ValueError, match="development"):
        select_agent_threshold(examples, candidates=(0.5,), split="test")
    with pytest.raises(ValueError, match="development"):
        RouterPolicy(calibration_split="test")


def test_hidden_evaluator_fields_cannot_enter_the_planner_prompt():
    with pytest.raises(ValueError, match="hidden evaluator field"):
        RoutingRequest(
            request_id="leak",
            public_input={"message": "hello", "expected_status": "OK"},
            agent_need_score=0.9,
        )
