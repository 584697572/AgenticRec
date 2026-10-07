import json

import pytest

from agenticrec.adapters.llm import ChatAdapter, StrictObjectSchema
from agenticrec.agent.executor import PlanExecutor, ToolDefinition
from agenticrec.agent.loop import AgentLimits, AgentLoop
from agenticrec.agent.router import Router, RouterPolicy, SystemMode
from agenticrec.agent.text_parser import TextRequestParser
from agenticrec.config import LLMBudget
from agenticrec.evaluation.episodes import PublicEpisode
from agenticrec.evaluation.system_executor import (
    BenchmarkSystemExecutor,
    FeedbackSchedule,
    TurnResult,
)
from agenticrec.evaluation.episodes import PrivateEpisode
from agenticrec.ranking.constraints import Constraints
from agenticrec.pipeline import FixedRequest
from agenticrec.protocols import ToolResult
from agenticrec.testing import FakeLLM


class Reply:
    def __init__(self, item_ids, status="OK", fallback_reason=None):
        self.value = {
            "status": status,
            "recommendations": [{"item_id": item_id} for item_id in item_ids],
            "fallback_reason": fallback_reason,
        }

    def to_dict(self):
        return dict(self.value)


class RecordingPipeline:
    def __init__(self):
        self.requests = []

    def recommend(self, request):
        self.requests.append(request)
        blocked = set(request.disliked_item_ids)
        selected = next(item for item in (1, 2, 3) if item not in blocked)
        return Reply([selected])


def structured_episode(*, multi_turn=False):
    return PublicEpisode(
        "test-001", "test", "structured", "user:1", "template:1",
        "exclusion_feedback" if multi_turn else "explicit_filter", multi_turn,
        {
            "schema_version": 1,
            "user_id": None,
            "history_authorized": False,
            "liked_item_ids": [],
            "disliked_item_ids": [],
            "seen_item_ids": [],
            "constraints": {"k": 1},
        },
        3 if multi_turn else 1,
    )


def text_episode(*, multi_turn=False):
    return PublicEpisode(
        "test-002", "test", "text", "user:2", "template:2",
        "exclusion_feedback" if multi_turn else "explicit_filter", multi_turn,
        {
            "mode": "text",
            "template_id": "test-explicit-filter-1",
            "message": "Choose one Action movie.",
            "user_id": None,
            "history_authorized": False,
        },
        3 if multi_turn else 1,
    )


def full_request():
    return {
        "schema_version": 1,
        "user_id": None,
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
            "k": 1,
            "required_fields": [],
        },
    }


def text_parser(*responses):
    fake = FakeLLM(list(responses))
    adapter = ChatAdapter(fake, LLMBudget())
    return TextRequestParser(adapter), adapter, fake


def plan(item_id=1, step_id="s1"):
    return json.dumps({
        "plan": [{
            "step_id": step_id,
            "tool_name": "recommend",
            "arguments": {"item_id": item_id},
        }]
    })


def agent_loop(mode, pipeline, responses, *, max_replans=1):
    fake = FakeLLM(list(responses))
    adapter = ChatAdapter(fake, LLMBudget())

    def recommend(arguments):
        request = FixedRequest.parse({
            "schema_version": 1,
            "user_id": None,
            "history_authorized": False,
            "liked_item_ids": [],
            "disliked_item_ids": [],
            "seen_item_ids": [],
            "constraints": {"k": 1},
        })
        reply = pipeline.recommend(request).to_dict()
        reply["recommendations"] = [{"item_id": arguments["item_id"]}]
        return ToolResult(True, data=reply, provenance="fixture")

    loop = AgentLoop(
        Router(RouterPolicy(system_mode=mode, agent_threshold=0.75)),
        pipeline,
        adapter,
        PlanExecutor([ToolDefinition("recommend", {"item_id": int}, recommend)]),
        limits=AgentLimits(
            max_planner_calls=1 if max_replans == 0 else 2,
            max_tool_calls=4,
            max_replans=max_replans,
        ),
        planned_cost_ceiling=0,
    )
    return loop, adapter, fake


def feedback(_episode_id, turn):
    assert turn == 2
    return {
        "turn": 2,
        "kind": "explicit_feedback",
        "patch": {"disliked_item_ids": [1]},
    }


def test_fixed_structured_path_has_zero_llm_calls():
    pipeline = RecordingPipeline()
    parser, adapter, fake = text_parser()
    executor = BenchmarkSystemExecutor(
        "F", fixed_pipeline=pipeline, text_parser=parser, ledger=adapter.ledger
    )

    attempt = executor(structured_episode())

    assert attempt.status == "OK" and attempt.item_ids == (1,)
    assert attempt.turns == 1 and attempt.tool_calls == 0
    assert attempt.request_count == 0 and fake.calls == []
    assert len(pipeline.requests) == 1


def test_fixed_text_path_parses_once_on_the_shared_ledger():
    pipeline = RecordingPipeline()
    parser, adapter, fake = text_parser(json.dumps({"request": full_request()}))
    executor = BenchmarkSystemExecutor(
        "F", fixed_pipeline=pipeline, text_parser=parser, ledger=adapter.ledger
    )

    attempt = executor(text_episode())

    assert attempt.status == "OK" and attempt.item_ids == (1,)
    assert attempt.request_count == 0 and len(fake.calls) == 1
    assert adapter.ledger.snapshot()["fixture_attempts"] == 1
    assert attempt.input_tokens is None and attempt.output_tokens is None


def test_always_agent_and_ours_use_distinct_routes_on_same_structured_input():
    pipeline_a = RecordingPipeline()
    loop_a, ledger_a, fake_a = agent_loop(
        SystemMode.ALWAYS_AGENT, pipeline_a, [plan(2)]
    )
    attempt_a = BenchmarkSystemExecutor(
        "A", fixed_pipeline=pipeline_a, agent_loop=loop_a, ledger=ledger_a.ledger
    )(structured_episode())

    pipeline_o = RecordingPipeline()
    loop_o, ledger_o, fake_o = agent_loop(SystemMode.OURS_ROUTER, pipeline_o, [])
    attempt_o = BenchmarkSystemExecutor(
        "O", fixed_pipeline=pipeline_o, agent_loop=loop_o, ledger=ledger_o.ledger
    )(structured_episode())

    assert attempt_a.item_ids == (2,) and len(fake_a.calls) == 1
    assert attempt_o.item_ids == (1,) and fake_o.calls == []
    assert attempt_a.tool_calls == 1 and attempt_o.tool_calls == 0


def test_ours_text_is_interpreted_inside_the_planner_call():
    pipeline = RecordingPipeline()
    loop, adapter, fake = agent_loop(SystemMode.OURS_ROUTER, pipeline, [plan(3)])
    attempt = BenchmarkSystemExecutor(
        "O", fixed_pipeline=pipeline, agent_loop=loop, ledger=adapter.ledger
    )(text_episode())

    assert attempt.item_ids == (3,) and attempt.tool_calls == 1
    assert len(fake.calls) == 1
    assert "Choose one Action movie" in fake.calls[0][1][1]


def test_multiturn_feedback_is_released_after_first_result_and_updates_fixed_state():
    pipeline = RecordingPipeline()
    parser, adapter, fake = text_parser(json.dumps({"request": full_request()}))
    calls = []

    def source(episode_id, turn):
        calls.append((episode_id, turn, len(pipeline.requests)))
        return feedback(episode_id, turn)

    executor = BenchmarkSystemExecutor(
        "F", fixed_pipeline=pipeline, text_parser=parser,
        feedback_source=source, ledger=adapter.ledger,
    )
    attempt = executor(text_episode(multi_turn=True))

    assert calls == [("test-002", 2, 1)]
    assert len(fake.calls) == 1
    assert [request.disliked_item_ids for request in pipeline.requests] == [(), (1,)]
    assert attempt.item_ids == (2,) and attempt.turns == 3
    assert attempt.preference_patch == {"disliked_item_ids": [1]}


def test_no_explicit_state_ablation_exposes_feedback_but_does_not_persist_patch():
    pipeline = RecordingPipeline()
    loop, adapter, fake = agent_loop(
        SystemMode.OURS_ROUTER, pipeline, [plan(1), plan(1, "s2")]
    )
    executor = BenchmarkSystemExecutor(
        "no_explicit_preference_state",
        fixed_pipeline=pipeline,
        agent_loop=loop,
        feedback_source=feedback,
        ledger=adapter.ledger,
    )
    attempt = executor(text_episode(multi_turn=True))

    assert len(fake.calls) == 2
    assert '"feedback_event"' in fake.calls[1][1][1]
    assert attempt.preference_patch is None
    assert attempt.item_ids == (1,)


def test_feedback_contract_rejects_evaluator_targets_before_second_system_call():
    pipeline = RecordingPipeline()
    parser, adapter, _fake = text_parser(json.dumps({"request": full_request()}))

    def leaking_source(_episode_id, _turn):
        return {
            "turn": 2,
            "kind": "explicit_feedback",
            "patch": {"disliked_item_ids": [1]},
            "acceptable_item_ids": [2],
        }

    executor = BenchmarkSystemExecutor(
        "F", fixed_pipeline=pipeline, text_parser=parser,
        feedback_source=leaking_source, ledger=adapter.ledger,
    )
    attempt = executor(text_episode(multi_turn=True))

    assert attempt.status == "INVALID_FEEDBACK"
    assert len(pipeline.requests) == 1
    assert attempt.item_ids == () and attempt.turns == 2


def test_missing_feedback_is_a_completed_failure_not_a_fake_three_turn_success():
    pipeline = RecordingPipeline()
    parser, adapter, _fake = text_parser(json.dumps({"request": full_request()}))
    executor = BenchmarkSystemExecutor(
        "F", fixed_pipeline=pipeline, text_parser=parser, ledger=adapter.ledger
    )
    attempt = executor(text_episode(multi_turn=True))

    assert attempt.status == "MISSING_FEEDBACK"
    assert attempt.turns == 2 and attempt.item_ids == ()
    assert len(pipeline.requests) == 1


def test_semantically_invalid_feedback_state_finishes_without_runner_interruption():
    value = structured_episode(multi_turn=True).to_dict()
    value["initial_input"]["liked_item_ids"] = [1]
    value["initial_input"]["disliked_item_ids"] = [1]
    episode = PublicEpisode.from_dict(value)
    pipeline = RecordingPipeline()
    parser, adapter, _fake = text_parser()
    executor = BenchmarkSystemExecutor(
        "F", fixed_pipeline=pipeline, text_parser=parser,
        feedback_source=feedback, ledger=adapter.ledger,
    )

    attempt = executor(episode)

    assert attempt.status == "INVALID_FEEDBACK"
    assert attempt.turns == 2 and attempt.item_ids == ()


def test_u1_uses_injected_upstream_turn_and_shared_ledger():
    fake = FakeLLM(['{"accepted":true}'])
    adapter = ChatAdapter(fake, LLMBudget())
    seen = []

    def upstream_turn(request):
        seen.append(request)
        adapter.chat_json(
            [{"role": "user", "content": "fixture upstream plan"}],
            schema=StrictObjectSchema({"accepted": bool}),
            planned_cost_ceiling=0,
        )
        return TurnResult("OK", (3,), tool_calls=2, fallback_kind=None)

    attempt = BenchmarkSystemExecutor(
        "U1", upstream_turn=upstream_turn, ledger=adapter.ledger
    )(structured_episode())

    assert len(seen) == 1 and len(fake.calls) == 1
    assert attempt.item_ids == (3,) and attempt.tool_calls == 2
    assert attempt.request_count == 0


def test_system_and_agent_loop_mode_must_match():
    pipeline = RecordingPipeline()
    loop, adapter, _fake = agent_loop(SystemMode.OURS_ROUTER, pipeline, [])
    with pytest.raises(ValueError, match="system and AgentLoop mode"):
        BenchmarkSystemExecutor(
            "A", fixed_pipeline=pipeline, agent_loop=loop, ledger=adapter.ledger
        )


def test_feedback_schedule_discards_targets_and_has_stable_identity():
    private = PrivateEpisode(
        episode_id="test-002",
        feasible=True,
        acceptable_item_ids=(999,),
        hard_constraints=Constraints(k=1),
        requested_k=1,
        required_count=1,
        expected_statuses=("OK",),
        expected_patch={"disliked_item_ids": [1]},
        evaluator_events=({
            "turn": 2,
            "kind": "explicit_feedback",
            "patch": {"disliked_item_ids": [1]},
        },),
    )
    first = FeedbackSchedule.from_private_episodes(
        [private], expected_episode_ids={"test-002"}
    )
    second = FeedbackSchedule.from_private_episodes(
        [private], expected_episode_ids={"test-002"}
    )

    assert first.sha256 == second.sha256
    assert first("test-002", 2)["patch"] == {"disliked_item_ids": [1]}
    released = first("test-002", 2)
    released["patch"]["disliked_item_ids"].append(2)
    assert first("test-002", 2)["patch"] == {"disliked_item_ids": [1]}
    assert "999" not in repr(first)
    with pytest.raises(KeyError):
        first("test-002", 3)
