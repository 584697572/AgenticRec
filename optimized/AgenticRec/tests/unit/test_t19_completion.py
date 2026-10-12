import json
from dataclasses import replace

import pytest

from agenticrec.adapters.llm import ChatAdapter
from agenticrec.config import LLMBudget
from agenticrec.agent.router import RoutingRequest, fixed_request_payload
from agenticrec.evaluation.system_executor import BenchmarkSystemExecutor, FeedbackSchedule
from agenticrec.evaluation.system_factory import build_benchmark_agent_loop
from agenticrec.evaluation.episodes import PublicEpisode
from agenticrec.pipeline import FixedRequest
from agenticrec.testing import FakeLLM


def payload():
    return fixed_request_payload(FixedRequest.parse({
        "schema_version": 1, "user_id": None, "history_authorized": False,
        "liked_item_ids": [1], "constraints": {"k": 1},
    }))


def plan(request):
    return json.dumps({"plan": [{"step_id": "s1", "tool_name": "recommend",
                                 "arguments": {"request": request}}]})


class Pipeline:
    def __init__(self):
        self.requests = []

    def recommend(self, request):
        self.requests.append(request)
        class Response:
            def to_dict(self):
                return {"status": "OK", "recommendations": [{"item_id": 3}],
                        "fallback_reason": None}
        return Response()


def test_text_state_is_bound_after_first_tool_execution_and_feedback():
    first = payload()
    final = {**first, "disliked_item_ids": [2]}
    fake = FakeLLM([plan(first), plan(final)])
    adapter = ChatAdapter(fake, LLMBudget())
    pipeline = Pipeline()
    loop = build_benchmark_agent_loop("O", fixed_pipeline=pipeline, planner=adapter,
                                     planned_cost_ceiling=0)
    schedule = FeedbackSchedule({("test-001", 2): {
        "turn": 2, "kind": "explicit_feedback", "patch": {"disliked_item_ids": [2]}}})
    executor = BenchmarkSystemExecutor("O", fixed_pipeline=pipeline, agent_loop=loop,
                                       feedback_source=schedule, ledger=adapter.ledger)
    episode = PublicEpisode("test-001", "test", "text", "user:1", "test:feedback",
                            "exclusion_feedback", True,
                            {"mode": "text", "template_id": "test-1",
                             "message": "Suggest one alternative to item 1",
                             "user_id": None, "history_authorized": False}, 3)
    attempt = executor(episode)
    assert attempt.status == "OK" and attempt.turns == 3
    assert pipeline.requests[-1].liked_item_ids == (1,)
    assert pipeline.requests[-1].disliked_item_ids == (2,)
    assert attempt.preference_patch == {"disliked_item_ids": [2]}
    assert len(fake.calls) == 2
    last_prompt = json.loads(fake.calls[-1][1][1].split("Public request: ")[1])
    assert last_prompt["disliked_item_ids"] == [2]


def test_structured_direct_pipeline_invocations_are_counted():
    adapter = ChatAdapter(FakeLLM([]), LLMBudget())
    pipeline = Pipeline()
    loop = build_benchmark_agent_loop("O", fixed_pipeline=pipeline, planner=adapter,
                                     planned_cost_ceiling=0)
    executor = BenchmarkSystemExecutor("O", fixed_pipeline=pipeline, agent_loop=loop,
                                       ledger=adapter.ledger)
    episode = PublicEpisode("test-001", "test", "structured", "user:1", "test:direct",
                            "explicit_filter", False, payload(), 1)
    assert executor(episode).tool_calls == 1


def test_provider_disables_thinking_and_uses_seed():
    from agenticrec.adapters.openai_compatible import OpenAICompatibleTransport, HttpResponse
    captured = []
    def sender(_url, _headers, body, _timeout):
        captured.append(json.loads(body))
        return HttpResponse(200, b'{"choices":[{"message":{"content":"{}"}}]}')
    transport = OpenAICompatibleTransport(api_base="https://api.deepseek.com", api_key="fixture",
                                         sender=sender, seed=7)
    transport.chat([{"role": "user", "content": "JSON"}], model="deepseek-flash",
                   max_output_tokens=1024, timeout_seconds=30, sdk_max_retries=0)
    assert captured[0]["thinking"] == {"type": "disabled"}
    assert captured[0]["seed"] == 7


def test_openai_transport_does_not_receive_deepseek_thinking_parameter():
    from agenticrec.adapters.openai_compatible import OpenAICompatibleTransport, HttpResponse
    captured = []
    def sender(_url, _headers, body, _timeout):
        captured.append(json.loads(body))
        return HttpResponse(200, b'{"choices":[{"message":{"content":"{}"}}]}')
    transport = OpenAICompatibleTransport(api_base="https://api.openai.com/v1", api_key="fixture", sender=sender)
    transport.chat([{"role": "user", "content": "JSON"}], model="fixture", max_output_tokens=10,
                   timeout_seconds=1, sdk_max_retries=0)
    assert "thinking" not in captured[0]


def test_fault_schedule_contains_only_execution_flags_and_is_immutable():
    from agenticrec.evaluation.faults import FaultSchedule
    events = {"test-001": "ranking_timeout"}
    schedule = FaultSchedule(events)
    events["test-001"] = None
    assert schedule("test-001") == "ranking_timeout"
    assert "ranking_timeout" not in repr(schedule)
    with pytest.raises(ValueError):
        FaultSchedule({"test-001": "arbitrary_execution"})


def test_full_request_ceiling_counts_response_turns_only():
    from agenticrec.evaluation.runner import episode_request_ceiling
    episode = PublicEpisode("test-001", "test", "text", "user:1", "test:feedback",
                            "exclusion_feedback", True, {"message": "fixture"}, 3)
    assert episode_request_ceiling("U1", episode) == 2
    assert episode_request_ceiling("F", episode) == 1
    assert episode_request_ceiling("A", episode) == 4
    assert episode_request_ceiling("O", episode) == 4
    assert episode_request_ceiling("no_content", episode) == 4


@pytest.mark.parametrize("system,component", [("F", "parser_calls"), ("U1", "planner_calls")])
def test_failed_llm_invocation_remains_in_component_trace(system, component):
    from agenticrec.agent.text_parser import TextRequestParser
    from agenticrec.evaluation.upstream_turn import UpstreamRebuiltTurn
    adapter = ChatAdapter(FakeLLM(["not valid JSON"]), LLMBudget())
    pipeline = Pipeline()
    extra = ({"fixed_pipeline": pipeline, "text_parser": TextRequestParser(adapter)}
             if system == "F" else {"upstream_turn": UpstreamRebuiltTurn(adapter, pipeline, planned_cost_ceiling=0)})
    executor = BenchmarkSystemExecutor(system, ledger=adapter.ledger, **extra)
    episode = PublicEpisode("test-001", "test", "text", "user:1", "test:direct",
                            "explicit_filter", False,
                            {"mode": "text", "template_id": "test-1", "message": "Suggest one movie", "user_id": None,
                             "history_authorized": False}, 1)
    assert executor(episode).status == "INVALID_PARSE"
    assert sum(turn.get(component, 0) for turn in executor.last_trace["turns"]) == 1
    if system == "F":
        assert sum(turn.get("planner_calls", 0) for turn in executor.last_trace["turns"]) == 0
