import hashlib
import json

from agenticrec.adapters.llm import ChatAdapter
from agenticrec.agent.executor import PlanExecutor, ToolDefinition
from agenticrec.agent.loop import AgentLoop
from agenticrec.agent.router import Router, RouterPolicy, RoutingRequest
from agenticrec.config import LLMBudget
from agenticrec.evaluation import runner
from agenticrec.evaluation.episodes import PublicEpisode
from agenticrec.pipeline import FixedRequest
from agenticrec.protocols import ToolResult
from agenticrec.testing import FakeLLM


class FixedReply:
    def to_dict(self):
        return {"status": "OK", "recommendations": [{"item_id": 1}]}


class FixedPipeline:
    def recommend(self, _request):
        return FixedReply()


def test_public_runner_records_fixed_and_fake_agent_trajectories(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    episodes = [
        PublicEpisode(
            "test-001", "test", "structured", "user:1", "template:1",
            "explicit_filter", False,
            {"schema_version": 1, "constraints": {"k": 1}}, 1,
        ),
        PublicEpisode(
            "test-002", "test", "text", "user:2", "template:2",
            "exclusion_feedback", True,
            {"mode": "text", "message": "recommend an action movie"}, 3,
        ),
    ]
    public_path = tmp_path / "test.public.jsonl"
    public_path.write_text(
        "".join(json.dumps(item.to_dict(), separators=(",", ":")) + "\n"
                for item in episodes),
        encoding="utf-8", newline="\n",
    )
    fake = FakeLLM([json.dumps({
        "plan": [{
            "step_id": "recommend-1",
            "tool_name": "recommend",
            "arguments": {"query": "action"},
        }]
    })])
    loop = AgentLoop(
        Router(RouterPolicy()),
        FixedPipeline(),
        ChatAdapter(fake, LLMBudget()),
        PlanExecutor([ToolDefinition(
            "recommend", {"query": str},
            lambda _args: ToolResult(
                True,
                data={"status": "OK", "recommendations": [{"item_id": 3}]},
                provenance="fixture",
            ),
        )]),
        planned_cost_ceiling=0,
    )

    def execute(episode):
        if episode.layer == "structured":
            request = RoutingRequest.from_fixed(
                episode.episode_id, FixedRequest.parse(episode.initial_input)
            )
        else:
            request = RoutingRequest(
                episode.episode_id, episode.initial_input,
                agent_need_score=0.9, needs_planning=True,
            )
        report = loop.run(request)
        return runner.attempt_from_loop_report(
            episode.episode_id, report, turns=episode.max_turns
        )

    spec = runner.RunSpec(
        run_id="integration-O-seed7",
        benchmark_id="fixture",
        system="O",
        seed=7,
        split="test",
        public_episodes=public_path.name,
        public_sha256=hashlib.sha256(public_path.read_bytes()).hexdigest(),
        episode_count=2,
        authorized_request_cap=8,
        money_budget_cny=0,
        per_request_cost_ceiling_cny=0,
        live=False,
        feedback_sha256="e" * 64,
    )
    journal = tmp_path / "attempts.jsonl"
    result = runner.run_benchmark(spec, journal, execute)
    attempts = runner.load_completed_attempts(spec, journal)

    assert result["status"] == "COMPLETE"
    assert result["remote_api_requests"] == 0
    assert [(item.episode_id, item.item_ids, item.request_count) for item in attempts] == [
        ("test-001", (1,), 0),
        ("test-002", (3,), 0),
    ]
    assert len(fake.calls) == 1
    evidence = journal.read_text(encoding="utf-8")
    assert "acceptable_item_ids" not in evidence
    assert "target_item_ids" not in evidence
