import hashlib
import json

from agenticrec.adapters.llm import ChatAdapter, TokenUsage, TransportReply
from agenticrec.agent.text_parser import TextRequestParser
from agenticrec.config import LLMBudget
from agenticrec.evaluation import runner
from agenticrec.evaluation.episodes import PublicEpisode
from agenticrec.evaluation.system_executor import BenchmarkSystemExecutor


class LiveFixtureTransport:
    is_live = True

    def __init__(self, content):
        self.content = content
        self.calls = []

    def chat(self, messages, **options):
        self.calls.append((messages, options))
        return TransportReply(
            content=self.content,
            usage=TokenUsage(20, 5, 25),
            usage_unknown_reason=None,
            cost=None,
            cost_unknown_reason="provider_bill_not_queried",
            is_fixture=False,
            remote_api_requests=1,
        )


class FixedResponse:
    def to_dict(self):
        return {
            "status": "OK",
            "recommendations": [{"item_id": 7}],
            "fallback_reason": None,
        }


class FixedPipeline:
    def recommend(self, _request):
        return FixedResponse()


def test_metered_text_parser_attempt_flows_through_resumable_runner(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    episode = PublicEpisode(
        "test-001", "test", "text", "user:1", "template:1",
        "explicit_filter", False,
        {
            "mode": "text",
            "template_id": "test-explicit-filter-1",
            "message": "Choose one Action movie.",
            "user_id": None,
            "history_authorized": False,
        },
        1,
    )
    public_path = tmp_path / "test.public.jsonl"
    public_path.write_text(
        json.dumps(episode.to_dict(), separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    parsed = {
        "request": {
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
    }
    transport = LiveFixtureTransport(json.dumps(parsed))
    config = LLMBudget(
        allow_paid_api=True,
        provider="fixture",
        model_id="fixture-model",
        api_request_cap=1,
        max_output_tokens=128,
        money_budget=1,
        max_retries=0,
    )
    adapter = ChatAdapter(transport, config)
    executor = BenchmarkSystemExecutor(
        "F",
        fixed_pipeline=FixedPipeline(),
        text_parser=TextRequestParser(adapter),
        ledger=adapter.ledger,
        planned_cost_ceiling=0.01,
    )
    spec = runner.RunSpec(
        run_id="fixture-F-seed7",
        benchmark_id="fixture",
        system="F",
        seed=7,
        split="test",
        public_episodes=public_path.name,
        public_sha256=hashlib.sha256(public_path.read_bytes()).hexdigest(),
        episode_count=1,
        authorized_request_cap=1,
        money_budget_cny=0.01,
        per_request_cost_ceiling_cny=0.01,
        live=True,
    )
    journal = tmp_path / "attempts.jsonl"

    report = runner.run_benchmark(spec, journal, executor)
    attempts = runner.load_completed_attempts(spec, journal)

    assert report["status"] == "COMPLETE"
    assert report["remote_api_requests"] == 1
    assert len(transport.calls) == 1
    assert len(attempts) == 1
    assert attempts[0].item_ids == (7,)
    assert attempts[0].request_count == 1
    assert attempts[0].input_tokens == 20
    assert attempts[0].output_tokens == 5
    evidence = journal.read_text(encoding="utf-8")
    assert "acceptable_item_ids" not in evidence
    assert "expected_statuses" not in evidence
