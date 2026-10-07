"""Mechanical F-system smoke on all frozen public episodes; no quality claim."""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from agenticrec.adapters.llm import ChatAdapter
from agenticrec.agent.text_parser import TextRequestParser
from agenticrec.config import LLMBudget
from agenticrec.evaluation.episodes import (
    load_private_episodes,
    load_public_episodes,
)
from agenticrec.evaluation.runner import RunSpec, load_completed_attempts, run_benchmark
from agenticrec.evaluation.system_executor import (
    BenchmarkSystemExecutor,
    FeedbackSchedule,
)
from agenticrec.testing import FakeLLM
from agenticrec.training import ROOT


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class _Response:
    item_id: int

    def to_dict(self):
        return {
            "status": "OK",
            "recommendations": [{"item_id": self.item_id}],
            "fallback_reason": None,
        }


class _FixturePipeline:
    def __init__(self):
        self.calls = 0

    def recommend(self, request):
        self.calls += 1
        blocked = set(request.disliked_item_ids) | set(request.constraints.excluded_item_ids)
        item_id = next(value for value in range(1, 20) if value not in blocked)
        return _Response(item_id)


def _parser_response(episode):
    value = episode.initial_input
    return json.dumps({
        "request": {
            "schema_version": 1,
            "user_id": value["user_id"],
            "history_authorized": value["history_authorized"],
            "liked_item_ids": [],
            "disliked_item_ids": [],
            "seen_item_ids": [],
            "constraints": {
                "include_genres": [],
                "exclude_genres": [],
                "year_min": None,
                "year_max": None,
                "excluded_item_ids": [],
                "exclude_seen": True,
                "k": 1,
                "required_fields": [],
            },
        }
    }, separators=(",", ":"))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", type=Path, required=True)
    args = parser.parse_args(argv)
    journal = args.journal.resolve()
    try:
        journal.relative_to(ROOT.resolve())
    except ValueError as error:
        raise SystemExit("journal must stay inside the workspace") from error
    if journal.exists():
        raise SystemExit("journal already exists; smoke evidence is append-only")

    manifest_path = ROOT / "artifacts/eval_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    public_entry = manifest["files"]["test_public"]
    private_entry = manifest["files"]["test_private"]
    public_path = ROOT / public_entry["path"]
    private_path = ROOT / private_entry["path"]
    if _sha256(public_path) != public_entry["sha256"]:
        raise ValueError("public frozen hash mismatch")
    if _sha256(private_path) != private_entry["sha256"]:
        raise ValueError("private evaluator hash mismatch")
    public = load_public_episodes(public_path)
    private = load_private_episodes(private_path)
    schedule = FeedbackSchedule.from_private_episodes(
        private, expected_episode_ids={episode.episode_id for episode in public}
    )

    text = [episode for episode in public if episode.layer == "text"]
    fake = FakeLLM([_parser_response(episode) for episode in text])
    adapter = ChatAdapter(fake, LLMBudget())
    pipeline = _FixturePipeline()
    executor = BenchmarkSystemExecutor(
        "F",
        fixed_pipeline=pipeline,
        text_parser=TextRequestParser(adapter),
        feedback_source=schedule,
        ledger=adapter.ledger,
    )
    spec = RunSpec(
        run_id="t19-F-frozen-fixture-smoke",
        benchmark_id="interactive-v1-t19",
        system="F",
        seed=42,
        split="test",
        public_episodes=public_entry["path"],
        public_sha256=public_entry["sha256"],
        episode_count=len(public),
        authorized_request_cap=sum(episode.max_turns for episode in text),
        money_budget_cny=0,
        per_request_cost_ceiling_cny=0,
        live=False,
        feedback_sha256=schedule.sha256,
    )
    first = run_benchmark(spec, journal, executor)
    resumed = run_benchmark(
        spec,
        journal,
        lambda _episode: (_ for _ in ()).throw(
            AssertionError("completed episode repeated")
        ),
    )
    attempts = load_completed_attempts(spec, journal)
    evidence = journal.read_text(encoding="utf-8")
    forbidden = (
        "acceptable_item_ids", "expected_statuses", "hard_constraints",
        "evaluator_events", "fault",
    )
    if any(name in evidence for name in forbidden):
        raise ValueError("evaluator-only field leaked into journal")
    report = {
        "status": "PASS",
        "scope": "mechanical_fixture_not_system_quality",
        "episodes": len(attempts),
        "multi_turn_attempts": sum(attempt.turns == 3 for attempt in attempts),
        "text_parser_fixture_calls": len(fake.calls),
        "fixed_pipeline_calls": pipeline.calls,
        "remote_api_requests": first["remote_api_requests"],
        "resume_executor_calls": 0,
        "resumed": resumed["resumed"],
        "feedback_sha256": schedule.sha256,
        "journal_sha256": _sha256(journal),
        "private_targets_in_journal": False,
        "metrics": "NOT EVALUATED",
    }
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
