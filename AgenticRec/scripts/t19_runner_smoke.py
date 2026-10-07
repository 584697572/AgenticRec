"""Exercise the resumable runner on all frozen public test episodes, offline."""

import hashlib
from pathlib import Path

from agenticrec.data import digest
from agenticrec.evaluation.episodes import EpisodeAttempt
from agenticrec.evaluation.runner import (
    ROOT,
    RunSpec,
    load_completed_attempts,
    run_benchmark,
)


def main():
    public_path = ROOT / "data/eval_private/interactive-v1/test.public.jsonl"
    journal = ROOT / "artifacts/runs/t19/runner_smoke_o_seed7_feedback_bound.jsonl"
    spec = RunSpec(
        run_id="t19-offline-smoke-O-seed7",
        benchmark_id="interactive-v1-t19-offline-fixture",
        system="O",
        seed=7,
        split="test",
        public_episodes=public_path.relative_to(ROOT).as_posix(),
        public_sha256=digest(public_path),
        episode_count=150,
        authorized_request_cap=460,
        money_budget_cny=0,
        per_request_cost_ceiling_cny=0,
        live=False,
        feedback_sha256=hashlib.sha256(b"runner-smoke-no-feedback").hexdigest(),
    )

    def fixture(episode):
        return EpisodeAttempt(
            episode_id=episode.episode_id,
            status="OFFLINE_FIXTURE_NOT_EVALUATED",
            item_ids=(),
            turns=1,
            tool_calls=0,
            latency_ms=0,
            input_tokens=None,
            output_tokens=None,
            request_count=0,
        )

    report = run_benchmark(spec, journal, fixture)
    attempts = load_completed_attempts(spec, journal)
    if (report["status"] != "COMPLETE" or len(attempts) != 150
            or report["remote_api_requests"] != 0
            or any(item.status != "OFFLINE_FIXTURE_NOT_EVALUATED" for item in attempts)):
        raise RuntimeError("offline runner smoke did not satisfy its contract")
    print({
        "status": "PASS",
        "scope": "runner engineering fixture only",
        "attempts": len(attempts),
        "journal_sha256": digest(journal),
        "remote_api_requests": 0,
        "metrics": "NOT EVALUATED",
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
