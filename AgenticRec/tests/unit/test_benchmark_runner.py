import hashlib
import json

import pytest

from agenticrec.evaluation import runner
from agenticrec.evaluation.episodes import EpisodeAttempt, PublicEpisode


def _public(episode_id, *, layer="structured", max_turns=1):
    return PublicEpisode(
        episode_id=episode_id,
        split="test",
        layer=layer,
        group_id="user:" + episode_id.rsplit("-", 1)[1],
        template_group_id=episode_id + ":template",
        scenario_family="exclusion_feedback" if max_turns > 1 else "explicit_filter",
        multi_turn=max_turns > 1,
        initial_input=(
            {"mode": "text", "message": "recommend"}
            if layer == "text"
            else {"schema_version": 1, "constraints": {"k": 2}}
        ),
        max_turns=max_turns,
    )


def _fixture(tmp_path):
    path = tmp_path / "public.jsonl"
    episodes = [_public("test-001"), _public("test-002", layer="text", max_turns=3)]
    path.write_text(
        "".join(json.dumps(item.to_dict(), separators=(",", ":")) + "\n"
                for item in episodes),
        encoding="utf-8", newline="\n",
    )
    return path, hashlib.sha256(path.read_bytes()).hexdigest(), episodes


def _spec(tmp_path, **overrides):
    public_path, public_sha, _episodes = _fixture(tmp_path)
    values = {
        "run_id": "t19-F-seed7",
        "benchmark_id": "interactive-v1-t19",
        "system": "F",
        "seed": 7,
        "split": "test",
        "public_episodes": public_path.name,
        "public_sha256": public_sha,
        "episode_count": 2,
        "authorized_request_cap": 3,
        "money_budget_cny": 0.0,
        "per_request_cost_ceiling_cny": 0.0,
        "live": False,
    }
    values.update(overrides)
    return runner.RunSpec(**values)


def _attempt(episode_id, *, requests=0, status="OK"):
    return EpisodeAttempt(
        episode_id=episode_id,
        status=status,
        item_ids=(1, 2),
        turns=1,
        tool_calls=0,
        latency_ms=1.5,
        input_tokens=None,
        output_tokens=None,
        request_count=requests,
    )


def test_dry_run_reads_only_public_input_and_has_no_side_effects(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    spec = _spec(tmp_path)
    invoked = []

    report = runner.run_benchmark(spec, tmp_path / "attempts.jsonl",
                                  lambda episode: invoked.append(episode), dry_run=True)

    assert report["status"] == "READY"
    assert report["episode_request_ceiling"] == {"test-001": 0, "test-002": 3}
    assert report["required_request_ceiling"] == 3
    assert report["private_targets_loaded"] is False
    assert report["remote_api_requests"] == 0
    assert invoked == []
    assert not (tmp_path / "attempts.jsonl").exists()


def test_completed_journal_resumes_without_repeating_episode_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    spec = _spec(tmp_path)
    journal = tmp_path / "attempts.jsonl"
    calls = []

    first = runner.run_benchmark(
        spec, journal,
        lambda episode: calls.append(episode.episode_id) or _attempt(episode.episode_id),
    )
    second = runner.run_benchmark(
        spec, journal,
        lambda episode: pytest.fail("completed episode was called again"),
    )

    assert first["status"] == second["status"] == "COMPLETE"
    assert calls == ["test-001", "test-002"]
    assert first["completed_episodes"] == second["completed_episodes"] == 2
    assert second["resumed"] is True
    assert second["remote_api_requests"] == 0
    assert len(journal.read_text(encoding="utf-8").splitlines()) == 5
    loaded = runner.load_completed_attempts(spec, journal)
    assert [item.episode_id for item in loaded] == ["test-001", "test-002"]
    assert all(item.item_ids == (1, 2) for item in loaded)


def test_interrupted_started_episode_blocks_resume_without_duplicate_charge(
        tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    spec = _spec(tmp_path)
    journal = tmp_path / "attempts.jsonl"

    with pytest.raises(RuntimeError, match="fixture crash"):
        runner.run_benchmark(spec, journal,
                             lambda _episode: (_ for _ in ()).throw(RuntimeError("fixture crash")))

    with pytest.raises(runner.UnresolvedAttemptError, match="test-001"):
        runner.run_benchmark(
            spec, journal,
            lambda _episode: pytest.fail("ambiguous attempt must not be repeated"),
        )
    events = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    assert [event["event"] for event in events] == ["HEADER", "STARTED", "INTERRUPTED"]
    assert events[-1]["error_type"] == "RuntimeError"
    assert "fixture crash" not in json.dumps(events)


def test_insufficient_full_run_authorization_stops_before_journal_or_executor(
        tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    spec = _spec(tmp_path, authorized_request_cap=2)
    journal = tmp_path / "attempts.jsonl"
    invoked = []

    with pytest.raises(runner.RunAuthorizationError, match="full run ceiling"):
        runner.run_benchmark(
            spec, journal, lambda episode: invoked.append(episode) or _attempt(episode.episode_id)
        )

    assert invoked == []
    assert not journal.exists()


def test_journal_hash_chain_and_spec_identity_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    spec = _spec(tmp_path)
    journal = tmp_path / "attempts.jsonl"
    runner.run_benchmark(spec, journal, lambda episode: _attempt(episode.episode_id))

    lines = journal.read_text(encoding="utf-8").splitlines()
    value = json.loads(lines[2])
    value["attempt"]["status"] = "TAMPERED"
    lines[2] = json.dumps(value, separators=(",", ":"), sort_keys=True)
    journal.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash chain"):
        runner.run_benchmark(spec, journal, lambda episode: _attempt(episode.episode_id))

    clean = tmp_path / "clean.jsonl"
    runner.run_benchmark(spec, clean, lambda episode: _attempt(episode.episode_id))
    changed = _spec(tmp_path, run_id="t19-F-seed42", seed=42)
    with pytest.raises(ValueError, match="identity"):
        runner.run_benchmark(changed, clean, lambda episode: _attempt(episode.episode_id))


def test_attempt_must_match_episode_and_reserved_request_ceiling(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    spec = _spec(tmp_path)

    with pytest.raises(ValueError, match="episode ID"):
        runner.run_benchmark(spec, tmp_path / "wrong.jsonl",
                             lambda _episode: _attempt("test-999"))

    live_like = _spec(
        tmp_path, run_id="t19-U1-seed7", system="U1",
        authorized_request_cap=4,
    )
    with pytest.raises(ValueError, match="reserved request ceiling"):
        runner.run_benchmark(
            live_like, tmp_path / "over.jsonl",
            lambda episode: _attempt(episode.episode_id, requests=2),
        )
