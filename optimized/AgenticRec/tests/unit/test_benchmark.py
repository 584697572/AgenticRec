import hashlib
import json

import pytest

from agenticrec import cli
from agenticrec.evaluation import benchmark


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path):
    data_dir = tmp_path / "data/eval"
    public_rows = [
        {
            "schema_version": 1,
            "episode_id": "test-001",
            "split": "test",
            "layer": "structured",
            "group_id": "user:1",
            "template_group_id": "test:a:structured",
            "scenario_family": "explicit_filter",
            "multi_turn": False,
            "initial_input": {"schema_version": 1},
            "max_turns": 1,
        },
        {
            "schema_version": 1,
            "episode_id": "test-002",
            "split": "test",
            "layer": "text",
            "group_id": "user:2",
            "template_group_id": "test:b:text",
            "scenario_family": "exclusion_feedback",
            "multi_turn": True,
            "initial_input": {"mode": "text", "message": "recommend"},
            "max_turns": 3,
        },
    ]
    public_text = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in public_rows)
    public_sha = _write(data_dir / "test.public.jsonl", public_text)
    private_sha = _write(data_dir / "test.private.jsonl", "{}\n")
    manifest = {
        "schema_version": 1,
        "frozen": True,
        "llm_simulator_enabled": False,
        "human_review_status": "VERIFIED",
        "human_review_checks": {"checked": True},
        "human_review_ids": ["test-001"],
        "root_relative_to_manifest": ".",
        "dataset_dir": "data/eval",
        "splits": {
            "test": {"episodes": 2, "layers": {"structured": 1, "text": 1}}
        },
        "files": {
            "test_public": {
                "name": "test.public.jsonl", "path": "data/eval/test.public.jsonl",
                "rows": 2, "sha256": public_sha, "visibility": "public",
            },
            "test_private": {
                "name": "test.private.jsonl", "path": "data/eval/test.private.jsonl",
                "rows": 1, "sha256": private_sha, "visibility": "evaluator_only",
            },
        },
    }
    _write(tmp_path / "eval_manifest.json", json.dumps(manifest))
    ledger = {
        "remote_api_requests": 93,
        "budget": {
            "provider": "deepseek", "model_id": "deepseek-flash",
            "api_request_cap": 120, "max_output_tokens": 1024,
            "money_budget": 3, "budget_unit": "CNY",
        },
    }
    _write(tmp_path / "ledger.json", json.dumps(ledger))
    config = {
        "schema_version": 1,
        "benchmark_id": "t19-test",
        "eval_manifest": "eval_manifest.json",
        "authorization_ledger": "ledger.json",
        "split": "test",
        "episodes": 2,
        "model_seeds": [7, 42, 2026],
        "agent_seeds": [7, 42, 2026],
        "systems": ["U1", "F", "A", "O"],
        "ablations": [
            "no_user_model", "no_collaborative", "no_explicit_preference_state",
            "always_agent", "no_replanning", "no_content",
        ],
        "max_replans": 1,
        "bootstrap_samples": 1000,
        "confidence": 0.95,
        "noninferiority_margin": -0.02,
    }
    config_path = tmp_path / "benchmark.json"
    _write(config_path, json.dumps(config))
    return config_path


def test_dry_run_counts_every_system_and_fails_closed_before_live_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "ROOT", tmp_path)
    plan = benchmark.build_dry_run(_fixture(tmp_path))

    # Three response turns; F parses the text episode only once. Three seeds.
    assert plan["request_ceiling_by_run"] == {
        "system:U1": 9,
        "system:F": 3,
        "system:A": 18,
        "system:O": 12,
        "ablation:no_user_model": 12,
        "ablation:no_collaborative": 12,
        "ablation:no_explicit_preference_state": 12,
        "ablation:always_agent": 0,
        "ablation:no_replanning": 6,
        "ablation:no_content": 12,
    }
    assert plan["required_new_requests_ceiling"] == 96
    assert len(plan["run_matrix"]) == 27
    assert sum(item["request_ceiling"] for item in plan["run_matrix"]) == 96
    assert len({item["run_id"] for item in plan["run_matrix"]}) == 27
    assert plan["run_aliases"] == {"always_agent": "A"}
    assert plan["authorization"]["requests_remaining"] == 27
    assert plan["authorization"]["request_shortfall"] == 69
    assert plan["status"] == "BLOCKED_AUTHORIZATION"
    assert plan["remote_api_requests"] == 0
    assert plan["test_private_loaded"] is False
    assert plan["live_run_authorized"] is False


def test_dry_run_rejects_changed_seed_order_or_missing_required_ablation(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "ROOT", tmp_path)
    path = _fixture(tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["model_seeds"] = [42, 7, 2026]
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="pre-registered model seeds"):
        benchmark.build_dry_run(path)


def test_paired_bootstrap_is_deterministic_and_preserves_pairing():
    result = benchmark.paired_bootstrap_ci(
        [0.0, 0.5, 1.0, 0.0], [0.25, 0.75, 1.25, 0.25],
        samples=2000, confidence=0.95, seed=42,
    )
    repeated = benchmark.paired_bootstrap_ci(
        [0.0, 0.5, 1.0, 0.0], [0.25, 0.75, 1.25, 0.25],
        samples=2000, confidence=0.95, seed=42,
    )
    assert result == repeated
    assert result["paired_count"] == 4
    assert result["mean_difference"] == pytest.approx(0.25)
    assert result["ci_lower"] == pytest.approx(0.25)
    assert result["ci_upper"] == pytest.approx(0.25)


def test_paired_bootstrap_rejects_unpaired_or_nonfinite_values():
    with pytest.raises(ValueError, match="same nonzero length"):
        benchmark.paired_bootstrap_ci([1.0], [], samples=100, confidence=0.95, seed=1)
    with pytest.raises(ValueError, match="finite"):
        benchmark.paired_bootstrap_ci([1.0], [float("nan")], samples=100,
                                      confidence=0.95, seed=1)


def test_non_dry_cli_fails_closed_on_authorization_shortfall(monkeypatch, capsys):
    monkeypatch.setattr(benchmark, "build_dry_run", lambda _path: {
        "status": "BLOCKED_AUTHORIZATION",
        "dry_run": True,
        "remote_api_requests": 0,
        "live_run_authorized": False,
    })

    assert cli.main(["benchmark", "--config", "unused.json"]) == 3
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "BLOCKED_AUTHORIZATION"
    assert report["dry_run"] is False
    assert report["planning_dry_run_completed"] is True
    assert report["remote_api_requests"] == 0
