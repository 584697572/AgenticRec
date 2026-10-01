"""Check native A1 evidence, source separation and honest incomplete acceptance."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "reproduction/native_runs/20261001"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    expected = "0959ecb05b0794748426e73e6efc1b6b35ec433d"
    assert sha(ROOT / "AgenticRec_完整复现与优化执行规范.md") == "d88f02c84855234673514519486e74e73c61ffd4e2ec75594020fdf9b8ebb677"
    git = ["git", "-c", "safe.directory=" + (ROOT / "RecAI").as_posix(), "-C", str(ROOT / "RecAI")]
    assert subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip() == expected
    assert not subprocess.run(git + ["status", "--porcelain"], capture_output=True, check=True).stdout
    subprocess.run(git + ["apply", "--check", str(ROOT / "reproduction/legacy_compat.patch")], check=True)
    diff = read(RUN / "compat_diff_manifest.json")
    assert diff["changed_files"] == ["requirements.txt"] and diff["python_source_unchanged"]
    for entry in diff["files"]:
        assert sha(ROOT / "RecAI/InteRecAgent" / entry["path"]) == entry["original_sha256"]
        assert sha(ROOT / "reproduction/.runtime/InteRecAgent-compat" / entry["path"]) == entry["compat_sha256"]
        if entry["path"].endswith(".py"):
            assert entry["original_sha256"] == entry["compat_sha256"]
    integrity = read(RUN / "complete_movie_integrity.json")
    for entry in integrity["full_movie_members"]:
        assert sha(ROOT / entry["path"]) == entry["sha256"]
    checkpoint = read(RUN / "native_checkpoint_contract.json")
    assert checkpoint["success"] and checkpoint["all_keys_and_weights_exactly_equal"]
    assert checkpoint["state_key_count"] == 36 and checkpoint["scores_shape"] == [1, 9888]
    assert checkpoint["missing_keys"] == checkpoint["unexpected_keys"] == []
    assert checkpoint["all_scores_finite"] and checkpoint["training_mode_preserved"]
    assert checkpoint["title_mapping"] == "NOT VERIFIED"
    native = read(RUN / "native_app_result.json")
    assert native["success"] and all(entry["status"] == "PASS" for entry in native["records"])
    assert native["llm"] == "mock_transport_only" and native["live_api_requests"] == 0
    assert native["live_agent"] == "NOT VERIFIED" and native["metrics"] == "NOT EVALUATED"
    app = read(RUN / "original_app_mock_result.json")
    assert app["loopback_only"] and app["http_status"] == 200 and len(app["turns"]) == 2
    assert [call["phase"] for call in app["sdk_calls"]] == ["plan_fixture", "summary_fixture"] * 2
    assert all(len(turn["tool_trace"]) == 3 for turn in app["turns"])
    config = read(RUN / "run_config.json")
    assert not config["allow_paid_api"] and config["api_request_cap"] == 0
    assert config["original_evaluation"] == config["real_api_connection"] == config["paper_reproduction_A2"] == "not_run"
    environment = read(ROOT / "reproduction/environment_legacy.json")
    assert environment["pip_check"]["exit_code"] == 0
    assert environment["rebuild"]["status"] == "PASS" and environment["rebuild"]["package_versions_identical"]
    tasks = {entry["id"]: entry for entry in read(ROOT / "TASKS.yaml")["tasks"]}
    assert tasks["T02"]["status"] == "DONE" and tasks["T03"]["status"] == "IN_PROGRESS" and tasks["T04"]["status"] == "BLOCKED"
    assert all(tasks["T%02d" % i]["status"] == "TODO" for i in range(5, 25))
    manifest = read(ROOT / "reproduction/resource_manifest.json")
    assert manifest["original_resource_integrity"] == "PASS" and not manifest["archive_downloaded"]
    assert manifest["route_final"] == "A1" and manifest["checks"]["similarity_id_order"] == "NOT VERIFIED"
    assert all(value is None for key, value in manifest["runtime_metrics"].items() if key != "status")
    for name in ("compatibility_diff", "complete_movie_integrity", "verify_legacy_rebuild", "offline_regressions", "original_sdk_mock",
                 "native_checkpoint_contract", "native_tools_and_original_app", "original_app_help", "original_eval_help_with_pythonpath"):
        assert (RUN / (name + ".exitcode.txt")).read_text().strip() == "0", name
    print("PASS: native original resources/tools/weights/app mock evidence, isolated rebuild, pinned pristine source, honest live/eval blockers")


if __name__ == "__main__":
    main()
