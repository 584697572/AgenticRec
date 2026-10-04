"""Verify this round's evidence; no imports of models, network calls or key reads."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def main():
    assert sha("AgenticRec_完整复现与优化执行规范.md") == "d88f02c84855234673514519486e74e73c61ffd4e2ec75594020fdf9b8ebb677"
    tasks = {t["id"]: t for t in read("TASKS.yaml")["tasks"]}
    assert tasks["T05"]["status"] == "DONE" and tasks["T05"]["depends_on"] == ["T00", "T01"]
    assert tasks["T03"]["status"] == "IN_PROGRESS" and tasks["T04"]["status"] == "BLOCKED"
    assert all(tasks[f"T{i:02d}"]["status"] == "TODO" for i in range(6, 25))
    env = read("reproduction/environment_dev.json")
    assert env["python"] == "3.11.14" and env["rebuild"]["package_versions_identical"]
    assert env["rebuild"]["locked_dependency_count"] == 7
    assert env["lock_sha256"] == sha("AgenticRec/requirements.dev.lock.txt")
    run = ROOT / "reproduction/native_runs/20261005"
    for name in ("dev_unit", "rebuilt_unit", "rebuilt_doctor", "rebuilt_fixture",
                 "dev_dependency_check_project_cache", "rebuilt_dependency_check_project_cache",
                 "legacy_offline_tests_final", "dtype_regression_after"):
        assert (run / (name + ".exitcode.txt")).read_text().strip() == "0", name
    assert "15 passed" in (run / "dev_unit.stdout.log").read_text()
    assert "15 passed" in (run / "rebuilt_unit.stdout.log").read_text()
    assert "Ran 23 tests" in (run / "legacy_offline_tests_final.stderr.log").read_text()
    offline = read("reproduction/native_runs/20261005/original_app_offline_summary.json")
    assert offline["status"] == "SUCCESS" and offline["validation"]["passed"]
    assert offline["remote_api_requests"] == 0 and offline["usage_is_fixture"]
    assert offline["http_attempts"] == 2 and offline["live_original_app"] == "NOT VERIFIED"
    assert offline["mapped_item_count"] == 3 and not offline["real_key_in_worker"]
    assert offline["private_result_sha256"] == sha("reproduction/logs/app_single_turn_offline_20261005_fixed/result.json")
    data = read("reproduction/redial_eval_preparation_20261005.json")
    assert data["source"]["sha256"] == sha(data["source"]["path"])
    assert data["output_sha256"] == sha(data["output"])
    assert data["output_rows"] == 45 and data["selected_test_dialogues"] == 736
    assert data["canonical_author_subset_equivalence"] == "NOT VERIFIED" and data["evaluation"] == "NOT EVALUATED"
    assert data["notebook_sha256"] == sha("RecAI/InteRecAgent/preprocess/preprocess_redial.ipynb")
    original = read("reproduction/native_runs/20261005/original_redial_mapping_failure_summary.json")
    assert original["mapping_error_count"] == 42 and original["status"] == "BLOCKED"
    assert original["error_types"] == ["TypeError"]
    logs = ROOT / "reproduction/logs/resource_recheck_20261005"
    assert (logs / "dtype_regression_before.exitcode.txt").read_text().strip() == "1"
    assert (logs / "prepare_a1_eval_replay.exitcode.txt").read_text().strip() == "0"
    print("PASS: T05 independent rebuild/tests, original-app offline evidence, original ReDial failure then compatibility replay, honest live/eval blockers")


if __name__ == "__main__":
    raise SystemExit(main())
