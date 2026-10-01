"""Publish factual A1 preflight evidence; never turn mock responses into formal metrics."""
import datetime
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "reproduction/logs/a1_20261001"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    required = ("compatibility_diff", "complete_movie_integrity", "verify_legacy_rebuild", "offline_regressions",
                "original_sdk_mock", "native_checkpoint_contract", "native_tools_and_original_app",
                "original_app_help", "original_eval_help_with_pythonpath")
    for name in required:
        assert (LOG / (name + ".exitcode.txt")).read_text().strip() == "0", name
    native = read(LOG / "native_app_result.json")
    checkpoint = read(LOG / "native_checkpoint_contract.json")
    integrity = read(LOG / "complete_movie_integrity.json")
    environment = read(ROOT / "reproduction/environment_legacy.json")
    assert native["success"] and checkpoint["success"] and integrity["success"]
    assert environment["rebuild"]["status"] == "PASS"
    run = ROOT / "reproduction/native_runs/20261001"
    run.mkdir(parents=True, exist_ok=True)
    for filename in ("native_app_result.json", "original_app_mock_result.json", "native_checkpoint_contract.json",
                     "sdk_mock_result.json", "complete_movie_integrity.json", "compat_diff_manifest.json",
                     "gte_model_manifest.json", "tokenizer_manifest.json"):
        shutil.copy2(LOG / filename, run / filename)
    for name in required:
        for suffix in (".stdout.log", ".stderr.log", ".command.json", ".exitcode.txt"):
            shutil.copy2(LOG / (name + suffix), run / (name + suffix))
    config = {"upstream_commit": native["upstream_commit"], "python": environment["python"],
              "torch": "1.13.1+cpu", "domain": "movie", "plan_first": 1, "langchain": 0,
              "demo_mode": "zero", "enable_reflection": 0, "enable_shorten": 0,
              "tool_smoke_seed": 20261001, "torch_threads": 4, "max_candidate_num": 1000,
              "rank_num": 100, "similar_ratio": 0.05, "engine": "offline-sdk-fixture",
              "allow_paid_api": False, "api_request_cap": 0,
              "llm_usage_fields": "synthetic HTTP fixture fields; not experimental token measurements",
              "real_provider": None, "real_model": None, "token_cap": None,
              "money_budget": None, "budget_unit": None, "real_api_connection": "not_run",
              "original_eval_data": "NOT VERIFIED: absent from pinned repository and resource ZIP",
              "original_evaluation": "not_run", "paper_reproduction_A2": "not_run",
              "shared_id_semantics": "NOT VERIFIED", "model_training_mode_preserved": True}
    write(run / "run_config.json", config)
    manifest_path = ROOT / "reproduction/resource_manifest.json"
    previous_path = ROOT / "reproduction/resource_manifest.before_native_a1.json"
    if not previous_path.exists():
        shutil.copy2(manifest_path, previous_path)
    manifest = read(previous_path)
    manifest.update(checked_date="2026-10-01", original_resource_integrity="PASS",
                    integrity_scope="complete movie member byte integrity, not shared-ID semantics or permission to redistribute",
                    route_candidate="A1", route_final="A1", a1_status="OFFLINE_PREFLIGHT_PASS_LIVE_BLOCKED",
                    local_resources_directory_exists=True, local_settings="data/raw/upstream_movie/settings.json",
                    route_decision_status="A1 original-resource compatibility path retained; shared-ID semantics, live LLM and evaluation unresolved",
                    native_a1_evidence="reproduction/native_runs/20261001/native_app_result.json")
    keys = {"movies.ftr": "GAME_INFO_FILE", "columns.json": "TABLE_COL_DESC_FILE",
            "SASRec-SASRec.pth": "MODEL_CKPT_FILE", "movie_sim.npy": "ITEM_SIM_FILE", "settings.json": "settings"}
    manifest["resources"] = [dict(entry, key=keys[entry["name"]], local_path=entry["path"],
                                  source=integrity["source"], license="NOT VERIFIED", id_integrity="NOT VERIFIED")
                             for entry in integrity["full_movie_members"]]
    manifest["checks"].update(similarity_shape="PASS: full (9889,9889), float64, all values finite",
                               similarity_id_order="NOT VERIFIED",
                               unirec_compatibility="PASS: native loader/predict; all 36 state keys and weights equal original checkpoint",
                               checkpoint_full_integrity="PASS", original_gallery="PASS: real gte-base and 9888 catalog rows")
    manifest["download_scope"] = {"domains": ["movie"], "full_zip_downloaded": False,
                                  "matrix_resume_cache": True, "tls_validation_disabled": False,
                                  "completed_initial_response_body_bytes": 63006266,
                                  "completed_resume_response_body_bytes": 385634205,
                                  "failed_partial_response_byte_count": "unknown; not included in completed-body accounting",
                                  "dependencies_total_download_bytes": "not fully metered; do not report as known total"}
    manifest["remaining_blockers"] = ["live provider/model and budget authorization missing", "original evaluation data missing",
                                      "authoritative checkpoint title mapping and matrix ID order NOT VERIFIED",
                                      "independent prebuilt-resource redistribution license NOT VERIFIED"]
    manifest["next_dependency"] = "Resolve T03 mapping/provenance and T04 live API/evaluation prerequisites; do not start rebuilt models"
    manifest["runtime_metrics"] = {"status": "not_run", "recall_at_k": None, "ndcg_at_k": None,
                                   "mrr": None, "hit_rate": None, "task_success_rate": None,
                                   "constraint_satisfaction_rate": None, "latency": None,
                                   "token_usage": None, "tool_calls": None, "replan_rate": None}
    write(manifest_path, manifest)
    tasks_path = ROOT / "TASKS.yaml"
    tasks = read(tasks_path)
    by_id = {task["id"]: task for task in tasks["tasks"]}
    by_id["T02"].update(status="DONE", evidence="reproduction/environment_legacy.json; reproduction/native_runs/20261001/",
                        verification="pip check, actual native imports, original SDK mock and offline rebuild with all 174 package versions identical PASS")
    by_id["T02"].pop("remaining", None)
    by_id["T03"].update(status="IN_PROGRESS", evidence="reproduction/resource_manifest.json; reproduction/native_runs/20261001/",
                        verification="All five movie member CRC/SHA checks, original gallery/matrix and exact real checkpoint state loading PASS",
                        remaining="Authoritative checkpoint title mapping, matrix row semantics and prebuilt resource license unresolved; A1 functional route retained")
    by_id["T04"].update(status="BLOCKED", evidence="reproduction/UPSTREAM_REPRODUCTION.md; reproduction/native_runs/20261001/",
                        reason="Offline native preflight PASS, including original app loopback HTTP200 and two mock-LLM turns. Live budget/provider/model, original evaluation data and T03 semantics/provenance prerequisites remain unresolved")
    write(tasks_path, tasks)
    print(json.dumps({"native_offline_preflight": "PASS", "T02": "DONE", "T03": "IN_PROGRESS", "T04": "BLOCKED",
                      "live_original_reproduction": "NOT VERIFIED", "formal_metrics": "NOT EVALUATED"}))


if __name__ == "__main__":
    main()
