"""Bounded A1 two-turn App run followed by the unchanged 45-row evaluator."""
import argparse
import hashlib
import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

from live_app_http import write_new
from live_app_single_turn import resource_checks, worker_environment
from live_llm_connection import scrub
from live_llm_preflight import check
from local_api_key import LocalKeyError, read_api_key
from reproduction_session_http import AUTHORIZATION_ID, AUTHORIZED_PROFILE, DATA_SHA256, QUERIES, SessionHTTP

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/eval_private/upstream_redial_a1_hashseed42/test_data_50.jsonl"
LOCAL_PROFILE = ROOT / "reproduction/.runtime/live_reproduction_20261005.json"
LIVE_DIRECTORY = ROOT / "reproduction/logs/live_reproduction_20261005"


def frozen_data():
    raw = DATA.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DATA_SHA256:
        raise ValueError("frozen_evaluation_input_hash_mismatch")
    rows = [json.loads(line) for line in raw.splitlines()]
    if len(rows) != 45 or any(set(r) != {"context", "target"} or not all(isinstance(v, str) and v for v in r.values()) for r in rows):
        raise ValueError("frozen_evaluation_input_schema_mismatch")
    if any(r["target"].casefold() in r["context"].casefold() for r in rows):
        raise ValueError("hidden_target_already_visible")
    return rows


def execute(directory, key, offline=False):
    directory = Path(directory)
    if (directory / "run.started.json").exists():
        return {"status": "BLOCKED", "reason": "authorized_session_already_started", "new_http_attempts": 0}
    records = frozen_data()
    provenance = resource_checks()
    source_paths = ["RecAI/InteRecAgent/app.py", "RecAI/InteRecAgent/eval/one_turn_eval.py",
        "reproduction/scripts/live_reproduction.py", "reproduction/scripts/original_reproduction_worker.py",
        "reproduction/scripts/reproduction_session_http.py"]
    provenance["source_sha256"] = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in source_paths}
    provenance["evaluation_data_sha256"] = DATA_SHA256
    directory.mkdir(parents=True, exist_ok=True)
    write_new(directory / "run.started.json", {"authorization_id": AUTHORIZATION_ID,
        "scope": "offline_fixture" if offline else "live_A1_multiturn_and_original_evaluation",
        "profile": AUTHORIZED_PROFILE, "provenance": provenance, "queries": QUERIES,
        "seed": 42, "sdk_max_retries": 0, "outer_retry_limits": 1, "http_transport_retries": 0})
    import httpx
    names, active = {}, {}
    def fixture(request):
        prompt = json.loads(request.content)["messages"][-1]["content"]
        if prompt.startswith("Previous Chat History:"):
            content = prompt.split("Execution Result: \n", 1)[1].split(".\n\nPlease", 1)[0]
        else:
            if active.get("stage") == "multiturn":
                year = ">= 1990" if active["index"] == 0 else "< 1980"
                sql = "SELECT id FROM movie_information WHERE tags LIKE '%Comedy%' AND release_date " + year + " ORDER BY visited_num DESC LIMIT 30"
                count = "3"
            else:
                sql = "SELECT id FROM movie_information ORDER BY visited_num DESC LIMIT 30"
                count = "5"
            plan = [{"tool_name": names["HardFilterTool"], "input": sql},
                {"tool_name": names["RankingTool"], "input": '{"schema":"popularity"}'},
                {"tool_name": names["MapTool"], "input": count}]
            content = "Action: ToolExecutor\nAction Input: " + json.dumps(plan)
        return httpx.Response(200, json={"id": "offline-fixture", "object": "chat.completion", "created": 0,
            "model": "deepseek-flash", "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}})
    contexts = list(QUERIES) + [r["context"] for r in records]
    broker = SessionHTTP(directory, key, contexts, transport=httpx.MockTransport(fixture) if offline else None)
    events = queue.Queue()
    worker = None
    started = time.perf_counter()
    result = {"status": "FAILED", "reason": "worker_did_not_complete", "original_evaluation": "NOT EVALUATED"}
    try:
        env = worker_environment()
        env["PYTHONHASHSEED"] = "42"
        with (directory / "worker.stderr.log").open("xb") as err, (directory / "events.jsonl").open("x", encoding="utf-8") as event_log:
            worker = subprocess.Popen([sys.executable, str(ROOT / "reproduction/scripts/original_reproduction_worker.py"),
                "--output", str(directory)], cwd=ROOT, env=env, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=err, text=True, encoding="utf-8", bufsize=1)
            def receive():
                for line in worker.stdout:
                    events.put(line)
                events.put(None)
            threading.Thread(target=receive, daemon=True).start()
            while True:
                if time.perf_counter() - started > 2400:
                    raise TimeoutError("session_worker_deadline")
                try:
                    line = events.get(timeout=1)
                except queue.Empty:
                    continue
                if line is None:
                    break
                event = json.loads(line)
                kind = event.get("event")
                if kind not in ("request", "worker_log", "result"):
                    event_log.write(json.dumps(event, ensure_ascii=True) + "\n")
                    event_log.flush()
                if kind == "context":
                    active.clear()
                    active.update(stage=event["stage"], index=event["index"])
                    context = QUERIES[event["index"]] if event["stage"] == "multiturn" else records[event["index"]]["context"]
                    broker.set_context(context)
                elif kind == "request":
                    packet = broker.send(event["body"])
                    worker.stdin.write(json.dumps(packet, ensure_ascii=True) + "\n")
                    worker.stdin.flush()
                    print(json.dumps({"stage": "llm_attempt_finished", "attempts": len(broker.calls), **active}), flush=True)
                elif kind == "ready":
                    names.update(event["tool_names"])
                    print(json.dumps({"stage": "original_app_ready"}), flush=True)
                elif kind in ("progress", "sample_completed"):
                    print(json.dumps(event), flush=True)
                elif kind == "worker_log":
                    with (directory / "worker.stdout.log").open("x", encoding="utf-8") as handle:
                        handle.write(scrub(event["text"], key))
                elif kind == "result":
                    result = event
                    break
                else:
                    raise ValueError("unexpected_worker_event")
            worker.wait(timeout=10)
            result["worker_exit_code"] = worker.returncode
            if worker.returncode:
                result["status"] = "FAILED"
    except Exception as error:
        result.update(status="FAILED", parent_error_type=type(error).__name__)
    finally:
        if worker is not None:
            if worker.poll() is None:
                worker.kill()
                worker.wait()
            worker.stdin.close()
            worker.stdout.close()
        broker.close()
    calls_ok = bool(broker.calls) and all(c["provider_response_valid"] for c in broker.calls)
    if not calls_ok:
        result["status"] = "FAILED"
    totals = {k: sum(c["response"]["usage"][k] for c in broker.calls)
              for k in ("prompt_tokens", "completion_tokens", "total_tokens")} if calls_ok else None
    estimate = sum(c["estimated_cost_cny_peak_ceiling"] for c in broker.calls) if calls_ok and not offline else None
    if estimate is not None and estimate > 3:
        result["status"] = "FAILED"
    result.update(authorization_id=AUTHORIZATION_ID, scope="offline_fixture" if offline else "live_A1_reproduction",
        http_attempts=len(broker.calls), remote_api_requests=0 if offline else
            (len(broker.calls) if all(c["http_status"] is not None for c in broker.calls) else None),
        usage=totals if not offline else None, usage_is_fixture=offline, calls=broker.calls,
        actual_cost_cny=None, actual_cost_reason="provider_bill_not_queried",
        estimated_cost_cny_peak_ceiling=estimate, price_checked_date="2026-10-05",
        price_source="https://api-docs.deepseek.com/zh-cn/quick_start/pricing/",
        provider_billing_hard_cap_enforced=False, budget=AUTHORIZED_PROFILE,
        total_elapsed_including_load_seconds=time.perf_counter() - started,
        provenance=provenance, canonical_author_subset_equivalence="NOT VERIFIED",
        shared_checkpoint_id_semantics="NOT VERIFIED", paper_A2="NOT VERIFIED")
    if offline:
        result["original_evaluation"] = "OFFLINE HARNESS CHECK; LIVE NOT EVALUATED"
    result = scrub(result, key)
    write_new(directory / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--offline-output", type=Path, default=ROOT / "reproduction/logs/a1_session_offline_20261005")
    args = parser.parse_args()
    directory = args.offline_output.resolve() if args.offline else LIVE_DIRECTORY
    if not directory.is_relative_to(ROOT / "reproduction/logs"):
        parser.error("offline output must be within reproduction/logs")
    if (directory / "run.started.json").exists():
        print(json.dumps({"status": "BLOCKED", "reason": "authorized_session_already_started", "new_http_attempts": 0}))
        return 2
    try:
        key = "offline-fixture-placeholder" if args.offline else read_api_key()
        profile = AUTHORIZED_PROFILE if args.offline else json.loads(LOCAL_PROFILE.read_text(encoding="utf-8-sig"))
    except (LocalKeyError, OSError, ValueError):
        print(json.dumps({"status": "BLOCKED", "reason": "local_profile_or_key_invalid", "new_http_attempts": 0}))
        return 2
    if profile != AUTHORIZED_PROFILE or (not args.offline and check(profile, bool(key))["status"] != "CONFIG_READY"):
        print(json.dumps({"status": "BLOCKED", "reason": "authorized_profile_or_key_missing", "new_http_attempts": 0}))
        return 2
    try:
        result = execute(directory, key, args.offline)
    except Exception as error:
        result = {"status": "FAILED", "error_type": type(error).__name__, "original_evaluation": "NOT EVALUATED"}
    summary = {k: result[k] for k in ("status", "scope", "http_attempts", "remote_api_requests", "usage",
        "usage_is_fixture", "actual_cost_cny", "estimated_cost_cny_peak_ceiling", "original_evaluation",
        "worker_exit_code", "parent_error_type", "error_type") if k in result}
    summary["evidence_directory"] = directory.relative_to(ROOT).as_posix()
    if "multiturn" in result:
        summary["multiturn_status"] = result["multiturn"]["status"]
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if result["status"] == "SUCCESS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
