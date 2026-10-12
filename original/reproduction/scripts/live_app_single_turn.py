"""Run exactly one original-app turn, with a keyless resource worker and capped HTTP."""
import argparse
import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

from live_app_http import AUTHORIZATION_ID, PROFILE, QUERY, BoundedHTTP, write_new
from live_llm_connection import scrub
from live_llm_preflight import check
from local_api_key import LocalKeyError, read_api_key

ROOT = Path(__file__).resolve().parents[2]
LOCAL_PROFILE = ROOT / "reproduction/.runtime/live_app_single_turn_20261005.json"
LIVE_DIRECTORY = ROOT / "reproduction/logs/live_app_single_turn_20261005"


def resource_checks():
    manifest = json.loads((ROOT / "reproduction/resource_manifest.json").read_text(encoding="utf-8"))
    records = []
    for entry in manifest["resources"]:
        path = ROOT / entry["path"]
        h = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                h.update(block)
        if h.hexdigest() != entry["sha256"]:
            raise ValueError("original_resource_hash_mismatch")
        linked = ROOT / "reproduction/.runtime/InteRecAgent-compat/resources/movie" / path.name
        if not os.path.samefile(path, linked):
            if hashlib.sha256(linked.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError("compat_resource_hash_mismatch")
        records.append({"name": path.name, "sha256": h.hexdigest()})
    upstream = ROOT / "RecAI/InteRecAgent"
    copied = ROOT / "reproduction/.runtime/InteRecAgent-compat"
    files = sorted(upstream.rglob("*.py"))
    for path in files:
        if path.read_bytes() != (copied / path.relative_to(upstream)).read_bytes():
            raise ValueError("compat_python_differs_from_pinned_upstream")
    return {"resources": records, "original_python_files_identical": len(files)}


def worker_environment():
    env = {k: os.environ[k] for k in ("SYSTEMROOT", "WINDIR", "TEMP", "TMP", "PATH", "COMSPEC") if k in os.environ}
    cache = ROOT / "reproduction/.runtime/hf_cache"
    profile = ROOT / "reproduction/.runtime/isolated_profile"
    env.update(USERPROFILE=str(profile), LOCALAPPDATA=str(profile / "AppData/Local"),
        APPDATA=str(profile / "AppData/Roaming"), MPLCONFIGDIR=str(profile / "matplotlib"),
        GRADIO_ANALYTICS_ENABLED="False", TIKTOKEN_CACHE_DIR=str(profile / "tiktoken"),
        DOMAIN="movie", PYTHONUTF8="1", PYTHONIOENCODING="utf-8", HF_HOME=str(cache),
        SENTENCE_TRANSFORMERS_HOME=str(cache / "sentence_transformers"),
        HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    return env


def execute(directory, key, offline=False):
    import httpx
    if (directory / "run.started.json").exists():
        return {"status": "BLOCKED", "reason": "single_turn_already_started", "new_http_attempts": 0}
    evidence = resource_checks()
    directory.mkdir(parents=True, exist_ok=True)
    write_new(directory / "run.started.json", {
        "authorization_id": AUTHORIZATION_ID, "scope": "offline_fixture" if offline else "live_original_app",
        "profile": PROFILE, "query": QUERY, "provenance": evidence})
    names = {}
    rejected = []
    def fixture(request):
        body = json.loads(request.content)
        prompt = body["messages"][-1]["content"]
        if prompt.startswith("Previous Chat History:"):
            content = prompt.split("Execution Result: \n", 1)[1].split(".\n\nPlease", 1)[0]
        else:
            plan = [{"tool_name": names["HardFilterTool"], "input":
                "SELECT id FROM movie_information WHERE tags LIKE '%Comedy%' AND release_date >= 1990 ORDER BY visited_num DESC LIMIT 30"},
                {"tool_name": names["RankingTool"], "input": '{"schema":"popularity"}'},
                {"tool_name": names["MapTool"], "input": "3"}]
            content = "Action: ToolExecutor\nAction Input: " + json.dumps(plan)
        return httpx.Response(200, json={"id": "offline-fixture", "object": "chat.completion", "created": 0,
            "model": PROFILE["model_id"], "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}})

    broker = BoundedHTTP(directory, key, httpx.MockTransport(fixture) if offline else None)
    messages = queue.Queue()
    worker = None
    result = {"status": "FAILED", "reason": "worker_did_not_complete"}
    started = time.perf_counter()
    try:
        with (directory / "worker.stderr.log").open("wb") as err:
            worker = subprocess.Popen([sys.executable, str(ROOT / "reproduction/scripts/original_app_worker.py")],
                cwd=ROOT, env=worker_environment(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=err, text=True, encoding="utf-8", bufsize=1)
            def receive():
                for line in worker.stdout:
                    messages.put(line)
                messages.put(None)
            reader = threading.Thread(target=receive, daemon=True)
            reader.start()
            while True:
                if time.perf_counter() - started > 900:
                    raise TimeoutError("worker_wait_deadline")
                try:
                    line = messages.get(timeout=1)
                except queue.Empty:
                    continue
                if line is None:
                    break
                event = json.loads(line)
                kind = event.get("event")
                if kind == "request":
                    try:
                        packet = broker.send(event["body"])
                    except Exception as error:
                        rejected.append({"error_type": type(error).__name__, "request": event["body"]})
                        packet = {"transport_error": type(error).__name__}
                    worker.stdin.write(json.dumps(packet, ensure_ascii=True) + "\n")
                    worker.stdin.flush()
                    print(json.dumps({"stage": "llm_attempt_finished", "attempts": len(broker.calls),
                                      "scope": "offline_fixture" if offline else "live_original_app"}), flush=True)
                elif kind == "ready":
                    names.update(event["tool_names"])
                    print(json.dumps({"stage": "original_app_ready", "query": QUERY}), flush=True)
                elif kind == "progress":
                    print(json.dumps(event), flush=True)
                elif kind == "worker_log":
                    (directory / "worker.stdout.log").write_text(scrub(event["text"], key), encoding="utf-8")
                elif kind == "result":
                    result = event
                    break
                else:
                    raise ValueError("unexpected_worker_event")
            worker.wait(timeout=10)
            result["worker_exit_code"] = worker.returncode
            if worker.returncode != 0:
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
    usages = [c["response"].get("usage") if isinstance(c["response"], dict) else None for c in broker.calls]
    usage_valid = bool(usages) and all(isinstance(u, dict) and
        all(type(u.get(k)) is int and u[k] >= 0 for k in ("prompt_tokens", "completion_tokens", "total_tokens"))
        and u["prompt_tokens"] + u["completion_tokens"] == u["total_tokens"] and u["completion_tokens"] <= 512 for u in usages)
    if not usage_valid or any(c["http_status"] != 200 or c["error_type"] for c in broker.calls):
        result["status"] = "FAILED"
    totals = {k: sum(u[k] for u in usages) for k in ("prompt_tokens", "completion_tokens", "total_tokens")} if usage_valid else None
    estimate = ((totals["prompt_tokens"] * 2 + totals["completion_tokens"] * 8) / 1000000
                if totals and not offline else None)
    if estimate is not None and estimate > 1:
        result["status"] = "FAILED"
    result.update(authorization_id=AUTHORIZATION_ID, scope="offline_fixture" if offline else "live_original_app",
        live_original_app="VERIFIED" if result["status"] == "SUCCESS" and not offline else "NOT VERIFIED",
        http_attempts=len(broker.calls), remote_api_requests=0 if offline else
            (len(broker.calls) if all(c["http_status"] is not None for c in broker.calls) else None),
        usage=totals, usage_is_fixture=offline, actual_cost_cny=None, actual_cost_reason="provider_bill_not_queried",
        estimated_cost_cny_peak_ceiling=estimate, price_checked_date="2026-10-05",
        price_source="https://api-docs.deepseek.com/zh-cn/quick_start/pricing/",
        budget=PROFILE, total_elapsed_including_load_seconds=time.perf_counter() - started,
        calls=broker.calls, rejected_requests=rejected,
        shared_checkpoint_id_semantics="NOT VERIFIED", original_evaluation="NOT EVALUATED")
    result = scrub(result, key)
    write_new(directory / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--offline-output", type=Path, default=ROOT / "reproduction/logs/app_single_turn_offline_20261005")
    args = parser.parse_args()
    try:
        key = "offline-fixture-placeholder" if args.offline else read_api_key()
    except LocalKeyError:
        print(json.dumps({"status": "BLOCKED", "reason": "local_key_configuration_invalid", "new_http_attempts": 0}))
        return 2
    if args.offline:
        directory = args.offline_output.resolve()
        if not directory.is_relative_to(ROOT / "reproduction/logs"):
            parser.error("offline output must be inside reproduction/logs")
    else:
        directory = LIVE_DIRECTORY
        try:
            profile = json.loads(LOCAL_PROFILE.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            profile = None
        if profile != PROFILE or check(profile, bool(key.strip()))["status"] != "CONFIG_READY":
            print(json.dumps({"status": "BLOCKED", "reason": "authorized_profile_or_key_missing", "new_http_attempts": 0}))
            return 2
    try:
        result = execute(directory, key, args.offline)
    except Exception as error:
        result = {"status": "FAILED", "error_type": type(error).__name__, "live_original_app": "NOT VERIFIED"}
    # Full prompts/traces stay local; the terminal gets a compact, safe summary.
    summary = {k: v for k, v in result.items() if k not in
               ("calls", "memory", "map_observations", "tool_trace", "budget", "rejected_requests")}
    summary["evidence_directory"] = directory.relative_to(ROOT).as_posix()
    print(json.dumps(summary, ensure_ascii=True, indent=2), flush=True)
    return 0 if result["status"] == "SUCCESS" else (2 if result["status"] == "BLOCKED" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
