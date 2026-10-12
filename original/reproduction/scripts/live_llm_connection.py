"""One authorized DeepSeek connection attempt through the unmodified upstream wrapper.

Run with a key in the environment or workspace .env, using the legacy Python. Repeated
invocations share an exclusive reservation file and cannot spend another attempt.
"""
import contextlib
import datetime
import hashlib
import importlib.util
import io
import json
import os
import time
from pathlib import Path
from unittest.mock import patch

from live_llm_preflight import check
from local_api_key import LocalKeyError, read_api_key

ROOT = Path(__file__).resolve().parents[2]
AUTHORIZATION_ID = "deepseek-connection-20261004"
AUTHORIZED_PROFILE = {
    "provider": "deepseek", "api_type": "open_ai", "api_base": "https://api.deepseek.com",
    "api_version": None, "model_id": "deepseek-flash", "allow_paid_api": True,
    "api_request_cap": 1, "max_output_tokens": 128, "money_budget": 1, "budget_unit": "CNY",
}
SOURCE = ROOT / "RecAI/InteRecAgent/llm4crs/utils/open_ai.py"
SOURCE_SHA256 = "60e7d6c6fbca1aaf334efb5106edaffccbf34cba7dcb5e983d5980ff94251b57"
MESSAGES = [{"role": "system", "content": "Reply briefly."},
            {"role": "user", "content": "Reply with OK."}]
PROFILE = ROOT / "reproduction/.runtime/live_llm.json"
LEDGER = ROOT / "reproduction/.runtime/live_llm_connection_20261004.request.json"
RESULT = ROOT / "reproduction/logs/live_llm_20261004/connection_result.json"
PRICE_SOURCE = "https://api-docs.deepseek.com/zh-cn/quick_start/pricing/"
# Official peak rates checked 2026-10-04; planning allowance, not measured input.
PLANNED_INPUT_ALLOWANCE = 1024
PLANNED_COST_CNY = (PLANNED_INPUT_ALLOWANCE * 2 + 128 * 8) / 1_000_000


def blocked(reason):
    return {"status": "BLOCKED", "reason": reason, "authorization_id": AUTHORIZATION_ID,
            "http_attempts": 0, "live_connection": "NOT VERIFIED"}


def scrub(value, key):
    """Only whitelist-derived output is saved; additionally redact an echoed key."""
    if isinstance(value, str):
        return value.replace(key, "[REDACTED]") if key else value
    if isinstance(value, dict):
        return {name: scrub(item, key) for name, item in value.items()}
    if isinstance(value, list):
        return [scrub(item, key) for item in value]
    return value


def run_once(profile, key, ledger=LEDGER, result_path=RESULT, transport=None,
             evidence_scope="live_connection"):
    if check(profile, bool(key and key.strip()))["status"] != "CONFIG_READY":
        return blocked("configuration_or_key_missing")
    if profile != AUTHORIZED_PROFILE:
        return blocked("profile_exceeds_or_differs_from_this_authorization")
    if ledger.exists():
        return blocked("authorized_attempt_already_reserved")
    if result_path.exists():
        return blocked("existing_evidence_must_be_preserved")
    if profile["money_budget"] < PLANNED_COST_CNY:
        return blocked("planning_estimate_exceeds_budget")
    try:
        source_bytes = SOURCE.read_bytes()
        if hashlib.sha256(source_bytes).hexdigest() != SOURCE_SHA256:
            return blocked("upstream_source_hash_mismatch")
        import httpx
        from openai import OpenAI
        spec = importlib.util.spec_from_file_location("connection_upstream_open_ai", SOURCE)
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        ledger.parent.mkdir(parents=True, exist_ok=True)
        result_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as error:
        result = blocked("local_prerequisite_failed")
        result["error_type"] = type(error).__name__
        return result

    # An injected transport is always an offline test, regardless of caller label.
    is_fixture = transport is not None
    scope = "offline_fixture" if is_fixture else evidence_scope
    inner = transport if is_fixture else httpx.HTTPTransport(retries=0)
    state = {"attempts": 0, "http_status": None, "usage": None,
             "finish_reason": None, "response_model": None, "error_type": None}

    class LimitedTransport(httpx.BaseTransport):
        def handle_request(self, request):
            body = json.loads(request.content)
            if (request.method != "POST" or str(request.url) != "https://api.deepseek.com/chat/completions"
                    or body.get("model") != "deepseek-flash" or body.get("max_tokens") != 128
                    or body.get("thinking") != {"type": "disabled"}
                    or body.get("messages") != MESSAGES or body.get("stream", False)
                    or set(body) - {"model", "messages", "temperature", "max_tokens", "thinking", "stream"}):
                raise RuntimeError("request_outside_authorized_connection_probe")
            if state["attempts"]:
                raise RuntimeError("authorized_attempt_already_reserved")
            # Reserve before DNS/TLS/HTTP. Crashes and timeouts never refund it.
            with ledger.open("x", encoding="utf-8") as handle:
                json.dump({"authorization_id": AUTHORIZATION_ID,
                    "scope": scope, "http_attempts_reserved": 1,
                    "reserved_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "request_body_sha256": hashlib.sha256(request.content).hexdigest(),
                    "money_budget_cny": 1, "max_output_tokens": 128}, handle, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            state["attempts"] = 1
            response = inner.handle_request(request)
            state["http_status"] = response.status_code
            return response

        def close(self):
            inner.close()

    def client_factory(**kwargs):
        client = OpenAI(**kwargs, max_retries=0, timeout=60,
            http_client=httpx.Client(transport=LimitedTransport(), follow_redirects=False))
        create = client.chat.completions.create

        def bounded_create(**call_kwargs):
            try:
                response = create(**call_kwargs, extra_body={"thinking": {"type": "disabled"}})
                data = response.model_dump()
                usage = data.get("usage")
                if isinstance(usage, dict):
                    state["usage"] = {name: usage[name] for name in
                        ("prompt_tokens", "completion_tokens", "total_tokens", "prompt_cache_hit_tokens",
                         "prompt_cache_miss_tokens") if type(usage.get(name)) is int and usage[name] >= 0}
                if isinstance(data.get("model"), str):
                    state["response_model"] = data["model"][:160]
                if data.get("choices"):
                    reason = data["choices"][0].get("finish_reason")
                    state["finish_reason"] = reason if isinstance(reason, str) else None
                return response
            except Exception as error:
                state["error_type"] = type(error).__name__
                raise

        client.chat.completions.create = bounded_create
        return client

    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    clock_start = time.perf_counter()
    reply = None
    caller = None
    try:
        # Original failure handling prints tracebacks; don't expose provider error
        # bodies or headers. Preserve sanitized error class/status instead.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with patch.object(original, "OpenAI", client_factory):
                caller = original.OpenAICall(model=profile["model_id"], api_key=key,
                    api_type="open_ai", api_base=profile["api_base"], model_type="chat",
                    retry_limits=1, timeout=60)
                reply = caller.call(MESSAGES[1]["content"], sys_prompt=MESSAGES[0]["content"], max_tokens=128)
    except Exception as error:
        state["error_type"] = type(error).__name__
    finally:
        if caller is not None:
            caller.openai_client.close()
        else:
            inner.close()

    usage = state["usage"]
    usage_valid = (isinstance(usage, dict) and
        all(type(usage.get(name)) is int for name in ("prompt_tokens", "completion_tokens", "total_tokens"))
        and usage["completion_tokens"] <= 128
        and usage["prompt_tokens"] + usage["completion_tokens"] == usage["total_tokens"])
    successful = (state["http_status"] == 200 and state["error_type"] is None
                  and bool(reply) and state["attempts"] == 1 and usage_valid)
    estimated_ceiling = ((usage["prompt_tokens"] * 2 + usage["completion_tokens"] * 8) / 1_000_000
                         if usage_valid else None)
    if estimated_ceiling is not None and estimated_ceiling > 1:
        successful = False
    report = {
        "status": "SUCCESS" if successful else "FAILED", "authorization_id": AUTHORIZATION_ID,
        "scope": scope, "live_connection": "VERIFIED" if successful and not is_fixture else "NOT VERIFIED",
        "http_attempts": state["attempts"],
        "remote_api_requests": 0 if is_fixture else (1 if state["http_status"] is not None else None),
        "provider_delivery": "offline_fixture" if is_fixture else
            ("HTTP_response_received" if state["http_status"] is not None else "NOT VERIFIED"),
        "http_status": state["http_status"], "error_type": state["error_type"],
        "requested_model": "deepseek-flash", "response_model": state["response_model"],
        "thinking": "disabled", "max_output_tokens": 128,
        "sdk_max_retries": 0, "upstream_retry_limits": 1, "http_transport_retries": 0,
        "started_at_utc": started, "elapsed_seconds": time.perf_counter() - clock_start,
        "reply": reply[:512] if successful else None,
        "usage": usage, "usage_is_fixture": is_fixture,
        "actual_cost_cny": None, "actual_cost_reason": "provider_bill_not_queried",
        "estimated_cost_cny_peak_ceiling": estimated_ceiling,
        "price_source": PRICE_SOURCE, "price_checked_date": "2026-10-04",
        "planning_input_token_allowance": PLANNED_INPUT_ALLOWANCE,
        "planning_estimated_cost_cny": PLANNED_COST_CNY,
        "money_budget_cny": 1, "provider_billing_hard_cap_enforced": False,
        "upstream_source_sha256": SOURCE_SHA256, "original_app": "NOT VERIFIED",
    }
    report = scrub(report, key)
    if state["attempts"]:
        # Never overwrite an earlier result. The reservation also survives errors.
        with result_path.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=True, indent=2)
            handle.write("\n")
    return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        profile = json.loads(PROFILE.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        report = blocked("local_profile_missing_or_invalid")
    else:
        try:
            key = read_api_key()
        except LocalKeyError:
            report = blocked("local_key_configuration_invalid")
        else:
            report = run_once(profile, key)
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0 if report["status"] == "SUCCESS" else (2 if report["status"] == "BLOCKED" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
