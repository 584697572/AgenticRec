"""HTTP and persistent budget boundary for the explicitly authorized A1 session."""
import hashlib
import json
import math
import os
import time
from pathlib import Path

from live_app_http import write_new
from live_llm_connection import scrub

AUTHORIZATION_ID = "deepseek-a1-multiturn-eval-20261005"
AUTHORIZED_PROFILE = {
    "provider": "deepseek", "api_type": "open_ai", "api_base": "https://api.deepseek.com",
    "api_version": None, "model_id": "deepseek-flash", "allow_paid_api": True,
    "api_request_cap": 120, "max_output_tokens": 1024, "money_budget": 3, "budget_unit": "CNY",
}
QUERIES = (
    "Recommend three comedy movies released since 1990.",
    "Now recommend three comedy movies released before 1980 instead. Keep the comedy requirement.",
)
DATA_SHA256 = "f63630aa236198145e07e3ebae18e9c9d044faff55e21adc961aec7e3980b095"
STOP = ["Obsersation", "observation", "Observation:", "observation:"]


def valid_usage(usage, maximum):
    return (isinstance(usage, dict) and all(type(usage.get(k)) is int and usage[k] >= 0
        for k in ("prompt_tokens", "completion_tokens", "total_tokens"))
        and usage["prompt_tokens"] + usage["completion_tokens"] == usage["total_tokens"]
        and usage["completion_tokens"] <= maximum)


class SessionHTTP:
    def __init__(self, directory, key, contexts, profile=None, transport=None):
        import httpx
        self.profile = dict(AUTHORIZED_PROFILE if profile is None else profile)
        if transport is None and self.profile != AUTHORIZED_PROFILE:
            raise ValueError("live_profile_does_not_match_authorization")
        for field in ("api_request_cap", "max_output_tokens"):
            if type(self.profile.get(field)) is not int or self.profile[field] <= 0:
                raise ValueError("invalid_request_or_token_budget")
        money = self.profile.get("money_budget")
        if type(money) not in (int, float) or not math.isfinite(money) or money <= 0:
            raise ValueError("invalid_money_budget")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.key = key
        self.contexts = frozenset(contexts)
        if not self.contexts or any(not isinstance(c, str) or not c for c in self.contexts):
            raise ValueError("invalid_visible_contexts")
        self.active_context = None
        self.is_fixture = transport is not None
        self.calls = []
        self.client = httpx.Client(transport=transport or httpx.HTTPTransport(retries=0),
                                   follow_redirects=False, timeout=60)

    def set_context(self, context):
        if context not in self.contexts:
            raise ValueError("context_outside_authorized_input")
        self.active_context = context

    def _previous_cost(self):
        cost = 0.0
        requests = sorted(self.directory.glob("request_*.json"))
        if len(requests) >= self.profile["api_request_cap"]:
            raise RuntimeError("request_budget_exhausted")
        for path in requests:
            reservation = json.loads(path.read_text(encoding="utf-8"))
            if reservation["authorization_id"] != AUTHORIZATION_ID:
                raise RuntimeError("authorization_ledger_mismatch")
            response_path = self.directory / path.name.replace("request_", "response_")
            if not response_path.exists():
                raise RuntimeError("unresolved_prior_attempt")
            record = json.loads(response_path.read_text(encoding="utf-8"))
            if not record.get("provider_response_valid"):
                raise RuntimeError("prior_provider_failure")
            cost += record["estimated_cost_cny_peak_ceiling"]
        return cost

    def send(self, body):
        payload = json.dumps(body, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
        messages = body.get("messages") if isinstance(body, dict) else None
        valid = (isinstance(body, dict) and body.get("model") == self.profile["model_id"]
            and type(body.get("max_tokens")) is int and 0 < body["max_tokens"] <= self.profile["max_output_tokens"]
            and body.get("thinking") == {"type": "disabled"} and body.get("stream", False) is False
            and len(payload) <= 100000
            and not (set(body) - {"messages", "model", "max_tokens", "temperature", "thinking", "stream", "stop"})
            and ("stop" not in body or body["stop"] == STOP)
            and isinstance(messages, list) and len(messages) == 2
            and all(isinstance(m, dict) and set(m) == {"role", "content"}
                    and isinstance(m["content"], str) for m in messages)
            and [m["role"] for m in messages] == ["system", "user"]
            and self.active_context is not None and self.active_context in messages[-1]["content"])
        if not valid:
            raise ValueError("request_outside_authorized_scope")
        lock = self.directory / ".send.lock"
        try:
            with lock.open("x", encoding="utf-8") as handle:
                handle.write(str(os.getpid()))
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError:
            raise RuntimeError("concurrent_or_unresolved_sender") from None
        try:
            previous_cost = self._previous_cost()
            # Conservative byte-based planning allowance, not the provider's tokenizer
            # or a billing hard cap. Actual usage reconciles this after each response.
            input_allowance = sum(len(m["content"].encode("utf-8")) for m in messages) + 256
            planned_cost = (input_allowance * 2 + body["max_tokens"] * 8) / 1000000
            if previous_cost + planned_cost > self.profile["money_budget"]:
                raise RuntimeError("money_budget_exhausted")
            number = len(list(self.directory.glob("request_*.json"))) + 1
            write_new(self.directory / ("request_%04d.json" % number), {
                "authorization_id": AUTHORIZATION_ID, "attempt": number,
                "usage_is_fixture": self.is_fixture,
                "request_body_sha256": hashlib.sha256(payload).hexdigest(),
                "visible_context_sha256": hashlib.sha256(self.active_context.encode()).hexdigest(),
                "max_output_tokens": body["max_tokens"], "input_token_planning_allowance": input_allowance,
                "planning_estimated_cost_cny_peak_ceiling": planned_cost,
                "money_budget_cny": self.profile["money_budget"]})
            record = {"attempt": number, "request": body, "http_status": None, "response": None,
                      "error_type": None, "usage_is_fixture": self.is_fixture,
                      "provider_response_valid": False, "estimated_cost_cny_peak_ceiling": None}
            clock = time.perf_counter()
            try:
                response = self.client.post(self.profile["api_base"] + "/chat/completions",
                    headers={"Authorization": "Bearer " + self.key}, json=body)
                record["http_status"] = response.status_code
                content = response.text.replace(self.key, "[REDACTED]") if self.key else response.text
                try:
                    parsed = json.loads(content)
                except ValueError:
                    parsed = {"unparseable_response": True}
                record["response"] = parsed
                usage = parsed.get("usage") if isinstance(parsed, dict) else None
                record["provider_response_valid"] = (response.status_code == 200
                    and valid_usage(usage, body["max_tokens"])
                    and parsed.get("model") == self.profile["model_id"])
                if record["provider_response_valid"]:
                    record["estimated_cost_cny_peak_ceiling"] = (
                        usage["prompt_tokens"] * 2 + usage["completion_tokens"] * 8) / 1000000
                return {"status_code": response.status_code, "content": content}
            except Exception as error:
                record["error_type"] = type(error).__name__
                return {"transport_error": type(error).__name__}
            finally:
                record["elapsed_seconds"] = time.perf_counter() - clock
                record = scrub(record, self.key)
                self.calls.append(record)
                write_new(self.directory / ("response_%04d.json" % number), record)
        finally:
            lock.unlink()

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def validate_turn(rows, answer, tool_names, required_tools, turn):
    year_check = (lambda y: y >= 1990) if turn == 0 else (lambda y: y < 1980)
    checks = {
        "three_unique_catalog_items": len(rows) == 3 and len({r["id"] for r in rows}) == 3,
        "all_comedy": bool(rows) and all("Comedy" in r["tags"] for r in rows),
        "current_year_range": bool(rows) and all(year_check(r["year"]) for r in rows),
        "mapped_titles_in_answer": bool(rows) and all(r["title"].casefold() in answer.casefold() for r in rows),
        "filter_rank_map_executed": all(name in tool_names for name in required_tools),
    }
    return {"passed": all(checks.values()), "checks": checks}
