"""HTTP boundary for the authorized original-app probe; no recommendation code."""
import hashlib
import json
import os
import time
from pathlib import Path

from live_llm_connection import scrub

PROFILE = {
    "provider": "deepseek", "api_type": "open_ai", "api_base": "https://api.deepseek.com",
    "api_version": None, "model_id": "deepseek-flash", "allow_paid_api": True,
    "api_request_cap": 2, "max_output_tokens": 512, "money_budget": 1, "budget_unit": "CNY",
}
QUERY = "Recommend three comedy movies released since 1990."
AUTHORIZATION_ID = "deepseek-original-app-single-turn-20261005"


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


class BoundedHTTP:
    def __init__(self, directory, key, transport=None):
        import httpx
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.key = key
        self.is_fixture = transport is not None
        self.calls = []
        self.client = httpx.Client(transport=transport or httpx.HTTPTransport(retries=0),
                                   follow_redirects=False, timeout=60)

    def send(self, body):
        payload = json.dumps(body, ensure_ascii=True, separators=(",", ":")).encode()
        valid = (isinstance(body, dict) and body.get("model") == PROFILE["model_id"]
                 and type(body.get("max_tokens")) is int and 0 < body["max_tokens"] <= 512
                 and body.get("thinking") == {"type": "disabled"}
                 and not body.get("stream", False)
                 and len(payload) <= 100000
                 and not (set(body) - {"messages", "model", "max_tokens", "temperature", "thinking", "stream", "stop"})
                 and ("stop" not in body or body["stop"] == ["Obsersation", "observation", "Observation:", "observation:"])
                 and isinstance(body.get("messages"), list) and len(body["messages"]) == 2
                 and all(isinstance(m, dict) and set(m) == {"role", "content"}
                         and isinstance(m["content"], str) for m in body["messages"])
                 and [m["role"] for m in body["messages"]] == ["system", "user"]
                 and QUERY in body["messages"][-1]["content"])
        if not valid:
            raise ValueError("request_outside_authorized_scope")
        # Atomic slots are shared by every invocation using this authorization.
        number = None
        for candidate in (1, 2):
            try:
                write_new(self.directory / ("request_%d.json" % candidate), {
                    "authorization_id": AUTHORIZATION_ID, "attempt": candidate,
                    "scope": "offline_fixture" if self.is_fixture else "live_original_app",
                    "request_body_sha256": hashlib.sha256(payload).hexdigest(),
                    "max_output_tokens": body["max_tokens"], "money_budget_cny": 1})
                number = candidate
                break
            except FileExistsError:
                continue
        if number is None:
            raise RuntimeError("request_budget_exhausted")
        record = {"attempt": number, "request": body, "http_status": None,
                  "response": None, "error_type": None, "usage_is_fixture": self.is_fixture}
        self.calls.append(record)
        clock = time.perf_counter()
        try:
            response = self.client.post(PROFILE["api_base"] + "/chat/completions",
                headers={"Authorization": "Bearer " + self.key}, json=body)
            record["http_status"] = response.status_code
            # Never forward a reflected credential into the resource worker.
            content = response.text.replace(self.key, "[REDACTED]") if self.key else response.text
            try:
                record["response"] = json.loads(content)
            except ValueError:
                record["response"] = {"unparseable_response": True}
            return {"status_code": response.status_code, "content": content}
        except Exception as error:
            record["error_type"] = type(error).__name__
            return {"transport_error": type(error).__name__}
        finally:
            record["elapsed_seconds"] = time.perf_counter() - clock
            cleaned = scrub(record, self.key)
            record.clear()
            record.update(cleaned)
            write_new(self.directory / ("response_%d.json" % number), record)

    def close(self):
        self.client.close()


def validate_recommendations(rows, answer, tool_names, required_tools):
    """Validate recorded original Map IDs against real catalog rows, never LLM self-rating."""
    checks = {
        "three_unique_catalog_items": len(rows) == 3 and len({r["id"] for r in rows}) == 3,
        "all_comedy": bool(rows) and all("Comedy" in r["tags"] for r in rows),
        "all_since_1990": bool(rows) and all(r["year"] >= 1990 for r in rows),
        "mapped_titles_in_answer": bool(rows) and all(r["title"].casefold() in answer.casefold() for r in rows),
        "filter_rank_map_executed": all(name in tool_names for name in required_tools),
    }
    return {"passed": all(checks.values()), "checks": checks}
