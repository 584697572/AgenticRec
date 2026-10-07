import json

import pytest

from agenticrec.adapters.openai_compatible import HttpResponse
from agenticrec.adapters.providers import build_live_chat_adapter
from agenticrec.adapters.llm import StrictObjectSchema
from agenticrec.config import LLMBudget
from agenticrec.runtime.errors import LiveCallsDisabled
from agenticrec.runtime.secrets import LocalKeyError


class FixtureSender:
    def __init__(self):
        self.calls = []

    def __call__(self, url, headers, body, timeout_seconds):
        self.calls.append((url, headers, json.loads(body), timeout_seconds))
        response = {
            "choices": [{"message": {"content": '{"ok":true}'}}],
            "usage": {
                "prompt_tokens": 8,
                "completion_tokens": 3,
                "total_tokens": 11,
            },
        }
        return HttpResponse(200, json.dumps(response).encode())


def live_config(**overrides):
    values = {
        "allow_paid_api": True,
        "provider": "deepseek",
        "model_id": "deepseek-flash",
        "api_request_cap": 1,
        "max_output_tokens": 128,
        "money_budget": 1.0,
        "max_retries": 0,
        "sdk_max_retries": 0,
    }
    values.update(overrides)
    return LLMBudget(**values)


def test_factory_provider_fixture_runs_through_the_shared_budget_ledger():
    sender = FixtureSender()
    adapter = build_live_chat_adapter(
        live_config(), api_key="private-fixture", sender=sender
    )
    result = adapter.chat_json(
        [{"role": "user", "content": "return JSON"}],
        schema=StrictObjectSchema({"ok": bool}),
        planned_cost_ceiling=0.01,
    )

    assert result.value == {"ok": True}
    assert result.remote_api_requests == 1
    snapshot = adapter.ledger.snapshot()
    assert snapshot["live_attempts"] == 1
    assert snapshot["usage"] == {
        "prompt_tokens": 8,
        "completion_tokens": 3,
        "total_tokens": 11,
    }
    assert snapshot["actual_cost"] is None
    assert snapshot["cost_unknown_reason"] == "provider_bill_not_queried"
    assert sender.calls[0][0] == "https://api.deepseek.com/chat/completions"


def test_factory_reads_key_only_when_no_explicit_test_key(monkeypatch):
    monkeypatch.setattr(
        "agenticrec.adapters.providers.read_api_key",
        lambda: "dotenv-fixture",
    )
    adapter = build_live_chat_adapter(live_config(), sender=FixtureSender())
    assert "dotenv-fixture" not in repr(adapter.transport)


def test_factory_rejects_missing_key_and_unregistered_provider_before_network(monkeypatch):
    monkeypatch.setattr("agenticrec.adapters.providers.read_api_key", lambda: "")
    with pytest.raises(LocalKeyError, match="OPENAI_API_KEY is missing"):
        build_live_chat_adapter(live_config())
    with pytest.raises(ValueError, match="unsupported live provider"):
        build_live_chat_adapter(live_config(provider="fixture"), api_key="fixture")


def test_disabled_live_configuration_still_fails_before_transport():
    sender = FixtureSender()
    adapter = build_live_chat_adapter(
        LLMBudget(provider="deepseek", model_id="deepseek-flash"),
        api_key="fixture",
        sender=sender,
    )
    with pytest.raises(LiveCallsDisabled):
        adapter.chat_json(
            [{"role": "user", "content": "fixture"}],
            schema=StrictObjectSchema({"ok": bool}),
            planned_cost_ceiling=0.01,
        )
    assert sender.calls == []
