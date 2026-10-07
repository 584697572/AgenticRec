import json
import importlib
from pathlib import Path
from unittest.mock import patch

import pytest

from agenticrec.adapters.openai_compatible import (
    HttpResponse,
    OpenAICompatibleTransport,
)
from agenticrec.runtime.errors import LLMTransportError
from agenticrec.runtime.secrets import LocalKeyError, read_api_key
import agenticrec.runtime.secrets as secrets_module


class RecordingSender:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def __call__(self, url, headers, body, timeout_seconds):
        self.calls.append((url, headers, body, timeout_seconds))
        if self.error is not None:
            raise self.error
        return self.response


def provider_reply(*, content='{"route":"DIRECT"}', usage=True):
    value = {
        "id": "fixture-response",
        "model": "deepseek-flash",
        "choices": [{"message": {"role": "assistant", "content": content}}],
    }
    if usage:
        value["usage"] = {
            "prompt_tokens": 12,
            "completion_tokens": 4,
            "total_tokens": 16,
            "prompt_cache_hit_tokens": 0,
        }
    return HttpResponse(200, json.dumps(value).encode("utf-8"))


def test_deepseek_request_is_exact_json_with_no_sdk_retry():
    sender = RecordingSender(provider_reply())
    transport = OpenAICompatibleTransport(
        api_base="https://api.deepseek.com",
        api_key="private-fixture-key",
        sender=sender,
    )

    reply = transport.chat(
        ({"role": "user", "content": "return JSON"},),
        model="deepseek-flash",
        max_output_tokens=128,
        timeout_seconds=9.5,
        sdk_max_retries=0,
    )

    assert len(sender.calls) == 1
    url, headers, body, timeout = sender.calls[0]
    assert url == "https://api.deepseek.com/chat/completions"
    assert headers == {
        "Accept": "application/json",
        "Authorization": "Bearer private-fixture-key",
        "Content-Type": "application/json",
    }
    assert json.loads(body) == {
        "model": "deepseek-flash",
        "messages": [{"role": "user", "content": "return JSON"}],
        "max_tokens": 128,
        "response_format": {"type": "json_object"},
    }
    assert timeout == 9.5
    assert reply.content == '{"route":"DIRECT"}'
    assert reply.usage.prompt_tokens == 12
    assert reply.usage.completion_tokens == 4
    assert reply.usage.total_tokens == 16
    assert reply.cost is None
    assert reply.cost_unknown_reason == "provider_bill_not_queried"
    assert reply.remote_api_requests == 1
    assert reply.is_fixture is False


def test_missing_provider_usage_stays_null_with_reason():
    transport = OpenAICompatibleTransport(
        api_base="https://api.deepseek.com/v1/",
        api_key="fixture",
        sender=RecordingSender(provider_reply(usage=False)),
    )
    reply = transport.chat(
        [{"role": "user", "content": "fixture"}],
        model="deepseek-flash",
        max_output_tokens=8,
        timeout_seconds=1,
        sdk_max_retries=0,
    )
    assert reply.usage is None
    assert reply.usage_unknown_reason == "provider_usage_missing"


@pytest.mark.parametrize(
    ("status", "retryable"),
    [(401, False), (429, True), (500, True), (302, False)],
)
def test_http_status_is_sanitized_and_classified(status, retryable):
    key = "private-fixture-key"
    body = ("provider echoed " + key).encode()
    transport = OpenAICompatibleTransport(
        api_base="https://api.deepseek.com",
        api_key=key,
        sender=RecordingSender(HttpResponse(status, body)),
    )
    with pytest.raises(LLMTransportError) as caught:
        transport.chat(
            [{"role": "user", "content": "fixture"}],
            model="deepseek-flash",
            max_output_tokens=8,
            timeout_seconds=1,
            sdk_max_retries=0,
        )
    assert caught.value.status_code == status
    assert caught.value.should_retry() is retryable
    assert key not in str(caught.value)


@pytest.mark.parametrize(
    "body",
    [
        b"not-json",
        b"{}",
        b'{"choices":[]}',
        b'{"choices":[{"message":{"content":3}}]}',
        b'{"choices":[{"message":{"content":"{}"}}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":9}}',
    ],
)
def test_malformed_provider_response_fails_closed(body):
    transport = OpenAICompatibleTransport(
        api_base="https://api.deepseek.com",
        api_key="fixture",
        sender=RecordingSender(HttpResponse(200, body)),
    )
    with pytest.raises(LLMTransportError, match="invalid provider response") as caught:
        transport.chat(
            [{"role": "user", "content": "fixture"}],
            model="deepseek-flash",
            max_output_tokens=8,
            timeout_seconds=1,
            sdk_max_retries=0,
        )
    assert caught.value.error_type == "invalid_response"
    assert caught.value.should_retry() is False


def test_network_errors_are_sanitized_and_classified():
    key = "private-fixture-key"
    for raw, kind in (
        (TimeoutError(key), "timeout"),
        (ConnectionError(key), "connection"),
        (OSError(key), "connection"),
    ):
        transport = OpenAICompatibleTransport(
            api_base="https://api.deepseek.com",
            api_key=key,
            sender=RecordingSender(error=raw),
        )
        with pytest.raises(LLMTransportError) as caught:
            transport.chat(
                [{"role": "user", "content": "fixture"}],
                model="deepseek-flash",
                max_output_tokens=8,
                timeout_seconds=1,
                sdk_max_retries=0,
            )
        assert caught.value.error_type == kind
        assert caught.value.should_retry() is True
        assert key not in str(caught.value)


@pytest.mark.parametrize(
    "api_base",
    [
        "http://api.deepseek.com",
        "https://user:pass@api.deepseek.com",
        "https://api.deepseek.com?key=fixture",
        "https://api.deepseek.com/#fragment",
    ],
)
def test_provider_base_must_be_plain_https(api_base):
    with pytest.raises(ValueError):
        OpenAICompatibleTransport(api_base=api_base, api_key="fixture")


def test_key_and_sdk_retry_are_never_exposed_or_silently_accepted():
    key = "private-fixture-key"
    transport = OpenAICompatibleTransport(
        api_base="https://api.deepseek.com", api_key=key,
        sender=RecordingSender(provider_reply()),
    )
    assert key not in repr(transport)
    with pytest.raises(ValueError, match="sdk_max_retries must be 0"):
        transport.chat(
            [{"role": "user", "content": "fixture"}],
            model="deepseek-flash",
            max_output_tokens=8,
            timeout_seconds=1,
            sdk_max_retries=1,
        )


def test_environment_key_wins_without_reading_dotenv():
    with patch.object(Path, "read_text", side_effect=AssertionError("secret file read")):
        assert read_api_key(environ={"OPENAI_API_KEY": "env-fixture"}) == "env-fixture"


def test_dotenv_loader_reads_only_key_and_never_echoes_malformed_value(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "allow_paid_api=true\napi_request_cap=999\n"
        "OPENAI_API_KEY='local-fixture' # ignored comment\n",
        encoding="utf-8",
    )
    values = {"UNRELATED": "unchanged"}
    assert read_api_key(env, values) == "local-fixture"
    assert values == {"UNRELATED": "unchanged"}

    env.write_text("OPENAI_API_KEY=hidden\nOPENAI_API_KEY=second\n", encoding="utf-8")
    with pytest.raises(LocalKeyError) as caught:
        read_api_key(env, {})
    assert "hidden" not in str(caught.value)
    assert "second" not in str(caught.value)


def test_importing_secret_module_does_not_read_dotenv():
    with patch.object(Path, "read_text", side_effect=AssertionError("secret file read")):
        importlib.reload(secrets_module)
