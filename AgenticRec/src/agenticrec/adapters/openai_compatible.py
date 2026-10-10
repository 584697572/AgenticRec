"""Minimal HTTPS transport for OpenAI-compatible chat-completions providers."""

from dataclasses import dataclass
import json
import socket
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .llm import TokenUsage, TransportReply
from ..runtime.errors import LLMTransportError


_MAX_RESPONSE_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True)
class HttpResponse:
    status_code: int
    body: bytes

    def __post_init__(self):
        if type(self.status_code) is not int or not 100 <= self.status_code <= 599:
            raise ValueError("status_code must be an HTTP status")
        if not isinstance(self.body, bytes):
            raise ValueError("body must be bytes")


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        return None


def _validate_base(api_base):
    if not isinstance(api_base, str) or not api_base.strip():
        raise ValueError("api_base is required")
    parsed = urlsplit(api_base)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("api_base must be a plain HTTPS URL")
    return api_base.rstrip("/")


def _validate_key(api_key):
    if not isinstance(api_key, str) or not api_key:
        raise ValueError("api_key is required")
    if any(character.isspace() or ord(character) < 32 for character in api_key):
        raise ValueError("api_key must be a single value")
    return api_key


def _messages(messages):
    if not isinstance(messages, (list, tuple)) or not messages:
        raise ValueError("messages must be a nonempty list")
    normalized = []
    for message in messages:
        if type(message) is not dict or set(message) != {"role", "content"}:
            raise ValueError("each message requires exactly role and content")
        if message["role"] not in {"system", "user", "assistant", "tool"}:
            raise ValueError("unsupported message role")
        if not isinstance(message["content"], str):
            raise ValueError("message content must be a string")
        normalized.append({"role": message["role"], "content": message["content"]})
    return normalized


class OpenAICompatibleTransport:
    """One-attempt transport. Retry ownership remains in ChatAdapter."""

    is_live = True

    def __init__(self, *, api_base, api_key, sender=None, seed=None):
        self.api_base = _validate_base(api_base)
        self._api_key = _validate_key(api_key)
        if seed is not None and (type(seed) is not int or seed < 0):
            raise ValueError("seed must be a nonnegative integer or null")
        self.seed = seed
        if sender is not None and not callable(sender):
            raise TypeError("sender must be callable")
        self._sender = sender or self._stdlib_send

    def __repr__(self):
        return f"OpenAICompatibleTransport(api_base={self.api_base!r})"

    def chat(
        self,
        messages,
        *,
        model,
        max_output_tokens,
        timeout_seconds,
        sdk_max_retries,
    ):
        if sdk_max_retries != 0:
            raise ValueError("sdk_max_retries must be 0")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model is required")
        if type(max_output_tokens) is not int or max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be a positive integer")
        if type(timeout_seconds) not in (int, float) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        payload = {
            "model": model,
            "messages": _messages(messages),
            "max_tokens": max_output_tokens,
            "response_format": {"type": "json_object"},
            "temperature": 0,
        }
        if urlsplit(self.api_base).hostname == "api.deepseek.com":
            payload["thinking"] = {"type": "disabled"}
        if self.seed is not None:
            payload["seed"] = self.seed
        body = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        headers = {
            "Accept": "application/json",
            "Authorization": "Bearer " + self._api_key,
            "Content-Type": "application/json",
        }
        try:
            response = self._sender(
                self.api_base + "/chat/completions",
                headers,
                body,
                timeout_seconds,
            )
        except (TimeoutError, socket.timeout) as error:
            raise LLMTransportError(
                "provider request timed out", error_type="timeout", retryable=True
            ) from error
        except (ConnectionError, OSError) as error:
            raise LLMTransportError(
                "provider connection failed", error_type="connection", retryable=True
            ) from error
        if not isinstance(response, HttpResponse):
            raise LLMTransportError(
                "invalid provider response", error_type="invalid_response", retryable=False
            )
        if not 200 <= response.status_code < 300:
            raise LLMTransportError(
                "provider returned an HTTP error",
                status_code=response.status_code,
                error_type="http_status",
            )
        return self._decode(response.body)

    @staticmethod
    def _decode(body):
        try:
            value = json.loads(body.decode("utf-8"))
            choices = value["choices"]
            if type(choices) is not list or not choices:
                raise ValueError
            content = choices[0]["message"]["content"]
            if not isinstance(content, str):
                raise ValueError
            raw_usage = value.get("usage")
            if raw_usage is None:
                usage = None
                usage_reason = "provider_usage_missing"
            else:
                if type(raw_usage) is not dict:
                    raise ValueError
                usage = TokenUsage(
                    prompt_tokens=raw_usage["prompt_tokens"],
                    completion_tokens=raw_usage["completion_tokens"],
                    total_tokens=raw_usage["total_tokens"],
                )
                usage_reason = None
        except (KeyError, TypeError, ValueError, UnicodeError, json.JSONDecodeError) as error:
            raise LLMTransportError(
                "invalid provider response", error_type="invalid_response", retryable=False
            ) from error
        return TransportReply(
            content=content,
            usage=usage,
            usage_unknown_reason=usage_reason,
            cost=None,
            cost_unknown_reason="provider_bill_not_queried",
            is_fixture=False,
            remote_api_requests=1,
        )

    @staticmethod
    def _stdlib_send(url, headers, body, timeout_seconds):
        request = Request(url, data=body, headers=headers, method="POST")
        opener = build_opener(_NoRedirect())
        try:
            with opener.open(request, timeout=timeout_seconds) as stream:
                response_body = stream.read(_MAX_RESPONSE_BYTES + 1)
                status_code = stream.status
        except HTTPError as error:
            response_body = error.read(_MAX_RESPONSE_BYTES + 1)
            status_code = error.code
        except URLError as error:
            if isinstance(error.reason, (TimeoutError, socket.timeout)):
                raise TimeoutError("provider request timed out") from error
            raise ConnectionError("provider connection failed") from error
        if len(response_body) > _MAX_RESPONSE_BYTES:
            raise LLMTransportError(
                "provider response exceeded size limit",
                error_type="invalid_response",
                retryable=False,
            )
        return HttpResponse(status_code, response_body)
