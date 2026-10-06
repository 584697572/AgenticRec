"""Provider-neutral chat adapter with strict JSON and one retry controller."""

from dataclasses import dataclass
import json
import math
import time
from typing import Protocol

from ..config import LLMBudget
from ..runtime.budget import BudgetLedger
from ..runtime.deadline import RoundDeadline
from ..runtime.errors import LLMTransportError, ResponseValidationError


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    def __post_init__(self):
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(name + " must be a nonnegative integer")
        if self.total_tokens != self.prompt_tokens + self.completion_tokens:
            raise ValueError("total_tokens must equal prompt_tokens + completion_tokens")


@dataclass(frozen=True)
class TransportReply:
    content: str
    usage: TokenUsage | None
    usage_unknown_reason: str | None
    cost: float | None
    cost_unknown_reason: str | None
    is_fixture: bool = False
    remote_api_requests: int = 1

    def __post_init__(self):
        if not isinstance(self.content, str):
            raise ValueError("content must be a string")
        if self.usage is not None and not isinstance(self.usage, TokenUsage):
            raise ValueError("usage must be TokenUsage or null")
        if self.usage is None and not self.usage_unknown_reason:
            raise ValueError("unknown usage requires a reason")
        if self.usage is not None and self.usage_unknown_reason is not None:
            raise ValueError("known usage cannot have an unknown reason")
        if self.cost is None and not self.cost_unknown_reason:
            raise ValueError("unknown cost requires a reason")
        if self.cost is not None:
            if (type(self.cost) not in (int, float)
                    or not math.isfinite(self.cost) or self.cost < 0):
                raise ValueError("cost must be finite and nonnegative")
            if self.cost_unknown_reason is not None:
                raise ValueError("known cost cannot have an unknown reason")
        if type(self.is_fixture) is not bool:
            raise ValueError("is_fixture must be boolean")
        if type(self.remote_api_requests) is not int or self.remote_api_requests < 0:
            raise ValueError("remote_api_requests must be a nonnegative integer")


class ChatTransport(Protocol):
    def chat(self, messages, *, model, max_output_tokens, timeout_seconds,
             sdk_max_retries): ...


class StrictObjectSchema:
    """Small exact-key JSON object schema used by agent response contracts."""

    def __init__(self, required_fields):
        if not isinstance(required_fields, dict) or not required_fields:
            raise ValueError("required_fields must be a nonempty mapping")
        for name, expected in required_fields.items():
            if not isinstance(name, str) or not name or not isinstance(expected, type):
                raise ValueError("schema fields must map names to Python types")
        self.required_fields = dict(required_fields)

    def validate(self, value):
        if not isinstance(value, dict):
            raise ResponseValidationError("response JSON must be an object")
        missing = sorted(set(self.required_fields) - set(value))
        if missing:
            raise ResponseValidationError("missing fields: " + ", ".join(missing))
        extra = sorted(set(value) - set(self.required_fields))
        if extra:
            raise ResponseValidationError("unknown fields: " + ", ".join(extra))
        for name, expected in self.required_fields.items():
            actual = value[name]
            valid = isinstance(actual, expected) and not (
                expected in (int, float) and isinstance(actual, bool)
            )
            if expected is float:
                valid = isinstance(actual, (int, float)) and not isinstance(actual, bool)
                valid = valid and math.isfinite(actual)
            if not valid:
                raise ResponseValidationError(
                    f"field {name} must be {expected.__name__}"
                )
        return dict(value)


@dataclass(frozen=True)
class ChatResult:
    value: dict
    content: str
    usage: TokenUsage | None
    usage_unknown_reason: str | None
    cost: float | None
    cost_unknown_reason: str | None
    attempts: int
    remote_api_requests: int


def _validate_messages(messages):
    if not isinstance(messages, (list, tuple)) or not messages:
        raise ValueError("messages must be a nonempty list")
    normalized = []
    for message in messages:
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ValueError("each message requires exactly role and content")
        if message["role"] not in {"system", "user", "assistant", "tool"}:
            raise ValueError("unsupported message role")
        if not isinstance(message["content"], str):
            raise ValueError("message content must be a string")
        normalized.append({"role": message["role"], "content": message["content"]})
    return tuple(normalized)


def _coerce_reply(value):
    if isinstance(value, TransportReply):
        return value
    try:
        return TransportReply(
            content=value.content,
            usage=value.usage,
            usage_unknown_reason=value.usage_unknown_reason,
            cost=value.cost,
            cost_unknown_reason=value.cost_unknown_reason,
            is_fixture=value.is_fixture,
            remote_api_requests=value.remote_api_requests,
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise ResponseValidationError("transport returned an invalid reply") from error


class ChatAdapter:
    def __init__(self, transport: ChatTransport, config: LLMBudget, *, ledger=None,
                 clock=time.monotonic, sleep=time.sleep):
        if not isinstance(config, LLMBudget):
            raise TypeError("config must be LLMBudget")
        self.transport = transport
        self.config = config
        self.ledger = ledger or BudgetLedger(config)
        if self.ledger.config != config:
            raise ValueError("ledger and adapter must share the same configuration")
        self._clock = clock
        self._sleep = sleep
        self._is_live = getattr(transport, "is_live", True)
        if type(self._is_live) is not bool:
            raise ValueError("transport is_live must be boolean")

    def chat_json(self, messages, *, schema, planned_cost_ceiling, deadline=None):
        if not isinstance(schema, StrictObjectSchema):
            raise TypeError("schema must be StrictObjectSchema")
        messages = _validate_messages(messages)
        deadline = deadline or RoundDeadline(
            self.config.round_deadline_seconds, clock=self._clock
        )
        if not isinstance(deadline, RoundDeadline):
            raise TypeError("deadline must be RoundDeadline")
        attempts_used = 0

        for retry_index in range(self.config.max_retries + 1):
            timeout = deadline.timeout_for_attempt(
                self.config.per_request_timeout_seconds
            )
            attempt = self.ledger.begin_attempt(
                planned_cost_ceiling=planned_cost_ceiling,
                is_live=self._is_live,
            )
            attempts_used += 1
            try:
                raw_reply = self.transport.chat(
                    messages,
                    model=self.config.model_id,
                    max_output_tokens=self.config.max_output_tokens,
                    timeout_seconds=timeout,
                    sdk_max_retries=self.config.sdk_max_retries,
                )
                transport_reply = _coerce_reply(raw_reply)
            except LLMTransportError as error:
                self.ledger.finish_attempt(
                    attempt,
                    status="transport_error",
                    usage=None,
                    usage_unknown_reason="transport_error_before_usage",
                    cost=None,
                    cost_unknown_reason="transport_error_before_cost",
                )
                if retry_index >= self.config.max_retries or not error.should_retry():
                    raise
                self._wait_for_retry(deadline)
                continue
            except (ConnectionError, TimeoutError) as error:
                self.ledger.finish_attempt(
                    attempt,
                    status="transport_error",
                    usage=None,
                    usage_unknown_reason="transport_error_before_usage",
                    cost=None,
                    cost_unknown_reason="transport_error_before_cost",
                )
                wrapped = LLMTransportError(
                    str(error),
                    error_type="timeout" if isinstance(error, TimeoutError) else "connection",
                )
                if retry_index >= self.config.max_retries:
                    raise wrapped from error
                self._wait_for_retry(deadline)
                continue
            except Exception:
                self.ledger.finish_attempt(
                    attempt,
                    status="transport_error",
                    usage=None,
                    usage_unknown_reason="unexpected_transport_error_before_usage",
                    cost=None,
                    cost_unknown_reason="unexpected_transport_error_before_cost",
                )
                raise

            self.ledger.finish_attempt(
                attempt,
                status="success",
                usage=transport_reply.usage,
                usage_unknown_reason=transport_reply.usage_unknown_reason,
                cost=transport_reply.cost,
                cost_unknown_reason=transport_reply.cost_unknown_reason,
            )
            try:
                decoded = json.loads(transport_reply.content)
            except json.JSONDecodeError as error:
                raise ResponseValidationError("response content is not valid JSON") from error
            value = schema.validate(decoded)
            return ChatResult(
                value=value,
                content=transport_reply.content,
                usage=transport_reply.usage,
                usage_unknown_reason=transport_reply.usage_unknown_reason,
                cost=transport_reply.cost,
                cost_unknown_reason=transport_reply.cost_unknown_reason,
                attempts=attempts_used,
                remote_api_requests=attempts_used if self._is_live else 0,
            )
        raise AssertionError("unreachable retry loop")

    def _wait_for_retry(self, deadline):
        delay = float(self.config.retry_backoff_seconds)
        deadline.require_backoff_window(delay)
        if delay:
            self._sleep(delay)
