import pytest

from agenticrec.adapters.llm import (
    ChatAdapter,
    StrictObjectSchema,
    TokenUsage,
    TransportReply,
)
from agenticrec.config import LLMBudget
from agenticrec.runtime.budget import BudgetLedger
from agenticrec.runtime.deadline import RoundDeadline
from agenticrec.runtime.errors import (
    DeadlineExceeded,
    LLMTransportError,
    LiveCallsDisabled,
    ResponseValidationError,
)
from agenticrec.testing import FakeLLM


def live_config(**overrides):
    values = {
        "allow_paid_api": True,
        "provider": "fixture",
        "model_id": "fixture-model",
        "api_request_cap": 4,
        "max_output_tokens": 128,
        "money_budget": 1.0,
        "budget_unit": "CNY",
        "per_request_timeout_seconds": 5.0,
        "round_deadline_seconds": 12.0,
        "max_retries": 1,
        "retry_backoff_seconds": 0.0,
        "sdk_max_retries": 0,
    }
    values.update(overrides)
    return LLMBudget(**values)


class ScriptedTransport:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def reply(content, usage=None):
    return TransportReply(
        content=content,
        usage=usage,
        usage_unknown_reason=None if usage else "fixture_has_no_provider_usage",
        cost=None,
        cost_unknown_reason="provider_bill_not_queried",
    )


def schema():
    return StrictObjectSchema({"route": str, "confidence": float})


def test_fake_llm_strict_json_success_and_sdk_retry_disabled():
    transport = FakeLLM(['{"route":"DIRECT","confidence":0.9}'])
    config = LLMBudget()
    ledger = BudgetLedger(config)
    result = ChatAdapter(transport, config, ledger=ledger).chat_json(
        [{"role": "user", "content": "fixture"}],
        schema=schema(),
        planned_cost_ceiling=0,
    )

    assert result.value == {"route": "DIRECT", "confidence": 0.9}
    assert result.attempts == 1
    assert result.remote_api_requests == 0
    assert transport.call_options[0]["sdk_max_retries"] == 0
    assert ledger.snapshot()["usage"] is None
    assert ledger.snapshot()["fixture_attempts"] == 1
    assert ledger.snapshot()["live_attempts"] == 0


def test_missing_required_response_field_is_rejected_after_counting_attempt():
    transport = ScriptedTransport([reply('{"route":"DIRECT"}')])
    config = live_config()
    ledger = BudgetLedger(config)

    with pytest.raises(ResponseValidationError, match="missing fields: confidence"):
        ChatAdapter(transport, config, ledger=ledger).chat_json(
            [{"role": "user", "content": "fixture"}],
            schema=schema(),
            planned_cost_ceiling=0.01,
        )

    assert ledger.snapshot()["attempts"] == 1


def test_401_is_not_retried():
    transport = ScriptedTransport([
        LLMTransportError("unauthorized", status_code=401),
        reply('{"route":"DIRECT","confidence":0.9}'),
    ])
    config = live_config(max_retries=3)
    ledger = BudgetLedger(config)

    with pytest.raises(LLMTransportError) as error:
        ChatAdapter(transport, config, ledger=ledger).chat_json(
            [{"role": "user", "content": "fixture"}],
            schema=schema(),
            planned_cost_ceiling=0.01,
        )

    assert error.value.status_code == 401
    assert len(transport.calls) == 1
    assert ledger.snapshot()["attempts"] == 1


def test_429_retries_once_and_counts_both_attempts():
    transport = ScriptedTransport([
        LLMTransportError("rate limited", status_code=429),
        reply(
            '{"route":"DIRECT","confidence":0.9}',
            TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        ),
    ])
    config = live_config(max_retries=1)
    ledger = BudgetLedger(config)
    result = ChatAdapter(transport, config, ledger=ledger).chat_json(
        [{"role": "user", "content": "fixture"}],
        schema=schema(),
        planned_cost_ceiling=0.01,
    )

    assert result.attempts == 2
    assert result.remote_api_requests == 2
    assert len(transport.calls) == 2
    assert ledger.snapshot()["attempts"] == 2
    # One failed attempt has unknown usage, so the aggregate remains unknown.
    assert ledger.snapshot()["usage"] is None


def test_unauthorized_live_call_stops_before_transport():
    transport = ScriptedTransport([reply('{"route":"DIRECT","confidence":0.9}')])
    config = LLMBudget()

    with pytest.raises(LiveCallsDisabled):
        ChatAdapter(transport, config).chat_json(
            [{"role": "user", "content": "fixture"}],
            schema=schema(),
            planned_cost_ceiling=0.01,
        )

    assert transport.calls == []


def test_per_request_timeout_is_capped_by_remaining_round_deadline():
    clock = FakeClock()
    deadline = RoundDeadline(3.0, clock=clock)
    clock.advance(1.0)
    transport = ScriptedTransport([reply('{"route":"DIRECT","confidence":0.9}')])
    config = live_config(per_request_timeout_seconds=5.0, round_deadline_seconds=20.0)

    ChatAdapter(transport, config, clock=clock).chat_json(
        [{"role": "user", "content": "fixture"}],
        schema=schema(),
        deadline=deadline,
        planned_cost_ceiling=0.01,
    )

    assert transport.calls[0]["timeout_seconds"] == pytest.approx(2.0)


def test_expired_round_deadline_stops_before_attempt():
    clock = FakeClock()
    deadline = RoundDeadline(1.0, clock=clock)
    clock.advance(1.0)
    transport = ScriptedTransport([reply('{"route":"DIRECT","confidence":0.9}')])
    config = live_config()
    ledger = BudgetLedger(config)

    with pytest.raises(DeadlineExceeded):
        ChatAdapter(transport, config, ledger=ledger, clock=clock).chat_json(
            [{"role": "user", "content": "fixture"}],
            schema=schema(),
            deadline=deadline,
            planned_cost_ceiling=0.01,
        )

    assert ledger.snapshot()["attempts"] == 0
    assert transport.calls == []


def test_round_deadline_covers_time_spent_by_failed_attempts_and_retries():
    clock = FakeClock()

    class SlowFailureTransport(ScriptedTransport):
        def chat(self, messages, **kwargs):
            clock.advance(2.0)
            return super().chat(messages, **kwargs)

    deadline = RoundDeadline(1.0, clock=clock)
    transport = SlowFailureTransport([
        LLMTransportError("rate limited", status_code=429),
        reply('{"route":"DIRECT","confidence":0.9}'),
    ])
    config = live_config(max_retries=1)
    ledger = BudgetLedger(config)

    with pytest.raises(DeadlineExceeded):
        ChatAdapter(transport, config, ledger=ledger, clock=clock).chat_json(
            [{"role": "user", "content": "fixture"}],
            schema=schema(),
            deadline=deadline,
            planned_cost_ceiling=0.01,
        )

    assert len(transport.calls) == 1
    assert ledger.snapshot()["attempts"] == 1


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        "[]",
        '{"route":"DIRECT","confidence":0.9,"extra":true}',
        '{"route":"DIRECT","confidence":"high"}',
        '{"route":"DIRECT","confidence":NaN}',
    ],
)
def test_malformed_or_nonconforming_json_is_rejected(content):
    transport = ScriptedTransport([reply(content)])
    config = live_config()
    with pytest.raises(ResponseValidationError):
        ChatAdapter(transport, config).chat_json(
            [{"role": "user", "content": "fixture"}],
            schema=schema(),
            planned_cost_ceiling=0.01,
        )
