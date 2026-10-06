import pytest

from agenticrec.config import LLMBudget
from agenticrec.runtime.budget import BudgetLedger
from agenticrec.runtime.errors import BudgetExceeded, LiveCallsDisabled


def live_config(**overrides):
    values = {
        "allow_paid_api": True,
        "provider": "fixture",
        "model_id": "fixture-model",
        "api_request_cap": 2,
        "max_output_tokens": 128,
        "money_budget": 1.0,
        "budget_unit": "CNY",
    }
    values.update(overrides)
    return LLMBudget(**values)


def test_default_budget_blocks_before_counting_an_attempt():
    ledger = BudgetLedger(LLMBudget())

    with pytest.raises(LiveCallsDisabled):
        ledger.begin_attempt(planned_cost_ceiling=0.01)

    assert ledger.snapshot()["attempts"] == 0


def test_every_started_attempt_is_counted_and_never_refunded():
    ledger = BudgetLedger(live_config(api_request_cap=1))
    attempt = ledger.begin_attempt(planned_cost_ceiling=0.1)
    ledger.finish_attempt(
        attempt,
        status="transport_error",
        usage=None,
        usage_unknown_reason="connection_failed_before_usage",
        cost=None,
        cost_unknown_reason="provider_bill_not_available",
    )

    assert ledger.snapshot()["attempts"] == 1
    with pytest.raises(BudgetExceeded):
        ledger.begin_attempt(planned_cost_ceiling=0.1)


def test_unknown_usage_and_cost_are_null_with_reasons_not_zero():
    ledger = BudgetLedger(live_config())
    attempt = ledger.begin_attempt(planned_cost_ceiling=0.2)
    ledger.finish_attempt(
        attempt,
        status="success",
        usage=None,
        usage_unknown_reason="fixture_has_no_provider_usage",
        cost=None,
        cost_unknown_reason="provider_bill_not_queried",
    )

    snapshot = ledger.snapshot()
    assert snapshot["usage"] is None
    assert snapshot["usage_unknown_reason"] == "fixture_has_no_provider_usage"
    assert snapshot["actual_cost"] is None
    assert snapshot["cost_unknown_reason"] == "provider_bill_not_queried"
    assert snapshot["estimated_cost_ceiling"] == pytest.approx(0.2)


def test_planned_cost_ceiling_fails_closed_before_transport():
    ledger = BudgetLedger(live_config(money_budget=0.15))

    with pytest.raises(BudgetExceeded):
        ledger.begin_attempt(planned_cost_ceiling=0.2)

    assert ledger.snapshot()["attempts"] == 0


def test_actual_cost_above_ceiling_is_recorded_before_failure():
    ledger = BudgetLedger(live_config())
    attempt = ledger.begin_attempt(planned_cost_ceiling=0.1)

    with pytest.raises(BudgetExceeded, match="actual_cost_exceeded"):
        ledger.finish_attempt(
            attempt,
            status="success",
            usage=None,
            usage_unknown_reason="provider_usage_missing",
            cost=0.11,
            cost_unknown_reason=None,
        )

    record = ledger.snapshot()["attempt_records"][0]
    assert record["status"] == "cost_ceiling_exceeded"
    assert record["actual_cost"] == pytest.approx(0.11)


@pytest.mark.parametrize("planned", [None, -1, 0, float("nan")])
def test_live_attempt_requires_a_finite_cost_ceiling(planned):
    ledger = BudgetLedger(live_config())
    with pytest.raises((BudgetExceeded, ValueError)):
        ledger.begin_attempt(planned_cost_ceiling=planned)
