"""Fail-closed request, token-observation and money-budget accounting."""

from dataclasses import dataclass
import math
from threading import Lock

from ..config import LLMBudget
from .errors import BudgetExceeded, LiveCallsDisabled


@dataclass
class _Attempt:
    number: int
    planned_cost_ceiling: float
    is_live: bool
    status: str = "started"
    usage: object | None = None
    usage_unknown_reason: str | None = None
    cost: float | None = None
    cost_unknown_reason: str | None = None


class BudgetLedger:
    """Counts an attempt before transport and never refunds request capacity."""

    def __init__(self, config: LLMBudget):
        if not isinstance(config, LLMBudget):
            raise TypeError("config must be LLMBudget")
        self.config = config
        self._attempts = []
        self._lock = Lock()

    @property
    def attempts(self):
        with self._lock:
            return len(self._attempts)

    def begin_attempt(self, *, planned_cost_ceiling, is_live=True):
        if type(is_live) is not bool:
            raise ValueError("is_live must be boolean")
        if (type(planned_cost_ceiling) not in (int, float)
                or not math.isfinite(planned_cost_ceiling)
                or planned_cost_ceiling < 0):
            raise BudgetExceeded("finite_nonnegative_planned_cost_ceiling_required")
        planned_cost_ceiling = float(planned_cost_ceiling)
        with self._lock:
            if is_live and not self.config.allow_paid_api:
                raise LiveCallsDisabled("live_llm_calls_disabled")
            if is_live and planned_cost_ceiling <= 0:
                raise BudgetExceeded("live_attempt_requires_positive_cost_ceiling")
            live_attempts = [item for item in self._attempts if item.is_live]
            if is_live and len(live_attempts) >= self.config.api_request_cap:
                raise BudgetExceeded("api_request_cap_exhausted")
            committed = sum(item.planned_cost_ceiling for item in live_attempts)
            if is_live and committed + planned_cost_ceiling > self.config.money_budget + 1e-12:
                raise BudgetExceeded("money_budget_would_be_exceeded")
            if not is_live and planned_cost_ceiling != 0:
                raise ValueError("offline fixture attempts require a zero cost ceiling")
            attempt = _Attempt(len(self._attempts) + 1, planned_cost_ceiling, is_live)
            self._attempts.append(attempt)
            return attempt.number

    def finish_attempt(self, number, *, status, usage, usage_unknown_reason,
                       cost, cost_unknown_reason):
        if not isinstance(status, str) or not status:
            raise ValueError("status is required")
        if usage is None and not usage_unknown_reason:
            raise ValueError("unknown usage requires a reason")
        if usage is not None and usage_unknown_reason is not None:
            raise ValueError("known usage cannot have an unknown reason")
        if cost is None and not cost_unknown_reason:
            raise ValueError("unknown cost requires a reason")
        if cost is not None:
            if (type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0):
                raise ValueError("cost must be finite and nonnegative")
            cost = float(cost)
            if cost_unknown_reason is not None:
                raise ValueError("known cost cannot have an unknown reason")
        with self._lock:
            if type(number) is not int or number < 1 or number > len(self._attempts):
                raise ValueError("unknown attempt number")
            attempt = self._attempts[number - 1]
            if attempt.status != "started":
                raise ValueError("attempt already finished")
            cost_exceeded = (cost is not None and attempt.is_live
                             and cost > attempt.planned_cost_ceiling + 1e-12)
            attempt.status = "cost_ceiling_exceeded" if cost_exceeded else status
            attempt.usage = usage
            attempt.usage_unknown_reason = usage_unknown_reason
            attempt.cost = cost
            attempt.cost_unknown_reason = cost_unknown_reason
            if cost_exceeded:
                raise BudgetExceeded("actual_cost_exceeded_planned_ceiling")

    def snapshot(self):
        with self._lock:
            attempts = list(self._attempts)
        finished = [item for item in attempts if item.status != "started"]
        usage_unknown = [item.usage_unknown_reason for item in finished if item.usage is None]
        cost_unknown = [item.cost_unknown_reason for item in finished if item.cost is None]

        usage = None
        usage_reason = "no_completed_attempts"
        if finished and not usage_unknown:
            usage = {
                "prompt_tokens": sum(item.usage.prompt_tokens for item in finished),
                "completion_tokens": sum(item.usage.completion_tokens for item in finished),
                "total_tokens": sum(item.usage.total_tokens for item in finished),
            }
            usage_reason = None
        elif usage_unknown:
            usage_reason = ";".join(dict.fromkeys(usage_unknown))

        actual_cost = None
        cost_reason = "no_completed_attempts"
        if finished and not cost_unknown:
            actual_cost = sum(item.cost for item in finished)
            cost_reason = None
        elif cost_unknown:
            cost_reason = ";".join(dict.fromkeys(cost_unknown))

        return {
            "allow_paid_api": self.config.allow_paid_api,
            "provider": self.config.provider,
            "model_id": self.config.model_id,
            "attempts": len(attempts),
            "live_attempts": sum(item.is_live for item in attempts),
            "fixture_attempts": sum(not item.is_live for item in attempts),
            "api_request_cap": self.config.api_request_cap,
            "usage": usage,
            "usage_unknown_reason": usage_reason,
            "actual_cost": actual_cost,
            "cost_unknown_reason": cost_reason,
            "estimated_cost_ceiling": sum(item.planned_cost_ceiling for item in attempts),
            "money_budget": self.config.money_budget,
            "budget_unit": self.config.budget_unit,
            "attempt_records": [
                {
                    "number": item.number,
                    "is_live": item.is_live,
                    "status": item.status,
                    "usage": None if item.usage is None else {
                        "prompt_tokens": item.usage.prompt_tokens,
                        "completion_tokens": item.usage.completion_tokens,
                        "total_tokens": item.usage.total_tokens,
                    },
                    "usage_unknown_reason": item.usage_unknown_reason,
                    "actual_cost": item.cost,
                    "cost_unknown_reason": item.cost_unknown_reason,
                    "planned_cost_ceiling": item.planned_cost_ceiling,
                }
                for item in attempts
            ],
        }
