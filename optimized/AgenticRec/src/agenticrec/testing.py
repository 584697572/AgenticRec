"""Deterministic FakeLLM and a fail-closed budget primitive for offline fixtures."""
from dataclasses import dataclass
from .config import LLMBudget
from .runtime.budget import BudgetLedger


@dataclass(frozen=True)
class FakeReply:
    content: str
    usage: None = None
    usage_unknown_reason: str = "fixture_has_no_provider_usage"
    cost: None = None
    cost_unknown_reason: str = "fixture_has_no_provider_cost"
    is_fixture: bool = True
    remote_api_requests: int = 0


class FakeLLM:
    is_live = False

    def __init__(self, responses):
        self.responses = list(responses)
        if not all(isinstance(item, str) for item in self.responses):
            raise ValueError("fixture responses must be strings")
        self.calls = []
        self.call_options = []

    def chat(self, messages, **options):
        if len(self.calls) >= len(self.responses):
            raise RuntimeError("fixture_responses_exhausted")
        content = self.responses[len(self.calls)]
        self.calls.append(tuple((m["role"], m["content"]) for m in messages))
        self.call_options.append(dict(options))
        return FakeReply(content)


class AttemptBudget:
    """Backward-compatible test facade over the unified runtime ledger."""
    def __init__(self, config: LLMBudget):
        self.config = config
        self._ledger = BudgetLedger(config)

    @property
    def attempts(self):
        return self._ledger.attempts

    def reserve(self):
        ceiling = (self.config.money_budget / self.config.api_request_cap
                   if self.config.api_request_cap else 0)
        return self._ledger.begin_attempt(planned_cost_ceiling=ceiling)
