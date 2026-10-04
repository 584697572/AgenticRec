"""Deterministic FakeLLM and a fail-closed budget primitive for offline fixtures."""
from dataclasses import dataclass
from .config import LLMBudget


@dataclass(frozen=True)
class FakeReply:
    content: str
    usage: None = None
    is_fixture: bool = True
    remote_api_requests: int = 0


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        if not all(isinstance(item, str) for item in self.responses):
            raise ValueError("fixture responses must be strings")
        self.calls = []

    def chat(self, messages):
        if len(self.calls) >= len(self.responses):
            raise RuntimeError("fixture_responses_exhausted")
        content = self.responses[len(self.calls)]
        self.calls.append(tuple((m["role"], m["content"]) for m in messages))
        return FakeReply(content)


class AttemptBudget:
    """Attempt accounting primitive only; contains no network implementation."""
    def __init__(self, config: LLMBudget):
        self.config = config
        self.attempts = 0

    def reserve(self):
        if not self.config.allow_paid_api or self.attempts >= self.config.api_request_cap:
            raise RuntimeError("paid_api_disabled_or_budget_exhausted")
        self.attempts += 1
        return self.attempts
