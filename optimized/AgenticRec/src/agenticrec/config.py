"""Strict configuration foundation. Files use the JSON subset of YAML 1.2."""
from dataclasses import dataclass, fields
import json
import math
from pathlib import Path


def exact_keys(data, cls):
    if not isinstance(data, dict) or set(data) - {f.name for f in fields(cls)}:
        raise ValueError("configuration must be an object with known fields only")


@dataclass(frozen=True)
class LLMBudget:
    allow_paid_api: bool = False
    provider: str | None = None
    model_id: str | None = None
    api_request_cap: int = 0
    max_output_tokens: int = 0
    money_budget: float = 0
    budget_unit: str = "CNY"
    per_request_timeout_seconds: float = 30.0
    round_deadline_seconds: float = 90.0
    max_retries: int = 1
    retry_backoff_seconds: float = 0.25
    sdk_max_retries: int = 0

    def __post_init__(self):
        if type(self.allow_paid_api) is not bool:
            raise ValueError("allow_paid_api must be boolean")
        for name in ("api_request_cap", "max_output_tokens", "max_retries", "sdk_max_retries"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(name + " must be a nonnegative integer")
        if (type(self.money_budget) not in (int, float) or not math.isfinite(self.money_budget)
                or self.money_budget < 0):
            raise ValueError("money_budget must be finite and nonnegative")
        for name in ("provider", "model_id"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(name + " must be a nonempty string or null")
        if not isinstance(self.budget_unit, str) or not self.budget_unit.strip():
            raise ValueError("budget_unit is required")
        for name in ("per_request_timeout_seconds", "round_deadline_seconds"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(name + " must be finite and positive")
        if (type(self.retry_backoff_seconds) not in (int, float)
                or not math.isfinite(self.retry_backoff_seconds)
                or self.retry_backoff_seconds < 0):
            raise ValueError("retry_backoff_seconds must be finite and nonnegative")
        if self.sdk_max_retries != 0:
            raise ValueError("sdk_max_retries must be 0; the adapter owns retry accounting")
        if self.allow_paid_api and (not self.provider or not self.model_id or
                min(self.api_request_cap, self.max_output_tokens, self.money_budget) <= 0):
            raise ValueError("live calls require explicit provider, model, request/token/money limits")


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int = 1
    seed: int = 42
    domain: str = "movie"
    llm: LLMBudget = LLMBudget()

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported schema_version")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.domain != "movie":
            raise ValueError("P0 supports only the movie domain")
        if not isinstance(self.llm, LLMBudget):
            raise ValueError("llm must conform to LLMBudget")


def parse_config(data):
    exact_keys(data, ExperimentConfig)
    values = dict(data)
    llm = values.pop("llm", {})
    exact_keys(llm, LLMBudget)
    return ExperimentConfig(**values, llm=LLMBudget(**llm))


def load_config(path: Path):
    return parse_config(json.loads(path.read_text(encoding="utf-8-sig")))
