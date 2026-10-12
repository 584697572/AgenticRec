"""Development-calibrated rule routing for the F/A/O system variants."""

from copy import deepcopy
from dataclasses import asdict, dataclass
from enum import Enum
import math
from types import MappingProxyType
from typing import Mapping

from ..pipeline import FixedRequest


class Route(str, Enum):
    DIRECT = "DIRECT"
    PERSONALIZED = "PERSONALIZED"
    AGENT = "AGENT"
    CLARIFY = "CLARIFY"


class SystemMode(str, Enum):
    FIXED_PIPELINE = "fixed_pipeline"
    ALWAYS_AGENT = "always_agent"
    OURS_ROUTER = "ours_router"


_HIDDEN_KEYS = frozenset({
    "target", "targets", "ground_truth", "accepted_item_ids",
    "relevant_item_ids", "expected_status", "expected_route", "private",
    "private_target", "evaluator_only",
})


def _finite_probability(value, name):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or value < 0 or value > 1):
        raise ValueError(name + " must be finite and between 0 and 1")
    return float(value)


def _reject_hidden_fields(value):
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError("public_input keys must be strings")
            normalized = key.casefold()
            if (normalized in _HIDDEN_KEYS or normalized.startswith("expected_")
                    or normalized.endswith("_target")):
                raise ValueError("hidden evaluator field cannot enter routing input: " + key)
            _reject_hidden_fields(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_hidden_fields(child)


def _json_value(value):
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_value(child) for key, child in value.items()}
    return value


def fixed_request_payload(request):
    if not isinstance(request, FixedRequest):
        raise TypeError("request must be FixedRequest")
    payload = asdict(request)
    # Keep the public JSON contract used by FixedRequest.parse.
    payload["constraints"] = asdict(request.constraints)
    return _json_value(payload)


@dataclass(frozen=True)
class RoutingRequest:
    request_id: str
    public_input: Mapping
    agent_need_score: float = 0.0
    fixed_request: FixedRequest | None = None
    needs_planning: bool = False
    unresolved_fields: tuple[str, ...] = ()
    conflict_reason: str | None = None

    def __post_init__(self):
        if (not isinstance(self.request_id, str) or not self.request_id.strip()
                or self.request_id != self.request_id.strip()):
            raise ValueError("request_id must be a nonempty trimmed string")
        if not isinstance(self.public_input, dict):
            raise ValueError("public_input must be a JSON object")
        _reject_hidden_fields(self.public_input)
        if self.fixed_request is not None and not isinstance(self.fixed_request, FixedRequest):
            raise ValueError("fixed_request must be FixedRequest or null")
        if type(self.needs_planning) is not bool:
            raise ValueError("needs_planning must be boolean")
        if (type(self.unresolved_fields) is not tuple
                or any(not isinstance(item, str) or not item.strip()
                       for item in self.unresolved_fields)
                or len(set(self.unresolved_fields)) != len(self.unresolved_fields)):
            raise ValueError("unresolved_fields must contain unique nonempty strings")
        if self.conflict_reason is not None and (
                not isinstance(self.conflict_reason, str)
                or not self.conflict_reason.strip()):
            raise ValueError("conflict_reason must be nonempty or null")
        object.__setattr__(self, "agent_need_score", _finite_probability(
            self.agent_need_score, "agent_need_score"
        ))
        object.__setattr__(self, "public_input", MappingProxyType(
            deepcopy(dict(self.public_input))
        ))

    @classmethod
    def from_fixed(cls, request_id, request, **overrides):
        values = {
            "request_id": request_id,
            "public_input": fixed_request_payload(request),
            "agent_need_score": 0.0,
            "fixed_request": request,
        }
        values.update(overrides)
        return cls(**values)


@dataclass(frozen=True)
class RouterPolicy:
    system_mode: SystemMode = SystemMode.OURS_ROUTER
    agent_threshold: float = 0.75
    calibration_split: str = "development"

    def __post_init__(self):
        try:
            mode = SystemMode(self.system_mode)
        except (TypeError, ValueError) as error:
            raise ValueError("unknown system_mode") from error
        if self.calibration_split != "development":
            raise ValueError("routing thresholds may be calibrated on development only")
        object.__setattr__(self, "system_mode", mode)
        object.__setattr__(self, "agent_threshold", _finite_probability(
            self.agent_threshold, "agent_threshold"
        ))


@dataclass(frozen=True)
class RouteDecision:
    route: Route
    reason: str
    system_mode: SystemMode
    agent_need_score: float
    agent_threshold: float


@dataclass(frozen=True)
class CalibrationExample:
    agent_need_score: float
    should_use_agent: bool

    def __post_init__(self):
        object.__setattr__(self, "agent_need_score", _finite_probability(
            self.agent_need_score, "agent_need_score"
        ))
        if type(self.should_use_agent) is not bool:
            raise ValueError("should_use_agent must be boolean")


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    correct: int
    total: int
    split: str = "development"


def select_agent_threshold(examples, *, candidates, split):
    """Select accuracy, then the largest tied threshold, on development only."""
    if split != "development":
        raise ValueError("agent threshold selection is development-only")
    examples = tuple(examples)
    if not examples or not all(isinstance(item, CalibrationExample) for item in examples):
        raise ValueError("development calibration examples are required")
    candidates = tuple(_finite_probability(value, "candidate threshold")
                       for value in candidates)
    if not candidates:
        raise ValueError("at least one candidate threshold is required")
    scored = []
    for threshold in sorted(set(candidates)):
        correct = sum(
            (example.agent_need_score >= threshold) == example.should_use_agent
            for example in examples
        )
        scored.append((correct, threshold))
    correct, threshold = max(scored)
    return ThresholdSelection(threshold, correct, len(examples))


def _fixed_conflict(request):
    if request is None:
        return None
    constraints = request.constraints
    included = {item.casefold() for item in constraints.include_genres}
    excluded = {item.casefold() for item in constraints.exclude_genres}
    if included & excluded:
        return "included_and_excluded_genre_overlap"
    if (constraints.year_min is not None and constraints.year_max is not None
            and constraints.year_min > constraints.year_max):
        return "year_range_is_reversed"
    return None


def _fixed_route(request):
    personalized = bool(
        request.liked_item_ids or request.disliked_item_ids
        or (request.history_authorized and request.user_id is not None)
    )
    return Route.PERSONALIZED if personalized else Route.DIRECT


class Router:
    def __init__(self, policy=None):
        self.policy = policy or RouterPolicy()
        if not isinstance(self.policy, RouterPolicy):
            raise TypeError("policy must be RouterPolicy")

    def decide(self, request):
        if not isinstance(request, RoutingRequest):
            raise TypeError("request must be RoutingRequest")
        mode = self.policy.system_mode
        if mode is SystemMode.ALWAYS_AGENT:
            return self._decision(Route.AGENT, "always_agent_baseline", request)

        conflict = request.conflict_reason or _fixed_conflict(request.fixed_request)
        if conflict:
            return self._decision(Route.CLARIFY, "conflict:" + conflict, request)
        if request.unresolved_fields:
            return self._decision(
                Route.CLARIFY,
                "unresolved:" + ",".join(request.unresolved_fields),
                request,
            )

        if mode is SystemMode.FIXED_PIPELINE:
            if request.fixed_request is None:
                return self._decision(Route.CLARIFY, "fixed_pipeline_requires_parse", request)
            return self._decision(
                _fixed_route(request.fixed_request), "fixed_pipeline_baseline", request
            )

        if request.fixed_request is not None and not request.needs_planning:
            return self._decision(
                _fixed_route(request.fixed_request), "structured_request_needs_no_plan", request
            )
        if request.agent_need_score >= self.policy.agent_threshold:
            return self._decision(Route.AGENT, "agent_need_meets_threshold", request)
        return self._decision(Route.CLARIFY, "agent_need_below_threshold", request)

    def _decision(self, route, reason, request):
        return RouteDecision(
            route, reason, self.policy.system_mode,
            request.agent_need_score, self.policy.agent_threshold,
        )
