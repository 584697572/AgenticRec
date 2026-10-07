"""Bounded recommendation agent loop with at most one tool-result replan."""

from copy import deepcopy
from dataclasses import dataclass
import json
import math
import time

from ..adapters.llm import ChatAdapter, StrictObjectSchema
from ..runtime.deadline import RoundDeadline
from ..runtime.errors import (
    BudgetExceeded,
    DeadlineExceeded,
    LLMTransportError,
    LiveCallsDisabled,
    ResponseValidationError,
)
from .executor import PlanExecutor, PlanValidationError, parse_plan
from .router import Route, Router, RoutingRequest


@dataclass(frozen=True)
class AgentLimits:
    max_planner_calls: int = 2
    max_tool_calls: int = 4
    max_replans: int = 1

    def __post_init__(self):
        for name in ("max_planner_calls", "max_tool_calls"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(name + " must be a positive integer")
        if type(self.max_replans) is not int or self.max_replans not in (0, 1):
            raise ValueError("max_replans must be 0 or 1")
        if self.max_planner_calls < self.max_replans + 1:
            raise ValueError("planner budget cannot cover the allowed replan")


def _result_dict(result):
    return {
        "ok": result.ok,
        "data": deepcopy(result.data),
        "error_code": result.error_code,
        "retryable": result.retryable,
        "elapsed_ms": result.elapsed_ms,
        "provenance": result.provenance,
    }


def _execution_dict(report):
    return {
        "ok": report.ok,
        "failed_step_id": report.failed_step_id,
        "traces": [
            {
                "position": trace.position,
                "step_id": trace.step_id,
                "tool_name": trace.tool_name,
                "arguments": deepcopy(trace.arguments),
                "result": _result_dict(trace.result),
                "elapsed_ms": trace.elapsed_ms,
            }
            for trace in report.traces
        ],
    }


@dataclass(frozen=True)
class LoopReport:
    request_id: str
    route: Route
    route_reason: str
    system_mode: str
    status: str
    response: dict | None
    error_code: str | None
    planner_calls: int
    tool_calls: int
    replans: int
    remote_api_requests: int
    plans: tuple[dict, ...]
    executions: tuple
    budget: dict
    elapsed_ms: float

    def to_dict(self):
        return {
            "request_id": self.request_id,
            "route": self.route.value,
            "route_reason": self.route_reason,
            "system_mode": self.system_mode,
            "status": self.status,
            "response": deepcopy(self.response),
            "error_code": self.error_code,
            "planner_calls": self.planner_calls,
            "tool_calls": self.tool_calls,
            "replans": self.replans,
            "remote_api_requests": self.remote_api_requests,
            "plans": deepcopy(list(self.plans)),
            "executions": [_execution_dict(item) for item in self.executions],
            "budget": deepcopy(self.budget),
            "elapsed_ms": self.elapsed_ms,
        }


def _default_result_validator(value, routing_request):
    if (not isinstance(value, dict)
            or not isinstance(value.get("status"), str)
            or not isinstance(value.get("recommendations"), list)):
        return False
    recommendations = value["recommendations"]
    if value["status"] != "OK":
        return recommendations == []
    item_ids = []
    for row in recommendations:
        if (not isinstance(row, dict) or type(row.get("item_id")) is not int
                or row["item_id"] <= 0):
            return False
        item_ids.append(row["item_id"])
    if len(item_ids) != len(set(item_ids)):
        return False
    fixed = routing_request.fixed_request
    if fixed is None:
        return True
    constraints = fixed.constraints
    if len(recommendations) > constraints.k:
        return False
    blocked = set(constraints.excluded_item_ids)
    if constraints.exclude_seen:
        blocked.update(fixed.seen_item_ids)
    if blocked & set(item_ids):
        return False
    include = {genre.casefold() for genre in constraints.include_genres}
    exclude = {genre.casefold() for genre in constraints.exclude_genres}
    for row in recommendations:
        genres = row.get("genres")
        if include or exclude or "genres" in constraints.required_fields:
            if (not isinstance(genres, list)
                    or any(not isinstance(genre, str) or not genre for genre in genres)):
                return False
            normalized = {genre.casefold() for genre in genres}
            if not include <= normalized or exclude & normalized:
                return False
        year = row.get("year")
        if (constraints.year_min is not None or constraints.year_max is not None
                or "year" in constraints.required_fields):
            if type(year) is not int or year <= 0:
                return False
            if constraints.year_min is not None and year < constraints.year_min:
                return False
            if constraints.year_max is not None and year > constraints.year_max:
                return False
        if "title" in constraints.required_fields and (
                not isinstance(row.get("title"), str) or not row["title"].strip()):
            return False
    return True


class AgentLoop:
    PLAN_SCHEMA = StrictObjectSchema({"plan": list})

    def __init__(self, router, fixed_pipeline, planner, executor, *, limits=None,
                 planned_cost_ceiling=0, result_validator=None, clock=time.monotonic):
        if not isinstance(router, Router):
            raise TypeError("router must be Router")
        if not hasattr(fixed_pipeline, "recommend"):
            raise TypeError("fixed_pipeline must provide recommend")
        if not isinstance(planner, ChatAdapter):
            raise TypeError("planner must be ChatAdapter")
        if not isinstance(executor, PlanExecutor):
            raise TypeError("executor must be PlanExecutor")
        limits = limits or AgentLimits()
        if not isinstance(limits, AgentLimits):
            raise TypeError("limits must be AgentLimits")
        if (type(planned_cost_ceiling) not in (int, float)
                or not math.isfinite(planned_cost_ceiling)
                or planned_cost_ceiling < 0):
            raise ValueError("planned_cost_ceiling must be finite and nonnegative")
        if result_validator is not None and not callable(result_validator):
            raise TypeError("result_validator must be callable")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self.router = router
        self.fixed_pipeline = fixed_pipeline
        self.planner = planner
        self.executor = executor
        self.limits = limits
        self.planned_cost_ceiling = float(planned_cost_ceiling)
        self.result_validator = result_validator or _default_result_validator
        self._clock = clock

    def run(self, request):
        if not isinstance(request, RoutingRequest):
            raise TypeError("request must be RoutingRequest")
        started = self._clock()
        initial_budget = self.planner.ledger.snapshot()
        decision = self.router.decide(request)
        if decision.route in (Route.DIRECT, Route.PERSONALIZED):
            if request.fixed_request is None:
                return self._report(request, decision, "CLARIFY", None,
                                    "fixed_request_missing", 0, 0, 0, (), (),
                                    initial_budget, started)
            response = self.fixed_pipeline.recommend(request.fixed_request)
            payload = response.to_dict()
            return self._report(
                request, decision, payload.get("status", "OK"), payload, None,
                0, 0, 0, (), (), initial_budget, started,
            )
        if decision.route is Route.CLARIFY:
            return self._report(
                request, decision, "CLARIFY", None, decision.reason,
                0, 0, 0, (), (), initial_budget, started,
            )
        return self._run_agent(request, decision, initial_budget, started)

    def _run_agent(self, request, decision, initial_budget, started):
        planner_calls = tool_calls = replans = 0
        plans = []
        executions = []
        previous_failure = None
        deadline = RoundDeadline(
            self.planner.config.round_deadline_seconds, clock=self._clock
        )
        while True:
            if planner_calls >= self.limits.max_planner_calls:
                return self._report(
                    request, decision, "BUDGET_EXHAUSTED", None,
                    "planner_call_budget_exhausted", planner_calls, tool_calls,
                    replans, plans, executions, initial_budget, started,
                )
            messages = self._planner_messages(request, previous_failure)
            planner_calls += 1
            try:
                chat = self.planner.chat_json(
                    messages,
                    schema=self.PLAN_SCHEMA,
                    planned_cost_ceiling=self.planned_cost_ceiling,
                    deadline=deadline,
                )
                steps = parse_plan(chat.value["plan"])
            except (BudgetExceeded, DeadlineExceeded, LiveCallsDisabled):
                return self._report(
                    request, decision, "BUDGET_EXHAUSTED", None,
                    "planner_runtime_budget_exhausted", planner_calls, tool_calls,
                    replans, plans, executions, initial_budget, started,
                )
            except (ResponseValidationError, PlanValidationError):
                return self._report(
                    request, decision, "INVALID_PLAN", None,
                    "plan_validation_failed", planner_calls, tool_calls,
                    replans, plans, executions, initial_budget, started,
                )
            except LLMTransportError:
                return self._report(
                    request, decision, "PLANNER_FAILED", None,
                    "planner_transport_failed", planner_calls, tool_calls,
                    replans, plans, executions, initial_budget, started,
                )

            plans.append(deepcopy(chat.value))
            if tool_calls + len(steps) > self.limits.max_tool_calls:
                return self._report(
                    request, decision, "BUDGET_EXHAUSTED", None,
                    "tool_call_budget_exhausted", planner_calls, tool_calls,
                    replans, plans, executions, initial_budget, started,
                )
            try:
                execution = self.executor.execute(steps)
            except PlanValidationError:
                return self._report(
                    request, decision, "INVALID_PLAN", None,
                    "plan_validation_failed", planner_calls, tool_calls,
                    replans, plans, executions, initial_budget, started,
                )
            executions.append(execution)
            tool_calls += len(execution.traces)

            if execution.ok:
                final_data = execution.traces[-1].result.data
                if self.result_validator(final_data, request):
                    payload = deepcopy(final_data)
                    return self._report(
                        request, decision, payload.get("status", "OK"), payload,
                        None, planner_calls, tool_calls, replans, plans,
                        executions, initial_budget, started,
                    )
                failure_code = "INVALID_FINAL_RESULT"
                retryable = True
            else:
                failed = execution.traces[-1].result
                failure_code = failed.error_code
                retryable = failed.retryable

            if not retryable or replans >= self.limits.max_replans:
                status = "INVALID_RESULT" if failure_code == "INVALID_FINAL_RESULT" else "TOOL_FAILED"
                return self._report(
                    request, decision, status, None, failure_code,
                    planner_calls, tool_calls, replans, plans, executions,
                    initial_budget, started,
                )
            replans += 1
            previous_failure = {
                "failed_step_id": execution.failed_step_id,
                "error_code": failure_code,
                "tool_calls_used": tool_calls,
                "tool_calls_remaining": self.limits.max_tool_calls - tool_calls,
            }

    def _planner_messages(self, request, previous_failure):
        tools = json.dumps(
            self.executor.describe_tools(), sort_keys=True, separators=(",", ":")
        )
        contract = (
            "Return exactly one JSON object with key plan. plan must be a nonempty "
            "array of exact step objects: step_id, tool_name, arguments. Use only "
            "registered tools. Do not add prose. Registered tools: " + tools
        )
        visible = json.dumps(
            dict(request.public_input), ensure_ascii=True, sort_keys=True,
            separators=(",", ":"),
        )
        user = "Public request: " + visible
        if previous_failure is not None:
            user += ". Previous execution failed: " + json.dumps(
                previous_failure, sort_keys=True, separators=(",", ":")
            )
        return [
            {"role": "system", "content": contract},
            {"role": "user", "content": user},
        ]

    def _report(self, request, decision, status, response, error_code,
                planner_calls, tool_calls, replans, plans, executions,
                initial_budget, started):
        budget = self.planner.ledger.snapshot()
        remote = budget["live_attempts"] - initial_budget["live_attempts"]
        elapsed_ms = max(0.0, (self._clock() - started) * 1000.0)
        return LoopReport(
            request_id=request.request_id,
            route=decision.route,
            route_reason=decision.reason,
            system_mode=decision.system_mode.value,
            status=status,
            response=deepcopy(response),
            error_code=error_code,
            planner_calls=planner_calls,
            tool_calls=tool_calls,
            replans=replans,
            remote_api_requests=remote,
            plans=tuple(deepcopy(list(plans))),
            executions=tuple(executions),
            budget=budget,
            elapsed_ms=elapsed_ms,
        )
