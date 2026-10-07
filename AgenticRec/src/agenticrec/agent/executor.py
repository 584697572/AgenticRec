"""Ordered, exact-whitelist execution for structured tool plans."""

from copy import deepcopy
from dataclasses import dataclass
import json
import math
import time
from types import MappingProxyType
from typing import Any, Callable, Mapping

from ..protocols import ToolResult
from ..runtime.errors import SyncTaskTimeout
from ..runtime.sync import SyncTaskRunner


class PlanValidationError(ValueError):
    """The entire plan is rejected before any tool runs."""


def _required_text(value, label):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise PlanValidationError(label + " must be a nonempty trimmed string")
    return value


@dataclass(frozen=True)
class PlanStep:
    step_id: str
    tool_name: str
    arguments: Mapping[str, Any]

    def __post_init__(self):
        object.__setattr__(self, "step_id", _required_text(self.step_id, "step_id"))
        object.__setattr__(self, "tool_name", _required_text(self.tool_name, "tool_name"))
        if not isinstance(self.arguments, dict):
            raise PlanValidationError("arguments must be an object")
        object.__setattr__(self, "arguments", MappingProxyType(deepcopy(self.arguments)))


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    parameters: Mapping[str, type]
    invoke: Callable[[Mapping[str, Any]], ToolResult]

    def __post_init__(self):
        object.__setattr__(self, "name", _required_text(self.name, "tool name"))
        if not isinstance(self.parameters, dict):
            raise ValueError("tool parameters must be a mapping")
        normalized = {}
        for name, expected in self.parameters.items():
            if not isinstance(name, str) or not name or name != name.strip():
                raise ValueError("parameter names must be nonempty trimmed strings")
            if not isinstance(expected, type):
                raise ValueError("parameter schemas must contain Python types")
            normalized[name] = expected
        if not callable(self.invoke):
            raise ValueError("tool invoke must be callable")
        object.__setattr__(self, "parameters", MappingProxyType(normalized))

    def validate_arguments(self, arguments):
        names = set(arguments)
        expected_names = set(self.parameters)
        missing = sorted(expected_names - names)
        if missing:
            raise PlanValidationError(
                f"tool {self.name} missing parameters: " + ", ".join(missing)
            )
        extra = sorted(names - expected_names)
        if extra:
            raise PlanValidationError(
                f"tool {self.name} has unknown parameters: " + ", ".join(extra)
            )
        for name, expected in self.parameters.items():
            value = arguments[name]
            valid = isinstance(value, expected)
            if expected in (int, float) and isinstance(value, bool):
                valid = False
            if expected is float:
                valid = isinstance(value, (int, float)) and not isinstance(value, bool)
                valid = valid and math.isfinite(value)
            if not valid:
                raise PlanValidationError(
                    f"tool {self.name} parameter {name} must be {expected.__name__}"
                )


@dataclass(frozen=True)
class StepTrace:
    position: int
    step_id: str
    tool_name: str
    arguments: dict
    result: ToolResult
    elapsed_ms: float


@dataclass(frozen=True)
class ExecutionReport:
    ok: bool
    traces: tuple[StepTrace, ...]
    failed_step_id: str | None = None

    def __post_init__(self):
        if type(self.ok) is not bool:
            raise ValueError("ok must be boolean")
        if self.ok and self.failed_step_id is not None:
            raise ValueError("successful execution cannot have a failed step")
        if not self.ok and not self.failed_step_id:
            raise ValueError("failed execution requires a failed step")


def parse_plan(payload):
    """Parse an exact-key JSON/list plan without collapsing repeated tools."""
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as error:
            raise PlanValidationError("plan is not valid JSON") from error
    if not isinstance(payload, list) or not payload:
        raise PlanValidationError("plan must be a nonempty list")
    steps = []
    expected_keys = {"step_id", "tool_name", "arguments"}
    for position, raw in enumerate(payload, start=1):
        if not isinstance(raw, dict):
            raise PlanValidationError(f"step {position} must be an object")
        if set(raw) != expected_keys:
            raise PlanValidationError(
                f"step {position} requires exactly step_id, tool_name and arguments"
            )
        steps.append(PlanStep(raw["step_id"], raw["tool_name"], raw["arguments"]))
    return tuple(steps)


class PlanExecutor:
    def __init__(self, tools, *, clock=time.monotonic, sync_runner=None,
                 tool_timeout_seconds=None):
        tools = tuple(tools)
        if not tools or not all(isinstance(tool, ToolDefinition) for tool in tools):
            raise ValueError("at least one ToolDefinition is required")
        if len({tool.name for tool in tools}) != len(tools):
            raise ValueError("tool registry contains duplicate names")
        if not callable(clock):
            raise ValueError("clock must be callable")
        if (sync_runner is None) != (tool_timeout_seconds is None):
            raise ValueError("sync_runner and tool_timeout_seconds must be set together")
        if sync_runner is not None and not isinstance(sync_runner, SyncTaskRunner):
            raise TypeError("sync_runner must be SyncTaskRunner")
        if tool_timeout_seconds is not None and (
                type(tool_timeout_seconds) not in (int, float)
                or not math.isfinite(tool_timeout_seconds)
                or tool_timeout_seconds <= 0):
            raise ValueError("tool_timeout_seconds must be finite and positive")
        self._tools = {tool.name: tool for tool in tools}
        self._clock = clock
        self._sync_runner = sync_runner
        self._tool_timeout_seconds = (
            None if tool_timeout_seconds is None else float(tool_timeout_seconds)
        )

    def describe_tools(self):
        """Return the exact public whitelist without exposing invoke callables."""
        return tuple(
            {
                "name": name,
                "parameters": {
                    parameter: expected.__name__
                    for parameter, expected in tool.parameters.items()
                },
            }
            for name, tool in sorted(self._tools.items())
        )

    def execute(self, steps):
        steps = self._preflight(steps)
        traces = []
        for position, step in enumerate(steps, start=1):
            tool = self._tools[step.tool_name]
            original_arguments = deepcopy(dict(step.arguments))
            invocation_arguments = deepcopy(original_arguments)
            started = self._clock()
            try:
                if self._sync_runner is None:
                    result = tool.invoke(invocation_arguments)
                else:
                    result = self._sync_runner.run(
                        tool.invoke,
                        invocation_arguments,
                        timeout_seconds=self._tool_timeout_seconds,
                    )
                if not isinstance(result, ToolResult):
                    result = ToolResult(
                        False,
                        error_code="INVALID_TOOL_RESULT",
                        provenance="executor",
                    )
            except SyncTaskTimeout as error:
                result = ToolResult(
                    False,
                    error_code="TOOL_TIMEOUT",
                    retryable=False,
                    provenance=(
                        "executor:timeout_work_may_continue"
                        if error.work_may_continue
                        else "executor:timeout_cancelled_before_start"
                    ),
                )
            except Exception as error:
                result = ToolResult(
                    False,
                    error_code="TOOL_EXCEPTION",
                    provenance="executor:" + type(error).__name__,
                )
            elapsed_ms = max(0.0, (self._clock() - started) * 1000.0)
            trace = StepTrace(
                position=position,
                step_id=step.step_id,
                tool_name=step.tool_name,
                arguments=original_arguments,
                result=result,
                elapsed_ms=elapsed_ms,
            )
            traces.append(trace)
            if not result.ok:
                return ExecutionReport(False, tuple(traces), step.step_id)
        return ExecutionReport(True, tuple(traces))

    def _preflight(self, steps):
        if not isinstance(steps, (list, tuple)) or not steps:
            raise PlanValidationError("steps must be a nonempty sequence")
        if not all(isinstance(step, PlanStep) for step in steps):
            raise PlanValidationError("every step must be PlanStep")
        step_ids = [step.step_id for step in steps]
        if len(step_ids) != len(set(step_ids)):
            raise PlanValidationError("duplicate step_id")
        for step in steps:
            tool = self._tools.get(step.tool_name)
            if tool is None:
                raise PlanValidationError("unknown tool: " + step.tool_name)
            tool.validate_arguments(step.arguments)
        return tuple(steps)
