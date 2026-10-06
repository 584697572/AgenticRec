import json

import pytest

from agenticrec.agent.executor import (
    PlanExecutor,
    PlanValidationError,
    ToolDefinition,
    parse_plan,
)
from agenticrec.protocols import ToolResult


def recording_tool(name, schema, events, *, fail_on=None):
    def invoke(arguments):
        events.append((name, dict(arguments)))
        if fail_on is not None and arguments == fail_on:
            return ToolResult(False, error_code="FIXTURE_FAILURE", retryable=False)
        return ToolResult(True, data={"received": dict(arguments)}, provenance="fixture")

    return ToolDefinition(name=name, parameters=schema, invoke=invoke)


def test_same_tool_runs_twice_in_original_order_with_original_arguments():
    events = []
    executor = PlanExecutor([
        recording_tool("lookup", {"query": str}, events),
        recording_tool("map", {"limit": int}, events),
    ])
    steps = parse_plan(json.dumps([
        {"step_id": "s1", "tool_name": "lookup", "arguments": {"query": "first-query"}},
        {"step_id": "s2", "tool_name": "lookup", "arguments": {"query": "second-query"}},
        {"step_id": "s3", "tool_name": "map", "arguments": {"limit": 5}},
    ]))

    report = executor.execute(steps)

    assert report.ok is True
    assert events == [
        ("lookup", {"query": "first-query"}),
        ("lookup", {"query": "second-query"}),
        ("map", {"limit": 5}),
    ]
    assert [trace.step_id for trace in report.traces] == ["s1", "s2", "s3"]
    assert [trace.tool_name for trace in report.traces] == ["lookup", "lookup", "map"]
    assert [trace.arguments for trace in report.traces] == [
        {"query": "first-query"}, {"query": "second-query"}, {"limit": 5}
    ]


@pytest.mark.parametrize(
    "payload,match",
    [
        ([{"step_id": "s1", "tool_name": "missing", "arguments": {}}], "unknown tool"),
        ([
            {"step_id": "same", "tool_name": "lookup", "arguments": {"query": "a"}},
            {"step_id": "same", "tool_name": "lookup", "arguments": {"query": "b"}},
        ], "duplicate step_id"),
        ([{"step_id": "s1", "tool_name": "lookup", "arguments": {}}], "missing parameters"),
        ([{"step_id": "s1", "tool_name": "lookup", "arguments": {"query": "a", "x": 1}}],
         "unknown parameters"),
        ([{"step_id": "s1", "tool_name": "lookup", "arguments": {"query": 3}}],
         "parameter query must be str"),
    ],
)
def test_invalid_plan_is_rejected_before_any_tool_executes(payload, match):
    events = []
    executor = PlanExecutor([recording_tool("lookup", {"query": str}, events)])

    with pytest.raises(PlanValidationError, match=match):
        executor.execute(parse_plan(payload))

    assert events == []


def test_parser_rejects_bad_json_non_list_and_non_exact_step_schema():
    for payload in (
        "not-json",
        '{"step_id":"s1"}',
        [{"step_id": "s1", "tool_name": "lookup"}],
        [{"step_id": "s1", "tool_name": "lookup", "arguments": {}, "extra": True}],
    ):
        with pytest.raises(PlanValidationError):
            parse_plan(payload)


def test_later_invalid_step_is_rejected_before_an_earlier_valid_step_runs():
    events = []
    executor = PlanExecutor([recording_tool("lookup", {"query": str}, events)])
    steps = parse_plan([
        {"step_id": "s1", "tool_name": "lookup", "arguments": {"query": "valid"}},
        {"step_id": "s2", "tool_name": "lookup", "arguments": {"query": 2}},
    ])

    with pytest.raises(PlanValidationError, match="parameter query must be str"):
        executor.execute(steps)

    assert events == []


def test_tool_failure_stops_later_steps_and_retains_failure_trace():
    events = []
    failing = {"query": "stop"}
    executor = PlanExecutor([
        recording_tool("lookup", {"query": str}, events, fail_on=failing),
        recording_tool("map", {"limit": int}, events),
    ])
    steps = parse_plan([
        {"step_id": "s1", "tool_name": "lookup", "arguments": failing},
        {"step_id": "s2", "tool_name": "map", "arguments": {"limit": 5}},
    ])

    report = executor.execute(steps)

    assert report.ok is False
    assert report.failed_step_id == "s1"
    assert len(report.traces) == 1
    assert report.traces[0].result.error_code == "FIXTURE_FAILURE"
    assert events == [("lookup", failing)]


def test_exact_whitelist_does_not_accept_substring_tool_names():
    events = []
    executor = PlanExecutor([recording_tool("lookup", {"query": str}, events)])
    steps = parse_plan([
        {"step_id": "s1", "tool_name": "prefix lookup suffix",
         "arguments": {"query": "fixture"}},
    ])

    with pytest.raises(PlanValidationError, match="unknown tool"):
        executor.execute(steps)

    assert events == []


def test_tool_exception_becomes_a_stopping_trace_without_leaking_message():
    def explode(arguments):
        raise RuntimeError("private fixture detail")

    executor = PlanExecutor([
        ToolDefinition("explode", {"value": int}, explode),
    ])
    report = executor.execute(parse_plan([
        {"step_id": "s1", "tool_name": "explode", "arguments": {"value": 1}},
    ]))

    assert report.ok is False
    assert report.failed_step_id == "s1"
    assert report.traces[0].result.error_code == "TOOL_EXCEPTION"
    assert report.traces[0].result.provenance == "executor:RuntimeError"
    assert "private fixture detail" not in repr(report)
