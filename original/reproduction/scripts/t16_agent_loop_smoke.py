"""T16 development-only routing calibration and replayable FakeLLM smoke."""

from collections import Counter
import json
from pathlib import Path

from agenticrec.adapters.llm import ChatAdapter
from agenticrec.agent.executor import PlanExecutor, ToolDefinition
from agenticrec.agent.loop import AgentLimits, AgentLoop
from agenticrec.agent.router import (
    CalibrationExample,
    Router,
    RouterPolicy,
    RoutingRequest,
    select_agent_threshold,
)
from agenticrec.config import LLMBudget
from agenticrec.data import digest
from agenticrec.pipeline import FixedRequest
from agenticrec.protocols import ToolResult
from agenticrec.testing import FakeLLM


ROOT = Path(__file__).resolve().parents[2]
DEVELOPMENT_PUBLIC = ROOT / "data/eval_private/interactive-v1/development.public.jsonl"
TRACE = ROOT / "artifacts/agent/t16_fake_trajectories_20261007.json"


class _UnusedPipeline:
    def recommend(self, _request):
        raise AssertionError("Fake agent trajectory unexpectedly entered fixed pipeline")


def _plan(query, *, two_steps=False):
    steps = [{
        "step_id": "s1", "tool_name": "recommend", "arguments": {"query": query}
    }]
    if two_steps:
        steps.append({
            "step_id": "s2", "tool_name": "recommend", "arguments": {"query": query + "-2"}
        })
    return json.dumps({"plan": steps}, sort_keys=True)


def _request(request_id="fixture", **overrides):
    values = {
        "request_id": request_id,
        "public_input": {"mode": "text", "message": "recommend an action movie"},
        "agent_need_score": 1.0,
    }
    values.update(overrides)
    return RoutingRequest(**values)


def _run_fixture(responses, invoke, request=None, limits=None):
    fake = FakeLLM(responses)
    loop = AgentLoop(
        Router(RouterPolicy(agent_threshold=0.75)),
        _UnusedPipeline(),
        ChatAdapter(fake, LLMBudget()),
        PlanExecutor([ToolDefinition("recommend", {"query": str}, invoke)]),
        limits=limits,
        planned_cost_ceiling=0,
    )
    return loop.run(request or _request())


def _load_development():
    if DEVELOPMENT_PUBLIC.name != "development.public.jsonl":
        raise AssertionError("Only the frozen development public file may be loaded")
    rows = [json.loads(line) for line in DEVELOPMENT_PUBLIC.read_text(
        encoding="utf-8"
    ).splitlines() if line]
    if not rows or any(row.get("split") != "development" for row in rows):
        raise AssertionError("Routing calibration received a non-development episode")
    return rows


def _calibrate(rows):
    examples = [
        CalibrationExample(1.0 if row["layer"] == "text" else 0.0,
                           row["layer"] == "text")
        for row in rows
    ]
    return select_agent_threshold(
        examples, candidates=(0.25, 0.5, 0.75), split="development"
    )


def _route_counts(rows, threshold):
    router = Router(RouterPolicy(agent_threshold=threshold))
    counts = Counter()
    for row in rows:
        initial = row["initial_input"]
        if row["layer"] == "structured":
            request = RoutingRequest.from_fixed(
                row["episode_id"], FixedRequest.parse(initial)
            )
        else:
            request = RoutingRequest(
                row["episode_id"], initial, agent_need_score=1.0
            )
        counts[router.decide(request).route.value] += 1
    return dict(sorted(counts.items()))


def main():
    rows = _load_development()
    selected = _calibrate(rows)
    route_counts = _route_counts(rows, selected.threshold)

    success = _run_fixture(
        [_plan("success")],
        lambda _: ToolResult(True, data={"status": "OK", "recommendations": []},
                             provenance="fixture"),
        _request("success"),
    )
    retry_calls = []

    def retry_once(arguments):
        retry_calls.append(dict(arguments))
        if len(retry_calls) == 1:
            return ToolResult(False, error_code="TEMPORARY", retryable=True,
                              provenance="fixture")
        return ToolResult(True, data={"status": "OK", "recommendations": []},
                          provenance="fixture")

    replan = _run_fixture(
        [_plan("first"), _plan("second")], retry_once, _request("replan")
    )
    clarify = _run_fixture(
        [],
        lambda _: ToolResult(True, data={}),
        _request("clarify", agent_need_score=0.1, unresolved_fields=("year",)),
    )
    exhausted = _run_fixture(
        [_plan("oversized", two_steps=True)],
        lambda _: ToolResult(True, data={"status": "OK", "recommendations": []}),
        _request("budget"),
        AgentLimits(max_planner_calls=2, max_tool_calls=1, max_replans=1),
    )
    trajectories = {
        "success": success.to_dict(),
        "replan": replan.to_dict(),
        "clarify": clarify.to_dict(),
        "budget_exhausted": exhausted.to_dict(),
    }
    expected = {
        "success": ("OK", 1, 1, 0),
        "replan": ("OK", 2, 2, 1),
        "clarify": ("CLARIFY", 0, 0, 0),
        "budget_exhausted": ("BUDGET_EXHAUSTED", 1, 0, 0),
    }
    for name, report in trajectories.items():
        actual = (report["status"], report["planner_calls"],
                  report["tool_calls"], report["replans"])
        if actual != expected[name] or report["remote_api_requests"] != 0:
            raise AssertionError(name + " trajectory violated its bounded contract")

    TRACE.parent.mkdir(parents=True, exist_ok=True)
    TRACE.write_text(json.dumps({
        "schema_version": 1,
        "task": "T16",
        "threshold_selection": {
            "split": selected.split,
            "threshold": selected.threshold,
            "correct": selected.correct,
            "total": selected.total,
        },
        "development_route_counts": route_counts,
        "trajectories": trajectories,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "status": "PASS",
        "task": "T16",
        "development_public_rows": len(rows),
        "development_public_sha256": digest(DEVELOPMENT_PUBLIC),
        "test_files_loaded": 0,
        "threshold": selected.threshold,
        "threshold_selection_correct": selected.correct,
        "threshold_selection_total": selected.total,
        "development_route_counts": route_counts,
        "trajectory_statuses": {
            name: report["status"] for name, report in trajectories.items()
        },
        "fixture_llm_attempts": sum(
            report["budget"]["fixture_attempts"] for report in trajectories.values()
        ),
        "remote_api_requests": 0,
        "trace_path": TRACE.relative_to(ROOT).as_posix(),
        "trace_sha256": digest(TRACE),
        "system_effect": "NOT EVALUATED",
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
