"""Metered rebuilt Plan -> original ToolBox/Buffer/Map benchmark adapter.

This is U1, not the paper's original prompt/resource reproduction. The prompt
adapts upstream tool-name/input plans to the complete mapped FixedRequest schema.
No extra LLM summarization is used in this ID-output benchmark.
"""

import json
import subprocess

from ..adapters.llm import ChatAdapter, StrictObjectSchema
from ..adapters.model import RouteResult
from ..adapters.upstream_bridge import run_upstream_plan
from ..agent.executor import PlanStep, PlanValidationError
from .system_factory import PLANNER_INSTRUCTIONS, _validate_benchmark_plan
from .system_executor import TurnResult
from ..pipeline import FixedRequest
from ..runtime.errors import ResponseValidationError


class UpstreamRebuiltTurn:
    SCHEMA = StrictObjectSchema({"request": dict, "plan": list})

    def __init__(self, adapter, pipeline, *, planned_cost_ceiling, bridge=run_upstream_plan):
        if not isinstance(adapter, ChatAdapter):
            raise TypeError("adapter must be ChatAdapter")
        self.adapter, self.pipeline, self.bridge = adapter, pipeline, bridge
        self.planned_cost_ceiling = planned_cost_ceiling
        self.last_request = None
        self.last_trace = None

    def __call__(self, routing_request):
        self.last_request = None
        self.last_trace = None
        prompt = (
            "Use the InteRecAgent rebuilt Plan First design. Return JSON with exactly "
            "request (complete FixedRequest) and plan (list of tool_name,input objects). "
            "Plan must start with Movie Candidates Ranking Tool input as the literal "
            "JSON string {\"schema\":\"model_scores\"}, and end with Mapping Tool input "
            "as a string equal to request.constraints.k. Use only these exact tool "
            "names, at most 4 steps. Model scores are computed from the validated "
            "request in the frozen mapped MovieLens space. " + PLANNER_INSTRUCTIONS
        )
        chat = self.adapter.chat_json(
            [{"role": "system", "content": prompt},
             {"role": "user", "content": "Public request: " + json.dumps(
                 dict(routing_request.public_input), sort_keys=True, separators=(",", ":"))}],
            schema=self.SCHEMA, planned_cost_ceiling=self.planned_cost_ceiling,
        )
        try:
            step = PlanStep("validate", "recommend", {"request": chat.value["request"]})
            _validate_benchmark_plan((step,), routing_request)
            request = FixedRequest.parse(chat.value["request"])
            plan = chat.value["plan"]
            expected_ranking = {"tool_name": "Movie Candidates Ranking Tool",
                                "input": '{"schema":"model_scores"}'}
            expected_map = {"tool_name": "Mapping Tool", "input": str(request.constraints.k)}
            if (not 1 <= request.constraints.k <= 20 or not 2 <= len(plan) <= 4
                    or plan[0] != expected_ranking or plan[-1] != expected_map
                    or any(step not in (expected_ranking, expected_map) for step in plan)):
                raise ValueError("upstream plan contract invalid")
        except (ValueError, TypeError, PlanValidationError):
            return TurnResult("INVALID_PLAN", (), 0)
        self.last_request = request
        response = self.pipeline.recommend(request).to_dict()
        self.last_trace = {"request": chat.value["request"], "plan": plan,
                           "response": response}
        if response["status"] != "OK":
            return TurnResult(response["status"], (), 1, response.get("fallback_reason"))
        catalog = self.pipeline.retriever.catalog
        candidates = list(range(1, catalog.n_items + 1))
        selected = [row["item_id"] for row in response["recommendations"]]
        scores = {row["item_id"]: row["personalized_score"] for row in response["recommendations"]}
        route = RouteResult(candidates, [scores.get(i, 0.0) for i in candidates], selected,
                            response["profile_source"], response.get("fallback_reason"))
        try:
            upstream = self.bridge(route, {i: catalog.items[i].title for i in candidates},
                                   top_k=request.constraints.k, tool_plan=plan)
        except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired):
            return TurnResult("TOOL_FAILED", (), 1)
        self.last_trace["upstream"] = upstream
        return TurnResult("OK", tuple(upstream["mapped_ids"]),
                          1 + len(upstream["tracker_tools"]), response.get("fallback_reason"))
