"""Self-authored six-item catalog: engineering demo, never quality evidence."""
from dataclasses import replace
import json

from .adapters.llm import ChatAdapter
from .adapters.model import RoutingScorer, SessionSeedScorer
from .agent.router import RoutingRequest, fixed_request_payload
from .agent.state import PreferencePatch, PreferenceState
from .config import LLMBudget
from .evaluation.system_factory import build_benchmark_agent_loop
from .pipeline import FixedRecommendationPipeline, FixedRequest
from .ranking.constraints import Catalog, Constraints, Item
from .retrieval.sources import CandidateRetriever
from .testing import FakeLLM


class DemoScorer:
    """Deterministic toy scores, explicitly not trained model weights."""
    n_users = 1
    n_items = 6
    train_seen = {1: {1}}

    def score(self, user_id, item_ids):
        if user_id != 1:
            raise ValueError("unknown demo user")
        return [float(item_id) for item_id in item_ids]


def build_demo_pipeline():
    catalog = Catalog([
        Item(1, "Demo Action Seed", frozenset({"Action"}), 1995),
        Item(2, "Demo Horror", frozenset({"Horror"}), 1996),
        Item(3, "Demo Action Horror", frozenset({"Action", "Horror"}), 1997),
        Item(4, "Demo Future Action", frozenset({"Action"}), 2001),
        Item(5, "Demo Drama", frozenset({"Drama"}), 1999),
        Item(6, "Demo Action Match", frozenset({"Action"}), 1998),
    ])
    edges = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 4), (2, 6),
             (3, 1), (3, 5), (3, 6)]
    known = DemoScorer()
    session = SessionSeedScorer(
        {i: item.genres for i, item in catalog.items.items()},
        {1: 3, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2}, 6,
    )
    return FixedRecommendationPipeline(
        CandidateRetriever(catalog, edges, known), RoutingScorer(known, session),
    )


def run_demo():
    pipeline = build_demo_pipeline()
    base = FixedRequest(user_id=1, history_authorized=True, constraints=Constraints(
        include_genres=("Action",), exclude_genres=("Horror",),
        year_min=1990, year_max=2005, k=2,
    ))
    state = PreferenceState.create(session_id="demo-feedback")
    updated = state.apply(PreferencePatch(
        event_id="demo-turn-2", source_turn=2, source="explicit_user",
        hard_updates={"year_max": 2000},
    )).state
    requests = {
        "known_user": base,
        "anonymous": replace(base, user_id=None, history_authorized=False, liked_item_ids=(1,)),
        "exclusion": replace(base, constraints=replace(base.constraints, excluded_item_ids=(6,))),
        "feedback": replace(base, constraints=replace(base.constraints, year_max=updated.hard_constraints.year_max)),
        "no_solution": replace(base, constraints=replace(base.constraints, year_min=2030, year_max=2040)),
    }
    cases = {}
    for name, request in requests.items():
        response = pipeline.recommend(request)
        cases[name] = {"status": response.status, "profile_source": response.profile_source,
                       "item_ids": [row.item_id for row in response.recommendations],
                       "response": response.to_dict()}
    payload = fixed_request_payload(base)
    # Two identical tool names with distinct step IDs must both execute.
    plan = json.dumps({"plan": [
        {"step_id": name, "tool_name": "recommend", "arguments": {"request": payload}}
        for name in ("first", "second")
    ]})
    routing = RoutingRequest.from_fixed("demo-routing", base)
    results = {}
    for system, replies in (("A", [plan]), ("O", [])):
        loop = build_benchmark_agent_loop(
            system, fixed_pipeline=pipeline,
            planner=ChatAdapter(FakeLLM(replies), LLMBudget()), planned_cost_ceiling=0,
        )
        results[system] = loop.run(routing)
    agent = {system: result.to_dict() for system, result in results.items()}
    agent["same_response"] = results["A"].response == results["O"].response
    if (cases["known_user"]["item_ids"] != [6, 4]
            or cases["no_solution"]["status"] != "NO_FEASIBLE_ITEMS"
            or results["A"].status != "OK" or results["A"].tool_calls != 2
            or results["O"].status != "OK" or not agent["same_response"]):
        raise RuntimeError("offline recommendation demo failed")
    return {"status": "PASS", "scope": "self_authored_offline_recommendation_demo",
            "is_fixture": True, "remote_api_requests": 0,
            "model": "deterministic_toy_scorer_not_trained_BPR_or_LightGCN",
            "metrics": "NOT EVALUATED", "cases": cases, "agent": agent,
            "feedback_state_version": updated.version}
