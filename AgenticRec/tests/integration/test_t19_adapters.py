import json

from agenticrec.adapters.llm import ChatAdapter
from agenticrec.agent.router import RoutingRequest, fixed_request_payload
from agenticrec.config import LLMBudget
from agenticrec.evaluation.faults import FaultAwarePipeline, FaultBoundExecutor, FaultSchedule
from agenticrec.evaluation.upstream_turn import UpstreamRebuiltTurn
from agenticrec.evaluation.variants import build_pipeline_variant
from agenticrec.pipeline import FixedRecommendationPipeline, FixedRequest
from agenticrec.testing import FakeLLM


def test_metered_u1_executes_original_toolbox_on_real_frozen_candidates():
    pipeline = FixedRecommendationPipeline.from_frozen("lightgcn", 42)
    request = FixedRequest.parse({"schema_version": 1, "constraints": {
        "include_genres": ["Action"], "k": 3}})
    planner = ChatAdapter(FakeLLM([json.dumps({
        "request": fixed_request_payload(request),
        "plan": [{"tool_name": "Movie Candidates Ranking Tool", "input": '{"schema":"model_scores"}'},
                 {"tool_name": "Mapping Tool", "input": "3"}],
    })]), LLMBudget())
    upstream = UpstreamRebuiltTurn(planner, pipeline, planned_cost_ceiling=0)
    turn = upstream(RoutingRequest.from_fixed("test-001", request))
    assert turn.status == "OK" and len(turn.item_ids) == 3
    assert turn.tool_calls == 3
    assert upstream.last_trace["upstream"]["plan_source"] == "metered_llm_rebuilt_plan"
    assert upstream.last_trace["upstream"]["credential_env_present"] is False


def test_recommendation_ablations_remove_the_claimed_signals_preserve_train_seen():
    base = FixedRecommendationPipeline.from_frozen("lightgcn", 42)
    request = FixedRequest.parse({"schema_version": 1, "user_id": 1,
        "history_authorized": True, "constraints": {"include_genres": ["Action"], "k": 3}})
    for condition in ("no_user_model", "no_collaborative", "no_content"):
        pipeline = build_pipeline_variant(base, condition)
        response = pipeline.recommend(request)
        assert response.status == "OK" and len(response.recommendations) == 3
        assert not ({row.item_id for row in response.recommendations}
                    & set(base.retriever.known_user.train_seen[1]))
        sources = response.candidate_counts["per_source"]
        if condition == "no_content":
            assert "content" not in sources and "cf_model" in sources
        else:
            assert "cf_model" not in sources and "content" in sources
        if condition == "no_user_model":
            assert pipeline.router is not base.router
        elif condition == "no_collaborative":
            assert pipeline.router is base.router


def test_local_ranking_fault_uses_safe_content_fallback_and_preserves_exclusions():
    base = FixedRecommendationPipeline.from_frozen("lightgcn", 42)
    fallback = build_pipeline_variant(base, "no_user_model")
    pipeline = FaultAwarePipeline(base, fallback)
    request = FixedRequest.parse({"schema_version": 1, "constraints": {
        "include_genres": ["Action"], "excluded_item_ids": [1, 2], "k": 3}})
    with pipeline.activate("ranking_timeout"):
        reply = pipeline.recommend(request)
    assert reply.status == "OK" and reply.fallback_reason == "fixed_pipeline"
    assert pipeline.invocation_count == 2 and pipeline.fault_count == 1
    assert all(row.item_id not in (1, 2) and "Action" in row.genres for row in reply.recommendations)
    assert pipeline._active is None


def test_persistent_original_worker_resets_each_request_and_strips_credentials():
    from agenticrec.adapters.upstream_bridge import UpstreamWorkerSession, run_upstream_plan
    from agenticrec.adapters.model import RouteResult
    titles={1:'First',2:'Second',3:'Third'}
    with UpstreamWorkerSession() as session:
        for ranked in ([2,1],[3,2]):
            route=RouteResult([1,2,3],[0.1,0.2,0.3],ranked,'cold_start',None)
            value=run_upstream_plan(route,titles,top_k=2,session=session)
            assert value['mapped_ids']==ranked
            assert value['buffer_reset'] and not value['credential_env_present']
