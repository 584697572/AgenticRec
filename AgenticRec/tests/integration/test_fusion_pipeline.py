"""New pipeline keeps final hard gates and works as an Agent dependency."""
from agenticrec.fusion_pipeline import FusionPipeline
from agenticrec.pipeline import FixedRequest
from agenticrec.ranking.constraints import Constraints, Catalog, Item
from agenticrec.retrieval.sources import CandidateRetriever


def retriever():
    catalog=Catalog([Item(i,str(i),frozenset({'Drama'} if i==4 else {'Comedy'}),1995)
                     for i in range(1,7)])
    return CandidateRetriever(catalog,[(1,1),(1,2),(1,3),(2,1),(2,3),(2,5),
                                       (3,2),(3,4),(3,6)])


def test_explicit_local_denial_and_final_hard_gate_with_diagnostics():
    pipeline=FusionPipeline(retriever(),mode='conditional')
    request=FixedRequest(Constraints(include_genres=('Comedy',),k=5),
                         liked_item_ids=(1,),disliked_item_ids=(2,))
    response=pipeline.recommend(request)
    ids=[r.item_id for r in response.recommendations]
    assert ids and not set(ids)&{1,2,4}
    assert response.remote_api_requests==0 and response.llm_requests==0
    trace=pipeline.diagnose(request)
    assert {r['item_id'] for r in trace['candidate_rows'] if r['final']}==set(ids)
    assert len(trace['candidate_rows'])>=len(ids)


def test_conflict_missing_field_and_invalid_id_preserve_old_contract():
    pipeline=FusionPipeline(retriever())
    assert pipeline.recommend(FixedRequest(Constraints(include_genres=('Comedy',),exclude_genres=('Comedy',)))).status=='CLARIFY_CONFLICT'
    assert pipeline.recommend(FixedRequest(Constraints(required_fields=('duration',)))).status=='MISSING_FIELD'
    assert pipeline.recommend(FixedRequest(Constraints(),liked_item_ids=(999,))).status=='INVALID_ITEM_ID'


def test_inherited_frozen_factory_constructs_compatible_pipeline(monkeypatch):
    from agenticrec.adapters.model import KnownUserScorer
    from agenticrec.retrieval.adaptive import CachedResearchRetriever
    monkeypatch.setattr(KnownUserScorer,'from_artifacts',lambda *_:None)
    monkeypatch.setattr(CachedResearchRetriever,'from_frozen',lambda *_:retriever())
    pipeline=FusionPipeline.from_frozen('bpr',42,mode='rrf')
    assert pipeline.mode=='rrf'
    assert pipeline.recommend(FixedRequest(Constraints(k=2),liked_item_ids=(1,))).status=='OK'


def test_existing_agent_executor_persists_local_feedback_and_does_not_leak_sessions():
    from agenticrec.adapters.llm import ChatAdapter
    from agenticrec.config import LLMBudget
    from agenticrec.evaluation.episodes import PublicEpisode
    from agenticrec.evaluation.system_executor import BenchmarkSystemExecutor
    from agenticrec.evaluation.system_factory import build_benchmark_agent_loop
    from agenticrec.agent.router import fixed_request_payload
    from agenticrec.testing import FakeLLM

    class RecordingFusion(FusionPipeline):
        def recommend(self,request):
            response=super().recommend(request)
            self.calls.append((request,response))
            return response
    pipeline=RecordingFusion(retriever(),mode='rrf'); pipeline.calls=[]
    fixed=FixedRequest(Constraints(include_genres=('Comedy',),k=1),liked_item_ids=(1,))
    episode=PublicEpisode('development-001','development','structured','user:new','template:new',
                          'exclusion_feedback',True,fixed_request_payload(fixed),3)
    transport=FakeLLM([]); planner=ChatAdapter(transport,LLMBudget())
    loop=build_benchmark_agent_loop('O',fixed_pipeline=pipeline,planner=planner,planned_cost_ceiling=0)
    def feedback(_id,turn):
        assert turn==2 and len(pipeline.calls)==1
        denied=pipeline.calls[0][1].recommendations[0].item_id
        return {'turn':2,'kind':'explicit_feedback','patch':{'disliked_item_ids':[denied]}}
    attempt=BenchmarkSystemExecutor('O',fixed_pipeline=pipeline,agent_loop=loop,
              ledger=planner.ledger,feedback_source=feedback)(episode)
    assert attempt.status=='OK' and attempt.turns==3 and attempt.request_count==0
    assert not transport.calls and len(pipeline.calls)==2
    denied=pipeline.calls[0][1].recommendations[0].item_id
    assert denied not in attempt.item_ids
    assert pipeline.calls[1][0].disliked_item_ids==(denied,)
    assert pipeline.calls[1][0].constraints==fixed.constraints
    assert all('Comedy' in row.genres for row in pipeline.calls[1][1].recommendations)
    # Sharing immutable train statistics must not retain session feedback.
    assert pipeline.recommend(fixed).to_dict()==pipeline.calls[0][1].to_dict()


def test_existing_strict_agent_tool_executes_fusion_and_preserves_authorization():
    import json
    from agenticrec.adapters.llm import ChatAdapter
    from agenticrec.agent.router import RoutingRequest,fixed_request_payload
    from agenticrec.config import LLMBudget
    from agenticrec.evaluation.system_factory import build_benchmark_agent_loop
    from agenticrec.testing import FakeLLM
    pipeline=FusionPipeline(retriever(),mode='rrf')
    request=FixedRequest(Constraints(include_genres=('Comedy',),k=2),liked_item_ids=(1,))
    plan={'plan':[{'step_id':'s1','tool_name':'recommend',
                    'arguments':{'request':fixed_request_payload(request)}}]}
    transport=FakeLLM([json.dumps(plan)])
    loop=build_benchmark_agent_loop('A',fixed_pipeline=pipeline,
          planner=ChatAdapter(transport,LLMBudget()),planned_cost_ceiling=0)
    response=loop.run(RoutingRequest.from_fixed('fusion-tool',request))
    assert response.status=='OK' and response.remote_api_requests==0
    assert response.planner_calls==response.tool_calls==1
    assert response.response==pipeline.recommend(request).to_dict()
    changed=fixed_request_payload(request); changed['history_authorized']=True; changed['user_id']=1
    plan['plan'][0]['arguments']['request']=changed
    bad_loop=build_benchmark_agent_loop('A',fixed_pipeline=pipeline,
            planner=ChatAdapter(FakeLLM([json.dumps(plan)]),LLMBudget()),planned_cost_ceiling=0)
    invalid=bad_loop.run(RoutingRequest.from_fixed('no-auth',request))
    assert invalid.status=='INVALID_PLAN' and invalid.tool_calls==0
