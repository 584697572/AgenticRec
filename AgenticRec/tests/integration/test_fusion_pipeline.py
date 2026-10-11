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
