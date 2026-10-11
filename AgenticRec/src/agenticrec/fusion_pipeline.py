"""Experimental fusion pipeline; stable CLI/default model remains unchanged."""
from .adapters.model import RoutingScorer, SessionSeedScorer
from .pipeline import FixedRecommendationPipeline, FixedResponse, RecommendationEvidence
from .retrieval.adaptive import (source_snapshot, gate_features, fusion_weights,
                                weighted_fusion, candidate_diagnostics)


def predict_snapshot(retriever,request,snapshot,*,mode,gate=None,content_weight=.5,
                     protect_cf=10,no_quality=False):
    features=gate_features(retriever,request,snapshot)
    if no_quality: features=features[:4]+[0.0]*5
    weights=fusion_weights(mode,snapshot,features,gate=gate,content_weight=content_weight)
    fused=weighted_fusion(snapshot['rankings'],weights,universe=snapshot['eligible_ids'],
                          per_route=snapshot['per_route'],union_cap=200,protect_cf=protect_cf)
    if snapshot['profile_source']=='known_user':
        scores=dict(zip(fused.candidate_ids,retriever.known_user.score(request.user_id,fused.candidate_ids)))
        ranked=sorted(fused.candidate_ids,key=lambda i:(-scores[i],-fused.rrf_scores[i],i))
    else:
        scores=fused.rrf_scores
        ranked=fused.candidate_ids
    final=retriever.gate.finalize(ranked,snapshot['effective_constraints'],snapshot['seen_ids'])
    return final,fused,scores,{'weights':weights,'features':features,
                             'feature_scope':'public state and train-only retrieval signals'}


class FusionPipeline(FixedRecommendationPipeline):
    """Explicit experimental entry, preserving the old request/response contract.

    Compatible with existing AgentLoop dependency injection, never auto-selected.
    Immutable model parameters can be shared; no mutable request cache is kept.
    """
    def __init__(self,retriever,*,mode='no_content',gate=None,content_weight=.5,protect_cf=10):
        session=SessionSeedScorer({i:item.genres for i,item in retriever.catalog.items.items()},
                                  retriever.popularity,retriever.catalog.n_items)
        super().__init__(retriever,RoutingScorer(retriever.known_user,session))
        self.mode,self.learned_gate,self.content_weight,self.protect_cf=mode,gate,content_weight,protect_cf

    def recommend(self,request):
        # Existing validation/error/clarification contract is preserved.
        from .pipeline import FixedRequest
        if not isinstance(request,FixedRequest): raise ValueError('FixedRequest required')
        supplied=request.liked_item_ids+request.disliked_item_ids+request.seen_item_ids
        if any(i>self.retriever.catalog.n_items for i in supplied): return super().recommend(request)
        snapshot=source_snapshot(self.retriever,request,positive_content=True)
        if snapshot['status']!='OK': return super().recommend(request)
        final,fused,scores,decision=predict_snapshot(self.retriever,request,snapshot,
            mode=self.mode,gate=self.learned_gate,content_weight=self.content_weight,protect_cf=self.protect_cf)
        old=snapshot['baseline_retrieval']
        counts={'universe':len(self.candidate_ids),'eligible':len(snapshot['eligible_ids']),
                'per_source':{s:len(ids) for s,ids in snapshot['rankings'].items()},
                'fused':len(fused.candidate_ids),'final':len(final.item_ids),
                'fusion_policy':self.mode,'fusion_weights':decision['weights']}
        if final.status!='OK':
            return FixedResponse(final.status,final.reason,[],old.profile_source,old.fallback_reason,
                                 counts,list(final.missing_fields),request.constraints.k)
        recommendations=[]
        for i in final.item_ids:
            item=self.retriever.catalog.items[i]
            recommendations.append(RecommendationEvidence(i,item.title,sorted(item.genres),item.year,
                scores[i],fused.rrf_scores[i],fused.source_ranks[i]))
        return FixedResponse('OK','FEWER_FEASIBLE_ITEMS' if final.shortfall else None,recommendations,
            old.profile_source,old.fallback_reason,counts,list(final.missing_fields),final.shortfall)

    def diagnose(self,request):
        snap=source_snapshot(self.retriever,request,positive_content=True)
        if snap['status']!='OK': return {'status':snap['status'],'candidate_rows':[]}
        final,fused,_scores,decision=predict_snapshot(self.retriever,request,snap,mode=self.mode,
            gate=self.learned_gate,content_weight=self.content_weight,protect_cf=self.protect_cf)
        return {'status':final.status,'decision':decision,**candidate_diagnostics(snap,fused,final.item_ids)}
