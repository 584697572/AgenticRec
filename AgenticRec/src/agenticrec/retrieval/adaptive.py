"""Public-signal reliability gating and bounded weighted rank fusion.

No evaluator target, scenario family or future interaction is an input.
Raw scores stay source-specific; fusion never sums incompatible scales.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
import math

import torch

from .fusion import FusionResult, reciprocal_rank_fusion


FEATURE_NAMES = ('liked_count', 'disliked_count', 'constraint_count', 'known_history',
                 'seed_support', 'cf_coverage', 'content_coverage', 'top20_overlap',
                 'content_margin')


def weighted_fusion(rankings, weights, *, universe, per_route=100, union_cap=200,
                    rank_constant=60, protect_cf=0):
    # Reuse baseline validation even for zero-weight sources.
    validated = reciprocal_rank_fusion(rankings, universe=universe, per_route=per_route,
                                      union_cap=max(1, len(universe)), rank_constant=rank_constant)
    if set(weights) != set(rankings) or any(type(w) not in (int,float) or not math.isfinite(w) or w < 0 for w in weights.values()):
        raise ValueError('Need one finite nonnegative weight per source')
    if type(union_cap) is not int or union_cap < 1 or type(protect_cf) is not int or not 0 <= protect_cf <= union_cap:
        raise ValueError('Invalid fusion capacity or protection')
    scores = {i:sum(weights[s]/(rank_constant+r) for s,r in rs.items())
              for i,rs in validated.source_ranks.items()}
    scores = {i:v for i,v in scores.items() if v > 0}
    ordered = sorted(scores,key=lambda i:(-scores[i],i))
    cf = next((ids for s,ids in rankings.items() if s.startswith('cf_') and weights[s] > 0), [])
    protected = list(cf[:min(protect_cf,per_route)])
    kept = set(protected)
    for i in ordered:
        if len(kept) >= union_cap: break
        kept.add(i)
    ordered = [i for i in ordered if i in kept]
    return FusionResult(ordered,{i:scores[i] for i in ordered},
                        {i:validated.source_ranks[i] for i in ordered})


def _cf_scores(retriever, eligible, liked, user_id, authorized):
    known = retriever.known_user
    if authorized and user_id is not None and known is not None and user_id <= known.n_users:
        return dict(zip(eligible,known.score(user_id,eligible))), 'cf_model'
    if not liked: return {},None
    # Cache immutable train co-occurrence only, never user state or labels.
    if not hasattr(retriever,'_research_cf_cache'): retriever._research_cf_cache = {}
    scores = defaultdict(float)
    for seed in liked:
        if seed not in retriever._research_cf_cache:
            counts = Counter()
            for u in sorted(retriever.item_users[seed]): counts.update(sorted(retriever.user_items[u]))
            retriever._research_cf_cache[seed] = dict(counts)
        for i,c in retriever._research_cf_cache[seed].items():
            scores[i] += c/(math.sqrt(retriever.popularity[i]*retriever.popularity[seed])*len(liked))
    allowed=set(eligible)
    return {i:s for i,s in scores.items() if i in allowed}, 'cf_item_item'


def source_snapshot(retriever, request, *, positive_content=True, per_route=100):
    """All recalled candidates (including later drops), raw scores and ranks.

    Public FixedRequest is validated by the unchanged baseline hard gate first.
    Candidate scores cover each source's capped output, not the entire catalog.
    """
    args = dict(user_id=request.user_id,history_authorized=request.history_authorized,
                liked_ids=request.liked_item_ids,disliked_ids=request.disliked_item_ids,
                seen_ids=request.seen_item_ids,per_route=per_route)
    old = retriever.retrieve(list(retriever.catalog.items),request.constraints,**args)
    snap={'status':old.status,'eligible_ids':old.pre_filter.item_ids,
          'rankings':{},'scores':{},'profile_source':old.profile_source,
          'effective_constraints':old.effective_constraints,'seen_ids':old.seen_ids,
          'baseline_retrieval':old,'per_route':per_route}
    if old.status != 'OK': return snap
    eligible=old.pre_filter.item_ids
    query=defaultdict(float)
    for sign,seeds in [(1,request.liked_item_ids),(-1,() if positive_content else request.disliked_item_ids)]:
        for seed in seeds:
            for term,w in retriever.content_vectors[seed].items(): query[term] += sign*w/len(seeds)
    for genre in request.constraints.include_genres: query['genre:'+genre.casefold()] += 1
    raw = {'content':{i:sum(query.get(t,0)*w for t,w in retriever.content_vectors[i].items()) for i in eligible}}
    cf,name=_cf_scores(retriever,eligible,request.liked_item_ids,request.user_id,request.history_authorized)
    if name: raw[name]=cf
    raw['popularity']={i:float(retriever.popularity[i]) for i in eligible}
    for source,values in raw.items():
        ids=sorted((i for i,s in values.items() if source not in ('content','cf_item_item') or s>0),key=lambda i:(-values[i],i))[:per_route]
        snap['rankings'][source]=ids
        snap['scores'][source]={i:values[i] for i in ids}
    return snap


def gate_features(retriever,request,snapshot):
    ranks=snapshot['rankings']; cf=next((ids for s,ids in ranks.items() if s.startswith('cf_')),[])
    content=ranks.get('content',[]); limit=snapshot['per_route']
    support=[math.log1p(retriever.popularity[i]) for i in request.liked_item_ids]
    denominator=math.log1p(max(retriever.popularity.values(),default=0)) or 1
    values=[snapshot['scores']['content'][i] for i in content[:2]]
    margin=(values[0]-values[1])/(abs(values[0])+1e-12) if len(values)>1 else 0
    return [min(len(request.liked_item_ids),10)/10, min(len(request.disliked_item_ids),10)/10,
            min(len(request.constraints.include_genres)+len(request.constraints.exclude_genres),5)/5,
            float(snapshot['profile_source']=='known_user'),
            sum(support)/len(support)/denominator if support else 0,
            len(cf)/limit,len(content)/limit,
            len(set(cf[:20])&set(content[:20]))/max(1,min(20,len(cf),len(content))), margin]


@dataclass(frozen=True)
class LogisticGate:
    weights: tuple[float,...]
    bias: float
    means: tuple[float,...]
    scales: tuple[float,...]

    @classmethod
    def fit(cls,features,labels,*,regularization,seed,steps=200):
        if not features or len(features)!=len(labels) or any(len(row)!=len(FEATURE_NAMES) for row in features):
            raise ValueError('Invalid fit arrays')
        if any(not math.isfinite(x) for row in features for x in row) or any(y not in (0,1) for y in labels):
            raise ValueError('Nonfinite features or invalid labels')
        if type(regularization) not in (int,float) or not math.isfinite(regularization) or regularization<=0:
            raise ValueError('Invalid regularization')
        torch.manual_seed(seed)
        x=torch.tensor(features,dtype=torch.float64); y=torch.tensor(labels,dtype=torch.float64)
        means=x.mean(0); std=x.std(0,unbiased=False)
        scales=torch.where(std>1e-6,std,torch.ones_like(std)); x=(x-means)/scales
        weights=torch.nn.Parameter(torch.randn(len(FEATURE_NAMES),dtype=torch.float64)*.01)
        with torch.no_grad(): weights[std<=1e-6]=0
        bias=torch.nn.Parameter(torch.zeros((),dtype=torch.float64))
        optimizer=torch.optim.Adam([weights,bias],lr=.05)
        for _ in range(steps):
            optimizer.zero_grad()
            loss=torch.nn.functional.binary_cross_entropy_with_logits(x@weights+bias,y)+regularization*weights.square().mean()
            loss.backward(); optimizer.step()
        return cls(tuple(weights.detach().tolist()),bias.item(),tuple(means.tolist()),tuple(scales.tolist()))

    def predict(self,features):
        if len(features)!=len(FEATURE_NAMES) or any(type(x) not in (int,float) or not math.isfinite(x) for x in features):
            raise ValueError('Invalid gate features')
        logit=self.bias+sum(w*(x-m)/s for w,x,m,s in zip(self.weights,features,self.means,self.scales))
        return 1/(1+math.exp(-max(-700,min(700,logit))))

    def to_dict(self):
        return {'feature_names':list(FEATURE_NAMES),'weights':list(self.weights),'bias':self.bias,
                'means':list(self.means),'scales':list(self.scales)}

    @classmethod
    def from_dict(cls,value):
        if set(value)!= {'feature_names','weights','bias','means','scales'} or value['feature_names']!=list(FEATURE_NAMES):
            raise ValueError('Gate schema mismatch')
        if any(len(value[k])!=len(FEATURE_NAMES) for k in ('weights','means','scales')):
            raise ValueError('Gate dimension mismatch')
        vals=[value['bias']]+value['weights']+value['means']+value['scales']
        if any(type(x) not in (int,float) or not math.isfinite(x) for x in vals) or any(x<=0 for x in value['scales']):
            raise ValueError('Invalid gate parameters')
        return cls(tuple(value['weights']),float(value['bias']),tuple(value['means']),tuple(value['scales']))


def fusion_weights(mode,snapshot,features,*,gate=None,content_weight=.5):
    if mode not in ('no_content','fixed','rrf','conditional','learned'):
        raise ValueError('Unknown fusion policy')
    if mode=='no_content': weight=0
    elif mode=='fixed': weight=content_weight
    elif mode=='rrf': weight=1
    elif mode=='conditional':
        # Fixed public-signal prior, not tuned on evaluator scenario names.
        weight=.5 if features[5]<.5 or features[4]<.25 else .25
        if features[1]>0 and features[7]<.1: weight=0
    else:
        if gate is None: raise ValueError('Trained gate required')
        weight=max(0,2*gate.predict(features)-1)
    if type(weight) not in (int,float) or not math.isfinite(weight) or not 0<=weight<=1:
        raise ValueError('Invalid content weight')
    return {s:(weight if s=='content' else (1 if mode=='rrf' else .1) if s=='popularity' else 1)
            for s in snapshot['rankings']}


def candidate_diagnostics(snapshot,fused,final_ids):
    ranks={s:{i:r for r,i in enumerate(ids,1)} for s,ids in snapshot['rankings'].items()}
    union=sorted({i for ids in snapshot['rankings'].values() for i in ids})
    retained=set(fused.candidate_ids); final=set(final_ids)
    if not final<=retained: raise ValueError('Final item outside fused candidates')
    sources=list(ranks)
    return {'eligible_count':len(snapshot['eligible_ids']),'source_counts':{s:len(v) for s,v in ranks.items()},
        'overlap_counts':{a+'|'+b:len(set(ranks[a])&set(ranks[b])) for n,a in enumerate(sources) for b in sources[n+1:]},
        'retention':{s:len(set(v)&retained)/len(v) if v else None for s,v in ranks.items()},
        'candidate_rows':[{'item_id':i,'source_ranks':{s:r[i] for s,r in ranks.items() if i in r},
            'source_scores':{s:snapshot['scores'][s][i] for s,r in ranks.items() if i in r},
            'retained':i in retained,'final':i in final,'fusion_score':fused.rrf_scores.get(i)} for i in union]}


from .sources import CandidateRetriever


class CachedResearchRetriever(CandidateRetriever):
    """Train-count cache with identical itemCF arithmetic and cap semantics."""
    def collaborative_rank(self,eligible_ids,liked_ids,*,user_id,history_authorized,limit):
        scores,name=_cf_scores(self,eligible_ids,liked_ids,user_id,history_authorized)
        if name is None: return [],None
        ids=sorted((i for i,s in scores.items() if name=='cf_model' or s>0),key=lambda i:(-scores[i],i))[:limit]
        return ids,name
