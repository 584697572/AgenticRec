from collections import Counter, defaultdict
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from agenticrec.evaluation.preference_ranker import constrained_ridge, PreferenceRanker, PreferenceSystem, baseline_sources, fuse
from agenticrec.evaluation.frontier_experiment import FrontierSystem
from agenticrec.evaluation.frontier_state import empty_state, merge_state, verify_delivery


def fixture():
    catalog = {i:{'item_id':i,'raw_id':100+i,'title':str(i),'year':1990+i,
                  'genres':['Drama'] if i%2 else ['Action']} for i in range(1,25)}
    base = SimpleNamespace(catalog=SimpleNamespace(n_items=24),known_user=None,
             popularity=Counter({i:30-i for i in catalog}),
             item_users=defaultdict(set,{1:{1,2},3:{2},5:{1}}),
             user_items=defaultdict(set,{1:{1,5,7},2:{1,3,9}}),
             content_vectors={i:{'genre:'+('drama' if i%2 else 'action'):1.} for i in catalog})
    return catalog,base


@pytest.mark.parametrize('general',[False,True])
def test_closed_form_matches_independent_column_ridge(general):
    torch.manual_seed(71)
    z = torch.randint(0,2,(15,6)).to(torch.float64)
    y = torch.randint(0,2,(15,6)).to(torch.float64) if general else z
    b,residual = constrained_ridge(z.T@z,z.T@y,7)
    reference = torch.zeros_like(b,dtype=torch.float64)
    for j in range(6):
        keep = [i for i in range(6) if i!=j]
        inputs = z[:,keep]
        reference[keep,j] = torch.linalg.solve(inputs.T@inputs+7*torch.eye(5,dtype=torch.float64),inputs.T@y[:,j])
    assert torch.allclose(b.to(torch.float64),reference,atol=1e-7)
    assert b.diagonal().abs().sum()==0 and residual<1e-8


@pytest.mark.parametrize('regularization',[0,-1,float('inf'),float('nan')])
def test_regularization_is_positive_finite(regularization):
    with pytest.raises(ValueError):constrained_ridge(torch.eye(2),torch.eye(2),regularization)


def system(cls,base,catalog,ranker=None):
    value = cls.__new__(cls)
    for k,v in dict(variant='C1',catalog=catalog,base=base,policy=None,sft_policy=None,
                   policy_stage='rl',stateful=True,hard_gate=True,align=True,verifier=True,
                   state=empty_state(),history=[]).items():setattr(value,k,v)
    if ranker is not None:value.preference_ranker=ranker
    return value


def delta(**kwargs):
    d = dict(include_genres=None,exclude_genres=None,year_min=None,year_max=None,k=None,
             liked_item_ids=[],seen_item_ids=[],exclude_previous=False,reset_constraints=False,
             genre_mode='replace',evidence={})
    d.update(kwargs);return d


def test_original_ranking_and_conversation_are_identical_with_original_weights():
    catalog,base=fixture()
    ranker=PreferenceRanker(base,catalog,{'weights':{'cf':1,'content':1,'popularity':1}})
    original=system(FrontierSystem,base,catalog);candidate=system(PreferenceSystem,base,catalog,ranker)
    public={'messages':['first','second'],'catalog_context':[]}
    for n,d in enumerate([delta(liked_item_ids=[1],seen_item_ids=[1],k=5),delta(exclude_previous=True,include_genres=['Drama'])]):
        a=original.execute(public,n,d);b=candidate.execute(public,n,d)
        assert a==b and original.state==candidate.state and original.history==candidate.history


def test_learned_ranking_only_reads_public_likes_and_never_relaxes_gates():
    catalog,base=fixture()
    checkpoint={'model':'ease','regularization':10,'item_ids':list(catalog),
                'weights':torch.arange(24*24,dtype=torch.float32).reshape(24,24)}
    ranker=PreferenceRanker(base,catalog,{'model':'ease','regularization':10,'weights':{'learned':1}},checkpoint)
    s=dict(empty_state(),liked_item_ids=[1],seen_item_ids=[1],include_genres=['Drama'],year_max=2010,k=5)
    ids=ranker.rank(s)
    assert ids==[19,17,15,13,11] and verify_delivery(ids,catalog,s)==[]
    # Hidden evaluator attributes cannot affect preference ranking.
    assert ranker.rank(dict(s,user_id=987,relevant_raw_ids=[103]))==ids
    short=dict(s,k=20)
    assert verify_delivery(ranker.rank(short),catalog,short)==['WRONG_QUANTITY']


def test_model_without_public_history_falls_back_to_exact_c1():
    catalog,base=fixture()
    checkpoint={'model':'ease','regularization':10,'item_ids':list(catalog),'weights':torch.ones(24,24)}
    r=PreferenceRanker(base,catalog,{'model':'ease','regularization':10,'weights':{'learned':1}},checkpoint)
    s=empty_state();eligible,sources=baseline_sources(base,catalog,s)
    assert r.rank(s)==fuse(base,eligible,sources,{'cf':1,'content':1,'popularity':1})


def test_mismatched_checkpoint_order_rejected():
    catalog,base=fixture()
    config={'model':'ease','regularization':10,'weights':{'learned':1}}
    cp={'model':'ease','regularization':10,'item_ids':list(reversed(catalog)),'weights':torch.eye(24)}
    with pytest.raises(ValueError,match='item order'):PreferenceRanker(base,catalog,config,cp)


def test_previous_batch_is_excluded_from_new_learned_delivery():
    catalog,base=fixture()
    cp={'model':'ease','regularization':10,'item_ids':list(catalog),'weights':torch.ones(24,24)}
    ranker=PreferenceRanker(base,catalog,{'model':'ease','regularization':10,'weights':{'learned':1}},cp)
    candidate=system(PreferenceSystem,base,catalog,ranker)
    p={'messages':['first','different batch'],'catalog_context':[]}
    first=candidate.execute(p,0,delta(liked_item_ids=[1],seen_item_ids=[1],k=5))
    second=candidate.execute(p,1,delta(exclude_previous=True))
    assert not set(first['raw_ids'])&set(second['raw_ids'])
    assert second['state']['liked_item_ids']==[1] and candidate.history[-1]['content']==second['answer']
