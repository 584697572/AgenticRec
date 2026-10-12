"""Meaningful fusion, trace, label isolation and item-local feedback contracts."""
from dataclasses import replace
import math

import pytest

from agenticrec.retrieval.adaptive import weighted_fusion, source_snapshot, gate_features, LogisticGate
from agenticrec.ranking.constraints import Catalog, Item, Constraints
from agenticrec.retrieval.sources import CandidateRetriever
from agenticrec.retrieval.fusion import reciprocal_rank_fusion
from agenticrec.pipeline import FixedRequest


def retriever():
    catalog = Catalog([Item(i, str(i), frozenset({'Comedy'} if i != 4 else {'Drama'}), 1995)
                       for i in range(1, 7)])
    return CandidateRetriever(catalog, [(1, 1), (1, 2), (1, 3), (2, 1), (2, 3), (2, 5),
                                     (3, 2), (3, 4), (3, 6)])


def test_uniform_weight_is_exact_rrf_and_scales_are_not_added():
    ranks = {'content': [2, 1, 3], 'cf_item_item': [3, 2], 'popularity': [1, 2]}
    assert weighted_fusion(ranks, {s: 1 for s in ranks}, universe=[1,2,3]) == reciprocal_rank_fusion(ranks, universe=[1,2,3])
    assert weighted_fusion(ranks, {'content':0, 'cf_item_item':1, 'popularity':0},
                           universe=[1,2,3]).candidate_ids == [3,2]


def test_protected_cf_keeps_capacity_and_rejects_invalid_weights():
    r = weighted_fusion({'cf_item_item':[1,2], 'content':[3,4], 'popularity':[3,4]},
                        {'cf_item_item':.01,'content':1,'popularity':1}, universe=[1,2,3,4],
                        union_cap=2, protect_cf=2)
    assert set(r.candidate_ids) == {1,2} and len(r.candidate_ids) == 2
    for value in [-1, float('nan'), float('inf'), True]:
        with pytest.raises(ValueError):
            weighted_fusion({'content':[1]}, {'content':value}, universe=[1])
    with pytest.raises(ValueError):
        weighted_fusion({'content':[1,1]}, {'content':1}, universe=[1])


def test_diagnostics_keep_dropped_candidates_raw_scores_and_overlap():
    base = retriever()
    request = FixedRequest(Constraints(k=2), liked_item_ids=(1,), disliked_item_ids=(2,))
    snap = source_snapshot(base, request, positive_content=True, per_route=3)
    assert 1 not in snap['eligible_ids'] and 2 not in snap['eligible_ids']
    assert set(snap['rankings']) == set(snap['scores'])
    assert all(math.isfinite(s) for scores in snap['scores'].values() for s in scores.values())
    # A disliked film is removed locally; another Comedy remains content-eligible.
    assert 3 in snap['rankings']['content']
    assert snap['scores']['content'][3] > 0
    assert snap['status'] == 'OK'


def test_gate_features_use_public_state_and_train_source_signals_only():
    base = retriever(); request = FixedRequest(Constraints(k=2), liked_item_ids=(1,))
    snap = source_snapshot(base, request, positive_content=True)
    features = gate_features(base, request, snap)
    changed = {**snap, 'hidden_acceptable_ids':[99], 'test_target':[999], 'scenario_family':'feedback'}
    assert features == gate_features(base, request, changed)
    assert len(features) == 9 and all(math.isfinite(x) for x in features)
    with pytest.raises(ValueError):
        FixedRequest.parse({'schema_version':1,'constraints':{},'test_target':[3]})


def test_logistic_gate_fit_serialization_and_bad_feature_dimensions():
    gate = LogisticGate.fit([[0.0]*9,[1.0]*9]*5, [0.0,1.0]*5, regularization=.01, seed=7)
    assert gate.predict([1.0]*9) > gate.predict([0.0]*9)
    loaded = LogisticGate.from_dict(gate.to_dict())
    assert loaded.predict([.5]*9) == gate.predict([.5]*9)
    with pytest.raises(ValueError): loaded.predict([1.0])
    with pytest.raises(ValueError): LogisticGate.from_dict({**gate.to_dict(), 'weights':[float('nan')]*9})


def test_constant_fit_features_do_not_create_untrained_confidence():
    gate=LogisticGate.fit([[0.0]*9]*5,[0.0]*5,regularization=.1,seed=42)
    assert gate.weights==(0.0,)*9
    assert gate.predict([1.0]*9)==gate.predict([0.0]*9)


def test_cached_itemcf_is_exact_baseline_including_input_order_and_caps():
    from agenticrec.retrieval.adaptive import CachedResearchRetriever
    base=retriever()
    edges=[(u,i) for u,items in base.user_items.items() for i in items]
    cached=CachedResearchRetriever(base.catalog,edges)
    for seeds in ((),(1,),(1,2),(2,1)):
        for limit in (1,6):
            assert cached.collaborative_rank([3,4,5,6],seeds,user_id=None,history_authorized=False,limit=limit)==base.collaborative_rank([3,4,5,6],seeds,user_id=None,history_authorized=False,limit=limit)
