from collections import Counter
from copy import deepcopy
import math

import pytest

from agenticrec.evaluation.frontier_state import empty_state, verify_delivery
from agenticrec.evaluation.simple_baselines import SimpleRanker, quality, summarize, public_state


def fixture():
    catalog = {i: {'item_id': i, 'raw_id': 100+i, 'year': 2000+i,
                   'title': str(i), 'genres': ['Drama'] if i % 2 else ['Action']}
               for i in range(1, 15)}
    edges = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (3, 2), (4, 2), (5, 4)]
    content = {i: {'odd' if i % 2 else 'even': 1.} for i in catalog}
    return catalog, edges, content


def test_itemcf_matches_independent_user_set_cosine_not_raw_counts():
    catalog, edges, content = fixture()
    ranker = SimpleRanker(catalog, edges, content)
    state = dict(empty_state(), liked_item_ids=[1], seen_item_ids=[1])
    scores = ranker.scores(list(catalog), state, 'itemcf')
    # Movie 2 has twice the overlap of 3, but four times its popularity.
    assert scores[2] == pytest.approx(2 / math.sqrt(2*4))
    assert scores[3] == pytest.approx(1 / math.sqrt(2*1))
    users = {i: {u for u, item in edges if item == i} for i in catalog}
    for i in catalog:
        expected = len(users[1] & users[i]) / math.sqrt(len(users[1])*len(users[i])) if users[i] else 0.
        assert scores[i] == pytest.approx(expected)
    assert ranker.rank(state, 'itemcf')[:2] == [2, 3]


def test_popularity_is_train_only_and_hidden_identity_cannot_affect_any_control():
    catalog, edges, content = fixture()
    ranker = SimpleRanker(catalog, edges, content)
    state = dict(empty_state(), liked_item_ids=[1], seen_item_ids=[1])
    hidden = dict(deepcopy(state), user_id=999, relevant_raw_ids=[114], complete_history=list(catalog))
    assert ranker.rank(state, 'popular')[:3] == [2, 3, 4]
    for method in ('popular', 'itemcf', 'content'):
        assert ranker.rank(state, method) == ranker.rank(hidden, method)


@pytest.mark.parametrize('method', ['popular', 'itemcf', 'content'])
def test_all_controls_share_seen_genre_year_and_previous_batch_gates(method):
    catalog, edges, content = fixture()
    ranker = SimpleRanker(catalog, edges, content)
    state = dict(empty_state(), liked_item_ids=[1], seen_item_ids=[1], excluded_previous_ids=[3],
                 include_genres=['Drama'], year_min=2004, year_max=2012, k=4)
    ids = ranker.rank(state, method)
    assert set(ids) == {5, 7, 9, 11}
    assert verify_delivery(ids, catalog, state) == []


def test_itemcf_has_no_popularity_blend_in_zero_scores_and_empty_history_fallback():
    catalog, edges, content = fixture()
    ranker = SimpleRanker(catalog, edges, content)
    state = dict(empty_state(), liked_item_ids=[1], seen_item_ids=[1], k=10)
    assert ranker.rank(state, 'itemcf')[2:] == list(range(4, 12))
    empty = empty_state()
    assert ranker.rank(empty, 'itemcf') == ranker.rank(empty, 'popular')


def test_quality_matches_hand_computed_multitarget_metrics_and_rejects_duplicates():
    values = quality(list(range(1, 11)), [2, 4, 11])
    assert values['hit'] == 1. and values['recall'] == pytest.approx(2/3)
    assert values['mrr'] == .5
    assert values['ndcg'] == pytest.approx((1/math.log2(3)+1/math.log2(5))/(1+1/math.log2(3)+.5))
    with pytest.raises(ValueError): quality([1]*10, [1])
    with pytest.raises(ValueError): quality(list(range(10)), [])


def test_duplicate_training_edges_or_public_history_rejected():
    catalog, edges, content = fixture()
    with pytest.raises(ValueError): SimpleRanker(catalog, edges + [edges[0]], content)
    with pytest.raises(ValueError): public_state({'catalog_context': [{'item_id': 1}]*2})


def test_incomplete_matched_cohort_is_rejected_instead_of_reported_as_complete():
    with pytest.raises(ValueError): summarize([])
