"""Hand-calculated examples for the frozen full-catalog ranking protocol."""
import math
import hashlib
import json
from pathlib import Path

import pytest

from agenticrec.evaluation.metrics import evaluate_cohort, rank_candidates, score_ranking
from agenticrec.evaluation.cohort import fixed_cohort


ROOT = Path(__file__).resolve().parents[3]


def test_two_relevant_items_can_be_scored_by_hand():
    result = score_ranking([2, 1, 3], {1, 3}, {1, 2, 3, 4}, k=5)
    ideal = 1 + 1 / math.log2(3)
    assert result['recall'] == 1
    assert result['hit'] == 1
    assert result['mrr'] == 0.5
    assert result['ndcg'] == pytest.approx((1 / math.log2(3) + 1 / math.log2(4)) / ideal)


def test_complete_hit_all_wrong_short_and_empty():
    assert score_ranking([1, 2], {1, 2}, {1, 2, 3}, 2)['ndcg'] == 1
    assert score_ranking([3], {1, 2}, {1, 2, 3}, 2)['recall'] == 0
    assert score_ranking([2], {1, 2}, {1, 2, 3}, 2)['recall'] == 0.5
    assert score_ranking([], {1}, {1, 2}, 2) == {'recall': 0, 'hit': 0, 'mrr': 0, 'ndcg': 0}


def test_stable_full_universe_topk_and_seen_filter():
    scores = {1: 0.5, 2: 0.8, 3: 0.8, 4: -1.0}
    assert rank_candidates(scores, {1, 2, 3, 4}, 2) == [2, 3]
    assert rank_candidates(scores, {1, 2, 3, 4}, 10, seen={2, 4}) == [3, 1]
    assert rank_candidates({1: 0.3}, {1, 2}, 1, seen={1, 2}) == []


@pytest.mark.parametrize('ranking', [[1, 1], [0], [7], [2]])
def test_duplicate_padding_unknown_and_seen_items_are_invalid(ranking):
    with pytest.raises(ValueError):
        score_ranking(ranking, {1}, {1, 2, 3}, 2, seen={2})


def test_nan_missing_score_duplicate_candidates_and_invalid_k_fail_closed():
    with pytest.raises(ValueError):
        rank_candidates({1: float('nan')}, {1}, 1)
    with pytest.raises(ValueError):
        rank_candidates({1: 1}, {1, 2}, 2)
    with pytest.raises(ValueError):
        rank_candidates({1: 1}, [1, 1], 2)
    with pytest.raises(ValueError):
        rank_candidates({1: 1}, {1}, 0)


def test_cohort_denominator_keeps_empty_and_invalid_attempts():
    result = evaluate_cohort([1, 2, 3, 4], {1: {1}, 2: {2}, 3: {3}, 4: set()},
                             {1: [1], 2: [], 3: [0]}, {1, 2, 3}, k=1)
    assert result['users_total'] == 4
    assert result['users_without_relevance'] == 1
    assert result['users_evaluated'] == 3
    assert result['failures'] == 2
    assert result['recall'] == pytest.approx(1 / 3)
    assert result['hit'] == pytest.approx(1 / 3)
    assert result['mrr'] == pytest.approx(1 / 3)
    assert result['ndcg'] == pytest.approx(1 / 3)


def test_relevance_outside_frozen_candidates_is_rejected():
    with pytest.raises(ValueError):
        evaluate_cohort([1], {1: {99}}, {1: [1]}, {1, 2}, k=1)


def test_evaluation_cohort_keeps_training_seen_and_empty_users():
    train = [{'raw_user_id': 10, 'raw_item_id': 5},
             {'raw_user_id': 10, 'raw_item_id': 6}]
    test = [{'raw_user_id': 10, 'raw_item_id': 7, 'rating': 5},
            {'raw_user_id': 11, 'raw_item_id': 6, 'rating': 2},
            {'raw_user_id': 13, 'raw_item_id': 7, 'rating': 5}]
    cohort = fixed_cohort({'10': 1, '11': 2}, {'5': 1, '6': 2, '7': 3}, train, test)
    assert cohort['user_ids'] == [1, 2]
    assert cohort['candidate_ids'] == [1, 2, 3]
    assert cohort['seen_by_user']['1'] == [1, 2]
    assert cohort['relevant_by_user']['1'] == [3]
    assert cohort['relevant_by_user']['2'] == []
    assert cohort['coverage_only']['users_without_relevance'] == 1


def test_frozen_cohort_is_bound_to_split_and_training_mapping():
    summary_path = ROOT / 'reproduction/native_runs/20261006_metrics/cohort_summary.json'
    if not summary_path.exists():
        pytest.skip('Local MovieLens1M cohort has not been frozen')
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    for split in ('valid', 'test'):
        entry = summary['splits'][split]
        path = ROOT / entry['private_path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256']
        cohort = json.loads(path.read_text(encoding='utf-8'))
        assert cohort['evaluator_only'] is True
        assert cohort['data_manifest_sha256'] == summary['data_manifest_sha256']
        assert cohort['id_map_sha256'] == summary['id_map_sha256']
        assert len(cohort['user_ids']) == entry['warm_users']
        assert len(cohort['candidate_ids']) == entry['candidate_items']
        assert sum(bool(v) for v in cohort['relevant_by_user'].values()) == entry['users_with_relevance']
        assert sum(len(v) for v in cohort['relevant_by_user'].values()) == entry['relevant_items']
        assert all(not (set(cohort['seen_by_user'][u]) & set(relevant))
                   for u, relevant in cohort['relevant_by_user'].items())
