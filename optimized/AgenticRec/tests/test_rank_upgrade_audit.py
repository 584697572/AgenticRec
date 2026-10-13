import copy
import importlib.util
from pathlib import Path
import sys

import pytest


def auditor():
    scripts = Path(__file__).resolve().parents[1]/'scripts'
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location('rank_upgrade_auditor', scripts/'audit_rank_upgrade.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def fixture_rows(module):
    public = {'e': {'liked_item_ids': [20], 'history_n': 3}}
    private = {'e': {'relevant_item_ids': [11], 'history_n': 3}}
    truth = {str(i): {'genres': [], 'year': 2000} for i in range(1, 21)}
    predictions = [{'system': system, 'seed': seed, 'episode_id': 'e', 'history_n': 3,
                    'item_ids': list(range(1, 11))}
                   for system, seeds in module.SYSTEM_SEEDS.items() for seed in seeds]
    scores = [dict(r, metrics={'hit': 0., 'recall': 0., 'ndcg': 0., 'mrr': 0.}) for r in predictions]
    return scores, predictions, public, private, truth


def test_independent_scores_accept_zero_hits_without_inventing_results():
    module = auditor()
    by, means = module.verify_scores(*fixture_rows(module), expected_users=1)
    assert len(by) == 9 and means['route1']['e']['hit'] == 0


@pytest.mark.parametrize('fault', ['fabricated_hit', 'missing_seed', 'duplicate_seed', 'seen_item'])
def test_independent_scores_reject_corrupted_or_incomplete_evidence(fault):
    module = auditor()
    scores, predictions, public, private, truth = copy.deepcopy(fixture_rows(module))
    if fault == 'fabricated_hit':
        scores[0]['metrics']['hit'] = 1.
    elif fault == 'missing_seed':
        scores.pop(); predictions.pop()
    elif fault == 'duplicate_seed':
        scores[-1] = scores[-2]; predictions[-1] = predictions[-2]
    else:
        scores[0]['item_ids'][0] = predictions[0]['item_ids'][0] = 20
    with pytest.raises(AssertionError):
        module.verify_scores(scores, predictions, public, private, truth, expected_users=1)


def test_selection_must_use_all_seeds_and_the_development_mean():
    module = auditor()
    candidates = [{'seed': seed, 'config': {'epoch': n}, 'metrics': {
        'hit': .2 if n == 0 else .1, 'ndcg': .1, 'recall': .1, 'mrr': .1}}
        for n in range(9) for seed in (7, 42, 2026)]
    value = {'status': 'DEVELOPED', 'test_opened': False, 'api_requests': 0, 'candidates': candidates,
             'selected': {'config': {'epoch': 0}, 'metrics': {'hit': sum([.2]*3)/3,
                          'ndcg': sum([.1]*3)/3, 'recall': sum([.1]*3)/3, 'mrr': sum([.1]*3)/3}}}
    assert module.verify_selection(value, 'route1') == value['selected']
    value['selected']['config']['epoch'] = 8
    with pytest.raises(AssertionError): module.verify_selection(value, 'route1')
