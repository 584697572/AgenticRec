"""Checks for the standalone auditor, independently of candidate system code."""
import importlib.util
import math
from pathlib import Path


path = Path(__file__).resolve().parents[1] / 'scripts/audit_three_way_comparison.py'
spec = importlib.util.spec_from_file_location('three_way_auditor_tests', path)
auditor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auditor)


def example():
    truth = {str(i): {'genres': ['Drama'], 'year': 1995} for i in range(1, 11)}
    label = {'genres': ['Drama'], 'exclude_genres': ['Horror'],
             'year_min': 1990, 'year_max': 2000, 'excluded_raw_ids': [],
             'exclude_previous': False, 'relevant_raw_ids': [3, 8, 99], 'k': 10}
    turn = {'raw_ids': list(range(1, 11)), 'status': 'OK', 'extraction_verified': True}
    return turn, label, truth


def test_independent_rank_metrics_include_all_future_targets():
    turn, label, truth = example()
    values, _ = auditor.assess(turn, label, truth, [])
    assert values['strict_success'] and values['hit'] == 1
    assert values['recall'] == 2 / 3 and values['mrr'] == 1 / 3
    expected = (1 / math.log2(4) + 1 / math.log2(9)) / sum(
        1 / math.log2(i + 1) for i in (1, 2, 3))
    assert math.isclose(values['ndcg'], expected)


def test_repeated_list_exclusion_and_failed_delivery():
    turn, label, truth = example()
    label['exclude_previous'] = True
    values, errors = auditor.assess(turn, label, truth, [8])
    assert not values['strict_success'] and errors['excluded_item_deliveries'] == 1
    turn['status'] = 'EXECUTION_FAILURE'
    values, _ = auditor.assess(turn, label, truth, [])
    assert not values['strict_success']
    assert all(values[key] == 0 for key in ('hit', 'recall', 'ndcg', 'mrr'))


def test_multiple_unknown_items_are_not_claimed_as_duplicate_films():
    turn, label, truth = example()
    turn['raw_ids'][-2:] = [None, None]
    values, errors = auditor.assess(turn, label, truth, [])
    assert not values['strict_success']
    assert errors['unknown_identity_items'] == 2
    assert errors['duplicate_items'] == 0
