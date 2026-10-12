"""Synthetic contract tests: future rows cannot shape IDs or training edges."""
from agenticrec.data import build_contract, choose_time_boundaries, split_interactions


def rows():
    # One warm user, one cold user and an item first liked in the future.
    train = [(1, 10, 5, 1), (1, 11, 4, 2), (1, 12, 5, 3),
             (1, 13, 4, 4), (1, 14, 5, 5), (1, 15, 2, 6),
             (2, 16, 5, 7)]
    valid = [(1, 99, 5, 8), (2, 10, 5, 9)]
    test = [(1, 10, 5, 10), (3, 14, 5, 11)]
    return train, valid, test


def test_global_ties_never_cross_windows():
    source = [(1, i, 5, t) for i, t in enumerate([1, 1, 2, 3, 4, 5, 6, 6, 7, 8], 1)]
    low, high = choose_time_boundaries(source)
    train, valid, test = split_interactions(source, low, high)
    assert max(x[3] for x in train) < min(x[3] for x in valid)
    assert max(x[3] for x in valid) < min(x[3] for x in test)
    assert len(train) + len(valid) + len(test) == len(source)


def test_train_only_ids_and_future_positives_remain_cold():
    train, valid, test = rows()
    result = build_contract(train, valid, test, min_train_positives=5)
    assert result['users'] == {1: 1}
    assert result['items'] == {i: i - 9 for i in range(10, 15)} | {16: 6}
    assert all(u == 1 and i in result['items'] for u, i, _r, _t in result['train_edges'])
    assert result['valid_labels'][0]['segment'] == 'COLD_ITEM'
    assert result['valid_labels'][1]['segment'] == 'COLD_USER'
    assert result['test_labels'][0]['segment'] == 'WARM'
    assert result['test_labels'][1]['segment'] == 'COLD_USER'
    changed = build_contract(train, [(42, 400, 5, 8)], [(43, 401, 5, 9)], min_train_positives=5)
    assert changed['users'] == result['users']
    assert changed['items'] == result['items']
    assert changed['train_edges'] == result['train_edges']


def test_observed_low_ratings_are_not_positive_or_unknown():
    train, valid, test = rows()
    result = build_contract(train, valid, test, min_train_positives=5)
    assert 15 in result['observed_by_user'][1]
    assert 15 not in result['items']
    assert len(result['train_edges']) == 5
