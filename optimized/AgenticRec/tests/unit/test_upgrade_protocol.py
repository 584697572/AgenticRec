"""Future item IDs, tied cutoffs and user grouping must not leak into research."""
import pytest

from agenticrec.research_protocol import split_ratings, train_space, user_partition


def test_external_freeze_counts_only_nonempty_relevance_users(tmp_path, monkeypatch):
    import zipfile
    import agenticrec.research_protocol as protocol
    raw = tmp_path/'raw'; raw.mkdir()
    (raw/'ml-100k-README.txt').write_text('Synthetic test terms')
    with zipfile.ZipFile(raw/'ml-100k.zip','w') as z:
        z.writestr('ml-100k/u.data','1 1 5 1\n'*100000)
        z.writestr('ml-100k/u.genre','Comedy|0\n')
        z.writestr('ml-100k/u.item','\n'.join(f'{i}|Movie {i} (1995)|||x|1' for i in range(1,7)))
    train = [(1,i,5,i) for i in range(1,6)] + [(2,i,5,i) for i in range(2,7)]
    monkeypatch.setattr(protocol,'split_ratings',lambda rows:(train,[(1,6,5,20)],[(1,6,5,30)]))
    result=protocol.prepare_external(raw,tmp_path/'out')
    assert result['warm_users']==2
    assert result['coverage']['valid']['users_with_relevance']==1
    assert result['coverage']['valid']['users_without_relevance']==1


def test_time_ties_enter_later_window_and_future_items_never_enter_mapping():
    rows = [(1, 1, 5, 1), (1, 2, 4, 2), (2, 1, 5, 3), (2, 2, 5, 4),
            (1, 99, 5, 5), (2, 99, 5, 5), (1, 100, 5, 6), (2, 100, 5, 7),
            (3, 200, 5, 8), (3, 201, 5, 9)]
    train, valid, test = split_ratings(rows, train_fraction=.5, valid_fraction=.3)
    assert max(r[3] for r in train) < min(r[3] for r in valid)
    assert max(r[3] for r in valid) < min(r[3] for r in test)
    users, items = train_space(train, min_positives=2)
    assert set(users) == {1, 2}
    assert set(items) == {1, 2}
    assert items[1] == 1 and 99 not in items and 200 not in items


def test_low_ratings_do_not_make_candidates_and_duplicate_interactions_fail():
    rows = [(1, 1, 5, 1), (1, 2, 4, 2), (1, 9, 1, 3)]
    assert train_space(rows, min_positives=2)[1] == {1: 1, 2: 2}
    with pytest.raises(ValueError, match='duplicate'):
        split_ratings(rows + [rows[0]])


def test_all_states_and_seeds_of_a_user_stay_in_one_partition():
    groups = user_partition(list(range(1, 101)))
    assert [len(groups[k]) for k in ('fit', 'select', 'audit')] == [60, 20, 20]
    assert not set(groups['fit']) & set(groups['select'])
    assert not set(groups['fit']) & set(groups['audit'])
    assert groups == user_partition(list(reversed(range(1, 101))))
