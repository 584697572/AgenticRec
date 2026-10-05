"""Cross-check the actual frozen dataset if present, plus source-independent gates."""
import json
import hashlib
from collections import Counter
from pathlib import Path

import pytest

from agenticrec.data import build_contract


ROOT = Path(__file__).resolve().parents[3]


def test_holdout_changes_cannot_change_training_mapping_or_graph():
    train = [(1, i, 5, i) for i in range(1, 6)] + [(1, 6, 1, 6)]
    a = build_contract(train, [(1, 200, 5, 7)], [(1, 300, 5, 8)])
    b = build_contract(train, [(5, 400, 5, 7)], [(6, 500, 5, 8)])
    assert a['users'] == b['users'] and a['items'] == b['items']
    assert a['train_edges'] == b['train_edges']
    assert a['observed_by_user'] == b['observed_by_user']


def test_frozen_dataset_has_no_future_graph_edges():
    manifest_path = ROOT / 'artifacts/data_manifest.json'
    if not manifest_path.exists():
        pytest.skip('Dataset not yet frozen; synthetic leakage test remains active')
    import pyarrow.parquet as pq
    m = json.loads(manifest_path.read_text(encoding='utf-8'))
    train = pq.read_table(ROOT / 'data/processed/ml-1m-v1/train.parquet').to_pydict()
    edges = pq.read_table(ROOT / 'data/processed/ml-1m-v1/train_positive.parquet').to_pydict()
    id_map = json.loads((ROOT / 'data/processed/ml-1m-v1/id_map.json').read_text(encoding='utf-8'))
    train_keys = set(zip(train['raw_user_id'], train['raw_item_id'], train['timestamp']))
    edge_keys = set(zip(edges['raw_user_id'], edges['raw_item_id'], edges['timestamp']))
    assert edge_keys <= train_keys
    assert all(t < m['split']['valid_start_timestamp'] for t in edges['timestamp'])
    assert all(r >= 4 for r in edges['rating'])
    assert set(edges['user_id']) <= {int(x) for x in id_map['user_to_model'].values()}
    assert set(edges['item_id']) <= {int(x) for x in id_map['item_to_model'].values()}
    assert 0 not in edges['user_id'] and 0 not in edges['item_id']
    positives = Counter()
    candidate_items = set()
    observed_by_user = {}
    for user, item, rating in zip(train['raw_user_id'], train['raw_item_id'], train['rating']):
        observed_by_user.setdefault(user, set()).add(item)
        if rating >= 4:
            positives[user] += 1
            candidate_items.add(item)
    assert {int(x) for x in id_map['item_to_model']} == candidate_items
    assert {int(x) for x in id_map['user_to_model']} == {u for u, n in positives.items() if n >= 5}
    assert sorted(id_map['item_to_model'].values()) == list(range(1, len(candidate_items) + 1))
    assert sorted(id_map['user_to_model'].values()) == list(range(1, len(id_map['user_to_model']) + 1))
    assert hashlib.sha256((ROOT / 'data/processed/ml-1m-v1/id_map.json').read_bytes()).hexdigest() == m['id_map_sha256']
    for group in ('valid', 'test'):
        labels = pq.read_table(ROOT / ('data/processed/ml-1m-v1/eval_' + group + '.parquet')).to_pydict()
        for user, item, rating, segment, target in zip(labels['raw_user_id'],
                labels['raw_item_id'], labels['rating'], labels['segment'], labels['is_target']):
            expected = ('WARM' if str(user) in id_map['user_to_model'] and str(item) in id_map['item_to_model']
                        else 'COLD_USER' if str(user) not in id_map['user_to_model'] and str(item) in id_map['item_to_model']
                        else 'COLD_ITEM' if str(user) in id_map['user_to_model'] else 'COLD_BOTH')
            assert segment == expected and target == (rating >= 4)
            assert item not in observed_by_user.get(user, ())
