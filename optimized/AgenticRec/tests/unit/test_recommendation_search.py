"""Validation-only search boundaries and independently reloaded checkpoints."""
import copy
import json
from pathlib import Path

import pytest
import torch

from agenticrec import recommendation_search as search


def plan():
    return {'schema_version': 1, 'run_id': 'fixture_search', 'purpose': 'fixture',
            'seed': 42, 'torch_threads': 1, 'top_k': 10,
            'selection_metric': 'valid_ndcg_at_10', 'test_access': False,
            'remote_api_requests': 0, 'trials': [
                {'name': 'gcn', 'model': 'lightgcn', 'layers': 1, 'embedding_dim': 4,
                 'learning_rate': .001, 'regularization': .0001,
                 'batch_size': 2, 'max_epochs': 2, 'patience': 2},
                {'name': 'bpr', 'model': 'bpr', 'layers': 0, 'embedding_dim': 4,
                 'learning_rate': .001, 'regularization': .0001,
                 'batch_size': 2, 'max_epochs': 2, 'patience': 2}]}


@pytest.mark.parametrize('field,value', [('test_access', True), ('remote_api_requests', 1),
                                       ('selection_metric', 'valid_hit_at_10'),
                                       ('run_id', '../escape'), ('seed', True)])
def test_search_rejects_leakage_metric_switch_and_path_escape(field, value):
    invalid = plan()
    invalid[field] = value
    with pytest.raises(ValueError):
        search.validate_plan(invalid)


def test_search_rejects_duplicate_trials_and_invalid_numerics():
    for field, value in [('layers', 0), ('max_epochs', 0),
                         ('learning_rate', float('nan')), ('regularization', -.1)]:
        invalid = plan()
        invalid['trials'][0][field] = value
        with pytest.raises(ValueError):
            search.validate_plan(invalid)
    invalid = plan()
    invalid['trials'].append(copy.deepcopy(invalid['trials'][0]))
    with pytest.raises(ValueError):
        search.validate_plan(invalid)


@pytest.mark.parametrize('model_name', ['gcn', 'bpr'])
def test_fixture_run_reloads_best_and_never_requests_test(tmp_path, monkeypatch, model_name):
    monkeypatch.setattr(search, 'ROOT', tmp_path)
    edges = [(1, 1), (1, 2), (2, 2)]
    seen = {1: {1, 2}, 2: {2}}
    manifest = {'id_map_sha256': 'fixture', 'files': {
        'train_positive.parquet': {'sha256': 'fixture'},
        'split_manifest.json': {'sha256': 'fixture'}}}
    monkeypatch.setattr(search, 'load_train_contract', lambda: (manifest, 2, 3, seen, edges))
    cohort = {'user_ids': [1, 2], 'candidate_ids': [1, 2, 3],
              'relevant_by_user': {'1': [3], '2': []},
              'seen_by_user': {'1': [1, 2], '2': [2]}}
    def loader(split, _path):
        assert split == 'valid'
        return cohort
    monkeypatch.setattr(search, 'load_frozen_cohort', loader)
    monkeypatch.setattr(search, 'digest', lambda _path: 'fixture-sha')
    config = tmp_path / 'plan.json'
    config.write_text(json.dumps(plan()))
    result = search.run(config, model_name)
    assert result['valid_metrics']['hit'] == 1
    assert result['valid_metrics']['users_evaluated'] == 1
    assert result['valid_metrics']['users_without_relevance'] == 1
    assert result['checkpoint_reload_metrics_exact']
    assert result['test_evaluation'] == 'NOT EVALUATED'
    assert result['remote_api_requests'] == 0
    checkpoint = tmp_path / result['checkpoint_path']
    payload = torch.load(checkpoint, weights_only=True)
    assert payload['identity']['seed'] == 42
    assert payload['trial']['model'] == ('lightgcn' if model_name == 'gcn' else 'bpr')
    with pytest.raises(FileExistsError):
        search.run(config, model_name)

