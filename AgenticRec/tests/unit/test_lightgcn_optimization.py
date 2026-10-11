"""Independent numerical and no-leakage oracles for the follow-up study."""
import random
import json

import pytest
import torch

from agenticrec.models.lightgcn import LightGCN, lightgcn_loss, normalized_bipartite_graph
from agenticrec.models.lightgcn_residual import ResidualLightGCN
from agenticrec.lightgcn_optimization import epoch_pairs, subset_cohort, validate_trial


@pytest.mark.parametrize('weights', [(0.5, 0.5), (0.8, 0.2), (1.0, 0.0)])
def test_residual_sparse_loss_and_gradients_match_independent_dense_oracle(weights):
    torch.manual_seed(42)
    graph = normalized_bipartite_graph(2, 3, [(1, 1), (1, 2), (2, 2)])
    model = ResidualLightGCN(2, 3, 4, 1, graph, weights)
    users, pos, neg = torch.tensor([1, 2]), torch.tensor([1, 2]), torch.tensor([3, 3])
    loss = lightgcn_loss(model, users, pos, neg, 0.0001)
    loss.backward()
    eu = model.users.weight.detach().clone().requires_grad_()
    ei = model.items.weight.detach().clone().requires_grad_()
    ego = torch.cat((eu[1:], ei[1:]))
    out = weights[0] * ego + weights[1] * (graph.to_dense() @ ego)
    uu, ii = out[:2], out[2:]
    expected = torch.nn.functional.softplus((uu[users-1] * (ii[neg-1]-ii[pos-1])).sum(1)).mean()
    expected += 0.0001 * (eu[users].square().sum()+ei[pos].square().sum()+ei[neg].square().sum())/(2*len(users))
    expected.backward()
    torch.testing.assert_close(loss, expected, atol=1e-7, rtol=0)
    torch.testing.assert_close(model.users.weight.grad, eu.grad, atol=1e-7, rtol=0)
    torch.testing.assert_close(model.items.weight.grad, ei.grad, atol=1e-7, rtol=0)
    assert model.users.weight.grad[0].count_nonzero() == 0
    assert model.items.weight.grad[0].count_nonzero() == 0


@pytest.mark.parametrize('layers', [1, 2, 3])
def test_uniform_weights_preserve_original_propagation_and_backward(layers):
    graph = normalized_bipartite_graph(2, 3, [(1, 1), (1, 2), (2, 2)])
    original = LightGCN(2, 3, 4, layers, graph)
    new = ResidualLightGCN(2, 3, 4, layers, graph, [1/(layers+1)]*(layers+1))
    new.load_state_dict(original.state_dict())
    for model in (original, new):
        ue, ie = model.propagate()
        (ue.square().sum()+ie.square().sum()).backward()
    for a, b in zip(original.propagate(), new.propagate()):
        torch.testing.assert_close(a, b, atol=1e-7, rtol=0)
    for a, b in zip(original.parameters(), new.parameters()):
        torch.testing.assert_close(a.grad, b.grad, atol=1e-7, rtol=0)


@pytest.mark.parametrize('weights', [[0.5], [-0.1, 1.1], [0.0, 0.0], [True, 0], [float('nan'), 0.5]])
def test_residual_weights_reject_invalid_or_ambiguous_normalization(weights):
    graph = normalized_bipartite_graph(2, 3, [(1, 1), (2, 2)])
    with pytest.raises(ValueError):
        ResidualLightGCN(2, 3, 4, 1, graph, weights)


def test_cached_transpose_handles_non_symmetric_fixed_graph():
    graph = torch.tensor([[0.0, 2.0], [3.0, 0.0]]).to_sparse()
    model = ResidualLightGCN(1, 1, 2, 1, graph, [0.8, 0.2])
    ego = torch.cat((model.users.weight[1:], model.items.weight[1:]))
    expected = (0.8*ego+0.2*(graph.to_dense()@ego)).sum()
    expected_grads = torch.autograd.grad(expected, tuple(model.parameters()))
    sum(v.sum() for v in model.propagate()).backward()
    for actual, wanted in zip(model.parameters(), expected_grads):
        torch.testing.assert_close(actual.grad, wanted)


def test_user_uniform_pairs_remove_interaction_count_bias_and_are_reproducible():
    edges = [(1, i) for i in range(1, 101)] + [(2, 101)]
    a = epoch_pairs(edges, 'uniform_user', random.Random(42), count=10000)
    assert a == epoch_pairs(edges, 'uniform_user', random.Random(42), count=10000)
    assert 0.47 < sum(u == 1 for u, _ in a)/len(a) < 0.53
    assert set(a) <= set(edges)
    shuffled = epoch_pairs(edges, 'edge', random.Random(42))
    assert len(shuffled) == len(edges) and set(shuffled) == set(edges)
    assert sum(u == 1 for u, _ in shuffled) == 100
    assert edges[0] == (1, 1)


def test_subset_cohort_does_not_expose_nonselection_users():
    cohort = {'user_ids': [1, 2, 3], 'candidate_ids': [1, 2],
              'relevant_by_user': {'1': [1], '2': [2], '3': []},
              'seen_by_user': {'1': [], '2': [], '3': []}}
    selected = subset_cohort(cohort, [1])
    assert selected['user_ids'] == [1]
    assert set(selected['relevant_by_user']) == {'1'}
    assert set(selected['seen_by_user']) == {'1'}
    with pytest.raises(ValueError):
        subset_cohort(cohort, [99])


def test_trial_validation_rejects_test_adaptive_and_unknown_configuration():
    trial = dict(name='gcn_edge_uniform', model='lightgcn', layers=1, embedding_dim=64,
                 layer_weights=[0.5, 0.5], sampling='edge', learning_rate=0.001,
                 regularization=0.0001, batch_size=2048, max_epochs=32, patience=8)
    validate_trial(trial)
    for key, value in [('sampling', 'test_hard_negatives'), ('layers', True), ('learning_rate', float('nan'))]:
        bad = {**trial, key: value}
        with pytest.raises(ValueError):
            validate_trial(bad)
    with pytest.raises(ValueError):
        validate_trial({**trial, 'test_access': True})


def test_tiny_train_reload_and_audit_isolation_without_real_data(tmp_path, monkeypatch):
    import agenticrec.lightgcn_optimization as study
    root = tmp_path
    (root/'artifacts').mkdir()
    (root/'artifacts/data_manifest.json').write_text('fixture manifest')
    valid = root/'data/eval_private/ml-1m-v1/valid.json'
    valid.parent.mkdir(parents=True)
    valid.write_text('fixture development labels')
    trial = dict(name='tiny', model='lightgcn', layers=1, embedding_dim=4,
                 layer_weights=[0.8, 0.2], sampling='uniform_user', learning_rate=0.001,
                 regularization=0.0001, batch_size=2, max_epochs=2, patience=2)
    plan = {'seeds': [7, 42, 2026], 'torch_threads': 1, 'trials': [trial]}
    path = root/'plan.json'
    path.write_text(json.dumps(plan))
    manifest = {'id_map_sha256': 'fixture map', 'files': {
        'train_positive.parquet': {'sha256': 'fixture train'}}}
    contract = (manifest, 2, 3, {1: {1}, 2: {2}}, [(1, 1), (2, 2)])
    cohort = {'user_ids': [1, 2], 'candidate_ids': [1, 2, 3],
              'relevant_by_user': {'1': [2], '2': [3]},
              'seen_by_user': {'1': [1], '2': [2]}}
    monkeypatch.setattr(study, 'ROOT', root)
    monkeypatch.setattr(study, 'BASE', root/'runs')
    monkeypatch.setattr(study, 'plan_and_groups', lambda _: (plan, {'fit': [1], 'select': [], 'audit': [2]}))
    monkeypatch.setattr(study, 'load_train_contract', lambda: contract)
    def only_valid(split, _):
        assert split == 'valid'
        return cohort
    monkeypatch.setattr(study, 'load_frozen_cohort', only_valid)
    monkeypatch.setattr(study.subprocess, 'check_output',
                        lambda args, **kwargs: 'fixture commit\n' if 'rev-parse' in args else '')
    result = study.train(path, 'tiny', 42)
    directory = study.BASE/'seed_42/tiny'
    assert result['checkpoint_reload_exact'] is True
    assert result['audit_evaluation'] == result['test_evaluation'] == 'NOT EVALUATED'
    assert set(json.loads((directory/'selection_predictions.json').read_text())) == {'1'}
    model, loaded = study.load_verified(directory, trial, contract, study.digest(path), 42)
    assert loaded['checkpoint_sha256'] == result['checkpoint_sha256']
    with pytest.raises(ValueError, match='identity'):
        study.load_verified(directory, {**trial, 'layer_weights': [0.5, 0.5]}, contract, study.digest(path), 42)
    with pytest.raises(FileExistsError):
        study.train(path, 'tiny', 42)
    with pytest.raises(FileNotFoundError):
        study.train(path, 'tiny', 7)
    (directory/'best.pt').write_bytes(b'corrupted fixture checkpoint')
    with pytest.raises(ValueError, match='SHA'):
        study.load_verified(directory, trial, contract, study.digest(path), 42)
