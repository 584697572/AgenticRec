"""Independent loss/gradient oracle and validation-only experiment boundary."""
import copy
import json
from pathlib import Path

import pytest
import torch

from agenticrec.lightgcn_diagnostic import validate_plan, validation_only_files
from agenticrec.models.lightgcn import LightGCN, lightgcn_loss, normalized_bipartite_graph


def test_lightgcn_dense_reference_matches_loss_and_every_gradient():
    torch.manual_seed(42)
    graph = normalized_bipartite_graph(2, 3, [(1, 1), (1, 2), (2, 2)])
    model = LightGCN(2, 3, 4, 3, graph)
    users, pos, neg = torch.tensor([1, 2]), torch.tensor([1, 2]), torch.tensor([3, 3])
    actual = lightgcn_loss(model, users, pos, neg, 0.0001)
    actual.backward()
    eu = model.users.weight.detach().clone().requires_grad_()
    ei = model.items.weight.detach().clone().requires_grad_()
    layer = torch.cat([eu[1:], ei[1:]])
    layers = [layer]
    for _ in range(3):
        layer = graph.to_dense() @ layer
        layers.append(layer)
    propagated = torch.stack(layers).mean(0)
    uu, ii = propagated[:2], propagated[2:]
    delta = (uu[users-1] * (ii[neg-1] - ii[pos-1])).sum(1)
    expected = torch.nn.functional.softplus(delta).mean()
    expected += 0.0001 * (eu[users].square().sum() + ei[pos].square().sum()
                          + ei[neg].square().sum()) / (2 * len(users))
    expected.backward()
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=0)
    torch.testing.assert_close(model.users.weight.grad, eu.grad, atol=1e-7, rtol=0)
    torch.testing.assert_close(model.items.weight.grad, ei.grad, atol=1e-7, rtol=0)


def test_plan_denies_test_or_remote_access_and_duplicate_trials():
    path = Path(__file__).resolve().parents[2] / 'configs/lightgcn_diagnostic_20261010.json'
    plan = json.loads(path.read_text())
    validate_plan(plan)
    for key, value in [('test_access', True), ('remote_api_requests', 1)]:
        invalid = copy.deepcopy(plan)
        invalid[key] = value
        with pytest.raises(ValueError):
            validate_plan(invalid)
    invalid = copy.deepcopy(plan)
    invalid['trials'].append(copy.deepcopy(invalid['trials'][0]))
    with pytest.raises(ValueError, match='Duplicate'):
        validate_plan(invalid)


def test_diagnostic_file_guard_blocks_canonical_test_but_restores_open(tmp_path):
    test = tmp_path / 'data/eval_private/ml-1m-v1/test.json'
    test.parent.mkdir(parents=True)
    test.write_text('secret test target')
    valid = test.with_name('valid.json')
    valid.write_text('valid')
    with validation_only_files():
        assert valid.read_text() == 'valid'
        with pytest.raises(PermissionError):
            test.read_text()
        with pytest.raises(PermissionError):
            open(test, 'rb')
    assert test.read_text() == 'secret test target'
