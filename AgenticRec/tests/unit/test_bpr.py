"""T08 synthetic checks: training inputs are deliberately independent of holdout."""
import random

import pytest
import torch

from agenticrec.models.bpr import BPRMF, TrainNegativeSampler, bpr_loss, load_checkpoint, save_checkpoint
from agenticrec.models.baselines import popularity_scores, random_ranking


def test_sampler_excludes_every_train_rating_but_never_consults_future():
    seen = {1: {1, 2, 3}, 2: {1, 2, 3, 4}}
    sampler = TrainNegativeSampler(5, seen, random.Random(42))
    assert {sampler.sample(1) for _ in range(100)} == {4, 5}
    assert {sampler.sample(2) for _ in range(20)} == {5}
    # Item 4 can be a future positive; it must remain eligible for user 1.
    assert 4 in {sampler.sample(1) for _ in range(100)}
    with pytest.raises(ValueError):
        TrainNegativeSampler(2, {1: {1, 2}}, random.Random(0))


def test_loss_shapes_and_gradient_flow():
    torch.manual_seed(42)
    model = BPRMF(2, 4, 8)
    users = torch.tensor([1, 2])
    positive = torch.tensor([1, 2])
    negative = torch.tensor([3, 4])
    assert model.score(users, positive).shape == (2,)
    loss = bpr_loss(model, users, positive, negative, 1e-4)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.count_nonzero(model.users.weight.grad[1:]) > 0
    assert torch.count_nonzero(model.items.weight.grad[1:]) > 0
    assert torch.count_nonzero(model.users.weight.grad[0]) == 0


def test_tiny_fixture_overfits_and_checkpoint_reload_is_identical(tmp_path):
    torch.manual_seed(7)
    model = BPRMF(1, 3, 8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    users = torch.tensor([1, 1])
    positives = torch.tensor([1, 1])
    negatives = torch.tensor([2, 3])
    initial = bpr_loss(model, users, positives, negatives, 0).item()
    for _ in range(120):
        optimizer.zero_grad()
        loss = bpr_loss(model, users, positives, negatives, 0)
        loss.backward()
        optimizer.step()
    assert loss.item() < initial * 0.1
    assert model.score(torch.tensor([1, 1]), torch.tensor([1, 1]))[0] > model.score(users, negatives).max()
    path = tmp_path / 'best.pt'
    identity = {'id_map_sha256': 'a' * 64, 'split_sha256': 'b' * 64,
                'code_sha256': 'c' * 64, 'seed': 7, 'torch_version': str(torch.__version__)}
    save_checkpoint(path, model, identity)
    restored = load_checkpoint(path, 1, 3, 8, identity)
    torch.testing.assert_close(restored.score(users, negatives), model.score(users, negatives), rtol=0, atol=0)
    with pytest.raises(ValueError, match='identity'):
        load_checkpoint(path, 1, 3, 8, identity | {'split_sha256': 'wrong'})


def test_random_and_popularity_are_deterministic_and_legal():
    a = random_ranking([1, 2, 3, 4], {2}, 3, 42, 1)
    assert a == random_ranking([1, 2, 3, 4], {2}, 3, 42, 1)
    assert len(a) == len(set(a)) == 3 and 2 not in a
    scores = popularity_scores([1, 2, 3], {2: 5, 1: 5})
    assert sorted(scores, key=lambda i: (-scores[i], i)) == [1, 2, 3]
