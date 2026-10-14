import pytest
import torch

from agenticrec.evaluation.top100_models import SequenceResidual, boundary_loss, ranked_pool, percentile


def test_zero_residual_preserves_original_scores_and_ties():
    model = SequenceResidual(torch.zeros(15), torch.ones(15))
    x = torch.randn(100, 15); base = torch.randn(100)
    assert torch.equal(model(x, base), base)
    with torch.no_grad(): model.net[-1].weight.fill_(10)
    assert torch.equal(model(x, base, 0), base)
    assert (model(x, base, .3)-base).abs().max() <= .300001
    assert ranked_pool([3, 1, 2], torch.tensor([1., 1., 0.])) == [1, 3, 2]


def test_boundary_compares_best_positive_to_tenth_nonpositive():
    score = torch.arange(12, dtype=torch.float32)[None].requires_grad_()
    pos = torch.zeros_like(score, dtype=torch.bool); pos[0, 0] = True
    loss = boundary_loss(score, pos); loss.sum().backward()
    # Nonpositive scores 11..1: tenth is 2, not the maximum (11).
    assert score.grad[0, 0] < 0 and score.grad[0, 2] > 0
    assert score.grad[0, 11] == 0
    raised = score.detach().clone(); raised[0, 0] = 3
    assert boundary_loss(raised, pos) < loss.detach()
    assert 0 in ranked_pool(list(range(12)), raised[0], 10)


def test_boundary_handles_multiple_positive_targets_and_rejects_bad_groups():
    scores = torch.arange(13.)[None].requires_grad_(); pos = torch.zeros_like(scores, dtype=torch.bool)
    pos[0, [0, 1]] = True
    boundary_loss(scores, pos).sum().backward()
    assert scores.grad[0, 1] < 0 and scores.grad[0, 0] == 0
    with pytest.raises(ValueError): boundary_loss(scores, torch.zeros_like(pos))
    with pytest.raises(ValueError): boundary_loss(scores, torch.ones_like(pos))
    with pytest.raises(ValueError): boundary_loss(scores*float('nan'), pos)


def test_pool_ties_have_deterministic_item_id_order_without_insertion():
    assert percentile(torch.tensor([2., 2., 1.]), [4, 2, 9]).tolist() == [.5, 1., 0.]
    ids = [100, 7, 4]; ordered = ranked_pool(ids, torch.tensor([.2, .9, .4]))
    assert ordered == [7, 4, 100] and set(ordered) == set(ids)
    with pytest.raises(ValueError): ranked_pool([1, 1], torch.ones(2))
