from collections import defaultdict

import pytest
import torch

from agenticrec.evaluation.rank_upgrade_models import (CFIndex, FeatureBuilder, PopularResidual,
    ShortSequence, GraphSession, padded_histories, multi_positive_loss, ordered_ids, mixed_scores)


def test_cf_matrix_matches_independent_user_intersections_and_iuf_weighting():
    edges = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (3, 2), (3, 4)]
    users = defaultdict(set)
    for u, i in edges: users[i].add(u)
    plain = CFIndex(5, edges)
    actual = plain.cosine([1, 3])
    for i in range(1, 6):
        expected = sum(len(users[i]&users[j])/(len(users[i])*len(users[j]))**.5
                       if users[i] else 0 for j in [1, 3])/2
        assert float(actual[i-1]) == pytest.approx(expected)
    weighted = CFIndex(5, edges, iuf=True)
    degree = {u: sum(v == u for v, i in edges) for u, i in edges}
    for i in range(1, 6):
        for j in range(1, 6):
            expected = sum(1/torch.log1p(torch.tensor(float(degree[u]))) for u in users[i]&users[j])
            assert float(weighted.co[i-1, j-1]) == pytest.approx(float(expected), abs=1e-6)


def test_neighbor_shrink_does_not_read_labels_and_keeps_only_top_k():
    index = CFIndex(5, [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (3, 4)])
    result = index.neighbors([1], 1, 10)
    assert result[0] == 0 and result[1] > 0 and int(result.gt(0).sum()) == 1
    assert index.neighbors([1], 1, 50)[1] < result[1]


def test_ordered_ids_never_returns_seen_even_if_requested_k_exceeds_catalog():
    assert ordered_ids(torch.tensor([1., 1., 0.]), [1], 100) == [2, 3]
    with pytest.raises(ValueError): ordered_ids(torch.tensor([float('nan')]), [], 1)


def test_zero_residual_exactly_preserves_popular_order():
    torch.manual_seed(4)
    x = torch.randn(15, 10)
    model = PopularResidual(torch.zeros(10), torch.ones(10)).eval()
    assert torch.equal(model(x, .25), x[:, 0])
    target = torch.zeros(1, 15); target[0, 4] = 1
    loss = -(torch.log_softmax(model(x).unsqueeze(0), 1)*target).sum()
    loss.backward()
    assert model.net[-1].weight.grad.abs().sum() > 0


def test_sequence_encoder_is_causal_and_uses_history_order():
    torch.manual_seed(19); model = ShortSequence(20).eval()
    first = []
    hook = model.encoder.register_forward_hook(lambda m, args, out: first.append(out[:, 0].detach().clone()))
    a = torch.tensor([[1, 2, 3, 0, 0, 0, 0, 0, 0, 0]])
    b = torch.tensor([[1, 5, 4, 0, 0, 0, 0, 0, 0, 0]])
    with torch.no_grad(): va, vb = model(a), model(b)
    hook.remove()
    assert torch.allclose(first[0], first[1], atol=1e-6)
    assert not torch.allclose(va, vb)
    reverse = torch.tensor([[3, 2, 1, 0, 0, 0, 0, 0, 0, 0]])
    with torch.no_grad(): assert not torch.allclose(va, model(reverse))


def test_anonymous_graph_readout_uses_only_public_liked_item_vectors():
    torch.manual_seed(9); vectors = torch.randn(20, 32)
    model = GraphSession(vectors).eval()
    a = {'liked_item_ids': [1, 4, 3], 'user_id': 999, 'relevant_item_ids': [20]}
    b = dict(a, user_id=1, relevant_item_ids=[2])
    with torch.no_grad():
        assert torch.equal(model(padded_histories([a])), model(padded_histories([b])))
        assert torch.equal(model(padded_histories([a]), False), model(padded_histories([a]), True))
    with pytest.raises(ValueError): model(torch.zeros((1, 10), dtype=torch.long))


def test_multitarget_group_loss_and_gradient_against_direct_dense_oracle():
    logits = torch.tensor([[.1, .2, .3, .4]], requires_grad=True)
    targets = torch.tensor([[0., 1., 1., 0.]])
    loss = multi_positive_loss(logits, targets, torch.tensor([[1, 0]]))
    expected = -torch.log_softmax(logits[0, 1:], 0)[:2].mean()
    assert torch.allclose(loss, expected)
    actual_grad = torch.autograd.grad(loss, logits, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, logits)[0]
    assert torch.allclose(actual_grad, expected_grad)
    with pytest.raises(ValueError): multi_positive_loss(logits, targets, torch.tensor([[2, 0]]))


def test_popularity_mixture_zero_alpha_preserves_popular_ranking():
    learned = torch.tensor([4., 0., 7., 2.]); pop = torch.tensor([3., 5., 1., 9.])
    assert ordered_ids(mixed_scores(learned, pop, 0), [], 4) == [4, 2, 1, 3]


def test_training_public_histories_reject_duplicates_and_excess_length():
    with pytest.raises(ValueError): padded_histories([{'liked_item_ids': [1]*3}])
    with pytest.raises(ValueError): padded_histories([{'liked_item_ids': list(range(1, 12))}])
