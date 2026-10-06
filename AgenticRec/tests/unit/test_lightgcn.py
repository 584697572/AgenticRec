"""Independent numeric and identity checks for T09 LightGCN."""
import math

import pytest
import torch

from agenticrec.evaluation.metrics import rank_candidates
from agenticrec.models.lightgcn import (LightGCN, lightgcn_loss,
                                        normalized_bipartite_graph,
                                        load_lightgcn_checkpoint,
                                        save_lightgcn_checkpoint)
from agenticrec.training import ROOT, load_train_contract


def test_sparse_graph_matches_hand_weights_and_two_dense_steps():
    # Train positives: u1-i1, u1-i2, u2-i2. The held-out u2-i1 is absent.
    graph = normalized_bipartite_graph(2, 2, [(1, 1), (1, 2), (2, 2)])
    assert graph.layout == torch.sparse_coo
    assert graph.shape == (4, 4)
    assert graph._nnz() == 6
    dense = graph.to_dense()
    half_root = 1 / math.sqrt(2)
    expected = torch.tensor([[0, 0, half_root, 0.5],
                             [0, 0, 0, half_root],
                             [half_root, 0, 0, 0],
                             [0.5, half_root, 0, 0]], dtype=torch.float32)
    torch.testing.assert_close(dense, expected, rtol=0, atol=1e-7)
    ego = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    sparse_1 = torch.sparse.mm(graph, ego)
    sparse_2 = torch.sparse.mm(graph, sparse_1)
    torch.testing.assert_close(sparse_1, expected @ ego, rtol=0, atol=1e-6)
    torch.testing.assert_close(sparse_2, expected @ expected @ ego, rtol=0, atol=1e-6)


def test_layer_zero_included_and_score_topk():
    graph = normalized_bipartite_graph(2, 2, [(1, 1), (1, 2), (2, 2)])
    model = LightGCN(2, 2, 2, 2, graph)
    with torch.no_grad():
        model.users.weight[1:] = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        model.items.weight[1:] = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    users, items = model.propagate()
    ego = torch.cat([model.users.weight[1:], model.items.weight[1:]])
    dense = graph.to_dense()
    expected = (ego + dense @ ego + dense @ dense @ ego) / 3
    torch.testing.assert_close(torch.cat([users, items]), expected, rtol=0, atol=1e-6)
    scores = (users[0] @ items.T).tolist()
    ranking = rank_candidates(dict(zip([1, 2], scores)), [1, 2], 2, seen=[1])
    assert ranking == [2]


def test_graph_rejects_duplicates_padding_and_out_of_range():
    with pytest.raises(ValueError):
        normalized_bipartite_graph(2, 2, [(1, 1), (1, 1)])
    with pytest.raises(ValueError):
        normalized_bipartite_graph(2, 2, [(0, 1)])
    with pytest.raises(ValueError):
        normalized_bipartite_graph(2, 2, [(2, 3)])


def test_bpr_gradient_and_checkpoint_identity(tmp_path):
    torch.manual_seed(42)
    graph = normalized_bipartite_graph(2, 3, [(1, 1), (1, 2), (2, 2)])
    model = LightGCN(2, 3, 4, 2, graph)
    users = torch.tensor([1, 2])
    pos = torch.tensor([1, 2])
    neg = torch.tensor([3, 3])
    loss = lightgcn_loss(model, users, pos, neg, 1e-4)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.count_nonzero(model.users.weight.grad[1:]) > 0
    assert torch.count_nonzero(model.items.weight.grad[1:]) > 0
    identity = {'id_map_sha256': 'a' * 64, 'split_sha256': 'b' * 64,
                'train_positive_sha256': 'c' * 64, 'seed': 42}
    config = {'embedding_dim': 4, 'layers_to_try': [1, 2, 3]}
    path = tmp_path / 'lightgcn.pt'
    save_lightgcn_checkpoint(path, model, identity, config)
    restored = load_lightgcn_checkpoint(path, graph, 2, 3, 4, 2, identity, config)
    original = model.propagate()
    reloaded = restored.propagate()
    for a, b in zip(original, reloaded):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    with pytest.raises(ValueError, match='identity'):
        load_lightgcn_checkpoint(path, graph, 2, 3, 4, 2,
                                 identity | {'train_positive_sha256': 'wrong'}, config)
    with pytest.raises(ValueError, match='config'):
        load_lightgcn_checkpoint(path, graph, 2, 3, 4, 2, identity,
                                 config | {'embedding_dim': 8})


def test_real_graph_has_no_valid_or_test_edges():
    """Audit-only holdout read; the trainer itself never uses these rows."""
    import pyarrow.parquet as pq

    if not (ROOT / 'artifacts/data_manifest.json').exists():
        pytest.skip('Frozen MovieLens1M train protocol is not installed')
    manifest, n_users, n_items, _seen, train_edges = load_train_contract()
    graph = normalized_bipartite_graph(n_users, n_items, train_edges)
    assert len(train_edges) == manifest['files']['train_positive.parquet']['rows']
    assert graph._nnz() == 2 * len(train_edges)
    train_pairs = set(train_edges)
    for split in ('valid', 'test'):
        path = ROOT / 'data/processed/ml-1m-v1' / ('eval_' + split + '.parquet')
        table = pq.read_table(path, columns=['user_id', 'item_id'])
        pairs = set(zip(table['user_id'].to_pylist(), table['item_id'].to_pylist()))
        assert not train_pairs.intersection(pairs)
