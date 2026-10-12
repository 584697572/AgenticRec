"""LightGCN: sparse normalized bipartite propagation with layer-0 averaging.

Method: He et al., SIGIR 2020, arXiv:2002.02126. Cross-checked against the
authors' PyTorch reference at 947ca2b3b1d2d3545b114145710cb06c4e57b3d2;
this implementation is independent and uses the frozen T06 model ID contract.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def normalized_bipartite_graph(n_users, n_items, train_positive_edges):
    """Return D^-1/2 A D^-1/2 in sparse COO; no self-loops or holdout edges.

    User node = user_id - 1; item node = n_users + item_id - 1. Input must
    come from the train-positive table, whose hash is checked by the trainer.
    """
    if type(n_users) is not int or type(n_items) is not int or min(n_users, n_items) < 1:
        raise ValueError('Graph dimensions must be positive integers')
    edges = list(train_positive_edges)
    if not edges or len(edges) != len(set(edges)):
        raise ValueError('Graph edges must be nonempty and unique')
    if any(type(u) is not int or type(i) is not int or not 1 <= u <= n_users or not 1 <= i <= n_items
           for u, i in edges):
        raise ValueError('Graph contains padding or out-of-range ID')
    user_nodes = torch.tensor([u - 1 for u, _ in edges], dtype=torch.long)
    item_nodes = torch.tensor([n_users + i - 1 for _, i in edges], dtype=torch.long)
    src = torch.cat([user_nodes, item_nodes])
    dst = torch.cat([item_nodes, user_nodes])
    n_nodes = n_users + n_items
    degree = torch.bincount(src, minlength=n_nodes).float()
    weights = (degree[src] * degree[dst]).rsqrt()
    graph = torch.sparse_coo_tensor(torch.stack([src, dst]), weights,
                                    (n_nodes, n_nodes), dtype=torch.float32).coalesce()
    if graph._nnz() != 2 * len(edges) or not torch.isfinite(graph.values()).all():
        raise ValueError('Invalid normalized graph')
    return graph


class LightGCN(nn.Module):
    """Uniform average of ego and 1..L linearly propagated embeddings."""
    def __init__(self, n_users, n_items, dim, n_layers, graph):
        super().__init__()
        if type(n_layers) is not int or n_layers < 1 or min(n_users, n_items, dim) < 1:
            raise ValueError('Invalid LightGCN shape or layer count')
        if graph.layout != torch.sparse_coo or graph.shape != (n_users + n_items,) * 2:
            raise ValueError('Graph shape/layout does not match model IDs')
        self.n_users = n_users
        self.n_items = n_items
        self.dim = dim
        self.n_layers = n_layers
        self.users = nn.Embedding(n_users + 1, dim, padding_idx=0)
        self.items = nn.Embedding(n_items + 1, dim, padding_idx=0)
        nn.init.normal_(self.users.weight[1:], std=0.1)
        nn.init.normal_(self.items.weight[1:], std=0.1)
        with torch.no_grad():
            self.users.weight[0].zero_()
            self.items.weight[0].zero_()
        self.register_buffer('graph', graph.coalesce(), persistent=False)

    def propagate(self):
        layer = torch.cat([self.users.weight[1:], self.items.weight[1:]], dim=0)
        output = layer
        for _ in range(self.n_layers):
            layer = torch.sparse.mm(self.graph, layer)
            output = output + layer
        output = output / (self.n_layers + 1)
        return output[:self.n_users], output[self.n_users:]


def lightgcn_loss(model, user_ids, positive_ids, negative_ids, regularization):
    """Pairwise BPR on propagated scores; batch-normalized L2 on ego weights."""
    if not math.isfinite(regularization) or regularization < 0:
        raise ValueError('Regularization must be finite and nonnegative')
    if user_ids.ndim != 1 or user_ids.shape != positive_ids.shape or user_ids.shape != negative_ids.shape:
        raise ValueError('Expected aligned rank-one pairwise tensors')
    if torch.any(user_ids <= 0) or torch.any(positive_ids <= 0) or torch.any(negative_ids <= 0):
        raise ValueError('Padding ID in training triple')
    users, items = model.propagate()
    u = users[user_ids - 1]
    pos = items[positive_ids - 1]
    neg = items[negative_ids - 1]
    pairwise = F.softplus((u * neg).sum(dim=1) - (u * pos).sum(dim=1)).mean()
    ego_u = model.users(user_ids)
    ego_pos = model.items(positive_ids)
    ego_neg = model.items(negative_ids)
    regularizer = ((ego_u.square() + ego_pos.square() + ego_neg.square()).sum(dim=1) / 2).mean()
    return pairwise + regularization * regularizer


def save_lightgcn_checkpoint(path, model, identity, model_config):
    torch.save({'state_dict': model.state_dict(), 'identity': dict(identity),
                'model_config': dict(model_config),
                'n_users': model.n_users, 'n_items': model.n_items,
                'dim': model.dim, 'n_layers': model.n_layers}, path)


def load_lightgcn_checkpoint(path, graph, n_users, n_items, dim, n_layers, identity, model_config):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    dimensions = (n_users, n_items, dim, n_layers)
    if payload['identity'] != identity or payload['model_config'] != model_config or tuple(payload[key] for key in
            ('n_users', 'n_items', 'dim', 'n_layers')) != dimensions:
        raise ValueError('LightGCN checkpoint identity, config or dimensions differ from frozen train graph')
    model = LightGCN(n_users, n_items, dim, n_layers, graph)
    model.load_state_dict(payload['state_dict'], strict=True)
    model.eval()
    return model
