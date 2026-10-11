"""Experimental weighted LightGCN; preserves the original model unchanged.

Layer weighting is allowed by He et al. (arXiv:2002.02126), not a new algorithm.
The fixed adjacency transpose is cached for mathematically identical backward.
"""
import math

import torch

from .lightgcn import LightGCN


class _FixedMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, graph, transpose, features):
        ctx.save_for_backward(transpose)
        if features.device.type == 'cpu':
            return torch.sparse.mm(graph, features, reduce='sum')
        return torch.sparse.mm(graph, features)

    @staticmethod
    def backward(ctx, gradient):
        (transpose,) = ctx.saved_tensors
        if gradient.device.type == 'cpu':
            return None, None, torch.sparse.mm(transpose, gradient, reduce='sum')
        return None, None, torch.sparse.mm(transpose, gradient)


def validate_weights(weights, layers):
    if not isinstance(weights, (list, tuple)) or len(weights) != layers + 1:
        raise ValueError('Exactly one weight per layer, including ego, is required')
    if any(type(w) not in (int, float) or not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError('Layer weights must be finite nonnegative numbers')
    if not math.isclose(sum(weights), 1.0, rel_tol=0, abs_tol=1e-8):
        raise ValueError('Layer weights must sum to one')
    return tuple(float(w) for w in weights)


class ResidualLightGCN(LightGCN):
    def __init__(self, n_users, n_items, dim, n_layers, graph, layer_weights):
        weights = validate_weights(layer_weights, n_layers)
        if graph.requires_grad:
            raise ValueError('Cached adjacency must be a fixed training graph')
        super().__init__(n_users, n_items, dim, n_layers, graph.clone())
        self.layer_weights = weights
        # PyTorch 2.0's native CSR backward repeatedly converts/transposes the
        # fixed graph. Cache BOTH CSR directions and differentiate only features.
        forward = self.graph.to_sparse_csr()
        transpose = self.graph.transpose(0, 1).coalesce().to_sparse_csr()
        self.register_buffer('graph_forward', forward, persistent=False)
        self.register_buffer('graph_transpose', transpose, persistent=False)

    def propagate(self):
        layer = torch.cat((self.users.weight[1:], self.items.weight[1:]), dim=0)
        # Uniform aggregation retains the original arithmetic ordering exactly.
        uniform = all(w == self.layer_weights[0] for w in self.layer_weights)
        output = layer if uniform else self.layer_weights[0] * layer
        for weight in self.layer_weights[1:]:
            layer = _FixedMultiply.apply(self.graph_forward, self.graph_transpose, layer)
            output = output + (layer if uniform else weight * layer)
        if uniform:
            output = output / (self.n_layers + 1)
        return output[:self.n_users], output[self.n_users:]
