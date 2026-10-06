"""Train-only BPR matrix factorization and checkpoint identity checks."""
import random

import torch
from torch import nn
from torch.nn import functional as F


class BPRMF(nn.Module):
    def __init__(self, n_users, n_items, dim):
        super().__init__()
        if min(n_users, n_items, dim) < 1:
            raise ValueError('Model dimensions must be positive')
        self.users = nn.Embedding(n_users + 1, dim, padding_idx=0)
        self.items = nn.Embedding(n_items + 1, dim, padding_idx=0)
        nn.init.normal_(self.users.weight[1:], std=0.01)
        nn.init.normal_(self.items.weight[1:], std=0.01)
        with torch.no_grad():
            self.users.weight[0].zero_()
            self.items.weight[0].zero_()

    def score(self, user_ids, item_ids):
        if user_ids.shape != item_ids.shape or user_ids.ndim != 1:
            raise ValueError('Expected aligned rank-one user and item tensors')
        if torch.any(user_ids <= 0) or torch.any(item_ids <= 0):
            raise ValueError('Padding/unknown ID cannot be scored')
        return (self.users(user_ids) * self.items(item_ids)).sum(dim=1)


def bpr_loss(model, users, positives, negatives, regularization):
    """Mean softplus(s_neg-s_pos) + lambda * mean per-triple squared L2.

    Every sampled triple contributes (||p_u||²+||q_i||²+||q_j||²)/2;
    both terms are averaged over the batch, so lambda is batch-size invariant.
    """
    if not 0 <= regularization < float('inf'):
        raise ValueError('regularization must be finite and nonnegative')
    p = model.users(users)
    q_pos = model.items(positives)
    q_neg = model.items(negatives)
    positive_score = (p * q_pos).sum(dim=1)
    negative_score = (p * q_neg).sum(dim=1)
    pairwise = F.softplus(negative_score - positive_score).mean()
    regularizer = ((p.square() + q_pos.square() + q_neg.square()).sum(dim=1) / 2).mean()
    return pairwise + regularization * regularizer


class TrainNegativeSampler:
    """Uniform candidates excluding all train-rated items, including low ratings.

    The caller supplies train-only seen sets. No valid/test labels are accepted.
    """
    def __init__(self, n_items, train_seen_by_user, rng: random.Random):
        if n_items < 1:
            raise ValueError('Empty item universe')
        self.n_items = n_items
        self.rng = rng
        self.seen = {int(u): frozenset(items) for u, items in train_seen_by_user.items()}
        if any(any(type(i) is not int or i < 1 or i > n_items for i in s) for s in self.seen.values()):
            raise ValueError('Train history contains an item outside candidate IDs')
        if any(len(s) == n_items for s in self.seen.values()):
            raise ValueError('A user has no train-unrated candidate')

    def sample(self, user_id):
        seen = self.seen[user_id]
        while True:
            item = self.rng.randint(1, self.n_items)
            if item not in seen:
                return item


def save_checkpoint(path, model, identity):
    payload = {'state_dict': model.state_dict(), 'identity': dict(identity),
               'n_users': model.users.num_embeddings - 1,
               'n_items': model.items.num_embeddings - 1,
               'dim': model.users.embedding_dim}
    torch.save(payload, path)


def load_checkpoint(path, n_users, n_items, dim, identity):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload['identity'] != identity or (payload['n_users'], payload['n_items'], payload['dim']) != (n_users, n_items, dim):
        raise ValueError('Checkpoint identity or dimensions differ from frozen training protocol')
    model = BPRMF(n_users, n_items, dim)
    model.load_state_dict(payload['state_dict'], strict=True)
    model.eval()
    return model
