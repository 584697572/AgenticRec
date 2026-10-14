"""Fixed-pool residual ranking; inference receives public history only."""
import torch
from torch import nn

FEATURE_NAMES = ('popularity', 'cf_mean', 'cf_max', 'content', 'co_max',
                 'genre_affinity', 'history_length', 'seed_pop_mean',
                 'seed_pop_min', 'supported_fraction', 'sequence_score',
                 'sequence_rank', 'cf_pool_rank', 'content_pool_rank', 'pop_pool_rank')


def percentile(values, ids):
    order = sorted(range(len(ids)), key=lambda n: (-float(values[n]), ids[n]))
    ranks = torch.empty(len(ids))
    for n, index in enumerate(order):
        ranks[index] = 1 - n / max(1, len(ids)-1)
    return ranks


def pool_features(builder, liked, ids, scores):
    if not ids or len(ids) != len(set(ids)) or set(ids) & set(liked):
        raise ValueError('Unique non-history fixed pool required')
    ix = torch.tensor(ids)-1
    cf, content, _, _ = builder.context(liked)
    extra = torch.stack((scores[ix].float(), percentile(scores[ix], ids),
                         percentile(cf[ix], ids), percentile(content[ix], ids),
                         percentile(builder.index.popularity[ix], ids)), 1)
    x = torch.cat((builder.features(liked, ids), extra), 1)
    if not torch.isfinite(x).all():
        raise ValueError('Nonfinite public ranking feature')
    return x


class SequenceResidual(nn.Module):
    def __init__(self, means, scales, linear=False):
        super().__init__()
        self.register_buffer('means', means)
        self.register_buffer('scales', scales)
        self.net = nn.Sequential(nn.Linear(len(means), 1)) if linear else nn.Sequential(
            nn.Linear(len(means), 32), nn.ReLU(), nn.Dropout(.1), nn.Linear(32, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def correction(self, x):
        return torch.tanh(self.net((x-self.means)/self.scales).squeeze(-1))

    def forward(self, x, base, strength=1.):
        if strength < 0:
            raise ValueError('Nonnegative residual strength required')
        return base + strength*self.correction(x)


def boundary_loss(scores, positive, k=10, margin=.1, temperature=.2):
    """Surrogate for at least one observed positive above the kth nonpositive.

    Nonpositives mean unobserved future positives, not verified dislikes.
    Ties are resolved outside this loss by item ID. This is not exact Hit@k.
    """
    if scores.shape != positive.shape or positive.dtype != torch.bool:
        raise ValueError('Compatible boolean targets required')
    if not torch.isfinite(scores).all() or not positive.any(1).all():
        raise ValueError('Finite scores and nonempty positives required')
    if k <= 0 or temperature <= 0 or not (~positive).sum(1).ge(k).all():
        raise ValueError('At least k observed nonpositives required')
    best = scores.masked_fill(~positive, -float('inf')).max(1).values
    threshold = scores.masked_fill(positive, -float('inf')).topk(k, dim=1).values[:, -1]
    return torch.nn.functional.softplus((threshold+margin-best)/temperature)*temperature


def mass_loss(scores, positive):
    if scores.shape != positive.shape or not positive.any(1).all():
        raise ValueError('Nonempty positive candidates required')
    return torch.logsumexp(scores, 1)-torch.logsumexp(scores.masked_fill(~positive, -float('inf')), 1)


def ranked_pool(ids, scores, k=10):
    if len(ids) != len(scores) or len(ids) != len(set(ids)) or not torch.isfinite(scores).all():
        raise ValueError('Finite unique fixed candidate pool required')
    values = scores.tolist()
    return sorted(ids, key=lambda i: (-values[ids.index(i)], i))[:k]
