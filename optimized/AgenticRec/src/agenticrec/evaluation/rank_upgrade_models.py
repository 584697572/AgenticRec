"""Offline public-history ranking components; labels are training-only inputs."""
from collections import Counter, defaultdict
import math

import torch
from torch import nn


class CFIndex:
    def __init__(self, n_items, edges, *, iuf=False):
        users = sorted({u for u, _ in edges})
        remap = {u: n for n, u in enumerate(users)}
        x = torch.zeros((len(users), n_items), dtype=torch.float32)
        if len(edges) != len(set(edges)):
            raise ValueError('Duplicate graph edge')
        x[[remap[u] for u, i in edges], [i-1 for u, i in edges]] = 1
        self.popularity = x.sum(0).to(torch.float64)
        if iuf:
            x *= (1 / torch.log1p(x.sum(1))).sqrt().unsqueeze(1)
        self.co = x.T @ x
        self.norm = self.co.diagonal().to(torch.float64).sqrt()
        self.neighbor_cache = {}

    def cosine(self, liked):
        if not liked:
            return torch.zeros(len(self.popularity), dtype=torch.float64)
        ids = torch.tensor([i-1 for i in liked])
        denom = self.norm[ids, None] * self.norm[None, :]
        values = self.co[ids].to(torch.float64) / denom.clamp_min(1e-12)
        values[denom == 0] = 0
        return values.mean(0)

    def neighbors(self, liked, k, shrink):
        result = torch.zeros(len(self.popularity), dtype=torch.float64)
        for seed in liked:
            key = (seed, shrink)
            if key not in self.neighbor_cache:
                denominator = self.norm[seed-1] * self.norm + shrink
                row = self.co[seed-1].to(torch.float64) / denominator.clamp_min(1e-12)
                row[seed-1] = 0
                ids = torch.argsort(row, descending=True, stable=True)[:200]
                self.neighbor_cache[key] = (ids, row[ids])
            ids, values = self.neighbor_cache[key]
            result[ids[:k]] += values[:k] / max(1, len(liked))
        return result


def zscore(values):
    return (values-values.mean()) / values.std(unbiased=False).clamp_min(1e-6)


def mixed_scores(values, popularity, alpha):
    return alpha*zscore(values.to(torch.float64)) + (1-alpha)*zscore(torch.log1p(popularity))


def ordered_ids(scores, liked, k=10):
    if not torch.isfinite(scores).all():
        raise ValueError('Nonfinite recommendation scores')
    values = scores.clone()
    values[[i-1 for i in liked]] = -float('inf')
    order = torch.argsort(values, descending=True, stable=True)
    order = order[torch.isfinite(values[order])]
    return (order[:k]+1).tolist()


class FeatureBuilder:
    """Train co-occurrence, metadata and explicit public likes; no user IDs."""
    def __init__(self, index, vectors, genres):
        self.index, self.vectors, self.genres = index, vectors, genres
        self.pop = torch.log1p(index.popularity)
        self.pop /= self.pop.max().clamp_min(1)
        self.cache = {}

    def context(self, liked):
        key = tuple(liked)
        if key in self.cache:
            return self.cache[key]
        cf = self.index.cosine(liked)
        query = defaultdict(float)
        for seed in liked:
            for term, w in self.vectors[seed].items():
                query[term] += w/len(liked)
        content = torch.tensor([sum(query.get(t, 0)*w for t, w in self.vectors[i].items())
                                for i in range(1, len(self.pop)+1)], dtype=torch.float64)
        seen = set(liked)
        routes = [ordered_ids(v, liked, 100) for v in (cf, content, self.pop)]
        union = set(i for route in routes for i in route)
        # Bounded pool, no target insertion. Keep every route's first 100.
        candidates = sorted(union)
        self.cache[key] = (cf, content, routes, candidates)
        return self.cache[key]

    def features(self, liked, candidates):
        cf, content, _, _ = self.context(liked)
        seed_counts = self.index.popularity[[i-1 for i in liked]]
        counts = self.index.co[[i-1 for i in liked]][:, [i-1 for i in candidates]].to(torch.float64)
        denom = self.index.norm[[i-1 for i in liked], None]*self.index.norm[[i-1 for i in candidates]][None, :]
        cosine = counts/denom.clamp_min(1e-12)
        cosine[denom == 0] = 0
        affinity = Counter(g for i in liked for g in self.genres[i])
        rows = []
        for n, item in enumerate(candidates):
            rows.append([float(self.pop[item-1]), float(cf[item-1]), float(cosine[:, n].max()),
                         float(content[item-1]), math.log1p(float(counts[:, n].max()))/10,
                         sum(affinity[g] for g in self.genres[item])/len(liked),
                         len(liked)/10, math.log1p(float(seed_counts.mean()))/10,
                         math.log1p(float(seed_counts.min()))/10,
                         float((counts[:, n] > 0).sum())/len(liked)])
        return torch.tensor(rows, dtype=torch.float32)


class PopularResidual(nn.Module):
    def __init__(self, means, scales):
        super().__init__()
        self.register_buffer('means', means)
        self.register_buffer('scales', scales)
        self.net = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Dropout(.1), nn.Linear(32, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, alpha=1.):
        standardized = (x-self.means)/self.scales
        return standardized[..., 0] + alpha*self.net(standardized).squeeze(-1)


class ShortSequence(nn.Module):
    """SASRec-style causal encoder; multi-positive objective differs from SASRec."""
    def __init__(self, n_items, dim=32):
        super().__init__()
        self.items = nn.Embedding(n_items+1, dim, padding_idx=0)
        self.positions = nn.Embedding(10, dim)
        block = nn.TransformerEncoderLayer(dim, 2, dim*2, dropout=.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(block, 1, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        nn.init.normal_(self.items.weight, std=.02)
        with torch.no_grad(): self.items.weight[0].zero_()

    def encode(self, history):
        pad = history == 0
        if (history < 0).any() or (history >= self.items.num_embeddings).any() or pad.all(1).any():
            raise ValueError('Invalid or empty public history')
        positions = torch.arange(history.shape[1], device=history.device)
        value = self.items(history)*math.sqrt(self.items.embedding_dim) + self.positions(positions)
        mask = torch.triu(torch.ones(history.shape[1], history.shape[1], dtype=torch.bool), diagonal=1)
        value = self.norm(self.encoder(value, mask=mask, src_key_padding_mask=pad))
        lengths = (~pad).sum(1)
        return value[torch.arange(len(history)), lengths-1]

    def forward(self, history):
        return self.encode(history) @ self.items.weight[1:].T


class GraphSession(nn.Module):
    """Only public liked item embeddings form the session representation."""
    def __init__(self, item_vectors):
        super().__init__()
        vectors = nn.functional.normalize(item_vectors, dim=1)
        self.register_buffer('vectors', torch.cat([torch.zeros_like(vectors[:1]), vectors]))
        dim = vectors.shape[1]
        self.adapter = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, dim))
        nn.init.zeros_(self.adapter[-1].weight); nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, history, adapted=True):
        mask = history != 0
        if (history < 0).any() or (history >= len(self.vectors)).any() or not mask.any(1).all():
            raise ValueError('Invalid or empty public history')
        pooled = self.vectors[history].sum(1)/mask.sum(1, keepdim=True)
        session = pooled+self.adapter(pooled) if adapted else pooled
        return nn.functional.normalize(session, dim=1) @ self.vectors[1:].T / .15


def padded_histories(examples):
    result = torch.zeros((len(examples), 10), dtype=torch.long)
    for n, row in enumerate(examples):
        liked = row['liked_item_ids']
        if not 1 <= len(liked) <= 10 or len(liked) != len(set(liked)):
            raise ValueError('History must contain 1..10 unique public IDs')
        result[n, :len(liked)] = torch.tensor(liked)
    return result


def multi_positive_loss(logits, targets, history):
    if logits.shape != targets.shape or not targets.sum(1).gt(0).all():
        raise ValueError('Every training group needs positive supervision')
    masked = logits.clone()
    rows = torch.arange(len(history))[:, None].expand_as(history)
    valid = history > 0
    if targets[rows[valid], history[valid]-1].any():
        raise ValueError('Input and future positive target overlap')
    masked[rows[valid], history[valid]-1] = -1e9
    return -(nn.functional.log_softmax(masked, dim=1)*targets/targets.sum(1, keepdim=True)).sum(1).mean()
