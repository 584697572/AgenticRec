"""Public-history-only ranking; frozen C1 remains an independent control."""
from collections import defaultdict
from copy import deepcopy
import math

import torch

from .frontier_experiment import FrontierSystem
from .frontier_state import feasible_ids, verify_delivery
from ..retrieval.adaptive import _cf_scores


def constrained_ridge(gram, cross, regularization):
    """min ||Y-ZB||^2 + lambda||B||^2, diag(B)=0; EASE if Z=Y."""
    if not math.isfinite(regularization) or regularization <= 0:
        raise ValueError('Positive finite regularization required')
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1] or cross.shape != gram.shape:
        raise ValueError('Square compatible Gram/cross matrices required')
    if not torch.isfinite(gram).all() or not torch.isfinite(cross).all():
        raise ValueError('Finite training statistics required')
    a = gram.to(torch.float64).clone()
    a.diagonal().add_(regularization)
    chol = torch.linalg.cholesky(a)
    inverse = torch.cholesky_inverse(chol)
    unconstrained = torch.cholesky_solve(cross.to(torch.float64), chol)
    b = unconstrained - inverse * (unconstrained.diagonal() / inverse.diagonal()).unsqueeze(0)
    b.diagonal().zero_()
    # Constrained first-order condition must hold on all off-diagonal entries.
    residual = a @ b - cross.to(torch.float64)
    residual.diagonal().zero_()
    relative = residual.abs().max().item() / max(1., cross.abs().max().item())
    if relative > 1e-8 or not torch.isfinite(b).all():
        raise ValueError('Constrained solver failed numeric verification')
    return b.to(torch.float32), relative


def baseline_sources(base, catalog, state):
    """Exactly the original C1 top-100 lists, including zero content scores."""
    eligible = feasible_ids(catalog, state)
    cf, _ = _cf_scores(base, eligible, state['liked_item_ids'], None, False)
    query = defaultdict(float)
    for seed in state['liked_item_ids']:
        for term, weight in base.content_vectors.get(seed, {}).items():
            query[term] += weight / max(1, len(state['liked_item_ids']))
    for genre in state['include_genres']:
        query['genre:' + genre.casefold()] += 1
    sources = {'popularity': sorted(eligible, key=lambda i: (-base.popularity[i], i))[:100],
               'content': sorted(eligible, key=lambda i: (-sum(query.get(t, 0)*w for t, w in base.content_vectors[i].items()), i))[:100]}
    if cf:
        sources['cf'] = sorted(cf, key=lambda i: (-cf[i], i))[:100]
    return eligible, sources


def fuse(base, eligible, sources, weights, k=10):
    if not weights or any(not math.isfinite(w) or w < 0 for w in weights.values()) or not any(weights.values()):
        raise ValueError('Nonnegative finite weights with a positive route required')
    scores = defaultdict(float)
    for name, ids in sources.items():
        for rank, item in enumerate(ids, 1):
            scores[item] += weights.get(name, 0) / (60 + rank)
    return sorted(eligible, key=lambda i: (-scores[i], -base.popularity[i], i))[:k]


class PreferenceRanker:
    def __init__(self, base, catalog, config, checkpoint=None):
        self.base, self.catalog, self.config = base, catalog, deepcopy(config)
        self.weights = None
        if config.get('model'):
            if checkpoint is None or checkpoint['model'] != config['model']:
                raise ValueError('Missing/mismatched preference checkpoint')
            if checkpoint['regularization'] != config['regularization']:
                raise ValueError('Checkpoint regularization mismatch')
            if checkpoint['item_ids'] != list(range(1, base.catalog.n_items+1)):
                raise ValueError('Checkpoint item order differs from retriever')
            if checkpoint['weights'].shape != (base.catalog.n_items, base.catalog.n_items):
                raise ValueError('Checkpoint shape differs from retriever')
            self.weights = checkpoint['weights']

    def rank(self, state, snapshot=None):
        eligible, sources = snapshot or baseline_sources(self.base, self.catalog, state)
        sources = dict(sources)
        if not state['liked_item_ids']:
            return fuse(self.base, eligible, sources, {'cf':1., 'content':1., 'popularity':1.}, state['k'])
        if self.weights is not None:
            liked = sorted(set(state['liked_item_ids']))
            if any(i not in self.catalog for i in liked):
                raise ValueError('Public history contains unconfirmed identity')
            scores = self.weights[[i-1 for i in liked]].sum(0).tolist()
            sources['learned'] = sorted(eligible, key=lambda i: (-scores[i-1], i))[:100]
        return fuse(self.base, eligible, sources, self.config['weights'], state['k'])


class PreferenceSystem(FrontierSystem):
    """Original C1 parsing/gates, with only the delivered preference order replaced."""
    def __init__(self, journal, catalog, base, models, ranker):
        super().__init__('C1', journal, catalog, base, models)
        self.preference_ranker = ranker

    def execute(self, public, index, delta):
        result = super().execute(public, index, delta)
        ids = self.preference_ranker.rank(self.state)
        verification = verify_delivery(ids, self.catalog, self.state)
        answer = '\n'.join(f"{n}. {self.catalog[i]['title']} ({self.catalog[i]['year']})" for n,i in enumerate(ids,1))
        self.state['previous_item_ids'] = ids
        self.history[-1]['content'] = answer
        result.update(status='OK' if not verification else 'CONSTRAINT_FAILURE', answer=answer,
                      mapped_optimized_ids=ids, raw_ids=[self.catalog[i]['raw_id'] for i in ids],
                      state=deepcopy(self.state), verification_errors=verification)
        return result
