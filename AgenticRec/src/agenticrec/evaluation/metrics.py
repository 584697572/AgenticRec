"""ID-based full-universe ranking metrics from specification section 10.2."""
import math


ZERO = {'recall': 0, 'hit': 0, 'mrr': 0, 'ndcg': 0}
FIELDS = tuple(ZERO)


def _positive_ids(values, label):
    ids = list(values)
    if any(type(value) is not int or value <= 0 for value in ids):
        raise ValueError(label + ' must contain positive integer IDs; padding is forbidden')
    if len(ids) != len(set(ids)):
        raise ValueError(label + ' contains duplicate IDs')
    return ids


def _k(value):
    if type(value) is not int or value <= 0:
        raise ValueError('k must be a positive integer')
    return value


def rank_candidates(scores, candidate_ids, k, seen=()):
    """Return score-descending, ID-ascending top-k over all eligible IDs."""
    _k(k)
    universe = _positive_ids(candidate_ids, 'candidate_ids')
    seen_set = set(_positive_ids(seen, 'seen'))
    if any(i not in universe for i in scores):
        raise ValueError('score for item outside frozen candidate universe')
    eligible = [i for i in universe if i not in seen_set]
    if any(i not in scores or type(scores[i]) not in (int, float)
           or not math.isfinite(scores[i]) for i in eligible):
        raise ValueError('Every eligible item needs one finite numeric score')
    return sorted(eligible, key=lambda i: (-scores[i], i))[:k]


def score_ranking(ranking, relevant, candidate_ids, k, seen=()):
    """Score one user; empty legal output has four zeros, not a dropped row."""
    _k(k)
    universe = set(_positive_ids(candidate_ids, 'candidate_ids'))
    seen_set = set(_positive_ids(seen, 'seen'))
    rel = set(_positive_ids(relevant, 'relevant'))
    if not rel:
        raise ValueError('No-relevance users must be counted separately from main mean')
    if not rel <= universe or rel & seen_set:
        raise ValueError('Relevant IDs must be feasible in the frozen candidate universe')
    ranked = _positive_ids(ranking, 'ranking')
    if any(item not in universe or item in seen_set for item in ranked):
        raise ValueError('Ranking contains unknown or train-seen item')
    top = ranked[:k]
    hits = [index for index, item in enumerate(top, 1) if item in rel]
    if not hits:
        return dict(ZERO)
    dcg = sum(1 / math.log2(rank + 1) for rank in hits)
    idcg = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(rel), k) + 1))
    return {'recall': len(hits) / len(rel), 'hit': 1,
            'mrr': 1 / hits[0], 'ndcg': dcg / idcg}


def evaluate_cohort(user_ids, relevant_by_user, rankings_by_user, candidate_ids, k,
                    seen_by_user=None):
    """Macro-average over a frozen cohort, retaining failed/empty attempts.

    A user with no relevant item is reported and excluded from the main mean.
    A missing, empty or invalid prediction for a relevant user stays in the
    denominator with all-zero metrics and increments failures.
    """
    _k(k)
    users = _positive_ids(user_ids, 'user_ids')
    candidates = _positive_ids(candidate_ids, 'candidate_ids')
    candidate_set = set(candidates)
    seen_by_user = {} if seen_by_user is None else seen_by_user
    totals = {field: 0.0 for field in FIELDS}
    excluded, failures, evaluated = 0, 0, 0
    for user in users:
        rel = set(_positive_ids(relevant_by_user.get(user, ()), 'relevant'))
        if not rel:
            excluded += 1
            continue
        if not rel <= candidate_set:
            raise ValueError('Relevant item outside frozen candidate universe')
        evaluated += 1
        ranking = rankings_by_user.get(user)
        if ranking is None:
            failures += 1
            score = ZERO
        else:
            try:
                score = score_ranking(ranking, rel, candidates, k,
                                      seen=seen_by_user.get(user, ()))
                if not ranking:
                    failures += 1
            except ValueError:
                failures += 1
                score = ZERO
        for field in FIELDS:
            totals[field] += score[field]
    return {'users_total': len(users), 'users_without_relevance': excluded,
            'users_evaluated': evaluated, 'failures': failures,
            **{field: totals[field] / evaluated if evaluated else None for field in FIELDS}}
