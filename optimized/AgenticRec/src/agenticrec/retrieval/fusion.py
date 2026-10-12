"""Reciprocal-rank fusion without mixing incompatible raw score scales."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class FusionResult:
    candidate_ids: list[int]
    rrf_scores: dict[int, float]
    source_ranks: dict[int, dict[str, int]]


def reciprocal_rank_fusion(rankings, *, universe, per_route=100, union_cap=200, rank_constant=60):
    """Fuse one-based ranks; unknown and duplicate IDs fail closed."""
    if (type(per_route) is not int or per_route <= 0
            or type(union_cap) is not int or union_cap <= 0
            or type(rank_constant) is not int or rank_constant <= 0):
        raise ValueError("RRF limits and rank constant must be positive integers")
    universe = list(universe)
    if any(type(i) is not int or i <= 0 for i in universe) or len(universe) != len(set(universe)):
        raise ValueError("Universe needs positive model item IDs")
    allowed = set(universe)
    if not isinstance(rankings, dict) or any(type(name) is not str or not name for name in rankings):
        raise ValueError("Rankings need named sources")
    scores, provenance = {}, {}
    for source, ranking in rankings.items():
        ranked = list(ranking)
        if (any(type(i) is not int or i not in allowed for i in ranked)
                or len(ranked) != len(set(ranked))):
            raise ValueError("Source ranking contains duplicate or unverified IDs")
        for rank, item_id in enumerate(ranked[:per_route], 1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (rank_constant + rank)
            provenance.setdefault(item_id, {})[source] = rank
    ordered = sorted(scores, key=lambda item_id: (-scores[item_id], item_id))[:union_cap]
    if any(not math.isfinite(scores[i]) for i in ordered):
        raise ValueError("Non-finite fusion score")
    return FusionResult(ordered, {i: scores[i] for i in ordered},
                        {i: provenance[i] for i in ordered})
