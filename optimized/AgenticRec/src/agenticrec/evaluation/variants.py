"""Explicit recommendation ablations using the identical frozen ID/history space."""

from ..adapters.model import RouteResult
from ..pipeline import FixedRecommendationPipeline
from ..retrieval.sources import CandidateRetriever


class AblationScorer:
    def __init__(self, base, retriever, mode):
        self.base, self.retriever, self.mode = base, retriever, mode
        self.session = base.session

    def rank(self, candidate_ids, k, *, user_id=None, history_authorized=False,
             liked_ids=(), disliked_ids=()):
        if self.mode == "no_content" and history_authorized and user_id is not None:
            return self.base.rank(candidate_ids, k, user_id=user_id,
                                  history_authorized=history_authorized,
                                  liked_ids=liked_ids, disliked_ids=disliked_ids)
        known = (history_authorized and user_id is not None
                 and self.retriever.known_user is not None
                 and user_id <= self.retriever.known_user.n_users)
        source = "known_user" if known else "cold_start" if liked_ids or disliked_ids else "fallback_popularity"
        if self.mode == "no_content":
            ranked, _ = self.retriever.collaborative_rank(
                candidate_ids, liked_ids, user_id=user_id,
                history_authorized=history_authorized, limit=len(candidate_ids))
        else:
            # No model scores or collaborative recall: content-only preference scoring.
            from ..ranking.constraints import Constraints
            ranked = self.retriever.content_rank(candidate_ids, liked_ids, disliked_ids,
                                                  Constraints(), len(candidate_ids))
        missing = [i for i in self.retriever.popularity_rank(candidate_ids, len(candidate_ids))
                   if i not in set(ranked)]
        ranked = ranked + missing
        scores = {i: float(len(ranked) - position) for position, i in enumerate(ranked)}
        return RouteResult(list(candidate_ids), [scores[i] for i in candidate_ids],
                           ranked[:k], source, None)


def build_pipeline_variant(base, condition):
    if condition not in ("no_user_model", "no_collaborative", "no_content"):
        return base
    original = base.retriever
    edges = [(u, i) for u, items in original.user_items.items() for i in items]
    retriever = CandidateRetriever(
        original.catalog, edges, original.known_user,
        use_content=condition != "no_content",
        use_collaborative=condition == "no_content",
    )
    scorer = (base.router if condition == "no_collaborative"
              else AblationScorer(base.router, retriever, condition))
    return FixedRecommendationPipeline(retriever, scorer)
