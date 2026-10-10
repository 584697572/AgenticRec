"""Train-only content, collaborative, popularity recall with pre-fusion hard gate."""
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
import json
import math
import re

from ..data import digest
from ..ranking.constraints import Catalog, ConstraintGate, ConstraintResult, Constraints
from ..training import ROOT, load_train_contract
from .fusion import FusionResult, reciprocal_rank_fusion


def _id_list(values, catalog, name):
    result = list(values)
    if (any(type(i) is not int or i not in catalog.items for i in result)
            or len(result) != len(set(result))):
        raise ValueError(name + " must be unique frozen model item IDs")
    return result


@dataclass(frozen=True)
class RetrievalResult:
    status: str
    reason: str | None
    missing_fields: tuple[str, ...]
    eligible_count: int
    source_rankings: dict[str, list[int]]
    fused: FusionResult
    effective_constraints: Constraints
    seen_ids: tuple[int, ...]
    profile_source: str
    fallback_reason: str | None
    pre_filter: ConstraintResult


class CandidateRetriever:
    """The initial 100/route, 200-union RRF baseline from spec section 8.1."""

    def __init__(self, catalog: Catalog, train_positive_edges, known_user=None, *,
                 use_content=True, use_collaborative=True):
        if type(use_content) is not bool or type(use_collaborative) is not bool:
            raise ValueError("recall switches must be boolean")
        self.use_content, self.use_collaborative = use_content, use_collaborative
        self.catalog = catalog
        self.gate = ConstraintGate(catalog)
        self.known_user = known_user
        if known_user is not None and known_user.n_items != catalog.n_items:
            raise ValueError("Model and content catalog use different frozen item IDs")
        edges = list(train_positive_edges)
        if len(edges) != len(set(edges)) or any(type(u) is not int or u <= 0
               or type(i) is not int or i not in catalog.items for u, i in edges):
            raise ValueError("Train-positive graph has duplicate or invalid IDs")
        self.popularity = Counter(i for _u, i in edges)
        self.user_items = defaultdict(set)
        self.item_users = defaultdict(set)
        for user_id, item_id in edges:
            self.user_items[user_id].add(item_id)
            self.item_users[item_id].add(user_id)
        self.content_vectors = self._make_content_vectors()

    @classmethod
    def from_frozen(cls, known_user=None):
        manifest, _n_users, n_items, _seen, edges = load_train_contract()
        baseline = json.loads((ROOT / "reports/rec_baselines/seed_42.json").read_text(encoding="utf-8"))
        if (baseline["identity"]["data_manifest_sha256"] != digest(ROOT / "artifacts/data_manifest.json")
                or baseline["identity"]["id_map_sha256"] != manifest["id_map_sha256"]):
            raise ValueError("Train graph differs from the public frozen protocol identity")
        catalog = Catalog.from_frozen()
        if catalog.n_items != n_items:
            raise ValueError("Catalog and train graph dimensions differ")
        return cls(catalog, edges, known_user)

    def _make_content_vectors(self):
        terms_by_id = {}
        frequency = Counter()
        for item_id, item in self.catalog.items.items():
            terms = {"title:" + word for word in re.findall(r"[a-z0-9]+", item.title.casefold())}
            terms |= {"genre:" + genre.casefold() for genre in item.genres}
            if item.year is not None:
                terms.add("year:" + str(item.year))
            terms_by_id[item_id] = terms
            frequency.update(terms)
        n_items = self.catalog.n_items
        idf = {term: math.log((n_items + 1) / (count + 1)) + 1.0
               for term, count in frequency.items()}
        vectors = {}
        for item_id, terms in terms_by_id.items():
            ordered = sorted(terms)
            norm = math.sqrt(sum(idf[term] ** 2 for term in ordered)) or 1.0
            vectors[item_id] = {term: idf[term] / norm for term in ordered}
        return vectors

    def content_rank(self, eligible_ids, liked_ids, disliked_ids, constraints, limit):
        if not liked_ids and not disliked_ids and not constraints.include_genres:
            return []
        query = defaultdict(float)
        for seed in liked_ids:
            for term, weight in self.content_vectors[seed].items():
                query[term] += weight / len(liked_ids)
        for seed in disliked_ids:
            for term, weight in self.content_vectors[seed].items():
                query[term] -= weight / len(disliked_ids)
        for genre in constraints.include_genres:
            query["genre:" + genre.casefold()] += 1.0
        scores = {item_id: sum(query.get(term, 0.0) * weight
                               for term, weight in self.content_vectors[item_id].items())
                  for item_id in eligible_ids}
        return sorted((i for i in eligible_ids if scores[i] > 0),
                      key=lambda i: (-scores[i], i))[:limit]

    def collaborative_rank(self, eligible_ids, liked_ids, *, user_id, history_authorized, limit):
        if (history_authorized and user_id is not None and self.known_user is not None
                and user_id <= self.known_user.n_users):
            scores = self.known_user.score(user_id, eligible_ids)
            aligned = dict(zip(eligible_ids, scores))
            return (sorted(eligible_ids, key=lambda i: (-aligned[i], i))[:limit],
                    "cf_model")
        if not liked_ids:
            return [], None
        allowed = set(eligible_ids)
        scores = defaultdict(float)
        for seed in liked_ids:
            cooccurrence = Counter()
            for other_user in self.item_users[seed]:
                cooccurrence.update(self.user_items[other_user])
            for item_id, count in cooccurrence.items():
                if item_id in allowed:
                    scores[item_id] += count / (math.sqrt(self.popularity[item_id] *
                                                           self.popularity[seed]) * len(liked_ids))
        return sorted((i for i in scores if scores[i] > 0),
                      key=lambda i: (-scores[i], i))[:limit], "cf_item_item"

    def popularity_rank(self, eligible_ids, limit):
        return sorted(eligible_ids, key=lambda i: (-self.popularity[i], i))[:limit]

    def retrieve(self, candidate_ids, constraints: Constraints, *, user_id=None,
                 history_authorized=False, liked_ids=(), disliked_ids=(), seen_ids=(),
                 per_route=100, union_cap=200):
        if type(history_authorized) is not bool:
            raise ValueError("history_authorized must be explicit boolean")
        if user_id is not None and (type(user_id) is not int or user_id <= 0):
            raise ValueError("user_id must be positive or absent; padding 0 forbidden")
        if not isinstance(constraints, Constraints):
            empty = self.gate.pre_filter([], constraints)
            return self._empty(empty, constraints, (), "unavailable", "invalid_constraints")
        if (type(per_route) is not int or per_route <= 0 or type(union_cap) is not int or union_cap <= 0):
            raise ValueError("Retrieval caps must be positive integers")
        validation = self.gate.pre_filter([], constraints)
        if validation.status not in ("NO_FEASIBLE_ITEMS", "OK"):
            return self._empty(validation, constraints, (), "unavailable", "invalid_constraints")
        liked = _id_list(liked_ids, self.catalog, "liked_ids")
        disliked = _id_list(disliked_ids, self.catalog, "disliked_ids")
        if set(liked) & set(disliked):
            empty = ConstraintResult("CLARIFY_CONFLICT", [], "Item liked and disliked simultaneously", (), {}, constraints.k)
            return self._empty(empty, constraints, (), "cold_start", "seed_conflict")
        effective = replace(constraints, excluded_item_ids=tuple(sorted(
            set(constraints.excluded_item_ids) | set(liked) | set(disliked))))
        seen = set(_id_list(seen_ids, self.catalog, "seen_ids"))
        known = (history_authorized and user_id is not None and self.known_user is not None
                 and user_id <= self.known_user.n_users)
        if known:
            seen.update(self.known_user.train_seen.get(user_id, ()))
        pre = self.gate.pre_filter(candidate_ids, effective, tuple(sorted(seen)))
        source = "known_user" if known else "cold_start" if liked or disliked else "fallback_popularity"
        reason = (None if known else "model_unavailable" if history_authorized and user_id is not None and self.known_user is None
                  else "unknown_user" if history_authorized and user_id is not None
                  else "identity_not_authorized" if user_id is not None else
                  "anonymous")
        if pre.status != "OK":
            return self._empty(pre, effective, tuple(sorted(seen)), source, reason)
        eligible = pre.item_ids
        content = (self.content_rank(eligible, liked, disliked, effective, per_route)
                   if self.use_content else [])
        collaborative, cf_name = (self.collaborative_rank(
            eligible, liked, user_id=user_id, history_authorized=history_authorized, limit=per_route)
            if self.use_collaborative else ([], None))
        rankings = {"content": content} if self.use_content else {}
        if cf_name is not None:
            rankings[cf_name] = collaborative
        rankings["popularity"] = self.popularity_rank(eligible, per_route)
        fused = reciprocal_rank_fusion(rankings, universe=eligible,
                                       per_route=per_route, union_cap=union_cap)
        # The final gate is repeated by T12 after personalized reranking.
        post = self.gate.pre_filter(fused.candidate_ids, effective, tuple(sorted(seen)))
        if post.status != "OK" or post.item_ids != fused.candidate_ids:
            raise ValueError("Fusion emitted an item that failed the post-rank hard gate")
        return RetrievalResult("OK", None, pre.missing_fields, len(eligible), rankings,
                               fused, effective, tuple(sorted(seen)), source, reason, pre)

    @staticmethod
    def _empty(pre, constraints, seen, source, reason):
        return RetrievalResult(pre.status, pre.reason, pre.missing_fields, 0, {},
                               FusionResult([], {}, {}), constraints, seen, source, reason, pre)
