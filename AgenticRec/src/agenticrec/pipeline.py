"""T12 deterministic fixed recommendation flow; structured input, zero LLM calls."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from .adapters.model import KnownUserScorer, RoutingScorer, SessionSeedScorer
from .ranking.constraints import Constraints
from .retrieval.sources import CandidateRetriever


REQUEST_KEYS = frozenset(("schema_version", "user_id", "history_authorized",
                          "liked_item_ids", "disliked_item_ids", "seen_item_ids",
                          "constraints"))
CONSTRAINT_KEYS = frozenset(("include_genres", "exclude_genres", "year_min", "year_max",
                             "excluded_item_ids", "exclude_seen", "k", "required_fields"))


def _id_tuple(value, name):
    if type(value) is not list or any(type(item) is not int or item <= 0 for item in value):
        raise ValueError(name + " must be a JSON list of positive integer model IDs")
    if len(value) != len(set(value)):
        raise ValueError(name + " contains duplicate IDs")
    return tuple(value)


def _string_tuple(value, name):
    if type(value) is not list or any(type(item) is not str or not item.strip() for item in value):
        raise ValueError(name + " must be a JSON list of nonempty strings")
    if len({item.casefold() for item in value}) != len(value):
        raise ValueError(name + " contains duplicate values")
    return tuple(value)


@dataclass(frozen=True)
class FixedRequest:
    constraints: Constraints
    user_id: int | None = None
    history_authorized: bool = False
    liked_item_ids: tuple[int, ...] = ()
    disliked_item_ids: tuple[int, ...] = ()
    seen_item_ids: tuple[int, ...] = ()
    schema_version: int = 1

    @classmethod
    def parse(cls, payload):
        if type(payload) is not dict or set(payload) - REQUEST_KEYS:
            raise ValueError("Request must be an object with only documented fields")
        if payload.get("schema_version") != 1 or type(payload.get("schema_version")) is not int:
            raise ValueError("schema_version must be integer 1")
        raw_constraints = payload.get("constraints")
        if type(raw_constraints) is not dict or set(raw_constraints) - CONSTRAINT_KEYS:
            raise ValueError("constraints must be an object with only documented fields")
        include = _string_tuple(raw_constraints.get("include_genres", []), "include_genres")
        exclude = _string_tuple(raw_constraints.get("exclude_genres", []), "exclude_genres")
        required = _string_tuple(raw_constraints.get("required_fields", []), "required_fields")
        excluded = _id_tuple(raw_constraints.get("excluded_item_ids", []), "excluded_item_ids")
        year_min, year_max = raw_constraints.get("year_min"), raw_constraints.get("year_max")
        if any(value is not None and (type(value) is not int or value <= 0)
               for value in (year_min, year_max)):
            raise ValueError("year_min/year_max must be positive integers or null")
        exclude_seen = raw_constraints.get("exclude_seen", True)
        k = raw_constraints.get("k", 5)
        if type(exclude_seen) is not bool or type(k) is not int or k <= 0:
            raise ValueError("exclude_seen must be boolean and k a positive integer")
        user_id = payload.get("user_id")
        if user_id is not None and (type(user_id) is not int or user_id <= 0):
            raise ValueError("user_id must be a positive model user ID or null")
        authorized = payload.get("history_authorized", False)
        if type(authorized) is not bool:
            raise ValueError("history_authorized must be boolean")
        return cls(Constraints(include, exclude, year_min, year_max, excluded,
                               exclude_seen, k, required), user_id, authorized,
                   _id_tuple(payload.get("liked_item_ids", []), "liked_item_ids"),
                   _id_tuple(payload.get("disliked_item_ids", []), "disliked_item_ids"),
                   _id_tuple(payload.get("seen_item_ids", []), "seen_item_ids"))

    @classmethod
    def from_json_file(cls, path):
        path = Path(path)
        return cls.parse(json.loads(path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class RecommendationEvidence:
    item_id: int
    title: str
    genres: list[str]
    year: int | None
    personalized_score: float
    rrf_score: float
    source_ranks: dict[str, int]


@dataclass(frozen=True)
class FixedResponse:
    status: str
    reason: str | None
    recommendations: list[RecommendationEvidence]
    profile_source: str
    fallback_reason: str | None
    candidate_counts: dict[str, object]
    missing_fields: list[str]
    hard_constraint_shortfall: int
    input_mode: str = "structured_json"
    planner: str = "not_invoked"
    llm_requests: int = 0
    remote_api_requests: int = 0

    def to_dict(self):
        return asdict(self)


class FixedRecommendationPipeline:
    """Recall -> RRF -> profile scoring -> final hard gate -> evidence."""

    def __init__(self, retriever, router):
        if retriever.catalog.n_items != router.session.n_items:
            raise ValueError("Retriever and scorer use different item ID spaces")
        self.retriever, self.router = retriever, router
        self.candidate_ids = list(range(1, retriever.catalog.n_items + 1))

    @classmethod
    def from_frozen(cls, model_kind="lightgcn", seed=42):
        known = KnownUserScorer.from_artifacts(model_kind, seed)
        session = SessionSeedScorer.from_frozen_train()
        return cls(CandidateRetriever.from_frozen(known), RoutingScorer(known, session))

    def recommend(self, request: FixedRequest):
        if not isinstance(request, FixedRequest):
            raise ValueError("Fixed pipeline requires a validated FixedRequest")
        supplied_ids = (request.liked_item_ids + request.disliked_item_ids
                        + request.seen_item_ids)
        if any(item_id > self.retriever.catalog.n_items for item_id in supplied_ids):
            return FixedResponse("INVALID_ITEM_ID",
                                 "Seed/seen item ID is outside the frozen catalog", [],
                                 "unavailable", "invalid_request_item_id",
                                 {"universe": len(self.candidate_ids), "eligible": 0,
                                  "per_source": {}, "fused": 0, "final": 0},
                                 [], request.constraints.k)
        retrieved = self.retriever.retrieve(
            self.candidate_ids, request.constraints, user_id=request.user_id,
            history_authorized=request.history_authorized,
            liked_ids=request.liked_item_ids, disliked_ids=request.disliked_item_ids,
            seen_ids=request.seen_item_ids)
        source_counts = {name: len(ids) for name, ids in retrieved.source_rankings.items()}
        counts = {"universe": len(self.candidate_ids), "eligible": retrieved.eligible_count,
                  "per_source": source_counts, "fused": len(retrieved.fused.candidate_ids),
                  "final": 0}
        if retrieved.status != "OK":
            return FixedResponse(retrieved.status, retrieved.reason, [],
                                 retrieved.profile_source, retrieved.fallback_reason,
                                 counts, list(retrieved.missing_fields),
                                 request.constraints.k)
        fused_ids = retrieved.fused.candidate_ids
        routed = self.router.rank(
            fused_ids, len(fused_ids), user_id=request.user_id,
            history_authorized=request.history_authorized,
            liked_ids=request.liked_item_ids, disliked_ids=request.disliked_item_ids)
        if routed.profile_source != retrieved.profile_source:
            raise ValueError("Retrieval and scoring selected different profile routes")
        personalized = dict(zip(routed.candidate_ids, routed.scores))
        reranked = sorted(routed.ranked_ids,
                          key=lambda item_id: (-personalized[item_id],
                                               -retrieved.fused.rrf_scores[item_id], item_id))
        final = self.retriever.gate.finalize(
            reranked, retrieved.effective_constraints, retrieved.seen_ids)
        if final.status != "OK":
            return FixedResponse(final.status, final.reason, [], routed.profile_source,
                                 routed.fallback_reason, counts,
                                 list(final.missing_fields), request.constraints.k)
        evidence = []
        for item_id in final.item_ids:
            item = self.retriever.catalog.items[item_id]
            evidence.append(RecommendationEvidence(
                item_id, item.title, sorted(item.genres), item.year,
                personalized[item_id], retrieved.fused.rrf_scores[item_id],
                retrieved.fused.source_ranks[item_id]))
        counts["final"] = len(evidence)
        reason = "FEWER_FEASIBLE_ITEMS" if final.shortfall else None
        return FixedResponse("OK", reason, evidence, routed.profile_source,
                             routed.fallback_reason, counts,
                             list(final.missing_fields), final.shortfall)
