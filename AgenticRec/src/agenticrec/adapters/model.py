"""Strict frozen-ID model loading and separate known/anonymous scoring."""
from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
import platform

import pyarrow
import pyarrow.parquet as pq
import torch

from ..data import digest
from ..evaluation.metrics import rank_candidates
from ..models.bpr import load_checkpoint
from ..models.lightgcn import load_lightgcn_checkpoint, normalized_bipartite_graph
from ..training import ROOT, _code_digest as bpr_code_digest, load_train_contract
from ..lightgcn_training import _code_digest as lightgcn_code_digest


def _ids(values, n_items, label):
    values = list(values)
    if any(type(i) is not int or not 1 <= i <= n_items for i in values):
        raise ValueError(f"{label} must be frozen, positive model item IDs")
    if len(values) != len(set(values)):
        raise ValueError(f"{label} contains duplicate IDs")
    return values


def _scores(values, size):
    values = list(values)
    if len(values) != size or any(type(x) not in (int, float) or not math.isfinite(x) for x in values):
        raise ValueError("Scorer must return one finite score per candidate in input order")
    return [float(x) for x in values]


class KnownUserScorer:
    """BPR-MF or LightGCN scored by *model user ID*, never item_seq/padding 0."""

    def __init__(self, model, kind, n_users, n_items, train_seen):
        if kind not in ("bpr", "lightgcn") or n_users < 1 or n_items < 1:
            raise ValueError("Invalid known-user model contract")
        self.model, self.kind = model, kind
        self.n_users, self.n_items = n_users, n_items
        self.train_seen = {int(u): frozenset(v) for u, v in train_seen.items()}
        if any(not 1 <= u <= n_users or any(not 1 <= i <= n_items for i in v)
               for u, v in self.train_seen.items()):
            raise ValueError("Train-seen IDs outside frozen contract")
        self.model.eval()
        with torch.no_grad():
            if kind == "lightgcn":
                self.user_vectors, self.item_vectors = self.model.propagate()

    @classmethod
    def from_artifacts(cls, kind="lightgcn", seed=42, root=ROOT):
        root = Path(root)
        if kind not in ("bpr", "lightgcn") or type(seed) is not int or seed < 0:
            raise ValueError("Unsupported frozen model or seed")
        if root.resolve() != ROOT.resolve():
            raise ValueError("Frozen trainer currently binds to the project root")
        manifest, n_users, n_items, train_seen, edges = load_train_contract()
        report_path = root / "reports/rec_baselines" / (
            f"seed_{seed}.json" if kind == "bpr" else f"lightgcn_seed_{seed}.json")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        config_path = root / "AgenticRec/configs" / ("bpr.yaml" if kind == "bpr" else "lightgcn.yaml")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if report["status"] != "PASS" or report["seed"] != seed or report["config"] != config:
            raise ValueError("Public report/config/seed mismatch")
        identity = {
            "id_map_sha256": manifest["id_map_sha256"],
            "split_sha256": manifest["files"]["split_manifest.json"]["sha256"],
            "data_manifest_sha256": digest(root / "artifacts/data_manifest.json"),
            "config_sha256": digest(config_path),
            "code_sha256": bpr_code_digest() if kind == "bpr" else lightgcn_code_digest(),
            "seed": seed,
            "torch_version": str(torch.__version__),
            "pyarrow_version": pyarrow.__version__,
            "python_version": platform.python_version(),
        }
        if kind == "lightgcn":
            identity["train_positive_sha256"] = manifest["files"]["train_positive.parquet"]["sha256"]
        if report["identity"] != identity:
            raise ValueError("Public model identity differs from frozen local data/code/dependencies")
        if kind == "bpr":
            checkpoint = root / "artifacts/models/bpr" / f"seed_{seed}.pt"
            expected_sha = report["checkpoint_sha256"]
        else:
            layers = report["selection"]["selected_layers"]
            if layers not in config["layers_to_try"]:
                raise ValueError("Selected layer outside frozen configuration")
            checkpoint = root / "artifacts/models/lightgcn" / f"seed_{seed}_layer_{layers}.pt"
            expected_sha = report["trials"][str(layers)]["checkpoint_sha256"]
        if digest(checkpoint) != expected_sha:
            raise ValueError("Checkpoint SHA256 differs from public report")
        if kind == "bpr":
            model = load_checkpoint(checkpoint, n_users, n_items, config["embedding_dim"], identity)
        else:
            graph = normalized_bipartite_graph(n_users, n_items, edges)
            model = load_lightgcn_checkpoint(checkpoint, graph, n_users, n_items,
                                             config["embedding_dim"], layers, identity, config)
        return cls(model, kind, n_users, n_items, train_seen)

    def score(self, user_id, candidate_ids):
        if type(user_id) is not int or not 1 <= user_id <= self.n_users:
            raise ValueError("Unknown user ID; padding user 0 is forbidden")
        candidates = _ids(candidate_ids, self.n_items, "candidate_ids")
        if not candidates:
            return []
        with torch.no_grad():
            if self.kind == "bpr":
                vector = self.model.users.weight[user_id]
                items = self.model.items.weight[torch.tensor(candidates)]
            else:
                vector = self.user_vectors[user_id - 1]
                items = self.item_vectors[torch.tensor(candidates) - 1]
            values = (items @ vector).tolist()
        return _scores(values, len(candidates))

    def recommend(self, user_id, candidate_ids, k):
        candidates = _ids(candidate_ids, self.n_items, "candidate_ids")
        scores = self.score(user_id, candidates)
        return rank_candidates(dict(zip(candidates, scores)), candidates, k,
                               self.train_seen.get(user_id, ()))


class SessionSeedScorer:
    """Independent genre Jaccard seed pooling plus train-only popularity prior."""

    def __init__(self, genres_by_item, train_popularity, n_items):
        if type(n_items) is not int or n_items < 1:
            raise ValueError("Invalid item universe")
        self.n_items = n_items
        self.genres = {i: frozenset(genres_by_item.get(i, ())) for i in range(1, n_items + 1)}
        self.popularity = {i: int(train_popularity.get(i, 0)) for i in range(1, n_items + 1)}
        if any(count < 0 for count in self.popularity.values()):
            raise ValueError("Negative train popularity")
        denominator = math.log1p(max(self.popularity.values())) or 1.0
        self.prior = {i: math.log1p(self.popularity[i]) / denominator for i in self.popularity}

    @classmethod
    def from_frozen_train(cls, root=ROOT):
        root = Path(root)
        if root.resolve() != ROOT.resolve():
            raise ValueError("Frozen train contract currently binds to the project root")
        manifest, _n_users, n_items, _seen, edges = load_train_contract()
        items_path = root / "data/processed/ml-1m-v1/items.parquet"
        if digest(items_path) != manifest["files"]["items.parquet"]["sha256"]:
            raise ValueError("Catalog metadata hash mismatch")
        table = pq.read_table(items_path, columns=["item_id", "genres"])
        genres = {int(i): g for i, g in zip(table["item_id"].to_pylist(),
                                            table["genres"].to_pylist()) if i is not None}
        if set(genres) != set(range(1, n_items + 1)):
            raise ValueError("Catalog model IDs differ from train candidate IDs")
        return cls(genres, Counter(i for _, i in edges), n_items)

    def score_from_seeds(self, liked_ids, disliked_ids, candidate_ids):
        liked = _ids(liked_ids, self.n_items, "liked_ids")
        disliked = _ids(disliked_ids, self.n_items, "disliked_ids")
        if set(liked) & set(disliked):
            raise ValueError("A seed cannot be both liked and disliked")
        candidates = _ids(candidate_ids, self.n_items, "candidate_ids")

        def similarity(a, b):
            left, right = self.genres[a], self.genres[b]
            return len(left & right) / len(left | right) if left and right else 0.0

        return _scores([self.prior[i] * 0.01
                        + (sum(similarity(i, seed) for seed in liked) / len(liked) if liked else 0.0)
                        - (sum(similarity(i, seed) for seed in disliked) / len(disliked) if disliked else 0.0)
                        for i in candidates], len(candidates))


@dataclass(frozen=True)
class RouteResult:
    candidate_ids: list[int]
    scores: list[float]
    ranked_ids: list[int]
    profile_source: str
    fallback_reason: str | None


class RoutingScorer:
    """Require explicit identity authorization before any known-user model access."""

    def __init__(self, known_user, session):
        if known_user is not None and known_user.n_items != session.n_items:
            raise ValueError("Known and anonymous scorers use different ID spaces")
        self.known_user, self.session = known_user, session

    def rank(self, candidate_ids, k, *, user_id=None, history_authorized=False,
             liked_ids=(), disliked_ids=()):
        candidates = _ids(candidate_ids, self.session.n_items, "candidate_ids")
        liked = _ids(liked_ids, self.session.n_items, "liked_ids")
        disliked = _ids(disliked_ids, self.session.n_items, "disliked_ids")
        if type(history_authorized) is not bool:
            raise ValueError("history_authorized must be explicit boolean")
        if user_id is not None and (type(user_id) is not int or user_id <= 0):
            raise ValueError("user_id must be positive or absent; padding 0 is forbidden")
        known = (history_authorized and user_id is not None and self.known_user is not None
                 and user_id <= self.known_user.n_users)
        if known:
            scores = self.known_user.score(user_id, candidates)
            excluded = self.known_user.train_seen.get(user_id, ())
            source, reason = "known_user", None
        else:
            scores = self.session.score_from_seeds(liked, disliked, candidates)
            excluded = set(liked) | set(disliked)
            if liked or disliked:
                source = "cold_start"
                reason = ("unknown_user" if history_authorized and user_id is not None else
                          "identity_not_authorized" if user_id is not None else "anonymous")
            else:
                source = "fallback_popularity"
                reason = ("model_unavailable" if history_authorized and user_id is not None and self.known_user is None else
                          "unknown_user" if history_authorized and user_id is not None else
                          "identity_not_authorized" if user_id is not None else "anonymous_no_seeds")
        scores = _scores(scores, len(candidates))
        ranked = rank_candidates(dict(zip(candidates, scores)), candidates, k, excluded)
        return RouteResult(candidates, scores, ranked, source, reason)
