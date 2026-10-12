"""T10 scorer contract on frozen artifacts and a tiny independent catalog."""
import math

import pytest

from agenticrec.adapters.model import KnownUserScorer, RoutingScorer, SessionSeedScorer


def test_frozen_bpr_and_lightgcn_scores_align_with_candidate_order():
    candidates = [3, 1, 2]
    for kind in ("bpr", "lightgcn"):
        scorer = KnownUserScorer.from_artifacts(kind, seed=42)
        scores = scorer.score(1, candidates)
        assert len(scores) == len(candidates) and all(math.isfinite(s) for s in scores)
        assert scores == pytest.approx([scorer.score(1, [i])[0] for i in candidates], abs=1e-6)
        with pytest.raises(ValueError, match="Unknown user"):
            scorer.score(0, candidates)
        with pytest.raises(ValueError, match="Unknown user"):
            scorer.score(scorer.n_users + 1, candidates)
        with pytest.raises(ValueError, match="positive model item IDs"):
            scorer.score(1, [0])
        assert not set(scorer.recommend(1, list(range(1, scorer.n_items + 1)), 10)) & scorer.train_seen[1]


def test_seed_scoring_and_unknown_user_fallback_are_explicit():
    session = SessionSeedScorer({1: ["A"], 2: ["A"], 3: ["B"], 4: ["B"]},
                                {1: 4, 2: 2, 3: 1, 4: 0}, 4)
    route = RoutingScorer(None, session)
    cold = route.rank([3, 2, 1, 4], 2, user_id=99, history_authorized=True,
                      liked_ids=[1], disliked_ids=[3])
    assert cold.profile_source == "cold_start" and cold.fallback_reason == "unknown_user"
    assert cold.ranked_ids == [2, 4]  # liked/disliked seeds excluded
    assert cold.scores == session.score_from_seeds([1], [3], [3, 2, 1, 4])
    no_history = route.rank([4, 3, 2, 1], 2, user_id=99, history_authorized=True)
    assert no_history.profile_source == "fallback_popularity"
    assert no_history.fallback_reason == "model_unavailable"
    assert no_history.ranked_ids == [1, 2]
    unauthorized = route.rank([1, 2, 3, 4], 2, user_id=1, history_authorized=False,
                              liked_ids=[1])
    assert unauthorized.profile_source == "cold_start"
    assert unauthorized.fallback_reason == "identity_not_authorized"
    with pytest.raises(ValueError, match="padding"):
        route.rank([1, 2], 1, user_id=0, history_authorized=True)
    with pytest.raises(ValueError, match="both liked and disliked"):
        session.score_from_seeds([1], [1], [2])


def test_public_checkpoint_hash_mismatch_is_rejected(monkeypatch):
    import agenticrec.adapters.model as adapter
    original = adapter.digest

    def wrong_checkpoint_hash(path):
        if str(path).endswith("seed_42.pt"):
            return "0" * 64
        return original(path)

    monkeypatch.setattr(adapter, "digest", wrong_checkpoint_hash)
    with pytest.raises(ValueError, match="Checkpoint SHA256"):
        KnownUserScorer.from_artifacts("bpr", seed=42)
