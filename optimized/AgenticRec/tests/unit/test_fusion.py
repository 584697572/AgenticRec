"""T11 RRF numeric and multi-route retrieval contracts."""
import os
import subprocess
import sys

import pytest

from agenticrec.ranking.constraints import Catalog, Constraints, Item
from agenticrec.retrieval.fusion import reciprocal_rank_fusion
from agenticrec.retrieval.sources import CandidateRetriever


def fixture_retriever():
    catalog = Catalog([
        Item(1, "Comedy A 1995", frozenset({"Comedy"}), 1995),
        Item(2, "Horror B 1996", frozenset({"Horror"}), 1996),
        Item(3, "Mixed C 1997", frozenset({"Comedy", "Horror"}), 1997),
        Item(4, "Comedy D 1980", frozenset({"Comedy"}), 1980),
        Item(5, "Drama E 1999", frozenset({"Drama"}), 1999),
        Item(6, "Comedy F 1995", frozenset({"Comedy"}), 1995),
    ])
    edges = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 3), (2, 4),
             (3, 2), (3, 4), (3, 5), (4, 1), (4, 6)]
    return CandidateRetriever(catalog, edges)


def test_rrf_hand_calculation_dedup_and_source_evidence():
    fused = reciprocal_rank_fusion({"content": [1, 2, 3], "cf": [3, 2]},
                                   universe=[1, 2, 3, 4], union_cap=3)
    assert fused.candidate_ids == [3, 2, 1]
    assert fused.rrf_scores[3] == pytest.approx(1 / 63 + 1 / 61)
    assert fused.rrf_scores[2] == pytest.approx(2 / 62)
    assert fused.source_ranks[3] == {"content": 3, "cf": 1}
    with pytest.raises(ValueError, match="duplicate"):
        reciprocal_rank_fusion({"content": [1, 1]}, universe=[1, 2])
    with pytest.raises(ValueError, match="unverified"):
        reciprocal_rank_fusion({"content": [99]}, universe=[1, 2])


def test_three_routes_filter_before_fusion_and_cap_union():
    retriever = fixture_retriever()
    rules = Constraints(include_genres=("Comedy",), exclude_genres=("Horror",),
                        excluded_item_ids=(4,), k=5)
    result = retriever.retrieve([1, 2, 3, 4, 5, 6], rules, liked_ids=(1,),
                                per_route=2, union_cap=2)
    assert result.status == "OK" and result.profile_source == "cold_start"
    assert result.eligible_count == 1 and result.pre_filter.item_ids == [6]
    assert set(result.source_rankings) == {"content", "cf_item_item", "popularity"}
    assert result.fused.candidate_ids == [6]
    assert retriever.gate.finalize(result.fused.candidate_ids,
                                   result.effective_constraints,
                                   result.seen_ids).item_ids == [6]
    assert 1 not in result.fused.candidate_ids  # seed excluded, even at k=5


def test_recall_caps_limit_reranking_space_and_no_seed_has_popularity_only():
    retriever = fixture_retriever()
    result = retriever.retrieve([1, 2, 3, 4, 5, 6], Constraints(k=5),
                                per_route=1, union_cap=2)
    assert result.status == "OK" and result.eligible_count == 6
    assert result.source_rankings == {"content": [], "popularity": [1]}
    assert result.fused.candidate_ids == [1]  # rest are outside recall, not rerankable
    assert retriever.gate.finalize(result.fused.candidate_ids,
                                   result.effective_constraints).shortfall == 4


def test_unknown_or_unauthorized_user_cannot_enter_known_user_model():
    class SentinelModel:
        n_users = 1
        n_items = 6
        train_seen = {1: {1}}

        def score(self, _user, _items):
            raise AssertionError("known-user model must not be called")

    retriever = fixture_retriever()
    retriever.known_user = SentinelModel()
    for user, authorized in ((99, True), (1, False)):
        result = retriever.retrieve([1, 2, 3, 4, 5, 6], Constraints(k=3),
                                    user_id=user, history_authorized=authorized,
                                    liked_ids=(1,))
        assert result.status == "OK"
        assert "cf_model" not in result.source_rankings
        assert result.profile_source == "cold_start"


def test_authorized_known_user_uses_aligned_model_scores_and_excludes_train_seen():
    class KnownFixture:
        n_users = 1
        n_items = 6
        train_seen = {1: {1, 3}}

        def score(self, user_id, item_ids):
            assert user_id == 1 and 1 not in item_ids and 3 not in item_ids
            return [float(item_id) for item_id in item_ids]

    retriever = fixture_retriever()
    retriever.known_user = KnownFixture()
    result = retriever.retrieve([1, 2, 3, 4, 5, 6], Constraints(k=10),
                                user_id=1, history_authorized=True)
    assert result.status == "OK" and result.profile_source == "known_user"
    assert result.source_rankings["cf_model"] == [6, 5, 4, 2]
    assert not set(result.fused.candidate_ids) & {1, 3}
    assert retriever.gate.finalize(result.fused.candidate_ids,
                                   result.effective_constraints,
                                   result.seen_ids).item_ids == result.fused.candidate_ids


def test_content_index_is_reproducible_across_python_hash_seeds():
    code = """import json
from agenticrec.ranking.constraints import Catalog, Item
from agenticrec.retrieval.sources import CandidateRetriever
catalog = Catalog([Item(1, 'A Long Story (1995)', frozenset({'Comedy', 'Drama'}), 1995),
                   Item(2, 'Another Great Story (1998)', frozenset({'Comedy'}), 1998)])
retriever = CandidateRetriever(catalog, [(1, 1), (1, 2)])
print(json.dumps(retriever.content_vectors, sort_keys=True))
"""
    env = {key: value for key, value in os.environ.items()
           if not key.endswith(("_API_KEY", "_ACCESS_TOKEN", "_SECRET"))}
    outputs = []
    for seed in ("1", "2"):
        env["PYTHONHASHSEED"] = seed
        completed = subprocess.run([sys.executable, "-c", code], env=env,
                                   text=True, capture_output=True, check=True)
        outputs.append(completed.stdout)
    assert outputs[0] == outputs[1]
