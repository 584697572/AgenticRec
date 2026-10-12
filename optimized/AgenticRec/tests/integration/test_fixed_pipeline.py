"""T12 fixed structured flow: known, anonymous, exclusion, no-solution and CLI."""
import json

from agenticrec.adapters.model import RoutingScorer, SessionSeedScorer
from agenticrec.cli import main
from agenticrec.pipeline import FixedRecommendationPipeline, FixedRequest
from agenticrec.ranking.constraints import Catalog, Item
from agenticrec.retrieval.sources import CandidateRetriever


class KnownFixture:
    n_users = 1
    n_items = 6
    train_seen = {1: {1}}

    def score(self, user_id, item_ids):
        assert user_id == 1
        return [float(item_id) for item_id in item_ids]


def fixed_pipeline():
    catalog = Catalog([
        Item(1, "Action seed", frozenset({"Action"}), 1995),
        Item(2, "Horror only", frozenset({"Horror"}), 1996),
        Item(3, "Action horror", frozenset({"Action", "Horror"}), 1997),
        Item(4, "Action future", frozenset({"Action"}), 2001),
        Item(5, "Drama", frozenset({"Drama"}), 1999),
        Item(6, "Action match", frozenset({"Action"}), 1998),
    ])
    edges = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 4), (2, 6),
             (3, 1), (3, 5), (3, 6)]
    known = KnownFixture()
    session = SessionSeedScorer({i: catalog.items[i].genres for i in catalog.items},
                                {1: 3, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2}, 6)
    return FixedRecommendationPipeline(CandidateRetriever(catalog, edges, known),
                                       RoutingScorer(known, session))


def request(**changes):
    payload = {"schema_version": 1, "user_id": 1, "history_authorized": True,
               "liked_item_ids": [], "disliked_item_ids": [], "seen_item_ids": [],
               "constraints": {"include_genres": ["Action"],
                               "exclude_genres": ["Horror"],
                               "year_min": 1990, "year_max": 2005,
                               "excluded_item_ids": [], "exclude_seen": True, "k": 2,
                               "required_fields": ["title", "genres", "year"]}}
    payload.update(changes)
    return FixedRequest.parse(payload)


def assert_zero_llm(response):
    assert response.input_mode == "structured_json"
    assert response.planner == "not_invoked"
    assert response.llm_requests == response.remote_api_requests == 0


def test_known_user_success_returns_metadata_sources_and_both_scores():
    response = fixed_pipeline().recommend(request())
    assert response.status == "OK" and response.profile_source == "known_user"
    assert [item.item_id for item in response.recommendations] == [6, 4]
    assert all(item.title and item.genres == ["Action"] and item.year for item in response.recommendations)
    assert all(item.source_ranks and isinstance(item.personalized_score, float)
               and item.rrf_score > 0 for item in response.recommendations)
    assert response.candidate_counts["final"] == 2
    assert_zero_llm(response)


def test_anonymous_seed_uses_cold_start_without_known_identity():
    response = fixed_pipeline().recommend(request(
        user_id=None, history_authorized=False, liked_item_ids=[1]))
    assert response.status == "OK" and response.profile_source == "cold_start"
    assert response.fallback_reason == "anonymous"
    assert 1 not in [item.item_id for item in response.recommendations]
    assert response.candidate_counts["per_source"]["cf_item_item"] > 0
    assert_zero_llm(response)


def test_explicit_exclusion_survives_recall_rerank_and_large_k():
    constraints = {"include_genres": ["Action"], "exclude_genres": ["Horror"],
                   "year_min": 1990, "year_max": 2005, "excluded_item_ids": [6],
                   "exclude_seen": True, "k": 20, "required_fields": []}
    response = fixed_pipeline().recommend(request(constraints=constraints))
    ids = [item.item_id for item in response.recommendations]
    assert response.status == "OK" and ids == [4]
    assert 6 not in ids and response.hard_constraint_shortfall == 19
    assert response.reason == "FEWER_FEASIBLE_ITEMS"
    assert_zero_llm(response)


def test_no_solution_is_structured_and_never_relaxes_year():
    constraints = {"year_min": 1800, "year_max": 1850, "k": 5}
    response = fixed_pipeline().recommend(request(constraints=constraints))
    assert response.status == "NO_FEASIBLE_ITEMS"
    assert response.recommendations == [] and response.candidate_counts["final"] == 0
    assert response.hard_constraint_shortfall == 5
    assert_zero_llm(response)


def test_cli_accepts_structured_json_and_rejects_free_text_without_parser(
        tmp_path, monkeypatch, capsys):
    pipeline = fixed_pipeline()
    monkeypatch.setattr(FixedRecommendationPipeline, "from_frozen",
                        classmethod(lambda cls, model_kind="lightgcn", seed=42: pipeline))
    path = tmp_path / "request.json"
    payload = {"schema_version": 1, "user_id": None, "history_authorized": False,
               "liked_item_ids": [1], "disliked_item_ids": [], "seen_item_ids": [],
               "constraints": {"k": 2}}
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert main(["recommend", "--request", str(path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "OK" and report["llm_requests"] == 0
    path.write_text(json.dumps("find me a movie"), encoding="utf-8")
    assert main(["recommend", "--request", str(path)]) == 2
    failure = json.loads(capsys.readouterr().out)
    assert failure["status"] == "INVALID_REQUEST"
    assert failure["planner"] == "not_invoked" and failure["llm_requests"] == 0


def test_request_schema_rejects_unknown_fields_and_implicit_authorization():
    payload = {"schema_version": 1, "history_authorized": "yes", "constraints": {}}
    try:
        FixedRequest.parse(payload)
        raise AssertionError("string authorization was accepted")
    except ValueError:
        pass
    invalid_id = fixed_pipeline().recommend(request(liked_item_ids=[99]))
    assert invalid_id.status == "INVALID_ITEM_ID"
    assert invalid_id.recommendations == [] and invalid_id.llm_requests == 0
    payload = {"schema_version": 1, "constraints": {"plot_mood": "happy"}}
    try:
        FixedRequest.parse(payload)
        raise AssertionError("unsupported constraint was accepted")
    except ValueError:
        pass
