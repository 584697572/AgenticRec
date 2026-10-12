"""T11 hard filters must fail closed, including undersized candidate sets."""
from agenticrec.evaluation.metrics import rank_candidates

from agenticrec.ranking.constraints import Catalog, ConstraintGate, Constraints, Item


def catalog():
    return Catalog([
        Item(1, "Comedy 1995", frozenset({"Comedy"}), 1995),
        Item(2, "Horror 1996", frozenset({"Horror"}), 1996),
        Item(3, "Mixed 1997", frozenset({"Comedy", "Horror"}), 1997),
        Item(4, "Comedy 1980", frozenset({"Comedy"}), 1980),
        Item(5, "Drama unknown year", frozenset({"Drama"}), None),
        Item(6, "Comedy 2000", frozenset({"Comedy"}), 2000),
    ])


def test_true_filter_removes_unwanted_at_any_k_even_when_one_candidate_remains():
    gate = ConstraintGate(catalog())
    # Reproduce the old low-score-mask failure when k exceeds feasible items.
    assert rank_candidates({1: -1e9, 6: 1.0}, [1, 6], 5) == [6, 1]
    for k in (1, 5, 20):
        rules = Constraints(include_genres=("comedy",), exclude_genres=("HORROR",),
                            year_min=1990, year_max=2000, excluded_item_ids=(1,), k=k)
        before = gate.pre_filter([1, 2, 3, 4, 5, 6], rules)
        assert before.status == "OK" and before.item_ids == [6]
        # A low score mask would still return 1/2/3 when k exceeds feasible items.
        after = gate.finalize([2, 3, 1, 6], rules)
        assert after.status == "OK" and after.item_ids == [6]
        assert after.shortfall == max(0, k - 1)
        assert not set(after.item_ids) & {1, 2, 3, 4, 5}


def test_conflicts_missing_fields_and_no_solution_are_structured():
    gate = ConstraintGate(catalog())
    universe = list(range(1, 7))
    conflict = gate.pre_filter(universe, Constraints(include_genres=("Comedy",),
                                                   exclude_genres=("comedy",)))
    assert conflict.status == "CLARIFY_CONFLICT" and conflict.item_ids == []
    assert gate.pre_filter(universe, Constraints(year_min=2001, year_max=1999)).status == "CLARIFY_CONFLICT"
    missing = gate.pre_filter(universe, Constraints(required_fields=("duration",)))
    assert missing.status == "MISSING_FIELD" and missing.missing_fields == ("duration",)
    no_solution = gate.pre_filter(universe, Constraints(year_min=1800, year_max=1850))
    assert no_solution.status == "NO_FEASIBLE_ITEMS" and no_solution.item_ids == []
    assert "year" in no_solution.missing_fields  # item 5 was not guessed into the range
    assert gate.pre_filter(universe, Constraints(include_genres=("Space Opera",))).status == "UNKNOWN_GENRE"
    assert gate.pre_filter(universe, Constraints(excluded_item_ids=(99,))).status == "UNKNOWN_ITEM_ID"


def test_seen_flag_and_invalid_ids_do_not_leak():
    gate = ConstraintGate(catalog())
    assert gate.finalize([1, 2, 3], Constraints(k=3), seen_ids=[2]).item_ids == [1, 3]
    assert gate.finalize([1, 2, 3], Constraints(exclude_seen=False, k=3), seen_ids=[2]).item_ids == [1, 2, 3]
    assert gate.pre_filter([1, 1], Constraints()).status == "INVALID_ITEM_ID"
    assert gate.pre_filter([0], Constraints()).status == "INVALID_ITEM_ID"
    assert gate.finalize([2], Constraints(excluded_item_ids=(2,), k=5)).status == "NO_FEASIBLE_ITEMS"
