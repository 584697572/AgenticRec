import pytest

from agenticrec.evaluation.episodes import (
    EpisodeAttempt,
    PrivateEpisode,
    PublicEpisode,
    evaluate_episodes,
    score_episode,
)
from agenticrec.ranking.constraints import Catalog, Constraints, Item


def catalog():
    return Catalog([
        Item(1, "Comedy 1995", frozenset({"Comedy"}), 1995),
        Item(2, "Horror 1996", frozenset({"Horror"}), 1996),
        Item(3, "Comedy 2001", frozenset({"Comedy"}), 2001),
        Item(4, "Drama 2000", frozenset({"Drama"}), 2000),
        Item(5, "Comedy 2002", frozenset({"Comedy"}), 2002),
    ])


def public(episode_id="development-001", *, family="explicit_filter"):
    return PublicEpisode(
        episode_id=episode_id,
        split="development",
        layer="structured",
        group_id="user:1",
        template_group_id="development:explicit_filter:0:structured",
        scenario_family=family,
        multi_turn=False,
        initial_input={"schema_version": 1, "constraints": {"k": 2}},
        max_turns=2,
    )


def private(episode_id="development-001", **kwargs):
    values = {
        "episode_id": episode_id,
        "feasible": True,
        "acceptable_item_ids": (1, 3),
        "hard_constraints": Constraints(include_genres=("Comedy",), k=2),
        "requested_k": 2,
        "required_count": 2,
        "expected_statuses": ("OK",),
    }
    values.update(kwargs)
    return PrivateEpisode(**values)


def attempt(episode_id="development-001", **kwargs):
    values = {
        "episode_id": episode_id,
        "status": "OK",
        "item_ids": (1, 3),
        "turns": 1,
        "tool_calls": 2,
        "latency_ms": 10.0,
        "input_tokens": 20,
        "output_tokens": 5,
        "request_count": 1,
    }
    values.update(kwargs)
    return EpisodeAttempt(**values)


def test_valid_success_scores_all_contract_metrics():
    result = score_episode(public(), private(), attempt(), catalog())

    assert result.item_validity == 1
    assert result.constraint_precision == 1
    assert result.fill_at_k == 1
    assert result.strict_success is True
    assert result.preference_update_accuracy is None


def test_empty_output_cannot_pass_a_feasible_episode():
    result = score_episode(public(), private(), attempt(item_ids=()), catalog())

    assert result.item_validity == 0
    assert result.constraint_precision == 0
    assert result.fill_at_k == 0
    assert result.strict_success is False


def test_output_above_requested_k_is_not_strict_success():
    result = score_episode(
        public(), private(required_count=1), attempt(item_ids=(1, 3, 5)), catalog()
    )

    assert result.fill_at_k == 1
    assert result.strict_success is False


def test_invalid_duplicate_and_constraint_violating_ids_are_not_silently_removed():
    result = score_episode(
        public(), private(), attempt(item_ids=(1, 1, 2, 99)), catalog()
    )

    assert result.item_validity == pytest.approx(2 / 4)
    assert result.constraint_precision == pytest.approx(1 / 4)
    assert result.fill_at_k == pytest.approx(0.5)
    assert result.strict_success is False
    assert result.invalid_item_count == 2
    assert result.constraint_violation_count == 1


def test_correct_clarification_succeeds_for_infeasible_episode():
    hidden = private(
        feasible=False,
        acceptable_item_ids=(),
        hard_constraints=Constraints(
            include_genres=("Comedy",), exclude_genres=("comedy",), k=2
        ),
        required_count=0,
        expected_statuses=("CLARIFY_CONFLICT",),
    )
    result = score_episode(
        public(), hidden,
        attempt(status="CLARIFY_CONFLICT", item_ids=(), request_count=0,
                input_tokens=0, output_tokens=0),
        catalog(),
    )

    assert result.strict_success is True
    assert result.item_validity == 1
    assert result.constraint_precision == 1
    assert result.fill_at_k == 0


def test_feedback_patch_uses_annotated_denominator_and_exact_match():
    visible = public(family="exclusion_feedback")
    hidden = private(expected_patch={"disliked_item_ids": [2]})
    correct = score_episode(
        visible, hidden,
        attempt(preference_patch={"disliked_item_ids": [2]}), catalog(),
    )
    missing = score_episode(visible, hidden, attempt(preference_patch=None), catalog())

    assert correct.preference_update_accuracy == 1
    assert missing.preference_update_accuracy == 0


def test_full_denominator_keeps_missing_attempts_and_unknown_usage_null():
    publics = [public("development-001"), public("development-002")]
    privates = [private("development-001"), private("development-002")]
    only_attempt = attempt(
        "development-001", input_tokens=None, output_tokens=None, latency_ms=20
    )

    result = evaluate_episodes(publics, privates, [only_attempt], catalog())

    assert result["episodes_total"] == 2
    assert result["attempts_missing"] == 1
    assert result["strict_success_rate"] == pytest.approx(0.5)
    assert result["fill_at_k"] == pytest.approx(0.5)
    assert result["latency_ms"]["p50"] == 10
    assert result["latency_ms"]["p95"] == 20
    assert result["usage"]["total_tokens"] is None
    assert result["usage"]["episodes_unknown_usage"] == 1
    assert result["usage"]["requests"] == 1


def test_unexpected_attempt_id_and_duplicate_attempt_fail_closed():
    with pytest.raises(ValueError, match="unknown episode"):
        evaluate_episodes([public()], [private()], [attempt("test-999")], catalog())
    with pytest.raises(ValueError, match="duplicate attempt"):
        evaluate_episodes([public()], [private()], [attempt(), attempt()], catalog())
