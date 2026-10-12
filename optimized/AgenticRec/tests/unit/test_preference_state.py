import pytest

from agenticrec.agent.state import (
    PreferencePatch,
    PreferenceState,
    SoftPreferenceSignal,
)


def state_with_history():
    return PreferenceState.create(
        session_id="session-a",
        user_id=7,
        history_item_ids=(10, 11),
        history_authorized=True,
    )


def patch(event_id, turn, **kwargs):
    return PreferencePatch(
        event_id=event_id,
        source_turn=turn,
        source="explicit_user",
        **kwargs,
    )


def test_later_explicit_hard_field_overrides_old_value_with_provenance():
    initial = state_with_history()
    first = initial.apply(patch(
        "e1", 1, hard_updates={"year_min": 1980, "year_max": 2000}
    ))
    second = first.state.apply(patch("e2", 2, hard_updates={"year_min": 1990}))

    assert first.status == "APPLIED" and first.state.version == 1
    assert second.status == "APPLIED" and second.state.version == 2
    assert second.state.hard_constraints.year_min == 1990
    assert second.state.hard_constraints.year_max == 2000
    evidence = second.state.latest_evidence("hard_constraints.year_min")
    assert evidence.event_id == "e2"
    assert evidence.source_turn == 2
    assert evidence.source == "explicit_user"


def test_older_explicit_hard_update_is_ignored_without_mutating_state():
    current = state_with_history().apply(patch(
        "e2", 2, hard_updates={"year_min": 1990}
    )).state

    result = current.apply(patch(
        "e1", 1, hard_updates={"year_min": 1980}, liked_item_ids=(25,)
    ))

    assert result.status == "APPLIED"
    assert result.state.hard_constraints.year_min == 1990
    assert result.state.liked_item_ids == frozenset({25})
    assert result.state.latest_evidence("hard_constraints.year_min").event_id == "e2"
    assert result.ignored_fields == ("hard_constraints.year_min",)


def test_model_inference_cannot_write_or_override_hard_constraints():
    explicit = state_with_history().apply(patch(
        "e1", 1, hard_updates={"exclude_genres": ("Horror",)}
    )).state
    inferred = PreferencePatch(
        event_id="m1",
        source_turn=2,
        source="model_inference",
        hard_updates={"exclude_genres": ()},
    )

    result = explicit.apply(inferred)

    assert result.status == "REJECTED_INFERENCE_HARD_CONSTRAINT"
    assert result.state is explicit
    assert result.state.hard_constraints.exclude_genres == ("Horror",)
    assert result.state.version == 1


def test_one_disliked_item_does_not_expand_to_its_genres():
    initial = state_with_history()
    result = initial.apply(patch("e1", 1, disliked_item_ids=(25,)))

    assert result.state.disliked_item_ids == frozenset({25})
    assert result.state.liked_item_ids == frozenset()
    assert result.state.hard_constraints.exclude_genres == ()
    assert result.state.soft_preferences == ()
    assert result.state.latest_evidence("disliked_item_ids:25").event_id == "e1"


def test_explicit_item_feedback_overrides_inference_and_cannot_be_reversed_by_it():
    initial = state_with_history()
    inferred_like = PreferencePatch(
        event_id="m1", source_turn=1, source="model_inference", liked_item_ids=(25,)
    )
    after_inference = initial.apply(inferred_like).state
    after_dislike = after_inference.apply(patch("e2", 2, disliked_item_ids=(25,))).state
    retry_inference = PreferencePatch(
        event_id="m3", source_turn=3, source="model_inference", liked_item_ids=(25,)
    )
    result = after_dislike.apply(retry_inference)

    assert after_dislike.liked_item_ids == frozenset()
    assert after_dislike.disliked_item_ids == frozenset({25})
    assert result.status == "IGNORED_LOWER_PRIORITY"
    assert result.state.version == after_dislike.version
    assert result.state.disliked_item_ids == frozenset({25})
    assert result.state.latest_evidence("disliked_item_ids:25").source == "explicit_user"


def test_conflicting_hard_patch_requires_clarification_and_is_atomic():
    initial = state_with_history()
    conflicting = patch(
        "e1",
        1,
        hard_updates={
            "include_genres": ("Comedy",),
            "exclude_genres": ("comedy",),
        },
        liked_item_ids=(25,),
    )

    result = initial.apply(conflicting)

    assert result.status == "CLARIFY_CONFLICT"
    assert result.state is initial
    assert result.state.version == 0
    assert result.state.liked_item_ids == frozenset()
    assert result.contradictions == ("genre:comedy",)


def test_event_retry_is_idempotent_and_changed_reuse_is_rejected():
    initial = state_with_history()
    event = patch("e1", 1, liked_item_ids=(25,))
    first = initial.apply(event)
    retry = first.state.apply(event)
    reused = first.state.apply(patch("e1", 1, liked_item_ids=(26,)))

    assert first.status == "APPLIED" and first.state.version == 1
    assert retry.status == "IDEMPOTENT" and retry.state is first.state
    assert retry.state.version == 1
    assert reused.status == "EVENT_ID_REUSE" and reused.state is first.state
    assert reused.state.liked_item_ids == frozenset({25})


def test_effective_feedback_changes_profile_version_for_cache_invalidation():
    initial = state_with_history()
    old_cache_key = (initial.user_id, initial.profile_version, "Comedy", 5)
    event = patch("e1", 1, liked_item_ids=(25,))
    updated = initial.apply(event).state
    new_cache_key = (updated.user_id, updated.profile_version, "Comedy", 5)
    retried = updated.apply(event).state

    assert old_cache_key != new_cache_key
    assert retried.profile_version == updated.profile_version


def test_soft_preference_keeps_evidence_and_explicit_signal_wins():
    inferred = SoftPreferenceSignal("genre", "Comedy", "like", 0.6)
    explicit = SoftPreferenceSignal("genre", "Comedy", "dislike", 1.0)
    state = state_with_history().apply(PreferencePatch(
        event_id="m1", source_turn=1, source="model_inference",
        soft_preferences=(inferred,),
    )).state
    state = state.apply(patch("e2", 2, soft_preferences=(explicit,))).state
    result = state.apply(PreferencePatch(
        event_id="m3", source_turn=3, source="model_inference",
        soft_preferences=(inferred,),
    ))

    assert len(result.state.soft_preferences) == 1
    signal = result.state.soft_preferences[0]
    assert signal.polarity == "dislike"
    assert signal.source == "explicit_user"
    assert signal.event_id == "e2"
    assert result.status == "IGNORED_LOWER_PRIORITY"


def test_clear_session_retains_authorized_training_history_only():
    populated = state_with_history().apply(patch(
        "e1", 1,
        hard_updates={"year_min": 1990},
        liked_item_ids=(25,),
        unresolved_fields=("duration",),
    )).state

    cleared = populated.clear_session()

    assert cleared.history_item_ids == (10, 11)
    assert cleared.user_id == 7 and cleared.session_id == "session-a"
    assert cleared.version == populated.version + 1
    assert cleared.liked_item_ids == frozenset()
    assert cleared.disliked_item_ids == frozenset()
    assert cleared.hard_constraints.year_min is None
    assert cleared.unresolved_fields == ()
    assert all(e.source == "training_history" for e in cleared.evidence)


def test_training_history_requires_authorization_and_patch_schema_fails_closed():
    with pytest.raises(ValueError, match="authorization"):
        PreferenceState.create("session-a", history_item_ids=(10,))
    with pytest.raises(ValueError):
        patch("e1", 1, liked_item_ids=(0,))
    with pytest.raises(ValueError):
        patch("e1", 1, liked_item_ids=(25,), disliked_item_ids=(25,))
    with pytest.raises(ValueError):
        patch("e1", 1, hard_updates={"unknown": 1})
    with pytest.raises(ValueError):
        patch("e1", 1, hard_updates={"include_genres": ["Comedy"]})
