"""T17 same-name, async feedback and cache isolation regressions."""

import asyncio

import pytest

from agenticrec.agent.session import SessionCache, SessionStore
from agenticrec.agent.state import PreferencePatch


def patch(event_id, item_id, turn=1):
    return PreferencePatch(
        event_id=event_id,
        source_turn=turn,
        source="explicit_user",
        liked_item_ids=(item_id,),
    )


def test_same_display_name_is_not_an_identity_or_storage_key():
    store = SessionStore()
    first = store.create("session-1", display_name="Alex", user_id=11)
    second = store.create("session-2", display_name="Alex", user_id=22)
    store.apply("session-1", patch("event-1", 101))

    assert first.display_name == second.display_name == "Alex"
    assert store.get("session-1").state.liked_item_ids == frozenset({101})
    assert store.get("session-2").state.liked_item_ids == frozenset()
    assert store.get("session-1").state.user_id == 11
    assert store.get("session-2").state.user_id == 22


def test_async_feedback_to_distinct_sessions_never_crosses_state():
    store = SessionStore()
    store.create("session-a", display_name="same")
    store.create("session-b", display_name="same")

    async def apply_both():
        return await asyncio.gather(
            store.apply_async("session-a", patch("a1", 201)),
            store.apply_async("session-b", patch("b1", 301)),
        )

    results = asyncio.run(apply_both())

    assert [result.status for result in results] == ["APPLIED", "APPLIED"]
    assert store.get("session-a").state.liked_item_ids == frozenset({201})
    assert store.get("session-b").state.liked_item_ids == frozenset({301})


def test_concurrent_feedback_on_one_session_is_atomic_without_lost_updates():
    store = SessionStore()
    store.create("session-a")

    async def apply_all():
        return await asyncio.gather(*(
            store.apply_async(
                "session-a", patch(f"event-{index}", 400 + index, turn=index)
            )
            for index in range(1, 33)
        ))

    results = asyncio.run(apply_all())
    state = store.get("session-a").state

    assert all(result.status == "APPLIED" for result in results)
    assert state.liked_item_ids == frozenset(range(401, 433))
    assert state.profile_version == 32
    assert {event_id for event_id, _digest in state.applied_events} == {
        f"event-{index}" for index in range(1, 33)
    }


def test_cache_key_includes_session_and_profile_version_and_copies_values():
    store = SessionStore()
    cache = SessionCache()
    first = store.create("session-a", display_name="Alex").state
    second = store.create("session-b", display_name="Alex").state
    value = {"recommendations": [1, 2]}

    cache.put(first, "query:action", value)
    value["recommendations"].append(999)

    assert cache.get(first, "query:action") == {"recommendations": [1, 2]}
    assert cache.get(second, "query:action") is None
    copied = cache.get(first, "query:action")
    copied["recommendations"].append(888)
    assert cache.get(first, "query:action") == {"recommendations": [1, 2]}

    updated = store.apply("session-a", patch("e1", 501)).state
    assert cache.get(updated, "query:action") is None
    cache.put(updated, "query:action", {"recommendations": [3]})
    assert cache.get(updated, "query:action") == {"recommendations": [3]}
    assert cache.get(first, "query:action") is None


def test_clear_and_unknown_session_operations_fail_closed_and_stay_local():
    store = SessionStore()
    store.create("session-a")
    store.create("session-b")
    store.apply("session-a", patch("a1", 601))
    store.apply("session-b", patch("b1", 701))

    cleared = store.clear("session-a")

    assert cleared.state.liked_item_ids == frozenset()
    assert store.get("session-b").state.liked_item_ids == frozenset({701})
    with pytest.raises(KeyError, match="unknown session"):
        store.get("missing")
    with pytest.raises(KeyError, match="unknown session"):
        store.apply("missing", patch("x1", 801))


def test_duplicate_session_id_is_rejected_even_if_display_name_differs():
    store = SessionStore()
    store.create("session-a", display_name="first")

    with pytest.raises(ValueError, match="already exists"):
        store.create("session-a", display_name="second")
