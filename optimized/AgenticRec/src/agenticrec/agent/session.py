"""Thread-safe session state and versioned per-session response cache."""

import asyncio
from copy import deepcopy
from dataclasses import dataclass
from threading import RLock

from .state import PreferencePatch, PreferenceState


def _text(value, label):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(label + " must be a nonempty trimmed string")
    return value


@dataclass(frozen=True)
class SessionRecord:
    session_id: str
    display_name: str | None
    state: PreferenceState

    def __post_init__(self):
        if self.state.session_id != self.session_id:
            raise ValueError("record and preference state have different session IDs")


class SessionStore:
    """Session ID is the sole storage identity; display names may collide."""

    def __init__(self):
        self._records = {}
        self._lock = RLock()

    def create(self, session_id, *, display_name=None, user_id=None,
               history_item_ids=(), history_authorized=False):
        session_id = _text(session_id, "session_id")
        if display_name is not None:
            display_name = _text(display_name, "display_name")
        state = PreferenceState.create(
            session_id,
            user_id=user_id,
            history_item_ids=history_item_ids,
            history_authorized=history_authorized,
        )
        record = SessionRecord(session_id, display_name, state)
        with self._lock:
            if session_id in self._records:
                raise ValueError("session already exists: " + session_id)
            self._records[session_id] = record
        return record

    def get(self, session_id):
        session_id = _text(session_id, "session_id")
        with self._lock:
            try:
                return self._records[session_id]
            except KeyError as error:
                raise KeyError("unknown session: " + session_id) from error

    def apply(self, session_id, patch):
        if not isinstance(patch, PreferencePatch):
            raise TypeError("patch must be PreferencePatch")
        session_id = _text(session_id, "session_id")
        with self._lock:
            record = self._require(session_id)
            result = record.state.apply(patch)
            self._records[session_id] = SessionRecord(
                record.session_id, record.display_name, result.state
            )
            return result

    async def apply_async(self, session_id, patch):
        return await asyncio.to_thread(self.apply, session_id, patch)

    def clear(self, session_id):
        session_id = _text(session_id, "session_id")
        with self._lock:
            record = self._require(session_id)
            state = record.state.clear_session()
            updated = SessionRecord(record.session_id, record.display_name, state)
            self._records[session_id] = updated
            return updated

    def _require(self, session_id):
        try:
            return self._records[session_id]
        except KeyError as error:
            raise KeyError("unknown session: " + session_id) from error


@dataclass(frozen=True)
class SessionCacheKey:
    session_id: str
    profile_version: int
    request_fingerprint: str


class SessionCache:
    """Copies values and never shares entries across session/version boundaries."""

    def __init__(self):
        self._entries = {}
        self._lock = RLock()

    def get(self, state, request_fingerprint):
        key = self._key(state, request_fingerprint)
        with self._lock:
            value = self._entries.get(key)
            return None if value is None else deepcopy(value)

    def put(self, state, request_fingerprint, value):
        key = self._key(state, request_fingerprint)
        stored = deepcopy(value)
        with self._lock:
            stale = [
                existing for existing in self._entries
                if (existing.session_id == key.session_id
                    and existing.profile_version != key.profile_version)
            ]
            for existing in stale:
                del self._entries[existing]
            self._entries[key] = stored

    def evict_session(self, session_id):
        session_id = _text(session_id, "session_id")
        with self._lock:
            keys = [key for key in self._entries if key.session_id == session_id]
            for key in keys:
                del self._entries[key]
            return len(keys)

    def _key(self, state, request_fingerprint):
        if not isinstance(state, PreferenceState):
            raise TypeError("state must be PreferenceState")
        request_fingerprint = _text(request_fingerprint, "request_fingerprint")
        return SessionCacheKey(
            state.session_id, state.profile_version, request_fingerprint
        )
