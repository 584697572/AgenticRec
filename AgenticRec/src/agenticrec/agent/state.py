"""Immutable, provenance-bearing preference state for multi-turn sessions."""

from copy import deepcopy
from dataclasses import dataclass, fields
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping

from ..ranking.constraints import Constraints


PATCH_SOURCES = frozenset(("explicit_user", "model_inference"))
SOURCE_PRIORITY = {"model_inference": 1, "explicit_user": 2, "training_history": 3}
HARD_FIELDS = frozenset(field.name for field in fields(Constraints))


def _text(value, label):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(label + " must be a nonempty trimmed string")
    return value


def _positive_ids(values, label):
    if (type(values) is not tuple
            or any(type(value) is not int or value <= 0 for value in values)
            or len(set(values)) != len(values)):
        raise ValueError(label + " must contain unique positive integer IDs")
    return values


def _unique_text(values, label):
    if (type(values) is not tuple
            or any(not isinstance(value, str) or not value.strip() for value in values)
            or len({value.casefold() for value in values}) != len(values)):
        raise ValueError(label + " must contain unique nonempty strings")
    return values


@dataclass(frozen=True)
class SoftPreferenceSignal:
    dimension: str
    value: str
    polarity: str
    confidence: float

    def __post_init__(self):
        object.__setattr__(self, "dimension", _text(self.dimension, "dimension"))
        object.__setattr__(self, "value", _text(self.value, "value"))
        if self.polarity not in {"like", "dislike"}:
            raise ValueError("polarity must be like or dislike")
        if (type(self.confidence) not in (int, float)
                or not math.isfinite(self.confidence)
                or not 0 <= self.confidence <= 1):
            raise ValueError("confidence must be finite and in [0, 1]")
        object.__setattr__(self, "confidence", float(self.confidence))

    @property
    def key(self):
        return self.dimension.casefold(), self.value.casefold()


@dataclass(frozen=True)
class StoredSoftPreference:
    dimension: str
    value: str
    polarity: str
    confidence: float
    source_turn: int
    source: str
    event_id: str

    @property
    def key(self):
        return self.dimension.casefold(), self.value.casefold()


@dataclass(frozen=True)
class PreferenceEvidence:
    field: str
    value: Any
    source_turn: int
    source: str
    confidence: float
    event_id: str


@dataclass(frozen=True)
class PreferencePatch:
    event_id: str
    source_turn: int
    source: str
    hard_updates: Mapping[str, Any] = None
    liked_item_ids: tuple[int, ...] = ()
    disliked_item_ids: tuple[int, ...] = ()
    soft_preferences: tuple[SoftPreferenceSignal, ...] = ()
    unresolved_fields: tuple[str, ...] = ()
    resolved_fields: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "event_id", _text(self.event_id, "event_id"))
        if type(self.source_turn) is not int or self.source_turn <= 0:
            raise ValueError("source_turn must be a positive integer")
        if self.source not in PATCH_SOURCES:
            raise ValueError("unsupported patch source")
        hard_updates = {} if self.hard_updates is None else self.hard_updates
        if not isinstance(hard_updates, dict):
            raise ValueError("hard_updates must be an object")
        unknown = set(hard_updates) - HARD_FIELDS
        if unknown:
            raise ValueError("unknown hard constraint fields: " + ", ".join(sorted(unknown)))
        _updated_constraints(Constraints(), hard_updates)
        object.__setattr__(self, "hard_updates", MappingProxyType(deepcopy(hard_updates)))
        _positive_ids(self.liked_item_ids, "liked_item_ids")
        _positive_ids(self.disliked_item_ids, "disliked_item_ids")
        if set(self.liked_item_ids) & set(self.disliked_item_ids):
            raise ValueError("one patch cannot both like and dislike an item")
        if (type(self.soft_preferences) is not tuple
                or any(not isinstance(value, SoftPreferenceSignal) for value in self.soft_preferences)
                or len({value.key for value in self.soft_preferences}) != len(self.soft_preferences)):
            raise ValueError("soft_preferences must contain unique SoftPreferenceSignal keys")
        _unique_text(self.unresolved_fields, "unresolved_fields")
        _unique_text(self.resolved_fields, "resolved_fields")
        if ({value.casefold() for value in self.unresolved_fields}
                & {value.casefold() for value in self.resolved_fields}):
            raise ValueError("one patch cannot resolve and mark the same field unresolved")
        if not (hard_updates or self.liked_item_ids or self.disliked_item_ids
                or self.soft_preferences or self.unresolved_fields or self.resolved_fields):
            raise ValueError("preference patch must contain an update")

    def digest(self):
        payload = {
            "event_id": self.event_id,
            "source_turn": self.source_turn,
            "source": self.source,
            "hard_updates": dict(self.hard_updates),
            "liked_item_ids": self.liked_item_ids,
            "disliked_item_ids": self.disliked_item_ids,
            "soft_preferences": [
                {
                    "dimension": signal.dimension,
                    "value": signal.value,
                    "polarity": signal.polarity,
                    "confidence": signal.confidence,
                }
                for signal in self.soft_preferences
            ],
            "unresolved_fields": self.unresolved_fields,
            "resolved_fields": self.resolved_fields,
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class PatchResult:
    status: str
    state: "PreferenceState"
    changed: bool
    contradictions: tuple[str, ...] = ()
    ignored_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class PreferenceState:
    session_id: str
    user_id: int | None
    version: int
    hard_constraints: Constraints
    liked_item_ids: frozenset[int]
    disliked_item_ids: frozenset[int]
    soft_preferences: tuple[StoredSoftPreference, ...]
    history_item_ids: tuple[int, ...]
    evidence: tuple[PreferenceEvidence, ...]
    unresolved_fields: tuple[str, ...]
    contradictions: tuple[str, ...]
    applied_events: tuple[tuple[str, str], ...]

    @classmethod
    def create(cls, session_id, *, user_id=None, history_item_ids=(),
               history_authorized=False):
        session_id = _text(session_id, "session_id")
        if user_id is not None and (type(user_id) is not int or user_id <= 0):
            raise ValueError("user_id must be a positive integer or null")
        history_item_ids = _positive_ids(tuple(history_item_ids), "history_item_ids")
        if history_item_ids and history_authorized is not True:
            raise ValueError("training history requires explicit authorization")
        evidence = tuple(
            PreferenceEvidence(
                field=f"history_item_ids:{item_id}",
                value=item_id,
                source_turn=0,
                source="training_history",
                confidence=1.0,
                event_id="initial_history",
            )
            for item_id in history_item_ids
        )
        return cls(
            session_id=session_id,
            user_id=user_id,
            version=0,
            hard_constraints=Constraints(),
            liked_item_ids=frozenset(),
            disliked_item_ids=frozenset(),
            soft_preferences=(),
            history_item_ids=history_item_ids,
            evidence=evidence,
            unresolved_fields=(),
            contradictions=(),
            applied_events=(),
        )

    @property
    def profile_version(self):
        return self.version

    def latest_evidence(self, field):
        for item in reversed(self.evidence):
            if item.field == field:
                return item
        return None

    def apply(self, patch):
        if not isinstance(patch, PreferencePatch):
            raise TypeError("patch must be PreferencePatch")
        digest = patch.digest()
        prior_events = dict(self.applied_events)
        if patch.event_id in prior_events:
            status = "IDEMPOTENT" if prior_events[patch.event_id] == digest else "EVENT_ID_REUSE"
            return PatchResult(status, self, False)
        if patch.source == "model_inference" and patch.hard_updates:
            return PatchResult("REJECTED_INFERENCE_HARD_CONSTRAINT", self, False)

        accepted_hard_updates = {}
        ignored = []
        for name, value in patch.hard_updates.items():
            field = "hard_constraints." + name
            existing = self.latest_evidence(field)
            if existing is not None and not _incoming_wins(
                    patch.source, patch.source_turn, existing.source, existing.source_turn):
                ignored.append(field)
                continue
            accepted_hard_updates[name] = value

        proposed_constraints = _updated_constraints(
            self.hard_constraints, accepted_hard_updates
        )
        conflicts = _constraint_conflicts(proposed_constraints)
        if conflicts:
            return PatchResult(
                "CLARIFY_CONFLICT", self, False, conflicts,
                ignored_fields=tuple(ignored),
            )

        liked = set(self.liked_item_ids)
        disliked = set(self.disliked_item_ids)
        soft = {item.key: item for item in self.soft_preferences}
        evidence = list(self.evidence)
        unresolved = {field.casefold(): field for field in self.unresolved_fields}
        changed = False

        for name, value in accepted_hard_updates.items():
            field = "hard_constraints." + name
            evidence.append(_evidence(field, deepcopy(value), patch, 1.0))
            changed = True

        for item_id in patch.liked_item_ids:
            accepted = _apply_item_feedback(
                item_id, True, patch, liked, disliked, evidence, self
            )
            if accepted:
                changed = True
            else:
                ignored.append(f"liked_item_ids:{item_id}")
        for item_id in patch.disliked_item_ids:
            accepted = _apply_item_feedback(
                item_id, False, patch, liked, disliked, evidence, self
            )
            if accepted:
                changed = True
            else:
                ignored.append(f"disliked_item_ids:{item_id}")

        for signal in patch.soft_preferences:
            field = f"soft_preferences:{signal.key[0]}:{signal.key[1]}"
            current = soft.get(signal.key)
            if current is not None and not _incoming_wins(
                    patch.source, patch.source_turn, current.source, current.source_turn):
                ignored.append(field)
                continue
            stored = StoredSoftPreference(
                dimension=signal.dimension,
                value=signal.value,
                polarity=signal.polarity,
                confidence=signal.confidence,
                source_turn=patch.source_turn,
                source=patch.source,
                event_id=patch.event_id,
            )
            soft[signal.key] = stored
            evidence.append(_evidence(field, signal.polarity, patch, signal.confidence))
            changed = True

        for field in patch.unresolved_fields:
            key = field.casefold()
            if key not in unresolved:
                unresolved[key] = field
                evidence.append(_evidence("unresolved_fields:" + key, True, patch, 1.0))
                changed = True
        for field in patch.resolved_fields:
            key = field.casefold()
            if key in unresolved:
                del unresolved[key]
                evidence.append(_evidence("unresolved_fields:" + key, False, patch, 1.0))
                changed = True

        events = self.applied_events + ((patch.event_id, digest),)
        new_state = PreferenceState(
            session_id=self.session_id,
            user_id=self.user_id,
            version=self.version + (1 if changed else 0),
            hard_constraints=(
                proposed_constraints if accepted_hard_updates else self.hard_constraints
            ),
            liked_item_ids=frozenset(liked),
            disliked_item_ids=frozenset(disliked),
            soft_preferences=tuple(sorted(soft.values(), key=lambda item: item.key)),
            history_item_ids=self.history_item_ids,
            evidence=tuple(evidence),
            unresolved_fields=tuple(unresolved[key] for key in sorted(unresolved)),
            contradictions=(),
            applied_events=events,
        )
        status = "APPLIED" if changed else "IGNORED_LOWER_PRIORITY"
        return PatchResult(status, new_state, changed, ignored_fields=tuple(ignored))

    def clear_session(self):
        history_evidence = tuple(
            item for item in self.evidence if item.source == "training_history"
        )
        return PreferenceState(
            session_id=self.session_id,
            user_id=self.user_id,
            version=self.version + 1,
            hard_constraints=Constraints(),
            liked_item_ids=frozenset(),
            disliked_item_ids=frozenset(),
            soft_preferences=(),
            history_item_ids=self.history_item_ids,
            evidence=history_evidence,
            unresolved_fields=(),
            contradictions=(),
            applied_events=self.applied_events,
        )


def _evidence(field, value, patch, confidence):
    return PreferenceEvidence(
        field=field,
        value=value,
        source_turn=patch.source_turn,
        source=patch.source,
        confidence=float(confidence),
        event_id=patch.event_id,
    )


def _incoming_wins(incoming_source, incoming_turn, existing_source, existing_turn):
    incoming_priority = SOURCE_PRIORITY[incoming_source]
    existing_priority = SOURCE_PRIORITY[existing_source]
    return (incoming_priority > existing_priority
            or (incoming_priority == existing_priority and incoming_turn >= existing_turn))


def _apply_item_feedback(item_id, liked_signal, patch, liked, disliked, evidence, state):
    if item_id in liked:
        current_field = f"liked_item_ids:{item_id}"
    elif item_id in disliked:
        current_field = f"disliked_item_ids:{item_id}"
    else:
        current_field = None
    if current_field is not None:
        current = state.latest_evidence(current_field)
        if current is not None and not _incoming_wins(
                patch.source, patch.source_turn, current.source, current.source_turn):
            return False
    if liked_signal:
        disliked.discard(item_id)
        liked.add(item_id)
        field = f"liked_item_ids:{item_id}"
        value = "like"
    else:
        liked.discard(item_id)
        disliked.add(item_id)
        field = f"disliked_item_ids:{item_id}"
        value = "dislike"
    evidence.append(_evidence(field, value, patch, 1.0))
    return True


def _updated_constraints(current, updates):
    values = {field.name: getattr(current, field.name) for field in fields(Constraints)}
    values.update(dict(updates))
    candidate = Constraints(**values)
    _validate_constraints(candidate)
    return candidate


def _validate_constraints(value):
    _unique_text(value.include_genres, "include_genres")
    _unique_text(value.exclude_genres, "exclude_genres")
    _unique_text(value.required_fields, "required_fields")
    _positive_ids(value.excluded_item_ids, "excluded_item_ids")
    if type(value.exclude_seen) is not bool:
        raise ValueError("exclude_seen must be boolean")
    if type(value.k) is not int or value.k <= 0:
        raise ValueError("k must be a positive integer")
    for name in ("year_min", "year_max"):
        year = getattr(value, name)
        if year is not None and (type(year) is not int or year <= 0):
            raise ValueError(name + " must be a positive integer or null")


def _constraint_conflicts(value):
    include = {genre.casefold() for genre in value.include_genres}
    exclude = {genre.casefold() for genre in value.exclude_genres}
    conflicts = ["genre:" + genre for genre in sorted(include & exclude)]
    if (value.year_min is not None and value.year_max is not None
            and value.year_min > value.year_max):
        conflicts.append("year_range")
    return tuple(conflicts)
