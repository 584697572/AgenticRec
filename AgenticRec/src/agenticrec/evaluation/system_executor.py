"""Compose frozen benchmark episodes into comparable U1/F/A/O attempts."""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import time
from types import MappingProxyType

from ..agent.loop import AgentLoop
from ..agent.router import RoutingRequest, SystemMode, fixed_request_payload
from ..agent.state import PreferencePatch, PreferenceState
from ..agent.text_parser import TextRequestParser
from ..pipeline import FixedRequest
from ..runtime.budget import BudgetLedger
from ..runtime.errors import (
    BudgetExceeded,
    DeadlineExceeded,
    LLMTransportError,
    LiveCallsDisabled,
    ResponseValidationError,
)
from .episodes import EpisodeAttempt, PrivateEpisode, PublicEpisode


SYSTEMS = frozenset({
    "U1", "F", "A", "O", "no_user_model", "no_collaborative",
    "no_explicit_preference_state", "always_agent", "no_replanning",
})
_ROUTED_SYSTEMS = frozenset({
    "O", "no_user_model", "no_collaborative",
    "no_explicit_preference_state", "no_replanning",
})
_FEEDBACK_KEYS = frozenset({"turn", "kind", "patch"})
_PATCH_KEYS = frozenset({"liked_item_ids", "disliked_item_ids"})


@dataclass(frozen=True)
class TurnResult:
    status: str
    item_ids: tuple[int, ...]
    tool_calls: int
    fallback_kind: str | None = None

    def __post_init__(self):
        if not isinstance(self.status, str) or not self.status.strip():
            raise ValueError("turn status is required")
        if (
            type(self.item_ids) is not tuple
            or any(type(item_id) is not int or item_id <= 0 for item_id in self.item_ids)
            or len(self.item_ids) != len(set(self.item_ids))
        ):
            raise ValueError("turn item_ids must be unique positive integers")
        if type(self.tool_calls) is not int or self.tool_calls < 0:
            raise ValueError("turn tool_calls must be nonnegative")
        if self.fallback_kind is not None and (
            not isinstance(self.fallback_kind, str) or not self.fallback_kind.strip()
        ):
            raise ValueError("fallback_kind must be nonempty or null")


class FeedbackSchedule:
    """Evaluator boundary retaining only releasable feedback, never targets."""

    __slots__ = ("_events", "sha256")

    def __init__(self, events):
        if type(events) is not dict:
            raise TypeError("events must be a mapping")
        normalized = {}
        for key, value in events.items():
            if (
                type(key) is not tuple
                or len(key) != 2
                or not isinstance(key[0], str)
                or not key[0]
                or type(key[1]) is not int
            ):
                raise ValueError("feedback schedule key is invalid")
            event = _feedback_event(value, key[1])
            normalized[key] = event
        rows = [
            {"episode_id": episode_id, "event": normalized[(episode_id, turn)]}
            for episode_id, turn in sorted(normalized)
        ]
        encoded = json.dumps(
            rows, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        self._events = MappingProxyType({
            key: json.dumps(
                value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
            )
            for key, value in normalized.items()
        })
        self.sha256 = hashlib.sha256(encoded).hexdigest()

    @classmethod
    def from_private_episodes(cls, episodes, *, expected_episode_ids):
        episodes = tuple(episodes)
        expected = set(expected_episode_ids)
        if (
            not all(isinstance(episode, PrivateEpisode) for episode in episodes)
            or len({episode.episode_id for episode in episodes}) != len(episodes)
            or {episode.episode_id for episode in episodes} != expected
        ):
            raise ValueError("private evaluator episodes do not match public IDs")
        events = {}
        for episode in episodes:
            for raw in episode.evaluator_events:
                turn = raw.get("turn") if type(raw) is dict else None
                key = (episode.episode_id, turn)
                if key in events:
                    raise ValueError("duplicate feedback event for episode turn")
                events[key] = raw
        return cls(events)

    def __call__(self, episode_id, turn):
        try:
            return json.loads(self._events[(episode_id, turn)])
        except KeyError as error:
            raise KeyError("feedback event is unavailable") from error

    def __repr__(self):
        return (
            "FeedbackSchedule(sha256=" + repr(self.sha256)
            + ", event_count=" + str(len(self._events)) + ")"
        )


def _ids(value, label):
    if (
        type(value) is not list
        or any(type(item_id) is not int or item_id <= 0 for item_id in value)
        or len(value) != len(set(value))
    ):
        raise ValueError(label + " must contain unique positive integer IDs")
    return tuple(value)


def _feedback_event(value, expected_turn):
    if type(value) is not dict or set(value) != _FEEDBACK_KEYS:
        raise ValueError("feedback event must use the exact public turn contract")
    if value["turn"] != expected_turn or type(value["turn"]) is not int:
        raise ValueError("feedback turn does not match the requested turn")
    if value["kind"] != "explicit_feedback":
        raise ValueError("unsupported feedback kind")
    patch = value["patch"]
    if type(patch) is not dict or not patch or set(patch) - _PATCH_KEYS:
        raise ValueError("feedback patch contains unsupported fields")
    normalized = {}
    for name in sorted(_PATCH_KEYS):
        if name in patch:
            normalized[name] = list(_ids(patch[name], name))
    liked = set(normalized.get("liked_item_ids", ()))
    disliked = set(normalized.get("disliked_item_ids", ()))
    if liked & disliked:
        raise ValueError("feedback cannot like and dislike the same item")
    return {
        "turn": expected_turn,
        "kind": "explicit_feedback",
        "patch": normalized,
    }


def _turn_from_payload(payload, *, tool_calls=0):
    if type(payload) is not dict or not isinstance(payload.get("status"), str):
        raise ValueError("system response must contain a status")
    rows = payload.get("recommendations")
    if type(rows) is not list:
        raise ValueError("system response recommendations must be a list")
    item_ids = []
    for row in rows:
        if type(row) is not dict or type(row.get("item_id")) is not int:
            raise ValueError("recommendation must contain an integer item_id")
        item_ids.append(row["item_id"])
    fallback = payload.get("fallback_reason")
    return TurnResult(payload["status"], tuple(item_ids), tool_calls, fallback)


def _turn_from_loop(report):
    value = report.to_dict()
    response = value.get("response")
    if response is None:
        return TurnResult(value["status"], (), value["tool_calls"], None)
    return _turn_from_payload(response, tool_calls=value["tool_calls"])


def _usage_delta(before, after):
    records = after["attempt_records"][before["attempts"]:]
    remote_requests = sum(record["is_live"] for record in records)
    if not records:
        return 0, 0, remote_requests
    if any(record["usage"] is None for record in records):
        return None, None, remote_requests
    return (
        sum(record["usage"]["prompt_tokens"] for record in records),
        sum(record["usage"]["completion_tokens"] for record in records),
        remote_requests,
    )


class BenchmarkSystemExecutor:
    """Callable passed to the append-only runner; evaluator targets stay outside."""

    def __init__(
        self,
        system,
        *,
        fixed_pipeline=None,
        text_parser=None,
        agent_loop=None,
        upstream_turn=None,
        feedback_source=None,
        ledger,
        planned_cost_ceiling=0,
        clock=time.monotonic,
    ):
        if system not in SYSTEMS:
            raise ValueError("unsupported benchmark system")
        if not isinstance(ledger, BudgetLedger):
            raise TypeError("ledger must be BudgetLedger")
        if (
            type(planned_cost_ceiling) not in (int, float)
            or not math.isfinite(planned_cost_ceiling)
            or planned_cost_ceiling < 0
        ):
            raise ValueError("planned_cost_ceiling must be finite and nonnegative")
        if not callable(clock):
            raise TypeError("clock must be callable")
        if feedback_source is not None and not callable(feedback_source):
            raise TypeError("feedback_source must be callable")
        if system == "F":
            if not hasattr(fixed_pipeline, "recommend"):
                raise TypeError("F requires a fixed pipeline")
            if not isinstance(text_parser, TextRequestParser):
                raise TypeError("F requires a metered text parser")
            if text_parser.adapter.ledger is not ledger:
                raise ValueError("text parser and system executor must share a ledger")
        elif system == "U1":
            if not callable(upstream_turn):
                raise TypeError("U1 requires an upstream turn adapter")
        else:
            if not hasattr(fixed_pipeline, "recommend"):
                raise TypeError("Agent systems require a fixed pipeline")
            if not isinstance(agent_loop, AgentLoop):
                raise TypeError("Agent systems require AgentLoop")
            expected = (
                SystemMode.ALWAYS_AGENT
                if system in {"A", "always_agent"}
                else SystemMode.OURS_ROUTER
            )
            if agent_loop.router.policy.system_mode is not expected:
                raise ValueError("system and AgentLoop mode do not match")
            if agent_loop.planner.ledger is not ledger:
                raise ValueError("AgentLoop and system executor must share a ledger")
            if agent_loop.fixed_pipeline is not fixed_pipeline:
                raise ValueError("AgentLoop and executor must share the fixed pipeline")
            if system == "no_replanning" and agent_loop.limits.max_replans != 0:
                raise ValueError("no_replanning requires max_replans=0")
        self.system = system
        self.fixed_pipeline = fixed_pipeline
        self.text_parser = text_parser
        self.agent_loop = agent_loop
        self.upstream_turn = upstream_turn
        self.feedback_source = feedback_source
        self.feedback_sha256 = (
            feedback_source.sha256
            if isinstance(feedback_source, FeedbackSchedule)
            else None
        )
        self.ledger = ledger
        self.planned_cost_ceiling = float(planned_cost_ceiling)
        self._clock = clock

    def __call__(self, episode):
        if not isinstance(episode, PublicEpisode):
            raise TypeError("executor requires PublicEpisode")
        started = self._clock()
        before = self.ledger.snapshot()
        turns = 1
        total_tool_calls = 0
        preference_patch = None
        public_input = deepcopy(episode.initial_input)
        fixed_request = None
        if episode.layer == "structured":
            fixed_request = FixedRequest.parse(public_input)

        try:
            first, fixed_request = self._run_turn(
                episode, public_input, fixed_request, parse_text=True
            )
        except ResponseValidationError:
            return self._attempt(
                episode, "INVALID_PARSE", (), turns, total_tool_calls,
                started, before, None, None,
            )
        except (BudgetExceeded, DeadlineExceeded, LiveCallsDisabled):
            return self._attempt(
                episode, "BUDGET_EXHAUSTED", (), turns, total_tool_calls,
                started, before, None, None,
            )
        except LLMTransportError:
            return self._attempt(
                episode, "LLM_FAILED", (), turns, total_tool_calls,
                started, before, None, None,
            )
        total_tool_calls += first.tool_calls
        if not episode.multi_turn:
            return self._attempt(
                episode, first.status, first.item_ids, turns, total_tool_calls,
                started, before, None, first.fallback_kind,
            )

        turns = 2
        if self.feedback_source is None:
            return self._attempt(
                episode, "MISSING_FEEDBACK", (), turns, total_tool_calls,
                started, before, None, None,
            )
        try:
            event = _feedback_event(
                self.feedback_source(episode.episode_id, turns), turns
            )
        except (KeyError, TypeError, ValueError):
            return self._attempt(
                episode, "INVALID_FEEDBACK", (), turns, total_tool_calls,
                started, before, None, None,
            )

        if self.system != "no_explicit_preference_state":
            preference_patch = deepcopy(event["patch"])
            try:
                fixed_request = self._apply_feedback(
                    episode.episode_id, fixed_request, event
                )
            except (TypeError, ValueError):
                return self._attempt(
                    episode, "INVALID_FEEDBACK", (), turns, total_tool_calls,
                    started, before, None, None,
                )
        public_input = deepcopy(public_input)
        public_input["feedback_event"] = deepcopy(event)
        turns = 3
        try:
            final, fixed_request = self._run_turn(
                episode, public_input, fixed_request, parse_text=False
            )
        except (ResponseValidationError, ValueError):
            return self._attempt(
                episode, "INVALID_FINAL_RESULT", (), turns, total_tool_calls,
                started, before, preference_patch, None,
            )
        except (BudgetExceeded, DeadlineExceeded, LiveCallsDisabled):
            return self._attempt(
                episode, "BUDGET_EXHAUSTED", (), turns, total_tool_calls,
                started, before, preference_patch, None,
            )
        except LLMTransportError:
            return self._attempt(
                episode, "LLM_FAILED", (), turns, total_tool_calls,
                started, before, preference_patch, None,
            )
        total_tool_calls += final.tool_calls
        return self._attempt(
            episode, final.status, final.item_ids, turns, total_tool_calls,
            started, before, preference_patch, final.fallback_kind,
        )

    def _run_turn(self, episode, public_input, fixed_request, *, parse_text):
        if self.system == "F":
            if fixed_request is None:
                if not parse_text:
                    raise ValueError("fixed text state was not established")
                parsed = self.text_parser.parse(
                    episode.initial_input,
                    planned_cost_ceiling=self.planned_cost_ceiling,
                )
                fixed_request = parsed.request
            response = self.fixed_pipeline.recommend(fixed_request).to_dict()
            return _turn_from_payload(response), fixed_request

        request = self._routing_request(
            episode.episode_id, public_input, fixed_request
        )
        if self.system == "U1":
            result = self.upstream_turn(request)
            if not isinstance(result, TurnResult):
                raise TypeError("upstream turn adapter must return TurnResult")
            return result, fixed_request
        return _turn_from_loop(self.agent_loop.run(request)), fixed_request

    def _routing_request(self, request_id, public_input, fixed_request):
        if fixed_request is None:
            return RoutingRequest(
                request_id=request_id,
                public_input=public_input,
                agent_need_score=1.0,
                needs_planning=True,
            )
        visible = fixed_request_payload(fixed_request)
        if "feedback_event" in public_input:
            visible["feedback_event"] = deepcopy(public_input["feedback_event"])
        return RoutingRequest(
            request_id=request_id,
            public_input=visible,
            agent_need_score=0.0,
            fixed_request=fixed_request,
            needs_planning=False,
        )

    def _apply_feedback(self, episode_id, fixed_request, event):
        patch = event["patch"]
        if fixed_request is None:
            return None
        state = PreferenceState.create(
            "benchmark:" + episode_id,
            user_id=fixed_request.user_id,
            history_authorized=fixed_request.history_authorized,
        )
        if fixed_request.liked_item_ids or fixed_request.disliked_item_ids:
            initial = PreferencePatch(
                event_id=episode_id + ":initial",
                source_turn=1,
                source="explicit_user",
                liked_item_ids=fixed_request.liked_item_ids,
                disliked_item_ids=fixed_request.disliked_item_ids,
            )
            state = state.apply(initial).state
        update = PreferencePatch(
            event_id=episode_id + ":turn:" + str(event["turn"]),
            source_turn=event["turn"],
            source="explicit_user",
            liked_item_ids=tuple(patch.get("liked_item_ids", ())),
            disliked_item_ids=tuple(patch.get("disliked_item_ids", ())),
        )
        result = state.apply(update)
        if result.status != "APPLIED":
            raise ValueError("feedback patch was not applied")
        return FixedRequest(
            constraints=fixed_request.constraints,
            user_id=fixed_request.user_id,
            history_authorized=fixed_request.history_authorized,
            liked_item_ids=tuple(sorted(result.state.liked_item_ids)),
            disliked_item_ids=tuple(sorted(result.state.disliked_item_ids)),
            seen_item_ids=fixed_request.seen_item_ids,
        )

    def _attempt(
        self, episode, status, item_ids, turns, tool_calls, started, before,
        preference_patch, fallback_kind,
    ):
        after = self.ledger.snapshot()
        input_tokens, output_tokens, remote_requests = _usage_delta(before, after)
        return EpisodeAttempt(
            episode_id=episode.episode_id,
            status=status,
            item_ids=tuple(item_ids),
            turns=turns,
            tool_calls=tool_calls,
            latency_ms=max(0.0, (self._clock() - started) * 1000.0),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            request_count=remote_requests,
            preference_patch=preference_patch,
            fallback_kind=fallback_kind,
        )
