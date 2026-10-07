"""Fail-closed, resumable runner for frozen public benchmark episodes."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re

from .episodes import EpisodeAttempt, load_public_episodes


ROOT = Path(__file__).resolve().parents[4]
SYSTEMS = frozenset((
    "U1", "F", "A", "O", "no_user_model", "no_collaborative",
    "no_explicit_preference_state", "always_agent", "no_replanning",
))
RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class RunAuthorizationError(RuntimeError):
    pass


class UnresolvedAttemptError(RuntimeError):
    pass


class RunLockedError(RuntimeError):
    pass


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value):
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _event_hash(value):
    payload = dict(value)
    payload.pop("event_hash", None)
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def _workspace_path(value, label):
    if not isinstance(value, (str, Path)):
        raise ValueError(label + " must be a workspace path")
    path = (ROOT / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError(label + " must stay inside the workspace") from error
    return path


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    benchmark_id: str
    system: str
    seed: int
    split: str
    public_episodes: str
    public_sha256: str
    episode_count: int
    authorized_request_cap: int
    money_budget_cny: float
    per_request_cost_ceiling_cny: float
    live: bool
    feedback_sha256: str | None = None
    schema_version: int = 1

    def __post_init__(self):
        if self.schema_version != 1 or type(self.schema_version) is not int:
            raise ValueError("schema_version must be integer 1")
        if not isinstance(self.run_id, str) or not RUN_ID.fullmatch(self.run_id):
            raise ValueError("run_id has an invalid format")
        if not isinstance(self.benchmark_id, str) or not self.benchmark_id.strip():
            raise ValueError("benchmark_id must be nonempty")
        if self.system not in SYSTEMS:
            raise ValueError("unsupported benchmark system")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.split not in ("development", "test"):
            raise ValueError("split must be development or test")
        if not isinstance(self.public_episodes, str) or not self.public_episodes.strip():
            raise ValueError("public_episodes must be a nonempty path")
        if (not isinstance(self.public_sha256, str)
                or not re.fullmatch(r"[0-9a-f]{64}", self.public_sha256)):
            raise ValueError("public_sha256 must be lowercase SHA256")
        for name in ("episode_count", "authorized_request_cap"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(name + " must be a nonnegative integer")
        if self.episode_count < 1:
            raise ValueError("episode_count must be positive")
        for name in ("money_budget_cny", "per_request_cost_ceiling_cny"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(name + " must be finite and nonnegative")
        if type(self.live) is not bool:
            raise ValueError("live must be boolean")
        if self.feedback_sha256 is not None and (
                not isinstance(self.feedback_sha256, str)
                or not re.fullmatch(r"[0-9a-f]{64}", self.feedback_sha256)):
            raise ValueError("feedback_sha256 must be lowercase SHA256 or null")
        if self.live and (self.authorized_request_cap < 1
                          or self.money_budget_cny <= 0
                          or self.per_request_cost_ceiling_cny <= 0):
            raise ValueError("live runs require positive request and money ceilings")
        if not self.live and (self.money_budget_cny != 0
                              or self.per_request_cost_ceiling_cny != 0):
            raise ValueError("offline runs require zero money ceilings")

    @property
    def identity(self):
        return hashlib.sha256(_canonical(asdict(self)).encode("utf-8")).hexdigest()


def episode_request_ceiling(system, episode):
    if system == "F":
        return episode.max_turns if episode.layer == "text" else 0
    if system in ("U1", "no_replanning"):
        return episode.max_turns
    if system in (
            "A", "O", "no_user_model", "no_collaborative",
            "no_explicit_preference_state", "always_agent"):
        return episode.max_turns * 2
    raise ValueError("unsupported benchmark system")


def _load_public(spec):
    path = _workspace_path(spec.public_episodes, "public_episodes")
    if not path.is_file() or _sha256(path) != spec.public_sha256:
        raise ValueError("frozen public episode hash mismatch")
    episodes = load_public_episodes(path)
    if len(episodes) != spec.episode_count:
        raise ValueError("frozen public episode count mismatch")
    if any(episode.split != spec.split for episode in episodes):
        raise ValueError("frozen public episode split mismatch")
    if len({episode.episode_id for episode in episodes}) != len(episodes):
        raise ValueError("duplicate public episode ID")
    return path, episodes


def _append_event(path, event, previous_hash):
    value = dict(event)
    value["sequence"] = event["sequence"]
    value["previous_hash"] = previous_hash
    value["event_hash"] = _event_hash(value)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(_canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return value["event_hash"]


def _load_events(path):
    if not path.exists():
        return []
    events = []
    previous_hash = None
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError("journal contains invalid JSON") from error
        if (type(event) is not dict or event.get("sequence") != index
                or event.get("previous_hash") != previous_hash
                or event.get("event_hash") != _event_hash(event)):
            raise ValueError("journal hash chain validation failed")
        previous_hash = event["event_hash"]
        events.append(event)
    return events


def _journal_state(events, spec):
    if not events:
        return set(), {}, 0
    header = events[0]
    if (header.get("event") != "HEADER" or header.get("run_identity") != spec.identity
            or header.get("private_targets_loaded") is not False):
        raise ValueError("journal run identity does not match the requested run")
    started = {}
    completed = set()
    remote = 0
    for event in events[1:]:
        episode_id = event.get("episode_id")
        if event.get("event") == "STARTED":
            if episode_id in started or episode_id in completed:
                raise ValueError("journal contains a duplicate episode start")
            started[episode_id] = event
        elif event.get("event") == "COMPLETED":
            if episode_id not in started or episode_id in completed:
                raise ValueError("journal completion has no unique start")
            completed.add(episode_id)
            remote += event["attempt"]["request_count"]
            started.pop(episode_id)
        elif event.get("event") == "INTERRUPTED":
            if episode_id not in started:
                raise ValueError("journal interruption has no active start")
        else:
            raise ValueError("journal contains an unsupported event")
    return completed, started, remote


def _report(spec, episodes, ceilings, completed, remote, *, status, resumed):
    return {
        "schema_version": 1,
        "run_id": spec.run_id,
        "benchmark_id": spec.benchmark_id,
        "system": spec.system,
        "seed": spec.seed,
        "split": spec.split,
        "status": status,
        "resumed": resumed,
        "episodes_total": len(episodes),
        "completed_episodes": len(completed),
        "remaining_episodes": len(episodes) - len(completed),
        "episode_request_ceiling": ceilings,
        "required_request_ceiling": sum(ceilings.values()),
        "authorized_request_cap": spec.authorized_request_cap,
        "planned_cost_ceiling_cny": (
            sum(ceilings.values()) * spec.per_request_cost_ceiling_cny
        ),
        "money_budget_cny": spec.money_budget_cny,
        "remote_api_requests": remote,
        "private_targets_loaded": False,
        "public_sha256": spec.public_sha256,
        "feedback_sha256": spec.feedback_sha256,
        "run_identity": spec.identity,
    }


def run_benchmark(spec, journal_path, executor, *, dry_run=False):
    if not isinstance(spec, RunSpec):
        raise TypeError("spec must be RunSpec")
    if not callable(executor):
        raise TypeError("executor must be callable")
    if type(dry_run) is not bool:
        raise ValueError("dry_run must be boolean")
    _public_path, episodes = _load_public(spec)
    if any(episode.multi_turn for episode in episodes) and spec.feedback_sha256 is None:
        raise ValueError("multi-turn runs require a feedback schedule hash")
    executor_feedback_sha = getattr(executor, "feedback_sha256", spec.feedback_sha256)
    if (not dry_run and any(episode.multi_turn for episode in episodes)
            and executor_feedback_sha != spec.feedback_sha256):
        raise ValueError("executor feedback schedule does not match RunSpec")
    ceilings = {
        episode.episode_id: episode_request_ceiling(spec.system, episode)
        for episode in episodes
    }
    required_requests = sum(ceilings.values())
    required_money = required_requests * spec.per_request_cost_ceiling_cny
    if required_requests > spec.authorized_request_cap:
        raise RunAuthorizationError("authorization does not cover the full run ceiling")
    if required_money > spec.money_budget_cny + 1e-12:
        raise RunAuthorizationError("money authorization does not cover the full run ceiling")
    if dry_run:
        return _report(
            spec, episodes, ceilings, set(), 0, status="READY", resumed=False
        )

    journal_path = _workspace_path(journal_path, "journal_path")
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = journal_path.with_suffix(journal_path.suffix + ".lock")
    try:
        lock_path.open("x", encoding="utf-8").close()
    except FileExistsError as error:
        raise RunLockedError("benchmark journal is locked") from error
    try:
        events = _load_events(journal_path)
        resumed = bool(events)
        if not events:
            previous_hash = _append_event(journal_path, {
                "sequence": 1,
                "event": "HEADER",
                "run_identity": spec.identity,
                "run_id": spec.run_id,
                "benchmark_id": spec.benchmark_id,
                "system": spec.system,
                "seed": spec.seed,
                "split": spec.split,
                "public_sha256": spec.public_sha256,
                "feedback_sha256": spec.feedback_sha256,
                "private_targets_loaded": False,
                "authorized_request_cap": spec.authorized_request_cap,
                "money_budget_cny": spec.money_budget_cny,
                "per_request_cost_ceiling_cny": spec.per_request_cost_ceiling_cny,
            }, None)
            events = _load_events(journal_path)
        else:
            previous_hash = events[-1]["event_hash"]
        completed, unresolved, remote = _journal_state(events, spec)
        if unresolved:
            episode_id = next(iter(unresolved))
            raise UnresolvedAttemptError(
                "unresolved paid-attempt state for episode " + episode_id
            )

        known_ids = {episode.episode_id for episode in episodes}
        if not completed <= known_ids:
            raise ValueError("journal contains an unknown episode ID")
        for episode in episodes:
            if episode.episode_id in completed:
                continue
            sequence = len(events) + 1
            previous_hash = _append_event(journal_path, {
                "sequence": sequence,
                "event": "STARTED",
                "episode_id": episode.episode_id,
                "reserved_requests": ceilings[episode.episode_id],
                "reserved_cost_ceiling_cny": (
                    ceilings[episode.episode_id] * spec.per_request_cost_ceiling_cny
                ),
            }, previous_hash)
            events.append(json.loads(journal_path.read_text(encoding="utf-8").splitlines()[-1]))
            try:
                attempt = executor(episode)
                if not isinstance(attempt, EpisodeAttempt):
                    raise TypeError("executor must return EpisodeAttempt")
                if attempt.episode_id != episode.episode_id:
                    raise ValueError("attempt episode ID does not match public episode")
                if attempt.request_count > ceilings[episode.episode_id]:
                    raise ValueError("attempt exceeded its reserved request ceiling")
            except Exception as error:
                previous_hash = _append_event(journal_path, {
                    "sequence": len(events) + 1,
                    "event": "INTERRUPTED",
                    "episode_id": episode.episode_id,
                    "error_type": type(error).__name__,
                }, previous_hash)
                raise
            attempt_value = asdict(attempt)
            attempt_value["item_ids"] = list(attempt.item_ids)
            previous_hash = _append_event(journal_path, {
                "sequence": len(events) + 1,
                "event": "COMPLETED",
                "episode_id": episode.episode_id,
                "attempt": attempt_value,
            }, previous_hash)
            events = _load_events(journal_path)
            completed, unresolved, remote = _journal_state(events, spec)
        return _report(
            spec, episodes, ceilings, completed, remote,
            status="COMPLETE", resumed=resumed,
        )
    finally:
        if lock_path.exists():
            lock_path.unlink()


def attempt_from_loop_report(episode_id, report, *, turns=1, preference_patch=None):
    """Convert an AgentLoop report to the evaluator's immutable attempt schema."""
    if not isinstance(episode_id, str) or not episode_id:
        raise ValueError("episode_id is required")
    if not hasattr(report, "to_dict"):
        raise TypeError("report must provide to_dict")
    if type(turns) is not int or turns < 0:
        raise ValueError("turns must be a nonnegative integer")
    value = report.to_dict()
    response = value.get("response")
    recommendations = response.get("recommendations", []) if isinstance(response, dict) else []
    item_ids = []
    for row in recommendations:
        if type(row) is not dict or type(row.get("item_id")) is not int:
            raise ValueError("loop report recommendation is missing an integer item_id")
        item_ids.append(row["item_id"])
    usage = value.get("budget", {}).get("usage")
    return EpisodeAttempt(
        episode_id=episode_id,
        status=value["status"],
        item_ids=tuple(item_ids),
        turns=turns,
        tool_calls=value["tool_calls"],
        latency_ms=value["elapsed_ms"],
        input_tokens=None if usage is None else usage["prompt_tokens"],
        output_tokens=None if usage is None else usage["completion_tokens"],
        request_count=value["remote_api_requests"],
        preference_patch=preference_patch,
        fallback_kind=(response.get("fallback_reason")
                       if isinstance(response, dict) else None),
    )


def load_completed_attempts(spec, journal_path):
    """Load only completed public attempts; evaluator targets are handled elsewhere."""
    if not isinstance(spec, RunSpec):
        raise TypeError("spec must be RunSpec")
    journal_path = _workspace_path(journal_path, "journal_path")
    events = _load_events(journal_path)
    completed, unresolved, _remote = _journal_state(events, spec)
    if unresolved:
        raise UnresolvedAttemptError("journal contains an unresolved attempt")
    attempts = []
    for event in events:
        if event.get("event") != "COMPLETED":
            continue
        value = dict(event["attempt"])
        value["item_ids"] = tuple(value["item_ids"])
        attempts.append(EpisodeAttempt(**value))
    if len(attempts) != len(completed):
        raise ValueError("journal completed-attempt count mismatch")
    return tuple(attempts)
