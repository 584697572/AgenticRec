"""Frozen interactive episodes, leakage guards, and full-denominator metrics."""

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
from typing import Any

from ..ranking.constraints import Catalog, ConstraintGate, Constraints


ROOT = Path(__file__).resolve().parents[4]
SPLITS = frozenset(("development", "test"))
LAYERS = frozenset(("structured", "text"))
SCENARIO_FAMILIES = frozenset((
    "explicit_filter", "personalized", "seed_similar", "exclusion_feedback",
    "constraint_conflict", "unknown_attribute", "cold_start", "tool_error",
))
PUBLIC_FORBIDDEN_KEYS = frozenset((
    "acceptable_item_ids", "target_item_ids", "target_item_id", "target_title",
    "expected_statuses", "expected_patch", "evaluator_events", "fault",
    "hard_constraints", "hidden_preferences", "hidden_target",
))
PUBLIC_ITEM_ID_KEYS = frozenset((
    "liked_item_ids", "disliked_item_ids", "seen_item_ids", "excluded_item_ids",
    "seed_item_ids", "candidate_item_ids",
))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text(value, label):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(label + " must be a nonempty trimmed string")
    return value


def _positive_int(value, label):
    if type(value) is not int or value <= 0:
        raise ValueError(label + " must be a positive integer")
    return value


def _positive_ids(values, label):
    if (type(values) is not tuple
            or any(type(value) is not int or value <= 0 for value in values)
            or len(values) != len(set(values))):
        raise ValueError(label + " must contain unique positive integer IDs")
    return values


def _string_tuple(values, label):
    if (type(values) is not tuple
            or any(not isinstance(value, str) or not value.strip() for value in values)
            or len(values) != len(set(values))):
        raise ValueError(label + " must contain unique nonempty strings")
    return values


def _json_object(value, label):
    if type(value) is not dict:
        raise ValueError(label + " must be a JSON object")
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise ValueError(label + " must be JSON serializable") from exc


def _constraints_dict(value):
    return {field.name: getattr(value, field.name) for field in fields(Constraints)}


def _constraints_from_dict(value):
    value = _json_object(value, "hard_constraints")
    expected = {field.name for field in fields(Constraints)}
    if set(value) != expected:
        raise ValueError("hard_constraints must contain the exact Constraints fields")
    for name in ("include_genres", "exclude_genres", "excluded_item_ids", "required_fields"):
        if type(value[name]) is not list:
            raise ValueError(name + " must be a list")
        value[name] = tuple(value[name])
    constraints = Constraints(**value)
    _validate_constraints(constraints)
    return constraints


def _validate_constraints(value):
    if not isinstance(value, Constraints):
        raise ValueError("hard_constraints must be Constraints")
    _string_tuple(value.include_genres, "include_genres")
    _string_tuple(value.exclude_genres, "exclude_genres")
    _positive_ids(value.excluded_item_ids, "excluded_item_ids")
    _string_tuple(value.required_fields, "required_fields")
    if type(value.exclude_seen) is not bool:
        raise ValueError("exclude_seen must be boolean")
    _positive_int(value.k, "k")
    for name in ("year_min", "year_max"):
        year = getattr(value, name)
        if year is not None:
            _positive_int(year, name)


@dataclass(frozen=True)
class PublicEpisode:
    episode_id: str
    split: str
    layer: str
    group_id: str
    template_group_id: str
    scenario_family: str
    multi_turn: bool
    initial_input: dict[str, Any]
    max_turns: int
    schema_version: int = 1

    def __post_init__(self):
        if self.schema_version != 1 or type(self.schema_version) is not int:
            raise ValueError("schema_version must be integer 1")
        _text(self.episode_id, "episode_id")
        if self.split not in SPLITS or not self.episode_id.startswith(self.split + "-"):
            raise ValueError("episode_id must be namespaced by a supported split")
        if self.layer not in LAYERS:
            raise ValueError("unsupported evaluation layer")
        _text(self.group_id, "group_id")
        _text(self.template_group_id, "template_group_id")
        if self.scenario_family not in SCENARIO_FAMILIES:
            raise ValueError("unsupported scenario family")
        if type(self.multi_turn) is not bool:
            raise ValueError("multi_turn must be boolean")
        object.__setattr__(self, "initial_input",
                           _json_object(self.initial_input, "initial_input"))
        _positive_int(self.max_turns, "max_turns")
        if self.multi_turn and self.max_turns < 2:
            raise ValueError("multi-turn episodes need at least two turns")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if type(value) is not dict or set(value) != {
            "schema_version", "episode_id", "split", "layer", "group_id",
            "template_group_id", "scenario_family", "multi_turn", "initial_input",
            "max_turns",
        }:
            raise ValueError("invalid public episode schema")
        return cls(**value)


@dataclass(frozen=True)
class PrivateEpisode:
    episode_id: str
    feasible: bool
    acceptable_item_ids: tuple[int, ...]
    hard_constraints: Constraints
    requested_k: int
    required_count: int
    expected_statuses: tuple[str, ...]
    expected_patch: dict[str, Any] | None = None
    evaluator_events: tuple[dict[str, Any], ...] = ()
    fault: str | None = None
    expected_fallback_kind: str | None = None
    schema_version: int = 1

    def __post_init__(self):
        if self.schema_version != 1 or type(self.schema_version) is not int:
            raise ValueError("schema_version must be integer 1")
        _text(self.episode_id, "episode_id")
        if type(self.feasible) is not bool:
            raise ValueError("feasible must be boolean")
        _positive_ids(self.acceptable_item_ids, "acceptable_item_ids")
        _validate_constraints(self.hard_constraints)
        _positive_int(self.requested_k, "requested_k")
        if self.hard_constraints.k != self.requested_k:
            raise ValueError("requested_k must match hard_constraints.k")
        if type(self.required_count) is not int or not 0 <= self.required_count <= self.requested_k:
            raise ValueError("required_count must be in [0, requested_k]")
        _string_tuple(self.expected_statuses, "expected_statuses")
        if self.feasible and (not self.acceptable_item_ids or self.required_count < 1):
            raise ValueError("feasible episodes require targets and a positive count")
        if not self.feasible and (self.acceptable_item_ids or self.required_count):
            raise ValueError("infeasible episodes cannot require recommendations")
        if self.expected_patch is not None:
            object.__setattr__(self, "expected_patch",
                               _json_object(self.expected_patch, "expected_patch"))
        if (type(self.evaluator_events) is not tuple
                or any(type(event) is not dict for event in self.evaluator_events)):
            raise ValueError("evaluator_events must be a tuple of objects")
        object.__setattr__(self, "evaluator_events", tuple(
            _json_object(event, "evaluator event") for event in self.evaluator_events
        ))
        if self.fault is not None:
            _text(self.fault, "fault")
        if self.expected_fallback_kind is not None:
            _text(self.expected_fallback_kind, "expected_fallback_kind")

    def to_dict(self):
        value = asdict(self)
        value["hard_constraints"] = _constraints_dict(self.hard_constraints)
        return value

    @classmethod
    def from_dict(cls, value):
        if type(value) is not dict or set(value) != {
            "schema_version", "episode_id", "feasible", "acceptable_item_ids",
            "hard_constraints", "requested_k", "required_count", "expected_statuses",
            "expected_patch", "evaluator_events", "fault", "expected_fallback_kind",
        }:
            raise ValueError("invalid private episode schema")
        value = dict(value)
        value["acceptable_item_ids"] = tuple(value["acceptable_item_ids"])
        value["hard_constraints"] = _constraints_from_dict(value["hard_constraints"])
        value["expected_statuses"] = tuple(value["expected_statuses"])
        value["evaluator_events"] = tuple(value["evaluator_events"])
        return cls(**value)


@dataclass(frozen=True)
class EpisodeAttempt:
    episode_id: str
    status: str
    item_ids: tuple[int, ...]
    turns: int
    tool_calls: int
    latency_ms: float
    input_tokens: int | None
    output_tokens: int | None
    request_count: int
    preference_patch: dict[str, Any] | None = None
    fallback_kind: str | None = None

    def __post_init__(self):
        _text(self.episode_id, "episode_id")
        _text(self.status, "status")
        if type(self.item_ids) is not tuple or any(type(value) is not int for value in self.item_ids):
            raise ValueError("item_ids must be a tuple of integers")
        for name in ("turns", "tool_calls", "request_count"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(name + " must be a nonnegative integer")
        if (type(self.latency_ms) not in (int, float)
                or not math.isfinite(self.latency_ms) or self.latency_ms < 0):
            raise ValueError("latency_ms must be finite and nonnegative")
        object.__setattr__(self, "latency_ms", float(self.latency_ms))
        for name in ("input_tokens", "output_tokens"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(name + " must be a nonnegative integer or null")
        if self.preference_patch is not None:
            object.__setattr__(self, "preference_patch",
                               _json_object(self.preference_patch, "preference_patch"))
        if self.fallback_kind is not None:
            _text(self.fallback_kind, "fallback_kind")


@dataclass(frozen=True)
class EpisodeScore:
    episode_id: str
    item_validity: float
    constraint_precision: float
    fill_at_k: float
    strict_success: bool
    preference_update_accuracy: int | None
    invalid_item_count: int
    constraint_violation_count: int
    output_count: int
    legal_output_count: int
    status: str
    turns: int
    tool_calls: int
    latency_ms: float
    input_tokens: int | None
    output_tokens: int | None
    request_count: int
    fallback_kind: str | None


def score_episode(public, private, attempt, catalog):
    if not isinstance(public, PublicEpisode) or not isinstance(private, PrivateEpisode):
        raise TypeError("public/private episodes must use the frozen schemas")
    if not isinstance(attempt, EpisodeAttempt) or not isinstance(catalog, Catalog):
        raise TypeError("attempt and catalog must use the frozen schemas")
    if not public.episode_id == private.episode_id == attempt.episode_id:
        raise ValueError("episode IDs do not align")

    gate = ConstraintGate(catalog)
    seen = set()
    legal = []
    invalid = 0
    violations = 0
    for item_id in attempt.item_ids:
        if item_id <= 0 or item_id not in catalog.items or item_id in seen:
            invalid += 1
            continue
        seen.add(item_id)
        result = gate.pre_filter([item_id], private.hard_constraints)
        if result.status != "OK":
            violations += 1
            continue
        legal.append(item_id)

    output_count = len(attempt.item_ids)
    correct_empty = (not private.feasible and not output_count
                     and attempt.status in private.expected_statuses)
    if output_count:
        item_validity = len(seen) / output_count
        constraint_precision = len(legal) / output_count
    else:
        item_validity = 1.0 if correct_empty else 0.0
        constraint_precision = 1.0 if correct_empty else 0.0
    fill = min(len(legal) / private.requested_k, 1.0)

    status_ok = attempt.status in private.expected_statuses
    fallback_ok = (private.expected_fallback_kind is None
                   or attempt.fallback_kind == private.expected_fallback_kind)
    if private.feasible:
        accepted = bool(set(legal) & set(private.acceptable_item_ids))
        strict = (status_ok and fallback_ok and invalid == 0 and violations == 0
                  and private.required_count <= len(legal) <= private.requested_k
                  and output_count <= private.requested_k and accepted)
    else:
        strict = status_ok and fallback_ok and output_count == 0
    preference = None
    if private.expected_patch is not None:
        preference = int(attempt.preference_patch == private.expected_patch)

    return EpisodeScore(
        episode_id=public.episode_id,
        item_validity=item_validity,
        constraint_precision=constraint_precision,
        fill_at_k=fill,
        strict_success=bool(strict),
        preference_update_accuracy=preference,
        invalid_item_count=invalid,
        constraint_violation_count=violations,
        output_count=output_count,
        legal_output_count=len(legal),
        status=attempt.status,
        turns=attempt.turns,
        tool_calls=attempt.tool_calls,
        latency_ms=attempt.latency_ms,
        input_tokens=attempt.input_tokens,
        output_tokens=attempt.output_tokens,
        request_count=attempt.request_count,
        fallback_kind=attempt.fallback_kind,
    )


def _nearest_rank(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def evaluate_episodes(public_episodes, private_episodes, attempts, catalog):
    public_episodes = list(public_episodes)
    private_episodes = list(private_episodes)
    public = {episode.episode_id: episode for episode in public_episodes}
    private = {episode.episode_id: episode for episode in private_episodes}
    if len(public) != len(public_episodes) or len(private) != len(private_episodes):
        raise ValueError("duplicate episode definition")
    if set(public) != set(private):
        raise ValueError("public/private episode sets differ")
    attempt_map = {}
    for attempt in attempts:
        if attempt.episode_id not in public:
            raise ValueError("attempt references unknown episode")
        if attempt.episode_id in attempt_map:
            raise ValueError("duplicate attempt for episode")
        attempt_map[attempt.episode_id] = attempt

    missing = 0
    scores = []
    for episode_id in public:
        attempt = attempt_map.get(episode_id)
        if attempt is None:
            missing += 1
            attempt = EpisodeAttempt(
                episode_id, "MISSING_ATTEMPT", (), 0, 0, 0.0, 0, 0, 0
            )
        scores.append(score_episode(public[episode_id], private[episode_id],
                                    attempt, catalog))
    total = len(scores)
    annotated = [score.preference_update_accuracy for score in scores
                 if score.preference_update_accuracy is not None]
    unknown_usage = sum(score.input_tokens is None or score.output_tokens is None
                        for score in scores)
    input_known = sum(score.input_tokens or 0 for score in scores)
    output_known = sum(score.output_tokens or 0 for score in scores)
    status_counts = {}
    for score in scores:
        status_counts[score.status] = status_counts.get(score.status, 0) + 1
    denominator = total or 1
    latencies = [score.latency_ms for score in scores]
    return {
        "episodes_total": total,
        "attempts_missing": missing,
        "item_validity": sum(score.item_validity for score in scores) / denominator,
        "constraint_precision": sum(score.constraint_precision for score in scores) / denominator,
        "fill_at_k": sum(score.fill_at_k for score in scores) / denominator,
        "strict_success_rate": sum(score.strict_success for score in scores) / denominator,
        "preference_update_accuracy": (
            sum(annotated) / len(annotated) if annotated else None
        ),
        "preference_update_denominator": len(annotated),
        "invalid_items": sum(score.invalid_item_count for score in scores),
        "constraint_violations": sum(score.constraint_violation_count for score in scores),
        "turns": {
            "total": sum(score.turns for score in scores),
            "mean": sum(score.turns for score in scores) / denominator,
        },
        "tool_calls": {
            "total": sum(score.tool_calls for score in scores),
            "mean": sum(score.tool_calls for score in scores) / denominator,
        },
        "latency_ms": {
            "p50": statistics.median(latencies) if latencies else None,
            "p95": _nearest_rank(latencies, 0.95),
        },
        "usage": {
            "input_tokens_known_sum": input_known,
            "output_tokens_known_sum": output_known,
            "total_tokens": None if unknown_usage else input_known + output_known,
            "episodes_unknown_usage": unknown_usage,
            "requests": sum(score.request_count for score in scores),
        },
        "rates": {
            "fallback": sum(score.fallback_kind is not None for score in scores) / denominator,
            "abstain": sum(score.status == "ABSTAIN" for score in scores) / denominator,
            "clarify": sum(score.status.startswith("CLARIFY") for score in scores) / denominator,
            "no_solution": sum(score.status == "NO_FEASIBLE_ITEMS" for score in scores) / denominator,
        },
        "status_counts": status_counts,
    }


def _walk_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def _public_item_ids(value):
    found = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key in PUBLIC_ITEM_ID_KEYS and isinstance(child, list):
                found.update(item for item in child if type(item) is int)
            found.update(_public_item_ids(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_public_item_ids(child))
    return found


def _validate_split(public_episodes, private_episodes, split, catalog):
    public = list(public_episodes)
    private = list(private_episodes)
    if len({episode.episode_id for episode in public}) != len(public):
        raise ValueError("duplicate public episode ID")
    if len({episode.episode_id for episode in private}) != len(private):
        raise ValueError("duplicate private episode ID")
    public_map = {episode.episode_id: episode for episode in public}
    private_map = {episode.episode_id: episode for episode in private}
    if set(public_map) != set(private_map):
        raise ValueError("public/private episode sets differ")
    for episode_id, visible in public_map.items():
        hidden = private_map[episode_id]
        if visible.split != split:
            raise ValueError("episode stored in wrong split")
        forbidden = set(_walk_keys(visible.initial_input)) & PUBLIC_FORBIDDEN_KEYS
        if forbidden:
            raise ValueError("hidden evaluator field leaked into public input")
        visible_ids = _public_item_ids(visible.initial_input)
        if visible_ids & set(hidden.acceptable_item_ids):
            raise ValueError("target item ID leaked into visible input")
        serialized = json.dumps(visible.to_dict(), ensure_ascii=False).casefold()
        for item_id in hidden.acceptable_item_ids:
            title = catalog.items[item_id].title.casefold()
            if title and title in serialized:
                raise ValueError("target title leaked into visible input")
        if any(item_id not in catalog.items for item_id in hidden.acceptable_item_ids):
            raise ValueError("hidden target outside frozen catalog")
    return public_map, private_map


def validate_benchmark(dev_public, dev_private, test_public, test_private, catalog):
    dev_map, _ = _validate_split(dev_public, dev_private, "development", catalog)
    test_map, _ = _validate_split(test_public, test_private, "test", catalog)
    if set(dev_map) & set(test_map):
        raise ValueError("episode leakage across splits")
    dev_groups = {episode.group_id for episode in dev_map.values()}
    test_groups = {episode.group_id for episode in test_map.values()}
    if dev_groups & test_groups:
        raise ValueError("group leakage across development and test")
    dev_templates = {episode.template_group_id for episode in dev_map.values()}
    test_templates = {episode.template_group_id for episode in test_map.values()}
    if dev_templates & test_templates:
        raise ValueError("template group leakage across development and test")
    return True


def _write_jsonl(path, rows):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")) + "\n")


def _read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"blank JSONL row at line {line_number}")
            rows.append(json.loads(line))
    return rows


def load_public_episodes(path):
    return [PublicEpisode.from_dict(row) for row in _read_jsonl(path)]


def load_private_episodes(path):
    return [PrivateEpisode.from_dict(row) for row in _read_jsonl(path)]


def write_frozen_benchmark(data_dir, manifest_path, dev_public, dev_private,
                           test_public, test_private, *, catalog,
                           human_review_ids, source_bindings=None):
    data_dir = Path(data_dir)
    manifest_path = Path(manifest_path)
    if data_dir.exists() or manifest_path.exists():
        raise FileExistsError("frozen interactive evaluation output already exists")
    validate_benchmark(dev_public, dev_private, test_public, test_private, catalog)
    all_ids = {episode.episode_id for episode in list(dev_public) + list(test_public)}
    review_ids = tuple(human_review_ids)
    if not review_ids or len(set(review_ids)) != len(review_ids) or not set(review_ids) <= all_ids:
        raise ValueError("human_review_ids must be a nonempty subset of episode IDs")
    data_dir.mkdir(parents=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "development_public": data_dir / "development.public.jsonl",
        "development_private": data_dir / "development.private.jsonl",
        "test_public": data_dir / "test.public.jsonl",
        "test_private": data_dir / "test.private.jsonl",
    }
    _write_jsonl(paths["development_public"], [item.to_dict() for item in dev_public])
    _write_jsonl(paths["development_private"], [item.to_dict() for item in dev_private])
    _write_jsonl(paths["test_public"], [item.to_dict() for item in test_public])
    _write_jsonl(paths["test_private"], [item.to_dict() for item in test_private])

    common_root = Path(os.path.commonpath((data_dir.resolve(), manifest_path.parent.resolve())))
    root_relative = Path(os.path.relpath(common_root, manifest_path.parent.resolve())).as_posix()
    split_values = {
        "development": list(dev_public),
        "test": list(test_public),
    }
    manifest = {
        "schema_version": 1,
        "protocol": "T18 interactive-v1",
        "frozen": True,
        "generation_seed": 42,
        "llm_simulator_enabled": False,
        "root_relative_to_manifest": root_relative,
        "dataset_dir": data_dir.resolve().relative_to(common_root).as_posix(),
        "human_review_ids": list(review_ids),
        "human_review_status": "PENDING",
        "human_review_checks": None,
        "source_bindings": {} if source_bindings is None else source_bindings,
        "splits": {},
        "files": {},
    }
    for split, episodes in split_values.items():
        families = {}
        layers = {}
        for episode in episodes:
            families[episode.scenario_family] = families.get(episode.scenario_family, 0) + 1
            layers[episode.layer] = layers.get(episode.layer, 0) + 1
        manifest["splits"][split] = {
            "episodes": len(episodes),
            "multi_turn": sum(episode.multi_turn for episode in episodes),
            "unique_groups": len({episode.group_id for episode in episodes}),
            "unique_template_groups": len({episode.template_group_id for episode in episodes}),
            "scenario_families": dict(sorted(families.items())),
            "layers": dict(sorted(layers.items())),
        }
    for name, path in paths.items():
        manifest["files"][name] = {
            "name": path.name,
            "path": path.resolve().relative_to(common_root).as_posix(),
            "sha256": _sha256(path),
            "rows": len(_read_jsonl(path)),
            "visibility": "public" if name.endswith("_public") else "evaluator_only",
        }
    with manifest_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return manifest


def verify_eval_manifest(path):
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != 1 or not manifest.get("frozen")
            or manifest.get("llm_simulator_enabled") is not False):
        raise ValueError("invalid interactive evaluation manifest")
    review_status = manifest.get("human_review_status")
    review_checks = manifest.get("human_review_checks")
    if review_status not in {"PENDING", "VERIFIED"}:
        raise ValueError("invalid human review status")
    if (review_status == "PENDING" and review_checks is not None) or (
            review_status == "VERIFIED"
            and (type(review_checks) is not dict or not review_checks
                 or any(value is not True for value in review_checks.values()))):
        raise ValueError("human review status and checks disagree")
    root = (path.parent / manifest["root_relative_to_manifest"]).resolve()
    for entry in manifest["files"].values():
        file_path = root / entry["path"]
        if not file_path.is_file() or _sha256(file_path) != entry["sha256"]:
            raise ValueError("frozen evaluation file hash mismatch")
        if len(_read_jsonl(file_path)) != entry["rows"]:
            raise ValueError("frozen evaluation row count mismatch")
    return manifest


def record_human_review(path, reviewed_ids, checks):
    """Finalize audit metadata after a person inspects the preselected subset."""
    path = Path(path)
    manifest = verify_eval_manifest(path)
    reviewed_ids = tuple(reviewed_ids)
    if (not reviewed_ids or reviewed_ids != tuple(manifest["human_review_ids"])
            or len(set(reviewed_ids)) != len(reviewed_ids)):
        raise ValueError("reviewed IDs must exactly match the preselected audit subset")
    checks = _json_object(checks, "human review checks")
    if not checks or any(value is not True for value in checks.values()):
        raise ValueError("every human review check must be explicitly true")
    manifest["human_review_status"] = "VERIFIED"
    manifest["human_review_checks"] = checks
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError("human review temporary file already exists")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)
    return manifest


def _feasible_ids(catalog, constraints, excluded=()):
    gate = ConstraintGate(catalog)
    result = gate.pre_filter(list(catalog.items), constraints, seen_ids=excluded)
    return tuple(result.item_ids) if result.status == "OK" else ()


def _similar_ids(catalog, seed_id, excluded=()):
    seed = catalog.items[seed_id]
    excluded = set(excluded) | {seed_id}
    scored = []
    for item_id, item in catalog.items.items():
        if item_id in excluded:
            continue
        union = seed.genres | item.genres
        score = len(seed.genres & item.genres) / len(union) if union else 0
        if score > 0:
            scored.append((score, item_id))
    return tuple(item_id for _, item_id in sorted(scored, key=lambda pair: (-pair[0], pair[1])))


def _visible_input(split, layer, family, index, user_id, genre, seed_id, k):
    if layer == "structured":
        payload = {
            "schema_version": 1,
            "user_id": user_id if family in {"personalized", "tool_error"} else None,
            "history_authorized": family in {"personalized", "tool_error"},
            "liked_item_ids": [seed_id] if family in {"seed_similar", "exclusion_feedback"} else [],
            "disliked_item_ids": [],
            "seen_item_ids": [],
            "constraints": {"k": k},
        }
        if family in {"explicit_filter", "cold_start"}:
            payload["constraints"]["include_genres"] = [genre]
        elif family == "constraint_conflict":
            payload["constraints"].update({"include_genres": [genre],
                                            "exclude_genres": [genre.casefold()]})
        elif family == "unknown_attribute":
            payload["constraints"]["required_fields"] = ["duration"]
        return payload

    development = {
        "explicit_filter": "Please return {k} {genre} movies from the verified catalog.",
        "personalized": "Using my authorized profile, suggest {k} movies I may enjoy.",
        "seed_similar": "Find {k} movies similar to catalog item {seed}.",
        "exclusion_feedback": "Start with {k} suggestions related to catalog item {seed}; I may correct you next.",
        "constraint_conflict": "Recommend {genre} movies, but exclude every {genre} movie.",
        "unknown_attribute": "Recommend {k} movies with a verified duration field.",
        "cold_start": "I have no saved history; suggest {k} {genre} movies.",
        "tool_error": "Use my authorized profile to suggest {k} movies and degrade safely if a tool fails.",
    }
    test = {
        "explicit_filter": "From the catalog, choose {k} titles whose genre includes {genre}.",
        "personalized": "Select {k} recommendations from my permitted historical profile.",
        "seed_similar": "Give me {k} alternatives related to item ID {seed}.",
        "exclusion_feedback": "Propose {k} options around item ID {seed}; expect a follow-up preference.",
        "constraint_conflict": "I require the {genre} genre and also forbid the {genre} genre.",
        "unknown_attribute": "Only suggest {k} catalog movies when their duration is known.",
        "cold_start": "Without a user profile, find {k} choices in the {genre} genre.",
        "tool_error": "Recommend {k} items from my approved profile with a safe fallback on tool failure.",
    }
    message = (development if split == "development" else test)[family].format(
        k=k, genre=genre, seed=seed_id
    )
    return {"mode": "text", "template_id": f"{split}-{family}-{index % 5}",
            "message": message, "user_id": user_id if family in {"personalized", "tool_error"} else None,
            "history_authorized": family in {"personalized", "tool_error"}}


def _build_split(split, users, cohort, catalog, seed):
    counts = {
        "explicit_filter": 20, "personalized": 20, "seed_similar": 20,
        "exclusion_feedback": 40, "constraint_conflict": 15,
        "unknown_attribute": 15, "cold_start": 10, "tool_error": 10,
    }
    families = [family for family, count in counts.items() for _ in range(count)]
    rng = random.Random(seed + (0 if split == "development" else 1000))
    rng.shuffle(families)
    public, private = [], []
    for index, (user_id, family) in enumerate(zip(users, families), 1):
        episode_id = f"{split}-{index:03d}"
        relevant = tuple(cohort["relevant_by_user"][str(user_id)])
        seen = tuple(cohort["seen_by_user"][str(user_id)])
        target_id = relevant[(index - 1) % len(relevant)]
        target = catalog.items[target_id]
        genre = sorted(target.genres)[0]
        seed_id = seen[(index * 17) % len(seen)]
        k = 5
        constraints = Constraints(k=k)
        accepted = relevant
        feasible = True
        required_count = 1
        statuses = ("OK",)
        patch = None
        events = ()
        fault = None
        expected_fallback = None

        if family in {"explicit_filter", "cold_start"}:
            constraints = Constraints(include_genres=(genre,), k=k)
            accepted = _feasible_ids(catalog, constraints)
            required_count = min(k, len(accepted))
        elif family == "seed_similar":
            accepted = _similar_ids(catalog, seed_id)
            required_count = min(k, len(accepted))
        elif family == "exclusion_feedback":
            similar = _similar_ids(catalog, seed_id)
            disliked_id = similar[0]
            accepted = tuple(item for item in similar[1:] if item != disliked_id)
            constraints = Constraints(excluded_item_ids=(disliked_id,), k=k)
            patch = {"disliked_item_ids": [disliked_id]}
            events = ({"turn": 2, "kind": "explicit_feedback", "patch": patch},)
            required_count = min(k, len(accepted))
        elif family == "constraint_conflict":
            constraints = Constraints(include_genres=(genre,), exclude_genres=(genre.casefold(),), k=k)
            accepted, feasible, required_count = (), False, 0
            statuses = ("CLARIFY_CONFLICT",)
        elif family == "unknown_attribute":
            constraints = Constraints(k=k, required_fields=("duration",))
            accepted, feasible, required_count = (), False, 0
            statuses = ("MISSING_FIELD",)
        elif family == "tool_error":
            fault = "ranking_timeout"
            statuses = ("OK", "FALLBACK")
            expected_fallback = "fixed_pipeline"

        if feasible and (not accepted or required_count < 1):
            raise ValueError("generated feasible episode has no acceptance set")
        layer = "structured" if index % 2 else "text"
        visible = _visible_input(split, layer, family, index, user_id, genre,
                                 seed_id, k)
        public.append(PublicEpisode(
            episode_id=episode_id,
            split=split,
            layer=layer,
            group_id=f"user:{user_id}",
            template_group_id=f"{split}:{family}:{index % 5}:{layer}",
            scenario_family=family,
            multi_turn=family == "exclusion_feedback",
            initial_input=visible,
            max_turns=3 if family == "exclusion_feedback" else 1,
        ))
        private.append(PrivateEpisode(
            episode_id=episode_id,
            feasible=feasible,
            acceptable_item_ids=tuple(accepted),
            hard_constraints=constraints,
            requested_k=k,
            required_count=required_count,
            expected_statuses=statuses,
            expected_patch=patch,
            evaluator_events=events,
            fault=fault,
            expected_fallback_kind=expected_fallback,
        ))
    return public, private


def build_interactive_benchmark(catalog, development_cohort, test_cohort, seed=42):
    dev_users = [int(user) for user, relevant in development_cohort["relevant_by_user"].items()
                 if relevant]
    test_users = [int(user) for user, relevant in test_cohort["relevant_by_user"].items()
                  if relevant]
    rng = random.Random(seed)
    rng.shuffle(dev_users)
    dev_users = dev_users[:150]
    test_users = [user for user in test_users if user not in set(dev_users)]
    rng.shuffle(test_users)
    test_users = test_users[:150]
    if len(dev_users) != 150 or len(test_users) != 150:
        raise ValueError("insufficient disjoint warm users for 150/150 episodes")
    dev_public, dev_private = _build_split(
        "development", dev_users, development_cohort, catalog, seed
    )
    test_public, test_private = _build_split(
        "test", test_users, test_cohort, catalog, seed
    )
    validate_benchmark(dev_public, dev_private, test_public, test_private, catalog)
    review_ids = []
    for split_public in (dev_public, test_public):
        for family in sorted(SCENARIO_FAMILIES):
            review_ids.append(next(item.episode_id for item in split_public
                                   if item.scenario_family == family))
    return dev_public, dev_private, test_public, test_private, tuple(review_ids)


def freeze_interactive_benchmark(root=ROOT):
    root = Path(root)
    catalog = Catalog.from_frozen(root)
    cohort_dir = root / "data/eval_private/ml-1m-v1"
    dev_cohort = json.loads((cohort_dir / "valid.json").read_text(encoding="utf-8"))
    test_cohort = json.loads((cohort_dir / "test.json").read_text(encoding="utf-8"))
    built = build_interactive_benchmark(catalog, dev_cohort, test_cohort)
    dev_public, dev_private, test_public, test_private, review_ids = built
    data_manifest = root / "artifacts/data_manifest.json"
    bindings = {
        "data_manifest_sha256": _sha256(data_manifest),
        "id_map_sha256": json.loads(data_manifest.read_text(encoding="utf-8"))["id_map_sha256"],
        "development_cohort_sha256": _sha256(cohort_dir / "valid.json"),
        "test_cohort_sha256": _sha256(cohort_dir / "test.json"),
        "source_split_mapping": {"development": "valid", "test": "test"},
        "test_used_for_generation_only_by_evaluator": True,
    }
    return write_frozen_benchmark(
        root / "data/eval_private/interactive-v1",
        root / "artifacts/eval_manifest.json",
        dev_public, dev_private, test_public, test_private,
        catalog=catalog, human_review_ids=review_ids, source_bindings=bindings,
    )


if __name__ == "__main__":
    print(json.dumps(freeze_interactive_benchmark(), ensure_ascii=False,
                     indent=2, sort_keys=True))
