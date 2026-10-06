"""Frozen-catalog constraints; excluded items are removed, never score-masked."""
from dataclasses import dataclass
import json
from pathlib import Path

import pyarrow.parquet as pq

from ..data import digest
from ..training import ROOT


@dataclass(frozen=True)
class Item:
    item_id: int
    title: str
    genres: frozenset[str]
    year: int | None

    def __post_init__(self):
        if type(self.item_id) is not int or self.item_id <= 0 or not isinstance(self.title, str) or not self.title:
            raise ValueError("Item needs a positive frozen ID and title")
        if type(self.genres) is not frozenset or any(type(g) is not str or not g for g in self.genres):
            raise ValueError("Item genres must be a frozen set of names")
        if self.year is not None and (type(self.year) is not int or self.year <= 0):
            raise ValueError("Item year must be positive or missing")


class Catalog:
    """Only T06 title/genres/year metadata in the frozen model item ID space."""

    def __init__(self, items):
        items = list(items)
        self.items = {item.item_id: item for item in items}
        if not items or len(self.items) != len(items) or sorted(self.items) != list(range(1, len(items) + 1)):
            raise ValueError("Catalog item IDs must be unique, contiguous and start at 1")
        self.genre_names = {genre.casefold(): genre for item in items for genre in item.genres}
        self.n_items = len(items)

    @classmethod
    def from_frozen(cls, root=ROOT):
        root = Path(root)
        manifest = json.loads((root / "artifacts/data_manifest.json").read_text(encoding="utf-8"))
        data_dir = root / "data/processed/ml-1m-v1"
        for name in ("items.parquet", "id_map.json"):
            if digest(data_dir / name) != manifest["files"][name]["sha256"]:
                raise ValueError("Frozen catalog/hash mismatch: " + name)
        table = pq.read_table(data_dir / "items.parquet", columns=["item_id", "title", "genres", "year"])
        rows = table.to_pylist()
        items = [Item(int(row["item_id"]), row["title"], frozenset(row["genres"]), row["year"])
                 for row in rows if row["item_id"] is not None]
        catalog = cls(items)
        if catalog.n_items != len(json.loads((data_dir / "id_map.json").read_text(encoding="utf-8"))["item_to_model"]):
            raise ValueError("Frozen catalog size differs from item ID map")
        return catalog


@dataclass(frozen=True)
class Constraints:
    include_genres: tuple[str, ...] = ()
    exclude_genres: tuple[str, ...] = ()
    year_min: int | None = None
    year_max: int | None = None
    excluded_item_ids: tuple[int, ...] = ()
    exclude_seen: bool = True
    k: int = 5
    required_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConstraintResult:
    status: str
    item_ids: list[int]
    reason: str | None
    missing_fields: tuple[str, ...]
    excluded_counts: dict[str, int]
    shortfall: int


class ConstraintGate:
    """Pre-filter and final output gate use the same strict rules."""

    SUPPORTED_FIELDS = frozenset(("title", "genres", "year"))

    def __init__(self, catalog: Catalog):
        self.catalog = catalog

    def _issue(self, constraints):
        if not isinstance(constraints, Constraints):
            return "INVALID_CONSTRAINTS", "Expected Constraints schema", ()
        for name in ("include_genres", "exclude_genres", "required_fields"):
            value = getattr(constraints, name)
            if type(value) is not tuple or any(type(part) is not str or not part.strip() for part in value):
                return "INVALID_CONSTRAINTS", f"{name} must be a tuple of nonempty strings", ()
        if (type(constraints.excluded_item_ids) is not tuple
                or any(type(i) is not int or i <= 0 for i in constraints.excluded_item_ids)
                or len(set(constraints.excluded_item_ids)) != len(constraints.excluded_item_ids)):
            return "INVALID_CONSTRAINTS", "excluded_item_ids must be unique positive IDs", ()
        if type(constraints.exclude_seen) is not bool or type(constraints.k) is not int or constraints.k <= 0:
            return "INVALID_CONSTRAINTS", "exclude_seen must be bool and k positive integer", ()
        if any(value is not None and (type(value) is not int or value <= 0)
               for value in (constraints.year_min, constraints.year_max)):
            return "INVALID_CONSTRAINTS", "Year bounds must be positive integers or absent", ()
        required = set(constraints.required_fields)
        missing = tuple(sorted(required - self.SUPPORTED_FIELDS))
        if missing:
            return "MISSING_FIELD", "Requested field is absent from the verified catalog", missing
        included = {genre.casefold() for genre in constraints.include_genres}
        excluded = {genre.casefold() for genre in constraints.exclude_genres}
        if included & excluded or (constraints.year_min is not None and constraints.year_max is not None
                                   and constraints.year_min > constraints.year_max):
            return "CLARIFY_CONFLICT", "Contradictory hard constraints require clarification", ()
        unknown_genres = (included | excluded) - self.catalog.genre_names.keys()
        if unknown_genres:
            return "UNKNOWN_GENRE", "Genre is outside the verified catalog enum", tuple(sorted(unknown_genres))
        if any(i not in self.catalog.items for i in constraints.excluded_item_ids):
            return "UNKNOWN_ITEM_ID", "Excluded item ID is outside the frozen catalog", ()
        return None

    def pre_filter(self, candidate_ids, constraints: Constraints, seen_ids=()):
        issue = self._issue(constraints)
        if issue is not None:
            status, reason, fields = issue
            return ConstraintResult(status, [], reason, fields, {}, constraints.k if isinstance(constraints, Constraints) and type(constraints.k) is int and constraints.k > 0 else 0)
        candidates = list(candidate_ids)
        seen = list(seen_ids)
        if (any(type(i) is not int or i not in self.catalog.items for i in candidates + seen)
                or len(candidates) != len(set(candidates)) or len(seen) != len(set(seen))):
            return ConstraintResult("INVALID_ITEM_ID", [], "Candidate/seen IDs must be unique frozen model IDs", (), {}, constraints.k)
        include = {name.casefold() for name in constraints.include_genres}
        exclude = {name.casefold() for name in constraints.exclude_genres}
        unwanted = set(constraints.excluded_item_ids)
        seen_set = set(seen) if constraints.exclude_seen else set()
        accepted, missing, removed = [], set(), {}
        for item_id in candidates:
            item = self.catalog.items[item_id]
            genres = {genre.casefold() for genre in item.genres}
            reason = None
            if item_id in unwanted:
                reason = "excluded_item"
            elif item_id in seen_set:
                reason = "seen_item"
            elif "genres" in constraints.required_fields and not item.genres:
                reason = "missing_genres"
                missing.add("genres")
            elif (include or exclude) and not item.genres:
                reason = "missing_genres"
                missing.add("genres")
            elif not include <= genres:
                reason = "include_genres"
            elif exclude & genres:
                reason = "exclude_genres"
            elif "year" in constraints.required_fields and item.year is None:
                reason = "missing_year"
                missing.add("year")
            elif (constraints.year_min is not None or constraints.year_max is not None) and item.year is None:
                reason = "missing_year"
                missing.add("year")
            elif constraints.year_min is not None and item.year < constraints.year_min:
                reason = "year_min"
            elif constraints.year_max is not None and item.year > constraints.year_max:
                reason = "year_max"
            elif "title" in constraints.required_fields and not item.title:
                reason = "missing_title"
                missing.add("title")
            if reason is None:
                accepted.append(item_id)
            else:
                removed[reason] = removed.get(reason, 0) + 1
        status = "OK" if accepted else "NO_FEASIBLE_ITEMS"
        return ConstraintResult(status, accepted,
                                None if accepted else "No item satisfies every hard condition",
                                tuple(sorted(missing)), removed,
                                max(0, constraints.k - len(accepted)))

    def finalize(self, ranked_ids, constraints: Constraints, seen_ids=()):
        """Reapply hard rules after any ranking step; never pad with bad items."""
        result = self.pre_filter(ranked_ids, constraints, seen_ids)
        if result.status != "OK":
            return result
        selected = result.item_ids[:constraints.k]
        return ConstraintResult("OK", selected, None, result.missing_fields,
                                result.excluded_counts, max(0, constraints.k - len(selected)))
