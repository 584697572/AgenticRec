"""Metadata identity crosswalks; never infer pretrained embedding row labels.

Aliases come only from explicit a.k.a. annotations or article placement in a
source title. Exact year, uniqueness and genre checks fail closed. No fuzzy
title matching, year rewriting, target filtering or new movie rows occurs.
"""
from collections import Counter, defaultdict
import re
import unicodedata


def normalized_title(title):
    return " ".join(unicodedata.normalize("NFKC", title).casefold().split())


def split_title_year(title):
    match = re.fullmatch(r"(.+)\s+\((\d{4})\)", title.strip())
    return (match[1].strip(), int(match[2])) if match else (title.strip(), None)


def source_title_forms(title):
    forms = {title.strip()}
    aliases = list(re.finditer(r"\s*\(a\.k\.a\.\s+([^()]+)\)", title, flags=re.IGNORECASE))
    if aliases:
        forms.add(re.sub(r"\s*\(a\.k\.a\.\s+[^()]+\)", "", title, flags=re.IGNORECASE).strip())
        forms.update(m[1].strip() for m in aliases)
    for form in list(forms):
        article = re.fullmatch(r"(.+),\s+(The|A|An)", form, flags=re.IGNORECASE)
        if article:
            forms.add(article[2] + " " + article[1])
    return {normalized_title(form) for form in forms}


def parse_movies_dat(raw):
    rows, seen = [], set()
    for line in raw.decode("utf-8").splitlines():
        fields = line.split("::")
        if len(fields) != 3:
            raise ValueError("invalid_movies_dat_fields")
        item_id = int(fields[0])
        if item_id <= 0 or item_id in seen:
            raise ValueError("duplicate_or_invalid_raw_movie_id")
        seen.add(item_id)
        title, year = split_title_year(fields[1])
        rows.append({"raw_id": item_id, "source_title": fields[1], "title": title,
                     "year": year, "genres": sorted(fields[2].split("|"))})
    if not rows:
        raise ValueError("empty_movie_metadata")
    return rows


class MetadataIndex:
    def __init__(self, rows):
        self.rows = {r["raw_id"]: r for r in rows}
        if len(self.rows) != len(rows):
            raise ValueError("duplicate_raw_movie_id")
        self.by_title = defaultdict(set)
        for row in rows:
            for form in source_title_forms(row["title"]):
                self.by_title[form].add(row["raw_id"])

    def resolve(self, title, year):
        candidates = self.by_title.get(normalized_title(title), set())
        if not candidates:
            return {"status": "TITLE_ABSENT", "raw_ids": []}
        exact = sorted(i for i in candidates if year is None or self.rows[i]["year"] == year)
        if not exact:
            return {"status": "YEAR_CONFLICT", "raw_ids": [], "candidate_years": sorted(
                {self.rows[i]["year"] for i in candidates if self.rows[i]["year"] is not None})}
        return {"status": "PASS" if len(exact) == 1 else "AMBIGUOUS", "raw_ids": exact}


def catalog_crosswalk(catalog_rows, index):
    records = []
    for row in catalog_rows:
        result = index.resolve(row["title"], int(row["release_date"]))
        raw_id = result["raw_ids"][0] if result["status"] == "PASS" else None
        status = result["status"]
        if raw_id is not None and sorted(row["tags"]) != index.rows[raw_id]["genres"]:
            status = "GENRE_CONFLICT"
            raw_id = None
        records.append({"catalog_id": int(row["id"]), "raw_movie_id": raw_id, "status": status})
    counts = Counter(r["raw_movie_id"] for r in records if r["status"] == "PASS")
    for record in records:
        if record["status"] == "PASS" and counts[record["raw_movie_id"]] != 1:
            record.update(status="NON_BIJECTIVE", raw_movie_id=None)
    return records


def resolve_catalog_target(title, year, index, crosswalk):
    result = index.resolve(title, year)
    if result["status"] != "PASS":
        return {"status": result["status"], "catalog_ids": [], "raw_ids": result["raw_ids"]}
    raw_id = result["raw_ids"][0]
    ids = sorted(r["catalog_id"] for r in crosswalk if r["status"] == "PASS" and r["raw_movie_id"] == raw_id)
    return {"status": "PASS" if len(ids) == 1 else "CATALOG_ABSENT", "catalog_ids": ids, "raw_ids": [raw_id]}
