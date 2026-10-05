"""Freeze source-backed metadata crosswalk and aggregate coverage diagnostics.

All aliases are built from the complete source/crosswalk before reading saved
evaluation inputs. These inputs remain post-hoc diagnostics, not a new test.
Metadata/mappings/targets remain local under MovieLens research terms.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path

import pandas as pd
from catalog_identity import (MetadataIndex, catalog_crosswalk, parse_movies_dat,
    resolve_catalog_target, source_title_forms, split_title_year)
from reproduction_session_http import DATA_SHA256

ROOT = Path(__file__).resolve().parents[2]
PRIVATE = ROOT / "data/metadata_private/catalog_identity_20261005"
PUBLIC = ROOT / "reproduction/native_runs/20261005_identity/catalog_identity_summary.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value, lines=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as f:
        f.write(("\n".join(json.dumps(r, ensure_ascii=True, sort_keys=True) for r in value) + "\n")
                if lines else json.dumps(value, ensure_ascii=True, indent=2) + "\n")
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": digest(path)}


def main():
    if PUBLIC.exists() or PRIVATE.exists():
        print(json.dumps({"status": "BLOCKED", "reason": "frozen_identity_artifacts_exist", "api_requests": 0}))
        return 2
    acquired = json.loads((ROOT / "reproduction/logs/catalog_identity_20261005/metadata_acquisition.json").read_text())
    for member in acquired["members"]:
        assert digest(ROOT / member["path"]) == member["sha256"]
    source = ROOT / "data/raw/metadata_audit/ml-10m/movies.dat"
    source_digest = digest(source)
    catalog = ROOT / "data/raw/upstream_movie/movies.ftr"
    assert digest(catalog) == "1a0e8687ba0c366cf4ab7837c591542f9bbb3b445ec4f4155fe1e932b58bb8c9"
    frame = pd.read_feather(catalog)
    rows = parse_movies_dat(source.read_bytes())
    index = MetadataIndex(rows)
    walk = catalog_crosswalk(frame.to_dict("records"), index)
    assert len(walk) == 9888 and len({r["catalog_id"] for r in walk}) == len(walk)
    crosswalk_record = save(PRIVATE / "catalog_to_ml10m.jsonl", walk, lines=True)
    aliases = []
    for row in walk:
        if row["status"] == "PASS":
            official = index.rows[row["raw_movie_id"]]
            for title in sorted(source_title_forms(official["title"])):
                # Do not approve an alias that is ambiguous in the source.
                if index.resolve(title, official["year"])["raw_ids"] == [row["raw_movie_id"]]:
                    aliases.append({"catalog_id": row["catalog_id"], "raw_movie_id": row["raw_movie_id"],
                        "title_form": title, "year": official["year"], "source_sha256": source_digest})
    alias_record = save(PRIVATE / "source_aliases.jsonl", aliases, lines=True)
    alias_hash_before_targets = alias_record["sha256"]
    data = ROOT / "data/eval_private/upstream_redial_a1_hashseed42/test_data_50.jsonl"
    assert digest(data) == DATA_SHA256
    targets = [json.loads(line) for line in data.read_text().splitlines()]
    previous = json.loads((ROOT / "reproduction/logs/repair_check_20261005/catalog_contract_cases.json").read_text())
    cases = []
    for i, row in enumerate(targets):
        title, year = split_title_year(row["target"])
        result = resolve_catalog_target(title, year, index, walk)
        direct = previous[i]["exact_catalog_contract"]
        if direct["status"] == "PASS":
            assert result["status"] == "PASS" and result["catalog_ids"] == direct["ids"]
        cases.append({"index": i, "target": row["target"], "resolution": result,
                      "previous_direct_contract": direct})
    assert digest(PRIVATE / "source_aliases.jsonl") == alias_hash_before_targets
    case_record = save(PRIVATE / "saved_target_diagnostics.json", cases)
    manifest = {"schema_version": 1, "purpose": "metadata_identity_audit_only", "catalog_sha256": digest(catalog),
        "source_url": acquired["source"], "source_sha256": source_digest,
        "terms_sha256": acquired["members"][1]["sha256"], "license": "MovieLens research terms, local only",
        "id_spaces": {"catalog_id": "unchanged upstream 1..9888", "raw_movie_id": "official MovieLens 10M ID",
                      "checkpoint_row": "NOT VERIFIED; no mapping assigned", "matrix_row": "NOT VERIFIED"},
        "alias_policy": "NFKC/case/space normalization; source English article placement and explicit a.k.a.; exact year; unique source row and genre; no fuzzy matching",
        "test_used_for_alias_creation": False, "artifacts": [crosswalk_record, alias_record, case_record],
        "training_or_model_id_map": False, "new_benchmark": False}
    manifest_record = save(PRIVATE / "manifest.json", manifest)
    summary = {"status": "PASS", "scope": "catalog provenance and post-hoc target diagnostics, not model evaluation",
        "catalog_rows": len(walk), "official_metadata_rows": len(rows),
        "catalog_crosswalk_status_counts": dict(Counter(r["status"] for r in walk)),
        "approved_source_alias_records": len(aliases),
        "saved_diagnostic_samples": len(cases),
        "saved_target_resolution_counts": dict(Counter(r["resolution"]["status"] for r in cases)),
        "direct_matches_preserved": sum(r["previous_direct_contract"]["status"] == "PASS" for r in cases),
        "additional_source_backed_alias_matches": sum(r["previous_direct_contract"]["status"] != "PASS" and r["resolution"]["status"] == "PASS" for r in cases),
        "private_manifest": manifest_record, "source_url": acquired["source"], "source_sha256": source_digest,
        "catalog_sha256": digest(catalog), "input_sha256": DATA_SHA256,
        "downloaded_bytes_including_initial_probe": acquired["downloaded_bytes_including_initial_probe"],
        "ratings_downloaded": False, "new_catalog_rows": 0, "rewritten_years": 0, "changed_upstream_ids": 0,
        "changed_evaluation_samples": 0, "api_requests": 0,
        "pretrained_model_id_semantics": "NOT VERIFIED", "similarity_row_semantics": "NOT VERIFIED",
        "formal_original_metrics": "UNCHANGED: three branches 0/45", "new_quality_metrics": None,
        "metadata_terms": "https://files.grouplens.org/datasets/movielens/ml-10m-README.html"}
    save(PUBLIC, summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
