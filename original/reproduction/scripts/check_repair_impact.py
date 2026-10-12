"""Compare opt-in mapping correction and exact catalog guard on saved inputs.

No dataset, existing prediction, baseline metric or upstream file is rewritten.
Private row details stay ignored. Public output contains aggregate facts only.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path

import pandas as pd
from audit_redial_preprocess import CATALOG, NOTEBOOK, original_functions
from redial_compat import adapt_release_dates
from reproduction_session_http import DATA_SHA256
from upstream_bugfixes import corrected_movie_map, require_reachable_targets, target_catalog_contract

ROOT = Path(__file__).resolve().parents[2]


def main():
    private = ROOT / "reproduction/logs/repair_check_20261005/catalog_contract_cases.json"
    output = ROOT / "reproduction/native_runs/20261005_repairs/catalog_contract_summary.json"
    if private.exists() or output.exists():
        print(json.dumps({"status": "BLOCKED", "reason": "existing_impact_evidence_preserved", "remote_api_requests": 0}))
        return 2
    data = ROOT / "data/eval_private/upstream_redial_a1_hashseed42/test_data_50.jsonl"
    raw = data.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == DATA_SHA256
    rows = [json.loads(line) for line in raw.splitlines()]
    original = original_functions()
    patched = corrected_movie_map()
    frame = pd.read_feather(CATALOG)
    adapted = adapt_release_dates(frame)
    adapted["title"] = adapted["title"].str.lower()
    records = []
    for index, row in enumerate(rows):
        title, year = original["separate_movie_and_year"](row["target"])
        query = {"title": title, "date": year}
        original_id = int(original["movie_map"](query, adapted)[0][title, year])
        patched_id = int(patched(query, adapted)[0][title, year])
        def mapped_year(item_id):
            return int(frame.loc[frame["id"] == item_id, "release_date"].iloc[0]) if item_id > 0 else None
        records.append({"index": index, "target": row["target"], "parsed_year": year,
            "original_mapping_id": original_id, "patched_mapping_id": patched_id,
            "original_mapping_year": mapped_year(original_id), "patched_mapping_year": mapped_year(patched_id),
            "exact_catalog_contract": target_catalog_contract(row["target"], frame, original["separate_movie_and_year"])})
    rejected = False
    try:
        require_reachable_targets(rows, frame, original["separate_movie_and_year"])
    except ValueError:
        rejected = True
    assert rejected and len(rows) == 45
    private.parent.mkdir(parents=True, exist_ok=True)
    with private.open("x", encoding="utf-8") as f:
        f.write(json.dumps(records, ensure_ascii=True, indent=2) + "\n")
    previous = json.loads((ROOT / "reproduction/a1_session_evidence_20261005.json").read_text(encoding="utf-8"))
    for record in previous["public_files"]:
        # Status documents will be updated after this read-only impact check.
        if record["path"].startswith("reproduction/native_runs/20261005_session/"):
            assert hashlib.sha256((ROOT / record["path"]).read_bytes()).hexdigest() == record["working_bytes_sha256"]
    for record in previous["private_files"]:
        assert hashlib.sha256((ROOT / record["path"]).read_bytes()).hexdigest() == record["sha256"]
    summary = {"scope": "isolated_upstream_correctness_patch_and_catalog_contract; not a new benchmark",
        "status": "PASS", "samples": len(rows), "input_sha256": DATA_SHA256,
        "notebook_sha256": hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest(),
        "catalog_sha256": hashlib.sha256(CATALOG.read_bytes()).hexdigest(),
        "private_cases_sha256": hashlib.sha256(private.read_bytes()).hexdigest(),
        "original_mapping_unmapped": sum(r["original_mapping_id"] < 0 for r in records),
        "patched_mapping_unmapped": sum(r["patched_mapping_id"] < 0 for r in records),
        "changed_mapping_count": sum(r["original_mapping_id"] != r["patched_mapping_id"] for r in records),
        "original_mapping_year_mismatch_count": sum(r["original_mapping_id"] > 0 and r["parsed_year"] != r["original_mapping_year"] for r in records),
        "patched_mapping_year_mismatch_count": sum(r["patched_mapping_id"] > 0 and r["parsed_year"] != r["patched_mapping_year"] for r in records),
        "exact_catalog_contract_counts": dict(Counter(r["exact_catalog_contract"]["reason"] for r in records)),
        "proposed_reachable_catalog_evaluation": "BLOCKED; full input rejected, zero samples silently filtered",
        "original_evaluation_input_preserved": True, "original_metrics_preserved": True,
        "previous_private_evidence_files_unchanged": len(previous["private_files"]),
        "archive_entries_inspected": len(json.loads((ROOT / "reproduction/logs/t03_20261001/archive_entries.json").read_text())),
        "authoritative_checkpoint_title_mapping": "NOT VERIFIED; source archive has no training ID map",
        "similarity_row_title_mapping": "NOT VERIFIED", "new_dataset_created": False,
        "new_prediction_metrics": None, "new_algorithm_evaluation": "NOT EVALUATED",
        "remote_api_requests": 0, "network_download_bytes": 0}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as f:
        f.write(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
