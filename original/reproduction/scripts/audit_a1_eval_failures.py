"""Diagnose the saved A1 zero-hit run without changing predictions or targets."""
import ast
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

from audit_redial_preprocess import CATALOG, NOTEBOOK, original_functions
from live_app_http import write_new
from redial_compat import adapt_release_dates
from reproduction_session_http import DATA_SHA256

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "reproduction/logs/live_reproduction_20261005"


def main():
    data_path = ROOT / "data/eval_private/upstream_redial_a1_hashseed42/test_data_50.jsonl"
    raw = data_path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == DATA_SHA256
    data = [json.loads(line) for line in raw.splitlines()]
    result = json.loads((RUN / "result.json").read_text(encoding="utf-8"))
    functions = original_functions()
    frame = adapt_release_dates(pd.read_feather(CATALOG))
    frame["title"] = frame["title"].str.lower()
    source = ROOT / "RecAI/InteRecAgent/eval/one_turn_eval.py"
    node = next(n for n in ast.parse(source.read_text(encoding="utf-8")).body
                if isinstance(n, ast.FunctionDef) and n.name == "hit_judge")
    namespace = {"re": re, "fuzz": fuzz}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    judge = namespace["hit_judge"]
    cases = []
    for i, (row, observation) in enumerate(zip(data, result["evaluation"]["recbot"]["observations"])):
        title, year = functions["separate_movie_and_year"](row["target"])
        mapping, approximate = functions["movie_map"]({"title": title, "date": year}, frame)
        item_id = int(mapping[title, year])
        mapped_year = int(frame.loc[frame["id"] == item_id, "release_date"].iloc[0].year) if item_id >= 0 else None
        context_hash = hashlib.sha256(row["context"].encode()).hexdigest()
        calls = []
        for call in result["calls"]:
            reserved = json.loads((RUN / ("request_%04d.json" % call["attempt"])).read_text(encoding="utf-8"))
            if reserved["visible_context_sha256"] == context_hash:
                calls.append(call)
        cases.append({"index": i, "target": row["target"], "parsed_title": title, "parsed_year": year,
            "original_notebook_mapped_id": item_id, "mapped_year": mapped_year,
            "approximate_title_match": bool(approximate), "original_text_hit": observation["hit"],
            "year_stripped_text_hit_diagnostic": bool(judge(observation["answer"], title)),
            "mapped_target_in_recorded_map_ids_diagnostic": item_id >= 0 and item_id in observation["mapped_ids"],
            "no_map": not observation["mapped_ids"], "finish_reasons": [c["response"]["choices"][0]["finish_reason"] for c in calls],
            "answer": observation["answer"]})
    assert len(cases) == 45 and sum(c["original_text_hit"] for c in cases) == result["evaluation"]["recbot"]["hits"]
    private = RUN / "failure_diagnostics.json"
    write_new(private, cases)
    public = {"scope": "post_run_diagnostics_only; formal original metrics and frozen data remain unchanged",
        "dataset_sha256": DATA_SHA256, "private_case_file_sha256": hashlib.sha256(private.read_bytes()).hexdigest(),
        "notebook_sha256": hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest(),
        "catalog_sha256": hashlib.sha256(CATALOG.read_bytes()).hexdigest(),
        "sample_count": 45, "formal_original_recbot_hits": sum(c["original_text_hit"] for c in cases),
        "targets_with_year_suffix": sum(c["parsed_year"] is not None for c in cases),
        "targets_mapped_by_original_notebook": sum(c["original_notebook_mapped_id"] >= 0 for c in cases),
        "targets_unmapped_by_original_notebook": sum(c["original_notebook_mapped_id"] < 0 for c in cases),
        "mapped_targets_with_different_catalog_year": sum(c["original_notebook_mapped_id"] >= 0 and c["parsed_year"] != c["mapped_year"] for c in cases),
        "year_stripped_text_hit_diagnostic_count": sum(c["year_stripped_text_hit_diagnostic"] for c in cases),
        "mapped_target_in_recorded_map_ids_diagnostic_count": sum(c["mapped_target_in_recorded_map_ids_diagnostic"] for c in cases),
        "no_map_sample_indices": [c["index"] for c in cases if c["no_map"]],
        "truncated_response_sample_indices": [c["index"] for c in cases if "length" in c["finish_reasons"]],
        "api_requests": 0, "diagnostic_metrics_are_not_formal_benchmark": True,
        "limitations": "Notebook title/year mapping retains original approximate and signed-year behavior; it is not authoritative checkpoint row labeling. Stripping a year changes the metric target representation and cannot replace the preserved original result."}
    output = ROOT / "reproduction/native_runs/20261005_session/failure_diagnostics_summary.json"
    write_new(output, public)
    print(json.dumps({k: v for k, v in public.items() if not k.endswith("sha256")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
