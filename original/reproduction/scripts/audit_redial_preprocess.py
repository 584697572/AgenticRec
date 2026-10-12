"""Exercise original notebook functions on pinned raw test data; never call an LLM.

Only independent test-item mappings are needed to audit selection. Train/valid
dialogues are not fetched, and no canonical evaluation file is claimed or written.
"""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from fetch_unicrs_test import DEST, ROOT, validate

NOTEBOOK = ROOT / "RecAI/InteRecAgent/preprocess/preprocess_redial.ipynb"
CATALOG = ROOT / "data/raw/upstream_movie/movies.ftr"
NAMES = {"extract_numbers", "separate_movie_and_year", "attach_labels", "movie_map",
         "get_first_time_appear", "merge_conversation", "process_conv"}


def original_functions():
    namespace = dict(re=re, deepcopy=deepcopy, np=np, pd=pd, fuzz=fuzz)
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    definitions = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            if isinstance(node, ast.FunctionDef) and node.name in NAMES:
                definitions.append(node)
    if {node.name for node in definitions} != NAMES:
        raise ValueError("upstream_function_set_changed")
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    return namespace


def main():
    source = validate(DEST.read_bytes())
    conversations = [json.loads(line) for line in DEST.read_text(encoding="utf-8").splitlines()]
    if hashlib.sha256(CATALOG.read_bytes()).hexdigest() != "1a0e8687ba0c366cf4ab7837c591542f9bbb3b445ec4f4155fe1e932b58bb8c9":
        raise ValueError("original_catalog_hash_mismatch")
    catalog = pd.read_feather(CATALOG)
    catalog["title"] = catalog["title"].str.lower()
    functions = original_functions()
    positives = [[item for item in functions["attach_labels"](c, functions["extract_numbers"](c["messages"]))
                  if item["liked"] == 1] for c in conversations]
    # Sorting only stabilizes the audit log; each original movie_map call is independent.
    movies = sorted({(i["title"].strip(), i["date"]) for row in positives for i in row}, key=repr)
    errors, mapping, approximate = [], {}, {}
    for title, year in movies:
        try:
            mapped, approx = functions["movie_map"]({"title": title, "date": year}, catalog)
            mapping.update(mapped)
            approximate.update(approx)
        except Exception as error:
            errors.append({"title": title, "year": year, "error_type": type(error).__name__,
                           "error": str(error)})
    result = {"scope": "original_notebook_test_mapping_audit", "status": "BLOCKED" if errors else "PASS",
              "raw_data": source, "notebook_sha256": hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest(),
              "catalog_sha256": hashlib.sha256(CATALOG.read_bytes()).hexdigest(),
              "release_date_dtype": str(catalog["release_date"].dtype),
              "original_functions_loaded": sorted(NAMES),
              "original_functions_executed": ["extract_numbers", "separate_movie_and_year", "attach_labels", "movie_map"],
              "distinct_positive_title_years": len(movies),
              "mapped": int(sum(v >= 0 for v in mapping.values())),
              "unmapped": int(sum(v < 0 for v in mapping.values())),
              "approximate_mappings": len(approximate), "mapping_errors": errors,
              "canonical_eval_data": "NOT VERIFIED", "api_requests": 0, "metrics": "NOT EVALUATED"}
    if not errors:
        selected = [c for c, items in zip(conversations, positives)
                    if sum(mapping[(i["title"].strip(), i["date"])] >= 0 for i in items) >= 2]
        derived = [functions["process_conv"](c) for c in selected[:50]]
        result.update(selected_test_count=len(selected), derived_first50_count=sum(c is not None for c in derived))
    serialized = json.dumps(result, ensure_ascii=True, indent=2) + "\n"
    out = ROOT / "reproduction/logs/resource_recheck_20261005/redial_preprocess_audit_v3.json"
    with out.open("x", encoding="utf-8") as handle:
        handle.write(serialized)
    print(json.dumps({k: v for k, v in result.items() if k != "mapping_errors"}, indent=2))
    print(json.dumps({"mapping_error_count": len(errors), "first_error": errors[:1]}, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
