"""Reconstruct the fixed notebook's test selection with an explicit dtype adapter.

The resulting artifact is A1 reconstructed input, NOT the author's canonical
50-sample file or a paper A2 replication. No evaluation or LLM call occurs here.
"""
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from audit_redial_preprocess import CATALOG, NOTEBOOK, original_functions
from fetch_unicrs_test import DEST, ROOT, validate
from redial_compat import adapt_release_dates


def save_or_verify(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError("refusing_to_replace_different_evaluation_artifact")
    else:
        with path.open("xb") as handle:
            handle.write(data)


def main():
    if os.environ.get("PYTHONHASHSEED") != "42":
        raise ValueError("set PYTHONHASHSEED=42 before starting Python; original set iteration affects tied targets")
    raw = DEST.read_bytes()
    provenance = validate(raw)
    conversations = [json.loads(line) for line in raw.splitlines()]
    catalog_hash = hashlib.sha256(CATALOG.read_bytes()).hexdigest()
    if catalog_hash != "1a0e8687ba0c366cf4ab7837c591542f9bbb3b445ec4f4155fe1e932b58bb8c9":
        raise ValueError("original_catalog_hash_mismatch")
    frame = adapt_release_dates(pd.read_feather(CATALOG))
    frame["title"] = frame["title"].str.lower()
    functions = original_functions()
    positives = [[i for i in functions["attach_labels"](c, functions["extract_numbers"](c["messages"]))
                  if i["liked"] == 1] for c in conversations]
    movies = sorted({(i["title"].strip(), i["date"]) for row in positives for i in row}, key=repr)
    mapping = {}
    for title, year in movies:
        mapping.update(functions["movie_map"]({"title": title, "date": year}, frame)[0])
    selected = [c for c, items in zip(conversations, positives)
                if sum(mapping[(i["title"].strip(), i["date"])] >= 0 for i in items) >= 2]
    processed, ids, ties, errors, skipped = [], [], [], [], []
    for c in selected[:50]:
        try:
            appear = functions["get_first_time_appear"](c["messages"])
            liked = [i for i, q in c["initiatorQuestions"].items() if q["liked"] == 1 and q["suggested"] == 1]
            turns = [appear[i] for i in liked]
            if turns and turns.count(max(turns)) > 1:
                ties.append(c["conversationId"])
            item = functions["process_conv"](c)
            if item:
                processed.append(item)
                ids.append(c["conversationId"])
            else:
                skipped.append(c["conversationId"])
        except Exception as error:
            errors.append({"conversation_id": c["conversationId"], "error_type": type(error).__name__})
    if errors:
        raise ValueError("original_process_conv_errors: " + json.dumps(errors))
    data = ("\n".join(json.dumps(row, ensure_ascii=True) for row in processed) + "\n").encode()
    directory = ROOT / "data/eval_private/upstream_redial_a1_hashseed42"
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "test_data_50.jsonl"
    manifest = {"scope": "A1_reconstructed_notebook_input", "status": "PREPARED",
        "source": provenance, "notebook_sha256": hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest(),
        "catalog_sha256": catalog_hash, "output": output.relative_to(ROOT).as_posix(),
        "output_sha256": hashlib.sha256(data).hexdigest(), "output_bytes": len(data),
        "raw_dialogues": len(conversations), "positive_title_year_pairs": len(movies),
        "mapped_positive_pairs": int(sum(v >= 0 for v in mapping.values())),
        "unmapped_positive_pairs": int(sum(v < 0 for v in mapping.values())),
        "selected_test_dialogues": len(selected), "selected_prefix_count": len(selected[:50]),
        "output_rows": len(processed), "conversation_ids": ids,
        "skipped_by_original_process_conv": skipped, "tied_latest_positive_conversations": ties,
        "python_hash_seed": 42, "pandas": pd.__version__,
        "adapter": "integer_year_to_Jan1_timestamp_copy_for_original_movie_map_only",
        "signed_date_difference_behavior": "PRESERVED; earlier remake can win even when query year matches later remake",
        "train_valid_not_downloaded": "mapping depends only on each title/year and fixed catalog; test selection has no cross-dialogue state",
        "canonical_author_subset_equivalence": "NOT VERIFIED",
        "paper_A2": "NOT VERIFIED", "shared_checkpoint_id_semantics": "NOT VERIFIED",
        "api_requests": 0, "evaluation": "NOT EVALUATED"}
    save_or_verify(output, data)
    save_or_verify(directory / "manifest.json", (json.dumps(manifest, indent=2) + "\n").encode())
    print(json.dumps({k: v for k, v in manifest.items() if k not in
                     ("conversation_ids", "skipped_by_original_process_conv", "tied_latest_positive_conversations")}, indent=2))
    print(json.dumps({"skipped_count": len(skipped), "tied_target_conversation_count": len(ties)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
