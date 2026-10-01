"""Audit downloaded resource metadata without unpickling or loading model weights."""
import argparse
import ast
import hashlib
import importlib.metadata
import json
import pickletools
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/raw/upstream_audit"
LOG = ROOT / "reproduction/logs/t03_20261001"


def npy_header(prefix):
    if prefix[:6] != b"\x93NUMPY":
        raise ValueError("Invalid NPY signature")
    version = tuple(prefix[6:8])
    start = 10 if version == (1, 0) else 12
    size = struct.unpack("<H" if start == 10 else "<I", prefix[8:start])[0]
    if start + size > len(prefix):
        raise ValueError("Incomplete NPY header")
    header = ast.literal_eval(prefix[start:start+size].decode("latin1"))
    if set(header) != {"descr", "fortran_order", "shape"}:
        raise ValueError("Unexpected NPY header")
    return {"format_version": list(version), "data_offset": start+size, **header}


def static_checkpoint(prefix):
    if prefix[:4] != b"PK\x03\x04":
        raise ValueError("Checkpoint prefix is not a ZIP local header")
    fields = struct.unpack("<4s5H3I2H", prefix[:30])
    name_len, extra_len = fields[-2:]
    offset = 30 + name_len + extra_len
    name = prefix[30:30+name_len].decode("utf-8")
    if not name.endswith("/data.pkl") or fields[3] != 0:
        raise ValueError("First checkpoint entry is not an uncompressed data.pkl")
    # genops only decodes pickle instructions; it executes no GLOBAL/REDUCE calls.
    operations = list(pickletools.genops(prefix[offset:]))
    if operations[-1][0].name != "STOP":
        raise ValueError("Incomplete metadata pickle")
    allowed = {"model", "dataset", "n_items", "n_users", "embedding_size", "max_seq_len", "hidden_size"}
    scalars, embedding_shape = {}, None
    integer_ops = {"BININT", "BININT1", "BININT2", "INT", "LONG", "LONG1", "LONG4"}
    for index, (opcode, value, _) in enumerate(operations):
        if isinstance(value, str) and value in allowed and value not in scalars:
            for next_op, argument, _ in operations[index+1:index+4]:
                if next_op.name in {"BINPUT", "LONG_BINPUT", "MEMOIZE"}:
                    continue
                if next_op.name in integer_ops or next_op.name in {"BINUNICODE", "SHORT_BINUNICODE", "UNICODE"}:
                    scalars[value] = argument
                break
        if value == "item_embedding.weight":
            following = operations[index+1:index+35]
            for step, (next_op, _, _) in enumerate(following):
                if next_op.name == "BINPERSID":
                    shape_ops = following[step+1:step+5]
                    if len(shape_ops) == 4 and all(op.name in integer_ops for op, _, _ in shape_ops[:3]) and shape_ops[3][0].name == "TUPLE2":
                        embedding_shape = [shape_ops[1][1], shape_ops[2][1]]
                    break
    if "n_items" not in scalars or embedding_shape is None:
        raise ValueError("Could not safely identify item count and embedding shape")
    return {"metadata": scalars, "item_embedding_shape": embedding_shape,
            "pickle_opcode_count": len(operations), "pickle_metadata_bytes": operations[-1][2]+1,
            "method": "pickletools.genops_static_only", "weights_loaded": False,
            "complete_checkpoint_crc_verified": False}


def audit():
    import pyarrow.feather as feather  # already installed; no package installation.
    settings = json.loads((DATA / "03_settings.json").read_text(encoding="utf-8"))
    required = {"GAME_INFO_FILE", "TABLE_COL_DESC_FILE", "MODEL_CKPT_FILE", "ITEM_SIM_FILE", "USE_COLS", "CATEGORICAL_COLS"}
    assert set(settings) == required
    entries = json.loads((LOG / "archive_entries.json").read_text(encoding="utf-8"))
    by_name = {item["name"]: item for item in entries}
    for key in required - {"USE_COLS", "CATEGORICAL_COLS"}:
        assert "movie/" + settings[key] in by_name
    table = feather.read_table(DATA / "movies.ftr")
    assert set(settings["USE_COLS"]) <= set(table.column_names)
    assert set(settings["CATEGORICAL_COLS"]) <= set(settings["USE_COLS"])
    ids = table.column("id").to_pylist()
    null_counts = {name: table.column(name).null_count for name in table.column_names}
    assert len(ids) == len(set(ids)) and min(ids) == 1 and set(ids) == set(range(1, max(ids)+1))
    assert not any(null_counts.values())
    assert min(table.column("visited_num").to_pylist()) >= 0
    matrix = npy_header((DATA / "movie_sim.header.bin").read_bytes())
    assert matrix["descr"] == "<f8" and matrix["fortran_order"] is False
    assert matrix["shape"] == (max(ids)+1, max(ids)+1)
    assert matrix["data_offset"] + matrix["shape"][0] * matrix["shape"][1] * 8 == by_name["movie/movie_sim.npy"]["bytes"]
    checkpoint = static_checkpoint((DATA / "checkpoint.prefix.bin").read_bytes())
    # Execute only the original CandidateBuffer class with delayed annotations.
    path = ROOT / "RecAI/InteRecAgent/llm4crs/buffer/base.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "CandidateBuffer")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    ns = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[])), str(path), "exec"), ns)
    class GallerySize:
        def __len__(self):
            return len(ids)
    buffer = ns["CandidateBuffer"](GallerySize())
    missing = sorted(set(ids) - set(buffer.get()))
    matching_dimensions = checkpoint["metadata"]["n_items"] == matrix["shape"][0] == checkpoint["item_embedding_shape"][0]
    result = {"settings_keys": sorted(settings), "settings": settings,
              "settings_resource_members_present": True,
              "table": {"rows": len(ids), "unique_ids": len(set(ids)), "id_min": min(ids), "id_max": max(ids),
                        "padding_row_present": 0 in ids, "null_counts": null_counts,
                        "schema": {field.name: str(field.type) for field in table.schema},
                        "duplicate_title_count": len(ids)-len(set(table.column("title").to_pylist())),
                        "sha256": hashlib.sha256((DATA/"movies.ftr").read_bytes()).hexdigest(), "member_crc_verified": True},
              "matrix": {**matrix, "catalog_dimensions_match": True, "full_matrix_downloaded": False,
                         "full_crc_verified": False, "row_order_matches_model": "NOT VERIFIED"},
              "checkpoint": checkpoint, "checkpoint_catalog_dimensions_match": matching_dimensions,
              "original_buffer": {"candidate_count": len(buffer.get()), "catalog_ids_missing": missing,
                                  "tested_original_method": "CandidateBuffer.__init__", "fixed": False},
              "license_members": [item["name"] for item in entries if any(word in item["name"].lower() for word in ("license", "readme", "terms"))],
              "mapping_members": [item["name"] for item in entries if "map" in item["name"].lower()],
              "pyarrow_version": importlib.metadata.version("pyarrow"),
              "runtime_model_compatibility": "NOT VERIFIED", "runtime_metrics": "NOT EVALUATED"}
    (LOG / "resource_contract_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-alignment", action="store_true", help="Fail if the original package dimensions cannot align.")
    args = parser.parse_args()
    result = audit()
    print(json.dumps(result, indent=2))
    if args.require_alignment:
        assert result["checkpoint_catalog_dimensions_match"], "Catalog/matrix vocabulary 9889 differs from checkpoint embedding/config vocabulary 36255; no mapping provided"
