"""Compare saved upstream provenance and test original ID lookup on explicit CPU fixtures.

No checkpoint is deserialized. Fixture scores are not recommendation results.
"""
import ast
import hashlib
import inspect
import json
import math
import types
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WHEEL = ROOT / "data/raw/upstream_audit/unirec-0.0.1a4-py3-none-any.whl"
NOTEBOOK = ROOT / "RecAI/InteRecAgent/preprocess/movies.ipynb"
LOG = ROOT / "reproduction/logs/t03_recheck_20261001"


def digest(body):
    return hashlib.sha256(body).hexdigest()


def wheel_source(member):
    with zipfile.ZipFile(WHEEL) as archive:
        return archive.read(member).decode("utf-8")


def compile_nodes(nodes, namespace):
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(module, "<isolated-pinned-source>", "exec"), namespace)


def fixture_model():
    import torch
    # Execute the published lookup/predict/scorer bodies, without their package imports.
    source = wheel_source("unirec/model/base/recommender.py")
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "BaseRecommender")
    names = {"predict", "forward_item_emb", "item_embedding_for_user", "_predict_layer"}
    namespace = {"torch": torch, "inspect": inspect, "math": math, "nn": torch.nn}
    compile_nodes([n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names], namespace)
    scorer = next(n for n in ast.parse(wheel_source("unirec/model/modules.py")).body if isinstance(n, ast.ClassDef) and n.name == "InnerProductScorer")
    compile_nodes([scorer], namespace)
    fixture = types.SimpleNamespace()
    for name in names:
        setattr(fixture, name, types.MethodType(namespace[name], fixture))
    # Deliberately self-authored weights. They only demonstrate index bounds/flow.
    weights = torch.arange(36255, dtype=torch.float32).unsqueeze(1).expand(-1, 32).clone()
    fixture.item_embedding = torch.nn.Embedding.from_pretrained(weights, freeze=True, padding_idx=0)
    fixture.use_features = fixture.use_text_emb = fixture.has_user_bias = fixture.has_item_bias = False
    fixture.time_seq, fixture.tau, fixture.SCORE_CLIP = 0, 1.0, -1
    fixture.scorer_layers = namespace["InnerProductScorer"]()
    fixture.calls = []
    def user_fixture(self, item_seq=None, item_seq_len=None):
        # Encoder is a fixture, NOT the original SASRec Transformer.
        self.calls.append(item_seq.detach().clone())
        return self.item_embedding_for_user(item_seq).mean(dim=1)
    fixture.forward_user_emb = types.MethodType(user_fixture, fixture)
    return fixture


def ranking_tool(model, candidates):
    import numpy as np
    import torch
    class Logger:
        def debug(self, *args):
            pass
    namespace = {"json": json, "np": np, "torch": torch, "logger": Logger()}
    path = ROOT / "RecAI/InteRecAgent/llm4crs/ranking/reco_model_tool.py"
    node = next(n for n in ast.parse(path.read_text(encoding="utf-8")).body if isinstance(n, ast.ClassDef) and n.name == "RecModelTool")
    compile_nodes([node], namespace)
    table = catalog()
    class GalleryFixture:
        def fuzzy_match(self, titles, column):
            return titles  # exact selected title; original fuzzy embeddings not tested.
        def convert_title_2_info(self, titles, col_names):
            return {"id": [next(row["id"] for row in table if row["title"] == title) for title in titles]}
    buffer_path = ROOT / "RecAI/InteRecAgent/llm4crs/buffer/base.py"
    buffer_node = next(n for n in ast.parse(buffer_path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "CandidateBuffer")
    buffer_ns = {"np": np}
    compile_nodes([buffer_node], buffer_ns)
    buffer = buffer_ns["CandidateBuffer"](table)
    buffer.push("explicit_fixture_candidates", list(candidates))
    tool = namespace["RecModelTool"].__new__(namespace["RecModelTool"])
    tool.item_corups, tool.buffer, tool.model = GalleryFixture(), buffer, model
    tool.name, tool.device, tool.rec_num, tool._mode = "original-ranking-fixture", torch.device("cpu"), 2, "accuracy"
    return tool


def catalog():
    import pyarrow.feather as feather
    return feather.read_table(ROOT / "data/raw/upstream_audit/movies.ftr").to_pylist()


def audit():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    outputs = {}
    for index in (5, 7, 8, 16):
        cell = notebook["cells"][index]
        outputs[str(index)] = "\n".join(
            "".join(output.get("text", output.get("data", {}).get("text/plain", [])))
            for output in cell.get("outputs", [])
        )
    assert "298073 36254" in outputs["7"]
    # Saved outputs identify these sample remappings; notebook is not run in this audit.
    assert "Toy Story" in outputs["16"] and "17" in outputs["16"]
    assert "Jumanji" in outputs["16"] and "105" in outputs["16"]
    assert "Grumpier Old Men" in outputs["16"] and "232" in outputs["16"]
    table = catalog()
    compare = []
    for title, notebook_id in [("Toy Story", 17), ("Jumanji", 105), ("Grumpier Old Men", 232)]:
        found = next(row for row in table if row["title"] == title)
        at_old_id = next(row for row in table if row["id"] == notebook_id)
        compare.append({"title": title, "notebook_saved_id": notebook_id,
                        "catalog_id": found["id"], "catalog_year": found["release_date"],
                        "catalog_title_at_notebook_id": at_old_id["title"]})
    summary = json.loads((ROOT / "reproduction/resource_contract_summary.json").read_text())
    source_records = []
    modules = {
        "unirec/utils/general.py": {"load_model_freely"},
        "unirec/model/base/recommender.py": {"predict", "forward_item_emb", "item_embedding_for_user"},
        "unirec/model/base/reco_abc.py": {"_init_modules"},
        "unirec/model/sequential/sasrec.py": {"forward_user_emb"},
    }
    for member, names in modules.items():
        text = wheel_source(member)
        source_records.append({"member": member, "sha256": digest(text.encode()),
                               "symbols": {n.name: [n.lineno, n.end_lineno] for n in ast.walk(ast.parse(text))
                                           if isinstance(n, ast.FunctionDef) and n.name in names}})
    checkpoint = summary["checkpoint"]["metadata"]
    return {"upstream_commit": "0959ecb05b0794748426e73e6efc1b6b35ec433d",
            "notebook_sha256": digest(NOTEBOOK.read_bytes()), "notebook_executed": False,
            "notebook_saved_counts": {"users": 298073, "items": 36254},
            "checkpoint_counts_equal_notebook_plus_padding": checkpoint["n_items"] == 36255 and checkpoint["n_users"] == 298074,
            "title_id_comparison": compare, "unirec_version_examined": "0.0.1a4",
            "checkpoint_training_unirec_version": "NOT VERIFIED", "source_records": source_records,
            "lookup_contract": "Direct item_seq/item_id embedding lookup; extra unused embedding rows are allowed",
            "authoritative_checkpoint_title_mapping": "NOT VERIFIED",
            "inference": "Saved notebook outputs and checkpoint counts suggest different catalog mappings; not proof of checkpoint row labels",
            "checkpoint_loaded": False, "native_runtime": "NOT VERIFIED", "metrics": "NOT EVALUATED"}


if __name__ == "__main__":
    result = audit()
    LOG.mkdir(parents=True, exist_ok=True)
    (LOG / "id_mapping_recheck.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result, indent=2))
