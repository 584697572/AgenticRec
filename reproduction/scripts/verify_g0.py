"""Check G0 deliverables against pinned source and the unmodified specification."""
import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPECTED = "0959ecb05b0794748426e73e6efc1b6b35ec433d"
SPEC_HASH = "d88f02c84855234673514519486e74e73c61ffd4e2ec75594020fdf9b8ebb677"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    spec = ROOT / "AgenticRec_完整复现与优化执行规范.md"
    assert digest(spec) == SPEC_HASH, "Specification changed"
    lock = json.loads((ROOT / "reproduction/upstream_lock.json").read_text(encoding="utf-8"))
    assert lock["commit"] == EXPECTED and lock["local_patches"] == []
    for record in lock["read_files"]:
        assert digest(ROOT / "RecAI" / record["path"]) == record["working_tree_sha256"], record["path"]
    assert digest(ROOT / "LICENSE.txt") == digest(ROOT / "RecAI/LICENSE.txt")
    p = subprocess.run(["git", "-C", str(ROOT / "RecAI"), "status", "--porcelain"], capture_output=True, check=True)
    assert not p.stdout, "Upstream working tree changed"
    tasks = json.loads((ROOT / "TASKS.yaml").read_text(encoding="utf-8"))["tasks"]
    expected_cards = re.findall(r"^### (T\d{2})｜[^\n]+\n(.*?)(?=^### T\d{2}｜|\Z)", spec.read_text(encoding="utf-8"), re.M | re.S)
    assert len(tasks) == len(expected_cards) == 25
    by_id = {task["id"]: task for task in tasks}
    for task_id, body in expected_cards:
        expected_deps = re.findall(r"T\d{2}", re.search(r"依赖：([^。\n]+)", body)[1])
        assert by_id[task_id]["depends_on"] == expected_deps, task_id
    def visit(task_id, chain):
        assert task_id not in chain, "Cyclic dependency: " + task_id
        for dependency in by_id[task_id]["depends_on"]:
            visit(dependency, chain + [task_id])
    for task in tasks:
        visit(task["id"], [])
        assert task["status"] in ("TODO", "IN_PROGRESS", "BLOCKED", "DONE", "SKIPPED")
        if task["status"] == "DONE":
            assert all(by_id[dependency]["status"] == "DONE" for dependency in task["depends_on"])
    assert by_id["T03"]["status"] == "IN_PROGRESS"
    assert "T18" in by_id["T16"]["depends_on"]
    checks = json.loads((ROOT / "reproduction/logs/g0/audit_result.json").read_text(encoding="utf-8"))
    assert checks["successful"] and checks["tests_run"] > 0
    failure = json.loads((ROOT / "reproduction/logs/g0/duplicate_requirement_result.json").read_text(encoding="utf-8"))
    assert failure["failures"] == 1 and failure["errors"] == 0 and not failure["successful"]
    expected_symbols = {
        "InteRecAgent/llm4crs/agent_plan_first_openai.py": ["ToolBox", "ToolBox.run", "DialogueMemory", "CRSAgentPlanFirstOpenAI.init_agent", "CRSAgentPlanFirstOpenAI.run", "CRSAgentPlanFirstOpenAI.plan_and_exe", "CRSAgentPlanFirstOpenAI._parse_llm_output", "CRSAgentPlanFirstOpenAI._summarize_recommendation"],
        "InteRecAgent/llm4crs/buffer/base.py": ["CandidateBuffer.push", "CandidateBuffer.clear", "CandidateBuffer.clear_tracks"],
        "InteRecAgent/llm4crs/query/query_tool.py": ["QueryTool.run"],
        "InteRecAgent/llm4crs/retrieval/sql_tool.py": ["SQLSearchTool.run"],
        "InteRecAgent/llm4crs/retrieval/itemcf_tool.py": ["SimilarItemTool.run"],
        "InteRecAgent/llm4crs/ranking/reco_model_tool.py": ["RecModelTool.run", "RecModelTool._rank_by_x"],
        "InteRecAgent/llm4crs/mapper/map_tool.py": ["MapTool.run"],
        "InteRecAgent/llm4crs/corups/base.py": ["BaseGallery.__call__", "BaseGallery._read_file"],
        "InteRecAgent/llm4crs/memory/memory.py": ["UserProfileMemory.update"],
        "InteRecAgent/llm4crs/critic/base.py": ["Critic.__call__"],
        "InteRecAgent/llm4crs/demo/base.py": ["DemoSelector.__call__"],
        "InteRecAgent/llm4crs/utils/open_ai.py": ["OpenAICall.call"],
        "InteRecAgent/eval/one_turn_eval.py": ["StaticAgent.run", "hit_judge", "one_turn_conversation_eval"],
        "InteRecAgent/eval/user_simulator.py": ["Simulator.__call__", "conversation_eval"],
    }
    locations = {}
    for path, required in expected_symbols.items():
        tree = ast.parse((ROOT / "RecAI" / path).read_text(encoding="utf-8"))
        symbols = {}
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                symbols[node.name] = node.lineno
                if isinstance(node, ast.ClassDef):
                    symbols.update({node.name + "." + child.name: child.lineno for child in node.body if isinstance(child, ast.FunctionDef)})
        assert all(symbol in symbols for symbol in required), path
        locations[path] = {symbol: symbols[symbol] for symbol in required}
    (ROOT / "reproduction/logs/g0/verified_source_symbols.json").write_text(json.dumps(locations, indent=2) + "\n", encoding="utf-8")
    manifest = json.loads((ROOT / "reproduction/resource_manifest.json").read_text(encoding="utf-8"))
    assert manifest["original_resource_integrity"] == "NOT VERIFIED"
    assert not manifest["archive_downloaded"]
    assert manifest["runtime_metrics"]["status"] == "not_run"
    assert all(value is None for key, value in manifest["runtime_metrics"].items() if key != "status")
    print("PASS: unchanged spec, pinned pristine upstream, license, 25 task dependencies, source symbols, audit results, honest resource status")


if __name__ == "__main__":
    verify()
