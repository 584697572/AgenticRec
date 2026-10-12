"""G0 tests of pinned source, isolated from imports, models and live APIs.

Only selected upstream AST definitions are executed with postponed annotations.
Method bodies remain unchanged. Gallery/filter/ranker/LLM are explicit fixtures.
This is control-flow evidence, never A1/A2 reproduction or recommendation quality.
"""
import argparse
import ast
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import types
import unittest
from ast import literal_eval
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "RecAI"
OUT = ROOT / "reproduction/logs/g0"
SHA = "0959ecb05b0794748426e73e6efc1b6b35ec433d"
SPEC_SHA = "d88f02c84855234673514519486e74e73c61ffd4e2ec75594020fdf9b8ebb677"
AGENT_PATH = "InteRecAgent/llm4crs/agent_plan_first_openai.py"
READ_PATHS = [
    "LICENSE.txt", "InteRecAgent/README.md", "InteRecAgent/requirements.txt",
    "InteRecAgent/app.py", "InteRecAgent/llm4crs/environ_variables.py", AGENT_PATH,
    "InteRecAgent/llm4crs/buffer/base.py", "InteRecAgent/llm4crs/query/query_tool.py",
    "InteRecAgent/llm4crs/retrieval/sql_tool.py", "InteRecAgent/llm4crs/retrieval/itemcf_tool.py",
    "InteRecAgent/llm4crs/ranking/reco_model_tool.py", "InteRecAgent/llm4crs/mapper/map_tool.py",
    "InteRecAgent/llm4crs/corups/base.py", "InteRecAgent/llm4crs/memory/memory.py",
    "InteRecAgent/llm4crs/critic/base.py", "InteRecAgent/llm4crs/demo/base.py",
    "InteRecAgent/llm4crs/utils/open_ai.py", "InteRecAgent/llm4crs/utils/util.py",
    "InteRecAgent/llm4crs/utils/__init__.py", "InteRecAgent/llm4crs/__init__.py",
    "InteRecAgent/llm4crs/prompt/__init__.py", "InteRecAgent/llm4crs/prompt/system.py",
    "InteRecAgent/llm4crs/prompt/tool.py", "InteRecAgent/eval/one_turn_eval.py",
    "InteRecAgent/eval/user_simulator.py", "InteRecAgent/eval/eval_single_turn.sh",
]


def git(*args):
    completed = subprocess.run(["git", "-C", str(UPSTREAM), *args], capture_output=True, check=True)
    return completed.stdout.decode("utf-8").strip()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_definitions(path, names, namespace):
    tree = ast.parse((UPSTREAM / path).read_text(encoding="utf-8"), filename=path)
    selected = [node for node in tree.body if getattr(node, "name", None) in names]
    assert {node.name for node in selected} == set(names)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[]))
    exec(compile(module, str(UPSTREAM / path), "exec"), namespace)


class QuietLogger:
    def debug(self, *args):
        pass


class ScriptedLLM:
    def __init__(self, **kwargs):
        self.outputs = []
        self.prompts = []

    def call(self, **kwargs):
        self.prompts.append(kwargs)
        return self.outputs.pop(0)


def namespace():
    ns = {"json": json, "re": re, "deepcopy": deepcopy, "literal_eval": literal_eval,
          "logger": QuietLogger(), "os": types.SimpleNamespace(environ={"OPENAI_API_KEY": "fixture-not-a-key"}),
          "OpenAICall": ScriptedLLM, "extract_integers_from_string": lambda text: [int(n) for n in re.findall(r"\d+", text)]}
    # Audited prompt modules contain constants and string formatting, no imports/I/O.
    for path in ("InteRecAgent/llm4crs/prompt/tool.py", "InteRecAgent/llm4crs/prompt/system.py"):
        tree = ast.parse((UPSTREAM / path).read_text(encoding="utf-8"))
        assert all(isinstance(node, (ast.Assign, ast.Expr)) for node in tree.body)
        exec(compile(tree, str(UPSTREAM / path), "exec"), ns)
    load_definitions(AGENT_PATH, ("ToolBox", "DialogueMemory", "CRSAgentPlanFirstOpenAI"), ns)
    load_definitions("InteRecAgent/llm4crs/buffer/base.py", ("CandidateBuffer",), ns)
    load_definitions("InteRecAgent/llm4crs/mapper/map_tool.py", ("MapTool",), ns)
    return ns


class FixtureGallery:
    """Self-authored catalog: zero real user data and no SQL/model implementation."""
    titles = {0: "Fixture padding", 1: "Fixture Comedy A", 2: "Fixture Drama B", 3: "Fixture Comedy C"}

    def __len__(self):
        return len(self.titles)

    def info(self, **kwargs):
        return "Fixture metadata: id, title; no real dataset."

    def convert_id_2_info(self, ids, col_names):
        return {"title": [self.titles[item] for item in ids]}


class FixtureTool:
    def __init__(self, name, buffer, events, kind):
        self.name, self.buffer, self.events, self.kind = name, buffer, events, kind
        self.desc = "Explicit G0 fixture, not an actual recommendation tool."

    def run(self, inputs):
        before = list(self.buffer.get())
        if self.kind == "filter":
            self.buffer.push(self.name, [item for item in before if item in (1, 3)])
        elif self.kind == "rank":
            self.buffer.push(self.name, list(reversed(before)))
        self.events.append({"tool": self.name, "input": inputs, "before": before, "after": list(self.buffer.get())})
        self.buffer.track(self.name, inputs, "G0 fixture execution")
        return "G0 fixture execution"


class UpstreamAudit(unittest.TestCase):
    def setUp(self):
        self.ns = namespace()
        self.gallery = FixtureGallery()
        self.buffer = self.ns["CandidateBuffer"](self.gallery)
        self.events = []
        names = {key: value.format(Item="Movie", item="movie", ITEM="MOVIE") for key, value in self.ns["TOOL_NAMES"].items()}
        self.tools = {
            "BufferStoreTool": FixtureTool(names["BufferStoreTool"], self.buffer, self.events, "store"),
            "LookUpTool": FixtureTool(names["LookUpTool"], self.buffer, self.events, "lookup"),
            "HardFilterTool": FixtureTool(names["HardFilterTool"], self.buffer, self.events, "filter"),
            "SoftFilterTool": FixtureTool(names["SoftFilterTool"], self.buffer, self.events, "similarity"),
            "RankingTool": FixtureTool(names["RankingTool"], self.buffer, self.events, "rank"),
            "MapTool": self.ns["MapTool"](names["MapTool"], "G0 real MapTool body", self.gallery, self.buffer),
        }

    def toolbox(self):
        return self.ns["ToolBox"]("ToolExecutor", "G0 fixture", {tool.name: tool for tool in self.tools.values()})

    def duplicate_plan(self):
        return json.dumps([{"tool_name": "Movie Information Look Up Tool", "input": "first-query"},
                           {"tool_name": "Movie Information Look Up Tool", "input": "second-query"}])

    def test_pin_and_pristine_source(self):
        self.assertEqual(git("rev-parse", "HEAD"), SHA)
        self.assertEqual(git("diff", "--name-only", SHA, "--", "InteRecAgent", "LICENSE.txt"), "")
        spec = ROOT / "AgenticRec_完整复现与优化执行规范.md"
        self.assertEqual(hashlib.sha256(spec.read_bytes()).hexdigest(), SPEC_SHA)

    def test_duplicate_overwrite_is_observed(self):
        success, _ = self.toolbox().run(self.duplicate_plan())
        self.assertTrue(success)
        self.assertEqual([event["input"] for event in self.events], ["second-query"])
        dump(OUT / "duplicate_observed.json", {"classification": "upstream_bug_observed_unfixed", "events": self.events})

    def test_duplicate_preservation_requirement(self):
        self.toolbox().run(self.duplicate_plan())
        self.assertEqual([event["input"] for event in self.events], ["first-query", "second-query"])

    def test_actual_agent_buffer_map_control_flow(self):
        agent = self.ns["CRSAgentPlanFirstOpenAI"]("movie", self.tools, self.buffer, self.gallery,
                  engine="fixture", bot_type="chat", enable_shorten=False, demo_mode="zero", critic=None)
        # Use actual initialization with a ScriptedLLM, never read the real environment.
        agent.init_agent()
        steps = [{"tool_name": self.tools[key].name, "input": value} for key, value in
                 (("HardFilterTool", "fixture-comedy-filter"), ("RankingTool", '{"schema":"popularity"}'), ("MapTool", "2"))]
        agent.agent.outputs = ["Action: ToolExecutor\nAction Input: " + json.dumps(steps), "Fixture answer: Fixture Comedy C; Fixture Comedy A"]
        response = agent.run({"input": "Fixture request for two comedies"})
        self.assertEqual([event["after"] for event in self.events], [[1, 3], [3, 1]])
        self.assertEqual([row["tool"] for row in self.buffer.tracker], [tool["tool_name"] for tool in steps])
        self.assertEqual(self.buffer.get(), [1, 2, 3])  # Map clears candidates to initial range.
        self.assertEqual(len(agent.agent.prompts), 2)
        self.assertEqual([row["role"] for row in agent.memory.memory], ["Human", "Assistent"])
        self.assertIn("Fixture Comedy C; Fixture Comedy A", self.buffer.tracker[-1]["output"])
        dump(OUT / "control_flow_trace.json", {"kind": "isolated_upstream_control_flow_fixture",
             "actual_bodies": ["CRSAgentPlanFirstOpenAI", "ToolBox", "DialogueMemory", "CandidateBuffer", "MapTool"],
             "substitutions": ["LLM", "Gallery", "LookUp", "HardFilter", "Ranking"],
             "events": self.events, "tracker": self.buffer.tracker, "memory": agent.memory.memory,
             "response": response, "real_api_requests": 0, "recommendation_metrics": "NOT EVALUATED"})

    def test_missing_settings_fails_before_any_model_or_api(self):
        path = UPSTREAM / "InteRecAgent/llm4crs/environ_variables.py"
        env = {"__file__": str(path), "__name__": "g0_isolated_resource_contract"}
        # This module has only stdlib imports and reads settings; no upstream import.
        previous = os.environ.get("DOMAIN")
        os.environ["DOMAIN"] = "movie"
        try:
            with self.assertRaises(FileNotFoundError) as caught:
                exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), env)
            self.assertEqual(Path(caught.exception.filename), UPSTREAM / "InteRecAgent/resources/movie/settings.json")
            dump(OUT / "missing_settings.json", {"exception": "FileNotFoundError", "missing": "RecAI/InteRecAgent/resources/movie/settings.json"})
        finally:
            if previous is None:
                os.environ.pop("DOMAIN", None)
            else:
                os.environ["DOMAIN"] = previous


def inventory():
    records = []
    for path in READ_PATHS:
        data = (UPSTREAM / path).read_bytes()
        records.append({"path": path, "working_tree_sha256": hashlib.sha256(data).hexdigest(),
                        "git_blob": git("rev-parse", SHA + ":" + path), "lines": len(data.splitlines())})
    dump(ROOT / "reproduction/upstream_lock.json", {
        "repository_url": "https://github.com/microsoft/RecAI", "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"), "verified_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "checkout_path": "RecAI", "source_acquisition": "git clone", "license": "MIT",
        "license_path": "RecAI/LICENSE.txt", "license_sha256": records[0]["working_tree_sha256"],
        "local_patches": [], "local_patch_sha256": None, "read_files": records})
    dump(ROOT / "reproduction/environment_inventory.json", {
        "python_version": platform.python_version(), "os": platform.system(), "os_release": platform.release(),
        "machine": platform.machine(), "legacy_environment": "not_created", "dev_environment": "not_created",
        "dependencies_installed": False, "torch_cuda_runtime": "NOT VERIFIED",
        "allow_paid_api": False, "api_request_cap": 0})
    print("Pinned source inventory and environment inventory written.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-duplicates", action="store_true", help="Run the unfixed preservation requirement; exit 1 is expected.")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    inventory()
    loader = unittest.TestLoader()
    if args.require_duplicates:
        suite = unittest.TestSuite([UpstreamAudit("test_duplicate_preservation_requirement")])
    else:
        suite = unittest.TestSuite(UpstreamAudit(name) for name in loader.getTestCaseNames(UpstreamAudit)
                                  if name != "test_duplicate_preservation_requirement")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    dump(OUT / ("duplicate_requirement_result.json" if args.require_duplicates else "audit_result.json"), {
        "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
        "successful": result.wasSuccessful(), "requirement_still_unfixed": args.require_duplicates})
    sys.exit(0 if result.wasSuccessful() else 1)
