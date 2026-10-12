"""Opt-in corrections, kept separate from pristine A1 execution and artifacts.

These adapters execute fixed upstream functions; they never rewrite upstream.
They are behavior patches, not a new recommender or a new scoring metric.
"""
import ast
import json
from pathlib import Path

from audit_redial_preprocess import NOTEBOOK, original_functions

ROOT = Path(__file__).resolve().parents[2]
AGENT_SOURCE = ROOT / "RecAI/InteRecAgent/llm4crs/agent_plan_first_openai.py"


def corrected_movie_map():
    namespace = original_functions()
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    definitions = [n for c in notebook["cells"] if c["cell_type"] == "code"
                   for n in ast.parse("".join(c["source"])).body
                   if isinstance(n, ast.FunctionDef) and n.name == "movie_map"]
    if len(definitions) != 1:
        raise ValueError("upstream_movie_map_definition_changed")
    definition = definitions[0]
    changes = 0
    for node in ast.walk(definition):
        if isinstance(node, ast.Call) and ast.dump(node) == ast.dump(ast.parse("res['date_diff'].idxmin()", mode="eval").body):
            node.func.value = ast.Call(func=ast.Attribute(value=node.func.value, attr="abs", ctx=ast.Load()), args=[], keywords=[])
            changes += 1
    if changes != 1:
        raise ValueError("upstream_signed_year_expression_changed")
    compiled = ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[]))
    exec(compile(compiled, str(NOTEBOOK), "exec"), namespace)
    return namespace["movie_map"]


def corrected_agent_run(namespace):
    tree = ast.parse(AGENT_SOURCE.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "CRSAgentPlanFirstOpenAI")
    run = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    changes = 0
    for node in ast.walk(run):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "run" and isinstance(node.func.value, ast.Name) and node.func.value.id == "self":
            if [k.arg for k in node.keywords] != ["reflection"]:
                raise ValueError("upstream_reflection_call_changed")
            node.keywords.insert(0, ast.keyword(arg="chat_history", value=ast.Name(id="chat_history", ctx=ast.Load())))
            changes += 1
    if changes != 1:
        raise ValueError("upstream_reflection_recursion_changed")
    compiled = ast.fix_missing_locations(ast.Module(body=[run], type_ignores=[]))
    local = {}
    exec(compile(compiled, str(AGENT_SOURCE), "exec"), namespace, local)
    return local["run"]


def target_catalog_contract(target, catalog, split_title_year):
    """Exact title/year coverage check; never fuzzy-map or drop a sample.

    A match validates catalog reachability only, not checkpoint row semantics.
    """
    title, year = split_title_year(target)
    titles = catalog["title"].str.strip().str.casefold()
    matched = catalog.loc[titles == title.strip().casefold()]
    if matched.empty:
        return {"status": "BLOCKED", "reason": "title_absent", "ids": []}
    if year is not None:
        matched = matched.loc[matched["release_date"] == year]
        if matched.empty:
            return {"status": "BLOCKED", "reason": "year_absent", "ids": []}
    if len(matched) != 1:
        return {"status": "BLOCKED", "reason": "ambiguous_title_year", "ids": []}
    item_id = int(matched.iloc[0]["id"])
    if item_id <= 0:
        return {"status": "BLOCKED", "reason": "invalid_or_padding_id", "ids": []}
    return {"status": "PASS", "reason": "exact_catalog_match", "ids": [item_id]}


def require_reachable_targets(records, catalog, split_title_year):
    """Reject the whole proposed experiment if any target lacks a unique match."""
    contracts = [target_catalog_contract(r["target"], catalog, split_title_year) for r in records]
    if not records or any(c["status"] != "PASS" for c in contracts):
        raise ValueError("evaluation_target_catalog_contract_failed; no samples removed")
    return contracts
