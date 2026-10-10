"""Run U1 model-scored plans through pinned upstream tools in legacy Python."""
import json
import math
import os
from pathlib import Path
import subprocess

from ..data import digest
from ..training import ROOT
from .model import RouteResult


UPSTREAM_SOURCES = {
    "llm4crs/agent_plan_first_openai.py": "14f6598930e31ff7d78f2235b0e76680ca86985de875d3ae6fa00c59cc6c8bb3",
    "llm4crs/buffer/base.py": "0a5739335a98071132470ef42861585aafd2d7eb0e623bedc61b6c9d8ba5682f",
    "llm4crs/mapper/map_tool.py": "804fae0dadc9dc563b0cd1193fad1667f4146fe703def5c32466b1be86b20b28",
}
UPSTREAM_SHA256 = UPSTREAM_SOURCES["llm4crs/agent_plan_first_openai.py"]


def run_upstream_plan(route: RouteResult, titles_by_id, *, top_k=3, root=ROOT,
                      tool_plan=None):
    root = Path(root)
    if type(top_k) is not int or not 1 <= top_k <= 20:
        raise ValueError("top_k must be in 1..20")
    if (len(route.candidate_ids) != len(route.scores)
            or any(type(score) not in (int, float) or not math.isfinite(score) for score in route.scores)
            or not set(route.ranked_ids) <= set(route.candidate_ids)
            or len(route.ranked_ids) != len(set(route.ranked_ids))):
        raise ValueError("Unaligned or illegal route")
    if set(titles_by_id) != set(route.candidate_ids):
        raise ValueError("Title map must cover exactly the scored candidate IDs")
    if any(type(title) is not str or not title for title in titles_by_id.values()):
        raise ValueError("Every candidate needs a nonempty catalog title")
    for relative, sha256 in UPSTREAM_SOURCES.items():
        original = root / "RecAI/InteRecAgent" / relative
        compat_source = root / "reproduction/.runtime/InteRecAgent-compat" / relative
        if digest(original) != sha256 or digest(compat_source) != sha256:
            raise ValueError("Pinned upstream tool source differs: " + relative)
    compat = root / "reproduction/.runtime/InteRecAgent-compat" / "llm4crs/agent_plan_first_openai.py"
    python = root / "reproduction/.venv-legacy/Scripts/python.exe"
    worker = root / "reproduction/scripts/t10_upstream_worker.py"
    if not python.is_file() or not worker.is_file():
        raise FileNotFoundError("Legacy bridge or its pinned environment missing")
    env = {key: os.environ[key] for key in
           ("SYSTEMROOT", "WINDIR", "TEMP", "TMP", "PATH", "COMSPEC") if key in os.environ}
    profile = root / "reproduction/.runtime/isolated_profile"
    profile.mkdir(parents=True, exist_ok=True)
    env.update(DOMAIN="movie", PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
               USERPROFILE=str(profile), LOCALAPPDATA=str(profile / "AppData/Local"),
               APPDATA=str(profile / "AppData/Roaming"),
               PYTHONPATH=str(compat.parents[1]), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")
    payload = {
        "candidate_ids": route.candidate_ids,
        "ranked_ids": route.ranked_ids,
        "titles": {str(i): titles_by_id[i] for i in route.candidate_ids},
        "top_k": top_k,
        "profile_source": route.profile_source,
        "fallback_reason": route.fallback_reason,
    }
    if tool_plan is not None:
        allowed = {"Movie Candidates Ranking Tool", "Mapping Tool"}
        if (type(tool_plan) is not list or not 2 <= len(tool_plan) <= 4
                or any(type(step) is not dict or set(step) != {"tool_name", "input"}
                       or step["tool_name"] not in allowed for step in tool_plan)
                or tool_plan[-1] != {"tool_name": "Mapping Tool", "input": str(top_k)}
                or any(step["input"] != ('{"schema":"model_scores"}'
                        if step["tool_name"] == "Movie Candidates Ranking Tool"
                        else str(top_k)) for step in tool_plan)
                or tool_plan[0]["tool_name"] != "Movie Candidates Ranking Tool"):
            raise ValueError("unsafe upstream rebuilt plan")
        payload["tool_plan"] = tool_plan
    completed = subprocess.run([str(python), str(worker)], input=json.dumps(payload),
                               text=True, encoding="utf-8", capture_output=True,
                               cwd=root, env=env, timeout=90, check=False)
    if completed.returncode != 0:
        raise RuntimeError("Legacy upstream worker failed: " + completed.stderr[-800:])
    output = json.loads(completed.stdout)
    if (not output["toolbox_success"] or output["selected_ids"] != route.ranked_ids
            or output["mapped_ids"] != route.ranked_ids[:top_k]
            or not output["buffer_reset"] or output["credential_env_present"]):
        raise ValueError("Upstream bridge violated ranking, mapping, buffer or environment contract")
    return output
