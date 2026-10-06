"""Offline U1 worker: execute original ToolBox/Buffer/Map with new ranking scores.

Input and detailed output are local only. The process receives no API key.
"""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "reproduction/.runtime/InteRecAgent-compat"
sys.path.insert(0, str(SOURCE))


def execute(payload):
    from llm4crs.agent_plan_first_openai import ToolBox
    from llm4crs.buffer import CandidateBuffer
    from llm4crs.mapper import MapTool

    candidates = payload["candidate_ids"]
    ranked = payload["ranked_ids"]
    titles = {int(key): value for key, value in payload["titles"].items()}
    if (not candidates or candidates != list(range(1, len(candidates) + 1))
            or len(ranked) != len(set(ranked)) or not set(ranked) <= set(candidates)
            or set(titles) != set(candidates)):
        raise ValueError("Worker requires a contiguous frozen model item universe")

    class Gallery:
        mapped_ids = []

        def __len__(self):
            return len(candidates) + 1  # upstream buffer uses range(1, len(gallery))

        def convert_id_2_info(self, ids, col_names=None):
            self.mapped_ids = list(ids)
            return {"title": [titles[i] for i in ids]}

    class ModelRankingTool:
        name = "Movie Candidates Ranking Tool"

        def __init__(self, buffer):
            self.buffer = buffer
            self.selected_ids = []

        def run(self, inputs):
            request = json.loads(inputs)
            if request != {"schema": "model_scores"}:
                raise ValueError("U1 ranking expects precomputed, identity-checked model scores")
            available = set(self.buffer.get())
            self.selected_ids = [i for i in ranked if i in available]
            self.buffer.push(self.name, self.selected_ids)
            self.buffer.track(self.name, inputs, f"ranked {len(self.selected_ids)} candidates")
            return f"ranked {len(self.selected_ids)} candidates"

    gallery = Gallery()
    buffer = CandidateBuffer(gallery)
    ranking = ModelRankingTool(buffer)
    mapping = MapTool("Mapping Tool", "original upstream ID-to-title mapper", gallery, buffer)
    toolbox = ToolBox("U1 ToolBox", "upstream plan executor",
                      {ranking.name: ranking, mapping.name: mapping})
    plan = [{"tool_name": ranking.name, "input": '{"schema":"model_scores"}'},
            {"tool_name": mapping.name, "input": str(payload["top_k"])}]
    success, answer = toolbox.run(json.dumps(plan, ensure_ascii=False))
    return {
        "variant": "U1_upstream_rebuilt",
        "plan_source": "scripted_offline_fixture_not_llm",
        "upstream_classes": ["ToolBox", "CandidateBuffer", "MapTool"],
        "replaced_tool": "RecModelTool_by_U1_ModelRankingTool",
        "plan": plan,
        "toolbox_success": success,
        "answer": answer,
        "selected_ids": ranking.selected_ids,
        "mapped_ids": gallery.mapped_ids,
        "buffer_reset": buffer.get() == candidates,
        "tracker_tools": [step["tool"] for step in buffer.tracker],
        "profile_source": payload["profile_source"],
        "fallback_reason": payload["fallback_reason"],
        "credential_env_present": any(key in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY")
                                      or key.endswith(("_API_KEY", "_SECRET", "_ACCESS_TOKEN"))
                                      for key in os.environ),
        "remote_api_requests": 0,
    }


def main():
    payload = json.load(sys.stdin)
    # Upstream imports and tools may print diagnostics; stdout remains one JSON record.
    with redirect_stdout(io.StringIO()):
        result = execute(payload)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
