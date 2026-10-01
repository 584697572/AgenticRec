"""Execute the full original tool modules against original movie resources on CPU.

No AST extraction, synthetic catalog, synthetic matrix, or synthetic model weights.
Optional scripted LLM replay is explicitly a fixture, never a live Agent evaluation.
"""
import argparse
import json
import os
import random
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "reproduction/.runtime/InteRecAgent-compat"
LOG = ROOT / "reproduction/logs/a1_20261001"
sys.path.insert(0, str(SOURCE))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-replay", action="store_true")
    parser.add_argument("--app-startup", action="store_true")
    args = parser.parse_args()
    import numpy as np
    import torch
    from llm4crs.buffer import CandidateBuffer
    from llm4crs.corups import BaseGallery
    from llm4crs.environ_variables import (GAME_INFO_FILE, TABLE_COL_DESC_FILE, USE_COLS,
                                           CATEGORICAL_COLS, ITEM_SIM_FILE, MODEL_CKPT_FILE)
    from llm4crs.mapper import MapTool
    from llm4crs.prompt import (TOOL_NAMES, CANDIDATE_STORE_TOOL_DESC, LOOK_UP_TOOL_DESC,
                                HARD_FILTER_TOOL_DESC, SOFT_FILTER_TOOL_DESC,
                                RANKING_TOOL_DESC, MAP_TOOL_DESC)
    from llm4crs.query import QueryTool
    from llm4crs.ranking import RecModelTool
    from llm4crs.retrieval import SQLSearchTool, SimilarItemTool
    from llm4crs.utils import FuncToolWrapper
    random.seed(20261001)
    np.random.seed(20261001)
    torch.manual_seed(20261001)
    torch.set_num_threads(4)
    records = []
    result = {"scope": "original_resources_native_tools_cpu", "llm": "not_run",
              "upstream_commit": "0959ecb05b0794748426e73e6efc1b6b35ec433d",
              "shared_checkpoint_id_semantics": "NOT VERIFIED", "metrics": "NOT EVALUATED",
              "records": records, "success": False}

    def record(stage, **evidence):
        records.append(dict(stage=stage, status="PASS", **evidence))
        print(json.dumps(records[-1]), flush=True)

    try:
        print("Loading original gallery and cached thenlper/gte-base...", flush=True)
        gallery = BaseGallery(GAME_INFO_FILE, TABLE_COL_DESC_FILE, "movie_information",
                              columns=USE_COLS, fuzzy_cols=["title"] + CATEGORICAL_COLS,
                              categorical_cols=CATEGORICAL_COLS)
        assert len(gallery) == 9888 and gallery.corups.index.is_unique
        assert gallery.corups.index.min() == 1 and gallery.corups.index.max() == 9888
        fuzzy = str(np.asarray(gallery.fuzzy_match("Toy Stroy", "title")).reshape(-1)[0])
        # Semantic embeddings do not guarantee correction of this typo. Record the
        # original output honestly; the smoke contract is a valid catalog title.
        assert fuzzy in gallery.corups_title.index, fuzzy
        assert str(np.asarray(gallery.fuzzy_match("Toy Story", "title")).reshape(-1)[0]) == "Toy Story"
        record("original_gallery", catalog_rows=len(gallery), fuzzy_query="Toy Stroy", fuzzy_result=fuzzy)
        buffer = CandidateBuffer(gallery, num_limit=1000)
        domain_map = {"item": "movie", "Item": "Movie", "ITEM": "MOVIE"}
        names = {k: v.format(**domain_map) for k, v in TOOL_NAMES.items()}
        tools = {
            "BufferStoreTool": FuncToolWrapper(buffer.init_candidates, names["BufferStoreTool"], CANDIDATE_STORE_TOOL_DESC.format(**domain_map)),
            "LookUpTool": QueryTool(names["LookUpTool"], LOOK_UP_TOOL_DESC.format(**domain_map), gallery, buffer),
            "HardFilterTool": SQLSearchTool(names["HardFilterTool"], HARD_FILTER_TOOL_DESC.format(**domain_map), gallery, buffer, max_candidates_num=1000),
            "SoftFilterTool": SimilarItemTool(names["SoftFilterTool"], SOFT_FILTER_TOOL_DESC.format(**domain_map), ITEM_SIM_FILE, gallery, buffer, top_ratio=0.05),
            "RankingTool": RecModelTool(names["RankingTool"], RANKING_TOOL_DESC.format(**domain_map), MODEL_CKPT_FILE, gallery, buffer, rec_num=100),
            "MapTool": MapTool(names["MapTool"], MAP_TOOL_DESC.format(**domain_map), gallery, buffer),
        }
        matrix = tools["SoftFilterTool"].item_sim
        assert matrix.shape == (9889, 9889) and matrix.dtype == np.float64
        for start in range(0, matrix.shape[0], 256):
            assert np.isfinite(matrix[start:start + 256]).all()
        record("original_matrix_loaded", shape=list(matrix.shape), dtype=str(matrix.dtype), all_finite=True)
        model = tools["RankingTool"].model
        record("original_sasrec_loaded", class_name=type(model).__name__,
               embedding_shape=list(model.item_embedding.weight.shape), training_mode=model.training)
        lookup_sql = "SELECT id, title, release_date FROM movie_information WHERE title = 'Toy Story'"
        lookup = tools["LookUpTool"].run(lookup_sql)
        rows = json.loads(lookup)
        assert rows and rows[0]["title"] == "Toy Story" and rows[0]["id"] == 1062
        record("original_query", sql=lookup_sql, output=rows)
        stored = tools["BufferStoreTool"].run("Toy Story;;Jumanji")
        assert set(buffer.get()) == {1062, 1023}, buffer.get()
        record("original_buffer_store", output=stored, ids=[int(i) for i in buffer.get()])
        buffer.clear()
        hard_sql = "SELECT id FROM movie_information WHERE tags LIKE '%Comedy%' AND release_date >= 1990 ORDER BY visited_num DESC LIMIT 30"
        hard = tools["HardFilterTool"].run(hard_sql)
        assert "broken" not in hard and len(buffer.get()) == 30, hard
        eligible = gallery.corups.loc[buffer.get()]
        assert eligible.release_date.ge(1990).all() and eligible.tags.str.contains("Comedy").all()
        record("original_sql_filter", sql=hard_sql, output=hard, ids=[int(i) for i in buffer.get()])
        pop = tools["RankingTool"].run('{"schema":"popularity"}')
        assert "broken" not in pop and len(buffer.get()) == 30, pop
        record("original_popularity_rank", output=pop)
        buffer.clear()
        similar = tools["SoftFilterTool"].run('["Toy Story"]')
        assert "broken" not in similar and len(buffer.get()) > 5, similar
        assert np.isfinite(buffer.similarity).all() and len(buffer.similarity) == len(buffer.get())
        record("original_similar_retrieval", output=similar, candidate_count=len(buffer.get()))
        sim_rank = tools["RankingTool"].run('{"schema":"similarity"}')
        assert "broken" not in sim_rank and len(buffer.get()) > 0, sim_rank
        record("original_similarity_rank", output=sim_rank)
        candidates = torch.tensor(buffer.get(), dtype=torch.long)
        scores = model.predict({"item_seq": torch.tensor([[1062, 1023]]),
                                "item_seq_len": torch.tensor([2], dtype=torch.int), "item_id": candidates})
        assert scores.shape[-1] == len(candidates) and np.isfinite(scores).all()
        record("real_checkpoint_predict", scores_shape=list(scores.shape), all_finite=True)
        preference = tools["RankingTool"].run('{"schema":"preference","prefer":["Toy Story","Jumanji"]}')
        assert "broken" not in preference and len(buffer.get()) > 0, preference
        ranked = [int(i) for i in buffer.get()]
        assert set(ranked).issubset(set(gallery.corups.index))
        record("original_preference_rank", output=preference, ranked_ids=ranked)
        mapped = tools["MapTool"].run("3")
        expected = gallery.convert_id_2_info(ranked[:3], "title")["title"]
        assert all(title in mapped for title in expected), mapped
        record("original_map", output=mapped, ids=ranked[:3], titles=expected)
        if args.agent_replay:
            from llm4crs import agent_plan_first_openai as agent_module
            original_llm = agent_module.OpenAICall

            class ScriptedLLM:
                def __init__(self, **kwargs):
                    self.calls = []

                def call(self, user_prompt, **kwargs):
                    self.calls.append({"user_prompt": user_prompt, "sys_prompt": kwargs.get("sys_prompt")})
                    plan = [{"tool_name": names["HardFilterTool"], "input": hard_sql},
                            {"tool_name": names["RankingTool"], "input": '{"schema":"popularity"}'},
                            {"tool_name": names["MapTool"], "input": "3"}]
                    return "Action: ToolExecutor\nAction Input: " + json.dumps(plan)

            # Fixture substitution ONLY at the LLM boundary. No remote client instantiated.
            os.environ["OPENAI_API_KEY"] = "offline-fixture-placeholder"
            agent_module.OpenAICall = ScriptedLLM
            agent = agent_module.CRSAgentPlanFirstOpenAI("movie", tools, buffer, gallery,
                engine="offline-scripted-fixture", bot_type="chat", demo_mode="zero",
                enable_shorten=False, critic=None, verbose=False, enable_summarize=0)
            agent.init_agent()
            replay = []
            for text in ("Recommend three comedy movies released since 1990.",
                         "Please give me three comedy movies again."):
                answer = agent.run({"input": text})
                assert "Here are recommendations:" in answer and not answer.startswith("Something went wrong")
                assert len(buffer.tracker) == 3
                replay.append({"input": text, "output": answer, "tool_trace": buffer.tracker.copy()})
            assert len(agent.memory.memory) == 4 and len(agent.agent.calls) == 2
            assert replay[0]["input"] in agent.agent.calls[1]["user_prompt"]
            result.update(llm="scripted_fixture_only", replay=replay, memory=agent.memory.memory,
                          live_api_requests=0, live_agent="NOT VERIFIED")
            record("original_agent_two_turn_scripted_replay", memory_messages=4, remote_requests=0)
            agent_module.OpenAICall = original_llm
        if args.app_startup:
            from native_app_smoke import run_app
            result["original_app"] = run_app(gallery, names, hard_sql)
            result.update(llm="mock_transport_only", live_api_requests=0, live_agent="NOT VERIFIED")
            record("original_app_started_and_two_turn_callbacks", **result["original_app"])
        result["success"] = True
    except Exception as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        traceback.print_exc()
    filename = ("native_app_result.json" if args.app_startup else
                "native_agent_replay_result.json" if args.agent_replay else "native_tools_result.json")
    (LOG / filename).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"success": result["success"], "evidence_file": filename}), flush=True)
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
