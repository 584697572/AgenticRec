"""Drive the unchanged original App and evaluator without a real API credential."""
import argparse
import contextlib
import hashlib
import io
import json
import os
import random
import runpy
import sys
import time
from pathlib import Path
from unittest.mock import patch

from reproduction_session_http import AUTHORIZED_PROFILE, DATA_SHA256, QUERIES, validate_turn

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "reproduction/.runtime/InteRecAgent-compat"
DATA = ROOT / "data/eval_private/upstream_redial_a1_hashseed42/test_data_50.jsonl"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "reproduction/logs"):
        parser.error("worker output must be within reproduction/logs")
    wire_in, wire_out = sys.stdin, sys.stdout
    def emit(event):
        wire_out.write(json.dumps(event, ensure_ascii=True) + "\n")
        wire_out.flush()
    captured = io.StringIO()
    clients, maps, prompts = [], [], []
    phase = {"stage": "initialization"}
    result = {"event": "result", "status": "FAILED", "real_key_in_worker": False,
              "ui_served": False, "original_evaluation": "NOT EVALUATED"}
    old_cwd, old_argv = os.getcwd(), sys.argv[:]
    try:
        if os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("real_key_must_not_enter_worker")
        raw = DATA.read_bytes()
        if hashlib.sha256(raw).hexdigest() != DATA_SHA256:
            raise ValueError("frozen_evaluation_input_hash_mismatch")
        records = [json.loads(line) for line in raw.splitlines()]
        if len(records) != 45 or any(set(r) != {"context", "target"} for r in records):
            raise ValueError("frozen_evaluation_input_schema_mismatch")
        if any(r["target"].casefold() in r["context"].casefold() for r in records):
            raise ValueError("evaluation_target_already_visible")
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            import gradio as gr
            import httpx
            import numpy as np
            import torch
            from openai import OpenAI
            sys.path.insert(0, str(SOURCE))
            os.chdir(SOURCE)
            from llm4crs.utils import open_ai as original
            from llm4crs.utils import extract_integers_from_string
            from llm4crs.mapper import MapTool
            torch.set_num_threads(4)
            def seed():
                random.seed(42)
                np.random.seed(42)
                torch.manual_seed(42)
            seed()

            class Relay(httpx.BaseTransport):
                def handle_request(self, request):
                    if str(request.url) != AUTHORIZED_PROFILE["api_base"] + "/chat/completions" or request.method != "POST":
                        raise RuntimeError("unexpected_sdk_request")
                    body = json.loads(request.content)
                    prompts.append({"stage": dict(phase), "prompt": body["messages"][-1]["content"]})
                    emit({"event": "request", "body": body})
                    packet = json.loads(wire_in.readline())
                    if "transport_error" in packet:
                        raise httpx.TransportError(packet["transport_error"])
                    return httpx.Response(packet["status_code"], content=packet["content"].encode("utf-8"),
                                          headers={"content-type": "application/json"})

            def client_factory(**kwargs):
                client = OpenAI(**kwargs, max_retries=0,
                    http_client=httpx.Client(transport=Relay(), follow_redirects=False))
                create = client.chat.completions.create
                def bounded_create(**kw):
                    kw["max_tokens"] = min(kw.get("max_tokens", 1024), 1024)
                    return create(**kw, extra_body={"thinking": {"type": "disabled"}})
                client.chat.completions.create = bounded_create
                clients.append(client)
                return client

            original_init = original.OpenAICall.__init__
            def bounded_init(self, *args, **kwargs):
                kwargs["retry_limits"] = 1
                kwargs["timeout"] = 60
                original_init(self, *args, **kwargs)

            original_map = MapTool.run
            def observe_map(self, inputs):
                ids = list(self.buffer.get())
                try:
                    count = extract_integers_from_string(inputs)[0]
                except Exception:
                    count = 5
                count = min(count, self.max_rec_num)
                answer = original_map(self, inputs)
                maps.append({"ids": [int(i) for i in ids[:count]], "output": answer,
                             "stage": dict(phase)})
                return answer

            os.environ.update(OPENAI_API_KEY="stdio-relay-placeholder", OPENAI_API_TYPE="open_ai",
                              OPENAI_API_BASE=AUTHORIZED_PROFILE["api_base"], AGENT_ENGINE="deepseek-flash")
            sys.argv = ["app.py", "--engine", "deepseek-flash", "--bot_type", "chat",
                "--plan_first", "1", "--langchain", "0", "--demo_mode", "zero",
                "--enable_reflection", "0", "--enable_shorten", "0", "--reply_style", "concise",
                "--max_output_tokens", "1024"]
            with patch.object(original, "OpenAI", client_factory), \
                 patch.object(original.OpenAICall, "__init__", bounded_init), \
                 patch.object(MapTool, "run", observe_map), \
                 patch.object(gr.Blocks, "launch", lambda *args, **kwargs: None):
                emit({"event": "progress", "stage": "loading_original_app_and_cached_models"})
                app = runpy.run_path(str(SOURCE / "app.py"), run_name="__main__")
                emit({"event": "ready", "tool_names": app["tool_names"]})
                gallery = app["item_corups"]
                required = [app["tool_names"][n] for n in ("HardFilterTool", "RankingTool", "MapTool")]
                state = [app["default_chat_value"]]
                turns = []
                for index, query in enumerate(QUERIES):
                    phase.clear()
                    phase.update(stage="multiturn", index=index)
                    emit({"event": "context", "stage": "multiturn", "index": index})
                    before = len(maps)
                    _, state, _ = app["user"](query, state)
                    clock = time.perf_counter()
                    _, state = app["bot"].run_gr(state)
                    answer = state[-1][1]
                    rows = []
                    if len(maps) > before:
                        for item_id in maps[-1]["ids"]:
                            row = gallery.corups.loc[item_id]
                            rows.append({"id": item_id, "title": str(row["title"]),
                                "year": int(row["release_date"]), "tags": str(row["tags"])})
                    trace = app["candidate_buffer"].tracker.copy()
                    validation = validate_turn(rows, answer, [t["tool"] for t in trace], required, index)
                    turns.append({"query": query, "answer": answer, "mapped_items": rows,
                        "tool_trace": trace, "validation": validation,
                        "memory_entries": len(app["bot"].memory.memory),
                        "elapsed_seconds": time.perf_counter() - clock})
                    emit({"event": "progress", "stage": "multiturn_completed", "index": index,
                          "validation_passed": validation["passed"]})
                second_prompts = [p["prompt"] for p in prompts
                    if p["stage"].get("stage") == "multiturn" and p["stage"].get("index") == 1]
                history_ok = bool(second_prompts) and any(QUERIES[0] in p and turns[0]["answer"] in p for p in second_prompts)
                multi_ok = all(t["validation"]["passed"] for t in turns) and history_ok and turns[-1]["memory_entries"] == 4
                result["multiturn"] = {"status": "SUCCESS" if multi_ok else "FAILED", "turns": turns,
                    "previous_input_and_answer_visible_on_second_turn": history_ok,
                    "memory": app["bot"].memory.memory}

                # Load the unchanged evaluator and invoke its actual main() with known args.
                namespace = runpy.run_path(str(SOURCE / "eval/one_turn_eval.py"), run_name="__a1_eval__")
                g = namespace["main"].__globals__
                original_eval = g["one_turn_conversation_eval"]
                evaluations = {}
                strategy = {"name": "recbot"}
                def reuse_gallery(*args, **kwargs):
                    if Path(args[0]).resolve() != Path(gallery.fpath).resolve() or args[2] != gallery.name:
                        raise ValueError("original_gallery_reuse_contract_mismatch")
                    if kwargs.get("columns") != list(gallery.column_meaning):
                        raise ValueError("original_gallery_schema_mismatch")
                    return gallery
                g["BaseGallery"] = reuse_gallery
                def observe_eval(data, agent):
                    if data != records:
                        raise ValueError("original_evaluator_input_changed")
                    observed = []
                    class ObservedAgent:
                        def run(self, context):
                            index = len(observed)
                            if context != records[index]["context"]:
                                raise ValueError("only_visible_context_may_enter_agent")
                            phase.clear()
                            phase.update(stage="evaluation", index=index, agent=strategy["name"])
                            if strategy["name"] == "recbot":
                                emit({"event": "context", "stage": "evaluation", "index": index})
                            clock = time.perf_counter()
                            offset = len(maps)
                            answer = agent.run(context)
                            item_ids = maps[-1]["ids"] if len(maps) > offset else []
                            row = {"index": index, "answer": answer,
                                "hit": bool(g["hit_judge"](answer, records[index]["target"])),
                                "mapped_ids": item_ids, "elapsed_seconds": time.perf_counter() - clock,
                                "error_reply": answer.startswith("Something went wrong"),
                                "tool_trace": agent.bot.candidate_buffer.tracker.copy() if hasattr(agent, "bot") else []}
                            observed.append(row)
                            emit({"event": "sample_completed", "agent": strategy["name"],
                                  "index": index, "hit": row["hit"], "error_reply": row["error_reply"]})
                            return answer
                    metrics, conversations = original_eval(data, ObservedAgent())
                    hits = sum(row["hit"] for row in observed)
                    if len(observed) != 45 or metrics["hit"] != hits / 45:
                        raise ValueError("original_metric_recomputation_mismatch")
                    evaluations[strategy["name"]] = {"metric_name": "upstream_text_hit",
                        "metrics_from_original_function": metrics, "samples": 45, "hits": hits,
                        "observations": observed, "conversations": conversations}
                    return metrics, conversations
                g["one_turn_conversation_eval"] = observe_eval
                for method in ("recbot", "random", "popularity"):
                    seed()
                    strategy["name"] = method
                    sys.argv = ["one_turn_eval.py", "--agent", method, "--data", str(DATA),
                        "--save", str(output / ("original_%s_conversations.jsonl" % method)),
                        "--bot_type", "chat", "--plan_first", "1", "--langchain", "0",
                        "--demo_mode", "zero", "--enable_reflection", "0", "--enable_shorten", "0",
                        "--num_rec", "5", "--max_output_tokens", "1024"]
                    namespace["main"]()
                    emit({"event": "progress", "stage": "original_evaluation_completed", "agent": method,
                          "samples": 45, "hits": evaluations[method]["hits"]})
                result.update(status="SUCCESS" if multi_ok else "FAILED", evaluation=evaluations,
                    original_evaluation="A1 COMPLETED", gallery_reused_from_original_constructor=True,
                    evaluation_data_sha256=DATA_SHA256, seed=42,
                    canonical_author_subset_equivalence="NOT VERIFIED", paper_A2="NOT VERIFIED",
                    shared_checkpoint_id_semantics="NOT VERIFIED")
    except Exception as error:
        result.update(error_type=type(error).__name__, failed_stage=dict(phase))
        import traceback
        captured.write(traceback.format_exc())
    finally:
        for client in clients:
            client.close()
        os.chdir(old_cwd)
        sys.argv = old_argv
        emit({"event": "worker_log", "text": captured.getvalue()})
        emit(result)
    return 0 if result["status"] == "SUCCESS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
