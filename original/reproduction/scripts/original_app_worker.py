"""Private resource worker: original app and SDK, with HTTP relayed over stdio.

The supervising process supplies no real credential. This file has no live API mode.
"""
import contextlib
import io
import json
import os
import runpy
import sys
import time
from pathlib import Path
from unittest.mock import patch

from live_app_http import PROFILE, QUERY, validate_recommendations

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "reproduction/.runtime/InteRecAgent-compat"


def main():
    wire_out, wire_in = sys.stdout, sys.stdin
    def emit(data):
        wire_out.write(json.dumps(data, ensure_ascii=True) + "\n")
        wire_out.flush()
    result = {"event": "result", "status": "FAILED", "entry": "InteRecAgent/app.py",
              "real_key_in_worker": False, "ui_served": False}
    captured = io.StringIO()
    clients = []
    mapped = []
    original_dir = os.getcwd()
    try:
        # The parent constructs an environment allowlist; fail if a key leaked.
        if os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("real_key_must_not_enter_worker")
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            import httpx
            import gradio as gr
            import numpy as np
            import torch
            import random
            from openai import OpenAI
            sys.path.insert(0, str(SOURCE))
            os.chdir(SOURCE)
            from llm4crs.utils import open_ai as original
            from llm4crs.utils import extract_integers_from_string
            from llm4crs.mapper import MapTool
            random.seed(20261005)
            np.random.seed(20261005)
            torch.manual_seed(20261005)
            torch.set_num_threads(4)

            class Relay(httpx.BaseTransport):
                def handle_request(self, request):
                    if str(request.url) != PROFILE["api_base"] + "/chat/completions" or request.method != "POST":
                        raise RuntimeError("unexpected_sdk_request")
                    emit({"event": "request", "body": json.loads(request.content)})
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
                    kw["max_tokens"] = min(kw.get("max_tokens", 512), 512)
                    return create(**kw, extra_body={"thinking": {"type": "disabled"}})
                client.chat.completions.create = bounded_create
                clients.append(client)
                return client

            old_init = original.OpenAICall.__init__
            def bounded_init(self, *args, **kwargs):
                kwargs["retry_limits"] = 1
                kwargs["timeout"] = 60
                old_init(self, *args, **kwargs)

            old_map = MapTool.run
            def observe_map(self, inputs):
                before = list(self.buffer.get())
                try:
                    n = extract_integers_from_string(inputs)[0]
                except Exception:
                    n = 5
                ids = before[:min(n, self.max_rec_num)]
                output = old_map(self, inputs)
                mapped.append({"ids": [int(i) for i in ids], "output": output})
                return output

            # Placeholder stays in the worker. Parent is the only live HTTP sender.
            os.environ.update(OPENAI_API_KEY="stdio-relay-placeholder", OPENAI_API_TYPE="open_ai",
                              OPENAI_API_BASE=PROFILE["api_base"])
            sys.argv = ["app.py", "--engine", PROFILE["model_id"], "--bot_type", "chat",
                        "--plan_first", "1", "--langchain", "0", "--demo_mode", "zero",
                        "--enable_reflection", "0", "--enable_shorten", "0",
                        "--reply_style", "concise", "--max_output_tokens", "512"]
            with patch.object(original, "OpenAI", client_factory), \
                 patch.object(original.OpenAICall, "__init__", bounded_init), \
                 patch.object(MapTool, "run", observe_map), \
                 patch.object(gr.Blocks, "launch", lambda *args, **kwargs: None):
                emit({"event": "progress", "stage": "loading_original_app_and_cached_models"})
                namespace = runpy.run_path(str(SOURCE / "app.py"), run_name="__main__")
                emit({"event": "ready", "tool_names": namespace["tool_names"]})
                state = [namespace["default_chat_value"]]
                _, state, _ = namespace["user"](QUERY, state)
                clock = time.perf_counter()
                _, state = namespace["bot"].run_gr(state)
                answer = state[-1][1]
                rows = []
                gallery = namespace["item_corups"].corups
                if mapped:
                    for item_id in mapped[-1]["ids"]:
                        row = gallery.loc[item_id]
                        rows.append({"id": item_id, "title": str(row["title"]),
                                     "year": int(row["release_date"]), "tags": str(row["tags"])})
                trace = namespace["candidate_buffer"].tracker.copy()
                required = [namespace["tool_names"][name] for name in ("HardFilterTool", "RankingTool", "MapTool")]
                validation = validate_recommendations(rows, answer, [t["tool"] for t in trace], required)
                result.update(status="SUCCESS" if validation["passed"] else "FAILED", query=QUERY,
                    answer=answer, mapped_items=rows, map_observations=mapped,
                    tool_trace=trace, validation=validation,
                    memory=namespace["bot"].memory.memory,
                    turn_elapsed_seconds=time.perf_counter() - clock)
    except Exception as error:
        result["error_type"] = type(error).__name__
        # A keyless worker can retain a private local traceback for debugging.
        import traceback
        captured.write(traceback.format_exc())
    finally:
        for client in clients:
            client.close()
        os.chdir(original_dir)
        emit({"event": "worker_log", "text": captured.getvalue()})
        emit(result)
    return 0 if result["status"] == "SUCCESS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
