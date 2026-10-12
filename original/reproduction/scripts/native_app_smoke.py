"""Run the untouched app.py with real tools; only HTTP LLM responses are fixtures."""
import json
import os
import runpy
import socket
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "reproduction/.runtime/InteRecAgent-compat"


def run_app(gallery, names, hard_sql):
    import gradio as gr
    import httpx
    from openai import OpenAI
    from llm4crs import corups
    from llm4crs.utils import open_ai as original
    calls = []
    clients = []
    launched = []
    plan = [{"tool_name": names["HardFilterTool"], "input": hard_sql},
            {"tool_name": names["RankingTool"], "input": '{"schema":"popularity"}'},
            {"tool_name": names["MapTool"], "input": "3"}]

    def transport(request):
        assert request.url.host == "offline.invalid" and request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        assert body["model"] == "offline-sdk-fixture"
        prompt = body["messages"][-1]["content"]
        if prompt.startswith("Previous Chat History:"):
            assert "Here are recommendations:" in prompt
            # Return the actual original mapping output as fixture summarization.
            content = prompt.split("Execution Result: \n", 1)[1].split(".\n\nPlease", 1)[0]
            phase = "summary_fixture"
        else:
            content = "Action: ToolExecutor\nAction Input: " + json.dumps(plan)
            phase = "plan_fixture"
        calls.append({"phase": phase, "body": body})
        return httpx.Response(200, json={"id": "fixture", "object": "chat.completion", "created": 0,
            "model": "offline-sdk-fixture", "choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}})

    def client_factory(**kwargs):
        client = OpenAI(**kwargs, max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(transport)))
        clients.append(client)
        return client

    def gallery_factory(*args, **kwargs):
        # Reuse the identical original object initialized and verified just above;
        # avoids encoding 9888 titles twice. No tool/catalog/model substitution.
        assert Path(args[0]).resolve() == Path(gallery.fpath).resolve()
        assert args[2] == gallery.name and kwargs["columns"] == list(gallery.column_meaning)
        return gallery

    actual_launch = gr.Blocks.launch

    def launch(blocks, **kwargs):
        assert kwargs == {"share": False}
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        value = actual_launch(blocks, share=False, server_name="127.0.0.1", server_port=port,
                              prevent_thread_lock=True, quiet=True, inbrowser=False)
        launched.append({"url": value[1], "share_url": value[2], "blocks": blocks})
        return value

    previous_cwd, previous_argv = os.getcwd(), sys.argv[:]
    os.environ.update(OPENAI_API_KEY="offline-placeholder", OPENAI_API_TYPE="open_ai",
                      OPENAI_API_BASE="https://offline.invalid/v1")
    os.chdir(SOURCE)
    sys.argv = ["app.py", "--engine", "offline-sdk-fixture", "--bot_type", "chat",
                "--plan_first", "1", "--langchain", "0", "--demo_mode", "zero",
                "--enable_reflection", "0", "--enable_shorten", "0", "--reply_style", "concise"]
    try:
        with patch.object(original, "OpenAI", client_factory), patch.object(corups, "BaseGallery", gallery_factory), patch.object(gr.Blocks, "launch", launch):
            namespace = runpy.run_path(str(SOURCE / "app.py"), run_name="__main__")
            assert len(launched) == 1 and launched[0]["share_url"] is None
            url = launched[0]["url"]
            with httpx.Client(trust_env=False, timeout=20) as local_http:
                assert local_http.get(url).status_code == 200
                config = local_http.get(url + "config")
                assert config.status_code == 200 and config.json()["version"] == "3.40.1"
            bot = namespace["bot"]
            state = [namespace["default_chat_value"]]
            turns = []
            for text in ("Recommend three comedy movies released since 1990.", "Give me three comedy movies again."):
                _, state, _ = namespace["user"](text, state)
                output_state, returned_state = bot.run_gr(state)
                assert output_state == returned_state
                state = returned_state
                assert "Here are recommendations:" in state[-1][1]
                turns.append({"input": text, "output": state[-1][1], "tool_trace": namespace["candidate_buffer"].tracker.copy()})
            assert len(bot.memory.memory) == 4
            assert [call["phase"] for call in calls] == ["plan_fixture", "summary_fixture"] * 2
            result = {"http_status": 200, "gradio_version": "3.40.1", "loopback_only": True,
                      "source_entry": "InteRecAgent/app.py", "remote_api_requests": 0,
                      "llm": "real_sdk_with_mock_http_responses", "live_original_app": "NOT VERIFIED",
                      "gallery_reused_after_original_constructor": True, "turns": turns, "sdk_calls": calls}
            (ROOT / "reproduction/logs/a1_20261001/original_app_mock_result.json").write_text(json.dumps(result, indent=2) + "\n")
            return {"http_status": 200, "turns": len(turns), "remote_api_requests": 0, "llm": "mock_transport_only"}
    finally:
        for entry in launched:
            entry["blocks"].close()
        for client in clients:
            client.close()
        os.chdir(previous_cwd)
        sys.argv = previous_argv
