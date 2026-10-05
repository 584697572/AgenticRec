"""Offline native fixed-demo/Critic/reflection probe; no API key or downloads.

Uses real upstream classes and prompts with fixture gallery/buffer/tools and
SDK MockTransport. This is a control-flow check, not recommendation evaluation.
"""
import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "reproduction/.runtime/InteRecAgent-compat"
DEMOS = ROOT / "RecAI/InteRecAgent/demonstration/seed_demos_placeholder.jsonl"


def worker():
    if os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("real_key_must_not_enter_feature_probe")
    captured = io.StringIO()
    results = []
    old_cwd = os.getcwd()
    clients = []
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured), \
         patch.object(socket.socket, "connect", side_effect=RuntimeError("network_forbidden_in_offline_probe")):
        os.chdir(SOURCE)
        sys.path.insert(0, str(SOURCE))
        import httpx
        from openai import OpenAI
        from llm4crs.utils import open_ai
        import llm4crs.agent_plan_first_openai as agent_module
        from llm4crs.critic import Critic
        from llm4crs.prompt import TOOL_NAMES
        from langchain.embeddings import HuggingFaceEmbeddings
        from upstream_bugfixes import corrected_agent_run
        original_run = agent_module.CRSAgentPlanFirstOpenAI.run
        os.environ.update(OPENAI_API_KEY="offline-fixture-placeholder", OPENAI_API_TYPE="open_ai")
        for behavior in ("upstream_pristine", "upstream_behavior_patch"):
            for mode in ("zero", "fixed"):
                calls, planner_history, critic_history, demos_present = [], [], [], []
                external_history = "User: EXTERNAL_HISTORY_FIXTURE. Recommend a movie."
                def fixture(request):
                    body = json.loads(request.content)
                    is_critic = body["max_tokens"] == 128
                    prompt = body["messages"][-1]["content"]
                    calls.append("critic" if is_critic else "planner")
                    if is_critic:
                        critic_history.append(external_history in prompt)
                        content = "No. Follow the original request." if len(critic_history) == 1 else "Yes."
                    else:
                        planner_history.append(external_history in prompt)
                        demos_present.append("Here are some demonstrations" in prompt)
                        content = "Final Answer: FIXTURE_RESPONSE"
                    return httpx.Response(200, json={"id": "offline-fixture", "object": "chat.completion", "created": 0,
                        "model": "deepseek-flash", "choices": [{"index": 0, "finish_reason": "stop",
                        "message": {"role": "assistant", "content": content}}],
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}})
                def factory(**kwargs):
                    client = OpenAI(**kwargs, max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(fixture)))
                    clients.append(client)
                    return client
                tools = {k: SimpleNamespace(name=v.format(item="movie", Item="Movie", ITEM="MOVIE"), desc="FIXTURE_TOOL")
                         for k, v in TOOL_NAMES.items()}
                buffer = SimpleNamespace(clear=lambda: None, clear_tracks=lambda: None, track_info="FIXTURE_TRACKS")
                gallery = SimpleNamespace(info=lambda **kwargs: "FIXTURE_TABLE")
                run = original_run if behavior == "upstream_pristine" else corrected_agent_run(agent_module.__dict__)
                with patch.object(open_ai, "OpenAI", factory), patch.object(agent_module.CRSAgentPlanFirstOpenAI, "run", run):
                    critic = Critic(model="gpt-3.5-turbo", engine="deepseek-flash", buffer=buffer, domain="movie")
                    bot = agent_module.CRSAgentPlanFirstOpenAI("movie", tools, buffer, gallery, engine="deepseek-flash",
                        bot_type="chat", demo_mode=mode, demo_dir_or_file=str(DEMOS), num_demos=3,
                        critic=critic, reflection_limits=1, enable_shorten=False, user_profile_update=-1)
                    bot.init_agent()
                    answer = bot.run({"input": "FIXTURE_QUERY"}, chat_history=external_history)
                    row = {"behavior": behavior, "demo_mode": mode, "mock_sdk_calls": len(calls),
                        "call_order": calls, "planner_external_history_present": planner_history,
                        "critic_external_history_present": critic_history, "demo_prompt_present": demos_present,
                        "memory_entries": len(bot.memory.memory), "reflection_counter_after_run": bot._reflection_cnt,
                        "fixture_response_returned": answer == "FIXTURE_RESPONSE"}
                    assert calls == ["planner", "critic", "planner", "critic"]
                    assert demos_present == [mode == "fixed"] * 2
                    assert row["memory_entries"] == 2 and row["reflection_counter_after_run"] == 0
                    expected = [True, behavior != "upstream_pristine"]
                    assert planner_history == critic_history == expected
                    results.append(row)
        model = HuggingFaceEmbeddings.__fields__["model_name"].default
        hub = Path(os.environ["HF_HOME"]) / "hub" / ("models--" + model.replace("/", "--"))
        st = Path(os.environ["SENTENCE_TRANSFORMERS_HOME"]) / model.replace("/", "_")
        model_present = hub.exists() or st.exists()
        for client in clients:
            client.close()
        os.chdir(old_cwd)
    return {"status": "PASS", "scope": "native_original_feature_control_flow_on_fixtures_only", "cases": results,
        "real_key_in_worker": False, "remote_api_requests": 0, "network_download_bytes": 0,
        "usage": None, "usage_is_fixture": True, "recommendation_quality": "NOT EVALUATED",
        "demo_source": DEMOS.relative_to(ROOT).as_posix(), "demo_sha256": hashlib.sha256(DEMOS.read_bytes()).hexdigest(),
        "demo_total_examples": len(DEMOS.read_text(encoding="utf-8").splitlines()), "selected_fixed_examples": 3,
        "dynamic_default_embedding_model": model, "dynamic_model_cache_present": model_present,
        "dynamic_default_runtime": "NOT VERIFIED; no model download attempted",
        "canonical_author_demo_equivalence": "NOT VERIFIED"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(), indent=2))
        return 0
    output = ROOT / "reproduction/native_runs/20261005_repairs/original_features_summary.json"
    if output.exists():
        print(json.dumps({"status": "BLOCKED", "reason": "existing_feature_probe_evidence_preserved", "remote_api_requests": 0}))
        return 2
    from live_app_single_turn import worker_environment
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker"],
        cwd=ROOT, env=worker_environment(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    if result.returncode:
        sys.stdout.buffer.write(result.stdout)
        sys.stderr.buffer.write(result.stderr)
        return result.returncode
    data = json.loads(result.stdout)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as f:
        f.write(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
