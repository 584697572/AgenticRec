"""Exercise the original OpenAICall with real SDK serialization and a mock HTTP transport."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "reproduction/.runtime/InteRecAgent-compat"))


def main():
    import httpx
    from openai import OpenAI
    from llm4crs.utils import open_ai as original
    captured = []

    def handler(request):
        assert request.url.host == "offline.invalid" and request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        assert body["model"] == "offline-sdk-fixture"
        assert body["messages"][1]["content"] == "test original SDK path"
        assert body["max_tokens"] == 16
        captured.append({"path": request.url.path, "body": body})
        return httpx.Response(200, json={"id": "fixture", "object": "chat.completion", "created": 0,
            "model": "offline-sdk-fixture", "choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": "SDK fixture response"}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}})

    def client_factory(**kwargs):
        return OpenAI(**kwargs, max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(handler)))

    with patch.object(original, "OpenAI", client_factory):
        caller = original.OpenAICall(model="offline-sdk-fixture", api_key="offline-placeholder",
                                    api_base="https://offline.invalid/v1", model_type="chat", retry_limits=1)
        try:
            output = caller.call("test original SDK path", max_tokens=16)
            assert output == "SDK fixture response" and len(captured) == 1
        finally:
            caller.openai_client.close()
    result = {"scope": "original_wrapper_real_sdk_mock_transport", "success": True,
              "remote_api_requests": 0, "live_connection": "NOT VERIFIED",
              "usage_is_fixture_not_measurement": True, "requests": captured, "output": output}
    (ROOT / "reproduction/logs/a1_20261001/sdk_mock_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
