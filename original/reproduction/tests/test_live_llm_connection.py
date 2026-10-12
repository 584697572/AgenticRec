"""Offline HTTP tests of the actual upstream wrapper and the authorized request cap."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import live_llm_connection as connection


class ConnectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import httpx
            import openai
        except ImportError:
            raise unittest.SkipTest("Run with the existing legacy interpreter for SDK/HTTP tests")
        cls.httpx = httpx

    def run_fixture(self, handler, directory, profile=None):
        return connection.run_once(
            profile or dict(connection.AUTHORIZED_PROFILE), "offline-test-key",
            Path(directory) / "request.json", Path(directory) / "result.json",
            transport=self.httpx.MockTransport(handler), evidence_scope="offline_fixture")

    def test_success_original_sdk_payload_and_repeat_denied(self):
        requests = []

        def handler(request):
            body = json.loads(request.content)
            requests.append(body)
            self.assertEqual(str(request.url), "https://api.deepseek.com/chat/completions")
            self.assertEqual(body["max_tokens"], 128)
            self.assertEqual(body["thinking"], {"type": "disabled"})
            self.assertEqual(body["model"], "deepseek-flash")
            self.assertEqual(body["messages"], connection.MESSAGES)
            return self.httpx.Response(200, json={
                "id": "offline", "object": "chat.completion", "created": 0,
                "model": "deepseek-flash", "choices": [{"index": 0,
                    "finish_reason": "stop", "message": {"role": "assistant", "content": "OK"}}],
                "usage": {"prompt_tokens": 8, "completion_tokens": 1, "total_tokens": 9}})

        with tempfile.TemporaryDirectory() as folder:
            first = self.run_fixture(handler, folder)
            self.assertEqual(first["status"], "SUCCESS")
            self.assertEqual(first["http_attempts"], 1)
            self.assertEqual(first["remote_api_requests"], 0)
            self.assertEqual(first["live_connection"], "NOT VERIFIED")
            self.assertEqual(first["usage"]["completion_tokens"], 1)
            self.assertIsNone(first["actual_cost_cny"])
            saved = (Path(folder) / "result.json").read_bytes()
            second = self.run_fixture(handler, folder)
            self.assertEqual(second["status"], "BLOCKED")
            self.assertEqual(second["reason"], "authorized_attempt_already_reserved")
            self.assertEqual(second["http_attempts"], 0)
            self.assertEqual(len(requests), 1)
            self.assertEqual(saved, (Path(folder) / "result.json").read_bytes())
            self.assertNotIn("offline-test-key", saved.decode())

    def test_server_error_no_retry_and_no_secret_in_report(self):
        requests = []

        def handler(request):
            requests.append(request)
            return self.httpx.Response(500, json={"error": {"message": "offline-test-key", "type": "server_error"}})

        with tempfile.TemporaryDirectory() as folder:
            result = self.run_fixture(handler, folder)
            self.assertEqual(result["status"], "FAILED")
            self.assertEqual(result["http_attempts"], 1)
            self.assertEqual(result["http_status"], 500)
            self.assertEqual(len(requests), 1)
            self.assertIsNone(result["usage"])
            self.assertNotIn("offline-test-key", json.dumps(result))
            self.assertEqual(self.run_fixture(handler, folder)["status"], "BLOCKED")
            self.assertEqual(len(requests), 1)

    def test_timeout_consumes_attempt_without_retry(self):
        requests = []

        def handler(request):
            requests.append(request)
            raise self.httpx.ReadTimeout("offline-test-key", request=request)

        with tempfile.TemporaryDirectory() as folder:
            result = self.run_fixture(handler, folder)
            self.assertEqual(result["status"], "FAILED")
            self.assertEqual(result["http_attempts"], 1)
            self.assertEqual(len(requests), 1)
            self.assertNotIn("offline-test-key", json.dumps(result))
            self.assertIsNone(result["actual_cost_cny"])
            self.assertEqual(self.run_fixture(handler, folder)["status"], "BLOCKED")

    def test_redirect_is_not_followed(self):
        requests = []

        def handler(request):
            requests.append(request)
            return self.httpx.Response(307, headers={"Location": "https://other.invalid/"})

        with tempfile.TemporaryDirectory() as folder:
            result = self.run_fixture(handler, folder)
            self.assertEqual(result["status"], "FAILED")
            self.assertEqual(result["http_status"], 307)
            self.assertEqual(len(requests), 1)

    def test_invalid_authorization_or_missing_key_has_no_attempt(self):
        def handler(request):
            self.fail("Unauthorized request reached HTTP transport")

        for field, value in (("api_request_cap", 2), ("max_output_tokens", 129),
                             ("money_budget", 2), ("allow_paid_api", False),
                             ("api_base", "https://other.invalid")):
            profile = dict(connection.AUTHORIZED_PROFILE)
            profile[field] = value
            with tempfile.TemporaryDirectory() as folder:
                result = self.run_fixture(handler, folder, profile)
                self.assertEqual(result["status"], "BLOCKED")
                self.assertEqual(result["http_attempts"], 0)
                self.assertFalse((Path(folder) / "request.json").exists())
        with tempfile.TemporaryDirectory() as folder:
            result = connection.run_once(dict(connection.AUTHORIZED_PROFILE), "",
                Path(folder) / "request.json", Path(folder) / "result.json",
                self.httpx.MockTransport(handler), evidence_scope="offline_fixture")
            self.assertEqual(result["status"], "BLOCKED")
            self.assertFalse((Path(folder) / "request.json").exists())


if __name__ == "__main__":
    unittest.main()
