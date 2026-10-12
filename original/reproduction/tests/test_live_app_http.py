"""Budget and result acceptance tests; all provider HTTP is mocked."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from live_app_http import BoundedHTTP, QUERY, validate_recommendations
from live_app_single_turn import worker_environment


class AppBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import httpx
        except ImportError:
            raise unittest.SkipTest("Use existing legacy environment")
        cls.httpx = httpx

    def body(self):
        return {"model": "deepseek-flash", "max_tokens": 512,
                "thinking": {"type": "disabled"}, "temperature": 0,
                "messages": [{"role": "system", "content": "fixture"}, {"role": "user", "content": QUERY}]}

    def test_two_attempts_across_instances_even_on_500(self):
        seen = []
        def handler(request):
            seen.append(request)
            return self.httpx.Response(500, json={"error": {"message": "fixture-secret"}})
        with tempfile.TemporaryDirectory() as directory:
            for number in (1, 2):
                client = BoundedHTTP(directory, "fixture-secret", self.httpx.MockTransport(handler))
                result = client.send(self.body())
                client.close()
                self.assertEqual(result["status_code"], 500)
                self.assertNotIn("fixture-secret", result["content"])
                self.assertTrue((Path(directory) / ("request_%d.json" % number)).exists())
            client = BoundedHTTP(directory, "fixture-secret", self.httpx.MockTransport(handler))
            with self.assertRaisesRegex(RuntimeError, "budget_exhausted"):
                client.send(self.body())
            client.close()
            self.assertEqual(len(seen), 2)
            for path in Path(directory).glob("*.json"):
                self.assertNotIn("fixture-secret", path.read_text())

    def test_timeout_consumes_slot_and_redirect_not_followed(self):
        seen = []
        def handler(request):
            seen.append(request)
            if len(seen) == 1:
                raise self.httpx.ReadTimeout("fixture-secret", request=request)
            return self.httpx.Response(307, headers={"location": "https://other.invalid"})
        with tempfile.TemporaryDirectory() as directory:
            client = BoundedHTTP(directory, "fixture-secret", self.httpx.MockTransport(handler))
            self.assertIn("transport_error", client.send(self.body()))
            self.assertEqual(client.send(self.body())["status_code"], 307)
            self.assertEqual(len(seen), 2)
            client.close()

    def test_invalid_payload_spends_no_attempt(self):
        def handler(request):
            self.fail("Bad payload reached transport")
        for name, value in (("max_tokens", 513), ("model", "other"), ("stream", True),
                            ("thinking", {"type": "enabled"}), ("tools", [])):
            with tempfile.TemporaryDirectory() as directory:
                client = BoundedHTTP(directory, "fixture", self.httpx.MockTransport(handler))
                body = self.body()
                body[name] = value
                with self.assertRaises(ValueError):
                    client.send(body)
                self.assertEqual(list(Path(directory).iterdir()), [])
                client.close()

    def test_original_stop_sequences_are_forwarded_unchanged(self):
        stop = ["Obsersation", "observation", "Observation:", "observation:"]
        def handler(request):
            self.assertEqual(json.loads(request.content)["stop"], stop)
            return self.httpx.Response(200, json={"ok": True})
        with tempfile.TemporaryDirectory() as directory:
            client = BoundedHTTP(directory, "fixture", self.httpx.MockTransport(handler))
            body = self.body()
            body["stop"] = stop
            self.assertEqual(client.send(body)["status_code"], 200)
            client.close()

    def test_key_not_in_resource_worker_environment(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "fixture-secret", "AZURE_OPENAI_API_KEY": "fixture-secret"}):
            env = worker_environment()
            self.assertNotIn("OPENAI_API_KEY", env)
            self.assertNotIn("AZURE_OPENAI_API_KEY", env)
            self.assertNotIn("fixture-secret", json.dumps(env))
            self.assertEqual(env["HF_HUB_OFFLINE"], "1")

    def test_validation_rejects_empty_wrong_year_duplicates_or_llm_only(self):
        rows = [{"id": n, "title": "Movie%d" % n, "tags": "Comedy", "year": 1995} for n in (1, 2, 3)]
        answer = "Movie1; Movie2; Movie3"
        tools = ["filter", "rank", "map"]
        self.assertTrue(validate_recommendations(rows, answer, tools, tools)["passed"])
        self.assertFalse(validate_recommendations([], answer, tools, tools)["passed"])
        self.assertFalse(validate_recommendations(rows, answer, [], tools)["passed"])
        self.assertFalse(validate_recommendations(rows[:1]*3, answer, tools, tools)["passed"])
        rows[0]["year"] = 1980
        self.assertFalse(validate_recommendations(rows, answer, tools, tools)["passed"])


if __name__ == "__main__":
    unittest.main()
