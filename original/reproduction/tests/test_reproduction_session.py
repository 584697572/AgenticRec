"""Real-reproduction budget, hidden-target, and multi-turn acceptance regressions."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from reproduction_session_http import AUTHORIZED_PROFILE, SessionHTTP, validate_turn


class SessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import httpx
        cls.httpx = httpx

    def body(self, text="visible fixture context"):
        return {"model": "deepseek-flash", "max_tokens": 1024,
                "thinking": {"type": "disabled"}, "temperature": 0,
                "messages": [{"role": "system", "content": "fixture"},
                             {"role": "user", "content": text}]}

    def reply(self, request):
        return self.httpx.Response(200, json={"model": "deepseek-flash",
            "choices": [{"finish_reason": "stop", "message": {"content": "fixture-secret"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}})

    def broker(self, directory, **overrides):
        profile = dict(AUTHORIZED_PROFILE, **overrides)
        return SessionHTTP(directory, "fixture-secret", ["visible fixture context"],
                           profile=profile, transport=self.httpx.MockTransport(self.reply))

    def test_request_slots_persist_across_instances(self):
        with tempfile.TemporaryDirectory() as directory:
            for _ in range(2):
                with self.broker(directory, api_request_cap=2) as broker:
                    broker.set_context("visible fixture context")
                    self.assertEqual(broker.send(self.body())["status_code"], 200)
            with self.broker(directory, api_request_cap=2) as broker:
                broker.set_context("visible fixture context")
                with self.assertRaisesRegex(RuntimeError, "request_budget_exhausted"):
                    broker.send(self.body())
            self.assertEqual(len(list(Path(directory).glob("request_*.json"))), 2)

    def test_crashed_reserved_request_blocks_further_spending(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.broker(directory) as broker:
                broker.set_context("visible fixture context")
                broker.send(self.body())
            (Path(directory) / "response_0001.json").unlink()
            with self.broker(directory) as broker:
                broker.set_context("visible fixture context")
                with self.assertRaisesRegex(RuntimeError, "unresolved_prior_attempt"):
                    broker.send(self.body())
            self.assertEqual(len(list(Path(directory).glob("request_*.json"))), 1)

    def test_scope_and_token_limits_reject_without_spending(self):
        for field, value in (("model", "other"), ("max_tokens", 1025),
                             ("max_tokens", True), ("tools", []), ("stream", True)):
            with tempfile.TemporaryDirectory() as directory:
                with self.broker(directory) as broker:
                    broker.set_context("visible fixture context")
                    body = self.body()
                    body[field] = value
                    with self.assertRaises(ValueError):
                        broker.send(body)
                self.assertFalse(list(Path(directory).glob("request_*.json")))

    def test_context_cannot_be_switched_to_hidden_target(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.broker(directory) as broker:
                with self.assertRaises(ValueError):
                    broker.set_context("hidden evaluation target")
                broker.set_context("visible fixture context")
                with self.assertRaises(ValueError):
                    broker.send(self.body("hidden evaluation target"))
            self.assertFalse(list(Path(directory).glob("request_*.json")))

    def test_money_guard_refuses_before_http(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.broker(directory, money_budget=0.000001) as broker:
                broker.set_context("visible fixture context")
                with self.assertRaisesRegex(RuntimeError, "money_budget_exhausted"):
                    broker.send(self.body())
            self.assertFalse(list(Path(directory).glob("request_*.json")))

    def test_401_and_unknown_usage_stop_later_requests(self):
        for status, body in ((401, {"error": "fixture-secret"}),
                             (200, {"model": "deepseek-flash", "choices": []})):
            with tempfile.TemporaryDirectory() as directory:
                broker = self.broker(directory)
                broker.client.close()
                broker.client = self.httpx.Client(transport=self.httpx.MockTransport(
                    lambda request: self.httpx.Response(status, json=body)))
                with broker:
                    broker.set_context("visible fixture context")
                    broker.send(self.body())
                    with self.assertRaisesRegex(RuntimeError, "prior_provider_failure"):
                        broker.send(self.body())
                self.assertEqual(len(list(Path(directory).glob("request_*.json"))), 1)

    def test_redirect_not_followed_and_secret_redacted(self):
        seen = []
        def handler(request):
            seen.append(request)
            return self.httpx.Response(307, headers={"location": "https://offline.invalid"},
                                       text="fixture-secret")
        with tempfile.TemporaryDirectory() as directory:
            with SessionHTTP(directory, "fixture-secret", ["visible fixture context"],
                             transport=self.httpx.MockTransport(handler)) as broker:
                broker.set_context("visible fixture context")
                packet = broker.send(self.body())
                self.assertNotIn("fixture-secret", packet["content"])
            self.assertEqual(len(seen), 1)
            for path in Path(directory).glob("*.json"):
                self.assertNotIn("fixture-secret", path.read_text())

    def test_turn_validation_requires_current_year_range_and_tools(self):
        rows = [{"id": n, "title": "Movie%d" % n, "year": 1970, "tags": "Comedy"}
                for n in (1, 2, 3)]
        tools = ["filter", "rank", "map"]
        self.assertTrue(validate_turn(rows, "Movie1 Movie2 Movie3", tools, tools, 1)["passed"])
        self.assertFalse(validate_turn(rows, "Movie1 Movie2 Movie3", tools, tools, 0)["passed"])
        self.assertFalse(validate_turn(rows, "Movie1 Movie2 Movie3", [], tools, 1)["passed"])
        self.assertFalse(validate_turn(rows[:1] * 3, "Movie1", tools, tools, 1)["passed"])


if __name__ == "__main__":
    unittest.main()
