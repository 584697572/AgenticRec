"""Read-only config checks must fail closed without exposing secrets or calling APIs."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import live_llm_preflight as preflight


class LivePreflightTests(unittest.TestCase):
    def profile(self):
        return {"provider": "fixture", "api_type": "open_ai", "api_base": "https://offline.invalid/v1",
                "api_version": None, "model_id": "fixture-model", "allow_paid_api": True,
                "api_request_cap": 1, "max_output_tokens": 64, "money_budget": 1, "budget_unit": "fixture-unit"}

    def test_default_profile_never_authorizes_calls(self):
        path = Path(__file__).parents[1] / "live_llm.example.json"
        result = preflight.check(json.loads(path.read_text()), False)
        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(result["api_requests"], 0)
        self.assertFalse(result["paid_calls_authorized_by_this_check"])

    def test_ready_is_configuration_only_and_has_no_secret(self):
        result = preflight.check(self.profile(), True)
        self.assertEqual(result["status"], "CONFIG_READY")
        self.assertEqual(result["live_connection"], "NOT VERIFIED")
        self.assertFalse(result["paid_calls_authorized_by_this_check"])
        self.assertNotIn("api_base", json.dumps(result))

    def test_rejects_zero_budget_boolean_integer_and_embedded_secret(self):
        for name, value in (("api_request_cap", 0), ("api_request_cap", True), ("money_budget", float("nan")),
                            ("max_output_tokens", False), ("allow_paid_api", False),
                            ("api_base", "https://user:self-authored-secret@offline.invalid/v1")):
            profile = self.profile()
            profile[name] = value
            result = preflight.check(profile, True)
            self.assertEqual(result["status"], "BLOCKED")
            self.assertNotIn("self-authored-secret", json.dumps(result))
        profile = self.profile()
        profile["api_key"] = "self-authored-secret"
        self.assertNotIn("self-authored-secret", json.dumps(preflight.check(profile, True)))
        self.assertEqual(preflight.check(profile, True)["status"], "BLOCKED")


if __name__ == "__main__":
    unittest.main()
