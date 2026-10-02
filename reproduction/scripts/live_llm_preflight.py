"""Check live-LLM prerequisites without importing the SDK or making network calls."""
import argparse
import json
import math
import os
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
FIELDS = {"provider", "api_type", "api_base", "api_version", "model_id", "allow_paid_api",
          "api_request_cap", "max_output_tokens", "money_budget", "budget_unit"}


def check(profile, key_present):
    problems = []
    if not isinstance(profile, dict) or set(profile) != FIELDS:
        return {"status": "BLOCKED", "api_requests": 0,
                "missing_or_invalid": ["profile schema: use the exact example keys; do not put an API key in JSON"]}
    for name in ("provider", "model_id", "budget_unit"):
        if not isinstance(profile[name], str) or not profile[name].strip():
            problems.append(name)
    if profile["api_type"] not in ("open_ai", "azure"):
        problems.append("api_type")
    if profile["api_type"] == "azure" and not profile["api_version"]:
        problems.append("api_version")
    base = profile["api_base"]
    try:
        url = urlsplit(base) if isinstance(base, str) else None
        if not url or url.scheme != "https" or not url.hostname or url.username or url.password or url.query or url.fragment:
            problems.append("api_base: HTTPS endpoint without embedded credentials/query/fragment")
    except ValueError:
        problems.append("api_base")
    if profile["allow_paid_api"] is not True:
        problems.append("allow_paid_api: explicit authorization required")
    for name in ("api_request_cap", "max_output_tokens"):
        if type(profile[name]) is not int or profile[name] <= 0:
            problems.append(name)
    budget = profile["money_budget"]
    if type(budget) not in (int, float) or not math.isfinite(budget) or budget <= 0:
        problems.append("money_budget")
    if not key_present:
        problems.append("OPENAI_API_KEY in this terminal")
    return {"status": "CONFIG_READY" if not problems else "BLOCKED", "api_requests": 0,
            "key_present": bool(key_present), "missing_or_invalid": problems,
            "live_connection": "NOT VERIFIED", "paid_calls_authorized_by_this_check": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=ROOT / "reproduction/.runtime/live_llm.json")
    args = parser.parse_args()
    path = args.profile.resolve()
    if not path.is_relative_to(ROOT):
        parser.error("profile must be inside the workspace")
    if not path.is_file():
        print(json.dumps({"status": "BLOCKED", "api_requests": 0,
                          "missing_or_invalid": ["local profile: copy reproduction/live_llm.example.json first"],
                          "live_connection": "NOT VERIFIED"}))
        return 2
    try:
        profile = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        print(json.dumps({"status": "BLOCKED", "api_requests": 0,
                          "missing_or_invalid": ["profile must be readable valid JSON; its content is not logged"]}))
        return 2
    result = check(profile, bool(os.environ.get("OPENAI_API_KEY", "").strip()))
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "CONFIG_READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
