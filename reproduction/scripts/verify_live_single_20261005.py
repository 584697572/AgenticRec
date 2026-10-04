"""Check saved live single-turn evidence and the spent-attempt gate without HTTP."""
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import live_app_single_turn as app

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "reproduction/logs/live_app_single_turn_20261005"


def main():
    raw = (RUN / "result.json").read_bytes()
    result = json.loads(raw)
    public = json.loads((ROOT / "reproduction/native_runs/20261005_live/original_app_live_summary.json").read_text(encoding="utf-8"))
    assert public["raw_result_sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["status"] == "SUCCESS" and result["live_original_app"] == "VERIFIED"
    assert result["scope"] == "live_original_app" and result["usage_is_fixture"] is False
    assert result["http_attempts"] == result["remote_api_requests"] == len(result["calls"]) == 2
    assert result["worker_exit_code"] == 0 and result["validation"]["passed"]
    assert result["usage"] == {"prompt_tokens": 3655, "completion_tokens": 225, "total_tokens": 3880}
    for call in result["calls"]:
        assert call["http_status"] == 200 and call["error_type"] is None
        assert call["request"]["max_tokens"] <= 512
        assert call["request"]["thinking"] == {"type": "disabled"}
        assert call["response"]["model"] == "deepseek-flash"
        assert call["usage_is_fixture"] is False
    assert len(result["tool_trace"]) == len(result["mapped_items"]) == 3
    assert len({row["id"] for row in result["mapped_items"]}) == 3
    assert all(row["year"] >= 1990 and "Comedy" in row["tags"] for row in result["mapped_items"])
    assert result["shared_checkpoint_id_semantics"] == "NOT VERIFIED"
    assert result["original_evaluation"] == "NOT EVALUATED" and result["actual_cost_cny"] is None
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in RUN.glob("*") if p.is_file()}
    with patch.object(app, "resource_checks", side_effect=AssertionError("repeated resources")), \
         patch.object(app, "BoundedHTTP", side_effect=AssertionError("repeated HTTP")):
        blocked = app.execute(RUN, "unused-fixture-placeholder")
    assert blocked["status"] == "BLOCKED" and blocked["new_http_attempts"] == 0
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in RUN.glob("*") if p.is_file()}
    print("PASS: real app single-turn evidence, 2 HTTP200, 3880 actual tokens, catalog constraints, spent-attempt gate, preserved evidence; multi-turn/evaluation unverified")


if __name__ == "__main__":
    raise SystemExit(main())
