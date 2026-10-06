"""Offline U1 plan executes actual pinned ToolBox, Buffer and Map classes."""
import pytest

from agenticrec.adapters.model import RouteResult
from agenticrec.adapters.upstream_bridge import run_upstream_plan


def test_original_plan_buffer_and_map_execute_in_legacy_process():
    route = RouteResult([1, 2, 3, 4], [0.2, 0.7, 0.3, 0.1], [2, 3, 1, 4],
                        "cold_start", "anonymous")
    output = run_upstream_plan(route, {i: f"Fixture movie {i}" for i in route.candidate_ids},
                               top_k=2)
    assert output["variant"] == "U1_upstream_rebuilt"
    assert output["plan_source"] == "scripted_offline_fixture_not_llm"
    assert output["upstream_classes"] == ["ToolBox", "CandidateBuffer", "MapTool"]
    assert output["toolbox_success"] is True
    assert output["selected_ids"] == [2, 3, 1, 4]
    assert output["mapped_ids"] == [2, 3]
    assert "Fixture movie 2; Fixture movie 3" in output["answer"]
    assert output["buffer_reset"] is True
    assert output["tracker_tools"] == ["Movie Candidates Ranking Tool", "Mapping Tool"]
    assert output["credential_env_present"] is False
    assert output["remote_api_requests"] == 0


def test_bridge_rejects_unaligned_catalog_before_legacy_process():
    route = RouteResult([1, 2], [0.5, 0.3], [1, 2], "fallback_popularity", None)
    with pytest.raises(ValueError, match="Title map"):
        run_upstream_plan(route, {1: "Only one"})
