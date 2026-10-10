"""The public quickstart must work without resources, credentials or network."""
import json
import socket

from agenticrec.cli import main


def test_quickstart_runs_recommendation_and_agent_without_external_resources(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("release fixture accessed an external resource")

    from agenticrec.pipeline import FixedRecommendationPipeline
    from agenticrec.runtime import secrets
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(secrets, "read_api_key", forbidden)
    monkeypatch.setattr(FixedRecommendationPipeline, "from_frozen", forbidden)
    assert main(["fixture"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "PASS"
    assert report["is_fixture"] and report["remote_api_requests"] == 0
    assert report["cases"]["known_user"]["item_ids"] == [6, 4]
    assert report["cases"]["anonymous"]["profile_source"] == "cold_start"
    assert report["cases"]["exclusion"]["item_ids"] == [4]
    assert report["cases"]["feedback"]["item_ids"] == [6]
    assert report["cases"]["no_solution"]["status"] == "NO_FEASIBLE_ITEMS"
    assert report["agent"]["A"]["tool_calls"] == 2
    assert report["agent"]["O"]["planner_calls"] == 0
    assert report["agent"]["same_response"] is True
