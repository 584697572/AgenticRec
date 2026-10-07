from agenticrec.agent.executor import PlanExecutor, PlanStep
from agenticrec.evaluation.system_factory import build_recommendation_tool
from agenticrec.pipeline import FixedRecommendationPipeline


def test_real_frozen_pipeline_runs_behind_agent_recommendation_tool():
    pipeline = FixedRecommendationPipeline.from_frozen("lightgcn", 42)
    tool = build_recommendation_tool(pipeline)
    request = {
        "schema_version": 1,
        "user_id": None,
        "history_authorized": False,
        "liked_item_ids": [],
        "disliked_item_ids": [],
        "seen_item_ids": [],
        "constraints": {
            "include_genres": ["Action"],
            "exclude_genres": [],
            "year_min": None,
            "year_max": None,
            "excluded_item_ids": [],
            "exclude_seen": True,
            "k": 3,
            "required_fields": [],
        },
    }

    report = PlanExecutor([tool]).execute([
        PlanStep("recommend-1", "recommend", {"request": request})
    ])

    assert report.ok and len(report.traces) == 1
    result = report.traces[0].result
    assert result.ok and result.provenance == "ours:fixed_pipeline"
    assert result.data["status"] == "OK"
    assert len(result.data["recommendations"]) == 3
    assert all(
        "Action" in recommendation["genres"]
        for recommendation in result.data["recommendations"]
    )
