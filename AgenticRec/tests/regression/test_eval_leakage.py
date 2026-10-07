import json
from pathlib import Path

import pytest

from agenticrec.evaluation.episodes import (
    PrivateEpisode,
    PublicEpisode,
    load_private_episodes,
    load_public_episodes,
    record_human_review,
    validate_benchmark,
    verify_eval_manifest,
    write_frozen_benchmark,
)
from agenticrec.ranking.constraints import Catalog, Constraints, Item


def catalog():
    return Catalog([
        Item(1, "Hidden Comedy", frozenset({"Comedy"}), 1995),
        Item(2, "Visible Seed", frozenset({"Drama"}), 1996),
        Item(3, "Hidden Drama", frozenset({"Drama"}), 2001),
        Item(4, "Other", frozenset({"Action"}), 2000),
    ])


def pair(split, number, group, target, *, multi_turn=False):
    episode_id = f"{split}-{number:03d}"
    visible = PublicEpisode(
        episode_id=episode_id,
        split=split,
        layer="text",
        group_id=group,
        template_group_id=f"{split}:template:{number}:text",
        scenario_family="exclusion_feedback" if multi_turn else "personalized",
        multi_turn=multi_turn,
        initial_input={
            "mode": "text",
            "template_id": f"{split}-template-{number}",
            "message": "Please recommend two suitable movies.",
        },
        max_turns=3 if multi_turn else 1,
    )
    hidden = PrivateEpisode(
        episode_id=episode_id,
        feasible=True,
        acceptable_item_ids=(target,),
        hard_constraints=Constraints(k=2),
        requested_k=2,
        required_count=1,
        expected_statuses=("OK",),
        expected_patch={"disliked_item_ids": [2]} if multi_turn else None,
        evaluator_events=({"turn": 2, "patch": {"disliked_item_ids": [2]}},)
        if multi_turn else (),
    )
    return visible, hidden


def fixture_benchmark():
    dev_public, dev_private = pair("development", 1, "user:11", 1)
    test_public, test_private = pair("test", 1, "user:22", 3, multi_turn=True)
    return [dev_public], [dev_private], [test_public], [test_private]


def test_public_and_private_serialization_are_separate_and_targets_do_not_leak(tmp_path):
    benchmark = fixture_benchmark()
    manifest = write_frozen_benchmark(
        tmp_path / "data", tmp_path / "eval_manifest.json", *benchmark,
        catalog=catalog(), human_review_ids=("development-001", "test-001"),
    )

    public_text = (tmp_path / "data" / "test.public.jsonl").read_text(encoding="utf-8")
    private_text = (tmp_path / "data" / "test.private.jsonl").read_text(encoding="utf-8")
    assert "acceptable_item_ids" not in public_text
    assert "evaluator_events" not in public_text
    assert "Hidden Drama" not in public_text
    assert '"acceptable_item_ids":[3]' in private_text
    assert manifest["llm_simulator_enabled"] is False
    assert manifest["human_review_status"] == "PENDING"
    assert manifest["splits"]["test"]["episodes"] == 1

    loaded_public = load_public_episodes(tmp_path / "data" / "test.public.jsonl")
    loaded_private = load_private_episodes(tmp_path / "data" / "test.private.jsonl")
    assert loaded_public == benchmark[2]
    assert loaded_private == benchmark[3]
    assert not hasattr(loaded_public[0], "acceptable_item_ids")
    assert verify_eval_manifest(tmp_path / "eval_manifest.json") == manifest
    reviewed = record_human_review(
        tmp_path / "eval_manifest.json",
        ("development-001", "test-001"),
        {"public_private_alignment": True, "expected_outcome": True},
    )
    assert reviewed["human_review_status"] == "VERIFIED"
    assert verify_eval_manifest(tmp_path / "eval_manifest.json") == reviewed


def test_user_groups_and_episode_ids_cannot_cross_splits():
    dev_public, dev_private, test_public, test_private = fixture_benchmark()
    validate_benchmark(dev_public, dev_private, test_public, test_private, catalog())

    crossed = PublicEpisode(
        episode_id=test_public[0].episode_id,
        split="test",
        layer=test_public[0].layer,
        group_id=dev_public[0].group_id,
        template_group_id=test_public[0].template_group_id,
        scenario_family=test_public[0].scenario_family,
        multi_turn=test_public[0].multi_turn,
        initial_input=test_public[0].initial_input,
        max_turns=test_public[0].max_turns,
    )
    with pytest.raises(ValueError, match="group leakage"):
        validate_benchmark(
            dev_public, dev_private, [crossed], test_private, catalog()
        )

    crossed_template = PublicEpisode(
        episode_id=test_public[0].episode_id,
        split="test",
        layer=test_public[0].layer,
        group_id=test_public[0].group_id,
        template_group_id=dev_public[0].template_group_id,
        scenario_family=test_public[0].scenario_family,
        multi_turn=test_public[0].multi_turn,
        initial_input=test_public[0].initial_input,
        max_turns=test_public[0].max_turns,
    )
    with pytest.raises(ValueError, match="template group leakage"):
        validate_benchmark(
            dev_public, dev_private, [crossed_template], test_private, catalog()
        )


def test_hidden_target_id_or_title_in_visible_input_is_rejected():
    dev_public, dev_private, test_public, test_private = fixture_benchmark()
    leaked_id = PublicEpisode(
        episode_id=test_public[0].episode_id,
        split="test",
        layer="structured",
        group_id=test_public[0].group_id,
        template_group_id=test_public[0].template_group_id,
        scenario_family=test_public[0].scenario_family,
        multi_turn=False,
        initial_input={"liked_item_ids": [3], "constraints": {"k": 2}},
        max_turns=1,
    )
    with pytest.raises(ValueError, match="target item ID"):
        validate_benchmark(dev_public, dev_private, [leaked_id], test_private, catalog())

    leaked_title = PublicEpisode(
        episode_id=test_public[0].episode_id,
        split="test",
        layer="text",
        group_id=test_public[0].group_id,
        template_group_id=test_public[0].template_group_id,
        scenario_family=test_public[0].scenario_family,
        multi_turn=False,
        initial_input={"mode": "text", "message": "Recommend Hidden Drama"},
        max_turns=1,
    )
    with pytest.raises(ValueError, match="target title"):
        validate_benchmark(dev_public, dev_private, [leaked_title], test_private, catalog())


def test_manifest_detects_post_freeze_mutation(tmp_path):
    benchmark = fixture_benchmark()
    manifest_path = tmp_path / "eval_manifest.json"
    write_frozen_benchmark(
        tmp_path / "data", manifest_path, *benchmark,
        catalog=catalog(), human_review_ids=("development-001",),
    )
    path = tmp_path / "data" / "test.public.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        verify_eval_manifest(manifest_path)


def test_frozen_local_benchmark_has_no_leakage_when_available():
    root = Path(__file__).resolve().parents[3]
    manifest_path = root / "artifacts/eval_manifest.json"
    if not manifest_path.exists():
        pytest.skip("T18 local frozen benchmark is not present")
    manifest = verify_eval_manifest(manifest_path)
    base = root / manifest["dataset_dir"]
    dev_public = load_public_episodes(base / manifest["files"]["development_public"]["name"])
    dev_private = load_private_episodes(base / manifest["files"]["development_private"]["name"])
    test_public = load_public_episodes(base / manifest["files"]["test_public"]["name"])
    test_private = load_private_episodes(base / manifest["files"]["test_private"]["name"])
    validate_benchmark(dev_public, dev_private, test_public, test_private,
                       Catalog.from_frozen(root))
    assert len(dev_public) == len(test_public) == 150
    assert sum(item.multi_turn for item in dev_public) >= 30
    assert sum(item.multi_turn for item in test_public) >= 30
    assert set(manifest["human_review_ids"])
