"""Self-created fixtures for opt-in upstream correctness patches."""
import logging
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from audit_redial_preprocess import original_functions
from redial_compat import adapt_release_dates
from upstream_bugfixes import corrected_agent_run, corrected_movie_map, require_reachable_targets, target_catalog_contract
import pandas as pd


def flow_fixture(run, history="EXTERNAL_HISTORY_FIXTURE", reject_count=1, limit=2):
    seen = []
    memory = SimpleNamespace(get=lambda: "LOCAL_MEMORY_FIXTURE", append=lambda *args: None)
    critic_calls = []
    def critic(request, answer, context, tracks):
        critic_calls.append(context)
        return len(critic_calls) <= reject_count, "Review the request."
    bot = SimpleNamespace(selector=None, toolbox=SimpleNamespace(failed_times=0, name="FixtureExecutor", desc="fixture"),
        item_corups=SimpleNamespace(info=lambda **kw: "FIXTURE_TABLE"),
        candidate_buffer=SimpleNamespace(clear_tracks=lambda: None, clear=lambda: None, track_info=""),
        memory=memory, _record_planning=False, critic=critic, _reflection_cnt=0,
        reflection_limits=limit, _k_turn=0, user_profile_update=-1)
    bot.plan_and_exe = lambda prompt, mapping: seen.append(mapping.copy()) or "FIXTURE_RESPONSE"
    bot.prompt = "fixture"
    bot.run = run.__get__(bot)
    response = bot.run({"input": "FIXTURE_QUERY"}, chat_history=history)
    return seen, critic_calls, bot, response


class UpstreamBugfixTests(unittest.TestCase):
    def setUp(self):
        self.functions = original_functions()
        self.frame = pd.DataFrame({"id": [1, 2], "title": ["fixture", "fixture"],
                                   "release_date": [1957, 2007]})

    def test_original_signed_year_bug_is_reproduced(self):
        mapped = self.functions["movie_map"]({"title": "fixture", "date": 2007}, adapt_release_dates(self.frame))[0]
        self.assertEqual(mapped[("fixture", 2007)], 1)

    def test_patch_selects_exact_remake_year(self):
        mapped = corrected_movie_map()({"title": "fixture", "date": 2007}, adapt_release_dates(self.frame))[0]
        self.assertEqual(mapped[("fixture", 2007)], 2)

    def test_patch_selects_nearest_absolute_year(self):
        frame = self.frame.copy()
        frame["release_date"] = [1900, 2005]
        mapped = corrected_movie_map()({"title": "fixture", "date": 2007}, adapt_release_dates(frame))[0]
        self.assertEqual(mapped[("fixture", 2007)], 2)

    def test_patch_preserves_non_duplicate_mapping_and_original_input(self):
        frame = pd.DataFrame({"id": [1], "title": ["fixture"], "release_date": [1957]})
        adapted = adapt_release_dates(frame)
        for title, year in [("fixture", 1957), ("fixture", None), ("unknown", 2007)]:
            q = {"title": title, "date": year}
            self.assertEqual(corrected_movie_map()(q, adapted), self.functions["movie_map"](q, adapted))
        self.assertEqual(frame["release_date"].tolist(), [1957])

    def test_reflection_keeps_external_history_every_attempt(self):
        run = corrected_agent_run({"Dict": Dict, "logger": logging.getLogger("fixture")})
        seen, critics, bot, answer = flow_fixture(run)
        self.assertEqual([m["history"] for m in seen], ["EXTERNAL_HISTORY_FIXTURE"] * 2)
        self.assertEqual(critics, ["EXTERNAL_HISTORY_FIXTURE"] * 2)
        self.assertEqual(bot._reflection_cnt, 0)
        self.assertEqual(answer, "FIXTURE_RESPONSE")

    def test_reflection_keeps_limit_and_memory_fallback(self):
        run = corrected_agent_run({"Dict": Dict, "logger": logging.getLogger("fixture")})
        seen, _, bot, _ = flow_fixture(run, history=None, reject_count=10, limit=2)
        self.assertEqual(len(seen), 3)
        self.assertTrue(all(m["history"] == "LOCAL_MEMORY_FIXTURE" for m in seen))
        self.assertEqual(bot._reflection_cnt, 0)

    def test_contract_requires_exact_year_and_unique_title(self):
        contract = target_catalog_contract("Fixture (2007)", self.frame, self.functions["separate_movie_and_year"])
        self.assertEqual(contract["ids"], [2])
        for target, reason in [("Fixture", "ambiguous_title_year"), ("Fixture (1999)", "year_absent"),
                               ("Fixtures (2007)", "title_absent")]:
            self.assertEqual(target_catalog_contract(target, self.frame, self.functions["separate_movie_and_year"])["reason"], reason)

    def test_contract_blocks_whole_input_and_never_filters(self):
        rows = [{"context": "fixture", "target": "Fixture (2007)"},
                {"context": "fixture", "target": "Absent (2007)"}]
        with self.assertRaisesRegex(ValueError, "no samples removed"):
            require_reachable_targets(rows, self.frame, self.functions["separate_movie_and_year"])
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(require_reachable_targets(rows[:1], self.frame, self.functions["separate_movie_and_year"])), 1)

    def test_original_selection_can_choose_an_unreachable_final_target(self):
        conv = {"initiatorWorkerId": 1, "respondentWorkerId": 2,
            "movieMentions": {"1": "Fixture (1957)", "2": "Fixture (2007)", "3": "Absent (2007)"},
            "initiatorQuestions": {str(i): {"liked": 1, "suggested": 1} for i in (1, 2, 3)},
            "messages": [{"senderWorkerId": 1, "text": "Fixture request"},
                         {"senderWorkerId": 2, "text": "@1"}, {"senderWorkerId": 2, "text": "@2"},
                         {"senderWorkerId": 2, "text": "@3"}]}
        proposed = self.functions["process_conv"](conv)
        self.assertEqual(proposed["target"], "Absent (2007)")
        with self.assertRaisesRegex(ValueError, "no samples removed"):
            require_reachable_targets([proposed], self.frame, self.functions["separate_movie_and_year"])

    def test_contract_refuses_empty_input_and_padding(self):
        split = self.functions["separate_movie_and_year"]
        with self.assertRaises(ValueError):
            require_reachable_targets([], self.frame, split)
        padding = pd.DataFrame({"id": [0], "title": ["fixture"], "release_date": [2007]})
        self.assertEqual(target_catalog_contract("Fixture (2007)", padding, split)["reason"], "invalid_or_padding_id")


if __name__ == "__main__":
    unittest.main()
