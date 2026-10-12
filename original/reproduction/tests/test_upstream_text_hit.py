"""Pin the original fuzzy metric, including its partial-title limitation."""
import ast
import re
import unittest
from pathlib import Path
from unittest.mock import patch


class OriginalTextHitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from rapidfuzz import fuzz
        cls.fuzz = fuzz
        path = Path(__file__).resolve().parents[2] / "RecAI/InteRecAgent/eval/one_turn_eval.py"
        function = next(node for node in ast.parse(path.read_text(encoding="utf-8")).body
                        if isinstance(node, ast.FunctionDef) and node.name == "hit_judge")
        namespace = {"re": re, "fuzz": fuzz}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
        cls.judge = staticmethod(namespace["hit_judge"])

    def test_normalizes_case_and_punctuation(self):
        self.assertTrue(self.judge("Recommend STAR-WARS! (1997).", "star wars 1997"))

    def test_threshold_is_strictly_greater_than_eighty(self):
        with patch.object(self.fuzz, "partial_ratio", return_value=80):
            self.assertFalse(self.judge("answer", "target"))
        with patch.object(self.fuzz, "partial_ratio", return_value=80.001):
            self.assertTrue(self.judge("answer", "target"))

    def test_partial_title_can_hit_a_different_title(self):
        # Original behavior is intentionally preserved; this is not an ID hit.
        self.assertTrue(self.judge("Recommended: Aliens.", "Alien"))


if __name__ == "__main__":
    unittest.main()
