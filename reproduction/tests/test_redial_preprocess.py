"""Exercise original notebook mapping on self-created duplicate-title fixtures."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))


class RedialPreprocessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import pandas as pd
            from audit_redial_preprocess import original_functions
            from redial_compat import adapt_release_dates
        except ImportError:
            raise unittest.SkipTest("Use the existing legacy environment")
        cls.pd = pd
        cls.mapping = staticmethod(original_functions()["movie_map"])
        cls.adapt = staticmethod(adapt_release_dates)

    def test_integer_years_match_original_datetime_contract_without_mutation(self):
        original = self.pd.DataFrame({"id": [1, 2], "title": ["fixture", "fixture"],
                                      "release_date": [1957, 2007]})
        reference = original.copy(deep=True)
        reference["release_date"] = self.pd.to_datetime(["1957-01-01", "2007-01-01"])
        adapted = self.adapt(original)
        for year in (1957, 2007):
            query = {"title": "Fixture", "date": year}
            self.assertEqual(self.mapping(query, adapted), self.mapping(query, reference))
        self.assertEqual(original["release_date"].tolist(), [1957, 2007])
        # Preserve upstream's signed-difference behavior, even for the later remake.
        self.assertEqual(self.mapping({"title": "Fixture", "date": 2007}, adapted)[0][("Fixture", 2007)], 1)


if __name__ == "__main__":
    unittest.main()
