"""Self-created metadata fixtures for title/year identity and safe aliases."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from catalog_identity import MetadataIndex, parse_movies_dat, catalog_crosswalk, resolve_catalog_target


class CatalogIdentityTests(unittest.TestCase):
    def setUp(self):
        self.rows = parse_movies_dat((
            "7::Fixture, The (1995)::Comedy|Drama\n"
            "8::Fixture, The (2005)::Comedy\n"
            "9::Primary (a.k.a. Alternative) (1999)::Action\n"
            "10::Primary Two (a.k.a. Alternative) (1999)::Action\n"
            "11::Unique Primary (a.k.a. Unique Alternative) (2001)::Comedy\n"
        ).encode())
        self.index = MetadataIndex(self.rows)

    def test_article_representation_and_year_select_exact_item(self):
        self.assertEqual(self.index.resolve("The Fixture", 2005)["raw_ids"], [8])
        self.assertEqual(self.index.resolve("fixture, the", 1995)["raw_ids"], [7])

    def test_alias_is_backed_by_same_source_row(self):
        result = self.index.resolve("Unique Alternative", 2001)
        self.assertEqual(result["raw_ids"], [11])
        self.assertEqual(result["status"], "PASS")

    def test_collision_is_ambiguous_not_arbitrarily_chosen(self):
        result = self.index.resolve("Alternative", 1999)
        self.assertEqual(result["status"], "AMBIGUOUS")
        self.assertEqual(result["raw_ids"], [9, 10])

    def test_missing_year_never_chooses_remake(self):
        self.assertEqual(self.index.resolve("The Fixture", None)["status"], "AMBIGUOUS")
        self.assertEqual(self.index.resolve("The Fixture", 2010)["status"], "YEAR_CONFLICT")

    def test_unlisted_and_fuzzy_title_are_not_invented_aliases(self):
        self.assertEqual(self.index.resolve("Uniqe Alternative", 2001)["status"], "TITLE_ABSENT")
        self.assertEqual(self.index.resolve("Invented translation", 2001)["raw_ids"], [])

    def test_invalid_or_duplicate_raw_id_is_rejected(self):
        for raw in [b"0::Fixture (1995)::Comedy\n", b"7::Fixture (1995)::Comedy\n7::Other (2005)::Comedy\n"]:
            with self.assertRaises(ValueError):
                parse_movies_dat(raw)

    def test_crosswalk_uses_metadata_and_genres_not_model_row_count(self):
        result = catalog_crosswalk([{"id": 41, "title": "The Fixture", "release_date": 1995,
                                     "tags": ["Comedy", "Drama"]}], self.index)
        self.assertEqual(result[0]["raw_movie_id"], 7)
        self.assertEqual(result[0]["status"], "PASS")
        self.assertNotIn("checkpoint_row", result[0])

    def test_crosswalk_rejects_genre_conflict_and_many_to_one(self):
        conflict = catalog_crosswalk([{"id": 1, "title": "The Fixture", "release_date": 1995,
                                       "tags": ["Action"]}], self.index)
        self.assertEqual(conflict[0]["status"], "GENRE_CONFLICT")
        dup = catalog_crosswalk([{"id": i, "title": "Unique Alternative", "release_date": 2001,
                                 "tags": ["Comedy"]} for i in (1, 2)], self.index)
        self.assertEqual([r["status"] for r in dup], ["NON_BIJECTIVE"] * 2)

    def test_source_alias_returns_existing_catalog_id_only(self):
        walk = catalog_crosswalk([{"id": 41, "title": "Unique Primary (a.k.a. Unique Alternative)",
                                   "release_date": 2001, "tags": ["Comedy"]}], self.index)
        result = resolve_catalog_target("Unique Alternative", 2001, self.index, walk)
        self.assertEqual(result["catalog_ids"], [41])
        self.assertEqual(result["raw_ids"], [11])

    def test_movie_in_official_source_but_outside_catalog_is_not_inserted(self):
        result = resolve_catalog_target("The Fixture", 2005, self.index, [])
        self.assertEqual(result["status"], "CATALOG_ABSENT")
        self.assertEqual(result["catalog_ids"], [])

    def test_year_conflict_does_not_rewrite_target_or_select_another_year(self):
        result = resolve_catalog_target("Unique Alternative", 2002, self.index, [])
        self.assertEqual(result["status"], "YEAR_CONFLICT")
        self.assertEqual(result["raw_ids"], [])


if __name__ == "__main__":
    unittest.main()
