"""CPU fixtures of original source; no real checkpoint or performance evaluation."""
import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("recheck", Path(__file__).parents[1] / "scripts/recheck_id_contract.py")
recheck = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recheck)


class OriginalLookupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def test_larger_embedding_accepts_catalog_subset(self):
        import torch
        model = recheck.fixture_model()
        candidates = torch.tensor(sorted(row["id"] for row in recheck.catalog()))
        scores = model.predict({"item_id": candidates, "item_seq": torch.tensor([[1062]]), "item_seq_len": torch.tensor([1])})
        self.assertEqual(scores.shape, (1, 9888))
        self.assertTrue(torch.isfinite(torch.from_numpy(scores)).all())

    def test_index_outside_embedding_is_rejected(self):
        import torch
        model = recheck.fixture_model()
        with self.assertRaises(IndexError):
            model.forward_item_emb(torch.tensor([36255]))

    def test_original_ranker_passes_catalog_ids_without_remapping(self):
        model = recheck.fixture_model()
        candidates = [17, 105, 232, 1062, 1023]
        tool = recheck.ranking_tool(model, candidates)
        message = tool.run('{"schema":"preference","prefer":["Toy Story"]}')
        self.assertNotIn("broken", message)
        self.assertEqual(model.calls[0].tolist(), [[1062]])
        self.assertEqual(len(tool.buffer.get()), 2)
        self.assertNotIn(1062, tool.buffer.get())
        self.assertTrue(set(tool.buffer.get()).issubset(candidates))


if __name__ == "__main__":
    unittest.main()
