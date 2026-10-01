"""Check download resume and retry behavior on self-authored compressed blocks."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from urllib.error import URLError
import sys

scripts = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(scripts))
import fetch_movie_matrix as matrix


class MatrixResumeTests(unittest.TestCase):
    def test_transient_retry_and_cached_block_reuse(self):
        import hashlib
        calls = []
        def fetch(start, end):
            calls.append((start, end))
            if len(calls) == 1:
                raise URLError("self-authored transient error")
            return start, b"abcd"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "0.deflate"
            body = matrix.cached_block(fetch, 10, 13, path, retry_delay=0)
            self.assertEqual(body, b"abcd")
            matrix.cached_block(fetch, 10, 13, path, hashlib.sha256(body).hexdigest())
            self.assertEqual(len(calls), 2)

    def test_wrong_bounds_or_changed_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "0.deflate"
            with self.assertRaises(ValueError):
                matrix.cached_block(lambda s, e: (s+1, b"abcd"), 10, 13, path)
            path.write_bytes(b"bad!")
            with self.assertRaises(ValueError):
                matrix.cached_block(lambda s, e: (s, b"abcd"), 10, 13, path, "wrong")


if __name__ == "__main__":
    unittest.main()
