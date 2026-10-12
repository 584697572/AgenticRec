"""Offline tests of bounded ZIP reads, using only self-authored data."""
import importlib.util
import io
import unittest
import zipfile
from pathlib import Path

module_spec = importlib.util.spec_from_file_location("remote", Path(__file__).parents[1] / "scripts/inspect_remote_resources.py")
remote = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(remote)


class RemoteZipTests(unittest.TestCase):
    def test_directory_and_selected_member_read(self):
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", zipfile.ZIP_STORED) as archive:
            archive.writestr("movie/settings.json", '{"fixture": true}')
            archive.writestr("unused_large_file.bin", b"x" * 100000)
        data, fetched = output.getvalue(), []
        def fetch(start, end):
            fetched.append((start, end))
            return start, data[start:end+1]
        reader = remote.RemoteZipFile(len(data), fetch, (len(data)-4096, data[-4096:]))
        with zipfile.ZipFile(reader) as archive:
            self.assertEqual(archive.namelist(), ["movie/settings.json", "unused_large_file.bin"])
            self.assertEqual(archive.read("movie/settings.json"), b'{"fixture": true}')
        self.assertLess(sum(end-start+1 for start, end in fetched), 1000)

    def test_wrong_or_unbounded_range_is_rejected(self):
        reader = remote.RemoteZipFile(10_000_000, lambda start, end: (start+1, b"x"))
        with self.assertRaises(remote.RangeUnavailable):
            reader.read(1)
        with self.assertRaises(ValueError):
            reader.read()

    def test_public_confirmation_form_scope(self):
        form = remote.DownloadForm()
        form.feed('<input name="ignored" type="hidden"><form id="download-form" action="https://drive.usercontent.google.com/download"><input name="id" type="hidden" value="fixture"></form>')
        self.assertEqual(form.fields, {"id": "fixture"})


if __name__ == "__main__":
    unittest.main()
