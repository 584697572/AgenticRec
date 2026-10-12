"""Credential loading tests use only self-authored secrets and no network."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import local_api_key as local
import live_app_single_turn as app


class LocalKeyTests(unittest.TestCase):
    def test_environment_takes_precedence_without_reading_file(self):
        with patch.object(Path, "read_text", side_effect=AssertionError("unnecessary secret file read")):
            self.assertEqual(local.read_api_key(environ={"OPENAI_API_KEY": "env-fixture"}), "env-fixture")

    def test_utf8_bom_quotes_crlf_and_comments(self):
        for assignment in ('OPENAI_API_KEY="local-fixture" # comment', "export OPENAI_API_KEY='local-fixture'", 'OPENAI_API_KEY=local-fixture # comment'):
            with self.subTest(assignment=assignment), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / ".env"
                path.write_bytes(("\ufeff# local file\r\n" + assignment + "\r\n").encode())
                self.assertEqual(local.read_api_key(path, {}), "local-fixture")

    def test_missing_or_empty_key_is_not_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            self.assertEqual(local.read_api_key(path, {}), "")
            path.write_text("OPENAI_API_KEY=\n")
            self.assertEqual(local.read_api_key(path, {}), "")

    def test_only_key_is_read_no_expansion_or_environment_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("allow_paid_api=true\napi_request_cap=999\nOPENAI_API_KEY=${PRIVATE_FIXTURE}\n")
            environment = {"PRIVATE_FIXTURE": "do-not-expand"}
            self.assertEqual(local.read_api_key(path, environment), "${PRIVATE_FIXTURE}")
            self.assertEqual(environment, {"PRIVATE_FIXTURE": "do-not-expand"})

    def test_malformed_or_duplicate_key_never_echoed(self):
        for text in ('OPENAI_API_KEY="hidden-fixture', 'OPENAI_API_KEY=hidden-fixture\nOPENAI_API_KEY=second-fixture', 'OPENAI_API_KEY hidden-fixture', 'OPENAI_API_KEY="hidden-fixture" trailing'):
            with self.subTest(), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / ".env"
                path.write_text(text)
                with self.assertRaises(local.LocalKeyError) as caught:
                    local.read_api_key(path, {})
                self.assertNotIn("hidden-fixture", str(caught.exception))
                self.assertNotIn("second-fixture", str(caught.exception))

    def test_decode_failure_is_sanitized(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_bytes(b'OPENAI_API_KEY=private-fixture\xff')
            with self.assertRaises(local.LocalKeyError) as caught:
                local.read_api_key(path, {})
            self.assertNotIn("private-fixture", str(caught.exception))

    def test_live_entry_uses_file_without_echoing_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("OPENAI_API_KEY=local-fixture\n")
            profile = Path(directory) / "profile.json"
            profile.write_text(json.dumps(app.PROFILE))
            output = io.StringIO()
            with patch.dict(os.environ, {}, clear=True), patch.object(local, "WORKSPACE_ENV_FILE", path), \
                 patch.object(app, "LOCAL_PROFILE", profile), patch.object(sys, "argv", ["live_app_single_turn.py"]), \
                 patch.object(app, "execute", return_value={"status": "SUCCESS", "remote_api_requests": 0}) as execute, \
                 contextlib.redirect_stdout(output):
                self.assertEqual(app.main(), 0)
            self.assertEqual(execute.call_args.args[1], "local-fixture")
            self.assertNotIn("local-fixture", output.getvalue())

    def test_offline_entry_never_reads_secret(self):
        with patch.object(app, "read_api_key", side_effect=AssertionError("offline secret read")), \
             patch.object(sys, "argv", ["live_app_single_turn.py", "--offline"]), \
             patch.object(app, "execute", return_value={"status": "SUCCESS"}), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(app.main(), 0)

    def test_worker_import_does_not_read_dotenv(self):
        code = "from unittest.mock import patch\nfrom pathlib import Path\nwith patch.object(Path, 'read_text', side_effect=AssertionError('unexpected file read')):\n import original_app_worker\nprint('keyless worker import PASS')"
        result = subprocess.run([sys.executable, "-c", code], cwd=SCRIPTS, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))


if __name__ == "__main__":
    unittest.main()
