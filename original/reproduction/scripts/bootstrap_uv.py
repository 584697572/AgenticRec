"""Fetch and verify a pinned project-local Windows uv executable, without system install."""
import hashlib
import io
import json
import os
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
URL = "https://files.pythonhosted.org/packages/27/33/f1b73f9d019dcfe83fb545e5d0d080952300248d547433bb975ad7fd3331/uv-0.9.2-py3-none-win_amd64.whl"
WHEEL_SHA = "27815edaa0be4f6346b24f3b6a4389f7747ae98538f4e2e8946090bb959389cc"
EXE_SHA = "c819fa6467c35683274f458ae7a6d0ba31d48e902933287abe99e9a49c3fdbf9"


def main():
    assert os.name == "nt", "This bootstrap is for the recorded Windows environment"
    target = ROOT / "reproduction/.runtime/uv.exe"
    downloaded = 0
    if target.exists():
        assert hashlib.sha256(target.read_bytes()).hexdigest() == EXE_SHA, "Unexpected existing executable; refusing overwrite"
    else:
        with urllib.request.urlopen(URL, timeout=120) as response:
            data = response.read(24 * 1024 * 1024 + 1)
        assert len(data) == 21359072 and hashlib.sha256(data).hexdigest() == WHEEL_SHA
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = [name for name in archive.namelist() if name.endswith("/uv.exe")]
            assert len(names) == 1
            executable = archive.read(names[0])
        assert hashlib.sha256(executable).hexdigest() == EXE_SHA
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(executable)
        downloaded = len(data)
    print(json.dumps({"uv_version": "0.9.2", "executable_sha256": EXE_SHA,
                      "downloaded_response_body_bytes": downloaded, "system_installed": False}))


if __name__ == "__main__":
    main()
