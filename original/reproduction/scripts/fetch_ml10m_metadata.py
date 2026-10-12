"""Extract only official movie metadata and terms using bounded HTTPS ranges.

Windows native certificate validation resolves the observed Python TLS-chain
failure. No ratings, tags, full ZIP, API key, model or embedding is downloaded.
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import zipfile

from inspect_remote_resources import RemoteZipFile

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "reproduction/logs/catalog_identity_20261005"
DEST = ROOT / "data/raw/metadata_audit/ml-10m"
URL = "https://files.grouplens.org/datasets/movielens/ml-10m.zip"


def main():
    if (LOG / "metadata_acquisition.json").exists():
        print(json.dumps({"status": "BLOCKED", "reason": "metadata_acquisition_already_recorded", "new_download_bytes": 0}))
        return 2
    LOG.mkdir(parents=True, exist_ok=True)
    probe_path = LOG / "ml10m_windows_probe.json"
    if not probe_path.exists():
        command = ["powershell.exe", "-NoProfile", "-NonInteractive", "-File",
            str(ROOT / "reproduction/scripts/windows_metadata_range.ps1"), "-Url", URL,
            "-SuffixBytes", "65557", "-OutputPath", str(LOG / "ml10m_windows_suffix.bin")]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=45)
        if result.returncode:
            raise RuntimeError("native_metadata_suffix_failed")
        probe = json.loads(result.stdout.decode("utf-8-sig"))
        with probe_path.open("x", encoding="utf-8") as f:
            f.write(json.dumps(probe, indent=2) + "\n")
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    matched = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", probe["content_range"])
    if probe["status"] != 206 or not matched:
        raise ValueError("verified_suffix_probe_required")
    start, end, total = map(int, matched.groups())
    initial = (LOG / "ml10m_windows_suffix.bin").read_bytes()
    assert len(initial) == end - start + 1 and end == total - 1
    requests, downloaded = [], len(initial)
    def fetch(first, last):
        nonlocal downloaded
        size = last - first + 1
        if downloaded + size > 1024 * 1024:
            raise ValueError("metadata_one_MiB_download_budget_exceeded")
        path = LOG / ("ml10m_range_%03d.bin" % (len(requests) + 1))
        command = ["powershell.exe", "-NoProfile", "-NonInteractive", "-File",
            str(ROOT / "reproduction/scripts/windows_metadata_range.ps1"), "-Url", URL,
            "-RangeStart", str(first), "-RangeEnd", str(last), "-OutputPath", str(path)]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=45)
        if result.returncode:
            raise RuntimeError("native_metadata_range_failed: " + result.stderr.decode("utf-8", errors="replace"))
        record = json.loads(result.stdout.decode("utf-8-sig"))
        requests.append(record)
        if record["status"] != 206 or record["content_range"] != "bytes %d-%d/%d" % (first, last, total):
            raise ValueError("metadata_range_not_honored; no full-download fallback")
        raw = path.read_bytes()
        if len(raw) != size:
            raise ValueError("metadata_range_truncated")
        downloaded += len(raw)
        record["sha256"] = hashlib.sha256(raw).hexdigest()
        return first, raw
    members = []
    DEST.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(RemoteZipFile(total, fetch, (start, initial))) as archive:
        for name in ("ml-10M100K/movies.dat", "ml-10M100K/README.html"):
            item = archive.getinfo(name)
            if item.file_size > 1024 * 1024 or item.compress_size > 1024 * 1024 or item.flag_bits & 1:
                raise ValueError("unexpected_metadata_member")
            raw = archive.read(item)  # Complete member read validates ZIP CRC.
            path = DEST / Path(name).name
            with path.open("xb") as handle:
                handle.write(raw)
            members.append({"name": name, "path": path.relative_to(ROOT).as_posix(), "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(), "crc32": "%08x" % item.CRC})
    summary = {"status": "PASS", "source": URL, "archive_bytes": total, "full_archive_downloaded": False,
        "downloaded_bytes_including_initial_probe": downloaded, "byte_cap": 1024 * 1024,
        "members": members, "requests": requests, "ratings_downloaded": False, "tags_downloaded": False,
        "tls_verification": "Windows default certificate validation; never disabled",
        "use": "local metadata provenance audit only; not training or benchmark replacement",
        "license": "MovieLens research terms; no dataset redistribution", "api_requests": 0}
    with (LOG / "metadata_acquisition.json").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
