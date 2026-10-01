"""Download the small, fixed UniRec wheel for source inspection, without installing it."""
import hashlib
import io
import json
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "reproduction/logs/t03_recheck_20261001"
DATA = ROOT / "data/raw/upstream_audit"
VERSION = "0.0.1a4"  # upstream requirements minimum, not an inferred training version.
SOURCES = ["unirec/utils/general.py", "unirec/model/base/recommender.py",
           "unirec/model/base/reco_abc.py", "unirec/model/sequential/seqrec_base.py",
           "unirec/model/sequential/sasrec.py",
           "unirec/model/modules.py"]


def bounded_get(url, limit):
    with urlopen(Request(url, headers={"User-Agent": "AgenticRec-source-audit/1.0"}), timeout=20) as response:
        if response.status != 200:
            raise ValueError("Unexpected HTTP status")
        if response.headers.get("Content-Length") and int(response.headers["Content-Length"]) > limit:
            raise ValueError("Declared response exceeds source audit limit")
        body = response.read(limit + 1)
        if len(body) > limit:
            raise ValueError("Response exceeds source audit limit")
        return body


def main():
    LOG.mkdir(parents=True, exist_ok=True)
    metadata_url = f"https://pypi.org/pypi/unirec/{VERSION}/json"
    metadata_bytes = bounded_get(metadata_url, 128 * 1024)
    metadata = json.loads(metadata_bytes)
    filename = f"unirec-{VERSION}-py3-none-any.whl"
    package = next(item for item in metadata["urls"] if item["filename"] == filename)
    if not package["url"].startswith("https://files.pythonhosted.org/") or package["size"] > 160 * 1024:
        raise ValueError("Unexpected wheel source or size")
    target = DATA / filename
    received = len(metadata_bytes)
    if target.exists():
        body = target.read_bytes()
    else:
        body = bounded_get(package["url"], package["size"])
        received += len(body)
    if len(body) != package["size"] or hashlib.sha256(body).hexdigest() != package["digests"]["sha256"]:
        raise ValueError("Wheel checksum/size mismatch")
    if not target.exists():
        target.write_bytes(body)
    sources = []
    with zipfile.ZipFile(io.BytesIO(body)) as wheel:
        for member in SOURCES:
            if member not in wheel.namelist():
                sources.append({"member": member, "status": "absent"})
                continue
            info = wheel.getinfo(member)
            if info.file_size > 256 * 1024:
                raise ValueError("Source member too large")
            source = wheel.read(member)  # CRC verified, no import or package execution.
            path = LOG / "unirec_source" / member
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(source)
            sources.append({"member": member, "local_path": path.relative_to(ROOT).as_posix(),
                            "sha256": hashlib.sha256(source).hexdigest(), "status": "read_crc_verified"})
    result = {"metadata_url": metadata_url, "wheel_url": package["url"], "version": VERSION,
              "filename": filename, "bytes": len(body), "sha256": package["digests"]["sha256"],
              "received_body_bytes": received, "installed": False, "sources": sources,
              "version_scope": "Upstream permitted minimum; checkpoint training library version NOT VERIFIED"}
    (LOG / "unirec_source_acquisition.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
