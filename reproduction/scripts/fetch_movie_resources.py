"""Fetch only original movie resources, with bounded ranges and complete member CRCs."""
import hashlib
import json
import os
import uuid
import zipfile
import zlib
from pathlib import Path

import inspect_remote_resources as remote

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "reproduction/logs/a1_20261001/movie_download"
DEST = ROOT / "data/raw/upstream_movie"
CAP = 420 * 1024 * 1024


def check_file(path):
    sha, crc, size = hashlib.sha256(), 0, 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
            crc = zlib.crc32(chunk, crc)
            size += len(chunk)
    return size, crc & 0xffffffff, sha.hexdigest()


def main():
    LOG.mkdir(parents=True, exist_ok=True)
    DEST.mkdir(parents=True, exist_ok=True)
    remote.LOG = LOG
    client = remote.BoundedHTTP(CAP)
    results = []
    record = {"source": "Pinned InteRecAgent README original all_resources.zip", "byte_cap": CAP,
              "full_archive_downloaded": False, "weights_loaded": False, "status": "failed"}
    try:
        initial = client.initialize()
        reader = remote.RemoteZipFile(client.total, lambda start, end: client.fetch_range(f"bytes={start}-{end}", end-start+1), initial)
        with zipfile.ZipFile(reader) as archive:
            expected = json.loads((ROOT / "reproduction/archive_inventory.json").read_text())
            for entry in expected:
                actual = archive.getinfo(entry["name"])
                if (actual.file_size, actual.CRC) != (entry["bytes"], int(entry["crc32"], 16)):
                    raise ValueError("Archive changed since resource audit")
            audit_files = {"settings.json": "03_settings.json", "columns.json": "02_columns.json", "movies.ftr": "movies.ftr"}
            for name in ("settings.json", "columns.json", "movies.ftr", "SASRec-SASRec.pth", "movie_sim.npy"):
                entry = archive.getinfo("movie/" + name)
                target = DEST / name
                if target.exists():
                    size, crc, sha = check_file(target)
                    if (size, crc) != (entry.file_size, entry.CRC):
                        raise ValueError("Existing resource differs; will not overwrite " + name)
                    mode = "existing_crc_verified"
                else:
                    partial = target.with_name(target.name + ".partial." + uuid.uuid4().hex)
                    with partial.open("xb") as output:
                        if name in audit_files:
                            body = (ROOT / "data/raw/upstream_audit" / audit_files[name]).read_bytes()
                            output.write(body)
                        else:
                            with archive.open(entry) as source:
                                while True:
                                    chunk = source.read(1024 * 1024)
                                    if not chunk:
                                        break
                                    output.write(chunk)
                                    reader.cache = reader.cache[-4:]
                    size, crc, sha = check_file(partial)
                    if (size, crc) != (entry.file_size, entry.CRC):
                        raise ValueError("Incomplete resource or CRC mismatch " + name)
                    if target.exists():
                        raise FileExistsError(target)
                    os.replace(partial, target)
                    mode = "copied_audited_member" if name in audit_files else "downloaded_crc_verified"
                results.append({"name": "movie/" + name, "path": target.relative_to(ROOT).as_posix(),
                                "bytes": size, "sha256": sha, "crc32": f"{crc:08x}", "status": mode})
                print(json.dumps(results[-1]), flush=True)
            record["status"] = "completed"
    except Exception as error:
        record.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        record.update(resources=results, received_body_bytes=client.bytes, requests=client.requests)
        (LOG / "download_result.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
