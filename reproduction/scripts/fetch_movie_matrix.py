"""Resume bounded compressed matrix ranges; verify the original member before use."""
import hashlib
import http.client
import json
import os
import ssl
import struct
import time
import uuid
import zipfile
import zlib
from pathlib import Path
from urllib.error import URLError

import inspect_remote_resources as remote

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "reproduction/logs/a1_20261001/matrix_resume"
CACHE = ROOT / "data/raw/upstream_movie/.download-cache"
DEST = ROOT / "data/raw/upstream_movie/movie_sim.npy"
BLOCK = 8 * 1024 * 1024


def cached_block(fetch, start, end, path, known=None, retry_delay=2):
    size = end - start + 1
    if path.exists():
        body = path.read_bytes()
        if len(body) != size or known is None or hashlib.sha256(body).hexdigest() != known:
            raise ValueError("Cached block integrity is not verified")
        return body
    for attempt in range(3):
        try:
            offset, body = fetch(start, end)
            if offset != start or len(body) != size:
                raise ValueError("Returned block bounds mismatch")
            temporary = path.with_name(path.name + ".partial." + uuid.uuid4().hex)
            temporary.write_bytes(body)
            if path.exists():
                raise FileExistsError(path)
            os.replace(temporary, path)
            return body
        except (URLError, TimeoutError, ssl.SSLError, http.client.IncompleteRead):
            if attempt == 2:
                raise
            time.sleep(retry_delay * (attempt + 1))


def main():
    LOG.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    remote.LOG = LOG
    client = remote.BoundedHTTP(420 * 1024 * 1024)
    state_path = CACHE / "matrix_blocks.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"blocks": {}}
    result = {"status": "failed", "full_archive_downloaded": False, "retries_per_block_max": 3}
    try:
        initial = client.initialize()
        reader = remote.RemoteZipFile(client.total, lambda s, e: client.fetch_range(f"bytes={s}-{e}", e-s+1), initial)
        with zipfile.ZipFile(reader) as archive:
            entry = archive.getinfo("movie/movie_sim.npy")
        expected = next(v for v in json.loads((ROOT / "reproduction/archive_inventory.json").read_text()) if v["name"] == entry.filename)
        assert entry.file_size == expected["bytes"] and entry.CRC == int(expected["crc32"], 16)
        assert entry.compress_type == zipfile.ZIP_DEFLATED and not entry.flag_bits & 1
        _, header = client.fetch_range(f"bytes={entry.header_offset}-{entry.header_offset+29}", 30)
        fields = struct.unpack("<4s5H3I2H", header)
        assert fields[0] == b"PK\x03\x04"
        start = entry.header_offset + 30 + fields[-2] + fields[-1]
        identity = {"archive_bytes": client.total, "compressed_start": start,
                    "compressed_bytes": entry.compress_size, "crc32": entry.CRC, "bytes": entry.file_size}
        if "identity" in state and state["identity"] != identity:
            raise ValueError("Archive changed; refusing cached blocks")
        state["identity"] = identity
        if DEST.exists():
            from fetch_movie_resources import check_file
            size, crc, sha = check_file(DEST)
            assert (size, crc) == (entry.file_size, entry.CRC)
        else:
            temporary = DEST.with_name(DEST.name + ".partial." + uuid.uuid4().hex)
            decompressor = zlib.decompressobj(-15)
            size, crc, sha_obj = 0, 0, hashlib.sha256()
            with temporary.open("xb") as output:
                for number, offset in enumerate(range(0, entry.compress_size, BLOCK)):
                    length = min(BLOCK, entry.compress_size-offset)
                    key = str(number)
                    body = cached_block(lambda s, e: client.fetch_range(f"bytes={s}-{e}", e-s+1), start+offset,
                                        start+offset+length-1, CACHE / (key + ".deflate"), state["blocks"].get(key))
                    state["blocks"][key] = hashlib.sha256(body).hexdigest()
                    state_path.write_text(json.dumps(state, indent=2) + "\n")
                    decoded = decompressor.decompress(body)
                    if size + len(decoded) > entry.file_size:
                        raise ValueError("Decoded matrix exceeds declared size")
                    output.write(decoded)
                    size += len(decoded)
                    crc = zlib.crc32(decoded, crc)
                    sha_obj.update(decoded)
                    print(json.dumps({"block": number, "compressed_completed": offset+length,
                                      "compressed_total": entry.compress_size}), flush=True)
                tail = decompressor.flush()
                output.write(tail)
                size += len(tail)
                crc = zlib.crc32(tail, crc)
                sha_obj.update(tail)
            assert decompressor.eof and not decompressor.unused_data
            assert size == entry.file_size and (crc & 0xffffffff) == entry.CRC
            sha = sha_obj.hexdigest()
            if DEST.exists():
                raise FileExistsError(DEST)
            os.replace(temporary, DEST)
        result.update(status="completed", bytes=size, sha256=sha, crc32=f"{entry.CRC:08x}",
                      path=DEST.relative_to(ROOT).as_posix(), all_original_member_crc_verified=True)
    except Exception as error:
        result.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        result.update(received_complete_response_body_bytes=client.bytes, requests=client.requests,
                      partial_failed_response_bytes="unknown; excluded from completed-body accounting")
        (LOG / "matrix_download_result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
