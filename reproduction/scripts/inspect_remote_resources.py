"""Inspect the public upstream ZIP using bounded HTTP ranges, without loading weights.

Source: Python zipfile and PKWARE ZIP specification. A missing Range response stops
the inspection; this script never silently falls back to a full archive download.
"""
import argparse
import hashlib
import io
import json
import re
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
FILE_ID = "1nSw2cuoi_WEOnHRg_eIyLWGBAjHdelsg"
LOG = ROOT / "reproduction/logs/t03_20261001"
DATA = ROOT / "data/raw/upstream_audit"


class DownloadForm(HTMLParser):
    def __init__(self):
        super().__init__()
        self.action, self.fields, self.active = None, {}, False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "form" and attrs.get("id") == "download-form":
            self.action, self.active = attrs.get("action"), True
        if tag == "input" and self.active and attrs.get("type") == "hidden":
            self.fields[attrs["name"]] = attrs.get("value", "")

    def handle_endtag(self, tag):
        if tag == "form":
            self.active = False


class RangeUnavailable(RuntimeError):
    pass


class BoundedHTTP:
    def __init__(self, cap):
        self.cap, self.bytes, self.requests = cap, 0, []
        self.url, self.total = None, None
        self.etag = None

    def read_response(self, response, limit):
        if self.bytes + limit > self.cap:
            raise ValueError("Network byte budget would be exceeded")
        body = response.read(limit)
        self.bytes += len(body)
        return body

    def initialize(self):
        public_url = "https://drive.google.com/uc?" + urlencode({"export": "download", "id": FILE_ID})
        with urlopen(Request(public_url, headers={"User-Agent": "AgenticRec-resource-audit/1.0"}), timeout=20) as response:
            page = self.read_response(response, 16384)
            self.requests.append({"kind": "public_download_form", "status": response.status, "bytes": len(page),
                                  "content_type": response.headers.get("Content-Type")})
        (LOG / "download_form.html").write_bytes(page)
        form = DownloadForm()
        form.feed(page.decode("utf-8"))
        if form.action != "https://drive.usercontent.google.com/download" or form.fields.get("id") != FILE_ID:
            raise RangeUnavailable("Expected public Google download form not found; no login attempted")
        self.url = form.action + "?" + urlencode(form.fields)
        return self.fetch_range("bytes=-65557", 65557)

    def fetch_range(self, requested, expected_limit):
        headers = {"User-Agent": "AgenticRec-resource-audit/1.0", "Range": requested, "Accept-Encoding": "identity"}
        if self.etag:
            headers["If-Match"] = self.etag
        record = {"kind": "range", "requested_range": requested}
        self.requests.append(record)
        with urlopen(Request(self.url, headers=headers), timeout=20) as response:
            record.update(status=response.status, content_range=response.headers.get("Content-Range"),
                          content_type=response.headers.get("Content-Type"), content_length=response.headers.get("Content-Length"))
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", record["content_range"] or "")
            if response.status != 206 or not match:
                body = self.read_response(response, min(4096, self.cap - self.bytes))
                record["bytes"] = len(body)
                (LOG / "range_failure_prefix.bin").write_bytes(body)
                raise RangeUnavailable("Server did not return a valid 206 Content-Range; stopped after small prefix")
            start, end, total = map(int, match.groups())
            expected = end - start + 1
            if expected > expected_limit or start < 0 or end >= total:
                raise RangeUnavailable("Unexpected range size or bounds")
            if requested.startswith("bytes=-"):
                if end != total - 1 or expected != min(total, int(requested[7:])):
                    raise RangeUnavailable("Incorrect suffix range")
            else:
                wanted = tuple(map(int, requested[6:].split("-")))
                if (start, end) != wanted:
                    raise RangeUnavailable("Server returned a different range")
            if self.total is not None and total != self.total:
                raise RangeUnavailable("Archive length changed during inspection")
            self.total = total
            self.etag = response.headers.get("ETag", self.etag)
            body = self.read_response(response, expected)
            record.update(bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
            if len(body) != expected:
                raise RangeUnavailable("Truncated range response")
            return start, body


class RemoteZipFile(io.RawIOBase):
    def __init__(self, size, fetch, initial=None):
        self.size, self.fetch, self.position = size, fetch, 0
        self.cache = [] if initial is None else [initial]

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset if whence == 2 else -1
        if position < 0:
            raise ValueError("Invalid seek")
        self.position = position
        return position

    def read(self, count=-1):
        count = self.size - self.position if count < 0 else min(count, self.size - self.position)
        if count <= 0:
            return b""
        if count > 2 * 1024 * 1024:
            raise ValueError("Single ZIP read exceeds inspection limit")
        start, end = self.position, self.position + count
        for offset, data in self.cache:
            if offset <= start and end <= offset + len(data):
                self.position = end
                return data[start-offset:end-offset]
        offset, data = self.fetch(start, end - 1)
        if offset != start or len(data) != count:
            raise RangeUnavailable("Read bounds mismatch")
        self.cache.append((offset, data))
        self.position = end
        return data


def inspect(client, details=False):
    initial = client.initialize()
    remote = RemoteZipFile(client.total, lambda start, end: client.fetch_range(f"bytes={start}-{end}", end-start+1), initial)
    with zipfile.ZipFile(remote) as archive:
        entries = [{"name": item.filename, "bytes": item.file_size, "compressed_bytes": item.compress_size,
                    "compression": item.compress_type, "crc32": f"{item.CRC:08x}", "header_offset": item.header_offset,
                    "encrypted": bool(item.flag_bits & 1)} for item in archive.infolist()]
        (LOG / "archive_entries.json").write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
        selected = []
        for item in archive.infolist():
            name = item.filename.lower()
            if details:
                if name == "movie/movies.ftr":
                    if item.file_size > 1024*1024:
                        raise ValueError("Movie table exceeds one MiB")
                    body = archive.read(item)
                    target = DATA / "movies.ftr"
                    if target.exists() and target.read_bytes() != body:
                        raise FileExistsError(target)
                    target.write_bytes(body)
                    selected.append({"name": item.filename, "status": "read_crc_verified", "bytes": len(body),
                                     "sha256": hashlib.sha256(body).hexdigest(), "local_path": target.relative_to(ROOT).as_posix()})
                elif name in ("movie/movie_sim.npy", "movie/sasrec-sasrec.pth"):
                    prefix_size = 512 if name.endswith(".npy") else 65536
                    with archive.open(item) as member:
                        body = member.read(prefix_size)
                    target = DATA / ("movie_sim.header.bin" if name.endswith(".npy") else "checkpoint.prefix.bin")
                    target.write_bytes(body)
                    selected.append({"name": item.filename, "status": "prefix_only_crc_not_verified", "bytes": len(body),
                                     "sha256": hashlib.sha256(body).hexdigest(), "local_path": target.relative_to(ROOT).as_posix()})
                continue
            metadata = any(word in Path(name).name for word in ("license", "readme", "terms", "copyright"))
            settings = name.endswith("settings.json")
            movie_json = ("movie" in name and name.endswith(".json"))
            if not (metadata or settings or movie_json) or item.is_dir():
                continue
            if item.file_size > 256 * 1024 or item.flag_bits & 1:
                selected.append({"name": item.filename, "status": "skipped_size_or_encryption"})
                continue
            body = archive.read(item)  # zipfile checks member CRC on complete read.
            safe_name = f"{len(selected):02d}_" + re.sub(r"[^a-zA-Z0-9_.-]", "_", Path(item.filename).name)
            target = DATA / safe_name
            if target.exists():
                raise FileExistsError(target)
            target.write_bytes(body)
            selected.append({"name": item.filename, "status": "read_crc_verified", "bytes": len(body),
                             "sha256": hashlib.sha256(body).hexdigest(), "local_path": target.relative_to(ROOT).as_posix()})
        return {"archive_bytes": client.total, "archive_entry_count": len(entries), "selected": selected,
                "full_archive_downloaded": False, "weights_loaded": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--byte-cap", type=int, default=2*1024*1024)
    parser.add_argument("--details", action="store_true", help="Read only small movie table and matrix/checkpoint prefixes.")
    args = parser.parse_args()
    global LOG
    if args.details:
        LOG = LOG / "details"
    LOG.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    client = BoundedHTTP(args.byte_cap)
    result = {"source": "upstream README GoogleDrive link", "byte_cap": args.byte_cap,
              "status": "failed", "full_archive_downloaded": False, "weights_loaded": False}
    try:
        result.update(inspect(client, details=args.details), status="completed")
    except Exception as error:
        result.update(error_type=type(error).__name__, error=str(error))
    finally:
        result.update(received_body_bytes=client.bytes, requests=client.requests)
        (LOG / "remote_inspection.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
