"""Fetch only the pinned raw ReDial test input referenced by the upstream notebook.

ReDial: Li et al., Towards Deep Conversational Recommendations, NeurIPS 2018.
Dataset terms: https://redialdata.github.io/website/datasheet.html (CC BY 4.0).
UniCRS code MIT does not license the original MovieLens resource bundle.
"""
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMMIT = "f7d8f95104d6eba5371cfa845b243057a632079e"
NAME = "test_data_dbpedia_raw.jsonl"
URL = f"https://raw.githubusercontent.com/RUCAIBox/UniCRS/{COMMIT}/data/redial/{NAME}"
SIZE = 8751948
BLOB = "e2a2eeab9a41a4fcdb121d3ffe8550e6ef47f1ab"
DEST = ROOT / "data/raw/unicrs" / COMMIT / NAME


def validate(data):
    if len(data) != SIZE:
        raise ValueError("pinned_test_size_mismatch")
    blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    if blob != BLOB:
        raise ValueError("pinned_git_blob_mismatch")
    rows = [json.loads(line) for line in data.splitlines() if line.strip()]
    required = {"conversationId", "messages", "movieMentions", "initiatorQuestions"}
    if not rows or not all(required <= row.keys() for row in rows):
        raise ValueError("unexpected_raw_dialogue_schema")
    if len({row["conversationId"] for row in rows}) != len(rows):
        raise ValueError("duplicate_conversation_ids")
    return {"url": URL, "upstream_commit": COMMIT, "git_blob_sha1": blob,
            "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
            "conversation_count": len(rows), "path": DEST.relative_to(ROOT).as_posix(),
            "original_50_sample_file": "NOT VERIFIED", "evaluation": "NOT EVALUATED"}


def main():
    downloaded = 0
    if DEST.exists():
        data = DEST.read_bytes()
    else:
        request = urllib.request.Request(URL, headers={"Range": f"bytes=0-{SIZE-1}",
                                                       "Accept-Encoding": "identity"})
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status == 206 and response.headers.get("Content-Range") != f"bytes 0-{SIZE-1}/{SIZE}":
                raise ValueError("unexpected_content_range")
            chunks, received = [], 0
            while received <= SIZE:
                chunk = response.read(min(1024 * 1024, SIZE + 1 - received))
                if not chunk:
                    break
                chunks.append(chunk)
                received += len(chunk)
            data = b"".join(chunks)
        validate(data)
        DEST.parent.mkdir(parents=True, exist_ok=True)
        with DEST.open("xb") as handle:
            handle.write(data)
        downloaded = len(data)
    result = validate(data)
    result["downloaded_bytes_this_run"] = downloaded
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
