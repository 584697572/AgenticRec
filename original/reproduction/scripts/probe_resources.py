"""Bounded public-resource probes; never fetch a full archive or load weights."""
import argparse
import hashlib
import json
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reproduction/logs/g0"
README = (ROOT / "RecAI/InteRecAgent/README.md").read_text(encoding="utf-8")
LINKS = dict(re.findall(r"\[(GoogleDrive|RecDrive)\]\((https://[^)]+)\)", README))
LINKS.update({
    "MovieLensTerms": "https://files.grouplens.org/datasets/movielens/ml-1m-README.txt",
    "Issue110": "https://api.github.com/repos/microsoft/RecAI/issues/110",
})


def probe(name, url):
    result = {"name": name, "url": url, "checked_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
              "timeout_seconds": 20, "body_limit_bytes": 65536, "http_status": None,
              "archive_downloaded": False, "resource_integrity": "NOT VERIFIED"}
    try:
        request = Request(url, headers={"User-Agent": "AgenticRec-resource-audit/1.0"})
        try:
            response = urlopen(request, timeout=20)
        except HTTPError as error:
            response = error
        with response:
            result.update(http_status=response.code, final_url=response.url,
                          content_type=response.headers.get("Content-Type"),
                          content_length=response.headers.get("Content-Length"))
            body = response.read(65536)
            body_path = OUT / (name + ".response.txt")
            body_path.write_bytes(body)
            result.update(body_sha256=hashlib.sha256(body).hexdigest(), body_bytes=len(body),
                          body_path=body_path.relative_to(ROOT).as_posix(), transport_status="completed")
    except (URLError, TimeoutError, OSError) as error:
        result.update(transport_status="failed", error_type=type(error).__name__, error=str(error))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-label", choices=("sandbox", "network", "download"), default="sandbox")
    args = parser.parse_args()
    OUT = OUT / args.run_label
    if args.run_label == "download":
        file_id = re.search(r"/file/d/([^/]+)/", LINKS["GoogleDrive"])[1]
        LINKS = {"GoogleDriveDownload": "https://drive.google.com/uc?export=download&id=" + file_id}
    OUT.mkdir(parents=True, exist_ok=True)
    results = [probe(name, url) for name, url in LINKS.items()]
    (OUT / "resource_access.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for result in results:
        print(json.dumps(result, ensure_ascii=False))
