"""Cache only the original default davinci token-count encoding (no API calls)."""
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
URL = "https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken"


def main():
    cache = ROOT / "reproduction/.runtime/isolated_profile/tiktoken"
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / hashlib.sha1(URL.encode()).hexdigest()
    downloaded = 0
    if not target.exists():
        with urllib.request.urlopen(URL, timeout=60) as response:
            data = response.read(2 * 1024 * 1024 + 1)
            assert len(data) <= 2 * 1024 * 1024
        assert len(data.splitlines()) == 50256
        target.write_bytes(data)
        downloaded = len(data)
    data = target.read_bytes()
    assert len(data.splitlines()) == 50256
    record = {"source": URL, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
              "downloaded_response_body_bytes": downloaded, "path": target.relative_to(ROOT).as_posix()}
    (ROOT / "reproduction/logs/a1_20261001/tokenizer_manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
