"""Download only the original Gallery model's required files from a fixed HF revision."""
import hashlib
import json
import os
import time
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "reproduction/logs/a1_20261001"
DEST = ROOT / "reproduction/.runtime/gte-base"
FILES = {"config.json", "modules.json", "sentence_bert_config.json", "special_tokens_map.json",
         "tokenizer.json", "tokenizer_config.json", "vocab.txt", "1_Pooling/config.json", "pytorch_model.bin"}


def main():
    metadata_url = "https://huggingface.co/api/models/thenlper/gte-base?blobs=true"
    with urlopen(Request(metadata_url, headers={"User-Agent": "AgenticRec-A1/1.0"}), timeout=30) as response:
        body = response.read(2*1024*1024+1)
    if len(body) > 2*1024*1024:
        raise ValueError("Model metadata exceeds bound")
    info = json.loads(body)
    revision = info["sha"]
    selected = [item for item in info["siblings"] if item["rfilename"] in FILES]
    if not FILES <= {item["rfilename"] for item in selected}:
        raise ValueError("Required original model files missing")
    (LOG / "gte_selected_metadata.json").write_text(json.dumps({"model": info["id"], "revision": revision,
        "license": info.get("cardData", {}).get("license"), "selected": selected}, indent=2)+"\n")
    DEST.mkdir(parents=True, exist_ok=True)
    records = []
    for item in selected:
        name = item["rfilename"]
        path = DEST / name
        path.parent.mkdir(parents=True, exist_ok=True)
        size = item.get("size", item.get("lfs", {}).get("size"))
        sha_expected = item.get("lfs", {}).get("sha256")
        if size is None or size > 500*1024*1024:
            raise ValueError("Unexpected model file size")
        if not path.exists():
            partial = path.with_name(path.name+".partial")
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > size:
                raise ValueError("Existing partial file too large")
            if offset < size:
                url = f"https://huggingface.co/thenlper/gte-base/resolve/{revision}/{name}"
                headers = {"User-Agent": "AgenticRec-A1/1.0", "Accept-Encoding": "identity"}
                if offset:
                    headers["Range"] = f"bytes={offset}-"
                with urlopen(Request(url, headers=headers), timeout=30) as response:
                    if offset and (response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
                        raise ValueError("Model resume range ignored")
                    if not offset and response.status != 200:
                        raise ValueError("Unexpected model response")
                    with partial.open("ab" if offset else "xb") as output:
                        while True:
                            chunk = response.read(512*1024)
                            if not chunk:
                                break
                            offset += len(chunk)
                            if offset > size:
                                raise ValueError("Model body exceeds size")
                            output.write(chunk)
            if offset != size:
                raise ValueError("Truncated model response")
            sha = hashlib.sha256(partial.read_bytes()).hexdigest()
            if sha_expected and sha != sha_expected:
                raise ValueError("LFS hash mismatch")
            os.replace(partial, path)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.stat().st_size != size or (sha_expected and sha != sha_expected):
            raise ValueError("Cached model integrity mismatch")
        records.append({"name": name, "bytes": size, "sha256": sha, "lfs_sha256_verified": bool(sha_expected)})
        print(json.dumps(records[-1]), flush=True)
    (LOG / "gte_model_manifest.json").write_text(json.dumps({"model": "thenlper/gte-base", "revision": revision,
        "files": records, "unnecessary_onnx_safetensors_tf_downloaded": False}, indent=2)+"\n")


if __name__ == "__main__":
    main()
