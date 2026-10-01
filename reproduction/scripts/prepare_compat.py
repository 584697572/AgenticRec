"""Keep a pinned untouched source copy and attach original resources for A1 checks."""
import hashlib
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "RecAI/InteRecAgent"
TARGET = ROOT / "reproduction/.runtime/InteRecAgent-compat"
LOG = ROOT / "reproduction/logs/a1_20261001"


def main():
    if not TARGET.exists():
        shutil.copytree(SOURCE, TARGET, ignore=shutil.ignore_patterns("__pycache__", "resources"))
        records = [{"path": p.relative_to(SOURCE).as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                   for p in SOURCE.rglob("*") if p.is_file() and "__pycache__" not in p.parts and "resources" not in p.parts]
        for entry in records:
            assert hashlib.sha256((TARGET / entry["path"]).read_bytes()).hexdigest() == entry["sha256"]
        (LOG / "compat_initial_copy.json").write_text(json.dumps({"commit": "0959ecb05b0794748426e73e6efc1b6b35ec433d", "files": records}, indent=2) + "\n")
    attached, missing = [], []
    resources = TARGET / "resources/movie"
    resources.mkdir(parents=True, exist_ok=True)
    for name in ("settings.json", "columns.json", "movies.ftr", "SASRec-SASRec.pth", "movie_sim.npy"):
        source, target = ROOT / "data/raw/upstream_movie" / name, resources / name
        if not source.exists():
            missing.append(name)
            continue
        if not target.exists():
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
        assert source.stat().st_size == target.stat().st_size
        attached.append(name)
    print(json.dumps({"source_copy": TARGET.relative_to(ROOT).as_posix(), "attached": attached, "missing": missing}))
    manifest = json.loads((LOG / "gte_model_manifest.json").read_text(encoding="utf-8"))
    model_cache = ROOT / "reproduction/.runtime/hf_cache/sentence_transformers/thenlper_gte-base"
    for entry in manifest["files"]:
        source = ROOT / "reproduction/.runtime/gte-base" / entry["name"]
        target = model_cache / entry["name"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
        assert hashlib.sha256(target.read_bytes()).hexdigest() == entry["sha256"]
    print(json.dumps({"original_gte_cache_attached": True, "revision": manifest["revision"]}))


if __name__ == "__main__":
    main()
