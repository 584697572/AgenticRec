"""Prepare only the source/settings needed by U1; never fetch original resources.

U1 supplies its own mapped Gallery/scorer to original ToolBox/Buffer/Map. The
settings paths below are explicitly unused, not claims that resources exist.
Existing compatibility copies (including A1 resources) are never overwritten.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
PINNED_FILES = {
    "llm4crs/agent_plan_first_openai.py": "14f6598930e31ff7d78f2235b0e76680ca86985de875d3ae6fa00c59cc6c8bb3",
    "llm4crs/buffer/base.py": "0a5739335a98071132470ef42861585aafd2d7eb0e623bedc61b6c9d8ba5682f",
    "llm4crs/mapper/map_tool.py": "804fae0dadc9dc563b0cd1193fad1667f4146fe703def5c32466b1be86b20b28",
}


def prepare(root=ROOT):
    root = Path(root)
    source = root / "RecAI/InteRecAgent"
    target = root / "reproduction/.runtime/InteRecAgent-compat"
    for name, expected in PINNED_FILES.items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise ValueError("upstream source differs from fixed revision: " + name)
    created = not target.exists()
    if created:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target, ignore=shutil.ignore_patterns(
            "resources", "__pycache__", ".git", "UniRec", ".pytest_cache"))
        settings = target / "resources/movie/settings.json"
        settings.parent.mkdir(parents=True)
        with settings.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump({
                "GAME_INFO_FILE": "UNUSED_BY_U1/items",
                "TABLE_COL_DESC_FILE": "UNUSED_BY_U1/columns",
                "MODEL_CKPT_FILE": "UNUSED_BY_U1/checkpoint",
                "ITEM_SIM_FILE": "UNUSED_BY_U1/similarity",
                "USE_COLS": ["id", "title"], "CATEGORICAL_COLS": [],
            }, stream, indent=2)
            stream.write("\n")
    for name, expected in PINNED_FILES.items():
        if hashlib.sha256((target / name).read_bytes()).hexdigest() != expected:
            raise ValueError("existing U1 component differs: " + name)
    if not (target / "resources/movie/settings.json").is_file():
        raise FileNotFoundError("existing compat copy has no settings; refusing mutation")
    return {"status": "PASS", "scope": "U1_method_source_only_not_A1_resources",
            "created": created, "original_resources_downloaded": 0,
            "upstream_commit": "0959ecb05b0794748426e73e6efc1b6b35ec433d"}


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    print(json.dumps(prepare(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
