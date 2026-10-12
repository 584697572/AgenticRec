"""Prepare the two-part checkout without changing measured model source or records.

Windows directory junctions and file hardlinks need no administrator rights. They
are local compatibility views, never a third implementation or public data copy.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
UPSTREAM_COMMIT = "0959ecb05b0794748426e73e6efc1b6b35ec433d"


def inside(root, path):
    root, path = Path(root).resolve(), Path(path).absolute()
    if not path.is_relative_to(root) or not path.resolve().is_relative_to(root):
        raise ValueError("Compatibility path outside checkout")
    return path


def hide_link(path):
    if os.name == "nt" and path.is_dir():
        # Change the link's attributes; canonical data/code stays visible.
        flags = ctypes.windll.kernel32.GetFileAttributesW(str(path))
        if flags != 0xFFFFFFFF:
            ctypes.windll.kernel32.SetFileAttributesW(str(path), flags | 2)


def link(root, relative, target_relative, *, hidden=True):
    destination = inside(root, Path(root) / relative)
    target = inside(root, Path(root) / target_relative)
    if not target.exists():
        return {"path": relative, "status": "target_absent"}
    if destination.exists() or destination.is_symlink():
        if not os.path.samefile(destination, target):
            raise ValueError("Refusing to replace existing path: " + relative)
        return {"path": relative, "status": "already_prepared"}
    destination.parent.mkdir(parents=True, exist_ok=True)
    if target.is_dir():
        if os.name == "nt":
            # Literal PowerShell arguments; no cross-shell filesystem operations.
            quote = lambda value: "'" + str(value).replace("'", "''") + "'"
            command = ("$ErrorActionPreference = 'Stop'; New-Item -ItemType Junction "
                       "-Path " + quote(destination) + " -Target " + quote(target)
                       + " | Out-Null")
            subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
                           check=True, capture_output=True)
        else:
            destination.symlink_to(os.path.relpath(target, destination.parent), target_is_directory=True)
    elif os.name == "nt":
        os.link(target, destination)
    else:
        destination.symlink_to(os.path.relpath(target, destination.parent))
    if hidden:
        hide_link(destination)
    return {"path": relative, "status": "created"}


def pairs(root):
    root = Path(root)
    fixed = {
        "RecAI": "original/RecAI", "reproduction": "original/reproduction",
        "AgenticRec": "optimized/AgenticRec", "reports": "optimized/reports",
        "data": "optimized/data", "artifacts": "optimized/artifacts",
        "optimized/RecAI": "original/RecAI",
        "optimized/reproduction": "original/reproduction",
        "original/AgenticRec": "optimized/AgenticRec",
        "original/artifacts": "optimized/artifacts",
    }
    history = "optimized/history/workspace"
    for name in ("STATUS.md", "DECISIONS.md", "TASKS.yaml", "AgenticRec_完整复现与优化执行规范.md"):
        for part in ("original", "optimized"):
            fixed[part + "/" + name] = history + "/" + name
    for part in ("original", "optimized"):
        fixed[part + "/LICENSE.txt"] = "LICENSE.txt"
    # Canonical original resources and independent experiment data have different
    # owners. Mirror only missing entries to keep historical relative paths valid.
    for suffix in ("data", "data/raw", "data/eval_private", "reports"):
        for owner, other in (("original", "optimized"), ("optimized", "original")):
            source = root / owner / suffix
            destination = root / other / suffix
            if not source.is_dir():
                continue
            for entry in sorted(source.iterdir()):
                if entry.is_symlink() or (os.name == "nt" and entry.stat().st_file_attributes & 0x400):
                    continue
                # Windows hardlinks are distinguished by canonical paths in the
                # migration manifest, rather than guessing from their names.
                manifest = root / "optimized/history/layout_20261012.json"
                owned = json.loads(manifest.read_text(encoding="utf-8"))["canonical_report_paths"] if manifest.exists() else []
                if suffix == "reports" and entry.is_file() and owned and entry.relative_to(root).as_posix() not in owned:
                    continue
                other_path = destination / entry.name
                if other_path.exists() and os.path.samefile(other_path, entry):
                    continue
                if other_path.exists():
                    if suffix == "data" and entry.name in ("raw", "eval_private"):
                        continue
                    raise ValueError("Conflicting compatibility entry: " + str(other_path.relative_to(root)))
                fixed[other_path.relative_to(root).as_posix()] = entry.relative_to(root).as_posix()
    return fixed


def repair_editable_paths(root):
    root = Path(root)
    changed = 0
    replacements = {
        str(root / "AgenticRec" / "src"): str(root / "optimized/AgenticRec/src"),
        (root / "AgenticRec" / "src").as_posix(): (root / "optimized/AgenticRec/src").as_posix(),
    }
    package = root / "optimized/AgenticRec"
    for environment in package.glob(".venv*"):
        for site in (environment / "Lib/site-packages",):
            for path in site.glob("__editable__*.pth"):
                before = path.read_text(encoding="utf-8")
                after = before
                for old, new in replacements.items():
                    after = after.replace(old, new)
                if after != before:
                    path.write_text(after, encoding="utf-8")
                    changed += 1
    return changed


def prepare(root=ROOT):
    root = Path(root).resolve()
    for part in ("original", "optimized"):
        if not (root / part).is_dir():
            raise ValueError("Missing canonical project part: " + part)
    for part in ("original", "optimized"):
        for suffix in ("data/raw", "data/eval_private", "reports"):
            (root / part / suffix).mkdir(parents=True, exist_ok=True)
    (root / "optimized/artifacts").mkdir(parents=True, exist_ok=True)
    # Compute the overlay before creating links, to avoid following new aliases.
    records = [link(root, name, target) for name, target in pairs(root).items()]
    return {"status": "PREPARED", "canonical_parts": ["original", "optimized"],
            "compatibility_links": records, "editable_paths_repaired": repair_editable_paths(root),
            "downloads": 0, "remote_api_requests": 0}


def status(root=ROOT):
    root = Path(root)
    paths = {
        "original_source": "original/RecAI/InteRecAgent/app.py",
        "original_environment": "original/reproduction/.venv-legacy/Scripts/python.exe",
        "optimized_source": "optimized/AgenticRec/src/agenticrec/cli.py",
        "optimized_environment": "optimized/AgenticRec/.venv-t20/Scripts/python.exe",
        "original_resources": "original/data/raw/upstream_movie/SASRec-SASRec.pth",
        "trained_models": "optimized/artifacts/models",
    }
    return {"canonical_parts": ["original", "optimized"], "upstream_commit": UPSTREAM_COMMIT,
            "available": {name: (root / path).exists() for name, path in paths.items()},
            "remote_api_requests": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "status"))
    args = parser.parse_args()
    try:
        result = prepare() if args.command == "prepare" else status()
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(json.dumps({"status": "ERROR", "reason": str(error)}, ensure_ascii=True))
        return 1
    if args.command == "prepare":
        records = result.pop("compatibility_links")
        result["created_links"] = sum(row["status"] == "created" for row in records)
        result["existing_links"] = sum(row["status"] == "already_prepared" for row in records)
        result["absent_optional_targets"] = [row["path"] for row in records if row["status"] == "target_absent"]
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
