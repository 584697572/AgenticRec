"""Apply dependency-only pins to the separate compatibility copy and export exact diff."""
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "RecAI/InteRecAgent"
TARGET = ROOT / "reproduction/.runtime/InteRecAgent-compat"
LOG = ROOT / "reproduction/logs/a1_20261001"


def main():
    relative = "requirements.txt"
    original = (SOURCE / relative).read_text(encoding="utf-8")
    desired = (ROOT / "reproduction/requirements.legacy.in").read_text(encoding="utf-8")
    current = (TARGET / relative).read_text(encoding="utf-8")
    assert current in (original, desired), "Unexpected compatibility-copy edits; refusing overwrite"
    (TARGET / relative).write_text(desired, encoding="utf-8")
    original_lines = [line + "\n" for line in original.splitlines()]
    desired_lines = [line + "\n" for line in desired.splitlines()]
    diff_lines = list(difflib.unified_diff(original_lines, desired_lines,
                                          fromfile="a/InteRecAgent/requirements.txt", tofile="b/InteRecAgent/requirements.txt"))
    # difflib otherwise concatenates the old final line and the next '+' line.
    # Preserve the original missing final newline using Git's standard marker.
    if original and not original.endswith("\n"):
        last_line = original.splitlines()[-1] + "\n"
        final_removal = "-" + last_line
        if final_removal in diff_lines:
            position = diff_lines.index(final_removal)
            diff_lines.insert(position + 1, "\\ No newline at end of file\n")
        else:
            position = diff_lines.index(" " + last_line)
            diff_lines[position:position + 1] = [final_removal, "\\ No newline at end of file\n", "+" + last_line]
    diff = "".join(diff_lines)
    with (ROOT / "reproduction/legacy_compat.patch").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(diff)
    files = []
    for entry in json.loads((LOG / "compat_initial_copy.json").read_text(encoding="utf-8"))["files"]:
        path = entry["path"]
        source_hash = hashlib.sha256((SOURCE / path).read_bytes()).hexdigest()
        target_hash = hashlib.sha256((TARGET / path).read_bytes()).hexdigest()
        assert source_hash == entry["sha256"]
        assert path == relative or source_hash == target_hash, path
        files.append({"path": path, "original_sha256": source_hash, "compat_sha256": target_hash,
                      "changed": source_hash != target_hash})
    record = {"changed_files": [entry["path"] for entry in files if entry["changed"]],
              "python_source_unchanged": True, "files": files}
    (LOG / "compat_diff_manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"changed_files": record["changed_files"], "python_source_unchanged": True}))


if __name__ == "__main__":
    main()
