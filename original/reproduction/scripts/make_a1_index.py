"""Hash the current reviewable artifacts and local raw A1 logs (no model redistribution)."""
import datetime
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "reproduction/a1_evidence_index.json"


def main():
    git = ["git", "-c", "safe.directory=" + ROOT.as_posix(), "-C", str(ROOT)]
    output = subprocess.run(git + ["ls-files", "-z", "--cached", "--others", "--exclude-standard"], capture_output=True, check=True).stdout
    paths = {ROOT / part.decode("utf-8") for part in output.split(b"\0") if part}
    paths.update(path for path in (ROOT / "reproduction/logs/a1_20261001").rglob("*") if path.is_file())
    paths.discard(TARGET)
    records = []
    for path in sorted(paths):
        if path.is_file():
            data = path.read_bytes()
            relative = path.relative_to(ROOT).as_posix()
            blob = subprocess.run(git + ["show", ":" + relative], capture_output=True)
            records.append({"path": relative, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                            "git_index_sha256": hashlib.sha256(blob.stdout).hexdigest() if blob.returncode == 0 else None})
    result = {"recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "pre_commit_parent": subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip(),
              "scope": "workspace byte hashes plus staged Git blob hashes (text may normalize to LF); index excludes itself",
              "raw_logs_are_local_not_committed": True,
              "resource_hashes": json.loads((ROOT / "reproduction/native_runs/20261001/complete_movie_integrity.json").read_text())["full_movie_members"],
              "files": records}
    with TARGET.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"files_indexed": len(records), "index": TARGET.relative_to(ROOT).as_posix()}))


if __name__ == "__main__":
    main()
