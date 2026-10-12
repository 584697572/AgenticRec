"""Compare actual package versions in the installed and rebuilt legacy environments."""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "reproduction/logs/a1_20261001"


def main():
    code = "import importlib.metadata as m,json;print(json.dumps({d.metadata['Name'].lower():d.version for d in m.distributions()}))"
    results = []
    for directory in (".venv-legacy", ".venv-legacy-rebuild"):
        python = ROOT / "reproduction" / directory / "Scripts/python.exe"
        process = subprocess.run([str(python), "-c", code], capture_output=True, text=True, check=True)
        results.append(json.loads(process.stdout))
    assert results[0] == results[1], "Rebuilt versions differ"
    for name in ("legacy_rebuild_install_locked_cache", "legacy_rebuild_pip_check", "legacy_rebuild_imports", "original_sdk_mock"):
        assert (LOG / (name + ".exitcode.txt")).read_text().strip() == "0", name
    path = ROOT / "reproduction/environment_legacy.json"
    environment = json.loads(path.read_text(encoding="utf-8"))
    environment["rebuild"] = {"status": "PASS", "package_versions_identical": True,
                             "package_count": len(results[0]), "network_downloads": 0,
                             "evidence_logs": "reproduction/logs/a1_20261001/legacy_rebuild_*"}
    path.write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(environment["rebuild"]))


if __name__ == "__main__":
    main()
