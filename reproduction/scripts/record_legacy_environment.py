"""Freeze the installed isolated environment; retain its native package check."""
import datetime
import importlib.metadata as metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    check = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
    (ROOT / "reproduction/requirements.legacy.lock.txt").write_text(freeze.stdout, encoding="utf-8")
    packages = sorted(({"name": dist.metadata["Name"], "version": dist.version} for dist in metadata.distributions()),
                      key=lambda entry: entry["name"].lower())
    record = {"recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "python": platform.python_version(), "platform": platform.platform(),
              "executable": str(Path(sys.executable).relative_to(ROOT)),
              "environment": "isolated_legacy", "system_python_modified": False,
              "torch_variant": "1.13.1+cpu", "pipeline": "movie / plan_first / no_langchain / zero_demo",
              "omitted_original_requirement": {"chromadb==0.4.7": "chroma-hnswlib MSVC build failed; dynamic demos not exercised"},
              "pip_check": {"exit_code": check.returncode, "stdout": check.stdout, "stderr": check.stderr},
              "rebuild": "NOT VERIFIED", "packages": packages}
    (ROOT / "reproduction/environment_legacy.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"python": record["python"], "packages": len(packages), "pip_check_exit_code": check.returncode}))
    return check.returncode


if __name__ == "__main__":
    raise SystemExit(main())
