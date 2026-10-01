"""Import the full original modules in legacy Python; record actual failures."""
import importlib
import importlib.metadata
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "reproduction/.runtime/InteRecAgent-compat"
LOG = ROOT / "reproduction/logs/a1_20261001"
sys.path.insert(0, str(SOURCE))


def main():
    records = []
    for name in ("numpy", "torch", "pandas", "faiss", "sentence_transformers", "gradio",
                 "unirec.utils.general", "llm4crs.agent_plan_first_openai"):
        try:
            module = importlib.import_module(name)
            records.append({"module": name, "status": "PASS", "version": getattr(module, "__version__", None)})
        except Exception as error:
            records.append({"module": name, "status": "FAIL", "exception": type(error).__name__, "message": str(error)})
            traceback.print_exc()
    result = {"python": sys.version.split()[0], "records": records, "successful": all(r["status"] == "PASS" for r in records)}
    (LOG / "native_import_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["successful"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
