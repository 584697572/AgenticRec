"""Execute a subprocess with a small allowlist environment, excluding credentials."""
import argparse
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cwd", default=str(ROOT))
    parser.add_argument("--online-model-download", action="store_true")
    parser.add_argument("--pythonpath", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    env = {k: os.environ[k] for k in ("SYSTEMROOT", "WINDIR", "TEMP", "TMP", "PATH", "COMSPEC") if k in os.environ}
    cache = ROOT / "reproduction/.runtime/hf_cache"
    profile = ROOT / "reproduction/.runtime/isolated_profile"
    profile.mkdir(parents=True, exist_ok=True)
    env.update(USERPROFILE=str(profile), LOCALAPPDATA=str(profile / "AppData/Local"),
               APPDATA=str(profile / "AppData/Roaming"), MPLCONFIGDIR=str(profile / "matplotlib"),
               GRADIO_ANALYTICS_ENABLED="False", TIKTOKEN_CACHE_DIR=str(profile / "tiktoken"))
    env.update(DOMAIN="movie", PYTHONUTF8="1", PYTHONIOENCODING="utf-8", HF_HOME=str(cache),
               SENTENCE_TRANSFORMERS_HOME=str(cache / "sentence_transformers"),
               HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")
    if args.pythonpath:
        pythonpath = args.pythonpath.resolve()
        if not pythonpath.is_dir() or not pythonpath.is_relative_to(ROOT):
            parser.error("pythonpath must be an existing directory inside the workspace")
        env["PYTHONPATH"] = str(pythonpath)
    if not args.online_model_download:
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    result = subprocess.run(command, cwd=args.cwd, env=env)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
