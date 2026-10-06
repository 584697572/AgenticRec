"""Implemented T05 commands only. Unknown commands/arguments fail explicitly."""
import argparse
from dataclasses import asdict
import importlib.util
import json
import platform
from pathlib import Path

from .config import ExperimentConfig, load_config
from .testing import FakeLLM


def main(argv=None):
    parser = argparse.ArgumentParser(prog="agenticrec")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor")
    doctor.add_argument("--offline", action="store_true", required=True)
    doctor.add_argument("--config", type=Path)
    commands.add_parser("fixture")
    training = commands.add_parser("train")
    training.add_argument("--config", type=Path, required=True)
    training.add_argument("--seed", type=int, required=True)
    args = parser.parse_args(argv)
    if args.command == "train":
        if json.loads(args.config.read_text(encoding="utf-8")).get("model") == "lightgcn":
            from .lightgcn_training import train
        else:
            from .training import train
        train(args.config, args.seed)
        return 0
    if args.command == "doctor":
        try:
            config = load_config(args.config) if args.config else ExperimentConfig()
        except (OSError, ValueError, TypeError):
            parser.error("invalid configuration; check schema and fields (contents are not logged)")
        report = {"status": "PASS", "scope": "T05_offline_foundation",
            "python": platform.python_version(), "system": platform.system(),
            "torch_installed": importlib.util.find_spec("torch") is not None,
            "gpu_runtime": "NOT CHECKED", "data_training_evaluation": "NOT VERIFIED",
            "config": asdict(config), "remote_api_requests": 0,
            "note": "offline check never grants live API authorization"}
    else:
        reply = FakeLLM(["fixture response"]).chat([{"role": "user", "content": "offline fixture"}])
        report = {"status": "PASS", "scope": "FakeLLM_fixture_only", "reply": asdict(reply),
                  "recommendation_pipeline": "NOT IMPLEMENTED"}
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
