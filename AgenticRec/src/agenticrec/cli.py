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
    recommend = commands.add_parser("recommend")
    recommend.add_argument("--request", type=Path, required=True,
                           help="strict structured JSON; natural language parsing is separate")
    recommend.add_argument("--model", choices=("bpr", "lightgcn"), default="lightgcn")
    recommend.add_argument("--seed", type=int, default=42)
    benchmark = commands.add_parser("benchmark")
    benchmark.add_argument("--config", type=Path, required=True)
    benchmark.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "benchmark":
        from .evaluation.benchmark import (
            ROOT as WORKSPACE_ROOT,
            _sha256,
            build_dry_run,
            build_model_summary,
            write_dry_run_artifact,
            write_model_summary,
            write_partial_reports,
        )
        try:
            report = build_dry_run(args.config)
        except (OSError, json.JSONDecodeError, ValueError, TypeError) as error:
            report = {
                "status": "INVALID_BENCHMARK_CONFIG",
                "reason": str(error),
                "remote_api_requests": 0,
                "live_run_authorized": False,
            }
            print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
            return 2
        if not args.dry_run:
            report["planning_dry_run_completed"] = True
            report["dry_run"] = False
            report["live_run_authorized"] = False
            if report["status"] == "BLOCKED_AUTHORIZATION":
                report["reason"] = (
                    "paid-call permission, remaining requests and conservative money "
                    "budget must all cover the complete audited batch"
                )
                print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
                return 3
            from .evaluation.live_benchmark import run_batch
            try:
                result = run_batch(args.config, live=True)
            except (OSError, ValueError, RuntimeError) as error:
                print(json.dumps({"status": "BLOCKED_BATCH", "error_type": type(error).__name__,
                                  "reason": str(error)}, ensure_ascii=True))
                return 3
            from .evaluation.system_reports import generate_system_reports
            generate_system_reports(args.config, result["batch_dir"])
            print(json.dumps(result, ensure_ascii=True, sort_keys=True))
            return 0
        write_dry_run_artifact(report)
        raw_predictions = WORKSPACE_ROOT / "artifacts/runs/t19/model_predictions.jsonl"
        saved_summary = WORKSPACE_ROOT / "artifacts/runs/t19/model_summary.json"
        model_summary = None
        if raw_predictions.is_file() and saved_summary.is_file():
            candidate = json.loads(saved_summary.read_text(encoding="utf-8"))
            raw_evidence = candidate.get("raw_predictions", {})
            if (candidate.get("status") == "PASS"
                    and raw_evidence.get("sha256") == _sha256(raw_predictions)):
                model_summary = candidate
        if model_summary is None:
            try:
                model_summary = build_model_summary()
            except FileNotFoundError:
                model_summary = None
        if model_summary is not None:
            write_model_summary(model_summary)
            # Dry-run must not overwrite completed reports from a real batch.
            if not (WORKSPACE_ROOT / "artifacts/runs/t19/live_20261010/summary.json").is_file():
                write_partial_reports(report, model_summary)
            report["offline_model_summary"] = "WRITTEN"
        print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
        return 0
    if args.command == "train":
        if json.loads(args.config.read_text(encoding="utf-8")).get("model") == "lightgcn":
            from .lightgcn_training import train
        else:
            from .training import train
        train(args.config, args.seed)
        return 0
    if args.command == "recommend":
        from .pipeline import FixedRecommendationPipeline, FixedRequest
        try:
            request = FixedRequest.from_json_file(args.request)
        except (OSError, json.JSONDecodeError, ValueError, TypeError) as error:
            report = {"status": "INVALID_REQUEST", "reason": str(error),
                      "input_mode": "structured_json", "planner": "not_invoked",
                      "llm_requests": 0, "remote_api_requests": 0}
            print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
            return 2
        try:
            report = FixedRecommendationPipeline.from_frozen(args.model, args.seed).recommend(request).to_dict()
        except (OSError, ValueError, RuntimeError) as error:
            report = {"status": "PIPELINE_ERROR", "reason": str(error),
                      "input_mode": "structured_json", "planner": "not_invoked",
                      "llm_requests": 0, "remote_api_requests": 0}
            print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
            return 3
        print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
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
