"""Rebuild the T19 model table from frozen per-user predictions."""

from pathlib import Path

from agenticrec.evaluation.benchmark import build_dry_run, write_partial_reports
from agenticrec.evaluation.model_benchmark import (
    export_predictions,
    score_predictions,
    write_scored_summary,
)


ROOT = Path(__file__).resolve().parents[2]


def main():
    prediction_path = ROOT / "artifacts/runs/t19/model_predictions.jsonl"
    prediction_manifest = prediction_path.with_suffix(
        prediction_path.suffix + ".manifest.json"
    )
    if prediction_path.exists() != prediction_manifest.exists():
        raise FileNotFoundError("prediction file and manifest must exist together")
    if not prediction_path.exists():
        export_predictions(prediction_path)
    summary = score_predictions(prediction_path)
    write_scored_summary(summary, ROOT / "artifacts/runs/t19/model_summary.json")
    plan = build_dry_run(ROOT / "AgenticRec/configs/benchmark.yaml")
    write_partial_reports(plan, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
