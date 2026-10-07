"""T19 benchmark planning and paired statistics.

The dry-run path reads only the frozen public episode file and the historical
authorization ledger. It never imports evaluator-only targets or sends a
remote request.
"""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

from .episodes import load_public_episodes


ROOT = Path(__file__).resolve().parents[4]
SYSTEMS = ("U1", "F", "A", "O")
ABLATIONS = (
    "no_user_model",
    "no_collaborative",
    "no_explicit_preference_state",
    "always_agent",
    "no_replanning",
)
PREREGISTERED_SEEDS = (7, 42, 2026)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _workspace_path(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(label + " must be a nonempty workspace-relative path")
    path = (ROOT / value).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError(label + " must stay inside the workspace") from error
    return path


@dataclass(frozen=True)
class BenchmarkConfig:
    benchmark_id: str
    eval_manifest: str
    authorization_ledger: str
    split: str
    episodes: int
    model_seeds: tuple[int, ...]
    agent_seeds: tuple[int, ...]
    systems: tuple[str, ...]
    ablations: tuple[str, ...]
    max_replans: int
    bootstrap_samples: int
    confidence: float
    noninferiority_margin: float
    schema_version: int = 1

    @classmethod
    def from_file(cls, path):
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        expected = {
            "schema_version", "benchmark_id", "eval_manifest",
            "authorization_ledger", "split", "episodes", "model_seeds",
            "agent_seeds", "systems", "ablations", "max_replans",
            "bootstrap_samples", "confidence", "noninferiority_margin",
        }
        if type(value) is not dict or set(value) != expected:
            raise ValueError("benchmark config must contain the exact T19 fields")
        for name in ("model_seeds", "agent_seeds", "systems", "ablations"):
            if type(value[name]) is not list:
                raise ValueError(name + " must be a list")
            value[name] = tuple(value[name])
        return cls(**value)

    def __post_init__(self):
        if self.schema_version != 1 or type(self.schema_version) is not int:
            raise ValueError("schema_version must be integer 1")
        if not isinstance(self.benchmark_id, str) or not self.benchmark_id.strip():
            raise ValueError("benchmark_id must be nonempty")
        if self.split != "test":
            raise ValueError("T19 final benchmark must use the frozen test split")
        if type(self.episodes) is not int or self.episodes <= 0:
            raise ValueError("episodes must be a positive integer")
        if self.model_seeds != PREREGISTERED_SEEDS:
            raise ValueError("pre-registered model seeds must be [7, 42, 2026]")
        if self.agent_seeds != PREREGISTERED_SEEDS:
            raise ValueError("pre-registered agent seeds must be [7, 42, 2026]")
        if self.systems != SYSTEMS:
            raise ValueError("systems must be the pre-registered U1/F/A/O order")
        if self.ablations != ABLATIONS:
            raise ValueError("required ablations must use the pre-registered order")
        if self.max_replans != 1 or type(self.max_replans) is not int:
            raise ValueError("max_replans must be integer 1")
        if type(self.bootstrap_samples) is not int or self.bootstrap_samples < 1000:
            raise ValueError("bootstrap_samples must be an integer of at least 1000")
        if type(self.confidence) not in (int, float) or not 0 < self.confidence < 1:
            raise ValueError("confidence must be between zero and one")
        if (type(self.noninferiority_margin) not in (int, float)
                or not math.isfinite(self.noninferiority_margin)
                or self.noninferiority_margin >= 0):
            raise ValueError("noninferiority_margin must be finite and negative")


def _load_public_contract(config):
    manifest_path = _workspace_path(config.eval_manifest, "eval_manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != 1 or manifest.get("frozen") is not True
            or manifest.get("llm_simulator_enabled") is not False
            or manifest.get("human_review_status") != "VERIFIED"):
        raise ValueError("benchmark manifest is not frozen and human-review verified")
    split = manifest.get("splits", {}).get(config.split)
    if type(split) is not dict or split.get("episodes") != config.episodes:
        raise ValueError("configured episode count disagrees with frozen manifest")
    entry = manifest.get("files", {}).get(config.split + "_public")
    if type(entry) is not dict or entry.get("visibility") != "public":
        raise ValueError("frozen public episode entry is missing")
    public_path = _workspace_path(entry.get("path"), "public episode path")
    if not public_path.is_file() or _sha256(public_path) != entry.get("sha256"):
        raise ValueError("frozen public episode hash mismatch")
    episodes = load_public_episodes(public_path)
    if len(episodes) != config.episodes or len(episodes) != entry.get("rows"):
        raise ValueError("frozen public episode row count mismatch")
    if any(item.split != config.split for item in episodes):
        raise ValueError("public episode split mismatch")
    return manifest_path, manifest, public_path, episodes


def _load_authorization(config):
    path = _workspace_path(config.authorization_ledger, "authorization_ledger")
    ledger = json.loads(path.read_text(encoding="utf-8"))
    budget = ledger.get("budget")
    if type(budget) is not dict:
        raise ValueError("authorization ledger has no budget")
    used = ledger.get("remote_api_requests")
    cap = budget.get("api_request_cap")
    if (type(used) is not int or used < 0 or type(cap) is not int or cap < used):
        raise ValueError("authorization request ledger is invalid")
    if (budget.get("provider") != "deepseek"
            or budget.get("model_id") != "deepseek-flash"
            or budget.get("max_output_tokens") != 1024
            or budget.get("money_budget") != 3
            or budget.get("budget_unit") != "CNY"):
        raise ValueError("authorization ledger does not match the frozen T19 provider budget")
    return path, ledger, used, cap


def build_dry_run(config_path):
    """Return a request-ceiling plan without loading targets or calling an LLM."""
    config = BenchmarkConfig.from_file(config_path)
    manifest_path, manifest, public_path, episodes = _load_public_contract(config)
    ledger_path, ledger, used, cap = _load_authorization(config)

    seed_count = len(config.agent_seeds)
    all_turns = sum(item.max_turns for item in episodes)
    text_turns = sum(item.max_turns for item in episodes if item.layer == "text")
    replanning_multiplier = 1 + config.max_replans
    one_call_all_turns = all_turns * seed_count
    routed_ceiling = one_call_all_turns * replanning_multiplier
    breakdown = {
        "system:U1": one_call_all_turns,
        # F has no planner; free-text intent extraction remains visible in cost.
        "system:F": text_turns * seed_count,
        "system:A": routed_ceiling,
        "system:O": routed_ceiling,
        "ablation:no_user_model": routed_ceiling,
        "ablation:no_collaborative": routed_ceiling,
        "ablation:no_explicit_preference_state": routed_ceiling,
        # This ablation is exactly the already scheduled A run.
        "ablation:always_agent": 0,
        "ablation:no_replanning": one_call_all_turns,
    }
    required = sum(breakdown.values())
    per_seed_ceiling = {
        "U1": all_turns,
        "F": text_turns,
        "A": all_turns * replanning_multiplier,
        "O": all_turns * replanning_multiplier,
        "no_user_model": all_turns * replanning_multiplier,
        "no_collaborative": all_turns * replanning_multiplier,
        "no_explicit_preference_state": all_turns * replanning_multiplier,
        "no_replanning": all_turns,
    }
    run_matrix = [
        {
            "run_id": "t19-{}-seed{}".format(condition, seed),
            "condition": condition,
            "seed": seed,
            "request_ceiling": per_seed_ceiling[condition],
            "journal": "artifacts/runs/t19/live/{}/seed_{}.jsonl".format(
                condition, seed
            ),
        }
        for condition in (
            "U1", "F", "A", "O", "no_user_model", "no_collaborative",
            "no_explicit_preference_state", "no_replanning",
        )
        for seed in config.agent_seeds
    ]
    remaining = cap - used
    shortfall = max(0, required - remaining)
    return {
        "schema_version": 1,
        "benchmark_id": config.benchmark_id,
        "status": "READY" if not shortfall else "BLOCKED_AUTHORIZATION",
        "dry_run": True,
        "remote_api_requests": 0,
        "live_run_authorized": shortfall == 0,
        "test_private_loaded": False,
        "split": config.split,
        "episodes": len(episodes),
        "episode_turn_ceiling": all_turns,
        "text_turn_ceiling": text_turns,
        "model_seeds": list(config.model_seeds),
        "agent_seeds": list(config.agent_seeds),
        "systems": list(config.systems),
        "ablations": list(config.ablations),
        "request_ceiling_by_run": breakdown,
        "run_matrix": run_matrix,
        "run_aliases": {"always_agent": "A"},
        "required_new_requests_ceiling": required,
        "authorization": {
            "provider": ledger["budget"]["provider"],
            "model_id": ledger["budget"]["model_id"],
            "max_output_tokens": ledger["budget"]["max_output_tokens"],
            "request_cap_total": cap,
            "requests_used": used,
            "requests_remaining": remaining,
            "request_shortfall": shortfall,
            "money_budget_cny_total": ledger["budget"]["money_budget"],
            "money_budget_sufficiency": "NOT VERIFIED",
        },
        "frozen_inputs": {
            "manifest_sha256": _sha256(manifest_path),
            "public_episodes_sha256": _sha256(public_path),
            "authorization_ledger_sha256": _sha256(ledger_path),
            "private_entry_sha256_recorded": manifest["files"][config.split + "_private"]["sha256"],
        },
        "statistics": {
            "bootstrap_samples": config.bootstrap_samples,
            "confidence": config.confidence,
            "noninferiority_margin": config.noninferiority_margin,
        },
        "u0": "NOT_RUN_RESOURCE_SEMANTICS_UNVERIFIED",
        "note": (
            "The ceiling reserves every public max_turn and one replan for A/O and "
            "router ablations. Actual routed calls may be lower; authorization is "
            "checked against the ceiling so a paid batch cannot overrun mid-run."
        ),
    }


def write_dry_run_artifact(report):
    """Persist the machine-readable plan under the ignored run-artifact tree."""
    if type(report) is not dict or report.get("dry_run") is not True:
        raise ValueError("a dry-run report is required")
    path = ROOT / "artifacts/runs/t19/dry_run_plan.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)
    return path


def build_model_summary(seeds=PREREGISTERED_SEEDS):
    """Aggregate the immutable per-seed T08/T09 reports without selecting a seed."""
    seeds = tuple(seeds)
    if seeds != PREREGISTERED_SEEDS:
        raise ValueError("model summary requires the pre-registered seeds")
    report_dir = ROOT / "reports/rec_baselines"
    methods = ("Random", "Popularity", "BPR-MF", "LightGCN")
    metrics = ("recall", "ndcg", "mrr", "hit")
    values = {method: {metric: [] for metric in metrics} for method in methods}
    source_runs = []
    identity = None
    per_seed = []
    for seed in seeds:
        bpr_path = report_dir / ("seed_" + str(seed) + ".json")
        light_path = report_dir / ("lightgcn_seed_" + str(seed) + ".json")
        bpr = json.loads(bpr_path.read_text(encoding="utf-8"))
        light = json.loads(light_path.read_text(encoding="utf-8"))
        if (bpr.get("status") != "PASS" or bpr.get("task") != "T08"
                or bpr.get("seed") != seed or bpr.get("candidate_items") != 3469
                or bpr.get("test_used_for_selection") is not False):
            raise ValueError("invalid T08 model report for seed " + str(seed))
        if (light.get("status") != "PASS" or light.get("task") != "T09"
                or light.get("seed") != seed
                or light.get("test_used_for_training_or_selection") is not False
                or light.get("baseline_report_sha256") != _sha256(bpr_path)):
            raise ValueError("invalid T09 model report for seed " + str(seed))
        current_identity = (
            bpr["identity"]["id_map_sha256"],
            bpr["identity"]["split_sha256"],
            bpr["cohort_sha256"]["test"],
        )
        if identity is None:
            identity = current_identity
        elif current_identity != identity:
            raise ValueError("model seeds use different frozen evaluation identities")
        seed_metrics = {}
        for method in methods:
            row = (light["test_metrics"] if method == "LightGCN"
                   else bpr["metrics"]["test"][method])
            if row.get("failures") != 0 or row.get("users_evaluated") != 954:
                raise ValueError("model report has an incomplete test denominator")
            seed_metrics[method] = {metric: float(row[metric]) for metric in metrics}
            for metric in metrics:
                values[method][metric].append(float(row[metric]))
        per_seed.append({
            "seed": seed,
            "selected_lightgcn_layers": light["selection"]["selected_layers"],
            "metrics": seed_metrics,
        })
        source_runs.append({
            "seed": seed,
            "bpr_report": bpr_path.relative_to(ROOT).as_posix(),
            "bpr_report_sha256": _sha256(bpr_path),
            "lightgcn_report": light_path.relative_to(ROOT).as_posix(),
            "lightgcn_report_sha256": _sha256(light_path),
            "bpr_checkpoint_sha256": bpr["checkpoint_sha256"],
            "lightgcn_checkpoint_sha256": light["trials"][
                str(light["selection"]["selected_layers"])
            ]["checkpoint_sha256"],
        })
    aggregate = {}
    for method in methods:
        aggregate[method] = {
            metric: {
                "mean": statistics.fmean(values[method][metric]),
                "sample_std": statistics.stdev(values[method][metric]),
            }
            for metric in metrics
        }
    deltas = {
        metric: [right - left for left, right in zip(
            values["BPR-MF"][metric], values["LightGCN"][metric]
        )]
        for metric in metrics
    }
    return {
        "schema_version": 1,
        "status": "PARTIAL",
        "scope": "T19 recommendation-model table",
        "seeds": list(seeds),
        "candidate_universe": "all 3469 mapped MovieLens1M items minus train-seen items",
        "test_users_total": 5351,
        "test_users_evaluated": 954,
        "test_users_without_relevance": 4397,
        "metrics_at_k": 10,
        "per_seed": per_seed,
        "aggregate": aggregate,
        "lightgcn_minus_bpr_by_seed": deltas,
        "negative_gain_present": any(value < 0 for value in deltas["ndcg"]),
        "episode_level_paired_ci": "NOT EVALUATED_RAW_PREDICTIONS_NOT_EXPORTED",
        "significance_claim": "NOT MADE",
        "source_runs": source_runs,
        "frozen_identity": {
            "id_map_sha256": identity[0],
            "split_sha256": identity[1],
            "test_cohort_sha256": identity[2],
        },
        "remote_api_requests": 0,
    }


def write_model_summary(summary):
    if (type(summary) is not dict
            or not str(summary.get("scope", "")).startswith("T19 recommendation-model table")):
        raise ValueError("a T19 model summary is required")
    path = ROOT / "artifacts/runs/t19/model_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    temporary.replace(path)
    return path


def write_partial_reports(plan, model_summary):
    """Write T19 reports with explicit NOT EVALUATED cells for blocked live work."""
    report_dir = ROOT / "AgenticRec/reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    table = [
        "| Method | Recall@10 | NDCG@10 | MRR@10 | HitRate@10 |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in ("Random", "Popularity", "BPR-MF", "LightGCN"):
        row = model_summary["aggregate"][method]
        table.append("| {} | {:.6f} +/- {:.6f} | {:.6f} +/- {:.6f} | {:.6f} +/- {:.6f} | {:.6f} +/- {:.6f} |".format(
            method,
            row["recall"]["mean"], row["recall"]["sample_std"],
            row["ndcg"]["mean"], row["ndcg"]["sample_std"],
            row["mrr"]["mean"], row["mrr"]["sample_std"],
            row["hit"]["mean"], row["hit"]["sample_std"],
        ))
    paired = model_summary.get("episode_level_paired_ci")
    if isinstance(paired, dict):
        ndcg_ci = paired["ndcg"]
        paired_text = (
            "Paired user bootstrap for LightGCN minus BPR-MF NDCG@10: "
            "mean {:+.6f}, 95% CI [{:+.6f}, {:+.6f}] over {:,} users, "
            "averaging the three pre-registered seeds per user.".format(
                ndcg_ci["mean_difference"], ndcg_ci["ci_lower"],
                ndcg_ci["ci_upper"], ndcg_ci["paired_count"],
            )
        )
        evidence_text = (
            "The table was regenerated from frozen per-user Top-10 predictions; "
            "the artifact records their SHA256 and evaluator denominator."
        )
        source_text = (
            "Generated from frozen per-user predictions for seeds 7, 42, and 2026. "
            "Values are mean +/- sample standard deviation across seeds."
        )
        delta_text = (
            "LightGCN minus BPR-MF NDCG@10 by seed: "
            + json.dumps(model_summary["lightgcn_minus_bpr_by_seed"]["ndcg"])
            + ". The mean is negative; the paired interval below tests that loss. "
            "No LightGCN improvement claim is made."
        )
    else:
        paired_text = "Episode-level paired confidence intervals: **NOT EVALUATED**."
        evidence_text = (
            "Raw per-user predictions still need export and independent regeneration "
            "before this table can satisfy final T19 acceptance."
        )
        source_text = (
            "Generated from the three immutable T08/T09 seed reports. Values are "
            "mean +/- sample standard deviation across seeds 7, 42, and 2026."
        )
        delta_text = (
            "LightGCN minus BPR-MF NDCG@10 by seed: "
            + json.dumps(model_summary["lightgcn_minus_bpr_by_seed"]["ndcg"])
            + ". The mean is negative and no significance claim is made."
        )
    benchmark_text = "\n".join([
        "# T19 Benchmark (IN PROGRESS)", "",
        source_text, "",
        *table, "",
        "Candidate protocol: full 3,469-item mapped MovieLens1M universe with train-seen filtering; 5,351 total users, 954 users with test relevance, and 4,397 reported separately without relevance.", "",
        delta_text, "",
        paired_text + " " + evidence_text, "",
        "## Interactive systems", "",
        "| System | Strict Success | Constraint Precision | Fill@K | Requests | Latency p50/p95 |",
        "|---|---:|---:|---:|---:|---:|",
        "| U1 | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED |",
        "| F | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED |",
        "| A | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED |",
        "| O | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED |", "",
        "The paid run is blocked before request 1: the audited ceiling is {:,} new requests, {:,} remain authorized, and the shortfall is {:,}.".format(
            plan["required_new_requests_ceiling"],
            plan["authorization"]["requests_remaining"],
            plan["authorization"]["request_shortfall"],
        ), "",
        "Machine-readable local evidence: `artifacts/runs/t19/model_summary.json` and `artifacts/runs/t19/dry_run_plan.json`.", "",
    ])
    ablation_text = "\n".join([
        "# T19 Ablations (NOT EVALUATED)", "",
        "The frozen ablations are `no_user_model`, `no_collaborative`, `no_explicit_preference_state`, `always_agent`, and `no_replanning`.", "",
        "No ablation metric has been produced. The dry-run reserves the same public episodes, seeds, DeepSeek model, 1,024-token output cap, tools, and hard constraints for each condition. `always_agent` reuses A rather than spending a duplicate run.", "",
        "Live execution is **BLOCKED_AUTHORIZATION** with a request-cap shortfall of {:,}. No row may be filled until raw attempts are written and rescored by the evaluator-only targets.".format(plan["authorization"]["request_shortfall"]), "",
    ])
    failure_text = "\n".join([
        "# T19 Failure Analysis (IN PROGRESS)", "",
        "Recommendation-model finding: LightGCN underperforms BPR-MF on NDCG@10 for seeds 7 and 2026, and is higher only for seed 42. The three-seed mean is therefore a negative gain; it is retained rather than omitted.", "",
        "Interactive failure categories are frozen as invalid item, constraint violation, short fill, wrong status, preference-update mismatch, timeout/429/tool error, fallback, abstention, and missing attempt. Counts are **NOT EVALUATED** because the paid U1/F/A/O and ablation attempts have not run.", "",
        "Attribution to recommendation model, hard constraints, or routing is **NOT EVALUATED**. It will require paired episode traces under identical conditions; no causal claim is made from the model-only table.", "",
    ])
    outputs = {
        "benchmark.md": benchmark_text,
        "ablations.md": ablation_text,
        "failure_analysis.md": failure_text,
    }
    for name, content in outputs.items():
        (report_dir / name).write_text(content, encoding="utf-8", newline="\n")
    return tuple(report_dir / name for name in outputs)


def _percentile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def paired_bootstrap_ci(baseline, treatment, *, samples, confidence, seed):
    """Paired episode bootstrap for treatment-minus-baseline differences."""
    baseline = tuple(baseline)
    treatment = tuple(treatment)
    if not baseline or len(baseline) != len(treatment):
        raise ValueError("paired samples must have the same nonzero length")
    if any(type(value) not in (int, float) or not math.isfinite(value)
           for value in baseline + treatment):
        raise ValueError("paired values must be finite numbers")
    if type(samples) is not int or samples <= 0:
        raise ValueError("samples must be a positive integer")
    if type(confidence) not in (int, float) or not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    differences = tuple(float(right - left) for left, right in zip(baseline, treatment))
    rng = random.Random(seed)
    bootstrapped = []
    for _ in range(samples):
        bootstrapped.append(statistics.fmean(
            differences[rng.randrange(len(differences))]
            for _ in range(len(differences))
        ))
    alpha = (1 - float(confidence)) / 2
    return {
        "paired_count": len(differences),
        "mean_difference": statistics.fmean(differences),
        "ci_lower": _percentile(bootstrapped, alpha),
        "ci_upper": _percentile(bootstrapped, 1 - alpha),
        "confidence": float(confidence),
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
    }
