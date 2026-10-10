"""Recompute T19 system/ablation/failure reports from immutable raw attempts."""

from dataclasses import asdict
import hashlib
import json
import statistics

from .benchmark import ROOT, PREREGISTERED_SEEDS, paired_bootstrap_ci, _sha256
from .episodes import evaluate_episodes, load_public_episodes, load_private_episodes, score_episode
from .live_benchmark import canonical, write_json
from .runner import RunSpec, load_completed_attempts
from ..ranking.constraints import Catalog


def verify_provider_evidence(path, attempts):
    """Reconcile the independent transport journal, including failed attempts."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    pending, finished, by_episode = {}, set(), {}
    failures = 0
    for row in rows:
        request_id = row.get("request_id")
        if row.get("event") == "STARTED":
            if request_id in pending or request_id in finished:
                raise ValueError("duplicate provider request identity")
            pending[request_id] = row
        elif row.get("event") in {"COMPLETED", "FAILED"}:
            if request_id not in pending:
                raise ValueError("provider terminal event has no request intent")
            start = pending.pop(request_id)
            finished.add(request_id)
            reply = row.get("reply")
            if row["event"] == "COMPLETED":
                if not isinstance(reply, dict) or reply.get("is_fixture") is not False:
                    raise ValueError("fixture delivery cannot be used for live metrics")
                if reply.get("remote_api_requests") != 1:
                    raise ValueError("provider delivery request count differs from zero-retry contract")
            else:
                failures += 1
            by_episode.setdefault(start["episode_id"], []).append(reply)
        else:
            raise ValueError("unknown provider audit event")
    if pending:
        raise ValueError("unresolved provider intent cannot generate a completed report")
    if set(by_episode) - {attempt.episode_id for attempt in attempts}:
        raise ValueError("provider journal contains an unknown episode")
    for attempt in attempts:
        replies = by_episode.get(attempt.episode_id, [])
        if len(replies) != attempt.request_count:
            raise ValueError("provider journal and episode request count differ")
        if any(reply is None or reply.get("usage") is None for reply in replies):
            expected = (None, None)
        else:
            expected = (sum(reply["usage"]["prompt_tokens"] for reply in replies),
                        sum(reply["usage"]["completion_tokens"] for reply in replies))
        if (attempt.input_tokens, attempt.output_tokens) != expected:
            raise ValueError("provider journal and episode token usage differ")
    return {"requests": len(finished), "failed_requests": failures}


def failure_categories(public, private, attempt, score):
    categories = []
    if score.invalid_item_count:
        categories.append("invalid_item")
    if score.constraint_violation_count:
        categories.append("constraint_violation")
    if private.feasible and score.legal_output_count < private.required_count:
        categories.append("short_fill")
    if attempt.status not in private.expected_statuses:
        categories.append("wrong_status")
    if score.preference_update_accuracy == 0:
        categories.append("preference_update_mismatch")
    if attempt.status in {"TOOL_FAILED", "LLM_FAILED", "PLANNER_FAILED", "BUDGET_EXHAUSTED"}:
        categories.append("runtime_failure")
    if attempt.status in {"INVALID_PLAN", "INVALID_PARSE", "INVALID_FINAL_RESULT"}:
        categories.append("parse_or_plan_failure")
    if private.feasible and score.legal_output_count and not (set(attempt.item_ids) & set(private.acceptable_item_ids)):
        categories.append("acceptance_miss")
    if private.expected_fallback_kind and attempt.fallback_kind != private.expected_fallback_kind:
        categories.append("fallback_mismatch")
    if not score.strict_success and not categories:
        categories.append("other_strict_failure")
    return categories


def _aggregate(rows):
    return {name: {"mean": statistics.fmean(row[name] for row in rows),
                   "sample_std": statistics.stdev(row[name] for row in rows)}
            for name in ("strict_success", "constraint_precision", "fill_at_k", "requests", "tool_calls",
                         "latency_p50_ms", "latency_p95_ms", "input_tokens", "output_tokens", "fallback_rate",
                         "planner_calls", "parser_calls", "replans")}


def generate_system_reports(config_path, batch_dir):
    batch_dir = ROOT / batch_dir
    bindings = json.loads((batch_dir / "provenance.json").read_text(encoding="utf-8"))
    provenance_sha = hashlib.sha256(canonical(bindings).encode()).hexdigest()
    if _sha256(config_path) != bindings["benchmark_config_sha256"]:
        raise ValueError("report config differs from frozen run")
    manifest = json.loads((ROOT / "artifacts/eval_manifest.json").read_text(encoding="utf-8"))
    if _sha256(ROOT / "artifacts/eval_manifest.json") != bindings["manifest_sha256"]:
        raise ValueError("report evaluator manifest differs from frozen run")
    public_path = ROOT / manifest["files"]["test_public"]["path"]
    private_path = ROOT / manifest["files"]["test_private"]["path"]
    if (_sha256(public_path) != bindings["public_sha256"]
            or _sha256(private_path) != manifest["files"]["test_private"]["sha256"]):
        raise ValueError("report evaluator file hash mismatch")
    public = load_public_episodes(public_path)
    private = load_private_episodes(private_path)
    hidden = {item.episode_id: item for item in private}
    catalog = Catalog.from_frozen()
    conditions = ("U1", "F", "A", "O", "no_user_model", "no_collaborative",
                  "no_explicit_preference_state", "no_replanning", "no_content")
    results, raw_scores, evidence = {}, {}, []
    failures = {}
    for condition in conditions:
        results[condition] = []
        raw_scores[condition] = []
        failures[condition] = {"categories": {}, "examples": [], "strata": {}}
        for seed in PREREGISTERED_SEEDS:
            run_dir = batch_dir / condition / ("seed_" + str(seed))
            spec = RunSpec(**json.loads((run_dir / "config.resolved.json").read_text(encoding="utf-8")))
            if not spec.live:
                raise ValueError("mechanical fixtures cannot generate live system reports")
            if (spec.system != condition or spec.seed != seed
                    or spec.provenance_sha256 != provenance_sha
                    or spec.public_sha256 != bindings["public_sha256"]):
                raise ValueError("system report provenance or condition identity mismatch")
            attempts = load_completed_attempts(spec, run_dir / "predictions.jsonl")
            if len(attempts) != len(public):
                raise ValueError("T19 report requires every frozen episode, including failures")
            provider_evidence = verify_provider_evidence(run_dir / "llm_attempts.jsonl", attempts)
            metrics = evaluate_episodes(public, private, attempts, catalog)
            traces = [json.loads(line) for line in (run_dir / "traces.jsonl").read_text(encoding="utf-8").splitlines()]
            if len(traces) != len(public) or len({row["episode_id"] for row in traces}) != len(public):
                raise ValueError("T19 trace denominator mismatch")
            planner = sum(turn.get("planner_calls", 0) for trace in traces for turn in trace["turns"])
            parser = sum(turn.get("parser_calls", 0) for trace in traces for turn in trace["turns"])
            replans = sum(turn.get("replans", 0) for trace in traces for turn in trace["turns"])
            row = {"seed": seed, "strict_success": metrics["strict_success_rate"],
                "constraint_precision": metrics["constraint_precision"], "fill_at_k": metrics["fill_at_k"],
                "requests": metrics["usage"]["requests"], "tool_calls": metrics["tool_calls"]["total"],
                "latency_p50_ms": metrics["latency_ms"]["p50"], "latency_p95_ms": metrics["latency_ms"]["p95"],
                "input_tokens": metrics["usage"]["input_tokens_known_sum"], "output_tokens": metrics["usage"]["output_tokens_known_sum"],
                "fallback_rate": metrics["rates"]["fallback"], "planner_calls": planner, "parser_calls": parser, "replans": replans,
                "metrics": metrics}
            results[condition].append(row)
            write_json(run_dir / "metrics.json", metrics)
            attempts_by_id = {attempt.episode_id: attempt for attempt in attempts}
            scores = {episode.episode_id: score_episode(episode, hidden[episode.episode_id], attempts_by_id[episode.episode_id], catalog)
                      for episode in public}
            raw_scores[condition].append(scores)
            for episode in public:
                score = scores[episode.episode_id]
                categories = failure_categories(episode, hidden[episode.episode_id], attempts_by_id[episode.episode_id], score)
                for category in categories:
                    failures[condition]["categories"][category] = failures[condition]["categories"].get(category, 0) + 1
                if categories and len(failures[condition]["examples"]) < 5:
                    failures[condition]["examples"].append({"seed": seed, "episode_id": episode.episode_id,
                        "layer": episode.layer, "family": episode.scenario_family, "status": score.status, "categories": categories})
                for stratum in ("layer:" + episode.layer, "family:" + episode.scenario_family):
                    counter = failures[condition]["strata"].setdefault(stratum, {"episodes": 0, "strict_successes": 0})
                    counter["episodes"] += 1
                    counter["strict_successes"] += int(score.strict_success)
            evidence.append({"condition": condition, "seed": seed,
                "raw_attempts_sha256": _sha256(run_dir / "predictions.jsonl"),
                "traces_sha256": _sha256(run_dir / "traces.jsonl"),
                "provider_attempts_sha256": _sha256(run_dir / "llm_attempts.jsonl"),
                "provider_evidence": provider_evidence,
                "run_identity": spec.identity})
    aggregate = {condition: _aggregate(rows) for condition, rows in results.items()}
    aggregate["always_agent"] = aggregate["A"]
    comparisons = {}
    for baseline, treatment in (("A", "O"), ("F", "O"), ("U1", "O"),
            ("no_user_model", "O"), ("no_content", "O"), ("no_explicit_preference_state", "O"), ("no_replanning", "O")):
        comparisons[treatment + "_minus_" + baseline] = {}
        for metric in ("strict_success", "constraint_precision", "fill_at_k", "request_count", "latency_ms"):
            def vector(condition):
                return [statistics.fmean(float(getattr(scores[episode.episode_id], metric))
                                         for scores in raw_scores[condition]) for episode in public]
            comparisons[treatment + "_minus_" + baseline][metric] = paired_bootstrap_ci(
                vector(baseline), vector(treatment), samples=10000, confidence=.95, seed=42)
    success_ci = comparisons["O_minus_A"]["strict_success"]
    summary = {"status": "COMPLETE", "scope": "T19 synthetic offline interactive benchmark with real provider and real tools",
        "episodes_per_run": 150, "runs": len(evidence), "agent_seeds": list(PREREGISTERED_SEEDS),
        "aggregate": aggregate, "per_seed": results, "paired_ci": comparisons, "failures": failures,
        "u0": "NOT_RUN_RESOURCE_SEMANTICS_UNVERIFIED", "provenance": bindings, "raw_evidence": evidence,
        "noninferiority_margin": -.02, "H2_noninferiority_supported": success_ci["ci_lower"] > -.02,
        "actual_cost_cny": None, "actual_cost_reason": "provider_bill_not_queried",
        "requests": sum(row["requests"] for rows in results.values() for row in rows),
        "input_tokens": sum(row["input_tokens"] for rows in results.values() for row in rows),
        "output_tokens": sum(row["output_tokens"] for rows in results.values() for row in rows)}
    summary["episodes_unknown_usage"] = sum(row["metrics"]["usage"]["episodes_unknown_usage"]
                                            for rows in results.values() for row in rows)
    summary["known_token_cost_cny_peak_estimate"] = (summary["input_tokens"]*2 + summary["output_tokens"]*8)/1e6
    summary["estimated_cost_cny_peak_ceiling"] = (
        None if summary["episodes_unknown_usage"] else summary["known_token_cost_cny_peak_estimate"])
    summary["cost_planning_reserved_cny"] = sum(
        row["requests"] * bindings["per_request_cost_ceiling_cny"]
        for rows in results.values() for row in rows)
    write_json(batch_dir / "summary.json", summary)
    _write_reports(summary)
    return summary


def _write_reports(summary):
    directory = ROOT / "AgenticRec/reports"
    old = (directory / "benchmark.md").read_text(encoding="utf-8")
    model_text = old.split("## Interactive systems")[0].replace("(IN PROGRESS)", "(COMPLETED)")
    lines = [model_text.rstrip(), "", "## Interactive systems", "",
        "Real DeepSeek deepseek-flash, non-thinking, 1024-token cap, zero network retries; frozen 150-episode synthetic test, three seeds. Mean +/- sample SD. All failures remain in the denominator.", "",
        "| System | Strict Success | Constraint Precision | Fill@K | Requests/run | Tools/run | Latency p50/p95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    def formatted(row, name):
        value = row[name]
        return "{:.4f} +/- {:.4f}".format(value["mean"], value["sample_std"])
    for name in ("U1", "F", "A", "O"):
        row = summary["aggregate"][name]
        lines.append("| {} | {} | {} | {} | {} | {} | {:.1f}/{:.1f} |".format(name,
            formatted(row, "strict_success"), formatted(row, "constraint_precision"), formatted(row, "fill_at_k"),
            formatted(row, "requests"), formatted(row, "tool_calls"), row["latency_p50_ms"]["mean"], row["latency_p95_ms"]["mean"]))
    lines += ["", "U0: NOT RUN; original resource semantics/license remain unverified. U1 is a rebuilt Plan First JSON adapter with common frozen recommendation pipeline followed by original ToolBox/Buffer/Map; no paper prompt or result claim.", "",
        "O minus A paired episode bootstrap: " + canonical(summary["paired_ci"]["O_minus_A"]), "",
        "H2 noninferiority at -0.02: " + ("supported" if summary["H2_noninferiority_supported"] else "NOT SUPPORTED; evidence is insufficient"), "",
        "Provider bill: NOT QUERIED. Peak-rate token estimate: {} CNY over {} request attempts; {} episodes have unknown usage. Conservative planning reservation: {:.6f} CNY. Unknown usage remains null; no partial token sum is presented as a total cost ceiling.".format(
            summary["estimated_cost_cny_peak_ceiling"], summary["requests"], summary["episodes_unknown_usage"], summary["cost_planning_reserved_cny"]), "",
        "Latency includes parsing, tools and failures; serial episodes, prewarmed model. U1 includes legacy subprocess startup. This synthetic offline benchmark does not measure online satisfaction or CTR.", ""]
    (directory / "benchmark.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    lines = ["# T19 Ablations (COMPLETED)", "", "Same frozen episodes, three seeds, LLM, caps and hard constraints. always_agent reuses A.", "",
        "| Condition | Strict Success | Constraint Precision | Fill@K | Requests/run |",
        "|---|---:|---:|---:|---:|"]
    for name in ("O", "no_user_model", "no_content", "no_collaborative", "no_explicit_preference_state", "always_agent", "no_replanning"):
        row = summary["aggregate"][name]
        lines.append("| {} | {} | {} | {} | {} |".format(name, *(formatted(row, field) for field in ("strict_success", "constraint_precision", "fill_at_k", "requests"))))
    lines += ["", "no_user_model removes trained-user and collaborative scoring/recall, retaining content plus train-popularity fallback. no_content removes TF-IDF/genre seed scoring, retaining collaborative recall/rank plus the same fallback. no_collaborative is the additional exploratory condition removing CF recall while retaining trained final ranking.", "",
        "no_explicit_preference_state exposes current feedback but does not commit its patch. no_replanning permits one planner attempt. Deterministic ranking_timeout is a local execution fixture with a content fallback; no failures are deliberately sent to the provider.", "",
        "Paired confidence intervals are generated in the local summary.json; no gain is asserted when intervals are inconclusive.", ""]
    (directory / "ablations.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    lines = ["# T19 Failure Analysis (COMPLETED)", "", "Categories may overlap; counts are over all 450 attempts per condition, including unsuccessful ones. Examples omit raw preferences/titles.", "",
        "Recommendation model finding remains negative: three-seed LightGCN NDCG@10 underperforms BPR-MF; it is retained.", ""]
    for name, value in summary["failures"].items():
        lines += ["## " + name, "", "Categories: " + canonical(value["categories"]), "",
                  "Coverage: " + canonical(value["strata"]), "", "Examples: " + canonical(value["examples"]), ""]
    lines += ["These are observed failures, not proven causal attribution. Parser/plan failures, acceptance misses, constraint violations and state mismatches are separated to guide analysis. Paired ablations supply limited attribution; results do not establish online gains.", ""]
    (directory / "failure_analysis.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
