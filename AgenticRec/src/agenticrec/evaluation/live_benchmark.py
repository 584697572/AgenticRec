"""T19 batch orchestration: frozen inputs, bounded calls, durable evidence."""

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import time

from ..adapters.llm import ChatAdapter
from ..adapters.providers import build_live_chat_adapter
from ..agent.text_parser import TextRequestParser
from ..config import LLMBudget
from ..pipeline import FixedRecommendationPipeline
from ..runtime.errors import LLMTransportError
from .benchmark import (ROOT, BenchmarkConfig, build_dry_run, _load_public_contract, _sha256)
from .episodes import load_private_episodes
from .faults import FaultAwarePipeline, FaultBoundExecutor, FaultSchedule
from .fixture_transport import PublicFixtureTransport
from .runner import RunSpec, run_benchmark, load_completed_attempts, _load_events, _journal_state
from .system_executor import BenchmarkSystemExecutor, FeedbackSchedule
from .system_factory import build_benchmark_agent_loop
from .upstream_turn import UpstreamRebuiltTurn
from .variants import build_pipeline_variant
from .cost_limits import (MAX_INPUT_TOKEN_BOUND, INPUT_CNY_PER_MILLION,
                          OUTPUT_CNY_PER_MILLION, PER_REQUEST_COST_CNY)


def canonical(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def write_json(path, value, *, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = canonical(value) + "\n"
    if immutable and path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise ValueError("immutable benchmark evidence changed: " + path.name)
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(path)


def append_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


class AuditedTransport:
    """Persist request intent before sending; keep keys and headers out of evidence."""

    is_live = True

    def __init__(self, transport, path):
        self.transport, self.path = transport, path
        self.current_episode = None

    def chat(self, messages, **options):
        input_size = len(canonical(list(messages)).encode("utf-8"))
        if input_size > 8192:
            raise LLMTransportError("prompt exceeded frozen input bound", error_type="input_bound", retryable=False)
        number = sum(1 for _ in self.path.open(encoding="utf-8")) if self.path.exists() else 0
        request_id = str(number + 1)
        append_json(self.path, {"event": "STARTED", "request_id": request_id,
            "episode_id": self.current_episode, "messages": list(messages), "options": options,
            "input_bytes": input_size, "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        started = time.monotonic()
        try:
            reply = self.transport.chat(messages, **options)
        except Exception as error:
            append_json(self.path, {"event": "FAILED", "request_id": request_id,
                "error_type": type(error).__name__, "elapsed_ms": (time.monotonic()-started)*1000})
            raise
        append_json(self.path, {"event": "COMPLETED", "request_id": request_id,
            "reply": asdict(reply), "elapsed_ms": (time.monotonic()-started)*1000})
        if reply.usage is not None and reply.usage.prompt_tokens > MAX_INPUT_TOKEN_BOUND:
            raise RuntimeError("provider prompt usage exceeded the frozen monetary bound")
        return reply


def provenance(config_path, plan, schedule, faults):
    import torch
    source_root = ROOT / "AgenticRec/src/agenticrec"
    sources = {str(path.relative_to(ROOT)): _sha256(path)
               for path in sorted(source_root.rglob("*.py"))}
    sources["reproduction/scripts/t10_upstream_worker.py"] = _sha256(ROOT / "reproduction/scripts/t10_upstream_worker.py")
    checkpoints = {}
    for seed in (7, 42, 2026):
        report_path = ROOT / "reports/rec_baselines" / ("lightgcn_seed_" + str(seed) + ".json")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        layers = report["selection"]["selected_layers"]
        path = ROOT / "artifacts/models/lightgcn" / ("seed_{}_layer_{}.pt".format(seed, layers))
        observed = _sha256(path)
        if observed != report["trials"][str(layers)]["checkpoint_sha256"]:
            raise ValueError("T19 checkpoint differs from frozen model report")
        checkpoints[str(seed)] = {"path": path.relative_to(ROOT).as_posix(), "sha256": observed,
                                  "report_sha256": _sha256(report_path), "identity": report["identity"]}
    return {"benchmark_config_sha256": _sha256(config_path),
        "source_sha256": sources, "manifest_sha256": plan["frozen_inputs"]["manifest_sha256"],
        "public_sha256": plan["frozen_inputs"]["public_episodes_sha256"],
        "feedback_sha256": schedule.sha256, "fault_sha256": faults.sha256,
        "python": platform.python_version(), "torch": str(torch.__version__),
        "lock_sha256": _sha256(ROOT / "AgenticRec/requirements.dev.lock.txt"),
        "model": "lightgcn", "model_seeds": [7,42,2026],
        "checkpoints": checkpoints, "data_manifest_sha256": _sha256(ROOT / "artifacts/data_manifest.json"),
        "provider": "deepseek", "model_id": "deepseek-flash", "thinking": "disabled",
        "output_cap": 1024, "temperature": 0, "network_retries": 0,
        "input_token_planning_bound": MAX_INPUT_TOKEN_BOUND, "per_request_cost_ceiling_cny": PER_REQUEST_COST_CNY,
        "pricing_source": "https://api-docs.deepseek.com/zh-cn/quick_start/pricing/", "price_checked_date": "2026-10-10",
        "peak_input_cny_per_million": INPUT_CNY_PER_MILLION, "peak_output_cny_per_million": OUTPUT_CNY_PER_MILLION,
        "concurrency": 1, "latency_scope": "prewarmed model, serial episodes; U1 includes legacy subprocess startup",
        "authorization_ledger_sha256": plan["frozen_inputs"]["authorization_ledger_sha256"],
        "u1_scope": "rebuilt request/planner schema and frozen recommendation pipeline followed by original ToolBox/Buffer/Map; no LLM answer summarization",
        "hardware": {"system": platform.system(), "machine": platform.machine(), "cpu_count": os.cpu_count()}}


def prepare_batch(config_path, *, live, batch_dir=None):
    config_path = Path(config_path).resolve()
    config = BenchmarkConfig.from_file(config_path)
    plan = build_dry_run(config_path)
    if live:
        ledger_path = ROOT / config.authorization_ledger
        authorization = json.loads(ledger_path.read_text(encoding="utf-8"))
        if not (authorization.get("allow_paid_api") is True
                and authorization.get("scope") == "T19_complete_batch"
                and authorization.get("authorization_id")):
            raise ValueError("T19 requires explicit scoped batch authorization")
        required_money = plan["required_new_requests_ceiling"] * PER_REQUEST_COST_CNY
        if not plan["live_run_authorized"] or required_money > authorization["budget"]["money_budget"]:
            raise ValueError("request or money authorization cannot cover the full batch")
    _, manifest, _, public = _load_public_contract(config)
    private_path = ROOT / manifest["files"]["test_private"]["path"]
    if _sha256(private_path) != manifest["files"]["test_private"]["sha256"]:
        raise ValueError("private evaluator hash mismatch")
    private = load_private_episodes(private_path)
    ids = {episode.episode_id for episode in public}
    schedule = FeedbackSchedule.from_private_episodes(private, expected_episode_ids=ids)
    faults = FaultSchedule({episode.episode_id: episode.fault for episode in private})
    bindings = provenance(config_path, plan, schedule, faults)
    binding_sha = hashlib.sha256(canonical(bindings).encode()).hexdigest()
    batch_dir = batch_dir or ROOT / ("artifacts/runs/t19/live_20261010" if live else "artifacts/runs/t19/fixture_20261010")
    batch_dir = Path(batch_dir).resolve()
    batch_dir.relative_to(ROOT.resolve())
    # Bind before any request. Never overwrite a previous run's source identity.
    write_json(batch_dir / "provenance.json", bindings, immutable=True)
    write_json(batch_dir / "config.resolved.json", json.loads(config_path.read_text(encoding="utf-8")), immutable=True)
    specs = []
    for row in plan["run_matrix"]:
        spec = RunSpec(run_id=row["run_id"] + ("-live1010" if live else "-fixture1010"),
            benchmark_id=config.benchmark_id, system=row["condition"], seed=row["seed"], split="test",
            public_episodes=manifest["files"]["test_public"]["path"],
            public_sha256=plan["frozen_inputs"]["public_episodes_sha256"], episode_count=len(public),
            authorized_request_cap=row["request_ceiling"], money_budget_cny=row["request_ceiling"]*PER_REQUEST_COST_CNY if live else 0,
            per_request_cost_ceiling_cny=PER_REQUEST_COST_CNY if live else 0, live=live,
            feedback_sha256=schedule.sha256, fault_sha256=faults.sha256, provenance_sha256=binding_sha)
        specs.append((spec, batch_dir / row["condition"] / ("seed_" + str(row["seed"]))))
    for spec, run_dir in specs:
        journal = run_dir / "predictions.jsonl"
        if journal.exists():
            _, unresolved, _ = _journal_state(_load_events(journal), spec)
            if unresolved:
                raise ValueError("unresolved episode prevents paid batch resume")
    return config, plan, schedule, faults, specs, batch_dir


def run_batch(config_path, *, live=True, batch_dir=None, condition=None, seed=None):
    import torch
    torch.set_num_threads(1)
    config, plan, schedule, faults, specs, batch_dir = prepare_batch(config_path, live=live, batch_dir=batch_dir)
    bases = {}
    reports = []
    for spec, run_dir in specs:
        if condition is not None and spec.system != condition:
            continue
        if seed is not None and spec.seed != seed:
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        journal = run_dir / "predictions.jsonl"
        events = _load_events(journal)
        complete, unresolved, _ = _journal_state(events, spec)
        if len(complete) == spec.episode_count:
            print(canonical({"run_id": spec.run_id, "status": "RESUMED_COMPLETE", "episodes": len(complete)}), flush=True)
            continue
        base = bases.get(spec.seed)
        if base is None:
            base = FixedRecommendationPipeline.from_frozen("lightgcn", spec.seed)
            bases[spec.seed] = base
        recommendation = build_pipeline_variant(base, spec.system)
        fallback = build_pipeline_variant(base, "no_user_model")
        pipeline = FaultAwarePipeline(recommendation, fallback)
        budget = LLMBudget(allow_paid_api=live, provider="deepseek", model_id="deepseek-flash",
            api_request_cap=spec.authorized_request_cap, max_output_tokens=1024,
            money_budget=spec.money_budget_cny, max_retries=0, per_request_timeout_seconds=30,
            round_deadline_seconds=90)
        if live:
            adapter = build_live_chat_adapter(budget, seed=spec.seed)
            audit = AuditedTransport(adapter.transport, run_dir / "llm_attempts.jsonl")
            adapter.transport = audit
        else:
            adapter = ChatAdapter(PublicFixtureTransport(), budget)
            audit = None
        kwargs = {"fixed_pipeline": pipeline, "feedback_source": schedule,
                  "ledger": adapter.ledger, "planned_cost_ceiling": spec.per_request_cost_ceiling_cny}
        if spec.system == "F":
            kwargs["text_parser"] = TextRequestParser(adapter)
        elif spec.system == "U1":
            kwargs["upstream_turn"] = UpstreamRebuiltTurn(adapter, pipeline,
                planned_cost_ceiling=spec.per_request_cost_ceiling_cny)
        else:
            kwargs["agent_loop"] = build_benchmark_agent_loop(spec.system, fixed_pipeline=pipeline,
                planner=adapter, planned_cost_ceiling=spec.per_request_cost_ceiling_cny)
        executor = BenchmarkSystemExecutor(spec.system, **kwargs)
        bound = FaultBoundExecutor(executor, faults, pipeline)
        class TracedExecutor:
            feedback_sha256 = bound.feedback_sha256
            fault_sha256 = bound.fault_sha256
            def __call__(self, episode):
                if audit is not None:
                    audit.current_episode = episode.episode_id
                attempt = bound(episode)
                append_json(run_dir / "traces.jsonl", executor.last_trace)
                print(canonical({"run_id": spec.run_id, "episode": episode.episode_id,
                    "status": attempt.status, "requests": attempt.request_count}), flush=True)
                return attempt
        write_json(run_dir / "config.resolved.json", asdict(spec), immutable=True)
        report = run_benchmark(spec, journal, TracedExecutor())
        write_json(run_dir / "ledger.json", adapter.ledger.snapshot())
        write_json(run_dir / "run_report.json", report)
        reports.append(report)
        print(canonical({"run_id": spec.run_id, "status": report["status"],
                         "requests": report["remote_api_requests"]}), flush=True)
    return {"status": "RUNS_COMPLETE", "live": live, "batch_dir": str(batch_dir.relative_to(ROOT)), "runs": reports}
