"""Export T19 per-user model predictions and regenerate metrics from them."""

from collections import Counter
import json
import math
from pathlib import Path
import statistics

import torch

from ..data import digest
from ..models.baselines import popularity_scores, random_ranking
from ..models.bpr import load_checkpoint
from ..models.lightgcn import load_lightgcn_checkpoint, normalized_bipartite_graph
from ..training import ROOT, load_frozen_cohort, load_train_contract
from .benchmark import PREREGISTERED_SEEDS, _sha256, paired_bootstrap_ci
from .metrics import evaluate_cohort, rank_candidates, score_ranking


METHODS = ("Random", "Popularity", "BPR-MF", "LightGCN")
METRICS = ("recall", "ndcg", "mrr", "hit")


def _rank_model(score_rows, user_ids, candidates, seen_by_user, k):
    predictions = {}
    for user_id, row in zip(user_ids, score_rows):
        predictions[user_id] = rank_candidates(
            dict(zip(candidates, row)), candidates, k, seen_by_user[user_id]
        )
    return predictions


def _load_reports(seed):
    report_dir = ROOT / "reports/rec_baselines"
    bpr_path = report_dir / ("seed_" + str(seed) + ".json")
    light_path = report_dir / ("lightgcn_seed_" + str(seed) + ".json")
    bpr = json.loads(bpr_path.read_text(encoding="utf-8"))
    light = json.loads(light_path.read_text(encoding="utf-8"))
    if (bpr.get("seed") != seed or light.get("seed") != seed
            or light.get("baseline_report_sha256") != digest(bpr_path)):
        raise ValueError("model reports do not form a same-seed frozen pair")
    return bpr_path, light_path, bpr, light


def export_predictions(output_path, seeds=PREREGISTERED_SEEDS):
    """Write rankings only; hidden item targets never enter model scoring."""
    seeds = tuple(seeds)
    if seeds != PREREGISTERED_SEEDS:
        raise ValueError("prediction export requires pre-registered seeds")
    output_path = Path(output_path)
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError("raw model prediction evidence already exists")

    data_manifest, n_users, n_items, train_seen, train_edges = load_train_contract()
    cohort = load_frozen_cohort("test", ROOT / "artifacts/data_manifest.json")
    candidates = list(range(1, n_items + 1))
    if cohort["candidate_ids"] != candidates:
        raise ValueError("test candidate universe differs from the train ID map")
    # Evaluator selects denominator membership, then discards targets before scoring.
    eval_user_ids = tuple(
        user_id for user_id in cohort["user_ids"]
        if cohort["relevant_by_user"][str(user_id)]
    )
    if len(eval_user_ids) != 954:
        raise ValueError("frozen test evaluator user count changed")
    seen = {user_id: tuple(sorted(train_seen[user_id])) for user_id in eval_user_ids}
    k = 10
    graph = normalized_bipartite_graph(n_users, n_items, train_edges)
    source_runs = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    rows = 0
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        for seed in seeds:
            bpr_path, light_path, bpr_report, light_report = _load_reports(seed)
            bpr_checkpoint = ROOT / "artifacts/models/bpr" / ("seed_" + str(seed) + ".pt")
            bpr = load_checkpoint(
                bpr_checkpoint, n_users, n_items,
                bpr_report["config"]["embedding_dim"], bpr_report["identity"],
            )
            selected_layers = light_report["selection"]["selected_layers"]
            light_checkpoint = ROOT / "artifacts/models/lightgcn" / (
                "seed_" + str(seed) + "_layer_" + str(selected_layers) + ".pt"
            )
            light = load_lightgcn_checkpoint(
                light_checkpoint, graph, n_users, n_items,
                light_report["config"]["embedding_dim"], selected_layers,
                light_report["identity"], light_report["config"],
            )
            user_tensor = torch.tensor(eval_user_ids, dtype=torch.long)
            with torch.no_grad():
                bpr_rows = (bpr.users.weight[user_tensor] @ bpr.items.weight[1:].T).tolist()
                light_users, light_items = light.propagate()
                light_rows = (light_users[user_tensor - 1] @ light_items.T).tolist()
            rankings = {
                "BPR-MF": _rank_model(bpr_rows, eval_user_ids, candidates, seen, k),
                "LightGCN": _rank_model(light_rows, eval_user_ids, candidates, seen, k),
                "Random": {},
                "Popularity": {},
            }
            popularity_path = ROOT / "artifacts/models/bpr" / (
                "seed_" + str(seed) + "_popularity.json"
            )
            counts = Counter({
                int(item_id): int(count)
                for item_id, count in json.loads(
                    popularity_path.read_text(encoding="utf-8")
                ).items()
            })
            pop_scores = popularity_scores(candidates, counts)
            for user_id in eval_user_ids:
                rankings["Random"][user_id] = random_ranking(
                    candidates, set(seen[user_id]), k, seed, user_id
                )
                rankings["Popularity"][user_id] = rank_candidates(
                    pop_scores, candidates, k, seen[user_id]
                )
                row = {
                    "schema_version": 1,
                    "seed": seed,
                    "user_id": user_id,
                    "rankings": {
                        method: rankings[method][user_id] for method in METHODS
                    },
                }
                stream.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
                rows += 1
            source_runs.append({
                "seed": seed,
                "bpr_report_sha256": digest(bpr_path),
                "lightgcn_report_sha256": digest(light_path),
                "bpr_checkpoint_sha256": digest(bpr_checkpoint),
                "lightgcn_checkpoint_sha256": digest(light_checkpoint),
                "popularity_counts_sha256": digest(popularity_path),
            })
    temporary.replace(output_path)
    manifest = {
        "schema_version": 1,
        "status": "PASS",
        "scope": "T19 raw recommendation-model predictions",
        "file": output_path.relative_to(ROOT).as_posix(),
        "sha256": digest(output_path),
        "rows": rows,
        "seeds": list(seeds),
        "methods": list(METHODS),
        "top_k": k,
        "evaluation_users": len(eval_user_ids),
        "target_item_ids_in_prediction_file": False,
        "candidate_items": n_items,
        "candidate_policy": "full mapped universe minus train-seen items",
        "id_map_sha256": data_manifest["id_map_sha256"],
        "test_cohort_sha256": digest(ROOT / "data/eval_private/ml-1m-v1/test.json"),
        "source_runs": source_runs,
        "remote_api_requests": 0,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return manifest


def score_predictions(prediction_path, *, bootstrap_samples=10000, bootstrap_seed=42):
    """Attach evaluator-only targets after prediction freeze and rebuild every metric."""
    prediction_path = Path(prediction_path)
    manifest_path = prediction_path.with_suffix(prediction_path.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("sha256") != _sha256(prediction_path)
            or manifest.get("target_item_ids_in_prediction_file") is not False):
        raise ValueError("raw prediction manifest or hash is invalid")
    rows = {}
    for line in prediction_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if set(row) != {"schema_version", "seed", "user_id", "rankings"}:
            raise ValueError("invalid raw prediction row")
        key = (row["seed"], row["user_id"])
        if key in rows or set(row["rankings"]) != set(METHODS):
            raise ValueError("duplicate row or changed model order")
        rows[key] = row["rankings"]
    if len(rows) != manifest["rows"]:
        raise ValueError("raw prediction row count mismatch")

    cohort = load_frozen_cohort("test", ROOT / "artifacts/data_manifest.json")
    user_ids = cohort["user_ids"]
    eval_user_ids = tuple(
        user_id for user_id in user_ids if cohort["relevant_by_user"][str(user_id)]
    )
    relevant = {user_id: cohort["relevant_by_user"][str(user_id)] for user_id in user_ids}
    seen = {user_id: cohort["seen_by_user"][str(user_id)] for user_id in user_ids}
    candidates = cohort["candidate_ids"]
    per_seed = []
    values = {method: {metric: [] for metric in METRICS} for method in METHODS}
    per_user_seed_scores = {
        method: {metric: {user_id: [] for user_id in eval_user_ids} for metric in METRICS}
        for method in ("BPR-MF", "LightGCN")
    }
    for seed in PREREGISTERED_SEEDS:
        rankings_by_method = {
            method: {user_id: rows[(seed, user_id)][method] for user_id in eval_user_ids}
            for method in METHODS
        }
        seed_metrics = {}
        for method in METHODS:
            result = evaluate_cohort(
                user_ids, relevant, rankings_by_method[method], candidates, 10, seen
            )
            seed_metrics[method] = result
            for metric in METRICS:
                values[method][metric].append(float(result[metric]))
            if method in per_user_seed_scores:
                for user_id in eval_user_ids:
                    score = score_ranking(
                        rankings_by_method[method][user_id], relevant[user_id],
                        candidates, 10, seen[user_id],
                    )
                    for metric in METRICS:
                        per_user_seed_scores[method][metric][user_id].append(score[metric])
        _bpr_path, _light_path, bpr_report, light_report = _load_reports(seed)
        for method in METHODS:
            expected = (light_report["test_metrics"] if method == "LightGCN"
                        else bpr_report["metrics"]["test"][method])
            actual = seed_metrics[method]
            if (actual["failures"] != expected["failures"]
                    or actual["users_evaluated"] != expected["users_evaluated"]
                    or any(not math.isclose(actual[metric], expected[metric],
                                            rel_tol=0, abs_tol=1e-15)
                           for metric in METRICS)):
                raise ValueError("raw predictions do not regenerate source metrics")
        per_seed.append({"seed": seed, "metrics": seed_metrics})

    aggregate = {
        method: {
            metric: {
                "mean": statistics.fmean(values[method][metric]),
                "sample_std": statistics.stdev(values[method][metric]),
            }
            for metric in METRICS
        }
        for method in METHODS
    }
    paired = {}
    for metric in METRICS:
        baseline = [statistics.fmean(per_user_seed_scores["BPR-MF"][metric][user_id])
                    for user_id in eval_user_ids]
        treatment = [statistics.fmean(per_user_seed_scores["LightGCN"][metric][user_id])
                     for user_id in eval_user_ids]
        paired[metric] = paired_bootstrap_ci(
            baseline, treatment, samples=bootstrap_samples,
            confidence=0.95, seed=bootstrap_seed,
        )
    result = {
        "schema_version": 1,
        "status": "PASS",
        "scope": "T19 recommendation-model table regenerated from raw predictions",
        "seeds": list(PREREGISTERED_SEEDS),
        "candidate_universe": "all 3469 mapped MovieLens1M items minus train-seen items",
        "test_users_total": len(user_ids),
        "test_users_evaluated": len(eval_user_ids),
        "test_users_without_relevance": len(user_ids) - len(eval_user_ids),
        "metrics_at_k": 10,
        "per_seed": per_seed,
        "aggregate": aggregate,
        "lightgcn_minus_bpr_by_seed": {
            metric: [right - left for left, right in zip(
                values["BPR-MF"][metric], values["LightGCN"][metric]
            )]
            for metric in METRICS
        },
        "episode_level_paired_ci": paired,
        "negative_gain_present": any(
            value < 0 for value in [
                right - left for left, right in zip(
                    values["BPR-MF"]["ndcg"], values["LightGCN"]["ndcg"]
                )
            ]
        ),
        "significance_claim": (
            "LightGCN NDCG@10 is lower than BPR-MF under the paired user bootstrap"
            if paired["ndcg"]["ci_upper"] < 0 else "NOT MADE"
        ),
        "raw_predictions": {
            "path": prediction_path.relative_to(ROOT).as_posix(),
            "sha256": _sha256(prediction_path),
            "manifest_sha256": _sha256(manifest_path),
            "rows": len(rows),
        },
        "source_metrics_exactly_regenerated": True,
        "frozen_identity": {
            "id_map_sha256": manifest["id_map_sha256"],
            "test_cohort_sha256": manifest["test_cohort_sha256"],
        },
        "remote_api_requests": 0,
    }
    return result


def write_scored_summary(summary, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    temporary.replace(output_path)
