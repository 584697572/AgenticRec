"""Verify real train-only three-route retrieval; keep ID-bearing trace private."""
from collections import Counter
import json

from agenticrec.adapters.model import KnownUserScorer
from agenticrec.data import digest
from agenticrec.ranking.constraints import Constraints
from agenticrec.retrieval.sources import CandidateRetriever
from agenticrec.training import ROOT, load_train_contract


def main():
    manifest, n_users, n_items, _seen, edges = load_train_contract()
    model = KnownUserScorer.from_artifacts("lightgcn", seed=42)
    retriever = CandidateRetriever.from_frozen(model)
    all_ids = list(range(1, n_items + 1))
    seed = sorted(Counter(i for _u, i in edges).items(),
                  key=lambda pair: (-pair[1], pair[0]))[0][0]
    base = Constraints(include_genres=("Comedy",), exclude_genres=("Horror",),
                       year_min=1990, year_max=2000, k=10)
    base_feasible = retriever.gate.pre_filter(all_ids, base)
    if base_feasible.status != "OK" or len(base_feasible.item_ids) < 11:
        raise ValueError("Real-data smoke has too few feasible catalog items")
    explicitly_excluded = next(i for i in base_feasible.item_ids if i != seed)
    rules = Constraints(include_genres=base.include_genres,
                        exclude_genres=base.exclude_genres,
                        year_min=base.year_min, year_max=base.year_max,
                        excluded_item_ids=(explicitly_excluded,), k=10)
    outputs = {}
    for name, options in (("known_user", {"user_id": 1, "history_authorized": True}),
                          ("anonymous_seed", {})):
        result = retriever.retrieve(all_ids, rules, liked_ids=(seed,), **options)
        final = retriever.gate.finalize(result.fused.candidate_ids,
                                        result.effective_constraints, result.seen_ids)
        if result.status != "OK" or final.status != "OK" or not final.item_ids:
            raise AssertionError("T11 real-data retrieval did not produce a feasible result")
        if (len(result.fused.candidate_ids) > 200
                or any(len(ids) > 100 for ids in result.source_rankings.values())
                or explicitly_excluded in final.item_ids or seed in final.item_ids):
            raise AssertionError("Recall cap or hard exclusion violated")
        for item_id in final.item_ids:
            item = retriever.catalog.items[item_id]
            if ("Comedy" not in item.genres or "Horror" in item.genres
                    or item.year is None or not 1990 <= item.year <= 2000):
                raise AssertionError("Final hard condition violated")
        outputs[name] = {"profile_source": result.profile_source,
                         "fallback_reason": result.fallback_reason,
                         "source_rankings": result.source_rankings,
                         "rrf_candidate_ids": result.fused.candidate_ids,
                         "source_ranks": result.fused.source_ranks,
                         "final_ids": final.item_ids,
                         "eligible_count": result.eligible_count,
                         "shortfall": final.shortfall}
    missing = retriever.gate.pre_filter(all_ids, Constraints(required_fields=("duration",)))
    conflict = retriever.gate.pre_filter(all_ids, Constraints(include_genres=("Horror",),
                                                             exclude_genres=("Horror",)))
    no_feasible = retriever.gate.pre_filter(all_ids, Constraints(year_min=1800, year_max=1850))
    if (missing.status, conflict.status, no_feasible.status) != (
            "MISSING_FIELD", "CLARIFY_CONFLICT", "NO_FEASIBLE_ITEMS"):
        raise AssertionError("Structured failure states differ from constraint contract")
    private = {"dataset": "ml-1m-v1", "seed_item_id": seed,
               "explicitly_excluded_item_id": explicitly_excluded,
               "rules": {"include_genres": rules.include_genres, "exclude_genres": rules.exclude_genres,
                         "year_min": rules.year_min, "year_max": rules.year_max, "k": rules.k},
               "outputs": outputs}
    trace_path = ROOT / "artifacts/candidate_fusion/t11_trace_20261006.json"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(json.dumps(private, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    public = {"status": "PASS", "task": "T11", "variant": "U1_upstream_rebuilt",
              "dataset": "ml-1m-v1", "candidate_count": n_items,
              "warm_user_count": n_users,
              "id_map_sha256": manifest["id_map_sha256"],
              "data_manifest_sha256": digest(ROOT / "artifacts/data_manifest.json"),
              "private_trace_path": trace_path.relative_to(ROOT).as_posix(),
              "private_trace_sha256": digest(trace_path),
              "per_route_cap": 100, "union_cap": 200, "rrf_constant": 60,
              "scenarios": {name: {"profile_source": row["profile_source"],
                                  "source_counts": {source: len(ids) for source, ids in row["source_rankings"].items()},
                                  "eligible_count": row["eligible_count"],
                                  "fused_count": len(row["rrf_candidate_ids"]),
                                  "final_count": len(row["final_ids"]),
                                  "final_shortfall": row["shortfall"],
                                  "hard_violations": 0} for name, row in outputs.items()},
              "structured_statuses": {"unsupported_field": missing.status,
                                      "conflict": conflict.status,
                                      "no_feasible": no_feasible.status},
              "holdout_labels_loaded": False, "remote_api_requests": 0,
              "recommendation_quality": "NOT EVALUATED"}
    report_path = ROOT / "reports/candidate_fusion/t11_20261006.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(public, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(public, sort_keys=True))


if __name__ == "__main__":
    main()
