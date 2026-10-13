"""Read-only independent quality and failure audit; never imports the runners."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def assess(turn, label, truth, previous):
    ids = turn['raw_ids']
    excluded = set(label['excluded_raw_ids'])
    if label.get('exclude_previous'):
        excluded.update(previous)
    errors = Counter()
    for raw in ids:
        item = truth.get(str(raw))
        if item is None:
            errors['unknown_identity_items'] += 1
            continue
        if not set(label['genres']).issubset(item['genres']):
            errors['missing_required_genre_items'] += 1
        if set(label.get('exclude_genres', [])).intersection(item['genres']):
            errors['forbidden_genre_items'] += 1
        if ((label['year_min'] is not None and item['year'] < label['year_min'])
                or (label['year_max'] is not None and item['year'] > label['year_max'])):
            errors['year_violation_items'] += 1
        if raw in excluded:
            errors['excluded_item_deliveries'] += 1
    known_ids = [raw for raw in ids if truth.get(str(raw)) is not None]
    errors['duplicate_items'] = len(known_ids) - len(set(known_ids))
    errors['quantity_wrong_turns'] = int(len(ids) != label['k'])
    errors['status_not_ok_turns'] = int(turn['status'] != 'OK')
    errors['unverified_extraction_turns'] = int(not turn['extraction_verified'])
    strict = not any(errors.values())
    targets = set(label['relevant_raw_ids'])
    unique = set()
    hit_ranks = []
    for rank, raw in enumerate(ids[:label['k']], 1):
        if raw in targets and raw not in unique:
            unique.add(raw)
            hit_ranks.append(rank)
    delivered = turn['status'] == 'OK' and turn['extraction_verified']
    dcg = sum(1 / math.log2(rank + 1) for rank in hit_ranks)
    ideal = sum(1 / math.log2(rank + 1)
                for rank in range(1, min(label['k'], len(targets)) + 1))
    values = {
        'strict_success': strict,
        'hit': float(bool(hit_ranks)) if delivered and targets else 0.,
        'recall': len(unique) / len(targets) if delivered and targets else 0.,
        'ndcg': dcg / ideal if delivered and ideal else 0.,
        'mrr': 1 / hit_ranks[0] if delivered and hit_ranks else 0.,
    }
    return values, errors


def audit(directory, output):
    root = Path(__file__).resolve().parents[3]
    protocol = json.loads((directory / 'protocol.json').read_text(encoding='utf-8'))
    for field in ('sources_sha256', 'weights_sha256', 'data_sha256'):
        for name, expected in protocol[field].items():
            assert digest(root / name) == expected, name
    for name, expected in protocol['files_sha256'].items():
        assert digest(directory / name) == expected, name
    labels = {r['episode_id']: r for r in json.loads(
        (directory / 'formal_private.json').read_text(encoding='utf-8'))}
    truth = json.loads((directory / 'truth_private.json').read_text(encoding='utf-8'))
    run = directory / 'formal_run'
    execution = json.loads((run / 'execution.json').read_text(encoding='utf-8'))
    rows = [json.loads(line) for line in (run / 'episodes.jsonl').read_text(
        encoding='utf-8').splitlines()]
    assert execution['status'] == 'COMPLETED' and len(rows) == 960
    assert len({(r['system'], r['episode_id']) for r in rows}) == len(rows)
    diagnostics = defaultdict(Counter)
    scenarios = defaultdict(lambda: defaultdict(lambda: Counter(n=0, success=0)))
    quality = defaultdict(lambda: Counter(n=0))
    checked = refusal = 0
    for row in rows:
        hidden = labels[row['episode_id']]
        assert row['category'] == hidden['category']
        assert len(row['turns']) == len(hidden['labels'])
        previous = []
        task_ok = []
        for turn, label in zip(row['turns'], hidden['labels']):
            if label['feasible']:
                expected, errors = assess(turn, label, truth, previous)
                for name, value in expected.items():
                    assert math.isclose(float(turn['scores'][name]), float(value),
                                        rel_tol=0, abs_tol=1e-12), (row['system'], row['episode_id'], name)
                task_ok.append(expected['strict_success'])
                diagnostics[row['system']].update(errors)
                if hidden['category'] == 'session_quality':
                    quality[row['system']]['n'] += 1
                    for metric in ('hit', 'recall', 'ndcg', 'mrr'):
                        quality[row['system']][metric] += expected[metric]
                    if expected['strict_success']:
                        quality[row['system']]['successful_delivery_n'] += 1
                        quality[row['system']]['successful_delivery_hits'] += expected['hit']
                checked += 1
            else:
                # Independent minimal invariants; refusal semantics are declared
                # by the frozen protocol, not redefined after observing results.
                scores = turn['scores']
                assert scores['safe_non_delivery'] == (not turn['raw_ids'])
                assert not scores['correct_refusal'] or scores['safe_non_delivery']
                assert not scores['explanatory_refusal'] or scores['correct_refusal']
                refusal += 1
            previous = [raw for raw in turn['raw_ids'] if raw is not None]
        if hidden['category'] != 'unanswerable':
            assert row['strict_success'] == all(task_ok)
            bucket = scenarios[row['system']][row['scenario']]
            bucket['n'] += 1
            bucket['success'] += int(all(task_ok))
    public = json.loads((root / 'optimized/reports/three_way_comparison_20261013.json').read_text(encoding='utf-8'))
    for system, values in quality.items():
        assert values['n'] == 180
        for metric in ('hit', 'recall', 'ndcg', 'mrr'):
            assert math.isclose(values[metric] / 180,
                public['summary']['systems'][system]['quality'][metric], abs_tol=1e-12)
    assert checked == 1140 and refusal == 60
    result = {
        'status': 'VERIFIED', 'feasible_turns_independently_recomputed': checked,
        'refusal_turns_invariants_checked': refusal,
        'metrics': ['strict_success', 'Hit@10', 'Recall@10', 'NDCG@10', 'MRR@10'],
        'failure_counts': {k: dict(v) for k, v in diagnostics.items()},
        'failure_count_note': 'Non-exclusive; one item/turn may violate several predicates.',
        'by_scenario': {k: {s: dict(v) for s, v in cases.items()} for k, cases in scenarios.items()},
        'quality_delivery_counts': {k: dict(v) for k, v in quality.items()},
        'conditional_metric_note': 'Successful-delivery counts are supplemental; headline denominator remains all 180 users.',
        'frozen_inputs_sources_weights': 'PASS',
        'protocol_sha256': digest(directory / 'protocol.json'),
        'episodes_sha256': digest(run / 'episodes.jsonl'),
        'auditor_sha256': digest(Path(__file__)),
        'provider_calls_by_auditor': 0,
    }
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.directory.resolve(), args.output.resolve())
    print(json.dumps({k: result[k] for k in ('status', 'feasible_turns_independently_recomputed',
                                             'refusal_turns_invariants_checked', 'provider_calls_by_auditor')}))
