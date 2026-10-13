"""Independent read-only audit of offline controls, train rankings and metrics."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import pyarrow.parquet as pq

from audit_three_way_comparison import assess, digest


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify_scores(records, predictions, labels, truth):
    assert len(records) == len(predictions) == 900
    saved = {(r['system'], r['episode_id']): r for r in predictions}
    assert len(saved) == 900
    seen = set()
    totals = defaultdict(Counter)
    for row in records:
        key = (row['system'], row['episode_id'])
        assert key not in seen
        seen.add(key)
        for field, value in saved[key].items():
            assert row[field] == value
        private = labels[row['episode_id']]
        assert private['category'] == 'session_quality' and row['history_n'] == private['history_n']
        turn = {'raw_ids': row['raw_ids'], 'status': 'OK', 'extraction_verified': True}
        expected, errors = assess(turn, private['labels'][0], truth, [])
        assert not any(errors.values()) and expected['strict_success']
        totals[row['system']]['n'] += 1
        totals[row['system']]['history_' + str(row['history_n'])] += 1
        for metric in ('hit', 'recall', 'ndcg', 'mrr'):
            assert math.isclose(row['metrics'][metric], expected[metric], rel_tol=0, abs_tol=1e-12)
            totals[row['system']][metric] += expected[metric]
    for counts in totals.values():
        assert counts['n'] == 180 and all(counts['history_' + str(h)] == 60 for h in (3, 5, 10))
    assert set(totals) == {'popular', 'itemcf', 'content', 'C1', 'candidate'}
    return totals


def audit(directory, output):
    root = Path(__file__).resolve().parents[3]
    protocol = read(directory/'protocol.json')
    source = Path(protocol['source_directory'])
    assert protocol['status'] == 'FROZEN' and protocol['scope'] == 'post_hoc_offline_controls'
    for name, expected in protocol['source_sha256'].items():
        assert digest(source/name) == expected, name
    for name, expected in protocol['implementation_sha256'].items():
        assert digest(root/name) == expected, name
    old = read(source/'protocol.json')
    for group, base in [('files', source), ('sources', root), ('resources', root)]:
        for name, expected in old[group].items():
            assert digest(base/name) == expected, name
    report = read(directory/'report.json')
    assert report['status'] == 'VERIFIED' and report['api_requests'] == 0
    assert report['predictions_sha256'] == digest(directory/'predictions.json')
    assert report['protocol_sha256'] == digest(directory/'protocol.json')
    labels = {r['episode_id']: r for r in read(source/'formal_private.json')}
    public = {r['episode_id']: r for r in read(source/'formal_public.json')}
    previous = root/'optimized/artifacts/comparison/three_way_20261013'
    truth = read(previous/'truth_private.json')
    catalog = {int(k): v for k, v in read(previous/'catalog.json')['items'].items()}
    records, predictions = read(directory/'scores.json'), read(directory/'predictions.json')
    totals = verify_scores(records, predictions, labels, truth)
    for system, counts in totals.items():
        for metric in ('hit', 'recall', 'ndcg', 'mrr'):
            assert math.isclose(report['systems'][system][metric], counts[metric]/180, abs_tol=1e-12)
        assert report['systems'][system]['hits'] == counts['hit']
    live = {(r['system'], r['episode_id']): r['turns'][0]['raw_ids']
            for r in (json.loads(line) for line in (source/'live_run/episodes.jsonl').read_text(encoding='utf-8').splitlines())
            if r['category'] == 'session_quality'}
    data = root/'optimized/data/processed/ml-1m-v1/train_positive.parquet'
    train = pq.read_table(data, columns=['user_id', 'item_id', 'rating'])
    item_users = defaultdict(set)
    edges = set()
    for user, item, rating in zip(train['user_id'].to_pylist(), train['item_id'].to_pylist(), train['rating'].to_pylist()):
        assert rating >= 4 and (user, item) not in edges
        edges.add((user, item)); item_users[item].add(user)
    population = {i: len(item_users[i]) for i in catalog}
    # Brute-force user-set intersections independently verify ItemCF ordering.
    cosine_cache = {}
    cf_checked = popular_checked = replayed = 0
    for row in records:
        assert row['raw_ids'] == [catalog[i]['raw_id'] for i in row['item_ids']]
        pub = public[row['episode_id']]
        liked = [r['item_id'] for r in pub['catalog_context']]
        allowed = sorted(set(catalog) - set(liked))
        assert set(row['item_ids']) <= set(allowed)
        if row['system'] == 'popular':
            assert row['item_ids'] == sorted(allowed, key=lambda i: (-population[i], i))[:10]
            popular_checked += 1
        elif row['system'] == 'itemcf':
            assert all(item_users[j] for j in liked)
            for seed in liked:
                if seed not in cosine_cache:
                    cosine_cache[seed] = {i: len(item_users[i] & item_users[seed]) / math.sqrt(len(item_users[i])*len(item_users[seed]))
                                          if item_users[i] else 0. for i in catalog}
            scores = {i: sum(cosine_cache[seed][i]/len(liked) for seed in liked) for i in allowed}
            assert row['item_ids'] == sorted(allowed, key=lambda i: (-scores[i], i))[:10]
            cf_checked += 1
        elif row['system'] in ('C1', 'candidate'):
            assert row['raw_ids'] == live[(row['system'], row['episode_id'])]
            replayed += 1
    assert (popular_checked, cf_checked, replayed) == (180, 180, 360)
    by = {s: {r['episode_id']: r for r in records if r['system'] == s} for s in totals}
    for name, comparison in report['comparisons'].items():
        a, b = name.split('_minus_')
        keys = sorted(by[a]); assert set(keys) == set(by[b])
        wins = sum(by[a][i]['metrics']['hit'] > by[b][i]['metrics']['hit'] for i in keys)
        losses = sum(by[a][i]['metrics']['hit'] < by[b][i]['metrics']['hit'] for i in keys)
        assert (comparison['wins'], comparison['losses']) == (wins, losses)
        for m in ('hit', 'recall', 'ndcg', 'mrr'):
            delta = sum(by[a][i]['metrics'][m]-by[b][i]['metrics'][m] for i in keys)/180
            assert math.isclose(delta, comparison['metrics'][m]['mean'], abs_tol=1e-12)
    result = {'status': 'VERIFIED', 'independent_quality_scores': 900,
              'independent_popular_rankings': popular_checked, 'brute_force_cosine_itemcf_rankings': cf_checked,
              'replayed_real_live_rankings': replayed, 'source_hashes_verified': True,
              'api_requests': 0, 'predictions_sha256': digest(directory/'predictions.json'),
              'protocol_sha256': digest(directory/'protocol.json'), 'auditor_sha256': digest(Path(__file__)),
              'scoring_helper_sha256': digest(Path(__file__).with_name('audit_three_way_comparison.py')),
              'hits': {s: int(v['hit']) for s, v in totals.items()}}
    with output.open('x', encoding='utf-8') as out:
        json.dump(result, out, indent=2); out.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.directory, args.output), indent=2))
