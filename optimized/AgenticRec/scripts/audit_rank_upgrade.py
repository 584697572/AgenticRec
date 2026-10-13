"""Read-only, independent raw-data, split and metric audit of the HetRec study.

Does not import the training runner or its scoring function. Never calls an API.
"""
import argparse
from collections import Counter, defaultdict
import csv
import io
import json
import math
from pathlib import Path
import random
import zipfile

import pyarrow.parquet as pq
import torch

from audit_three_way_comparison import assess, digest

METRICS = ('hit', 'recall', 'ndcg', 'mrr')
SYSTEM_SEEDS = {s: (None,) for s in ('popular', 'itemcf', 'content', 'c1_rrf', 'c1_weighted', 'route2')}
SYSTEM_SEEDS.update({s: (7, 42, 2026) for s in ('route1', 'route3', 'route4')})


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def close(a, b):
    assert math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=0, abs_tol=1e-12), (a, b)


def bootstrap_interval(delta, seed):
    rng = random.Random(seed)
    n = len(delta)
    # Index sampling is intentionally implemented separately from the runner.
    samples = sorted(sum(delta[min(n-1, int(rng.random()*n))] for _ in range(n))/n for _ in range(10000))
    result = []
    for q in (.025, .975):
        position = (len(samples)-1)*q
        lo = math.floor(position)
        result.append(samples[lo]*(1-(position-lo))+samples[lo+1]*(position-lo))
    return result


def verify_scores(scores, predictions, public, private, catalog, expected_users=300):
    """Reject missing seeds, duplicate rows, invalid deliveries and invented scores."""
    expected = {(s, seed, episode) for s, seeds in SYSTEM_SEEDS.items() for seed in seeds for episode in public}
    assert len(public) == len(private) == expected_users and set(public) == set(private)
    saved = {(r['system'], r['seed'], r['episode_id']): r for r in predictions}
    assert len(saved) == len(predictions) == len(expected) and set(saved) == expected
    seen, by = set(), defaultdict(lambda: defaultdict(list))
    for row in scores:
        key = (row['system'], row['seed'], row['episode_id'])
        assert key not in seen and key in saved
        seen.add(key)
        assert {k: v for k, v in row.items() if k != 'metrics'} == saved[key]
        pub, label = public[row['episode_id']], private[row['episode_id']]
        assert row['history_n'] == pub['history_n'] == label['history_n']
        oracle_label = {'k': 10, 'genres': [], 'year_min': None, 'year_max': None,
                        'excluded_raw_ids': pub['liked_item_ids'], 'relevant_raw_ids': label['relevant_item_ids']}
        values, errors = assess({'raw_ids': row['item_ids'], 'status': 'OK', 'extraction_verified': True},
                               oracle_label, catalog, [])
        assert values['strict_success'] and not any(errors.values()), (key, errors)
        for m in METRICS:
            close(row['metrics'][m], values[m])
        by[row['system']][row['episode_id']].append(row)
    assert seen == expected and len(scores) == len(expected)
    means = {s: {u: {m: sum(r['metrics'][m] for r in rs)/len(rs) for m in METRICS}
                 for u, rs in users.items()} for s, users in by.items()}
    return by, means


def verify_selection(value, route):
    assert value['status'] == 'DEVELOPED' and value['test_opened'] is False and value['api_requests'] == 0
    groups = defaultdict(list)
    for row in value['candidates']:
        assert all(math.isfinite(v) and 0 <= v <= 1 for v in row['metrics'].values())
        groups[json.dumps(row['config'], sort_keys=True)].append(row)
    assert len(groups) == {'route1': 9, 'route2': 36, 'route3': 9, 'route4': 12}[route]
    expected_seeds = set(SYSTEM_SEEDS[route])
    summaries = []
    for key, rows in groups.items():
        assert len(rows) == len(expected_seeds) and {r['seed'] for r in rows} == expected_seeds
        summaries.append({'config': json.loads(key), 'metrics': {
            m: sum(r['metrics'][m] for r in rows)/len(rows) for m in METRICS}})
    winner = max(enumerate(summaries), key=lambda p: (p[1]['metrics']['hit'], p[1]['metrics']['ndcg'], -p[0]))[1]
    assert value['selected'] == winner
    return winner


def verify_raw(directory, archive):
    mapping = read(directory/'id_map_private.json')
    items = {int(k): v for k, v in mapping['item_to_model'].items()}
    users = {int(k): v for k, v in mapping['user_to_model'].items()}
    catalog = read(directory/'catalog.json')
    with zipfile.ZipFile(archive) as z:
        def rows(name):
            return csv.DictReader(io.StringIO(z.read(name).decode('latin-1')), delimiter='\t')
        records = [(int(r['timestamp']), int(r['userID']), int(r['movieID']), float(r['rating']))
                   for r in rows('user_ratedmovies-timestamps.dat')]
        movies = {int(r['id']): r for r in rows('movies.dat')}
        genres = defaultdict(set)
        for row in rows('movie_genres.dat'):
            genres[int(row['movieID'])].add(row['genre'])
        assert 'non-commercial use' in z.read('readme.txt').decode('latin-1')
        assert z.testzip() is None
    assert len(records) == 855598 and len({(u, i) for ts, u, i, r in records}) == len(records)
    records.sort()
    cutoffs = [records[int(len(records)*q)][0] for q in (.5, .7, .85)]
    assert cutoffs == read(directory/'prepared.json')['cutoffs_ms']
    warm = sorted({i for ts, u, i, r in records if ts < cutoffs[1] and r >= 4})
    assert items == {raw: n+1 for n, raw in enumerate(warm)} and len(catalog) == len(items) == 7541
    graph_users = sorted({u for ts, u, i, r in records if ts < cutoffs[1] and r >= 4})
    assert users == {raw: n+1 for n, raw in enumerate(graph_users)}
    for raw, model_id in items.items():
        actual, source = catalog[str(model_id)], movies[raw]
        assert set(actual) == {'raw_id', 'item_id', 'title', 'year', 'genres'}
        assert actual['raw_id'] == raw and actual['item_id'] == model_id and actual['title'] == source['title']
        year = int(source['year']) if source['year'].isdigit() and int(source['year']) > 0 else None
        assert actual['year'] == year and set(actual['genres']) == genres[raw]
    for name, cutoff in zip(('graph50', 'graph70'), cutoffs):
        table = pq.read_table(directory/(name+'.parquet')).to_pydict()
        actual = list(zip(table['user_id'], table['item_id'], table['timestamp_ms']))
        expected = [(users[u], items[i], ts) for ts, u, i, r in records if ts < cutoff and r >= 4]
        assert actual == expected and len({(u, i) for u, i, ts in actual}) == len(actual)
    phases = {}
    for phase, lo, hi in zip(('training', 'development', 'test'), cutoffs, (cutoffs[1], cutoffs[2], 10**20)):
        history, seen, targets = defaultdict(list), defaultdict(set), defaultdict(set)
        for ts, u, raw, rating in records:
            if ts < lo:
                seen[u].add(raw)
                if rating >= 4 and raw in items:
                    history[u].append(items[raw])
        for ts, u, raw, rating in records:
            if lo <= ts < hi and rating >= 4 and raw in items and raw not in seen[u]:
                targets[u].add(items[raw])
        phases[phase] = (history, targets, sorted(u for u in targets if len(history[u]) >= 10))
    rng = random.Random(2026101306)
    test = phases['test'][2][:]; rng.shuffle(test); test = test[:300]
    dev = [u for u in phases['development'][2] if u not in test]; rng.shuffle(dev); dev = dev[:180]
    train = [u for u in phases['training'][2] if u not in set(test+dev)]; rng.shuffle(train); train = train[:600]
    assert not set(test)&set(dev) and not set(train)&set(test+dev)
    assert read(directory/'training_users_private.json') == train and len(train) == 414
    history, targets, _ = phases['training']
    examples = [{'liked_item_ids': history[u][-h:], 'relevant_item_ids': sorted(targets[u])}
                for u in train for h in (3, 5, 10)]
    assert read(directory/'train_examples.json') == examples
    for phase, selected, per_h in [('development', dev, 60), ('test', test, 100)]:
        history, targets, _ = phases[phase]
        public, private = read(directory/(phase+'_public.json')), read(directory/(phase+'_private.json'))
        assert len(public) == len(private) == len(selected)
        assert Counter(r['history_n'] for r in public) == {3: per_h, 5: per_h, 10: per_h}
        for n, (p, q, u) in enumerate(zip(public, private, selected)):
            h = (3, 5, 10)[n % 3]
            assert p == {'episode_id': phase+'-'+str(n).zfill(3), 'liked_item_ids': history[u][-h:], 'history_n': h}
            assert q == {'episode_id': p['episode_id'], 'evaluator_raw_user_id': u,
                         'relevant_item_ids': sorted(targets[u]), 'history_n': h}
            assert not set(p['liked_item_ids'])&targets[u]
    return {'raw_ratings': len(records), 'catalog_items': len(items), 'training_users': len(train),
            'development_users': len(dev), 'test_users': len(test), 'training_examples': len(examples),
            'raw_mapping_and_temporal_splits_verified': True}


def audit(directory, output):
    root = Path(__file__).resolve().parents[3]
    prepared, frozen, report = [read(directory/name) for name in ('prepared.json', 'frozen.json', 'report.json')]
    assert frozen['status'] == 'FROZEN' and frozen['test_opened'] is False
    for names, base in [(prepared['source_sha256'], root), (prepared['input_sha256'], directory),
                        (frozen['files_sha256'], directory)]:
        for name, expected in names.items():
            assert digest(base/name) == expected, name
    archive = root/'optimized/data/raw/hetrec2011-upgrade-20261013/hetrec2011-movielens-2k-v2.zip'
    assert digest(archive) == prepared['archive_sha256'] == report['archive_sha256']
    raw = verify_raw(directory, archive)
    assert report['predictions_sha256'] == digest(directory/'test_predictions.json')
    assert report['frozen_sha256'] == digest(directory/'frozen.json')
    assert (directory/'test_predictions.json').stat().st_mtime_ns >= (directory/'frozen.json').stat().st_mtime_ns
    assert (directory/'test_scores.json').stat().st_mtime_ns >= (directory/'test_predictions.json').stat().st_mtime_ns
    for route in ('route1', 'route2', 'route3', 'route4'):
        selection = read(directory/route/'selection.json')
        assert verify_selection(selection, route) == frozen['selected'][route] == report['selected'][route]
        if route != 'route2':
            assert {r['seed'] for r in selection['training']} == {7, 42, 2026}
            for run in selection['training']:
                assert all(math.isfinite(r['loss']) and r['loss'] >= 0 for r in run['curves'])
                for epoch in ({'route1': (10, 20, 40), 'route3': (10, 20, 30), 'route4': (10, 20, 30)}[route]):
                    name = str(Path(route)/f'seed_{run["seed"]}_epoch_{epoch}.pt')
                    assert name in frozen['files_sha256']
                    weights = torch.load(directory/name, map_location='cpu')['state_dict']
                    assert weights and all(torch.isfinite(t).all() for t in weights.values())
                if route == 'route4':
                    stats = run['graph_training']
                    assert stats['item_weights_changed'] and stats['updates'] == 150 and stats['triples_per_update'] == 8192
                    assert all(math.isfinite(r['loss']) and r['loss'] > 0 for r in stats['curves'])
    public = {r['episode_id']: r for r in read(directory/'test_public.json')}
    private = {r['episode_id']: r for r in read(directory/'test_private.json')}
    catalog = read(directory/'catalog.json')
    scores, predictions = read(directory/'test_scores.json'), read(directory/'test_predictions.json')
    by, means = verify_scores(scores, predictions, public, private, catalog)
    for system, users in by.items():
        summary = report['systems'][system]
        assert summary['independent_users'] == 300 and summary['seeds'] == list(SYSTEM_SEEDS[system])
        for m in METRICS:
            close(summary['metrics'][m], sum(v[m] for v in means[system].values())/300)
        for seed in SYSTEM_SEEDS[system]:
            assert summary['per_seed_hits'][str(seed)] == sum(r['metrics']['hit'] for rs in users.values() for r in rs if r['seed'] == seed)
        for h in (3, 5, 10):
            keys = [u for u in users if public[u]['history_n'] == h]
            assert len(keys) == summary['history'][str(h)]['n'] == 100
            close(summary['history'][str(h)]['hit'], sum(means[system][u]['hit'] for u in keys)/100)
    # Independent training-count control, without the train runner's CF cache.
    table = pq.read_table(directory/'graph70.parquet').to_pydict()
    popularity = Counter(table['item_id'])
    item_users, user_items = defaultdict(set), defaultdict(set)
    for u, i in zip(table['user_id'], table['item_id']):
        item_users[i].add(u); user_items[u].add(i)
    cosine_cache = {}
    cf_checked = 0
    for row in predictions:
        if row['system'] == 'popular':
            allowed = set(map(int, catalog)) - set(public[row['episode_id']]['liked_item_ids'])
            assert row['item_ids'] == sorted(allowed, key=lambda i: (-popularity[i], i))[:10]
        elif row['system'] == 'itemcf':
            liked = public[row['episode_id']]['liked_item_ids']
            for seed in liked:
                if seed not in cosine_cache:
                    counts = Counter()
                    for u in sorted(item_users[seed]):
                        counts.update(user_items[u])
                    cosine_cache[seed] = {i: counts[i]/math.sqrt(popularity[i]*popularity[seed])
                                          for i in range(1, len(catalog)+1)}
            allowed = set(map(int, catalog)) - set(liked)
            values = {i: sum(cosine_cache[seed][i] for seed in liked)/len(liked) for i in allowed}
            assert row['item_ids'] == sorted(allowed, key=lambda i: (-values[i], i))[:10]
            cf_checked += 1
    assert cf_checked == 300
    expected_names = {r+'_minus_'+b for r in ('route1', 'route2', 'route3', 'route4') for b in ('popular', 'c1_weighted')}
    assert set(report['comparisons']) == expected_names
    for name, comparison in report['comparisons'].items():
        a, b = name.split('_minus_')
        rn = ('route1', 'route2', 'route3', 'route4').index(a)
        bn = ('popular', 'c1_weighted').index(b)
        assert comparison['paired_users'] == 300 and comparison['unit'] == 'user mean across seeds'
        keys = sorted(public)
        for mn, m in enumerate(METRICS):
            delta = [means[a][u][m]-means[b][u][m] for u in keys]
            close(comparison['metrics'][m]['mean'], sum(delta)/300)
            interval = bootstrap_interval(delta, 2026101307+100*rn+10*bn+mn)
            for actual, expected in zip(comparison['metrics'][m]['ci95'], interval):
                close(actual, expected)
        delta = [means[a][u]['hit']-means[b][u]['hit'] for u in keys]
        rng = random.Random(2026101310+rn*10+bn)
        observed = abs(sum(delta))
        extremes = sum(abs(sum(d*((-1, 1)[rng.randrange(2)]) for d in delta)) >= observed-1e-12
                       for _ in range(10000))
        close(comparison['permutation_p'], (extremes+1)/10001)
    running = 0.
    for n, name in enumerate(sorted(report['comparisons'], key=lambda k: report['comparisons'][k]['permutation_p'])):
        comp = report['comparisons'][name]
        running = max(running, min(1., (8-n)*comp['permutation_p']))
        close(comp['hit_holm_p'], running)
    winners = [r for r in ('route1', 'route2', 'route3', 'route4') if
               report['comparisons'][r+'_minus_popular']['metrics']['hit']['ci95'][0] > 0 and
               report['comparisons'][r+'_minus_popular']['hit_holm_p'] < .05]
    assert report['routes_verified_above_popular'] == winners
    assert report['api_requests'] == 0 and report['default_changed'] is False and report['strict_agent_tasks_evaluated'] is False
    result = {'status': 'VERIFIED', **raw, 'independent_quality_scores': len(scores),
              'independent_popular_rankings': 300, 'configurations_verified': 66,
              'independent_itemcf_rankings': cf_checked,
              'all_training_checkpoints_verified': True, 'source_and_frozen_hashes_verified': True,
              'user_mean_seed_aggregation_verified': True, 'api_requests': 0,
              'independent_bootstrap_permutation_holm_verified': True,
              'routes_verified_above_popular': winners, 'auditor_sha256': digest(Path(__file__)),
              'scoring_helper_sha256': digest(Path(__file__).with_name('audit_three_way_comparison.py')),
              'report_sha256': digest(directory/'report.json')}
    with output.open('x', encoding='utf-8') as out:
        json.dump(result, out, indent=2); out.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.directory, args.output), indent=2))
