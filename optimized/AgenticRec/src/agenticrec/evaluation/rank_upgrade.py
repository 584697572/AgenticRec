"""Finite four-route ranking study on a newly downloaded official HetRec v2."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import io
import itertools
import json
import math
from pathlib import Path
import random
import time
import zipfile

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from .simple_baselines import ROOT, digest, load, write_new, quality, paired_interval
from .rank_upgrade_models import (CFIndex, FeatureBuilder, PopularResidual, ShortSequence,
                                 GraphSession, padded_histories, multi_positive_loss,
                                 mixed_scores, ordered_ids)
from ..ranking.constraints import Catalog, Item
from ..retrieval.sources import CandidateRetriever
from ..models.lightgcn import normalized_bipartite_graph, lightgcn_loss
from ..models.lightgcn_residual import ResidualLightGCN

ARCHIVE = ROOT/'optimized/data/raw/hetrec2011-upgrade-20261013/hetrec2011-movielens-2k-v2.zip'
ARCHIVE_SHA = '6f03917faa2a2914a4f100a130f4fd40412785b0629950ef72e605406c91f52e'
SEEDS = (7, 42, 2026)
ALPHAS = (.25, .5, 1.)
BASELINES = ('popular', 'itemcf', 'content', 'c1_rrf', 'c1_weighted')
EPOCHS = {'route1': (10, 20, 40), 'route3': (10, 20, 30), 'route4': (10, 20, 30)}


def check_sources(directory):
    contract = load(directory/'prepared.json')
    if digest(ARCHIVE) != ARCHIVE_SHA:
        raise ValueError('Official archive changed')
    for name, expected in contract['source_sha256'].items():
        if digest(ROOT/name) != expected:
            raise ValueError('Source changed: '+name)
    for name, expected in contract['input_sha256'].items():
        if digest(directory/name) != expected:
            raise ValueError('Prepared input changed: '+name)
    return contract


def archive_rows(archive=ARCHIVE):
    with zipfile.ZipFile(archive) as z:
        def rows(name):
            # Original metadata contains Latin-1 names; do not drop characters.
            return csv.DictReader(io.StringIO(z.read(name).decode('latin-1')), delimiter='\t')
        metadata = {}
        for r in rows('movies.dat'):
            metadata[int(r['id'])] = {'raw_id': int(r['id']), 'title': r['title'],
                'year': int(r['year']) if r['year'].isdigit() and int(r['year']) > 0 else None, 'genres': []}
        for r in rows('movie_genres.dat'):
            metadata[int(r['movieID'])]['genres'].append(r['genre'])
        ratings = sorted((int(r['timestamp']), int(r['userID']), int(r['movieID']), float(r['rating']))
                         for r in rows('user_ratedmovies-timestamps.dat'))
        license_text = z.read('readme.txt').decode('latin-1')
    if len(ratings) != 855598 or len({(u, i) for ts, u, i, r in ratings}) != len(ratings):
        raise ValueError('Unexpected official ratings count or duplicates')
    if any(ts <= 0 or u <= 0 or i not in metadata or not .5 <= r <= 5 or r*2 != int(r*2)
           for ts, u, i, r in ratings):
        raise ValueError('Invalid official data')
    if 'non-commercial use' not in license_text:
        raise ValueError('Expected research-use terms missing')
    return metadata, ratings, license_text


def temporal_cases(ratings, mapping, lo, hi):
    seen, histories, targets = defaultdict(set), defaultdict(list), defaultdict(set)
    for ts, u, raw, rating in ratings:
        if ts < lo:
            seen[u].add(raw)
            if rating >= 4 and raw in mapping:
                histories[u].append((ts, mapping[raw]))
    for ts, u, raw, rating in ratings:
        if lo <= ts < hi and rating >= 4 and raw in mapping and raw not in seen[u]:
            targets[u].add(mapping[raw])
    eligible = sorted(u for u in targets if len(histories[u]) >= 10)
    return histories, targets, eligible


def prepare(directory):
    if directory.exists(): raise FileExistsError('Use a fresh study directory')
    if digest(ARCHIVE) != ARCHIVE_SHA: raise ValueError('Archive identity mismatch')
    metadata, ratings, license_text = archive_rows()
    t50, t70, t85 = [ratings[int(len(ratings)*q)][0] for q in (.5, .7, .85)]
    warm = sorted({i for ts, u, i, r in ratings if ts < t70 and r >= 4})
    mapping = {raw: n+1 for n, raw in enumerate(warm)}
    catalog = {mapping[raw]: dict(metadata[raw], item_id=mapping[raw]) for raw in warm}
    users = sorted({u for ts, u, i, r in ratings if ts < t70 and r >= 4})
    umap = {u: n+1 for n, u in enumerate(users)}
    directory.mkdir(parents=True)
    for name, cutoff in [('graph50', t50), ('graph70', t70)]:
        records = [(umap[u], mapping[i], ts) for ts, u, i, r in ratings if ts < cutoff and r >= 4]
        table = pa.table({'user_id': [r[0] for r in records], 'item_id': [r[1] for r in records],
                          'timestamp_ms': [r[2] for r in records]})
        pq.write_table(table, directory/(name+'.parquet'))
    write_new(directory/'catalog.json', catalog)
    write_new(directory/'id_map_private.json', {'item_to_model': mapping, 'user_to_model': umap})
    phases = {name: temporal_cases(ratings, mapping, lo, hi) for name, lo, hi in
              [('training', t50, t70), ('development', t70, t85), ('test', t85, 10**20)]}
    rng = random.Random(2026101306)
    test_users = phases['test'][2][:]; rng.shuffle(test_users); test_users = test_users[:300]
    dev_users = [u for u in phases['development'][2] if u not in test_users]
    rng.shuffle(dev_users); dev_users = dev_users[:180]
    train_users = [u for u in phases['training'][2] if u not in set(test_users+dev_users)]
    rng.shuffle(train_users); train_users = train_users[:600]
    if len(test_users) != 300 or len(dev_users) != 180 or len(train_users) < 100:
        raise ValueError('Insufficient independent users')
    for phase, selected in [('development', dev_users), ('test', test_users)]:
        public, private = [], []
        histories, targets, _ = phases[phase]
        for n, u in enumerate(selected):
            h = (3, 5, 10)[n % 3]
            liked = [i for ts, i in histories[u][-h:]]
            public.append({'episode_id': phase+'-'+str(n).zfill(3), 'liked_item_ids': liked, 'history_n': h})
            private.append({'episode_id': public[-1]['episode_id'], 'evaluator_raw_user_id': u,
                'relevant_item_ids': sorted(targets[u]), 'history_n': h})
        write_new(directory/(phase+'_public.json'), public)
        write_new(directory/(phase+'_private.json'), private)
    histories, targets, _ = phases['training']
    training = [{'liked_item_ids': [i for ts, i in histories[u][-h:]], 'relevant_item_ids': sorted(targets[u])}
                for u in train_users for h in (3, 5, 10)]
    write_new(directory/'train_examples.json', training)
    write_new(directory/'training_users_private.json', train_users)
    assert not set(train_users)&set(dev_users+test_users) and not set(dev_users)&set(test_users)
    sources = list((ROOT/'optimized/AgenticRec/src/agenticrec').rglob('*.py'))
    sources.append(ROOT/'optimized/reports/rank_upgrade_protocol_20261013.md')
    inputs = {str(p.relative_to(directory)): digest(p) for p in directory.iterdir() if p.is_file()}
    result = {'status': 'PREPARED', 'dataset': 'official HetRec2011 MovieLens 2k v2',
              'archive_sha256': ARCHIVE_SHA, 'catalog_items': len(warm), 'raw_ratings': len(ratings),
              'cutoffs_ms': [t50, t70, t85], 'cutoffs_utc': [datetime.fromtimestamp(t/1000, timezone.utc).isoformat() for t in (t50, t70, t85)],
              'development_users': 180, 'test_users': 300, 'training_users': len(train_users),
              'training_examples': len(training), 'seeds': list(SEEDS), 'api_requests': 0,
              'user_groups_disjoint': True, 'license': 'non-commercial research; raw data not redistributed',
              'input_sha256': inputs, 'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in sources}}
    write_new(directory/'prepared.json', result)
    return result


def environment(directory, half=False, iuf=False):
    check_sources(directory)
    catalog = {int(k): v for k, v in load(directory/'catalog.json').items()}
    graph = directory/('graph50.parquet' if half else 'graph70.parquet')
    table = pq.read_table(graph)
    edges = list(zip(table['user_id'].to_pylist(), table['item_id'].to_pylist()))
    base = CandidateRetriever(Catalog([Item(i, r['title'], frozenset(r['genres']), r['year'])
                                       for i, r in catalog.items()]), edges)
    cache = directory/'cache'/('index50.pt' if half else 'index70_iuf.pt' if iuf else 'index70.pt')
    if cache.exists():
        value = torch.load(cache, map_location='cpu')
        if value['graph_sha256'] != digest(graph): raise ValueError('Graph cache mismatch')
        index = CFIndex.__new__(CFIndex)
        index.popularity, index.co, index.norm = value['popularity'], value['co'], value['norm']
        index.neighbor_cache = {}
    else:
        index = CFIndex(len(catalog), edges, iuf=iuf)
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'graph_sha256': digest(graph), 'popularity': index.popularity,
                    'co': index.co, 'norm': index.norm}, cache)
    features = FeatureBuilder(index, base.content_vectors, {i: r['genres'] for i, r in catalog.items()})
    return {'catalog': catalog, 'base': base, 'index': index, 'features': features, 'edges': edges}


def rank(method, liked, env, *, model=None, config=None):
    if method not in (*BASELINES, 'route1', 'route2', 'route3', 'route4'):
        raise ValueError('Unknown ranking route')
    if not liked or len(liked) != len(set(liked)) or any(i not in env['catalog'] for i in liked):
        raise ValueError('Only unique public catalog likes are allowed')
    index = env['index']
    if method == 'popular': return ordered_ids(index.popularity, liked)
    if method == 'itemcf': return ordered_ids(index.cosine(liked), liked)
    if method == 'route2':
        v = index.neighbors(liked, config['neighbors'], config['shrink'])
        return ordered_ids(mixed_scores(v, index.popularity, config['alpha']), liked)
    if method in ('content', 'c1_rrf', 'c1_weighted', 'route1'):
        cf, content, routes, pool = env['features'].context(liked)
        if method == 'content': return ordered_ids(content, liked)
        if method == 'route1':
            x = env['features'].features(liked, pool)
            model.eval()
            with torch.no_grad(): scores = model(x, config['alpha']).tolist()
            return [pool[n] for n in sorted(range(len(pool)), key=lambda n: (-scores[n], pool[n]))[:10]]
        weights = (1., 1., 1.) if method == 'c1_rrf' else (1., .5, .25)
        scores = defaultdict(float)
        for w, route in zip(weights, routes):
            for r, i in enumerate(route, 1): scores[i] += w/(60+r)
        pop = index.popularity.tolist()
        return sorted((i for i in env['catalog'] if i not in liked), key=lambda i: (-scores[i], -pop[i-1], i))[:10]
    history = padded_histories([{'liked_item_ids': liked}])
    model.eval()
    with torch.no_grad():
        scores = model(history, adapted=config.get('adapted', True)) if method == 'route4' else model(history)
    return ordered_ids(mixed_scores(scores[0], index.popularity, config['alpha']), liked)


def validation(method, env, public, private, model=None, config=None):
    labels = {r['episode_id']: r['relevant_item_ids'] for r in private}
    values = []
    for row in public:
        ids = rank(method, row['liked_item_ids'], env, model=model, config=config)
        values.append(quality(ids, labels[row['episode_id']]))
    return {m: sum(v[m] for v in values)/len(values) for m in ('hit', 'ndcg', 'recall', 'mrr')}


def finish_selection(directory, route, candidates, details):
    grouped = defaultdict(list)
    for row in candidates: grouped[json.dumps(row['config'], sort_keys=True)].append(row)
    expected = 1 if route == 'route2' else len(SEEDS)
    if any(len(v) != expected for v in grouped.values()): raise ValueError('Incomplete seeds')
    summaries = []
    for key, rows in grouped.items():
        summaries.append({'config': json.loads(key), 'metrics': {m: sum(r['metrics'][m] for r in rows)/expected
                          for m in ('hit', 'ndcg', 'recall', 'mrr')}})
    best = max(enumerate(summaries), key=lambda p: (p[1]['metrics']['hit'], p[1]['metrics']['ndcg'], -p[0]))[1]
    result = {'status': 'DEVELOPED', 'route': route, 'selected': best, 'candidates': candidates,
              'selection': 'development mean Hit across seeds; then mean NDCG; fixed order', 'training': details,
              'test_opened': False, 'api_requests': 0}
    write_new(directory/route/'selection.json', result)
    return {'route': route, 'selected': best, 'training': details}


def features_for_training(directory, env):
    examples = load(directory/'train_examples.json')
    xs, ys, kept = [], [], []
    for n, row in enumerate(examples):
        liked = row['liked_item_ids']
        _, _, _, pool = env['features'].context(liked)
        y = torch.tensor([float(i in row['relevant_item_ids']) for i in pool])
        if not y.any(): continue
        if set(liked)&set(row['relevant_item_ids']): raise ValueError('Training input/target overlap')
        xs.append(env['features'].features(liked, pool)); ys.append(y); kept.append(len(pool))
        if n%300 == 0: print(json.dumps({'stage': 'training_features', 'groups': n}), flush=True)
    flattened = torch.cat(xs)
    means, scales = flattened.mean(0), flattened.std(0, unbiased=False).clamp_min(1e-5)
    x = torch.zeros((len(xs), 300, 10)); y = torch.zeros((len(xs), 300)); mask = torch.zeros((len(xs), 300), dtype=torch.bool)
    for n, (features, target) in enumerate(zip(xs, ys)):
        x[n, :len(features)] = features; y[n, :len(target)] = target; mask[n, :len(target)] = True
    return x, y, mask, means, scales, {'total_train_queries': len(examples), 'retained_queries': len(xs),
          'no_positive_in_unmodified_pool': len(examples)-len(xs), 'future_target_insertion': False}


def route1(directory):
    check_sources(directory); target = directory/'route1'
    if target.exists(): raise FileExistsError('Earlier route evidence cannot be overwritten')
    target.mkdir()
    half = environment(directory, half=True)
    x, y, mask, means, scales, stats = features_for_training(directory, half)
    del half
    env = environment(directory)
    public, private = load(directory/'development_public.json'), load(directory/'development_private.json')
    candidates, details = [], []
    for seed in SEEDS:
        torch.manual_seed(seed); model = PopularResidual(means, scales)
        opt = torch.optim.Adam(model.parameters(), lr=.002, weight_decay=1e-4)
        curves = []
        for epoch in range(1, 41):
            model.train(); losses = []
            for batch in torch.randperm(len(x)).split(32):
                scores = model(x[batch]).masked_fill(~mask[batch], -1e9)
                loss = -(torch.log_softmax(scores, 1)*y[batch]/y[batch].sum(1, keepdim=True)).sum(1).mean()
                opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss.detach()))
            if epoch in EPOCHS['route1']:
                torch.save({'state_dict': model.state_dict(), 'means': means, 'scales': scales}, target/f'seed_{seed}_epoch_{epoch}.pt')
                curves.append({'epoch': epoch, 'loss': sum(losses)/len(losses)})
                for alpha in ALPHAS:
                    config = {'epoch': epoch, 'alpha': alpha}
                    candidates.append({'seed': seed, 'config': config, 'metrics': validation('route1', env, public, private, model, config)})
                print(json.dumps({'route': 'route1', 'seed': seed, 'epoch': epoch, 'loss': curves[-1]['loss']}), flush=True)
        details.append({'seed': seed, 'epochs': 40, 'curves': curves, **stats})
    return finish_selection(directory, 'route1', candidates, details)


def route2(directory):
    check_sources(directory); target = directory/'route2'
    if load(directory/'route1/selection.json')['status'] != 'DEVELOPED':
        raise ValueError('Route 1 must finish first')
    if target.exists(): raise FileExistsError('Earlier route evidence cannot be overwritten')
    target.mkdir()
    public, private = load(directory/'development_public.json'), load(directory/'development_private.json')
    candidates = []
    for iuf in (False, True):
        env = environment(directory, iuf=iuf)
        for k, shrink, alpha in itertools.product((50, 100, 200), (10., 50.), ALPHAS):
            config = {'iuf': iuf, 'neighbors': k, 'shrink': shrink, 'alpha': alpha}
            candidates.append({'seed': None, 'config': config, 'metrics': validation('route2', env, public, private, config=config)})
        print(json.dumps({'route': 'route2', 'iuf': iuf, 'configs_done': len(candidates)}), flush=True)
    return finish_selection(directory, 'route2', candidates, {'configs': 36, 'training': 'train70 binary graph only'})


def graph_pretrain(env, seed):
    torch.manual_seed(seed)
    edges = env['edges']; n_users = max(u for u, i in edges)
    graph = normalized_bipartite_graph(n_users, len(env['catalog']), edges)
    model = ResidualLightGCN(n_users, len(env['catalog']), 32, 2, graph, (1/3, 1/3, 1/3))
    optimizer = torch.optim.Adam(model.parameters(), lr=.005)
    pairs = torch.tensor(edges, dtype=torch.long)
    seen = defaultdict(set)
    for u, i in edges: seen[u].add(i)
    rng = random.Random(seed); curves = []
    initial = model.items.weight.detach().clone()
    for step in range(150):
        sample = pairs[torch.randint(len(pairs), (8192,))]
        users, positives = sample[:, 0], sample[:, 1]
        negatives = [rng.randint(1, len(env['catalog'])) for _ in range(len(sample))]
        for n, u in enumerate(users.tolist()):
            while negatives[n] in seen[u]: negatives[n] = rng.randint(1, len(env['catalog']))
        loss = lightgcn_loss(model, users, positives, torch.tensor(negatives), 1e-4)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step in (0, 49, 99, 149):
            curves.append({'step': step+1, 'loss': float(loss.detach())})
            print(json.dumps({'route': 'route4', 'stage': 'LightGCN_BPR', 'seed': seed, **curves[-1]}), flush=True)
    with torch.no_grad(): _, items = model.propagate()
    return items.detach(), {'updates': 150, 'triples_per_update': 8192, 'layers': 2, 'dim': 32,
                            'item_weights_changed': not torch.equal(initial, model.items.weight), 'curves': curves}


def neural_route(directory, route):
    check_sources(directory); target = directory/route
    previous = 'route2' if route == 'route3' else 'route3'
    if load(directory/previous/'selection.json')['status'] != 'DEVELOPED':
        raise ValueError('Previous route must finish first')
    if target.exists(): raise FileExistsError('Earlier route evidence cannot be overwritten')
    target.mkdir(); env = environment(directory)
    examples = load(directory/'train_examples.json')
    histories = padded_histories(examples)
    targets = torch.zeros((len(examples), len(env['catalog'])))
    for n, row in enumerate(examples): targets[n, [i-1 for i in row['relevant_item_ids']]] = 1
    public, private = load(directory/'development_public.json'), load(directory/'development_private.json')
    candidates, details = [], []
    for seed in SEEDS:
        torch.manual_seed(seed)
        if route == 'route4':
            vectors, graph_stats = graph_pretrain(env, seed)
            torch.save({'item_vectors': vectors, 'training': graph_stats}, target/f'graph_seed_{seed}.pt')
            model = GraphSession(vectors)
            for alpha in ALPHAS:
                config = {'epoch': 0, 'alpha': alpha, 'adapted': False}
                candidates.append({'seed': seed, 'config': config, 'metrics': validation(route, env, public, private, model, config)})
        else:
            model = ShortSequence(len(env['catalog'])); graph_stats = None
        optimizer = torch.optim.Adam(model.parameters(), lr=.002, weight_decay=1e-4)
        curves = []
        for epoch in range(1, 31):
            model.train(); losses = []
            for batch in torch.randperm(len(examples)).split(32):
                logits = model(histories[batch])
                loss = multi_positive_loss(logits, targets[batch], histories[batch])
                optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
            if epoch in EPOCHS[route]:
                torch.save({'state_dict': model.state_dict()}, target/f'seed_{seed}_epoch_{epoch}.pt')
                curves.append({'epoch': epoch, 'loss': sum(losses)/len(losses)})
                for alpha in ALPHAS:
                    config = {'epoch': epoch, 'alpha': alpha}
                    if route == 'route4': config['adapted'] = True
                    candidates.append({'seed': seed, 'config': config, 'metrics': validation(route, env, public, private, model, config)})
                print(json.dumps({'route': route, 'seed': seed, 'epoch': epoch, 'loss': curves[-1]['loss']}), flush=True)
        details.append({'seed': seed, 'epochs': 30, 'train_queries': len(examples), 'curves': curves, 'graph_training': graph_stats})
    return finish_selection(directory, route, candidates, details)


def freeze(directory):
    check_sources(directory)
    selected = {r: load(directory/r/'selection.json')['selected'] for r in ('route1', 'route2', 'route3', 'route4')}
    files = {str(p.relative_to(directory)): digest(p) for p in directory.rglob('*') if p.is_file()}
    write_new(directory/'frozen.json', {'status': 'FROZEN', 'selected': selected, 'files_sha256': files,
                                     'api_requests': 0, 'test_opened': False})
    return {'status': 'FROZEN', 'selected': selected}


def verify_frozen(directory):
    check_sources(directory)
    value = load(directory/'frozen.json')
    for name, expected in value['files_sha256'].items():
        if digest(directory/name) != expected: raise ValueError('Frozen artifact changed: '+name)
    return value


def load_model(directory, route, seed, config, n_items):
    if route == 'route1':
        saved = torch.load(directory/route/f'seed_{seed}_epoch_{config["epoch"]}.pt', map_location='cpu')
        model = PopularResidual(saved['means'], saved['scales'])
    elif route == 'route3':
        saved = torch.load(directory/route/f'seed_{seed}_epoch_{config["epoch"]}.pt', map_location='cpu')
        model = ShortSequence(n_items)
    else:
        saved_graph = torch.load(directory/route/f'graph_seed_{seed}.pt', map_location='cpu')
        model = GraphSession(saved_graph['item_vectors'])
        if not config['adapted']: return model.eval()
        saved = torch.load(directory/route/f'seed_{seed}_epoch_{config["epoch"]}.pt', map_location='cpu')
    model.load_state_dict(saved['state_dict']); model.eval()
    return model


def evaluate(directory):
    frozen = verify_frozen(directory)
    public = load(directory/'test_public.json')
    env = environment(directory)
    records = []
    for system in BASELINES:
        for row in public:
            ids = rank(system, row['liked_item_ids'], env)
            records.append({'episode_id': row['episode_id'], 'history_n': row['history_n'],
                            'system': system, 'seed': None, 'item_ids': ids})
    for route in ('route1', 'route2', 'route3', 'route4'):
        config = frozen['selected'][route]['config']
        current = environment(directory, iuf=config['iuf']) if route == 'route2' else env
        for seed in ((None,) if route == 'route2' else SEEDS):
            model = None if route == 'route2' else load_model(directory, route, seed, config, len(env['catalog']))
            for row in public:
                ids = rank(route, row['liked_item_ids'], current, model=model, config=config)
                records.append({'episode_id': row['episode_id'], 'history_n': row['history_n'],
                                'system': route, 'seed': seed, 'item_ids': ids})
        print(json.dumps({'stage': 'test_predictions', 'route': route, 'records': len(records)}), flush=True)
    # All frozen predictions precede any test label parsing.
    write_new(directory/'test_predictions.json', records)
    private = {r['episode_id']: r for r in load(directory/'test_private.json')}
    pubs = {r['episode_id']: r for r in public}
    for row in records:
        ids = row['item_ids']
        if len(ids) != 10 or len(set(ids)) != 10 or set(ids)&set(pubs[row['episode_id']]['liked_item_ids']):
            raise ValueError('Invalid public delivery')
        if any(i not in env['catalog'] for i in ids): raise ValueError('Unknown catalog identity')
        row['metrics'] = quality(ids, private[row['episode_id']]['relevant_item_ids'])
    if len(records) != 4500: raise ValueError('Incomplete formal comparison')
    write_new(directory/'test_scores.json', records)
    return {'status': 'EVALUATED', 'predictions': 4500, 'independent_users': 300, 'api_requests': 0}


def report(directory):
    frozen = verify_frozen(directory)
    rows = load(directory/'test_scores.json')
    systems = (*BASELINES, 'route1', 'route2', 'route3', 'route4')
    by = {s: defaultdict(list) for s in systems}
    for row in rows: by[row['system']][row['episode_id']].append(row)
    summary = {}
    per_user = {}
    for s, users in by.items():
        expected = 3 if s in ('route1', 'route3', 'route4') else 1
        if len(users) != 300 or any(len(v) != expected for v in users.values()): raise ValueError('Incomplete user/seed pairing')
        per_user[s] = {u: {m: sum(r['metrics'][m] for r in rs)/expected for m in ('hit', 'recall', 'ndcg', 'mrr')}
                       for u, rs in users.items()}
        summary[s] = {'independent_users': 300, 'seeds': list(SEEDS) if expected == 3 else [None],
            'metrics': {m: sum(v[m] for v in per_user[s].values())/300 for m in ('hit', 'recall', 'ndcg', 'mrr')},
            'per_seed_hits': {str(seed): int(sum(r['metrics']['hit'] for rs in users.values() for r in rs if r['seed'] == seed))
                              for seed in (SEEDS if expected == 3 else (None,))},
            'history': {str(h): {'n': 100, 'hit': sum(per_user[s][u]['hit'] for u, rs in users.items() if rs[0]['history_n'] == h)/100}
                        for h in (3, 5, 10)}}
    comparisons = {}
    for rn, route in enumerate(('route1', 'route2', 'route3', 'route4')):
        for bn, baseline in enumerate(('popular', 'c1_weighted')):
            keys = sorted(per_user[route]); assert keys == sorted(per_user[baseline])
            differences = {m: [per_user[route][u][m]-per_user[baseline][u][m] for u in keys]
                           for m in ('hit', 'recall', 'ndcg', 'mrr')}
            cis = {m: paired_interval(delta, 2026101307+100*rn+10*bn+mn) for mn, (m, delta) in enumerate(differences.items())}
            delta = differences['hit']; observed = abs(sum(delta))
            rng = random.Random(2026101310+rn*10+bn)
            extremes = sum(abs(sum(d*rng.choice((-1, 1)) for d in delta)) >= observed-1e-12 for _ in range(10000))
            comparisons[route+'_minus_'+baseline] = {'metrics': cis, 'paired_users': 300,
                'permutation_p': (extremes+1)/10001, 'resamples': 10000, 'unit': 'user mean across seeds'}
    running = 0.
    for n, name in enumerate(sorted(comparisons, key=lambda k: comparisons[k]['permutation_p'])):
        running = max(running, min(1., (len(comparisons)-n)*comparisons[name]['permutation_p']))
        comparisons[name]['hit_holm_p'] = running
    winners = [r for r in ('route1', 'route2', 'route3', 'route4') if
               comparisons[r+'_minus_popular']['metrics']['hit']['ci95'][0] > 0 and
               comparisons[r+'_minus_popular']['hit_holm_p'] < .05]
    result = {'status': 'VERIFIED', 'dataset': load(directory/'prepared.json')['dataset'],
              'api_requests': 0, 'new_dataset_download_bytes': 18856822, 'training_seeds': list(SEEDS),
              'test_users': 300, 'test_predictions': len(rows), 'systems': summary, 'comparisons': comparisons,
              'selected': frozen['selected'], 'routes_verified_above_popular': winners,
              'default_changed': False, 'strict_agent_tasks_evaluated': False,
              'archive_sha256': ARCHIVE_SHA, 'predictions_sha256': digest(directory/'test_predictions.json'),
              'frozen_sha256': digest(directory/'frozen.json'), 'scope': 'new-dataset offline ranking, not original app or paper reproduction'}
    write_new(directory/'report.json', result)
    return result


def main():
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'route1', 'route2', 'route3', 'route4', 'freeze', 'evaluate', 'report'))
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    operations = {'prepare': prepare, 'route1': route1, 'route2': route2,
                  'route3': lambda d: neural_route(d, 'route3'), 'route4': lambda d: neural_route(d, 'route4'),
                  'freeze': freeze, 'evaluate': evaluate, 'report': report}
    start = time.perf_counter(); value = operations[args.command](args.directory)
    print(json.dumps({'elapsed_seconds': time.perf_counter()-start, **value}, indent=2))


if __name__ == '__main__': main()
