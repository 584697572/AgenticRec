"""Train-only diagnostics; no optimizer, holdout evaluator or serving changes."""
import argparse
import bisect
import builtins
from collections import Counter, defaultdict
from contextlib import contextmanager
import io
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess

import pyarrow.parquet as pq
import torch

from .data import digest
from .lightgcn_optimization import build_model, code_identity
from .training import ROOT, load_train_contract


STRATEGIES = ('uniform', 'median_pool', 'hardest_pool')


def source_paths():
    return list(Path(__file__).parent.rglob('*.py'))


def write_once(path, value):
    serialized = json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n'
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        f.write(serialized)
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()


def validate_plan(plan):
    # Fixed preregistration, not a hyperparameter-search interface.
    expected = {'schema_version': 1, 'run_id': 'lightgcn_training_diagnostics_20261011',
                'seed': 42, 'model_seeds': [7, 42, 2026],
                'models': ['gcn_edge_uniform', 'bpr_edge'], 'positives_per_user': 8,
                'negative_pool_size': 8, 'low_rating_max': 2, 'low_samples_per_user': 8,
                'easy_gradient_below': .05, 'time_min_positives': 20, 'time_min_group': 5,
                'top_k': 10, 'torch_threads': 2, 'score_batch_size': 128,
                'train_only': True, 'training_runs': 0, 'remote_api_requests': 0,
                'downloads': 0, 'source_plan': 'AgenticRec/configs/lightgcn_optimization_20261011.json'}
    if plan != expected or any(type(plan[k]) is not type(v) for k, v in expected.items()):
        raise ValueError('Only the fixed train-only diagnostic plan is allowed')
    return plan


@contextmanager
def train_only_files(allowed):
    """Whitelist Python and Arrow reads; controlled code never bypasses this guard."""
    paths = {str(Path(p).resolve()) for p in allowed}
    reads = set()
    original_open, original_io = builtins.open, io.open
    original_table, original_schema = pq.read_table, pq.read_schema

    def check(file):
        if not isinstance(file, (str, bytes, os.PathLike)):
            raise PermissionError('Only whitelisted file paths are accepted')
        name = str(Path(os.fsdecode(file)).resolve())
        if name not in paths:
            raise PermissionError('Train-only diagnostic blocked an unregistered file')
        reads.add(name)

    def guarded_open(opener):
        def guarded(file, mode='r', *args, **kwargs):
            if not any(c in mode for c in 'wax+'):
                check(file)
            return opener(file, mode, *args, **kwargs)
        return guarded

    def arrow_read(reader):
        def guarded(file, *args, **kwargs):
            check(file)
            return reader(file, *args, **kwargs)
        return guarded

    builtins.open, io.open = guarded_open(original_open), guarded_open(original_io)
    pq.read_table, pq.read_schema = arrow_read(original_table), arrow_read(original_schema)
    try:
        yield reads
    finally:
        builtins.open, io.open = original_open, original_io
        pq.read_table, pq.read_schema = original_table, original_schema


def pair_summary(positive, negative, easy_threshold):
    if len(positive) != len(negative):
        raise ValueError('Pair lengths differ')
    if not positive:
        return None
    margins = [p-n for p, n in zip(positive, negative)]
    if any(not math.isfinite(v) for v in margins):
        raise ValueError('Nonfinite pair score')
    losses = [max(-v, 0)+math.log1p(math.exp(-abs(v))) for v in margins]
    gradients = [math.exp(-v)/(1+math.exp(-v)) if v >= 0 else 1/(1+math.exp(v)) for v in margins]
    return {'pairs': len(margins), 'margin': statistics.mean(margins),
            'loss': statistics.mean(losses), 'gradient': statistics.mean(gradients),
            'easy_fraction': sum(g < easy_threshold for g in gradients)/len(gradients),
            'negative_outranks_positive': sum(v < 0 for v in margins)/len(margins)}


def choose_negatives(pool, scores):
    if not pool or len(pool) != len(scores) or any(not math.isfinite(s) for s in scores):
        raise ValueError('Invalid fixed negative pool')
    ordered = sorted(zip(scores, pool))
    return {'uniform': pool[0], 'median_pool': ordered[(len(pool)-1)//2][1],
            'hardest_pool': ordered[-1][1]}


def sample_users(n_items, seen, positive, low, positives_per_user, pool_size, low_count, seed):
    output = []
    for user in sorted(positive):
        rng = random.Random(seed*1000003+user)
        unrated = sorted(set(range(1, n_items+1))-set(seen[user]))
        if not unrated or not set(low.get(user, [])) <= set(seen[user]):
            raise ValueError('No unknown candidate or low ratings outside seen history')
        items = sorted(positive[user])
        pos = rng.sample(items, min(positives_per_user, len(items)))
        pools = [rng.sample(unrated, min(pool_size, len(unrated))) for _ in pos]
        low_ids = sorted(low.get(user, []))
        low_ids = rng.sample(low_ids, min(low_count, len(low_ids)))
        output.append({'user_id': user, 'positive_ids': pos, 'pools': pools, 'low_ids': low_ids})
    return output


def genre_distance(left, right, genres):
    def distribution(items):
        if not items:
            raise ValueError('Empty genre sample')
        counts = Counter()
        for item in items:
            labels = genres[item]
            if not labels:
                raise ValueError('Missing official genre')
            for label in labels:
                counts[label] += 1/(len(items)*len(labels))
        return counts
    a, b = distribution(left), distribution(right)
    # Python hash randomization changes set order; stable order and fsum preserve
    # byte identity across interpreters as well as the mathematical definition.
    return math.fsum(abs(a[g]-b[g]) for g in sorted(set(a)|set(b)))/2


def temporal_row(user, records, genres, min_positives, min_group, seed):
    times = [t for _, _, t in records]
    days = Counter(t//86400 for t in times)
    positives = sorted((t, i) for i, r, t in records if r >= 4)
    row = {'user_id': user, 'positive_count': len(positives), 'rated_count': len(records),
           'span_days': (max(times)-min(times))/86400, 'distinct_days': len(days),
           'largest_day_fraction': max(days.values())/len(times),
           'split_status': 'INSUFFICIENT_POSITIVES', 'genre_tv': None, 'random_genre_tv': None}
    if len(positives) < min_positives:
        return row
    early_cut = positives[math.ceil(len(positives)/4)-1][0]
    recent_cut = positives[(3*len(positives))//4][0]
    if early_cut >= recent_cut:
        row['split_status'] = 'TIMESTAMP_TIES_NOT_SEPARABLE'
        return row
    early = [i for t, i in positives if t <= early_cut]
    recent = [i for t, i in positives if t >= recent_cut]
    if min(len(early), len(recent)) < min_group:
        row['split_status'] = 'INSUFFICIENT_TIME_GROUP'
        return row
    shuffled = [i for _, i in positives]
    random.Random(seed*1000003+user).shuffle(shuffled)
    row.update(split_status='SEPARABLE', early_count=len(early), recent_count=len(recent),
               early_max_timestamp=early_cut, recent_min_timestamp=recent_cut,
               genre_tv=genre_distance(early, recent, genres),
               random_genre_tv=genre_distance(shuffled[:len(early)], shuffled[-len(recent):], genres))
    return row


def _mean_fields(rows):
    if not rows:
        return None
    return {key: statistics.mean(row[key] for row in rows) for key in rows[0] if key != 'pairs'}


def aggregate_rows(rows, with_strata=True):
    low_rows = [r['low'] for r in rows if r['low'] is not None]
    result = {'users': len(rows), 'negative': {}, 'low': {'users': len(low_rows)}}
    if rows:
        for strategy in rows[0]['negative']:
            values = [r['negative'][strategy] for r in rows]
            result['negative'][strategy] = {**_mean_fields(values),
                                          'pairs': sum(v['pairs'] for v in values)}
    if low_rows:
        result['low'].update(pair=_mean_fields([r['pair'] for r in low_rows]),
                             high_score_fraction=statistics.mean(r['high_score_fraction'] for r in low_rows),
                             score_percentile=statistics.mean(r['score_percentile'] for r in low_rows),
                             rated_items=sum(r['count'] for r in low_rows))
    if with_strata:
        result['strata'] = {name: aggregate_rows([r for r in rows if lo <= r['positive_count'] < hi], False)
                            for name, lo, hi in [('5_19',5,20),('20_99',20,100),('100_plus',100,10**9)]}
    return result


def temporal_summary(rows):
    separated = [r for r in rows if r['split_status'] == 'SEPARABLE']
    deltas = [r['genre_tv']-r['random_genre_tv'] for r in separated]
    spans = sorted(r['span_days'] for r in rows)
    return {'users': len(rows), 'status_counts': dict(Counter(r['split_status'] for r in rows)),
            'median_span_days': statistics.median(spans),
            'span_at_least_30_days_fraction': sum(s >= 30 for s in spans)/len(rows),
            'span_at_most_one_day_fraction': sum(s <= 1 for s in spans)/len(rows),
            'at_least_half_ratings_same_day_fraction': sum(r['largest_day_fraction'] >= .5 for r in rows)/len(rows),
            'genre_users': len(separated), 'mean_genre_tv': statistics.mean(r['genre_tv'] for r in separated) if separated else None,
            'mean_random_genre_tv': statistics.mean(r['random_genre_tv'] for r in separated) if separated else None,
            'mean_genre_tv_minus_random': statistics.mean(deltas) if deltas else None,
            'long_span_genre_users': sum(r['span_days'] >= 30 for r in separated)}


def verified_model(directory, trial, contract, source_plan_sha, seed):
    """Verify frozen weights without calling the historical holdout-hashing loader."""
    result = json.loads((directory/'result.json').read_text(encoding='utf-8'))
    if digest(directory/'best.pt') != result['checkpoint_sha256']:
        raise ValueError('Checkpoint SHA mismatch')
    manifest, nu, ni, _, _ = contract
    required = {'seed': seed, 'plan_sha256': source_plan_sha, 'code_sha256': code_identity(),
                'data_manifest_sha256': digest(ROOT/'artifacts/data_manifest.json'),
                'id_map_sha256': manifest['id_map_sha256'],
                'train_positive_sha256': manifest['files']['train_positive.parquet']['sha256'],
                'torch_version': str(torch.__version__)}
    if result['status'] != 'VERIFIED' or result['trial'] != trial or any(
            result['identity'].get(k) != v for k,v in required.items()):
        raise ValueError('Frozen training identity mismatch')
    payload = torch.load(directory/'best.pt', map_location='cpu', weights_only=True)
    if (payload['identity'] != result['identity'] or payload['trial'] != trial or
            payload['dimensions'] != [nu, ni] or payload['best_epoch'] != result['best_epoch']):
        raise ValueError('Checkpoint payload mismatch')
    model = build_model(trial, contract)
    model.load_state_dict(payload['state_dict'], strict=True)
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('Nonfinite frozen weights')
    model.eval()
    return model, result


def model_rows(model, kind, samples, positive, low, seen, n_items, plan):
    rows = []
    with torch.no_grad():
        ue, ie = model.propagate() if kind == 'lightgcn' else (model.users.weight[1:], model.items.weight[1:])
        for offset in range(0, len(samples), plan['score_batch_size']):
            batch = samples[offset:offset+plan['score_batch_size']]
            scores = (ue[torch.tensor([s['user_id']-1 for s in batch])]@ie.T).tolist()
            for sample, score in zip(batch, scores):
                u = sample['user_id']
                chosen = {name: [] for name in STRATEGIES}
                for pool in sample['pools']:
                    for name, item in choose_negatives(pool, [score[i-1] for i in pool]).items():
                        chosen[name].append(score[item-1])
                pos = [score[i-1] for i in sample['positive_ids']]
                row = {'user_id':u, 'positive_count':len(positive[u]), 'negative':{
                    name:pair_summary(pos, values, plan['easy_gradient_below']) for name,values in chosen.items()}, 'low':None}
                if low[u]:
                    unknown_scores = sorted(score[i-1] for i in range(1,n_items+1) if i not in seen[u])
                    low_scores = [score[i-1] for i in low[u]]
                    threshold = unknown_scores[-min(plan['top_k'],len(unknown_scores))]
                    low_sample = [score[i-1] for i in sample['low_ids']]
                    n = min(len(pos),len(low_sample))
                    row['low'] = {'count':len(low_scores),
                                  'pair':pair_summary(pos[:n],low_sample[:n],plan['easy_gradient_below']),
                                  'high_score_fraction':sum(s >= threshold for s in low_scores)/len(low_scores),
                                  'score_percentile':statistics.mean(bisect.bisect_left(unknown_scores,s)/len(unknown_scores) for s in low_scores)}
                rows.append(row)
    return rows


def run(plan_path, out):
    plan_path, out = Path(plan_path).resolve(), Path(out).resolve()
    plan = validate_plan(json.loads(plan_path.read_text(encoding='utf-8')))
    out.mkdir(parents=True, exist_ok=False)
    source_plan_path = ROOT/plan['source_plan']
    base = ROOT/'artifacts/runs/lightgcn_optimization_20261011'
    data = ROOT/'data/processed/ml-1m-v1'
    allowed = [plan_path, source_plan_path, ROOT/'artifacts/data_manifest.json', base/'selection_frozen.json']
    allowed += [data/name for name in ('train.parquet','train_positive.parquet','items.parquet','id_map.json','split_manifest.json')]
    allowed += source_paths()
    allowed += [base/f'seed_{seed}'/name/file for seed in plan['model_seeds'] for name in plan['models'] for file in ('result.json','best.pt')]
    git = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT,text=True).strip()
    dirty = bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT))
    torch.set_num_threads(plan['torch_threads'])
    with train_only_files(allowed) as reads:
        contract = load_train_contract()
        manifest, nu, ni, seen, edges = contract
        if manifest['train_positive_threshold'] != 4:
            raise ValueError('Frozen positive threshold changed')
        source_plan = json.loads(source_plan_path.read_text(encoding='utf-8'))
        frozen = json.loads((base/'selection_frozen.json').read_text(encoding='utf-8'))
        if set(frozen['selected'].values()) != set(plan['models']) or (
                frozen['plan_sha256'] != digest(source_plan_path) or frozen['code_sha256'] != code_identity()):
            raise ValueError('Use only the previous frozen development selections')
        for name in ('items.parquet',):
            if digest(data/name) != manifest['files'][name]['sha256']:
                raise ValueError('Official metadata hash changed')
        mapping = json.loads((data/'id_map.json').read_text(encoding='utf-8'))
        um, im = mapping['user_to_model'], mapping['item_to_model']
        genres = {r['item_id']:r['genres'] for r in pq.read_table(data/'items.parquet').to_pylist() if r['item_id'] is not None}
        records, positives, lows = defaultdict(list), defaultdict(list), defaultdict(list)
        all_ratings, warm_ratings = Counter(), Counter()
        train = pq.read_table(data/'train.parquet').to_pylist()
        for r in train:
            rating = r['rating']; all_ratings[rating] += 1
            u, i = um.get(str(r['raw_user_id'])), im.get(str(r['raw_item_id']))
            if u is None or i is None:
                continue
            records[u].append((i,rating,r['timestamp'])); warm_ratings[rating] += 1
            if rating >= 4: positives[u].append(i)
            elif rating <= 2: lows[u].append(i)
        if set((u,i) for u,items in positives.items() for i in items) != set(edges) or len(positives) != nu:
            raise ValueError('Train rating reconstruction differs from frozen graph')
        temporal = [temporal_row(u,records[u],genres,plan['time_min_positives'],plan['time_min_group'],plan['seed']) for u in sorted(positives)]
        samples = sample_users(ni,seen,positives,lows,plan['positives_per_user'],plan['negative_pool_size'],plan['low_samples_per_user'],plan['seed'])
        samples_sha = write_once(out/'samples.json',samples)
        temporal_sha = write_once(out/'temporal_rows.json',temporal)
        result = {'schema_version':1, 'status':'VERIFIED', 'scope':'training-signal diagnostics, not recommendation evaluation',
                  'plan_sha256':digest(plan_path), 'source_sha256':digest(Path(__file__)),
                  'git_commit':git, 'git_dirty':dirty, 'train_users':nu, 'candidate_items':ni,
                  'train_positive_edges':len(edges), 'sample_pairs':sum(len(s['positive_ids']) for s in samples),
                  'samples_sha256':samples_sha, 'temporal_rows_sha256':temporal_sha,
                  'rating_counts_all_train':dict(all_ratings), 'rating_counts_warm_candidates':dict(warm_ratings),
                  'users_with_low_ratings':sum(bool(lows[u]) for u in positives),
                  'users_with_at_least_five_low_ratings':sum(len(lows[u]) >= 5 for u in positives),
                  'temporal':temporal_summary(temporal), 'models':{},
                  'accuracy_gain':'NOT EVALUATED', 'training_runs':0,'remote_api_requests':0,'downloads':0,
                  'holdout_labels_read':False, 'serving_replaced':False,
                  'input_sha256':{p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else p.name: digest(p)
                                  for p in allowed if p.is_file()}}
        for name in plan['models']:
            trial = next(t for t in source_plan['trials'] if t['name'] == name)
            for seed in plan['model_seeds']:
                model, original = verified_model(base/f'seed_{seed}'/name,trial,contract,digest(source_plan_path),seed)
                rows = model_rows(model,trial['model'],samples,positives,lows,seen,ni,plan)
                filename = f'{name}_seed_{seed}_rows.json'
                rows_sha = write_once(out/filename,rows)
                result['models'][f'{name}_seed_{seed}'] = {'summary':aggregate_rows(rows),
                    'rows_file':filename,'rows_sha256':rows_sha,
                    'checkpoint_sha256':original['checkpoint_sha256']}
                print(json.dumps({'model':name,'seed':seed,'users':len(rows),'status':'diagnosed'}),flush=True)
        result['files_read'] = sorted(Path(p).relative_to(ROOT).as_posix() if Path(p).is_relative_to(ROOT) else Path(p).name for p in reads)
    write_once(out/'result.json',result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    run(args.plan,args.output)


if __name__ == '__main__':
    main()
