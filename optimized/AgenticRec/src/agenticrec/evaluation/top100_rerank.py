"""Finite fixed-top100 ranking experiment: no LLM or candidate insertion."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import shutil
import time

import pyarrow.parquet as pq
import torch

from .simple_baselines import ROOT, load, write_new, digest, quality, paired_interval
from .sequence_ml1m import verify as verify_old, training_examples
from .temporal_repair import env_for, ordering, metrics_on
from .rank_upgrade_models import ShortSequence, padded_histories, multi_positive_loss, mixed_scores
from .top100_models import FEATURE_NAMES, SequenceResidual, pool_features, boundary_loss, mass_loss, ranked_pool

OLD = ROOT/'optimized/artifacts/comparison/sequence_ml1m_20261013_run2'
PREVIOUS = ROOT/'optimized/artifacts/comparison/temporal_repair_20261013'
DATA = ROOT/'optimized/data/processed/ml-1m-v1'
PROTOCOL = ROOT/'optimized/reports/top100_rerank_protocol_20261014.md'
SEEDS = (7, 42, 2026)
ROUTES = ('linear_boundary', 'mlp_boundary', 'mlp_mass')
METRICS = ('hit', 'recall', 'ndcg', 'mrr')


def verify(directory):
    contract = load(directory/'contract.json')
    for name, sha in contract['files'].items():
        if digest(ROOT/name) != sha:
            raise ValueError('Frozen source changed: '+name)
    return contract


def prepare(directory):
    if directory.exists():
        raise FileExistsError('Use a fresh study directory')
    old = verify_old(OLD)
    previous = load(PREVIOUS/'contract.json')
    for name, sha in previous['files'].items():
        if digest(ROOT/name) != sha:
            raise ValueError('Previous frozen source changed: '+name)
    directory.mkdir(parents=True)
    for name in ('catalog.json', 'development_public.json', 'development_private.json',
                 'formal_public.json', 'formal_private.json', 'graph0.parquet', 'graph1.parquet',
                 'rerank_queries0.json', 'rerank_queries1.json'):
        shutil.copyfile(PREVIOUS/name, directory/name)
    samples = []
    for number in (0, 1):
        rows = pq.read_table(directory/f'graph{number}.parquet').to_pylist()
        examples = training_examples(rows)
        if not examples:
            raise ValueError('No prefix backbone examples')
        write_new(directory/f'prefix_examples{number}.json', examples)
        samples.append(dict(snapshot=number, edges=len(rows), examples=len(examples), users=len(examples)//3))
    paths = list((ROOT/'optimized/AgenticRec/src/agenticrec').rglob('*.py'))
    paths += [PROTOCOL, ROOT/'optimized/AgenticRec/scripts/audit_top100_rerank.py',
              ROOT/'optimized/AgenticRec/scripts/report_top100_rerank.py',
              ROOT/'optimized/AgenticRec/scripts/audit_three_way_comparison.py']
    paths += [DATA/n for n in ('train.parquet', 'train_positive.parquet', 'eval_valid.parquet', 'eval_test.parquet', 'id_map.json')]
    paths += [OLD/n for n in ('contract.json', 'predictions.json', 'report.json')]
    paths += [OLD/'models'/f'seed_{s}.pt' for s in SEEDS]
    paths += [p for p in directory.iterdir() if p.is_file()]
    result = dict(status='FROZEN', scope='post_hoc_original_180', model_items=old['model_items'],
                  feature_names=list(FEATURE_NAMES), seeds=list(SEEDS), primary_seed=42,
                  snapshot_samples=samples, hidden_history_filter=False, candidate_insertion=False,
                  api_requests=0, downloads=0, files={p.relative_to(ROOT).as_posix():digest(p) for p in paths})
    write_new(directory/'contract.json', result)
    return {k:v for k,v in result.items() if k!='files'}


def prefix(directory):
    contract = verify(directory)
    target = directory/'prefix_models'
    if target.exists():
        raise FileExistsError('Prefix training cannot overwrite')
    target.mkdir(); torch.set_num_threads(2)
    curves = []; started = time.perf_counter()
    for number in (0, 1):
        examples = load(directory/f'prefix_examples{number}.json')
        x = padded_histories(examples)
        y = torch.zeros((len(examples), contract['model_items']))
        for n, row in enumerate(examples):
            y[n, [i-1 for i in row['relevant_item_ids']]] = 1
        for seed in SEEDS:
            torch.manual_seed(seed); model = ShortSequence(contract['model_items'])
            opt = torch.optim.Adam(model.parameters(), lr=.002, weight_decay=1e-4)
            for epoch in range(1, 11):
                model.train(); total = 0.
                for batch in torch.randperm(len(examples)).split(32):
                    loss = multi_positive_loss(model(x[batch]), y[batch], x[batch])
                    if not torch.isfinite(loss): raise ValueError('Nonfinite prefix loss')
                    opt.zero_grad(); loss.backward(); opt.step()
                    total += float(loss.detach())*len(batch)
                step = dict(snapshot=number, seed=seed, epoch=epoch, loss=total/len(examples))
                curves.append(step); print(json.dumps(step), flush=True)
            torch.save(dict(state_dict=model.state_dict(), seed=seed, snapshot=number, epoch=10),
                       target/f'snapshot_{number}_seed_{seed}.pt')
    result = dict(status='TRAINED', curves=curves, elapsed_seconds=time.perf_counter()-started,
                  targets_outside_prefix_not_used=True, api_requests=0)
    write_new(directory/'prefix_training.json', result)
    return dict(status='TRAINED', epochs=len(curves), elapsed_seconds=result['elapsed_seconds'])


def score_history(checkpoint, model_items, histories, popularity):
    model = ShortSequence(model_items)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu')['state_dict']); model.eval()
    with torch.no_grad():
        raw = torch.cat([model(batch) for batch in padded_histories(histories).split(128)])
    return torch.stack([mixed_scores(v, popularity, .5) for v in raw])


def make_cache(env, histories, values):
    ids = [ordering(values[n], r['liked_item_ids'], env['catalog'], 100) for n,r in enumerate(histories)]
    if any(len(pool)!=100 for pool in ids): raise ValueError('Expected exact top100')
    xs = [pool_features(env['features'], r['liked_item_ids'], pool, values[n])
          for n,(r,pool) in enumerate(zip(histories,ids))]
    base = torch.stack([values[n, torch.tensor(pool)-1] for n,pool in enumerate(ids)])
    return dict(ids=ids, x=torch.stack(xs), base=base)


def cache_training(directory):
    contract = verify(directory); target = directory/'feature_cache'
    if target.exists(): raise FileExistsError('Feature evidence cannot overwrite')
    target.mkdir(); torch.set_num_threads(2); manifest = []; stats = []
    started = time.perf_counter()
    for number in (0, 1):
        env = env_for(directory, f'graph{number}.parquet')
        queries = load(directory/f'rerank_queries{number}.json')
        for seed in SEEDS:
            values = score_history(directory/'prefix_models'/f'snapshot_{number}_seed_{seed}.pt',
                                   contract['model_items'], queries, env['index'].popularity)
            cache = make_cache(env, queries, values)
            pos = torch.tensor([[i in q['relevant_item_ids'] for i in pool] for q,pool in zip(queries,cache['ids'])])
            usable = pos.any(1)&(~pos).sum(1).ge(10)
            groups = [(number,q['training_user_id']) for q in queries]
            torch.save(dict(cache, positive=pos, usable=usable, groups=groups), target/f'train_{number}_{seed}.pt')
            stats.append(dict(snapshot=number, seed=seed, queries=len(queries), usable=int(usable.sum()),
                              target_coverage=float(pos.any(1).float().mean()), no_positive=int((~pos.any(1)).sum())))
            for n,q in enumerate(queries):
                manifest.append(dict(snapshot=number, seed=seed, query_index=n, item_ids=cache['ids'][n],
                                     usable=bool(usable[n]), positives=int(pos[n].sum())))
            print(json.dumps(stats[-1]), flush=True)
    public = load(directory/'development_public.json'); histories = public_histories(public)
    env = env_for(directory)
    for seed in SEEDS:
        values = score_history(OLD/'models'/f'seed_{seed}.pt', contract['model_items'], histories, env['index'].popularity)
        torch.save(make_cache(env, histories, values), target/f'development_{seed}.pt')
    write_new(directory/'train_pool_manifest.json', manifest)
    write_new(directory/'feature_training.json', dict(status='BUILT', statistics=stats,
        elapsed_seconds=time.perf_counter()-started, candidate_insertion=False, hidden_history_filter=False,
        sequence_future_labels_in_features=False, feature_names=list(FEATURE_NAMES)))
    return dict(status='BUILT', statistics=stats)


def public_histories(public):
    return [dict(liked_item_ids=[r['item_id'] for r in p['catalog_context']]) for p in public]


def train_arrays(directory, seed):
    values = [torch.load(directory/'feature_cache'/f'train_{n}_{seed}.pt', map_location='cpu') for n in (0,1)]
    x = torch.cat([v['x'][v['usable']] for v in values])
    base = torch.cat([v['base'][v['usable']] for v in values])
    pos = torch.cat([v['positive'][v['usable']] for v in values])
    groups = [g for v in values for g,ok in zip(v['groups'],v['usable']) if ok]
    if not groups: raise ValueError('No real candidate supervision')
    counts = Counter(groups)
    weights = torch.tensor([len(groups)/len(counts)/counts[g] for g in groups])
    means = x.flatten(0,1).mean(0); scales = x.flatten(0,1).std(0,unbiased=False).clamp_min(1e-6)
    return x,base,pos,weights,means,scales


def residual(directory):
    verify(directory); target = directory/'residual_models'
    if target.exists(): raise FileExistsError('Residual training cannot overwrite')
    target.mkdir(); torch.set_num_threads(2); curves = []; started = time.perf_counter()
    for seed in SEEDS:
        x,base,pos,weights,means,scales = train_arrays(directory,seed)
        for route in ROUTES:
            torch.manual_seed(seed); model = SequenceResidual(means,scales,linear=route=='linear_boundary')
            opt = torch.optim.Adam(model.parameters(),lr=.002,weight_decay=1e-4)
            for epoch in range(1,21):
                model.train(); total = 0.
                for batch in torch.randperm(len(x)).split(32):
                    correction = model.correction(x[batch]); scores = base[batch]+correction
                    losses = mass_loss(scores,pos[batch]) if route=='mlp_mass' else boundary_loss(scores,pos[batch])
                    loss = ((losses+.02*correction.square().mean(1))*weights[batch]).mean()
                    if not torch.isfinite(loss): raise ValueError('Nonfinite residual loss')
                    opt.zero_grad(); loss.backward(); opt.step(); total += float(loss.detach())*len(batch)
                curves.append(dict(seed=seed,route=route,epoch=epoch,loss=total/len(x),queries=len(x)))
                if epoch in (10,20):
                    torch.save(dict(state_dict=model.state_dict(),route=route,seed=seed,epoch=epoch),
                               target/f'{route}_{seed}_{epoch}.pt')
            print(json.dumps(dict(route=route,seed=seed,epochs=20,queries=len(x))),flush=True)
    write_new(directory/'residual_training.json',dict(status='TRAINED',curves=curves,
              elapsed_seconds=time.perf_counter()-started,api_requests=0))
    return dict(status='TRAINED',epochs=len(curves),elapsed_seconds=time.perf_counter()-started)


def config_grid():
    return [dict(route='sequence',epoch=0,strength=0.)]+[
        dict(route=r,epoch=e,strength=a) for r in ROUTES for e in (10,20) for a in (.1,.3,.6)]


def infer_cache(directory, config, seed, cache):
    if config['route']=='sequence': return [pool[:10] for pool in cache['ids']]
    checkpoint = torch.load(directory/'residual_models'/f'{config["route"]}_{seed}_{config["epoch"]}.pt',map_location='cpu')['state_dict']
    model = SequenceResidual(checkpoint['means'],checkpoint['scales'],linear=config['route']=='linear_boundary')
    model.load_state_dict(checkpoint); model.eval()
    with torch.no_grad(): scores = model(cache['x'],cache['base'],config['strength'])
    return [ranked_pool(pool,scores[n]) for n,pool in enumerate(cache['ids'])]


def choose(rows):
    return max(rows,key=lambda r:(r['metrics']['hit'],r['metrics']['ndcg'],-r['grid_index']))


def freeze(directory):
    verify(directory); torch.set_num_threads(2)
    if (directory/'selection.json').exists(): raise FileExistsError('Selection cannot overwrite')
    public = load(directory/'development_public.json'); labels = load(directory/'development_private.json')
    catalog = {int(k):v for k,v in load(directory/'catalog.json').items()}
    caches = {s:torch.load(directory/'feature_cache'/f'development_{s}.pt',map_location='cpu') for s in SEEDS}
    candidates = []; predictions = []
    for index,config in enumerate(config_grid()):
        perseed = []
        for seed in SEEDS:
            ids = infer_cache(directory,config,seed,caches[seed])
            perseed.append(metrics_on(ids,public,labels,catalog))
            predictions += [dict(grid_index=index,seed=seed,episode_id=p['episode_id'],item_ids=row) for p,row in zip(public,ids)]
        candidates.append(dict(grid_index=index,config=config,metrics={m:sum(r[m] for r in perseed)/3 for m in METRICS},seed_metrics=perseed))
    best = choose(candidates)
    routes = {r:choose([c for c in candidates if c['config']['route']==r])['config'] for r in ROUTES}
    selection = dict(status='DEVELOPED',selected=best['config'],routes=routes,candidates=candidates,test_used_for_selection=False)
    write_new(directory/'development_predictions.json',predictions);write_new(directory/'selection.json',selection)
    files = [directory/'selection.json',directory/'development_predictions.json',directory/'prefix_training.json',
             directory/'residual_training.json',directory/'feature_training.json',directory/'train_pool_manifest.json']
    files += list((directory/'prefix_models').glob('*.pt'))+list((directory/'residual_models').glob('*.pt'))+list((directory/'feature_cache').glob('*.pt'))
    write_new(directory/'frozen.json',dict(status='FROZEN',selected=best['config'],routes=routes,
              selected_development_metrics=best['metrics'],files={p.relative_to(directory).as_posix():digest(p) for p in files}))
    return dict(status='FROZEN',selected=best['config'],development=best['metrics'],routes=routes,configurations=len(candidates))


def check_frozen(directory):
    contract = verify(directory); frozen = load(directory/'frozen.json')
    for name,sha in frozen['files'].items():
        if digest(directory/name)!=sha: raise ValueError('Frozen learned evidence changed: '+name)
    return contract,frozen


def evaluate(directory):
    contract,frozen = check_frozen(directory); torch.set_num_threads(2)
    if (directory/'predictions.json').exists(): raise FileExistsError('Formal evaluation cannot overwrite')
    env = env_for(directory); public = load(directory/'formal_public.json'); histories = public_histories(public)
    predictions = []; pools = []
    for seed in SEEDS:
        values = score_history(OLD/'models'/f'seed_{seed}.pt',contract['model_items'],histories,env['index'].popularity)
        cache = make_cache(env,histories,values)
        torch.save(cache,directory/f'formal_cache_{seed}.pt')
        pools += [dict(seed=seed,episode_id=p['episode_id'],item_ids=pool) for p,pool in zip(public,cache['ids'])]
        for config in [config_grid()[0]]+list(frozen['routes'].values()):
            ids = infer_cache(directory,config,seed,cache)
            for p,row in zip(public,ids):
                predictions.append(dict(system=config['route']+str(seed),seed=seed,episode_id=p['episode_id'],
                    history_n=len(p['catalog_context']),item_ids=row,raw_ids=[env['catalog'][i]['raw_id'] for i in row]))
    controls = [r for r in load(OLD/'predictions.json') if r['system'] in ('C1','candidate','popular')]
    write_new(directory/'formal_pools.json',pools)
    write_new(directory/'predictions.json',predictions+controls)
    # Formal labels are consumed only after all predictions have been persisted.
    labels = {r['episode_id']:r for r in load(directory/'formal_private.json')}; records = []
    for row in predictions+controls:
        records.append(dict(row,metrics=quality(row['raw_ids'],labels[row['episode_id']]['labels'][0]['relevant_raw_ids'])))
    write_new(directory/'scores.json',records); by = defaultdict(dict)
    for r in records: by[r['system']][r['episode_id']]=r
    systems = {s:dict(n=len(v),hits=int(sum(r['metrics']['hit'] for r in v.values())),
               **{m:sum(r['metrics'][m] for r in v.values())/len(v) for m in METRICS}) for s,v in by.items()}
    selected = frozen['selected']['route']+'42'; main = by[selected]; keys = sorted(main); comparisons = {}
    for bn,name in enumerate(('sequence42','candidate','popular')):
        wins = sum(main[k]['metrics']['hit']>by[name][k]['metrics']['hit'] for k in keys)
        losses = sum(main[k]['metrics']['hit']<by[name][k]['metrics']['hit'] for k in keys); n = wins+losses
        comparisons[name] = dict(wins=wins,losses=losses,
            hit_exact_p=min(1.,2*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/2**n),
            metrics={m:paired_interval([main[k]['metrics'][m]-by[name][k]['metrics'][m] for k in keys],2026101401+bn*10+mn) for mn,m in enumerate(METRICS)})
    last = 0.
    for rank,name in enumerate(sorted(comparisons,key=lambda k:comparisons[k]['hit_exact_p'])):
        last = max(last,min(1.,comparisons[name]['hit_exact_p']*(3-rank)));comparisons[name]['hit_holm_p']=last
    poolby = {(r['seed'],r['episode_id']):r['item_ids'] for r in pools}
    coverage = {}
    for seed in SEEDS:
        hits = sum(bool({env['catalog'][i]['raw_id'] for i in poolby[seed,p['episode_id']]} &
                        set(labels[p['episode_id']]['labels'][0]['relevant_raw_ids'])) for p in public)
        coverage[str(seed)] = dict(n=180,hits=hits,hit100=hits/180)
    result = dict(status='EVALUATED',scope='post_hoc_original_180',selected=frozen['selected'],primary_system=selected,
                  systems=systems,comparisons=comparisons,top100_coverage=coverage,routes=frozen['routes'],
                  api_requests=0,downloads=0,default_changed=False,strict_agent_tasks_evaluated=False,
                  predictions_sha256=digest(directory/'predictions.json'),pools_sha256=digest(directory/'formal_pools.json'),
                  frozen_sha256=digest(directory/'frozen.json'),contract_sha256=digest(directory/'contract.json'))
    write_new(directory/'report.json',result); return result


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','prefix','cache_training','residual','freeze','evaluate'))
    parser.add_argument('--directory',type=Path,required=True)
    args = parser.parse_args(); print(json.dumps(globals()[args.phase](args.directory.resolve()),indent=2),flush=True)
