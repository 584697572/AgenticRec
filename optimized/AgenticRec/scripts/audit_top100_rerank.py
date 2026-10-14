"""Audit raw chronology, fixed pools, selection, independent scoring and replay."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random

import pyarrow.parquet as pq
import torch

from audit_three_way_comparison import assess, digest


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rebuild_examples(rows):
    by = defaultdict(list)
    for row in rows:
        by[row['user_id']].append((row['timestamp'],row['item_id']))
    result = []
    for user,seq in sorted(by.items()):
        seq = sorted(seq); cut = int(len(seq)*.8)
        if cut<10 or cut>=len(seq): continue
        for h in (3,5,10):
            result.append(dict(liked_item_ids=[i for ts,i in seq[cut-h:cut]],
                               relevant_item_ids=sorted(i for ts,i in seq[cut:])))
    return result


def audit_cache(cache, env, histories, checkpoint, n_items):
    from agenticrec.evaluation.rank_upgrade_models import ShortSequence, padded_histories
    model = ShortSequence(n_items)
    model.load_state_dict(torch.load(checkpoint,map_location='cpu')['state_dict']);model.eval()
    with torch.no_grad(): raw = torch.cat([model(v) for v in padded_histories(histories).split(128)])
    pop = torch.log1p(env['index'].popularity)
    pz = (pop-pop.mean())/pop.std(unbiased=False).clamp_min(1e-6)
    for n,q in enumerate(histories):
        v = raw[n].double(); values = .5*((v-v.mean())/v.std(unbiased=False).clamp_min(1e-6))+.5*pz
        scores = values.tolist()
        pool = sorted(set(env['catalog'])-set(q['liked_item_ids']),key=lambda i:(-scores[i-1],i))[:100]
        assert pool==cache['ids'][n] and len(pool)==100
        ix = torch.tensor(pool)-1
        assert torch.equal(values[ix],cache['base'][n])
        cf,content,_,_ = env['features'].context(q['liked_item_ids'])
        extras = []
        for source in (values[ix],values[ix],cf[ix],content[ix],env['index'].popularity[ix]):
            if not extras:
                extras.append(source.float()); continue
            order = sorted(range(100),key=lambda k:(-float(source[k]),pool[k]))
            result = torch.empty(100)
            for rank,k in enumerate(order): result[k]=1-rank/99
            extras.append(result)
        expected = torch.cat((env['features'].features(q['liked_item_ids'],pool),torch.stack(extras,1)),1)
        assert torch.equal(expected,cache['x'][n])
    return len(histories)


def audit(directory, output):
    root = Path(__file__).resolve().parents[3]; data = root/'optimized/data/processed/ml-1m-v1'
    contract,frozen = read(directory/'contract.json'),read(directory/'frozen.json')
    for name,sha in contract['files'].items(): assert digest(root/name)==sha,name
    for name,sha in frozen['files'].items(): assert digest(directory/name)==sha,name
    catalog = {int(k):v for k,v in read(directory/'catalog.json').items()}
    mapping = read(data/'id_map.json'); umap = {int(k):int(v) for k,v in mapping['user_to_model'].items()}
    imap = {int(k):int(v) for k,v in mapping['item_to_model'].items()}
    rows = pq.read_table(data/'train.parquet').to_pylist(); positives = []
    seen = defaultdict(set); history = defaultdict(list); by = defaultdict(list)
    for r in rows:
        seen[r['raw_user_id']].add(r['raw_item_id'])
        if r['rating']>=4 and r['raw_user_id'] in umap and r['raw_item_id'] in imap:
            u,i = umap[r['raw_user_id']],imap[r['raw_item_id']]
            positives.append(dict(user_id=u,item_id=i,timestamp=r['timestamp'],rating=r['rating']))
            by[u].append((r['timestamp'],i))
            if i in catalog: history[r['raw_user_id']].append((r['timestamp'],i))
    times = sorted(r['timestamp'] for r in positives); queriesby = {}
    for number,ratio in enumerate((.35,.5)):
        anchor = times[int(len(times)*ratio)]
        graph = pq.read_table(directory/f'graph{number}.parquet').to_pylist()
        # Parquet schema includes original processed columns; compare semantic rows.
        assert sorted((r['user_id'],r['item_id'],r['timestamp'],r['rating']) for r in graph)==sorted(
            (r['user_id'],r['item_id'],r['timestamp'],r['rating']) for r in positives if r['timestamp']<anchor)
        assert max(r['timestamp'] for r in graph)<anchor
        assert rebuild_examples(graph)==read(directory/f'prefix_examples{number}.json')
        queries = []
        for u,seq in sorted(by.items()):
            hist = sorted((ts,i) for ts,i in seq if ts<anchor and i in catalog)
            targets = sorted(i for ts,i in seq if ts>=anchor+30*86400 and i in catalog)
            if len(hist)<10 or not targets: continue
            for h in (3,5,10):
                queries.append(dict(training_user_id=u,anchor=anchor,liked_item_ids=[i for ts,i in hist[-h:]],
                    masked_item_ids=sorted(i for ts,i in seq if ts<anchor),relevant_item_ids=targets,gap_days=30))
        assert queries==read(directory/f'rerank_queries{number}.json');queriesby[number]=queries
    for name,length in (('prefix_training',60),('residual_training',180)):
        curves=read(directory/(name+'.json'))['curves']
        assert len(curves)==length and all(math.isfinite(r['loss']) for r in curves)
    from agenticrec.evaluation.temporal_repair import env_for
    from agenticrec.evaluation.top100_rerank import infer_cache, config_grid, public_histories, OLD
    torch.set_num_threads(2); cache_replays = 0; manifest = []
    for number in (0,1):
        env = env_for(directory,f'graph{number}.parquet'); queries = queriesby[number]
        for seed in (7,42,2026):
            cache = torch.load(directory/'feature_cache'/f'train_{number}_{seed}.pt',map_location='cpu')
            cache_replays += audit_cache(cache,env,queries,directory/'prefix_models'/f'snapshot_{number}_seed_{seed}.pt',contract['model_items'])
            assert cache['groups']==[(number,q['training_user_id']) for q in queries]
            for n,q in enumerate(queries):
                p = torch.tensor([i in q['relevant_item_ids'] for i in cache['ids'][n]])
                assert torch.equal(p,cache['positive'][n])
                usable = bool(p.any() and (~p).sum()>=10)
                assert usable==bool(cache['usable'][n])
                manifest.append(dict(snapshot=number,seed=seed,query_index=n,item_ids=cache['ids'][n],usable=usable,positives=int(p.sum())))
    assert manifest==read(directory/'train_pool_manifest.json')
    env = env_for(directory); truths = read(root/'optimized/artifacts/comparison/three_way_20261013/truth_private.json')
    devpub = read(directory/'development_public.json'); devlab = {r['episode_id']:r for r in read(directory/'development_private.json')}
    public = read(directory/'formal_public.json'); labels = {r['episode_id']:r for r in read(directory/'formal_private.json')}
    assert len(devpub)==150 and len(public)==180
    assert not {devlab[p['episode_id']]['evaluator_raw_user_id'] for p in devpub}&{labels[p['episode_id']]['evaluator_raw_user_id'] for p in public}
    for phase,pubs,labs in (('eval_valid.parquet',devpub,devlab),('eval_test.parquet',public,labels)):
        targets = defaultdict(set)
        for r in pq.read_table(data/phase).to_pylist():
            if r['is_target'] and r['segment']=='WARM' and r['raw_item_id'] in imap and imap[r['raw_item_id']] in catalog and r['raw_item_id'] not in seen[r['raw_user_id']]:
                targets[r['raw_user_id']].add(r['raw_item_id'])
        for p in pubs:
            lab = labs[p['episode_id']]; u = lab['evaluator_raw_user_id']; h = len(p['catalog_context'])
            assert [r['item_id'] for r in p['catalog_context']]==[i for ts,i in sorted(history[u])[-h:]]
            assert set(lab['labels'][0]['relevant_raw_ids'])==targets[u]
    selection = read(directory/'selection.json'); dev = read(directory/'development_predictions.json')
    assert len(dev)==8550 and len(selection['candidates'])==19
    scores = defaultdict(lambda:defaultdict(list)); keys = set(); devrows = {}
    for r in dev:
        key = (r['grid_index'],r['seed'],r['episode_id']);assert key not in keys;keys.add(key);devrows[key]=r
        met,errors = assess(dict(raw_ids=[catalog[i]['raw_id'] for i in r['item_ids']],status='OK',extraction_verified=True),
                            devlab[r['episode_id']]['labels'][0],truths,[])
        assert not any(errors.values());scores[r['grid_index']][r['seed']].append(met)
    for index,c in enumerate(selection['candidates']):
        assert c['grid_index']==index and c['config']==config_grid()[index]
        for m in ('hit','recall','ndcg','mrr'):
            for sn,s in enumerate((7,42,2026)):
                v=sum(r[m] for r in scores[index][s])/150
                assert math.isclose(c['seed_metrics'][sn][m],v,abs_tol=1e-12)
            expected = sum(sum(r[m] for r in v)/150 for v in scores[index].values())/3
            assert math.isclose(c['metrics'][m],expected,abs_tol=1e-12)
    choose = lambda cs:max(cs,key=lambda c:(c['metrics']['hit'],c['metrics']['ndcg'],-c['grid_index']))
    assert choose(selection['candidates'])['config']==selection['selected']==frozen['selected']
    for route in ('linear_boundary','mlp_boundary','mlp_mass'):
        assert choose([c for c in selection['candidates'] if c['config']['route']==route])['config']==selection['routes'][route]==frozen['routes'][route]
    for seed in (7,42,2026):
        cache=torch.load(directory/'feature_cache'/f'development_{seed}.pt',map_location='cpu')
        cache_replays+=audit_cache(cache,env,public_histories(devpub),OLD/'models'/f'seed_{seed}.pt',contract['model_items'])
        for index,config in enumerate(config_grid()):
            predictions=infer_cache(directory,config,seed,cache)
            for p,ids in zip(devpub,predictions):assert ids==devrows[index,seed,p['episode_id']]['item_ids']
    pools=read(directory/'formal_pools.json');poolby={(r['seed'],r['episode_id']):r['item_ids'] for r in pools}
    assert len(pools)==len(poolby)==540
    pred=read(directory/'predictions.json');scored=read(directory/'scores.json');report=read(directory/'report.json');systems=defaultdict(dict)
    assert len(pred)==len(scored)==2700
    assert digest(directory/'predictions.json')==report['predictions_sha256'] and digest(directory/'formal_pools.json')==report['pools_sha256']
    assert (directory/'predictions.json').stat().st_mtime_ns<=(directory/'scores.json').stat().st_mtime_ns
    for p,r in zip(pred,scored):
        assert all(r[k]==v for k,v in p.items())
        assert r['raw_ids']==[catalog[i]['raw_id'] for i in r['item_ids']]
        assert len(r['item_ids'])==10 and len(set(r['item_ids']))==10
        met,errors=assess(dict(raw_ids=r['raw_ids'],status='OK',extraction_verified=True),labels[r['episode_id']]['labels'][0],truths,[])
        assert not any(errors.values())
        for m in ('hit','recall','ndcg','mrr'):assert math.isclose(met[m],r['metrics'][m],abs_tol=1e-12)
        assert r['episode_id'] not in systems[r['system']];systems[r['system']][r['episode_id']]=r
        if r['system'] not in ('candidate','popular','C1'):
            assert set(r['item_ids'])<=set(poolby[r['seed'],r['episode_id']])
    old={(r['system'],r['episode_id']):r for r in read(OLD/'predictions.json')}
    for seed in (7,42,2026):
        cache=torch.load(directory/f'formal_cache_{seed}.pt',map_location='cpu')
        cache_replays+=audit_cache(cache,env,public_histories(public),OLD/'models'/f'seed_{seed}.pt',contract['model_items'])
        for config in [config_grid()[0]]+list(frozen['routes'].values()):
            ids=infer_cache(directory,config,seed,cache);name=config['route']+str(seed)
            for p,row,pool in zip(public,ids,cache['ids']):
                assert row==systems[name][p['episode_id']]['item_ids'] and pool==poolby[seed,p['episode_id']]
                if config['route']=='sequence':assert row==old[name,p['episode_id']]['item_ids']
        covered=sum(bool({catalog[i]['raw_id'] for i in poolby[seed,p['episode_id']]}&set(labels[p['episode_id']]['labels'][0]['relevant_raw_ids'])) for p in public)
        assert report['top100_coverage'][str(seed)]==dict(n=180,hits=covered,hit100=covered/180)
    for name in ('candidate','popular','C1'):
        for eid,r in systems[name].items():assert r['raw_ids']==old[name,eid]['raw_ids']
    for name,rowsby in systems.items():
        assert len(rowsby)==180 and report['systems'][name]['hits']==sum(r['metrics']['hit'] for r in rowsby.values())
        for m in ('hit','recall','ndcg','mrr'):assert math.isclose(report['systems'][name][m],sum(r['metrics'][m] for r in rowsby.values())/180,abs_tol=1e-12)
    assert report['primary_system']==frozen['selected']['route']+'42'
    main=systems[report['primary_system']];eids=sorted(main);pvalues={}
    for bn,name in enumerate(('sequence42','candidate','popular')):
        c=report['comparisons'][name];w=sum(main[k]['metrics']['hit']>systems[name][k]['metrics']['hit'] for k in eids)
        l=sum(main[k]['metrics']['hit']<systems[name][k]['metrics']['hit'] for k in eids);n=w+l
        assert (w,l)==(c['wins'],c['losses']);pvalues[name]=min(1.,2*sum(math.comb(n,k) for k in range(min(w,l)+1))/2**n)
        assert math.isclose(pvalues[name],c['hit_exact_p'],abs_tol=1e-12)
        for mn,m in enumerate(('hit','recall','ndcg','mrr')):
            delta=[main[k]['metrics'][m]-systems[name][k]['metrics'][m] for k in eids]
            rng=random.Random(2026101401+bn*10+mn);samples=sorted(sum(rng.choices(delta,k=180))/180 for _ in range(10000))
            ci=[]
            for q in (.025,.975):
                pos=q*9999;ix=int(pos);ci.append(samples[ix]+(samples[ix+1]-samples[ix])*(pos-ix))
            assert math.isclose(sum(delta)/180,c['metrics'][m]['mean'],abs_tol=1e-12)
            assert all(math.isclose(a,b,abs_tol=1e-12) for a,b in zip(ci,c['metrics'][m]['ci95']))
    last=0.
    for rank,name in enumerate(sorted(pvalues,key=pvalues.get)):
        last=max(last,min(1.,pvalues[name]*(3-rank)));assert math.isclose(last,report['comparisons'][name]['hit_holm_p'],abs_tol=1e-12)
    result=dict(status='VERIFIED',source_hashes_verified=True,raw_prefix_graphs_and_queries_verified=True,
        prefix_training_samples_rebuilt=sum(r['examples'] for r in contract['snapshot_samples']),
        prefix_epochs_verified=60,residual_epochs_verified=180,feature_query_replays=cache_replays,
        development_predictions_scored_and_replayed=8550,configurations_verified=19,
        formal_scores_independently_verified=2700,new_formal_predictions_replayed=2160,
        historical_controls_matched=540,baseline_predictions_matched=540,fixed_top100_pools_verified=540,
        raw_public_inputs_and_targets_verified=True,statistics_verified=True,api_requests=0,
        report_sha256=digest(directory/'report.json'),auditor_sha256=digest(Path(__file__)))
    with output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();audit(args.directory.resolve(),args.output.resolve())
