"""Read-only raw chronology, model replay and independent scoring audit."""
import argparse
from collections import defaultdict, Counter
import json
import math
from pathlib import Path
import random

import pyarrow.parquet as pq
import torch

from audit_three_way_comparison import assess, digest


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def audit(directory, output):
    root = Path(__file__).resolve().parents[3]
    source = root/'optimized/artifacts/comparison/c1_hit_20261013'
    controls = root/'optimized/artifacts/comparison/simple_baselines_20261013'
    data = root/'optimized/data/processed/ml-1m-v1'
    olddir = root/'optimized/artifacts/comparison/three_way_20261013'
    contract = read(directory/'contract.json')
    for name, sha in contract['files'].items(): assert digest(root/name) == sha, name
    truth = read(olddir/'truth_private.json')
    catalog = {int(k):v for k,v in read(olddir/'catalog.json')['items'].items()}
    rawmap = {v['raw_id']:i for i,v in catalog.items()}
    rawrows = pq.read_table(data/'train.parquet').to_pylist()
    histories, seen = defaultdict(list), defaultdict(set)
    for r in rawrows:
        seen[r['raw_user_id']].add(r['raw_item_id'])
        if r['rating'] >= 4 and r['raw_item_id'] in rawmap:
            histories[r['raw_user_id']].append((r['timestamp'],r['raw_item_id']))
    targets = defaultdict(set)
    for r in pq.read_table(data/'eval_test.parquet').to_pylist():
        if r['is_target'] and r['segment']=='WARM' and r['raw_item_id'] in rawmap and r['raw_item_id'] not in seen[r['raw_user_id']]:
            targets[r['raw_user_id']].add(r['raw_item_id'])
    mapping = read(data/'id_map.json')['user_to_model']
    fullmap = {int(k):int(v) for k,v in read(data/'id_map.json')['item_to_model'].items()}
    fullhist = defaultdict(list)
    for r in rawrows:
        if r['rating']>=4 and r['raw_item_id'] in fullmap and str(r['raw_user_id']) in mapping:
            fullhist[r['raw_user_id']].append((r['timestamp'],r['raw_item_id']))
    examples = []
    for rawuser in sorted(fullhist, key=lambda u:int(mapping.get(str(u), 10**9))):
        if str(rawuser) not in mapping: continue
        seq = sorted(fullhist[rawuser]); cut = len(seq)*4//5
        if cut < 10 or cut >= len(seq): continue
        for h in (3,5,10):
            examples.append({'liked_item_ids': [fullmap[i] for _,i in seq[cut-h:cut]],
                             'relevant_item_ids': sorted(fullmap[i] for _,i in seq[cut:])})
    assert read(directory/'train_examples.json') == examples
    assert len(examples) == contract['training_examples']
    labels = {r['episode_id']:r for r in read(source/'formal_private.json')}
    public = {r['episode_id']:r for r in read(directory/'public.json')}
    assert len(public) == 180 and len({labels[i]['evaluator_raw_user_id'] for i in public}) == 180
    for eid,pub in public.items():
        lab = labels[eid]; user = lab['evaluator_raw_user_id']; h = lab['history_n']
        expected = [rawmap[i] for _,i in sorted(histories[user])[-h:]]
        assert [r['item_id'] for r in pub['catalog_context']] == expected
        assert set(lab['labels'][0]['relevant_raw_ids']) == targets[user]
        assert lab['labels'][0]['excluded_raw_ids'] == [catalog[i]['raw_id'] for i in expected]
    predictions, scores = read(directory/'predictions.json'), read(directory/'scores.json')
    assert len(predictions) == len(scores) == 1260
    keys = {(r['system'],r['episode_id']) for r in predictions}; assert len(keys) == 1260
    report = read(directory/'report.json')
    assert report['predictions_sha256'] == digest(directory/'predictions.json')
    assert (directory/'predictions.json').stat().st_mtime_ns <= (directory/'scores.json').stat().st_mtime_ns
    totals = defaultdict(Counter); by = defaultdict(dict)
    for pred,row in zip(predictions,scores):
        assert all(row[k] == v for k,v in pred.items())
        assert row['raw_ids'] == [catalog[i]['raw_id'] for i in row['item_ids']]
        expected,errors = assess({'raw_ids':row['raw_ids'],'status':'OK','extraction_verified':True},
                                  labels[row['episode_id']]['labels'][0],truth,[])
        assert not any(errors.values()) and expected['strict_success']
        for m in ('hit','recall','ndcg','mrr'):
            assert math.isclose(row['metrics'][m],expected[m],abs_tol=1e-12)
            totals[row['system']][m] += expected[m]
        totals[row['system']]['n'] += 1
        by[row['system']][row['episode_id']] = row
    for s,v in totals.items():
        assert v['n'] == 180 and report['systems'][s]['hits'] == v['hit']
        for m in ('hit','recall','ndcg','mrr'): assert math.isclose(report['systems'][s][m],v[m]/180,abs_tol=1e-12)
    previous = {(r['system'],r['episode_id']):r for r in read(controls/'predictions.json')}
    live = {(r['system'],r['episode_id']):r['turns'][0]['raw_ids'] for r in
            map(json.loads,(source/'live_run/episodes.jsonl').read_text(encoding='utf-8').splitlines())
            if r['category']=='session_quality'}
    for r in predictions:
        if not r['system'].startswith('sequence'):
            assert r == previous[r['system'],r['episode_id']]
            if r['system'] in ('candidate','C1'): assert r['raw_ids'] == live[r['system'],r['episode_id']]
    from agenticrec.evaluation.rank_upgrade_models import ShortSequence
    torch.set_num_threads(2)
    train = read(directory/'training.json')
    pop = torch.zeros(contract['model_items'],dtype=torch.float64)
    for item in pq.read_table(data/'train_positive.parquet',columns=['item_id']).column(0).to_pylist(): pop[item-1]+=1
    for run in train['runs']:
        seed = run['seed']; path = directory/'models'/f'seed_{seed}.pt'
        assert digest(path) == run['checkpoint_sha256']
        checkpoint = torch.load(path,map_location='cpu')
        assert checkpoint['epoch']==10 and checkpoint['contract_sha256']==digest(directory/'contract.json')
        model = ShortSequence(contract['model_items']); model.load_state_dict(checkpoint['state_dict']); model.eval()
        pubs = list(public.values()); x = torch.zeros((180,10),dtype=torch.long)
        for n,p in enumerate(pubs):
            ids = [r['item_id'] for r in p['catalog_context']]; x[n,:len(ids)] = torch.tensor(ids)
        with torch.no_grad(): values = model(x).to(torch.float64)
        logpop = torch.log1p(pop); logpop = (logpop-logpop.mean())/logpop.std(unbiased=False).clamp_min(1e-6)
        for n,p in enumerate(pubs):
            v = values[n]; v = .5*(v-v.mean())/v.std(unbiased=False).clamp_min(1e-6)+.5*logpop
            exclude = {r['item_id'] for r in p['catalog_context']}
            ids = sorted(set(catalog)-exclude,key=lambda i:(-float(v[i-1]),i))[:10]
            assert ids == by['sequence'+str(seed)][p['episode_id']]['item_ids']
    pvalues = {}
    for bn,baseline in enumerate(('candidate','popular','itemcf')):
        keys = sorted(public); a,b = by['sequence42'],by[baseline]
        win = sum(a[i]['metrics']['hit'] > b[i]['metrics']['hit'] for i in keys)
        loss = sum(a[i]['metrics']['hit'] < b[i]['metrics']['hit'] for i in keys); n = win+loss
        pvalues[baseline] = min(1.,2*sum(math.comb(n,k) for k in range(min(win,loss)+1))/2**n)
        comp = report['comparisons'][baseline]; assert (win,loss)==(comp['wins'],comp['losses'])
        assert math.isclose(pvalues[baseline],comp['hit_exact_p'],abs_tol=1e-12)
        for mn,m in enumerate(('hit','recall','ndcg','mrr')):
            delta = [a[i]['metrics'][m]-b[i]['metrics'][m] for i in keys]
            rng = random.Random(2026101314+bn*10+mn)
            samples = sorted(sum(rng.choices(delta,k=180))/180 for _ in range(10000))
            ci = []
            for q in (.025,.975):
                pos = q*9999; lo = int(pos); ci.append(samples[lo]+(samples[lo+1]-samples[lo])*(pos-lo))
            assert all(math.isclose(x,y,abs_tol=1e-12) for x,y in zip(ci,comp['metrics'][m]['ci95']))
    last = 0.
    for n,name in enumerate(sorted(pvalues,key=pvalues.get)):
        last = max(last,min(1.,pvalues[name]*(3-n)))
        assert math.isclose(last,report['comparisons'][name]['hit_holm_p'],abs_tol=1e-12)
    result = {'status':'VERIFIED','raw_chronology_users':180,'training_examples_checked':len(examples),
              'independent_scores':1260,'model_rankings_replayed':540,'historical_controls_matched':720,
              'historical_live_rankings_matched':360,'statistics_verified':True,'source_hashes_verified':True,
              'api_requests':0,'report_sha256':digest(directory/'report.json'),'auditor_sha256':digest(Path(__file__))}
    with output.open('x',encoding='utf-8') as f: json.dump(result,f,indent=2); f.write('\n')
    return result


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.directory,args.output),indent=2))
