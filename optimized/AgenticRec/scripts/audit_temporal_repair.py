"""Independent raw sample rebuild, selection scoring and frozen-model replay."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random

import pyarrow.parquet as pq
import torch

from audit_three_way_comparison import assess,digest


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def audit(directory,output):
    root=Path(__file__).resolve().parents[3]; data=root/'optimized/data/processed/ml-1m-v1'
    contract=read(directory/'contract.json');frozen=read(directory/'frozen.json')
    for name,sha in contract['files'].items():assert digest(root/name)==sha,name
    for name,sha in frozen['files'].items():assert digest(directory/name)==sha,name
    catalog={int(k):v for k,v in read(directory/'catalog.json').items()}
    mapping=read(data/'id_map.json');users={int(k):int(v) for k,v in mapping['user_to_model'].items()};items={int(k):int(v) for k,v in mapping['item_to_model'].items()}
    rawtrain=pq.read_table(data/'train.parquet').to_pylist();seqs=defaultdict(list);rawseen=defaultdict(set);rawhist=defaultdict(list)
    for r in rawtrain:
        rawseen[r['raw_user_id']].add(r['raw_item_id'])
        if r['rating']>=4 and r['raw_user_id'] in users and r['raw_item_id'] in items:
            u,i=users[r['raw_user_id']],items[r['raw_item_id']];seqs[u].append((r['timestamp'],i))
            if i in catalog:rawhist[r['raw_user_id']].append((r['timestamp'],i))
    ordinal=[];long=[];unique=set()
    for u,seq in sorted(seqs.items()):
        seq=sorted(seq);cut=int(len(seq)*.8)
        if cut>=10 and cut<len(seq):
            for h in (3,5,10):ordinal.append(dict(training_user_id=u,liked_item_ids=[i for ts,i in seq[cut-h:cut]],
                relevant_item_ids=sorted(i for ts,i in seq[cut:]),masked_item_ids=sorted(i for ts,i in seq[:cut]),kind='ordinal'))
        visible=[r for r in seq if r[1] in catalog]
        for ratio in (.25,.5,.75,.8):
            ix=int(len(visible)*ratio)
            if ix<10:continue
            anchor=visible[ix-1][0];history=[i for ts,i in visible if ts<=anchor];past=sorted(i for ts,i in seq if ts<=anchor)
            for gap in (30,90):
                target=sorted(i for ts,i in visible if ts>=anchor+gap*86400)
                if not target:continue
                for h in (3,5,10):
                    key=(u,anchor,gap,h)
                    if key in unique:continue
                    unique.add(key);long.append(dict(training_user_id=u,anchor=anchor,gap_days=gap,liked_item_ids=history[-h:],
                        relevant_item_ids=target,masked_item_ids=past,kind='long'))
    assert ordinal==read(directory/'ordinal.json') and long==read(directory/'long.json')
    rows=pq.read_table(data/'train_positive.parquet').to_pylist();times=sorted(r['timestamp'] for r in rows)
    for number,ratio in enumerate((.35,.5)):
        anchor=times[int(len(times)*ratio)];graph=pq.read_table(directory/f'graph{number}.parquet').to_pylist()
        assert graph==[r for r in rows if r['timestamp']<anchor]
        queries=[]
        for u,seq in sorted(seqs.items()):
            hist=sorted((ts,i) for ts,i in seq if ts<anchor and i in catalog)
            target=sorted(i for ts,i in seq if ts>=anchor+30*86400 and i in catalog)
            if len(hist)<10 or not target:continue
            for h in (3,5,10):queries.append(dict(training_user_id=u,anchor=anchor,liked_item_ids=[i for ts,i in hist[-h:]],
                masked_item_ids=sorted(i for ts,i in seq if ts<anchor),relevant_item_ids=target,gap_days=30))
        assert queries==read(directory/f'rerank_queries{number}.json')
    truth=read(root/'optimized/artifacts/comparison/three_way_20261013/truth_private.json')
    selections={}
    for phase in ('sequence','reranker'):
        selection=read(directory/(phase+'_selection.json'));dev=read(directory/(phase+'_dev_predictions.json'))
        labels={r['episode_id']:r for r in read(directory/'development_private.json')};totals=defaultdict(lambda:defaultdict(list));keys=set()
        assert len(dev)==len(selection['candidates'])*450
        for row in dev:
            key=(row['grid_index'],row['seed'],row['episode_id']);assert key not in keys;keys.add(key)
            metrics,errors=assess(dict(raw_ids=[catalog[i]['raw_id'] for i in row['item_ids']],status='OK',extraction_verified=True),labels[row['episode_id']]['labels'][0],truth,[])
            assert not any(errors.values())
            totals[row['grid_index']][row['seed']].append(metrics)
        for candidate in selection['candidates']:
            for m in ('hit','recall','ndcg','mrr'):
                expected=sum(sum(r[m] for r in totals[candidate['grid_index']][seed])/150 for seed in (7,42,2026))/3
                assert math.isclose(candidate['metrics'][m],expected,abs_tol=1e-12)
        best=max(selection['candidates'],key=lambda c:(c['metrics']['hit'],c['metrics']['ndcg'],-c['grid_index']))
        assert selection['selected']==best['config']==frozen[phase]
        selections[phase]=best
    a,b=selections['sequence'],selections['reranker']
    assert frozen['final']==('reranker' if (b['metrics']['hit'],b['metrics']['ndcg'])>(a['metrics']['hit'],a['metrics']['ndcg']) else 'sequence')
    public={r['episode_id']:r for r in read(directory/'formal_public.json')}
    labels={r['episode_id']:r for r in read(directory/'formal_private.json')}
    devusers={r['evaluator_raw_user_id'] for r in read(directory/'development_private.json')}
    assert len(public)==180 and not devusers&{labels[i]['evaluator_raw_user_id'] for i in public}
    targets=defaultdict(set)
    for r in pq.read_table(data/'eval_test.parquet').to_pylist():
        if r['is_target'] and r['segment']=='WARM' and r['raw_item_id'] in items and items[r['raw_item_id']] in catalog and r['raw_item_id'] not in rawseen[r['raw_user_id']]:targets[r['raw_user_id']].add(r['raw_item_id'])
    for eid,p in public.items():
        lab=labels[eid];u=lab['evaluator_raw_user_id'];h=lab['history_n']
        assert [r['item_id'] for r in p['catalog_context']]==[i for ts,i in sorted(rawhist[u])[-h:]]
        assert set(lab['labels'][0]['relevant_raw_ids'])==targets[u]
    predictions,scores=read(directory/'predictions.json'),read(directory/'scores.json');report=read(directory/'report.json')
    assert len(predictions)==len(scores)==1980
    assert digest(directory/'predictions.json')==report['predictions_sha256']
    assert (directory/'predictions.json').stat().st_mtime_ns<=(directory/'scores.json').stat().st_mtime_ns
    by=defaultdict(dict)
    for pred,row in zip(predictions,scores):
        assert all(row[k]==v for k,v in pred.items());assert row['raw_ids']==[catalog[i]['raw_id'] for i in row['item_ids']]
        expected,errors=assess(dict(raw_ids=row['raw_ids'],status='OK',extraction_verified=True),labels[row['episode_id']]['labels'][0],truth,[])
        assert not any(errors.values())
        for m in ('hit','recall','ndcg','mrr'):assert math.isclose(expected[m],row['metrics'][m],abs_tol=1e-12)
        assert row['episode_id'] not in by[row['system']];by[row['system']][row['episode_id']]=row
    for s,rows in by.items():
        assert len(rows)==180 and report['systems'][s]['hits']==sum(r['metrics']['hit'] for r in rows.values())
        for m in ('hit','recall','ndcg','mrr'):assert math.isclose(report['systems'][s][m],sum(r['metrics'][m] for r in rows.values())/180,abs_tol=1e-12)
    # Frozen inference replay; scorer above does not import the experiment executor.
    from agenticrec.evaluation.temporal_repair import env_for,sequence_values,ordering,rank_reranker
    from agenticrec.evaluation.rank_upgrade_models import PopularResidual
    torch.set_num_threads(2);env=env_for(directory);pubs=list(public.values())
    for seed in (7,42,2026):
        values=sequence_values(directory,frozen['sequence'],seed,pubs,env['index'].popularity)
        extras=[ordering(values[n],[r['item_id'] for r in p['catalog_context']],catalog,100) for n,p in enumerate(pubs)]
        checkpoint=torch.load(directory/'reranker'/f'seed_{seed}_epoch_{frozen["reranker"]["epoch"]}.pt',map_location='cpu')['state_dict']
        model=PopularResidual(checkpoint['means'],checkpoint['scales']);model.load_state_dict(checkpoint);model.eval()
        ranked=rank_reranker(model,env,pubs,extras if frozen['reranker']['include_sequence'] else [[] for p in pubs],frozen['reranker']['alpha'])
        for n,p in enumerate(pubs):
            assert extras[n][:10]==by['sequence'+str(seed)][p['episode_id']]['item_ids']
            assert ranked[n]==by['reranker'+str(seed)][p['episode_id']]['item_ids']
    old={(r['system'],r['episode_id']):r for r in read(root/'optimized/artifacts/comparison/sequence_ml1m_20261013_run2/predictions.json')}
    for s in ('candidate','popular','itemcf','C1','old_sequence42'):
        for eid,row in by[s].items():assert row['raw_ids']==old['sequence42' if s=='old_sequence42' else s,eid]['raw_ids']
    main=by[frozen['final']+'42'];keys=sorted(main);pvalues={}
    for bn,s in enumerate(('candidate','popular','old_sequence42')):
        c=report['comparisons'][s];win=sum(main[i]['metrics']['hit']>by[s][i]['metrics']['hit'] for i in keys);loss=sum(main[i]['metrics']['hit']<by[s][i]['metrics']['hit'] for i in keys);n=win+loss
        assert (win,loss)==(c['wins'],c['losses']);pvalues[s]=min(1.,2*sum(math.comb(n,k) for k in range(min(win,loss)+1))/2**n)
        assert math.isclose(c['hit_exact_p'],pvalues[s],abs_tol=1e-12)
        for mn,m in enumerate(('hit','recall','ndcg','mrr')):
            delta=[main[i]['metrics'][m]-by[s][i]['metrics'][m] for i in keys];rng=random.Random(2026101320+bn*10+mn)
            samples=sorted(sum(rng.choices(delta,k=180))/180 for _ in range(10000));ci=[]
            for q in (.025,.975):
                pos=q*9999;k=int(pos);ci.append(samples[k]+(samples[k+1]-samples[k])*(pos-k))
            assert all(math.isclose(a,b,abs_tol=1e-12) for a,b in zip(ci,c['metrics'][m]['ci95']))
    last=0.
    for rank,name in enumerate(sorted(pvalues,key=pvalues.get)):
        last=max(last,min(1.,pvalues[name]*(3-rank)));assert math.isclose(last,report['comparisons'][name]['hit_holm_p'],abs_tol=1e-12)
    result=dict(status='VERIFIED',ordinal_samples_rebuilt=len(ordinal),long_samples_rebuilt=len(long),
        train_snapshots_and_supervision_verified=True,development_predictions_scored=9000,development_configurations_verified=20,
        independent_formal_scores=1980,frozen_new_predictions_replayed=1080,historical_predictions_matched=900,
        source_hashes_verified=True,statistics_verified=True,api_requests=0,report_sha256=digest(directory/'report.json'),auditor_sha256=digest(Path(__file__)))
    with output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();audit(args.directory.resolve(),args.output.resolve())
