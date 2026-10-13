"""Finite supervised repair: prepare, sequence, reranker, freeze, evaluate."""
import argparse
from collections import defaultdict,Counter
import json
from pathlib import Path
import time

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from .simple_baselines import ROOT,load,write_new,digest,quality,paired_interval
from .sequence_ml1m import verify as verify_old
from .rank_upgrade_models import ShortSequence,PopularResidual,FeatureBuilder,CFIndex,padded_histories,mixed_scores
from .temporal_repair_models import temporal_examples,ordinal_examples,corrected_positive_loss,any_positive_loss,group_weights
from ..retrieval.sources import CandidateRetriever
from ..ranking.constraints import Catalog

OLD = ROOT/'optimized/artifacts/comparison/sequence_ml1m_20261013_run2'
SOURCE = ROOT/'optimized/artifacts/comparison/c1_hit_20261013'
DATA = ROOT/'optimized/data/processed/ml-1m-v1'
PROTOCOL = ROOT/'optimized/reports/temporal_repair_protocol_20261013.md'
SEEDS = (7,42,2026)
ROUTES = ('masked_ordinal','long_mixed')


def verify(directory):
    contract = load(directory/'contract.json')
    for name,sha in contract['files'].items():
        if digest(ROOT/name)!=sha:raise ValueError('Frozen source changed: '+name)
    return contract


def env_for(directory,graph=None):
    catalog = {int(k):v for k,v in load(directory/'catalog.json').items()}
    base = CandidateRetriever.from_frozen()
    if graph:
        table = pq.read_table(directory/graph)
        edges = list(zip(table['user_id'].to_pylist(),table['item_id'].to_pylist()))
        base = CandidateRetriever(Catalog.from_frozen(),edges)
    else:
        edges = [(u,i) for u,items in base.user_items.items() for i in sorted(items)]
    index = CFIndex(base.catalog.n_items,edges)
    features = FeatureBuilder(index,base.content_vectors,{i:list(v.genres) for i,v in base.catalog.items.items()})
    return dict(catalog=catalog,base=base,index=index,features=features)


def source_pool(env,liked,extra=()):
    cf,content,_,_ = env['features'].context(liked)
    eligible = sorted(set(env['catalog'])-set(liked))
    routes = [sorted(eligible,key=lambda i:(-float(v[i-1]),i))[:100] for v in (cf,content,env['index'].popularity)]
    return sorted(set(extra)-set(liked)|set().union(*map(set,routes)))


def prepare(directory):
    if directory.exists():raise FileExistsError('Use fresh directory')
    old = verify_old(OLD)
    catalog = load(ROOT/'optimized/artifacts/comparison/three_way_20261013/catalog.json')['items']
    confirmed = set(map(int,catalog))
    rows = pq.read_table(DATA/'train_positive.parquet').to_pylist()
    ordinal,long = ordinal_examples(rows),temporal_examples(rows,confirmed)
    if not long:raise ValueError('No real long-horizon supervision')
    directory.mkdir(parents=True)
    write_new(directory/'catalog.json',catalog)
    write_new(directory/'ordinal.json',ordinal);write_new(directory/'long.json',long)
    for phase in ('development_public','development_private','formal_private'):
        write_new(directory/(phase+'.json'),load(SOURCE/(phase+'.json')))
    write_new(directory/'formal_public.json',load(OLD/'public.json'))
    ordered = sorted(r['timestamp'] for r in rows)
    for number,q in enumerate((.35,.5)):
        anchor = ordered[int(len(ordered)*q)]
        past = [r for r in rows if r['timestamp']<anchor]
        pq.write_table(pa.Table.from_pylist(past),directory/f'graph{number}.parquet')
        histories,prefix,targets = defaultdict(list),defaultdict(set),defaultdict(set)
        for r in rows:
            u,i = r['user_id'],r['item_id']
            if r['timestamp']<anchor:
                prefix[u].add(i)
                if i in confirmed:histories[u].append((r['timestamp'],i))
            elif r['timestamp']>=anchor+30*86400 and i in confirmed:targets[u].add(i)
        queries = []
        for u in sorted(targets):
            hist = sorted(histories[u])
            if len(hist)<10:continue
            for h in (3,5,10):
                queries.append(dict(training_user_id=u,anchor=anchor,liked_item_ids=[i for t,i in hist[-h:]],
                    masked_item_ids=sorted(prefix[u]),relevant_item_ids=sorted(targets[u]),gap_days=30))
        write_new(directory/f'rerank_queries{number}.json',queries)
    paths = list((ROOT/'optimized/AgenticRec/src/agenticrec').rglob('*.py'))
    paths += [PROTOCOL,ROOT/'optimized/AgenticRec/scripts/audit_temporal_repair.py',
              ROOT/'optimized/AgenticRec/scripts/audit_three_way_comparison.py']
    paths += [DATA/n for n in ('train.parquet','train_positive.parquet','eval_valid.parquet','eval_test.parquet','id_map.json')]
    paths += [OLD/n for n in ('contract.json','predictions.json','report.json','models/seed_42.pt')]
    paths += [p for p in directory.iterdir() if p.is_file()]
    result = {'status':'FROZEN','model_items':old['model_items'],'catalog_items':len(catalog),
        'ordinal_examples':len(ordinal),'long_examples':len(long),
        'long_users':len({r['training_user_id'] for r in long}),
        'long_gap_counts':dict(Counter(r['gap_days'] for r in long)),
        'seeds':list(SEEDS),'primary_seed':42,'files':{p.relative_to(ROOT).as_posix():digest(p) for p in paths}}
    write_new(directory/'contract.json',result)
    return {k:v for k,v in result.items() if k!='files'}


def history_tensor(public):return padded_histories([dict(liked_item_ids=[r['item_id'] for r in p['catalog_context']]) for p in public])


def sequence_values(directory,config,seed,public,pop):
    n = verify(directory)['model_items']
    checkpoint = directory/config['route']/f'seed_{seed}_epoch_{config["epoch"]}.pt'
    model = ShortSequence(n); model.load_state_dict(torch.load(checkpoint,map_location='cpu')['state_dict']);model.eval()
    with torch.no_grad():raw = model(history_tensor(public))
    return torch.stack([mixed_scores(v,pop,config['alpha']) for v in raw])


def ordering(values,liked,catalog,k=10):
    if not torch.isfinite(values).all():raise ValueError('Nonfinite ranking score')
    v = values.tolist()
    return sorted(set(catalog)-set(liked),key=lambda i:(-v[i-1],i))[:k]


def metrics_on(ids,public,labels,catalog):
    if [p['episode_id'] for p in public]!=[p['episode_id'] for p in labels]:raise ValueError('Unpaired development labels')
    rows = [quality([catalog[i]['raw_id'] for i in prediction],label['labels'][0]['relevant_raw_ids'])
            for prediction,label in zip(ids,labels)]
    return {m:sum(r[m] for r in rows)/len(rows) for m in ('hit','recall','ndcg','mrr')}


def choose(candidates):return max(candidates,key=lambda r:(r['metrics']['hit'],r['metrics']['ndcg'],-r['grid_index']))


def sequence(directory):
    contract = verify(directory)
    if (directory/'sequence_selection.json').exists() or (directory/ROUTES[0]).exists():raise FileExistsError('Training cannot overwrite')
    torch.set_num_threads(2)
    ordinal,long = load(directory/'ordinal.json'),load(directory/'long.json')
    curves = []; started = time.perf_counter()
    for route in ROUTES:
        target = directory/route;target.mkdir()
        examples = ordinal if route=='masked_ordinal' else ordinal+long
        x = padded_histories(examples); y = torch.zeros((len(examples),contract['model_items']))
        mask = torch.zeros_like(y,dtype=torch.bool)
        for n,row in enumerate(examples):
            y[n,[i-1 for i in row['relevant_item_ids']]]=1
            mask[n,[i-1 for i in row['masked_item_ids']]]=True
        weights = group_weights(examples)
        for seed in SEEDS:
            torch.manual_seed(seed); model = ShortSequence(contract['model_items'])
            opt = torch.optim.Adam(model.parameters(),lr=.002,weight_decay=1e-4)
            for epoch in range(1,11):
                model.train(); total = 0.
                batches = torch.randperm(len(examples)) if route=='masked_ordinal' else torch.multinomial(weights,len(examples),replacement=True)
                for batch in batches.split(32):
                    loss = corrected_positive_loss(model(x[batch]),y[batch],mask[batch])
                    opt.zero_grad();loss.backward();opt.step();total+=float(loss.detach())*len(batch)
                step = dict(route=route,seed=seed,epoch=epoch,loss=total/len(examples));curves.append(step)
                print(json.dumps(step),flush=True)
                if epoch in (5,10):torch.save({'state_dict':model.state_dict(),'route':route,'seed':seed,'epoch':epoch},target/f'seed_{seed}_epoch_{epoch}.pt')
    public,labels = load(directory/'development_public.json'),load(directory/'development_private.json')
    env = env_for(directory); candidates = []; dev_predictions=[]
    for route in ROUTES:
        for epoch in (5,10):
            for alpha in (.25,.5,.75):
                config = dict(route=route,epoch=epoch,alpha=alpha); perseed=[]
                for seed in SEEDS:
                    scores = sequence_values(directory,config,seed,public,env['index'].popularity)
                    ids = [ordering(scores[n],[r['item_id'] for r in p['catalog_context']],env['catalog']) for n,p in enumerate(public)]
                    dev_predictions += [dict(grid_index=len(candidates),seed=seed,episode_id=p['episode_id'],item_ids=prediction)
                                        for p,prediction in zip(public,ids)]
                    perseed.append(metrics_on(ids,public,labels,env['catalog']))
                candidates.append(dict(config=config,grid_index=len(candidates),
                    metrics={m:sum(r[m] for r in perseed)/3 for m in perseed[0]},seed_metrics=perseed))
    result = dict(status='DEVELOPED',selected=choose(candidates)['config'],candidates=candidates,
                  curves=curves,elapsed_seconds=time.perf_counter()-started,test_used_for_selection=False)
    write_new(directory/'sequence_dev_predictions.json',dev_predictions)
    write_new(directory/'sequence_selection.json',result)
    return result


def rank_reranker(model,env,public,extras,alpha):
    predictions=[]
    for p,extra in zip(public,extras):
        liked=[r['item_id'] for r in p['catalog_context']]; candidates=source_pool(env,liked,extra)
        x=env['features'].features(liked,candidates)
        with torch.no_grad(): values=model(x,alpha=alpha).tolist()
        predictions.append([i for i,v in sorted(zip(candidates,values),key=lambda pair:(-pair[1],pair[0]))[:10]])
    return predictions


def reranker(directory):
    verify(directory)
    if (directory/'reranker').exists():raise FileExistsError('Reranker cannot overwrite')
    torch.set_num_threads(2); target=directory/'reranker';target.mkdir()
    examples=[]; stats=[]
    for number in (0,1):
        env=env_for(directory,f'graph{number}.parquet'); queries=load(directory/f'rerank_queries{number}.json')
        usable=0
        for q in queries:
            candidates=source_pool(env,q['liked_item_ids'])
            positive=torch.tensor([i in q['relevant_item_ids'] for i in candidates],dtype=torch.bool)
            if not positive.any():continue
            allowed=torch.tensor([i not in q['masked_item_ids'] for i in candidates],dtype=torch.bool)
            features=env['features'].features(q['liked_item_ids'],candidates)
            examples.append((features,positive,allowed));usable+=1
        stats.append(dict(snapshot=number,queries=len(queries),usable=usable,graph_edges=sum(env['base'].popularity.values())))
    if not examples:raise ValueError('No real reranking supervision')
    flat=torch.cat([e[0] for e in examples]); means=flat.mean(0);scales=flat.std(0,unbiased=False).clamp_min(1e-6)
    maxlen=max(len(e[0]) for e in examples)
    x=torch.zeros((len(examples),maxlen,10)); pos=torch.zeros((len(examples),maxlen),dtype=torch.bool); allowed=torch.zeros_like(pos)
    for n,(f,p,a) in enumerate(examples):x[n,:len(f)]=f;pos[n,:len(p)]=p;allowed[n,:len(a)]=a
    curves=[];started=time.perf_counter()
    for seed in SEEDS:
        torch.manual_seed(seed);model=PopularResidual(means,scales);opt=torch.optim.Adam(model.parameters(),lr=.002,weight_decay=1e-4)
        for epoch in range(1,21):
            model.train();total=0.
            for batch in torch.randperm(len(examples)).split(32):
                loss=any_positive_loss(model(x[batch]),pos[batch],allowed[batch])
                opt.zero_grad();loss.backward();opt.step();total+=float(loss.detach())*len(batch)
            curves.append(dict(seed=seed,epoch=epoch,loss=total/len(examples)))
            if epoch in (10,20):torch.save(dict(state_dict=model.state_dict(),seed=seed,epoch=epoch),target/f'seed_{seed}_epoch_{epoch}.pt')
        print(json.dumps(dict(route='reranker',seed=seed,epochs=20,queries=len(examples))),flush=True)
    write_new(directory/'reranker_training.json',dict(status='TRAINED',queries=len(examples),snapshot_stats=stats,curves=curves,elapsed_seconds=time.perf_counter()-started))
    env=env_for(directory);public=load(directory/'development_public.json');labels=load(directory/'development_private.json')
    sequence_config=load(directory/'sequence_selection.json')['selected']
    extras={}
    for seed in SEEDS:
        scores=sequence_values(directory,sequence_config,seed,public,env['index'].popularity)
        extras[seed]=[ordering(scores[n],[r['item_id'] for r in p['catalog_context']],env['catalog'],100) for n,p in enumerate(public)]
    candidates=[];dev_predictions=[]
    for epoch in (10,20):
        for alpha in (.5,1.):
            for include_sequence in (False,True):
                perseed=[]
                for seed in SEEDS:
                    model=PopularResidual(means,scales);model.load_state_dict(torch.load(target/f'seed_{seed}_epoch_{epoch}.pt',map_location='cpu')['state_dict']);model.eval()
                    ids=rank_reranker(model,env,public,extras[seed] if include_sequence else [[] for _ in public],alpha)
                    dev_predictions += [dict(grid_index=len(candidates),seed=seed,episode_id=p['episode_id'],item_ids=prediction)
                                        for p,prediction in zip(public,ids)]
                    perseed.append(metrics_on(ids,public,labels,env['catalog']))
                candidates.append(dict(config=dict(epoch=epoch,alpha=alpha,include_sequence=include_sequence),grid_index=len(candidates),
                    metrics={m:sum(r[m] for r in perseed)/3 for m in perseed[0]},seed_metrics=perseed))
    result=dict(status='DEVELOPED',selected=choose(candidates)['config'],candidates=candidates,test_used_for_selection=False)
    write_new(directory/'reranker_dev_predictions.json',dev_predictions)
    write_new(directory/'reranker_selection.json',result)
    return result


def freeze(directory):
    verify(directory)
    a,b=load(directory/'sequence_selection.json'),load(directory/'reranker_selection.json')
    va,vb=choose(a['candidates']),choose(b['candidates'])
    final='reranker' if (vb['metrics']['hit'],vb['metrics']['ndcg'])>(va['metrics']['hit'],va['metrics']['ndcg']) else 'sequence'
    files=[directory/'sequence_selection.json',directory/'reranker_selection.json',directory/'reranker_training.json',
           directory/'sequence_dev_predictions.json',directory/'reranker_dev_predictions.json']
    files+=list(directory.glob('masked_ordinal/*.pt'))+list(directory.glob('long_mixed/*.pt'))+list(directory.glob('reranker/*.pt'))
    result=dict(status='FROZEN',sequence=a['selected'],reranker=b['selected'],final=final,
                dev_metrics=dict(sequence=va['metrics'],reranker=vb['metrics']),
                files={p.relative_to(directory).as_posix():digest(p) for p in files})
    write_new(directory/'frozen.json',result)
    return result


def evaluate(directory):
    verify(directory);frozen=load(directory/'frozen.json')
    for name,sha in frozen['files'].items():
        if digest(directory/name)!=sha:raise ValueError('Changed frozen model: '+name)
    if (directory/'predictions.json').exists():raise FileExistsError('Predictions cannot overwrite')
    torch.set_num_threads(2);env=env_for(directory);public=load(directory/'formal_public.json');predictions=[]
    for seed in SEEDS:
        scores=sequence_values(directory,frozen['sequence'],seed,public,env['index'].popularity)
        extra=[ordering(scores[n],[r['item_id'] for r in p['catalog_context']],env['catalog'],100) for n,p in enumerate(public)]
        checkpoint=torch.load(directory/'reranker'/f'seed_{seed}_epoch_{frozen["reranker"]["epoch"]}.pt',map_location='cpu')
        model=PopularResidual(checkpoint['state_dict']['means'],checkpoint['state_dict']['scales']);model.load_state_dict(checkpoint['state_dict']);model.eval()
        reranked=rank_reranker(model,env,public,extra if frozen['reranker']['include_sequence'] else [[] for p in public],frozen['reranker']['alpha'])
        for n,p in enumerate(public):
            for stage,ids in [('sequence',extra[n][:10]),('reranker',reranked[n])]:
                predictions.append(dict(system=stage+str(seed),episode_id=p['episode_id'],history_n=len(p['catalog_context']),
                    item_ids=ids,raw_ids=[env['catalog'][i]['raw_id'] for i in ids]))
    controls=[r for r in load(OLD/'predictions.json') if r['system'] in ('C1','candidate','popular','itemcf','sequence42')]
    controls=[dict(r,system='old_sequence42' if r['system']=='sequence42' else r['system']) for r in controls]
    write_new(directory/'predictions.json',predictions+controls)
    labels={r['episode_id']:r for r in load(directory/'formal_private.json')};records=[]
    for row in predictions+controls:
        records.append(dict(row,metrics=quality(row['raw_ids'],labels[row['episode_id']]['labels'][0]['relevant_raw_ids'])))
    write_new(directory/'scores.json',records)
    by=defaultdict(dict)
    for r in records:by[r['system']][r['episode_id']]=r
    systems={s:dict(n=180,hits=int(sum(r['metrics']['hit'] for r in v.values())),
        **{m:sum(r['metrics'][m] for r in v.values())/180 for m in ('hit','recall','ndcg','mrr')}) for s,v in by.items()}
    main=by[frozen['final']+'42'];keys=sorted(main);comparisons={}
    import math
    for bn,s in enumerate(('candidate','popular','old_sequence42')):
        wins=sum(main[i]['metrics']['hit']>by[s][i]['metrics']['hit'] for i in keys);losses=sum(main[i]['metrics']['hit']<by[s][i]['metrics']['hit'] for i in keys);n=wins+losses
        comparisons[s]=dict(wins=wins,losses=losses,hit_exact_p=min(1.,2*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/2**n),
            metrics={m:paired_interval([main[i]['metrics'][m]-by[s][i]['metrics'][m] for i in keys],2026101320+bn*10+mn)
                     for mn,m in enumerate(('hit','recall','ndcg','mrr'))})
    last=0.
    for rank,name in enumerate(sorted(comparisons,key=lambda k:comparisons[k]['hit_exact_p'])):
        last=max(last,min(1.,comparisons[name]['hit_exact_p']*(3-rank)));comparisons[name]['hit_holm_p']=last
    result=dict(status='EVALUATED',scope='post_hoc_original_180',final_stage=frozen['final'],sequence_config=frozen['sequence'],
        reranker_config=frozen['reranker'],systems=systems,comparisons=comparisons,api_requests=0,downloads=0,
        strict_agent_tasks_evaluated=False,default_changed=False,predictions_sha256=digest(directory/'predictions.json'),
        frozen_sha256=digest(directory/'frozen.json'),contract_sha256=digest(directory/'contract.json'))
    write_new(directory/'report.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','sequence','reranker','freeze','evaluate'))
    parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(globals()[args.phase](args.directory.resolve()),indent=2),flush=True)
