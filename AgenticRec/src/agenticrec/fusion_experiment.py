"""Finite development/held-out evaluation for anonymous candidate fusion.

External evaluation cannot start until development selection is frozen.
Every stage owns an append-only directory and saves raw predictions/traces.
"""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import platform
import random
import statistics
import subprocess
import time

import pyarrow.parquet as pq
import torch

from .adapters.model import SessionSeedScorer
from .data import digest
from .evaluation.metrics import score_ranking, FIELDS
from .fusion_pipeline import predict_snapshot
from .pipeline import FixedRequest
from .ranking.constraints import Catalog, Constraints, Item
from .retrieval.adaptive import (_cf_scores, source_snapshot, weighted_fusion, CachedResearchRetriever,
                                gate_features, LogisticGate, candidate_diagnostics)
from .retrieval.sources import CandidateRetriever
from .training import ROOT, load_train_contract
from .lightgcn_diagnostic import validation_only_files


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        stream.write(json.dumps(value,sort_keys=True,indent=2)+'\n')


def code_identity():
    paths=[Path(__file__),Path(__file__).with_name('fusion_pipeline.py'),
           Path(__file__).with_name('research_protocol.py'),
           Path(__file__).parent/'retrieval/adaptive.py',Path(__file__).parent/'retrieval/sources.py',
           Path(__file__).parent/'ranking/constraints.py',Path(__file__).parent/'evaluation/metrics.py']
    return {p.relative_to(ROOT).as_posix():digest(p) for p in paths}


def load_development():
    with validation_only_files():
        manifest,_nu,_ni,_seen,edges=load_train_contract()
        cohort=json.loads((ROOT/'data/eval_private/ml-1m-v1/valid.json').read_text(encoding='utf-8'))
        directory=ROOT/'data/eval_private/fusion-development-20261011'
        frozen=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
        if digest(directory/'user_groups.json')!=frozen['user_groups_sha256'] or digest(ROOT/'data/eval_private/ml-1m-v1/valid.json')!=frozen['valid_sha256']:
            raise ValueError('Development freeze mismatch')
        groups=json.loads((directory/'user_groups.json').read_text(encoding='utf-8'))
        mapping=json.loads((ROOT/'data/processed/ml-1m-v1/id_map.json').read_text(encoding='utf-8'))
        users,items=mapping['user_to_model'],mapping['item_to_model']
        rows=[]
        for r in pq.read_table(ROOT/'data/processed/ml-1m-v1/train.parquet').to_pylist():
            u=users.get(str(r['raw_user_id'])); i=items.get(str(r['raw_item_id']))
            if u is not None and i is not None: rows.append((u,i,r['rating'],r['timestamp']))
        retriever=CachedResearchRetriever(Catalog.from_frozen(),edges)
    return retriever,rows,cohort,groups,{'data_manifest_sha256':digest(ROOT/'artifacts/data_manifest.json'),
                                       'development_manifest_sha256':digest(directory/'manifest.json')}


def load_external():
    directory=ROOT/'data/eval_private/fusion-external-100k-20261011-v2'
    manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    values={}
    for name in ('catalog.json','train.json','train_positive.json','test.json'):
        if digest(directory/name)!=manifest['files'][name]: raise ValueError('External data hash mismatch')
        values[name]=json.loads((directory/name).read_text(encoding='utf-8'))
    catalog=Catalog([Item(r['item_id'],r['title'],frozenset(r['genres']),r['year'])
                     for r in values['catalog.json'].values()])
    return CachedResearchRetriever(catalog,[tuple(e) for e in values['train_positive.json']]),values['train.json'],values['test.json'],{
        'external_manifest_sha256':digest(directory/'manifest.json'),'source_zip_sha256':manifest['source_zip_sha256']}


def requests_for_users(rows,catalog,users):
    history=defaultdict(list)
    for u,i,r,t in rows: history[u].append((t,i,r))
    result=[]
    for u in sorted(users):
        ordered=sorted(history[u]); positive=[i for t,i,r in ordered if r>=4][-3:][::-1]
        negative=[i for t,i,r in ordered if r<=2][-1:]
        seen=tuple(sorted({i for t,i,r in ordered}))
        genre=tuple(sorted(catalog.items[positive[0]].genres)[:1]) if positive else ()
        for state in ('positive_seeds','local_negative','genre_constraint'):
            result.append((u,state,FixedRequest(Constraints(k=10,include_genres=genre if state=='genre_constraint' else ()),
                liked_item_ids=tuple(positive),disliked_item_ids=tuple(negative) if state=='local_negative' else (),
                seen_item_ids=seen)))
    return result


def _score(ids,relevant,universe,seen):
    return score_ranking(ids,relevant,universe,10,seen) if relevant else None


def _legacy_prediction(base,request,snapshot,mode):
    ranks=dict(snapshot['rankings'])
    if mode in ('legacy_no_content','recall_only_off'): ranks.pop('content',None)
    fused=weighted_fusion(ranks,{s:1 for s in ranks},universe=snapshot['eligible_ids'])
    if mode=='legacy_no_content':
        scores,_=_cf_scores(base,fused.candidate_ids,request.liked_item_ids,None,False)
        ranked=sorted((i for i in scores if scores[i]>0),key=lambda i:(-scores[i],i))
        ranked += [i for i in base.popularity_rank(fused.candidate_ids,len(fused.candidate_ids)) if i not in set(ranked)]
    else:
        session=SessionSeedScorer({i:v.genres for i,v in base.catalog.items.items()},base.popularity,base.catalog.n_items)
        scores=dict(zip(fused.candidate_ids,session.score_from_seeds(request.liked_item_ids,
            () if mode=='negative_penalty_off' else request.disliked_item_ids,fused.candidate_ids)))
        ranked=sorted(fused.candidate_ids,key=lambda i:(-scores[i],-fused.rrf_scores[i],i))
    final=base.gate.finalize(ranked,snapshot['effective_constraints'],snapshot['seen_ids'])
    return final.item_ids,fused


def collect(base,rows,cohort,users,out):
    records=[]
    path=out/'source_records.jsonl'
    with path.open('x',encoding='utf-8') as stream:
        for index,(user,state,request) in enumerate(requests_for_users(rows,base.catalog,users),1):
            started=time.perf_counter()
            old=source_snapshot(base,request,positive_content=False)
            positive=source_snapshot(base,request,positive_content=True)
            if old['status']!='OK' or positive['status']!='OK':
                raise ValueError('Unexpected infeasible data-only request; must explicitly audit')
            eligible=set(positive['eligible_ids'])
            relevant=sorted(set(cohort['relevant_by_user'][str(user)])&eligible)
            record={'user':user,'state':state,'request':json.loads(json.dumps(asdict(request))),'relevant':relevant,
                    'train_activity':len(request.seen_item_ids),'features':gate_features(base,request,positive),
                    'predictions':{},'metrics':{},'latency_seconds':{}}
            for name in ('legacy_full','legacy_no_content','recall_only_off','negative_penalty_off'):
                ids,fused=_legacy_prediction(base,request,old,name)
                record['predictions'][name]=ids
                record['metrics'][name]=_score(ids,relevant,cohort['candidate_ids'],request.seen_item_ids)
                if name=='legacy_full': record['legacy_diagnostics']=candidate_diagnostics(old,fused,ids)
            popular=base.popularity_rank(positive['eligible_ids'],10)
            record['predictions']['popularity']=popular
            record['metrics']['popularity']=_score(popular,relevant,cohort['candidate_ids'],request.seen_item_ids)
            for name,mode,weight in [('no_content','no_content',0),('rrf','rrf',1),('conditional','conditional',.5)]+[
                    (f'fixed_{weight}','fixed',weight) for weight in (0,.25,.5,1)]:
                tick=time.perf_counter()
                final,fused,_scores,decision=predict_snapshot(base,request,positive,mode=mode,content_weight=weight)
                record['predictions'][name]=final.item_ids
                record['metrics'][name]=_score(final.item_ids,relevant,cohort['candidate_ids'],request.seen_item_ids)
                record['latency_seconds'][name]=time.perf_counter()-tick
                if name=='rrf': record['diagnostics']=candidate_diagnostics(positive,fused,final.item_ids)
            # Keep the frozen capped source signals sufficient to rebuild all policies.
            record['snapshot']={'rankings':positive['rankings'],'scores':positive['scores'],
                                'eligible_ids':positive['eligible_ids'],'profile_source':positive['profile_source'],
                                'per_route':positive['per_route']}
            record['source_elapsed_seconds']=time.perf_counter()-started
            stream.write(json.dumps(record,sort_keys=True)+'\n'); stream.flush()
            records.append(record)
            if index%50==0: print(json.dumps({'collected_queries':index,'total_queries':len(users)*3}),flush=True)
    return records


def reconstruct(record):
    request=FixedRequest.parse({'schema_version':1,**{k:v for k,v in record['request'].items() if k!='schema_version'}})
    snap=record['snapshot']
    snap={**snap,'scores':{s:{int(i):v for i,v in vals.items()} for s,vals in snap['scores'].items()},
          'effective_constraints':request.constraints,'seen_ids':request.seen_item_ids}
    return request,snap


def add_policy(records,base,cohort,name,*,mode,gate=None,content_weight=.5,protect_cf=10,no_quality=False):
    for rec in records:
        request,snap=reconstruct(rec)
        tick=time.perf_counter()
        final,fused,_scores,decision=predict_snapshot(base,request,snap,mode=mode,gate=gate,
            content_weight=content_weight,protect_cf=protect_cf,no_quality=no_quality)
        rec['predictions'][name]=final.item_ids
        rec['metrics'][name]=_score(final.item_ids,rec['relevant'],cohort['candidate_ids'],request.seen_item_ids)
        rec['latency_seconds'][name]=time.perf_counter()-tick
        rec.setdefault('gate_decisions',{})[name]=decision


def user_means(records,name,field='ndcg'):
    per_user=defaultdict(list)
    for r in records:
        if r['metrics'][name] is not None: per_user[r['user']].append(r['metrics'][name][field])
    return {u:statistics.mean(values) for u,values in per_user.items()}


def quality(records,name):
    vals=user_means(records,name)
    return statistics.mean(vals.values()) if vals else 0


def develop(out):
    with validation_only_files():
        base,rows,cohort,groups,identity=load_development()
        users=sorted(u for values in groups.values() for u in values)
        records=collect(base,rows,cohort,users,out)
        fit=[r for r in records if r['user'] in groups['fit'] and r['metrics']['no_content'] is not None]
        select=[r for r in records if r['user'] in groups['select']]
        labels=[int(r['metrics']['fixed_0.5']['ndcg']>r['metrics']['no_content']['ndcg']) for r in fit]
        frozen={'schema_version':1,'method':'public-signal reliability rank fusion','identity':identity,
                'seeds':[7,42,2026],'code_sha256':code_identity(),'gates':{},'fit_label_counts':dict(Counter(labels)),
                'fixed_weight':max((0,.25,.5,1),key=lambda w:(quality(select,f'fixed_{w}'),-w)),
                'fit_users':len(groups['fit']),'select_users':len(groups['select']),'audit_users':len(groups['audit']),
                'external_test_access':False,'selection_metric':'select user-mean NDCG@10','all_trials':[]}
        for seed in frozen['seeds']:
            for no_quality in (False,True):
                candidate=[]
                features=[r['features'][:4]+[0.0]*5 if no_quality else r['features'] for r in fit]
                for regularization in (.01,.1,1,10):
                    gate=LogisticGate.fit(features,labels,regularization=regularization,seed=seed)
                    name=f'trial_{seed}_{no_quality}_{regularization}'
                    add_policy(select,base,cohort,name,mode='learned',gate=gate,no_quality=no_quality)
                    value=quality(select,name)
                    frozen['all_trials'].append({'seed':seed,'no_quality':no_quality,'regularization':regularization,'select_ndcg':value})
                    candidate.append((value,-regularization,gate,regularization))
                best=max(candidate,key=lambda x:(x[0],x[1]))
                key=f'{seed}_'+('no_quality' if no_quality else 'full')
                frozen['gates'][key]={'gate':best[2].to_dict(),'regularization':best[3],'select_ndcg':best[0]}
                add_policy(records,base,cohort,'learned_'+key,mode='learned',gate=best[2],no_quality=no_quality)
                if not no_quality:
                    add_policy(records,base,cohort,f'learned_{seed}_no_protection',mode='learned',gate=best[2],protect_cf=0)
        add_policy(records,base,cohort,'fixed_best',mode='fixed',content_weight=frozen['fixed_weight'])
        write(out/'selection_frozen.json',frozen)
        write(out/'predictions.json',records)
        write(out/'identity.json',{**identity,'code_sha256':code_identity(),'groups':{k:len(v) for k,v in groups.items()}})
        write(out/'group_summary.json',{g:summarize([r for r in records if r['user'] in us]) for g,us in groups.items()})
    return frozen


def summarize(records):
    names=[name for name in records[0]['metrics'] if not name.startswith('trial_')]
    result={}
    for name in names:
        values={field:user_means(records,name,field) for field in FIELDS}
        result[name]={'users':len(values['ndcg']),'queries':len(records),
                     'queries_without_relevance':sum(r['metrics'][name] is None for r in records),
                     **{f:statistics.mean(v.values()) if v else None for f,v in values.items()},
                     'distinct_recommended_items':len({i for r in records for i in r['predictions'][name]})}
    return result


def evaluate_external(out,development):
    frozen_path=development/'selection_frozen.json'
    frozen=json.loads(frozen_path.read_text(encoding='utf-8'))
    if frozen['external_test_access'] is not False or frozen['code_sha256']!=code_identity():
        raise ValueError('Method/config/source changed after freezing')
    base,rows,cohort,identity=load_external()
    users=[u for u in cohort['user_ids'] if cohort['relevant_by_user'][str(u)]]
    records=collect(base,rows,cohort,users,out)
    for key,value in frozen['gates'].items():
        gate=LogisticGate.from_dict(value['gate'])
        add_policy(records,base,cohort,'learned_'+key,mode='learned',gate=gate,no_quality=key.endswith('no_quality'))
        if key.endswith('full'):
            seed=key.split('_')[0]
            add_policy(records,base,cohort,f'learned_{seed}_no_protection',mode='learned',gate=gate,protect_cf=0)
    add_policy(records,base,cohort,'fixed_best',mode='fixed',content_weight=frozen['fixed_weight'])
    write(out/'predictions.json',records)
    write(out/'summary.json',summarize(records))
    write(out/'identity.json',{**identity,'selection_sha256':digest(frozen_path),'code_sha256':code_identity(),
                               'users_evaluated':len(users),'warm_users':len(cohort['user_ids']),
                               'users_without_relevance':len(cohort['user_ids'])-len(users)})


def run(stage,run_id):
    if not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in run_id):
        raise ValueError('Unsafe experiment ID')
    if stage not in ('develop','external'): raise ValueError('Invalid stage')
    torch.set_num_threads(2)
    out=ROOT/'artifacts/runs/fusion_research'/run_id/stage
    out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    dirty=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    if dirty: raise ValueError('Formal research run requires committed source')
    write(out/'runtime.json',{'git_commit':git,'experiment_id':run_id,'stage':stage,'python':platform.python_version(),
        'torch':str(torch.__version__),'platform':platform.platform(),'cpu_threads':2,'cuda_available':torch.cuda.is_available(),
        'remote_api_requests':0,'model_parameters':10,'source_sha256':code_identity()})
    if stage=='develop': develop(out)
    else: evaluate_external(out,out.parent/'develop')
    write(out/'completion.json',{'status':'COMPLETED','elapsed_seconds':time.perf_counter()-started,
        'files':{p.name:digest(p) for p in out.iterdir() if p.is_file()},'remote_api_requests':0})
    print(json.dumps({'status':'COMPLETED','stage':stage,'elapsed_seconds':time.perf_counter()-started}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['develop','external'])
    parser.add_argument('--run-id',required=True)
    args=parser.parse_args(); run(args.stage,args.run_id)
