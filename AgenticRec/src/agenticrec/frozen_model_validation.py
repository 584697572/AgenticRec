"""Verify frozen models without additional search; exposed test is exploratory."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import random
import statistics
import time

import torch

from .data import digest
from .models.bpr import BPRMF
from .models.lightgcn import LightGCN, normalized_bipartite_graph
from .recommendation_search import evaluate
from .lightgcn_training import _code_digest
from .training import ROOT, load_train_contract, load_frozen_cohort


def trial_directory(kind,seed):
    name='bpr_l0_d32' if kind=='bpr' else 'lightgcn_l1_d64'
    run='lightgcn_search_20261010' if seed==42 else f'frozen_model_validation_20261011_seed_{seed}'
    return ROOT/'artifacts/runs/recommendation_search'/run/name


def load_search_model(directory,expected_trial,expected_seed,contract):
    directory=Path(directory)
    manifest,n_users,n_items,seen,edges=contract
    result=json.loads((directory/'result.json').read_text(encoding='utf-8'))
    checkpoint=directory/'best.pt'
    if result['status']!='VERIFIED' or result['trial']!=expected_trial or result['seed']!=expected_seed:
        raise ValueError('Search result/config/seed mismatch')
    if digest(checkpoint)!=result['checkpoint_sha256']: raise ValueError('Checkpoint SHA mismatch')
    identity=result['identity']
    expected_code=hashlib.sha256((_code_digest()+digest(Path(__file__).with_name('recommendation_search.py'))+
                  digest(Path(__file__).with_name('lightgcn_diagnostic.py'))).encode()).hexdigest()
    checks={'id_map_sha256':manifest['id_map_sha256'],
        'split_sha256':manifest['files']['split_manifest.json']['sha256'],
        'train_positive_sha256':manifest['files']['train_positive.parquet']['sha256'],
        'data_manifest_sha256':digest(ROOT/'artifacts/data_manifest.json'),
        'valid_cohort_sha256':digest(ROOT/'data/eval_private/ml-1m-v1/valid.json'),
        'code_sha256':expected_code,'seed':expected_seed,'torch_version':str(torch.__version__)}
    if any(identity[k]!=v for k,v in checks.items()): raise ValueError('Frozen model identity mismatch')
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if payload['identity']!=identity or payload['trial']!=expected_trial or payload['dimensions']!=[n_users,n_items]:
        raise ValueError('Checkpoint internal identity/dimensions mismatch')
    if payload['best_epoch']!=result['best_epoch']: raise ValueError('Best epoch mismatch')
    if expected_trial['model']=='bpr': model=BPRMF(n_users,n_items,expected_trial['embedding_dim'])
    else:
        graph=normalized_bipartite_graph(n_users,n_items,edges)
        model=LightGCN(n_users,n_items,expected_trial['embedding_dim'],expected_trial['layers'],graph)
    model.load_state_dict(payload['state_dict'],strict=True); model.eval()
    return model,result


def clustered_pair(rows_a,rows_b,field,*,samples=2000,seed=42):
    if set(rows_a)!=set(rows_b) or not rows_a: raise ValueError('Pair cohort mismatch')
    ids=sorted(rows_a); diffs=[rows_a[u][field]-rows_b[u][field] for u in ids]
    rng=random.Random(seed)
    boots=sorted(statistics.mean(rng.choices(diffs,k=len(diffs))) for _ in range(samples))
    return {'difference':statistics.mean(diffs),'ci95':[boots[int(.025*samples)],boots[min(samples-1,int(.975*samples))]],
            'users':len(ids),'bootstrap_samples':samples,'seed':seed,'unit':'user after within-user seed average'}


def average_rows(seed_rows):
    ids=set(seed_rows[0])
    if any(set(r)!=ids for r in seed_rows): raise ValueError('Seed cohorts differ')
    return {u:{field:statistics.mean(r[u][field] for r in seed_rows) for field in ('recall','hit','mrr','ndcg')}
            for u in sorted(ids)}


def strata(rows,seen):
    result={}
    for name,lo,hi in [('low_activity',0,100),('medium_activity',100,300),('high_activity',300,float('inf'))]:
        subset=[v for u,v in rows.items() if lo<=len(seen[int(u)])<hi]
        result[name]={'users':len(subset),**{f:statistics.mean(v[f] for v in subset) if subset else None
                                          for f in ('recall','hit','mrr','ndcg')}}
    return result


def run(output):
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    config_path=ROOT/'AgenticRec/configs/recommendation_search_selected_20261010.json'
    config=json.loads(config_path.read_text(encoding='utf-8'))
    contract=load_train_contract();manifest,nu,ni,seen,edges=contract
    cohorts={s:load_frozen_cohort(s,ROOT/'artifacts/data_manifest.json') for s in ('valid','test')}
    popular=Counter(i for _,i in edges)
    report={'schema_version':1,'status':'VERIFIED','scope':'Fixed configs; no new search',
        'test_status':'Previously Exposed Test','selection':'Configs frozen by valid NDCG on 2026-10-10',
        'config_sha256':digest(config_path),'data_manifest_sha256':digest(ROOT/'artifacts/data_manifest.json'),
        'code_sha256':digest(Path(__file__)),'python':platform.python_version(),'torch':str(torch.__version__),
        'seeds':[7,42,2026],'runs':{},'aggregate':{},'paired':{},'activity_strata':{},'remote_api_requests':0}
    all_rows={}
    for kind in ('bpr','lightgcn'):
        trial=config['selected'][kind]
        for seed in report['seeds']:
            directory=trial_directory(kind,seed)
            model,original=load_search_model(directory,trial,seed,contract)
            run={'checkpoint_sha256':original['checkpoint_sha256'],'source_result_sha256':digest(directory/'result.json'),
                 'best_epoch':original['best_epoch'],'actual_epochs':original['actual_epochs'],
                 'training_elapsed_seconds':original['elapsed_seconds'],'training_timing_limit':'Concurrent development runs, not isolated speed comparison',
                 'parameter_count':sum(p.numel() for p in model.parameters()),'metrics':{},'diversity':{},'inference_seconds':{}}
            for split,cohort in cohorts.items():
                tick=time.perf_counter();metrics,rows,diversity=evaluate(model,cohort,popular)
                run['inference_seconds'][split]=time.perf_counter()-tick
                if split=='valid' and metrics!=original['valid_metrics']: raise ValueError('Reloaded valid metrics differ')
                path=output/f'{kind}_seed_{seed}_{split}_predictions.json'
                path.write_text(json.dumps(rows,sort_keys=True)+'\n',encoding='utf-8')
                run['metrics'][split]=metrics; run['diversity'][split]=diversity
                all_rows[kind,seed,split]=rows
                run[split+'_predictions_sha256']=digest(path)
            report['runs'][f'{kind}_{seed}']=run
            print(json.dumps({'model':kind,'seed':seed,'metrics':run['metrics']}),flush=True)
    for split in cohorts:
        averaged={kind:average_rows([all_rows[kind,s,split] for s in report['seeds']]) for kind in ('bpr','lightgcn')}
        report['aggregate'][split]={kind:{f:{'mean':statistics.mean(report['runs'][f'{kind}_{s}']['metrics'][split][f] for s in report['seeds']),
            'std':statistics.stdev(report['runs'][f'{kind}_{s}']['metrics'][split][f] for s in report['seeds'])}
            for f in ('recall','hit','mrr','ndcg')} for kind in ('bpr','lightgcn')}
        report['paired'][split]={f:clustered_pair(averaged['lightgcn'],averaged['bpr'],f) for f in ('recall','hit','mrr','ndcg')}
        report['activity_strata'][split]={kind:strata(rows,seen) for kind,rows in averaged.items()}
    report['raw_directory']=output.relative_to(ROOT).as_posix()
    report['metric_denominators']={s:{k:c['coverage_only'][k] for k in ('users_with_relevance','users_without_relevance')}
                                  for s,c in cohorts.items()}
    report['gpu_peak_memory']='NOT APPLICABLE: CPU execution'
    report['inference_timing_limit']='Includes batch scoring and Python full-universe TopK; not online latency/QPS'
    report['checkpoint_reload_exact']=True
    (output/'summary.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args().output)
