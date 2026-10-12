"""Bounded matched explicit-low-feedback study; historical trainers untouched."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import statistics
import subprocess
import time

import torch
import pyarrow.parquet as pq
from .explicit_feedback import LowSampler, explicit_loss, mapped_lows
from .hard_negative_training import PoolSampler, paired_streams, tensor_bytes

from .data import digest
from .frozen_model_validation import average_rows, clustered_pair, strata
from .lightgcn_diagnostic import validation_only_files
from .lightgcn_optimization import build_model, code_identity as baseline_code_identity, subset_cohort, validate_trial
from .recommendation_search import evaluate
from .training import ROOT, load_frozen_cohort, load_train_contract


BASE = ROOT/'artifacts/runs/lightgcn_explicit_low_20261012'


def write(path, value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False)
        f.write('\n')


def validate_plan(plan):
    keys={'schema_version','run_id','seeds','torch_threads','negative_pool_size','group_sha256',
          'selection_metric','test_access','remote_api_requests','trials'}
    if set(plan)!=keys or type(plan['schema_version']) is not int or plan['schema_version']!=1:
        raise ValueError('Invalid plan schema')
    if (plan['run_id']!='lightgcn_explicit_low_20261012' or plan['seeds']!=[7,42,2026]
            or plan['torch_threads']!=2 or type(plan['torch_threads']) is not int
            or plan['negative_pool_size']!=8 or type(plan['negative_pool_size']) is not int
            or plan['test_access'] is not False or type(plan['remote_api_requests']) is not int
            or plan['remote_api_requests']!=0 or plan['selection_metric']!='development_selection_ndcg_at_10'):
        raise ValueError('Only preregistered train/development budget is permitted')
    expected={name:(kind,frac) for kind,prefix in [('lightgcn','gcn'),('bpr','bpr')]
              for name,frac in [(prefix+'_uniform',0.),(prefix+'_low25',.25)]}
    if len(plan['trials'])!=4 or {t['name'] for t in plan['trials']}!=set(expected):
        raise ValueError('Only four fixed conditions permitted')
    for trial in plan['trials']:
        base={k:v for k,v in trial.items() if k!='low_weight'}
        validate_trial(base)
        kind,frac=expected[trial['name']]
        exact={'model':kind,'low_weight':frac,'embedding_dim':64 if kind=='lightgcn' else 32,
               'layers':1 if kind=='lightgcn' else 0,'sampling':'edge','learning_rate':.001,
               'regularization':.0001,'batch_size':2048,'max_epochs':32,'patience':8,
               'layer_weights':[.5,.5] if kind=='lightgcn' else [1.]}
        if any(trial.get(k)!=v for k,v in exact.items()) or type(trial['low_weight']) not in (float,int):
            raise ValueError('Experiment must remain inside frozen range')
    return plan


def code_identity():
    return hashlib.sha256((baseline_code_identity()+digest(Path(__file__))+digest(Path(__file__).with_name('explicit_feedback.py'))+digest(Path(__file__).with_name('hard_negative_training.py'))).encode()).hexdigest()


def plan_and_groups(path):
    plan=validate_plan(json.loads(Path(path).read_text(encoding='utf-8')))
    group_path=ROOT/'data/eval_private/fusion-development-20261011/user_groups.json'
    if digest(group_path)!=plan['group_sha256']:
        raise ValueError('Development group identity mismatch')
    groups=json.loads(group_path.read_text(encoding='utf-8'))
    selected=groups['fit']+groups['select']
    if len(selected)!=len(set(selected)) or set(selected)&set(groups['audit']):
        raise ValueError('Selection/audit overlap')
    return plan,groups


def load_lows(contract):
    manifest,nu,ni,seen,edges = contract
    data = ROOT/'data/processed/ml-1m-v1'
    if digest(data/'train.parquet') != manifest['files']['train.parquet']['sha256']:
        raise ValueError('Frozen train rating SHA mismatch')
    mapping = json.loads((data/'id_map.json').read_text(encoding='utf-8'))
    rows = pq.read_table(data/'train.parquet',columns=['raw_user_id','raw_item_id','rating']).to_pylist()
    return mapped_lows(rows,mapping['user_to_model'],mapping['item_to_model'],seen,edges)


def load_verified(directory,trial,contract,plan_sha,seed):
    original=json.loads((directory/'result.json').read_text(encoding='utf-8'))
    if digest(directory/'best.pt')!=original['checkpoint_sha256']:
        raise ValueError('Checkpoint SHA mismatch')
    manifest,nu,ni,_,_=contract
    required={'seed':seed,'plan_sha256':plan_sha,'code_sha256':code_identity(),
              'data_manifest_sha256':digest(ROOT/'artifacts/data_manifest.json'),
              'id_map_sha256':manifest['id_map_sha256'],
              'train_positive_sha256':manifest['files']['train_positive.parquet']['sha256'],
              'train_ratings_sha256':manifest['files']['train.parquet']['sha256'],
              'valid_sha256':digest(ROOT/'data/eval_private/ml-1m-v1/valid.json'),
              'torch_version':str(torch.__version__)}
    if original['status']!='VERIFIED' or original['trial']!=trial or any(original['identity'].get(k)!=v for k,v in required.items()):
        raise ValueError('Training result identity mismatch')
    payload=torch.load(directory/'best.pt',map_location='cpu',weights_only=True)
    if (payload['identity']!=original['identity'] or payload['trial']!=trial
            or payload['dimensions']!=[nu,ni] or payload['best_epoch']!=original['best_epoch']):
        raise ValueError('Checkpoint payload identity mismatch')
    model=build_model(trial,contract); model.load_state_dict(payload['state_dict'],strict=True); model.eval()
    return model,original


def train(plan_path,name,seed):
    plan,groups=plan_and_groups(plan_path)
    matches=[t for t in plan['trials'] if t['name']==name]
    if len(matches)!=1 or seed not in plan['seeds']:raise ValueError('Unregistered trial/seed')
    if seed!=42:
        frozen=json.loads((BASE/'selection_frozen.json').read_text(encoding='utf-8'))
        if frozen['plan_sha256']!=digest(plan_path) or frozen['code_sha256']!=code_identity():
            raise ValueError('Freeze seed42 selection before later seeds')
    trial=matches[0]; out=BASE/f'seed_{seed}'/name; out.mkdir(parents=True,exist_ok=False)
    (out/'plan.json').write_bytes(Path(plan_path).read_bytes())
    torch.set_num_threads(plan['torch_threads']); torch.manual_seed(seed)
    edge_rng,pool_seed,mask_generator=paired_streams(seed)
    started=time.perf_counter()
    with validation_only_files():
        contract=load_train_contract(); manifest,nu,ni,seen,edges=contract
        cohort=load_frozen_cohort('valid',ROOT/'artifacts/data_manifest.json')
        selection=subset_cohort(cohort,groups['fit']+groups['select'])
        identity={'seed':seed,'plan_sha256':digest(plan_path),'code_sha256':code_identity(),
                  'data_manifest_sha256':digest(ROOT/'artifacts/data_manifest.json'),
                  'id_map_sha256':manifest['id_map_sha256'],
                  'train_positive_sha256':manifest['files']['train_positive.parquet']['sha256'],
              'train_ratings_sha256':manifest['files']['train.parquet']['sha256'],
                  'valid_sha256':digest(ROOT/'data/eval_private/ml-1m-v1/valid.json'),
                  'torch_version':str(torch.__version__),'python':platform.python_version(),
                  'platform':platform.platform(),
                  'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  'git_dirty':bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT))}
        model=build_model(trial,contract)
        initial_digest=hashlib.sha256()
        for key,value in sorted(model.state_dict().items()):
            initial_digest.update(key.encode()); initial_digest.update(tensor_bytes(value))
        identity['initial_state_sha256']=initial_digest.hexdigest()
        write(out/'identity.json',identity)
        sampler=PoolSampler(nu,ni,seen,plan['negative_pool_size'],pool_seed)
        lows=load_lows((manifest,nu,ni,seen,edges))
        low_sampler=LowSampler(nu,ni,lows,seed+3000003)
        optimizer=torch.optim.Adam(model.parameters(),lr=trial['learning_rate'])
        popularity=Counter(i for _,i in edges)
        best,best_epoch,stale,steps,curve=-1.,0,0,0,[]
        for epoch in range(1,trial['max_epochs']+1):
            edge_rng.shuffle(edges); model.train()
            totals=Counter(); stream=hashlib.sha256()
            for offset in range(0,len(edges),trial['batch_size']):
                batch=edges[offset:offset+trial['batch_size']]
                users=torch.tensor([u for u,_ in batch]); positive=torch.tensor([i for _,i in batch])
                pools=sampler.sample(users)
                coins=torch.rand(len(batch),generator=mask_generator)
                low=low_sampler.sample(users)
                for value in (users,positive,pools,coins,low):stream.update(tensor_bytes(value))
                optimizer.zero_grad()
                loss,diagnostic=explicit_loss(model,trial['model'],users,positive,pools[:,0],low,trial['low_weight'],trial['regularization'])
                if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                loss.backward(); optimizer.step(); steps+=1
                totals['loss']+=loss.item()*len(batch)
                totals['low_pairs']+=diagnostic['low_pairs']
                for key in ('data_loss','margin','gradient','easy_fraction','low_loss'):totals[key]+=diagnostic[key]*len(batch)
            metrics,_,_=evaluate(model,selection,popularity)
            row={'epoch':epoch,'train_loss':totals['loss']/len(edges),
                 'low_pair_fraction':totals['low_pairs']/len(edges),
                 **{key:totals[key]/len(edges) for key in ('data_loss','margin','gradient','easy_fraction','low_loss')},
                 'selection_ndcg':metrics['ndcg'],'optimizer_steps':steps,
                 'paired_stream_sha256':stream.hexdigest(),'elapsed_seconds':time.perf_counter()-started}
            curve.append(row)
            with (out/'curve.jsonl').open('a',encoding='utf-8',newline='\n') as f:f.write(json.dumps(row,sort_keys=True)+'\n')
            print(json.dumps({'trial':name,'seed':seed,**row}),flush=True)
            if metrics['ndcg']>best:
                best,best_epoch,stale=metrics['ndcg'],epoch,0
                torch.save({'state_dict':model.state_dict(),'trial':trial,'identity':identity,
                            'dimensions':[nu,ni],'best_epoch':epoch},out/'best.pt')
            else:
                stale+=1
                if stale>=trial['patience']:break
        payload=torch.load(out/'best.pt',map_location='cpu',weights_only=True)
        reloaded=build_model(trial,contract); reloaded.load_state_dict(payload['state_dict'],strict=True); reloaded.eval()
        metrics,rows,diversity=evaluate(reloaded,selection,popularity)
        if metrics['ndcg']!=best:raise ValueError('Reloaded NDCG mismatch')
        write(out/'selection_predictions.json',rows)
        result={'status':'VERIFIED','trial':trial,'identity':identity,'best_epoch':best_epoch,
                'actual_epochs':len(curve),'optimizer_steps':steps,'curve':curve,
                'selection_metrics':metrics,'diversity':diversity,'checkpoint_reload_exact':True,
                'checkpoint_sha256':digest(out/'best.pt'),'predictions_sha256':digest(out/'selection_predictions.json'),
                'elapsed_seconds':time.perf_counter()-started,'parameter_count':sum(p.numel() for p in model.parameters()),
                'test_evaluation':'NOT EVALUATED','audit_evaluation':'NOT EVALUATED','remote_api_requests':0}
        write(out/'result.json',result)
    return result


def freeze(plan_path):
    plan,groups=plan_and_groups(plan_path); candidates={}
    for trial in plan['trials']:
        directory=BASE/'seed_42'/trial['name']
        result=json.loads((directory/'result.json').read_text(encoding='utf-8'))
        if result['identity']['code_sha256']!=code_identity() or result['identity']['plan_sha256']!=digest(plan_path) or result['trial']!=trial or result['status']!='VERIFIED':
            raise ValueError('Selection identity mismatch')
        candidates[trial['name']]={'ndcg':result['selection_metrics']['ndcg'],'result_sha256':digest(directory/'result.json')}
    selected={kind:max(sorted(t['name'] for t in plan['trials'] if t['model']==kind),
                       key=lambda name:candidates[name]['ndcg']) for kind in ('lightgcn','bpr')}
    result={'selected':selected,'all_trials':candidates,'plan_sha256':digest(plan_path),
            'code_sha256':code_identity(),'selection_users':len(groups['fit'])+len(groups['select']),
            'audit_users':len(groups['audit']),'audit_used_for_selection':False}
    write(BASE/'selection_frozen.json',result); print(json.dumps(result),flush=True); return result


def audit(plan_path):
    plan,groups=plan_and_groups(plan_path)
    frozen=json.loads((BASE/'selection_frozen.json').read_text(encoding='utf-8'))
    if frozen['code_sha256']!=code_identity() or frozen['plan_sha256']!=digest(plan_path):raise ValueError('Frozen selection changed')
    out=BASE/'audit'; out.mkdir(exist_ok=False); torch.set_num_threads(plan['torch_threads'])
    with validation_only_files():
        contract=load_train_contract(); cohort=load_frozen_cohort('valid',ROOT/'artifacts/data_manifest.json')
        popularity=Counter(i for _,i in contract[4]); runs={}; raw={}
        cohorts={'selection':groups['fit']+groups['select'],'audit':groups['audit'],'all_valid':cohort['user_ids']}
        for trial in plan['trials']:
            for seed in plan['seeds']:
                name=trial['name']; model,original=load_verified(BASE/f'seed_{seed}'/name,trial,contract,digest(plan_path),seed)
                run={'metrics':{},'diversity':{},'actual_epochs':original['actual_epochs'],'best_epoch':original['best_epoch'],
                     'training_elapsed_seconds':original['elapsed_seconds'],'checkpoint_sha256':original['checkpoint_sha256'],
                     'source_result_sha256':digest(BASE/f'seed_{seed}'/name/'result.json')}
                for group,users in cohorts.items():
                    metrics,rows,diversity=evaluate(model,subset_cohort(cohort,users),popularity)
                    if group=='selection' and metrics!=original['selection_metrics']:raise ValueError('Selection metrics changed')
                    write(out/f'{name}_{seed}_{group}_predictions.json',rows)
                    raw[name,seed,group]=rows; run['metrics'][group]=metrics; run['diversity'][group]=diversity
                    run[group+'_predictions_sha256']=digest(out/f'{name}_{seed}_{group}_predictions.json')
                runs[f'{name}_{seed}']=run
        aggregate,paired,bands={},{},{}
        comparisons=[('gcn_low25','gcn_uniform'),('bpr_low25','bpr_uniform'),
                     (frozen['selected']['lightgcn'],frozen['selected']['bpr'])]
        for group in cohorts:
            averaged={t['name']:average_rows([raw[t['name'],s,group] for s in plan['seeds']]) for t in plan['trials']}
            aggregate[group]={t['name']:{field:{'mean':statistics.mean(runs[f"{t['name']}_{s}"]['metrics'][group][field] for s in plan['seeds']),
                'std':statistics.stdev(runs[f"{t['name']}_{s}"]['metrics'][group][field] for s in plan['seeds'])}
                for field in ('recall','hit','mrr','ndcg')} for t in plan['trials']}
            paired[group]={a+'_minus_'+b:clustered_pair(averaged[a],averaged[b],'ndcg') for a,b in comparisons}
            bands[group]={name:strata(rows,contract[3]) for name,rows in averaged.items()}
        # Assert common random numbers on every epoch both methods actually ran.
        paired_streams_verified={}
        for kind in ('gcn','bpr'):
            for seed in plan['seeds']:
                a=json.loads((BASE/f'seed_{seed}'/(kind+'_uniform')/'result.json').read_text(encoding='utf-8'))
                b=json.loads((BASE/f'seed_{seed}'/(kind+'_low25')/'result.json').read_text(encoding='utf-8'))
                valid=a['identity']['initial_state_sha256']==b['identity']['initial_state_sha256'] and all(
                    x['paired_stream_sha256']==y['paired_stream_sha256'] for x,y in zip(a['curve'],b['curve']))
                if not valid:raise ValueError('Paired initialization or samples drifted')
                paired_streams_verified[f'{kind}_{seed}']={'verified':True,'common_epochs':min(a['actual_epochs'],b['actual_epochs'])}
        summary={'status':'VERIFIED','scope':'Exploratory development explicit-low supervision',
                 'plan_sha256':digest(plan_path),'code_sha256':code_identity(),'selected':frozen['selected'],
                 'selection_frozen_sha256':digest(BASE/'selection_frozen.json'),'seeds':plan['seeds'],'runs':runs,
                 'aggregate':aggregate,'paired_ndcg':paired,'activity_strata':bands,
                 'paired_randomness':paired_streams_verified,'test_evaluation':'NOT EVALUATED',
                 'independent_quality_evaluation':'NOT EVALUATED','remote_api_requests':0,
                 'audit_limit':'Previously exposed validation; not independent confirmation',
                 'multiple_comparisons':'Exploratory unadjusted intervals; no quality claims against Microsoft paper'}
        write(out/'summary.json',summary)
    print(json.dumps({'status':'VERIFIED','aggregate':aggregate['audit'],'paired':paired['audit']}),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['train','freeze','audit']); parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--trial'); parser.add_argument('--seed',type=int,default=42); args=parser.parse_args()
    if args.action=='train':train(args.plan,args.trial,args.seed)
    elif args.action=='freeze':freeze(args.plan)
    else:audit(args.plan)


if __name__=='__main__':main()
