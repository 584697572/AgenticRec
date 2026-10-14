"""Supplementary formula replay without producer ranking/inference functions."""
import argparse
import json
import math
from pathlib import Path

import torch
from audit_three_way_comparison import digest


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def score(directory,config,seed,cache):
    if config['route']=='sequence': return cache['base']
    state=torch.load(directory/'residual_models'/f'{config["route"]}_{seed}_{config["epoch"]}.pt',map_location='cpu')['state_dict']
    x=(cache['x']-state['means'])/state['scales']
    with torch.no_grad():
        logits=torch.nn.functional.linear(x,state['net.0.weight'],state['net.0.bias'])
        if config['route']!='linear_boundary':
            logits=torch.nn.functional.linear(torch.relu(logits),state['net.3.weight'],state['net.3.bias'])
        correction=torch.tanh(logits.squeeze(-1))*config['strength']
    values=cache['base']+correction
    assert (values-cache['base']).abs().max()<=config['strength']+1e-7
    return values


def predict(directory,config,seed,cache):
    values=score(directory,config,seed,cache).tolist()
    return [[i for i,s in sorted(zip(pool,v),key=lambda pair:(-pair[1],pair[0]))[:10]] for pool,v in zip(cache['ids'],values)]


def audit(directory,output):
    torch.set_num_threads(2)
    contract=read(directory/'contract.json');frozen=read(directory/'frozen.json')
    root=Path(__file__).resolve().parents[3]
    for name,sha in contract['files'].items():assert digest(root/name)==sha,name
    for name,sha in frozen['files'].items():assert digest(directory/name)==sha,name
    selection=read(directory/'selection.json');dev=read(directory/'development_predictions.json')
    devby={(r['grid_index'],r['seed'],r['episode_id']):r for r in dev}
    devpub=read(directory/'development_public.json');public=read(directory/'formal_public.json')
    formalby={(r['system'],r['episode_id']):r for r in read(directory/'predictions.json')}
    count=0
    for seed in (7,42,2026):
        cache=torch.load(directory/'feature_cache'/f'development_{seed}.pt',map_location='cpu')
        for c in selection['candidates']:
            for p,ids in zip(devpub,predict(directory,c['config'],seed,cache)):
                assert ids==devby[c['grid_index'],seed,p['episode_id']]['item_ids'];count+=1
        cache=torch.load(directory/f'formal_cache_{seed}.pt',map_location='cpu')
        configs=[dict(route='sequence',epoch=0,strength=0.)]+list(frozen['routes'].values())
        for config in configs:
            for p,ids in zip(public,predict(directory,config,seed,cache)):
                assert ids==formalby[config['route']+str(seed),p['episode_id']]['item_ids'];count+=1
    # Scalar reference losses on every real usable training query, not a toy fixture.
    from agenticrec.evaluation.top100_models import boundary_loss, mass_loss
    groups=0
    for number in (0,1):
        for seed in (7,42,2026):
            cache=torch.load(directory/'feature_cache'/f'train_{number}_{seed}.pt',map_location='cpu')
            s=cache['base'][cache['usable']];p=cache['positive'][cache['usable']]
            boundary=boundary_loss(s,p).tolist();mass=mass_loss(s,p).tolist()
            for values,positive,b,m in zip(s.tolist(),p.tolist(),boundary,mass):
                good=[v for v,yes in zip(values,positive) if yes]
                other=sorted((v for v,yes in zip(values,positive) if not yes),reverse=True)
                z=(other[9]+.1-max(good))/.2
                assert math.isclose(b,.2*(max(z,0)+math.log1p(math.exp(-abs(z)))),abs_tol=1e-10)
                shift=max(values)
                expected=math.log(sum(math.exp(v-shift) for v in values))-math.log(sum(math.exp(v-shift) for v in good))
                assert math.isclose(m,expected,abs_tol=1e-10);groups+=1
    report=read(directory/'report.json')
    result=dict(status='VERIFIED',independent_ranking_formula_predictions_replayed=count,
                real_training_groups_loss_formula_verified=groups,producer_inference_helpers_used=False,
                bounded_residual_verified=True,api_requests=0,report_sha256=digest(directory/'report.json'),
                auditor_sha256=digest(Path(__file__)),predictions_sha256=report['predictions_sha256'])
    with output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();audit(args.directory.resolve(),args.output.resolve())
