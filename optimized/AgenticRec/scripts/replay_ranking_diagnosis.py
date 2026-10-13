"""Rebuild all diagnostic ranking orders from fixed weights and train sources."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import torch

from audit_three_way_comparison import digest
from agenticrec.evaluation.rank_upgrade_models import ShortSequence
from agenticrec.retrieval.sources import CandidateRetriever


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def replay(directory,output):
    root = Path(__file__).resolve().parents[3]
    old = root/'optimized/artifacts/comparison/sequence_ml1m_20261013_run2'
    oldcontract = read(old/'contract.json')
    protocol = read(directory/'contract.json')
    for name,sha in protocol['files'].items(): assert digest(root/name)==sha,name
    records = read(directory/'rankings.json')
    stored = {(r['system'],r['episode_id']):r['item_ids'] for r in records}
    catalog = {int(k):v for k,v in read(root/'optimized/artifacts/comparison/three_way_20261013/catalog.json')['items'].items()}
    public = read(old/'public.json')
    inputs = [[r['item_id'] for r in p['catalog_context']] for p in public]
    groups = defaultdict(list)
    for n,v in enumerate(inputs): groups[len(v)].append(n)
    rotated = [None]*180
    for group in groups.values():
        for j,n in enumerate(group): rotated[n]=inputs[group[(j+1)%len(group)]]
    def tensor(histories):
        x = torch.zeros((180,10),dtype=torch.long)
        for n,ids in enumerate(histories): x[n,:len(ids)]=torch.tensor(ids)
        return x
    torch.set_num_threads(2)
    base = CandidateRetriever.from_frozen()
    pop = torch.tensor([base.popularity[i] for i in range(1,oldcontract['model_items']+1)],dtype=torch.float64)
    logpop = torch.log1p(pop); logpop = (logpop-logpop.mean())/logpop.std(unbiased=False).clamp_min(1e-6)
    model_count = 0
    for seed in (7,42,2026):
        model = ShortSequence(oldcontract['model_items'])
        model.load_state_dict(torch.load(old/'models'/f'seed_{seed}.pt',map_location='cpu')['state_dict']); model.eval()
        variants = [('sequence'+str(seed),inputs,True)]
        if seed==42:
            variants += [('sequence42_pure',inputs,False),('sequence42_reverse',[list(reversed(v)) for v in inputs],True),
                         ('sequence42_rotated',rotated,True)]
        for name,histories,mix in variants:
            with torch.no_grad(): values = model(tensor(histories)).to(torch.float64)
            assert torch.isfinite(values).all()
            if mix:
                values = .5*(values-values.mean(1,keepdim=True))/values.std(1,unbiased=False,keepdim=True).clamp_min(1e-6)+.5*logpop
            for n,p in enumerate(public):
                v = values[n].tolist()
                eligible = set(catalog)-set(inputs[n])
                expected = sorted(eligible,key=lambda i:(-v[i-1],i))
                assert expected==stored[name,p['episode_id']],(name,p['episode_id'])
                model_count+=1
    co = {}
    unions = read(directory/'mixed_unions.json')
    baseline_count = 0
    for n,p in enumerate(public):
        liked = inputs[n]; eligible = sorted(set(catalog)-set(liked))
        cf,adaptive = defaultdict(float),defaultdict(float)
        for seed in liked:
            if seed not in co:
                counts = Counter()
                for u in sorted(base.item_users[seed]): counts.update(sorted(base.user_items[u]))
                co[seed] = counts
            for i,count in co[seed].items():
                denom = math.sqrt(base.popularity[seed]*base.popularity[i])
                cf[i] += count/denom/len(liked)
                adaptive[i] += count/(denom*len(liked))
        cf_order = sorted(eligible,key=lambda i:(-cf[i],i))
        assert cf_order==stored['itemcf',p['episode_id']]
        pop_order = sorted(eligible,key=lambda i:(-base.popularity[i],i))
        assert pop_order==stored['popular',p['episode_id']]
        query = defaultdict(float)
        for seed in liked:
            for t,w in base.content_vectors[seed].items(): query[t] += w/len(liked)
        content_scores = {i:sum(query.get(t,0)*w for t,w in base.content_vectors[i].items()) for i in eligible}
        content_order = sorted(eligible,key=lambda i:(-content_scores[i],i))
        source_cf = sorted((i for i in adaptive if i in set(eligible)),key=lambda i:(-adaptive[i],i))[:100]
        routes = dict(cf=source_cf,content=content_order[:100],popularity=pop_order[:100])
        assert sorted(set().union(*map(set,routes.values())))==unions[p['episode_id']]
        for name,weights in [('C1',dict(cf=1.,content=1.,popularity=1.)),('mixed',dict(cf=1.,content=.5,popularity=.25))]:
            scores = defaultdict(float)
            for s,ids in routes.items():
                for rank,item in enumerate(ids,1): scores[item] += weights[s]/(60+rank)
            expected = sorted(eligible,key=lambda i:(-scores[i],-base.popularity[i],i))
            assert expected==stored[name,p['episode_id']]
        baseline_count += 4
    assert (model_count,baseline_count)==(1080,720)
    result = {'status':'VERIFIED','frozen_model_full_rankings_replayed':model_count,
              'independently_rebuilt_baseline_full_rankings':baseline_count,'mixed_candidate_unions_rebuilt':180,
              'api_requests':0,'auditor_sha256':digest(Path(__file__)),'rankings_sha256':digest(directory/'rankings.json')}
    with output.open('x',encoding='utf-8') as f: json.dump(result,f,indent=2); f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    replay(args.directory.resolve(),args.output.resolve())
