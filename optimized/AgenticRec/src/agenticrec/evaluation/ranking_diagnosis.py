"""Finite post-hoc failure diagnostics; frozen models, no fitting or API calls."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import pyarrow.parquet as pq
import torch

from .simple_baselines import ROOT, digest, load, write_new, public_state, SimpleRanker, quality
from .sequence_ml1m import verify, SEEDS
from .rank_upgrade_models import ShortSequence, padded_histories, mixed_scores
from .preference_ranker import baseline_sources, fuse
from ..retrieval.sources import CandidateRetriever

STUDY = ROOT/'optimized/artifacts/comparison/sequence_ml1m_20261013_run2'
SOURCE = ROOT/'optimized/artifacts/comparison/c1_hit_20261013'
DATA = ROOT/'optimized/data/processed/ml-1m-v1'
CATALOG = ROOT/'optimized/artifacts/comparison/three_way_20261013/catalog.json'
PROTOCOL = ROOT/'optimized/reports/ranking_diagnosis_protocol_20261013.md'
SYSTEMS = ('sequence42','sequence7','sequence2026','mixed','C1','itemcf','popular',
           'sequence42_pure','sequence42_reverse','sequence42_rotated')
KS = (10,20,50,100,500)


def describe(values):
    if not values:
        return {'n':0}
    values = sorted(values)
    def pct(q):
        p = (len(values)-1)*q; lo = int(p)
        return values[lo]+(values[min(lo+1,len(values)-1)]-values[lo])*(p-lo)
    return {'n':len(values),'min':values[0],'median':pct(.5),'p90':pct(.9),'max':values[-1],
            'mean':sum(values)/len(values)}


def rank_bucket(position):
    if position is None: return 'no_feasible_target'
    for lo,hi in ((1,10),(11,20),(21,50),(51,100),(101,500)):
        if position <= hi: return f'{lo}-{hi}'
    return '501+'


def rotate_inputs(public):
    groups = defaultdict(list)
    for n,p in enumerate(public): groups[len(p['catalog_context'])].append(n)
    result = [None]*len(public)
    for group in groups.values():
        if len(group)<2: raise ValueError('Cannot rotate singleton group')
        for n,other in zip(group,group[1:]+group[:1]):
            result[n] = [r['item_id'] for r in public[other]['catalog_context']]
    return result


def summarize(records, public, catalog):
    by = {s:{r['episode_id']:r for r in records if r['system']==s} for s in SYSTEMS}
    keys = sorted(by['sequence42'])
    if len(records)!=len(SYSTEMS)*180 or any(sorted(v)!=keys for v in by.values()):
        raise ValueError('Incomplete diagnostic study')
    summary = {}
    for s,rows in by.items():
        summary[s] = {'n':180,'hits10':sum(r['quality']['hit'] for r in rows.values()),
            'quality':{m:sum(r['quality'][m] for r in rows.values())/180 for m in ('hit','recall','ndcg','mrr')},
            'hits_at_k':{str(k):sum(r['hits_at_k'][str(k)] for r in rows.values()) for k in KS},
            'first_target_rank_buckets':dict(Counter(rank_bucket(r['first_target_rank']) for r in rows.values())),
            'catalog_top10_distinct_items':len({i for r in rows.values() for i in r['top10']}),
            'popular_top10_overlap_mean':sum(r['popular_overlap10'] for r in rows.values())/180,
            'popular_top100_overlap_mean':sum(r['popular_overlap100'] for r in rows.values())/180,
            'hidden_train_seen_top10_mean':sum(r['hidden_seen_count10'] for r in rows.values())/180,
            'hidden_train_seen_top10_slot_fraction':sum(r['hidden_seen_count10'] for r in rows.values())/1800,
            'oracle_all_train_seen_filter_hits10':sum(r['oracle_seen_filter_quality']['hit'] for r in rows.values()),
            'target_popularity_rank':describe([r['target_popularity_median'] for r in rows.values()]),
            'history':{str(h):{'n':60,'hits10':sum(r['quality']['hit'] for r in rows.values() if r['history_n']==h),
                             'hits100':sum(r['hits_at_k']['100'] for r in rows.values() if r['history_n']==h)}
                       for h in (3,5,10)}}
    main = by['sequence42']; misses = [i for i in keys if not main[i]['quality']['hit']]
    alternatives = {}
    for s in ('mixed','itemcf','popular','sequence42_pure','sequence42_reverse','sequence42_rotated'):
        alternatives[s] = {'gains_vs_main':sum(by[s][i]['quality']['hit']>main[i]['quality']['hit'] for i in keys),
            'losses_vs_main':sum(by[s][i]['quality']['hit']<main[i]['quality']['hit'] for i in keys),
            'main_top10_overlap_mean':sum(len(set(by[s][i]['top10'])&set(main[i]['top10'])) for i in keys)/180,
            'identical_top10_lists':sum(by[s][i]['top10']==main[i]['top10'] for i in keys)}
    union = {str(k):sum(any(by[s][i]['hits_at_k'][str(k)] for s in ('sequence42','mixed','itemcf','popular')) for i in keys)
             for k in (10,100)}
    return {'systems':summary,'main_misses':len(misses),
            'main_misses_target_in_11_to_100':sum(main[i]['first_target_rank'] is not None and main[i]['first_target_rank']<=100 for i in misses),
            'main_misses_no_target_top100':sum(not main[i]['hits_at_k']['100'] for i in misses),
            'mixed_recall_union100_hit_users':sum(main[i]['mixed_union_hit'] for i in keys),
            'main_misses_recoverable_in_mixed_union':sum(main[i]['mixed_union_hit'] for i in misses),
            'oracle_four_method_union_hit_users':union,'mechanism_probes':alternatives,
            'input_stats':{k:describe([main[i][k] for i in keys]) for k in
                           ('target_count','full_train_seen_count','public_input_time_span_days','test_gap_days',
                            'public_unique_timestamps','public_all_timestamp_equal')},
            'main_hit_target_count':describe([main[i]['target_count'] for i in keys if main[i]['quality']['hit']]),
            'main_miss_target_count':describe([main[i]['target_count'] for i in misses])}


def run(directory):
    if directory.exists(): raise FileExistsError('Fresh evidence directory required')
    contract = verify(STUDY)
    audit = load(STUDY/'independent_audit.json')
    if audit['status']!='VERIFIED' or audit['report_sha256']!=digest(STUDY/'report.json'):
        raise ValueError('Unverified original study')
    public = load(STUDY/'public.json')
    catalog = {int(k):v for k,v in load(CATALOG)['items'].items()}
    paths = [PROTOCOL,Path(__file__),ROOT/'optimized/AgenticRec/scripts/audit_ranking_diagnosis.py',CATALOG,
             STUDY/'contract.json',STUDY/'training.json',STUDY/'predictions.json',STUDY/'scores.json',
             STUDY/'report.json',STUDY/'independent_audit.json',STUDY/'public.json',
             SOURCE/'formal_private.json',DATA/'train.parquet',DATA/'train_positive.parquet',DATA/'eval_test.parquet']
    paths += [STUDY/'models'/f'seed_{s}.pt' for s in SEEDS]
    paths += list((ROOT/'optimized/AgenticRec/src/agenticrec').rglob('*.py'))
    directory.mkdir(parents=True)
    write_new(directory/'contract.json',{'status':'FROZEN','scope':'post_hoc_failure_diagnosis','api_requests':0,
        'systems':list(SYSTEMS),'files':{p.relative_to(ROOT).as_posix():digest(p) for p in paths}})
    torch.set_num_threads(2)
    base = CandidateRetriever.from_frozen()
    edges = [(u,i) for u,items in base.user_items.items() for i in sorted(items)]
    simple = SimpleRanker(catalog,edges,base.content_vectors)
    likes = [[r['item_id'] for r in p['catalog_context']] for p in public]
    x = padded_histories([{'liked_item_ids':v} for v in likes])
    pop = torch.tensor([base.popularity[i] for i in range(1,contract['model_items']+1)],dtype=torch.float64)
    rotations = rotate_inputs(public)
    modelscores = {}
    for seed in SEEDS:
        model = ShortSequence(contract['model_items'])
        model.load_state_dict(torch.load(STUDY/'models'/f'seed_{seed}.pt',map_location='cpu')['state_dict']); model.eval()
        with torch.no_grad(): raw = model(x)
        if not torch.isfinite(raw).all(): raise ValueError('Nonfinite model scores')
        modelscores['sequence'+str(seed)] = torch.stack([mixed_scores(v,pop,.5) for v in raw])
        if seed==42:
            modelscores['sequence42_pure'] = raw.to(torch.float64)
            for label,hist in [('sequence42_reverse',[list(reversed(v)) for v in likes]),
                               ('sequence42_rotated',rotations)]:
                with torch.no_grad(): perturbed = model(padded_histories([{'liked_item_ids':v} for v in hist]))
                modelscores[label] = torch.stack([mixed_scores(v,pop,.5) for v in perturbed])
    rankings,unions = [], {}
    for n,pub in enumerate(public):
        state = public_state(pub); eligible,sources = baseline_sources(base,catalog,state)
        unions[pub['episode_id']] = sorted(set().union(*map(set,sources.values())))
        for s in SYSTEMS:
            if s in modelscores:
                values = modelscores[s][n]
                ids = sorted(eligible,key=lambda i:(-float(values[i-1]),i))
            elif s in ('mixed','C1'):
                weights = {'cf':1.,'content':.5,'popularity':.25} if s=='mixed' else dict.fromkeys(('cf','content','popularity'),1.)
                ids = fuse(base,eligible,sources,weights,k=len(eligible))
            else:
                values = simple.scores(eligible,state,s)
                ids = sorted(eligible,key=lambda i:(-values[i],i))
            rankings.append({'system':s,'episode_id':pub['episode_id'],'history_n':len(likes[n]),'item_ids':ids})
    # Persist without labels; compact JSON prevents verbose full-universe output.
    with (directory/'rankings.json').open('x',encoding='utf-8') as f: json.dump(rankings,f,separators=(',',':'))
    write_new(directory/'mixed_unions.json',unions)
    old = {(r['system'],r['episode_id']):r for r in load(STUDY/'predictions.json')}
    replayed = 0
    for r in rankings:
        name = 'candidate' if r['system']=='mixed' else r['system']
        if (name,r['episode_id']) in old:
            if r['item_ids'][:10] != old[name,r['episode_id']]['item_ids']: raise ValueError('Historical ranking differs')
            replayed += 1
    if replayed!=1260: raise ValueError('Missing old rank replays')
    # Hidden user history and future labels enter only evaluator diagnostics.
    labels = {r['episode_id']:r for r in load(SOURCE/'formal_private.json')}
    hist, seen = defaultdict(list),defaultdict(set)
    rawmap = {v['raw_id']:i for i,v in catalog.items()}
    for r in pq.read_table(DATA/'train.parquet').to_pylist():
        seen[r['raw_user_id']].add(r['raw_item_id'])
        if r['rating']>=4 and r['raw_item_id'] in rawmap:
            hist[r['raw_user_id']].append((r['timestamp'],r['raw_item_id']))
    future = defaultdict(list)
    for r in pq.read_table(DATA/'eval_test.parquet').to_pylist():
        if r['is_target'] and r['segment']=='WARM' and r['raw_item_id'] in rawmap and r['raw_item_id'] not in seen[r['raw_user_id']]:
            future[r['raw_user_id']].append(r['timestamp'])
    poporder = sorted(catalog,key=lambda i:(-base.popularity[i],i)); poprank = {i:n+1 for n,i in enumerate(poporder)}
    poprows = {r['episode_id']:r for r in rankings if r['system']=='popular'}
    results = []
    for r in rankings:
        label = labels[r['episode_id']]; u = label['evaluator_raw_user_id']
        targets = set(label['labels'][0]['relevant_raw_ids']); ids = r['item_ids']
        raws = [catalog[i]['raw_id'] for i in ids]; positions = [n+1 for n,raw in enumerate(raws) if raw in targets]
        visible = set(label['labels'][0]['excluded_raw_ids'])
        hidden_count = sum(raw in seen[u] and raw not in visible for raw in raws[:10])
        oracle = [raw for raw in raws if raw not in seen[u]][:10]
        seq = sorted(hist[u]); h = r['history_n']; timestamps = [ts for ts,_ in seq[-h:]]
        target_pops = [poprank[rawmap[i]] for i in targets]
        results.append({'system':r['system'],'episode_id':r['episode_id'],'history_n':h,'top10':ids[:10],
            'quality':quality(raws[:10],targets),'first_target_rank':min(positions) if positions else None,
            'hits_at_k':{str(k):int(any(p<=k for p in positions)) for k in KS},
            'popular_overlap10':len(set(ids[:10])&set(poprows[r['episode_id']]['item_ids'][:10])),
            'popular_overlap100':len(set(ids[:100])&set(poprows[r['episode_id']]['item_ids'][:100])),
            'hidden_seen_count10':hidden_count,'oracle_seen_filter_quality':quality(oracle,targets),
            'target_popularity_median':describe(target_pops)['median'],
            'mixed_union_hit':bool({catalog[i]['raw_id'] for i in unions[r['episode_id']]}&targets),
            'target_count':len(targets),'full_train_seen_count':len(seen[u]),
            'public_input_time_span_days':(max(timestamps)-min(timestamps))/86400,
            'public_unique_timestamps':len(set(timestamps)), 'public_all_timestamp_equal':int(len(set(timestamps))==1),
            'test_gap_days':(min(future[u])-max(ts for ts,_ in seq))/86400})
    write_new(directory/'scores.json',results)
    # Training diagnostics are descriptive, not model selection.
    trainrows = pq.read_table(DATA/'train_positive.parquet').to_pylist(); groups = defaultdict(list)
    for r in trainrows: groups[r['user_id']].append((r['timestamp'],r['item_id']))
    trainstats = []
    for u,seq in sorted(groups.items()):
        seq = sorted(seq); cut = int(len(seq)*.8)
        if cut<10 or cut>=len(seq): continue
        for h in (3,5,10):
            trainstats.append({'history_n':h,'target_count':len(seq)-cut,'earlier_positive_competing_count':cut-h,
                              'input_time_span_days':(seq[cut-1][0]-seq[cut-h][0])/86400,
                              'input_unique_timestamps':len({ts for ts,_ in seq[cut-h:cut]}),
                              'tied_input_target_boundary':int(seq[cut-1][0]==seq[cut][0]),
                              'target_same_input_last_timestamp_fraction':sum(ts==seq[cut-1][0] for ts,_ in seq[cut:])/(len(seq)-cut),
                              'target_span_days':(seq[-1][0]-seq[cut][0])/86400})
    write_new(directory/'training_stats_private.json',trainstats)
    report = {'status':'EVALUATED','scope':'post_hoc_failure_diagnosis','api_requests':0,'new_training':False,
        'downloads':0,'default_changed':False,'historical_top10_replayed':replayed,
        'rankings_sha256':digest(directory/'rankings.json'),'contract_sha256':digest(directory/'contract.json'),
        'training':{k:describe([r[k] for r in trainstats]) for k in
                    ('target_count','earlier_positive_competing_count','input_time_span_days','target_span_days',
                     'input_unique_timestamps','tied_input_target_boundary','target_same_input_last_timestamp_fraction')},
        **summarize(results,public,catalog)}
    write_new(directory/'report.json',report)
    return report


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.directory.resolve()),indent=2))
