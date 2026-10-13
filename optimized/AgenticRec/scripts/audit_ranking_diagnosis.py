"""Independent frozen-ranking, raw-history and evaluator diagnostic audit."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import pyarrow.parquet as pq

from audit_three_way_comparison import assess, digest


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def describe(values):
    values = sorted(values)
    def p(q):
        x = q*(len(values)-1); n = int(x)
        return values[n]+(values[min(n+1,len(values)-1)]-values[n])*(x-n)
    return dict(n=len(values),min=values[0],median=p(.5),p90=p(.9),max=values[-1],mean=sum(values)/len(values))


def bucket(rank):
    if rank is None: return 'no_feasible_target'
    return next(name for lo,hi,name in [(1,10,'1-10'),(11,20,'11-20'),(21,50,'21-50'),
                                       (51,100,'51-100'),(101,500,'101-500'),(501,10**9,'501+')] if lo<=rank<=hi)


def audit(directory,output):
    root = Path(__file__).resolve().parents[3]
    old = root/'optimized/artifacts/comparison/sequence_ml1m_20261013_run2'
    source = root/'optimized/artifacts/comparison/c1_hit_20261013'
    previous = root/'optimized/artifacts/comparison/three_way_20261013'
    data = root/'optimized/data/processed/ml-1m-v1'
    contract,report = read(directory/'contract.json'),read(directory/'report.json')
    for name,sha in contract['files'].items(): assert digest(root/name)==sha,name
    assert digest(directory/'rankings.json')==report['rankings_sha256']
    assert digest(directory/'contract.json')==report['contract_sha256']
    rankings,scores = read(directory/'rankings.json'),read(directory/'scores.json')
    assert len(rankings)==len(scores)==1800
    ranks = {(r['system'],r['episode_id']):r for r in rankings}; assert len(ranks)==1800
    assert (directory/'rankings.json').stat().st_mtime_ns <= (directory/'scores.json').stat().st_mtime_ns
    catalog = {int(k):v for k,v in read(previous/'catalog.json')['items'].items()}
    rawmap = {v['raw_id']:i for i,v in catalog.items()}
    public = {r['episode_id']:r for r in read(old/'public.json')}
    labels = {r['episode_id']:r for r in read(source/'formal_private.json')}
    truth = read(previous/'truth_private.json')
    rawhist,seen,future = defaultdict(list),defaultdict(set),defaultdict(list)
    for row in pq.read_table(data/'train.parquet').to_pylist():
        seen[row['raw_user_id']].add(row['raw_item_id'])
        if row['rating']>=4 and row['raw_item_id'] in rawmap: rawhist[row['raw_user_id']].append((row['timestamp'],row['raw_item_id']))
    for row in pq.read_table(data/'eval_test.parquet').to_pylist():
        if row['is_target'] and row['segment']=='WARM' and row['raw_item_id'] in rawmap and row['raw_item_id'] not in seen[row['raw_user_id']]:
            future[row['raw_user_id']].append(row['timestamp'])
    population = Counter(pq.read_table(data/'train_positive.parquet',columns=['item_id']).column(0).to_pylist())
    poprank = {i:n+1 for n,i in enumerate(sorted(catalog,key=lambda i:(-population[i],i)))}
    unions = read(directory/'mixed_unions.json')
    totals = defaultdict(list)
    for row in scores:
        s,eid = row['system'],row['episode_id']; ids = ranks[s,eid]['item_ids']
        liked = {r['item_id'] for r in public[eid]['catalog_context']}
        assert len(ids)==len(set(ids)) and set(ids)==set(catalog)-liked
        assert row['top10']==ids[:10]
        lab = labels[eid]; u = lab['evaluator_raw_user_id']; targets = set(lab['labels'][0]['relevant_raw_ids'])
        assert row['history_n']==len(liked)
        raws = [catalog[i]['raw_id'] for i in ids]
        expected,errors = assess({'raw_ids':raws[:10],'status':'OK','extraction_verified':True},lab['labels'][0],truth,[])
        assert not any(errors.values())
        for m in ('hit','recall','ndcg','mrr'): assert math.isclose(expected[m],row['quality'][m],abs_tol=1e-12)
        positions = [n for n,raw in enumerate(raws,1) if raw in targets]
        assert row['first_target_rank']==(min(positions) if positions else None)
        for k in (10,20,50,100,500): assert row['hits_at_k'][str(k)]==int(bool(targets&set(raws[:k])))
        popids = ranks['popular',eid]['item_ids']
        for k in (10,100): assert row['popular_overlap'+str(k)]==len(set(ids[:k])&set(popids[:k]))
        assert row['hidden_seen_count10']==len(set(raws[:10])&seen[u])
        oracle = [r for r in raws if r not in seen[u]][:10]
        turn,errors = assess({'raw_ids':oracle,'status':'OK','extraction_verified':True},lab['labels'][0],truth,[])
        assert not any(errors.values())
        for m in ('hit','recall','ndcg','mrr'): assert math.isclose(turn[m],row['oracle_seen_filter_quality'][m],abs_tol=1e-12)
        assert row['mixed_union_hit']==bool(targets&{catalog[i]['raw_id'] for i in unions[eid]})
        assert row['target_count']==len(targets) and row['full_train_seen_count']==len(seen[u])
        timestamps = [ts for ts,_ in sorted(rawhist[u])[-len(liked):]]
        assert row['public_unique_timestamps']==len(set(timestamps))
        assert row['public_all_timestamp_equal']==int(len(set(timestamps))==1)
        assert row['public_input_time_span_days']==(max(timestamps)-min(timestamps))/86400
        assert row['test_gap_days']==(min(future[u])-max(ts for ts,_ in rawhist[u]))/86400
        assert row['target_popularity_median']==describe([poprank[rawmap[i]] for i in targets])['median']
        totals[s].append(row)
    oldpred = {(r['system'],r['episode_id']):r for r in read(old/'predictions.json')}
    matched = 0
    for s,rows in totals.items():
        saved = report['systems'][s]; assert len(rows)==180
        assert saved['hits10']==sum(r['quality']['hit'] for r in rows)
        assert saved['first_target_rank_buckets']==dict(Counter(bucket(r['first_target_rank']) for r in rows))
        assert saved['catalog_top10_distinct_items']==len({i for r in rows for i in r['top10']})
        assert saved['hidden_train_seen_top10_mean']==sum(r['hidden_seen_count10'] for r in rows)/180
        assert saved['oracle_all_train_seen_filter_hits10']==sum(r['oracle_seen_filter_quality']['hit'] for r in rows)
        for k in (10,20,50,100,500): assert saved['hits_at_k'][str(k)]==sum(r['hits_at_k'][str(k)] for r in rows)
        for m in ('hit','recall','ndcg','mrr'): assert math.isclose(saved['quality'][m],sum(r['quality'][m] for r in rows)/180,abs_tol=1e-12)
        for r in rows:
            oldname = 'candidate' if s=='mixed' else s
            if (oldname,r['episode_id']) in oldpred:
                assert r['top10']==oldpred[oldname,r['episode_id']]['item_ids']; matched+=1
    assert matched==report['historical_top10_replayed']==1260
    main = {r['episode_id']:r for r in totals['sequence42']}
    misses = [i for i,r in main.items() if not r['quality']['hit']]
    assert len(misses)==report['main_misses']
    assert report['main_misses_target_in_11_to_100']==sum(main[i]['hits_at_k']['100'] for i in misses)
    assert report['main_misses_no_target_top100']==sum(not main[i]['hits_at_k']['100'] for i in misses)
    for field,stats in report['input_stats'].items(): assert stats==describe([r[field] for r in main.values()])
    by = {s:{r['episode_id']:r for r in rows} for s,rows in totals.items()}
    for k in (10,100):
        assert report['oracle_four_method_union_hit_users'][str(k)]==sum(any(by[s][i]['hits_at_k'][str(k)] for s in
            ('sequence42','mixed','itemcf','popular')) for i in main)
    for s,probe in report['mechanism_probes'].items():
        assert probe['gains_vs_main']==sum(by[s][i]['quality']['hit']>main[i]['quality']['hit'] for i in main)
        assert probe['losses_vs_main']==sum(by[s][i]['quality']['hit']<main[i]['quality']['hit'] for i in main)
        assert probe['identical_top10_lists']==sum(by[s][i]['top10']==main[i]['top10'] for i in main)
    # Recompute training chronology without importing the diagnostic runner.
    groups = defaultdict(list)
    for r in pq.read_table(data/'train_positive.parquet').to_pylist(): groups[r['user_id']].append((r['timestamp'],r['item_id']))
    expected_training = []
    for u,seq in sorted(groups.items()):
        seq = sorted(seq); cut = int(len(seq)*.8)
        if cut<10 or cut>=len(seq): continue
        for h in (3,5,10):
            expected_training.append({'history_n':h,'target_count':len(seq)-cut,'earlier_positive_competing_count':cut-h,
                'input_time_span_days':(seq[cut-1][0]-seq[cut-h][0])/86400,
                'input_unique_timestamps':len({ts for ts,_ in seq[cut-h:cut]}),
                'tied_input_target_boundary':int(seq[cut-1][0]==seq[cut][0]),
                'target_same_input_last_timestamp_fraction':sum(ts==seq[cut-1][0] for ts,_ in seq[cut:])/(len(seq)-cut),
                'target_span_days':(seq[-1][0]-seq[cut][0])/86400})
    assert expected_training==read(directory/'training_stats_private.json')
    for field,stats in report['training'].items(): assert stats==describe([r[field] for r in expected_training])
    result = {'status':'VERIFIED','independent_scores':1800,'full_catalog_permutations_checked':1800,
              'historical_top10_matched':matched,'training_samples_rebuilt':len(expected_training),
              'hidden_history_oracles_evaluator_only':True,'source_hashes_verified':True,'api_requests':0,
              'report_sha256':digest(directory/'report.json'),'auditor_sha256':digest(Path(__file__))}
    with output.open('x',encoding='utf-8') as f: json.dump(result,f,indent=2); f.write('\n')
    return result


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.directory.resolve(),args.output.resolve()),indent=2))
