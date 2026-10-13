"""Compare identical input-to-first-target horizon definitions using raw files."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import pyarrow.parquet as pq

from audit_three_way_comparison import digest


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def describe(values):
    values = sorted(values)
    def p(q):
        pos = (len(values)-1)*q; n = int(pos)
        return values[n]+(values[min(n+1,len(values)-1)]-values[n])*(pos-n)
    return {'n':len(values),'median':p(.5),'p90':p(.9),'mean':sum(values)/len(values),'min':values[0],'max':values[-1]}


def audit(directory,output):
    root = Path(__file__).resolve().parents[3]
    data = root/'optimized/data/processed/ml-1m-v1'
    contract = read(directory/'contract.json')
    for name,sha in contract['files'].items(): assert digest(root/name)==sha,name
    mapping = read(data/'id_map.json')
    users = {int(k):int(v) for k,v in mapping['user_to_model'].items()}
    items = {int(k):int(v) for k,v in mapping['item_to_model'].items()}
    graph,rawgraph = defaultdict(list),defaultdict(list)
    for r in pq.read_table(data/'train_positive.parquet').to_pylist(): graph[r['user_id']].append((r['timestamp'],r['item_id']))
    for r in pq.read_table(data/'train.parquet').to_pylist():
        if r['rating']>=4 and r['raw_user_id'] in users and r['raw_item_id'] in items:
            rawgraph[users[r['raw_user_id']]].append((r['timestamp'],items[r['raw_item_id']]))
    assert {u:sorted(v) for u,v in graph.items()}=={u:sorted(v) for u,v in rawgraph.items()}
    firstgap,lastgap = [],[]
    for u,seq in sorted(graph.items()):
        seq = sorted(seq); cut = int(len(seq)*.8)
        if cut<10 or cut>=len(seq): continue
        firstgap.append((seq[cut][0]-seq[cut-1][0])/86400)
        lastgap.append((seq[-1][0]-seq[cut-1][0])/86400)
    scores = read(directory/'scores.json')
    main = [r for r in scores if r['system']=='sequence42']
    result = {'status':'VERIFIED','scope':'descriptive_temporal_mismatch','training_distinct_users':len(firstgap),
              'training_last_input_to_first_target_days':describe(firstgap),
              'training_last_input_to_last_target_days':describe(lastgap),
              'training_first_target_within_one_hour_users':sum(v<=1/24 for v in firstgap),
              'training_first_target_same_timestamp_users':sum(v==0 for v in firstgap),
              'training_last_target_within_one_hour_users':sum(v<=1/24 for v in lastgap),
              'test_last_input_to_first_target_days':describe([r['test_gap_days'] for r in main]),
              'test_first_target_within_one_hour_users':sum(r['test_gap_days']<=1/24 for r in main),
              'api_requests':0,'auditor_sha256':digest(Path(__file__)),
              'raw_and_processed_train_sequences_matched':True,'source_hashes_verified':True}
    with output.open('x',encoding='utf-8') as f: json.dump(result,f,indent=2); f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    audit(args.directory.resolve(),args.output.resolve())
