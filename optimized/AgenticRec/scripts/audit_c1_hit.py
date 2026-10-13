"""Read-only independent C1 preference audit. No runners or network imports."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

from audit_three_way_comparison import assess, digest


def audit(directory, output):
    root = Path(__file__).resolve().parents[3]
    protocol = json.loads((directory/'protocol.json').read_text(encoding='utf-8'))
    for group,base in [('files',directory),('sources',root),('resources',root)]:
        for name,expected in protocol[group].items():
            assert digest(base/name)==expected, name
    execution = json.loads((directory/'live_run/execution.json').read_text(encoding='utf-8'))
    assert execution['status']=='COMPLETE'
    labels = {r['episode_id']:r for r in json.loads((directory/'formal_private.json').read_text(encoding='utf-8'))}
    rows = [json.loads(l) for l in (directory/'live_run/episodes.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows)==488 and len({(r['system'],r['episode_id']) for r in rows})==488
    truth = json.loads((root/'optimized/artifacts/comparison/three_way_20261013/truth_private.json').read_text(encoding='utf-8'))
    counts = defaultdict(Counter);errors = defaultdict(Counter);scenarios = defaultdict(lambda:defaultdict(Counter))
    provider_by_turn = defaultdict(list)
    events = [json.loads(l) for l in (directory/'live_run/provider.jsonl').read_text(encoding='utf-8').splitlines()]
    started = [r for r in events if r['status']=='STARTED']
    completed = [r for r in events if r['status']=='COMPLETED']
    assert len(started)==len(completed)==execution['remote_requests']
    assert {r['request_id'] for r in started}=={r['request_id'] for r in completed}
    assert len({r['request_id'] for r in completed})==len(completed)
    assert not any(r['status']=='FAILED' for r in events)
    for r in completed:
        assert r['http_status']==200 and r['usage']
        request = r['request']
        assert request['model']=='deepseek-flash' and request['max_tokens']==1024 and request['temperature']==0
        assert request['thinking']=={'type':'disabled'} and not request.get('stream',False)
        provider_by_turn[(r['system'],r['episode_id'],r['turn'])].append(r)
    feasible = refusal = 0
    for row in rows:
        private = labels[row['episode_id']]
        assert len(row['turns'])==len(private['labels']) and row['category']==private['category']
        previous = [];task = []
        for n,(turn,label) in enumerate(zip(row['turns'],private['labels']),1):
            actual = provider_by_turn[(row['system'],row['episode_id'],n)]
            assert turn['requests']==len(actual)
            for field,provider_field in [('input_tokens','prompt_tokens'),('output_tokens','completion_tokens')]:
                assert turn[field]==sum(r['usage'][provider_field] for r in actual)
                counts[row['system']][field]+=turn[field]
            counts[row['system']]['requests']+=turn['requests']
            if label['feasible']:
                value,problems = assess(turn,label,truth,previous)
                for metric,expected in value.items():
                    assert math.isclose(float(turn['scores'][metric]),float(expected),abs_tol=1e-12), metric
                errors[row['system']].update(problems);task.append(value['strict_success']);feasible+=1
                if row['category']=='session_quality':
                    counts[row['system']]['quality_users']+=1
                    for metric in ('hit','recall','ndcg','mrr'):counts[row['system']][metric]+=value[metric]
            else:
                assert turn['scores']['safe_non_delivery']==(not turn['raw_ids'])
                assert not turn['scores']['correct_refusal'] or not turn['raw_ids']
                counts[row['system']]['refusal_passed']+=turn['scores']['correct_refusal'];refusal+=1
            previous = turn['raw_ids']
        if private['category']!='unanswerable':
            assert row['strict_success']==all(task)
            counts[row['system']]['main_tasks']+=1;counts[row['system']]['main_passed']+=all(task)
            bucket = scenarios[row['system']][private['scenario']]
            bucket['n']+=1;bucket['passed']+=all(task)
    assert feasible==548 and refusal==20
    formal_users = {p['evaluator_raw_user_id'] for p in labels.values() if 'evaluator_raw_user_id' in p}
    development = json.loads((directory/'development_private.json').read_text(encoding='utf-8'))
    dev_users = {p['evaluator_raw_user_id'] for p in development}
    assert len(formal_users)==210 and len(dev_users)==150 and not formal_users&dev_users
    final = json.loads((directory/'final_report.json').read_text(encoding='utf-8'))
    for s in ('C1','candidate'):
        assert counts[s]['quality_users']==180 and counts[s]['main_tasks']==234
        for metric in ('hit','recall','ndcg','mrr'):
            assert math.isclose(counts[s][metric]/180,final['live_quality'][s][metric],abs_tol=1e-12)
        assert counts[s]['main_passed']==final['live_systems'][s]['strict_passed']
    result = {'status':'VERIFIED','independent_feasible_turns':feasible,'refusal_turns':refusal,
              'distinct_paired_tasks':244,'quality_users_per_system':180,'http200_known_usage':len(completed),
              'counts':{k:dict(v) for k,v in counts.items()},'failure_counts':{k:dict(v) for k,v in errors.items()},
              'scenarios':{s:{k:dict(v) for k,v in b.items()} for s,b in scenarios.items()},
              'development_formal_users_disjoint':True,'frozen_hashes_verified':True,
              'protocol_sha256':digest(directory/'protocol.json'),
              'episodes_sha256':digest(directory/'live_run/episodes.jsonl'),
              'auditor_sha256':digest(Path(__file__)),'scoring_helper_sha256':digest(Path(__file__).with_name('audit_three_way_comparison.py')),
              'provider_calls_by_auditor':0}
    with output.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2,ensure_ascii=False);stream.write('\n')
    return result


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args();print(json.dumps(audit(args.directory,args.output),indent=2))
