"""Rebuild and verify the completed comparison without any provider calls."""
import argparse
from collections import Counter
import json
from pathlib import Path

from agenticrec.evaluation.comparison_cases import ROOT, sha, write_new
from agenticrec.evaluation.comparison_run import summarize, verify
from agenticrec.evaluation.comparison_scoring import AnswerIdentity, score_turn


def rebuild(directory):
    protocol=verify(directory)
    run=directory/'formal_run'
    public=json.loads((directory/'formal_public.json').read_text())
    labels={r['episode_id']:r for r in json.loads((directory/'formal_private.json').read_text())}
    identity=json.loads((directory/'identity.json').read_text())
    records=[json.loads(line) for line in (run/'episodes.jsonl').read_text().splitlines()]
    events=[json.loads(line) for line in (run/'provider.jsonl').read_text().splitlines()]
    latest={}
    intents=[]
    for event in events:
        latest[event['request_id']]=event
        if event['status']=='STARTED':
            intents.append(event)
    assert len(intents)==len(latest)
    assert [r['request_id'] for r in intents]==list(range(1,len(intents)+1))
    assert all(r['status'] in ('COMPLETED','FAILED') for r in latest.values())
    calls=list(latest.values())
    assert len(records)==len(public)*2
    assert Counter((r['episode_id'],r['system']) for r in records)==Counter(
        (r['episode_id'],system) for r in public for system in ('original','optimized'))
    index=AnswerIdentity(identity['truth'])
    observed_calls=Counter()
    for record in records:
        expected=labels[record['episode_id']]
        assert record['category']==expected['category']
        assert len(record['turns'])==len(expected['labels'])
        for number,(turn,label) in enumerate(zip(record['turns'],expected['labels']),start=1):
            if record['system']=='original':
                parsed=index.extract(turn['answer'],turn['map_ids'])
                assert parsed['raw_ids']==turn['raw_ids']
            assert score_turn(turn,label,identity['truth'])==turn['scores']
            turn_calls=[r for r in calls if (r['episode_id'],r['system'],r['turn'])==
                        (record['episode_id'],record['system'],number)]
            observed_calls[(record['episode_id'],record['system'],number)]=len(turn_calls)
            assert turn['requests']==len(turn_calls)
            assert all(r['request']['model']==protocol['llm_model'] and
                       r['request']['max_tokens']==512 and r['request']['temperature']==0 and
                       r['request']['thinking']=={'type':'disabled'} for r in turn_calls)
            if all(r['usage'] for r in turn_calls):
                assert turn['input_tokens']==sum(r['usage']['prompt_tokens'] for r in turn_calls)
                assert turn['output_tokens']==sum(r['usage']['completion_tokens'] for r in turn_calls)
        assert record['strict_success']==all(t['scores']['strict_success'] for t in record['turns'])
    assert sum(observed_calls.values())==len(calls)
    expected=summarize(records,public,calls,'formal')
    stored=json.loads((run/'summary.json').read_text())
    assert stored['status']=='COMPLETED' and stored['blocker'] is None
    assert all(stored[k]==v for k,v in expected.items())
    assert all(sha(directory/'source_snapshot'/name)==value for name,value in protocol['sources_sha256'].items())
    balance=[json.loads(line) for line in (run/'balance.jsonl').read_text().splitlines()]
    def cny(row):
        from decimal import Decimal
        return sum((Decimal(r['total_balance']) for r in row['balance']['balance_infos'] if r['currency']=='CNY'),Decimal(0))
    return stored, {
        'status':'VERIFIED', 'new_api_requests':0, 'formal_cases':len(public),
        'episode_executions':len(records),'recomputed_turns':sum(len(r['turns']) for r in records),
        'reconciled_provider_requests':len(calls), 'unresolved_provider_intents':0,
        'metrics_rebuild_exact':True, 'frozen_source_snapshots_match':True,
        'protocol_sha256':sha(directory/'protocol.json'),
        'raw_episode_sha256':sha(run/'episodes.jsonl'),
        'raw_provider_sha256':sha(run/'provider.jsonl'),
        'report_script_sha256':sha(Path(__file__)),
        'observed_balance_decrease_cny':str(cny(balance[0])-cny(balance[-1])),
        'actual_invoice_cost_cny':None,
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    directory=args.directory.resolve()
    output=args.output.resolve()
    if not directory.is_relative_to((ROOT/'optimized/artifacts/comparison').resolve()):
        raise ValueError('Private input directory outside comparison artifacts')
    if not output.is_relative_to((ROOT/'optimized/reports').resolve()):
        raise ValueError('Public aggregate report outside optimized/reports')
    summary,evidence=rebuild(directory)
    write_new(output,{'summary':summary,'verification':evidence})
    print(json.dumps(evidence,indent=2))


if __name__=='__main__':
    main()
