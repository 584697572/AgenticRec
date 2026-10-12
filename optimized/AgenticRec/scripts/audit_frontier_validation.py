"""Independent predicate/quality/accounting audit; zero provider calls."""
from collections import Counter
import json
import math
from pathlib import Path

from agenticrec.evaluation.comparison_cases import ROOT,sha,write_new
from agenticrec.evaluation.comparison_scoring import AnswerIdentity


def independent_scores(turn,label,truth):
    ids=turn['raw_ids'];good=[]
    for raw in ids:
        item=truth.get(str(raw))
        good.append(bool(item and set(label['genres']).issubset(item['genres'])
                         and not set(label.get('exclude_genres',[])).intersection(item['genres'])
                         and (label['year_min'] is None or item['year']>=label['year_min'])
                         and (label['year_max'] is None or item['year']<=label['year_max'])
                         and raw not in label['excluded_raw_ids']))
    strict=(turn['status']=='OK' and turn['extraction_verified'] and len(ids)==label['k']
            and len(set(ids))==len(ids) and all(good))
    targets=set(label['relevant_raw_ids']);credited=set();dcg=0;hits=0
    for index,raw in enumerate(ids[:label['k']],1):
        if raw in targets and raw not in credited:
            credited.add(raw);hits+=1;dcg+=1/math.log2(index+1)
    idcg=sum(1/math.log2(index+1) for index in range(1,min(label['k'],len(targets))+1))
    delivered=turn['status']=='OK' and turn['extraction_verified']
    return {'strict_success':bool(strict),'constraint_precision':sum(good)/len(good) if good else 0.,
            'hit':float(bool(hits)) if delivered and targets else 0.,
            'recall':hits/len(targets) if delivered and targets else 0.,
            'ndcg':dcg/idcg if delivered and idcg else 0.}


def audit(directory):
    protocol=json.loads((directory/'protocol.json').read_text())
    for name,digest in protocol['sources_sha256'].items():assert sha(ROOT/name)==digest,name
    for name,digest in protocol['files_sha256'].items():assert sha(directory/name)==digest,name
    labels={v['episode_id']:v for v in json.loads((directory/'formal_private.json').read_text())}
    truth=json.loads((directory/'truth_private.json').read_text())
    catalog=json.loads((directory/'catalog.json').read_text())
    parser=AnswerIdentity(truth)
    run=directory/'formal_run'
    rows=[json.loads(s) for s in (run/'episodes.jsonl').read_text().splitlines()]
    assert len({(r['system'],r['episode_id']) for r in rows})==len(rows)
    turns=0;structured_answers=0
    for row in rows:
        previous=[];success=[]
        for turn,original_label in zip(row['turns'],labels[row['episode_id']]['labels']):
            label=dict(original_label)
            if label.get('exclude_previous'):label['excluded_raw_ids']=label['excluded_raw_ids']+previous
            computed=independent_scores(turn,label,truth)
            for key,value in computed.items():
                assert math.isclose(float(value),float(turn['scores'][key]),rel_tol=0,abs_tol=1e-12),(row['system'],row['episode_id'],key)
            if row['system']=='original':
                backed=[catalog['original_ids'].get(str(i)) for i in turn.get('mapped_catalog_ids',[])]
                assert parser.extract(turn['answer'],backed)['raw_ids']==turn['raw_ids']
            elif row['system']=='optimized':
                assert [catalog['optimized_ids'].get(str(i)) for i in turn.get('mapped_optimized_ids',[])]==turn['raw_ids']
            elif turn.get('mapped_optimized_ids'):
                ids=turn['mapped_optimized_ids']
                assert [catalog['items'][str(i)]['raw_id'] for i in ids]==turn['raw_ids']
                expected='\n'.join(f"{n}. {catalog['items'][str(i)]['title']} ({catalog['items'][str(i)]['year']})" for n,i in enumerate(ids,1))
                assert turn['answer']==expected
                structured_answers+=1
            previous=[i for i in turn['raw_ids'] if i is not None]
            success.append(computed['strict_success']);turns+=1
        assert all(success)==row['strict_success']
    latest={};counts=Counter()
    for text in (run/'provider.jsonl').read_text().splitlines():
        row=json.loads(text);counts[(row['request_id'],row['status'])]+=1;latest[row['request_id']]=row
        request=row['request']
        assert request['model']=='deepseek-flash' and request['max_tokens']==512
        assert request['temperature']==0 and request['thinking']=={'type':'disabled'}
    assert all(counts[(i,'STARTED')]==1 and row['status']!='STARTED' for i,row in latest.items())
    for row in rows:
        for number,turn in enumerate(row['turns'],1):
            matched=[v for v in latest.values() if (v['system'],v['episode_id'],v['turn'])==(row['system'],row['episode_id'],number)]
            assert len(matched)==turn['requests']
            assert all(v['status']=='COMPLETED' and v['usage'] for v in matched)
            assert sum(v['usage']['prompt_tokens'] for v in matched)==turn['input_tokens']
            assert sum(v['usage']['completion_tokens'] for v in matched)==turn['output_tokens']
    assert sum(t['requests'] for r in rows for t in r['turns'])==len(latest)
    # No test targets or raw user IDs may enter planner input; catalog IDs are
    # legitimate public context, unlike target/evaluator-labelled fields.
    forbidden=('relevant_raw_ids','evaluator_raw_user_id','eval_test.parquet','formal_private.json')
    assert not any(token in json.dumps(v['request']) for v in latest.values() for token in forbidden)
    balance=[json.loads(s) for s in (run/'balance.jsonl').read_text().splitlines()]
    summary={'status':'VERIFIED','episodes':len(rows),'turns':turns,
             'structured_delivery_identity_checked':structured_answers,'generation_requests':len(latest),
             'unresolved_request_intents':0,'all_predicates_and_quality_recomputed':True,
             'requests_and_tokens_reconciled':True,'planner_target_field_scan':'PASS',
             'source_and_input_hashes':'PASS','protocol_sha256':sha(directory/'protocol.json'),
             'episodes_sha256':sha(run/'episodes.jsonl'),'provider_sha256':sha(run/'provider.jsonl'),
             'auditor_sha256':sha(Path(__file__)),'first_balance':balance[0]['balance'],'last_balance':balance[-1]['balance'],
             'invoice_cost_cny':None}
    write_new(directory/'independent_audit.json',summary)
    public={k:v for k,v in summary.items() if k not in ('first_balance','last_balance')}
    from decimal import Decimal
    initial=next(r for r in summary['first_balance']['balance_infos'] if r['currency']=='CNY')
    final=next(r for r in summary['last_balance']['balance_infos'] if r['currency']=='CNY')
    public['observed_balance_decrease_cny']=str(Decimal(initial['total_balance'])-Decimal(final['total_balance']))
    write_new(ROOT/'optimized/reports/frontier_independent_audit_20261012.json',public)
    return summary


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(audit(args.directory.resolve()),indent=2))
