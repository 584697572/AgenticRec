"""Fix evaluator coverage using independent source-backed catalog identities.

Preserve frozen predictions, labels and initial scores. No inference or APIs.
"""
from collections import Counter
import copy
import importlib.util
import json
from pathlib import Path

import pyarrow.feather as feather
import pyarrow.parquet as parquet

from agenticrec.evaluation.comparison_cases import ROOT, sha, write_new
from agenticrec.evaluation.comparison_run import summarize
from agenticrec.evaluation.comparison_scoring import AnswerIdentity, score_turn
from report_original_comparison import rebuild


def identity_maps(original,optimized,index,split_title_year,source_title_forms=lambda title:{title}):
    original_map,optimized_map={},{}
    statuses=Counter()
    def resolve(title,year):
        exact=index.resolve(title,year)
        if exact['status']=='PASS':
            return exact
        candidates=set()
        for form in source_title_forms(title):
            result=index.resolve(form,year)
            candidates.update(result['raw_ids'])
        if len(candidates)==1:
            return {'status':'PASS','raw_ids':list(candidates)}
        if len(candidates)>1:
            return {'status':'AMBIGUOUS','raw_ids':sorted(candidates)}
        return index.resolve(title,year)
    for row in original:
        result=resolve(row['title'],int(row['release_date']))
        if result['status']=='PASS':
            original_map[int(row['id'])]=result['raw_ids'][0]
        else:
            statuses['original_'+result['status']]+=1
    # Multiple catalog records for one source film do not establish a unique
    # crosswalk. Drop them all, rather than choose based on test outcomes.
    counts=Counter(original_map.values())
    original_map={key:raw for key,raw in original_map.items() if counts[raw]==1}
    for row in optimized:
        if row['item_id'] is None:
            continue
        title,year=split_title_year(row['title'])
        result=resolve(title,year)
        if result['status']=='PASS' and result['raw_ids'][0]==row['raw_item_id']:
            optimized_map[int(row['item_id'])]=result['raw_ids'][0]
        else:
            statuses['optimized_'+result['status']]+=1
    assert len(set(optimized_map.values()))==len(optimized_map)
    return original_map,optimized_map,dict(statuses)


def correct(directory,output):
    initial,evidence=rebuild(directory)
    spec=importlib.util.spec_from_file_location('source_identity',ROOT/'original/reproduction/scripts/catalog_identity.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    metadata=ROOT/'original/data/raw/metadata_audit/ml-10m/movies.dat'
    index=module.MetadataIndex(module.parse_movies_dat(metadata.read_bytes()))
    original_path=ROOT/'original/data/raw/upstream_movie/movies.ftr'
    optimized_path=ROOT/'optimized/data/processed/ml-1m-v1/items.parquet'
    original_map,optimized_map,statuses=identity_maps(
        feather.read_table(original_path).to_pylist(),parquet.read_table(optimized_path).to_pylist(),
        index,module.split_title_year,module.source_title_forms)
    run=directory/'formal_run'
    records=[json.loads(line) for line in (run/'episodes.jsonl').read_text().splitlines()]
    old=copy.deepcopy(records)
    identity=json.loads((directory/'identity.json').read_text())
    old_original={int(k):v for k,v in identity['original_ids'].items()}
    old_optimized={int(k):v for k,v in identity['optimized_ids'].items()}
    assert all(original_map[key]==raw for key,raw in old_original.items())
    assert all(optimized_map[key]==raw for key,raw in old_optimized.items())
    labels={r['episode_id']:r for r in json.loads((directory/'formal_private.json').read_text())}
    parser=AnswerIdentity(identity['truth'])
    changes=[]
    for record in records:
        prior=next(r for r in old if (r['system'],r['episode_id'])==(record['system'],record['episode_id']))
        for number,(turn,label,old_turn) in enumerate(zip(record['turns'],labels[record['episode_id']]['labels'],prior['turns']),start=1):
            if record['system']=='original':
                mapped=[original_map.get(i) for i in turn['mapped_catalog_ids']]
                turn.update(parser.extract(turn['answer'],mapped))
            else:
                turn['raw_ids']=[optimized_map.get(i) for i in turn['mapped_optimized_ids']]
            turn['scores']=score_turn(turn,label,identity['truth'])
            if turn['raw_ids']!=old_turn['raw_ids']:
                changes.append({'episode_id':record['episode_id'],'system':record['system'],'turn':number,
                                'before_ids':old_turn['raw_ids'],'after_ids':turn['raw_ids']})
            for name in ('answer','requests','input_tokens','output_tokens','tool_calls','end_to_end_seconds','status'):
                assert turn[name]==old_turn[name]
        record['strict_success']=all(t['scores']['strict_success'] for t in record['turns'])
    public=json.loads((directory/'formal_public.json').read_text())
    latest={}
    for line in (run/'provider.jsonl').read_text().splitlines():
        event=json.loads(line)
        latest[event['request_id']]=event
    corrected=summarize(records,public,list(latest.values()),'formal')
    corrected.update(status='COMPLETED',blocker=None,executed_systems=['original','optimized'])
    private=directory/'identity_correction'
    private.mkdir(exist_ok=False)
    write_new(private/'maps.json',{'original_ids':original_map,'optimized_ids':optimized_map,'unresolved_statuses':statuses})
    write_new(private/'scored_records.json',records)
    write_new(private/'changed_identity_rows.json',changes)
    evidence.update(identity_correction='source-backed evaluator correction after execution; not a new preregistered run',
                    unchanged_predictions_labels_usage_and_latency=True,
                    original_initial_map_items=len(old_original),original_corrected_map_items=len(original_map),
                    optimized_initial_map_items=len(old_optimized),optimized_corrected_map_items=len(optimized_map),
                    changed_turn_identity_rows=len(changes),
                    correction_script_sha256=sha(Path(__file__)),
                    corrected_private_records_sha256=sha(private/'scored_records.json'))
    write_new(output,{'summary':corrected,'initial_summary':initial,'verification':evidence})
    print(json.dumps({'summary':corrected,'verification':evidence},indent=2))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    directory=args.directory.resolve()
    output=args.output.resolve()
    if not directory.is_relative_to((ROOT/'optimized/artifacts/comparison').resolve()):
        raise ValueError('Private comparison directory outside workspace artifacts')
    if not output.is_relative_to((ROOT/'optimized/reports').resolve()):
        raise ValueError('Public aggregate outside reports')
    correct(directory,output)
