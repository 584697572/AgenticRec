"""Freeze shared public requests separately from private identity/quality labels."""
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import random

import pyarrow.feather as feather
import pyarrow.parquet as parquet

ROOT = Path(__file__).resolve().parents[5]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=True, sort_keys=True, indent=2)
        stream.write('\n')


def load_assets():
    readiness = ROOT/'optimized/artifacts/comparison/readiness_20261012.json'
    identity = json.loads(readiness.read_text(encoding='utf-8'))
    for name, expected in identity['source_sha256'].items():
        from_root = {
            'original_catalog': 'original/data/raw/upstream_movie/movies.ftr',
            'crosswalk': 'original/data/metadata_private/catalog_identity_20261005/catalog_to_ml10m.jsonl',
            'crosswalk_manifest': 'original/data/metadata_private/catalog_identity_20261005/manifest.json',
            'optimized_catalog': 'optimized/data/processed/ml-1m-v1/items.parquet',
            'train_positive': 'optimized/data/processed/ml-1m-v1/train_positive.parquet',
            'split_manifest': 'optimized/data/processed/ml-1m-v1/split_manifest.json',
            'identity_source': 'original/reproduction/scripts/catalog_identity.py',
        }
        if sha(ROOT/from_root[name]) != expected:
            raise ValueError('Readiness input changed: '+name)
    spec = importlib.util.spec_from_file_location('comparison_catalog_identity',
        ROOT/'original/reproduction/scripts/catalog_identity.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    metadata = ROOT/'original/data/raw/metadata_audit/ml-10m/movies.dat'
    truth = {r['raw_id']: r for r in module.parse_movies_dat(metadata.read_bytes())}
    rows = feather.read_table(ROOT/'original/data/raw/upstream_movie/movies.ftr').to_pylist()
    rows = {int(r['id']): r for r in rows}
    crosswalk_path = ROOT/'original/data/metadata_private/catalog_identity_20261005/catalog_to_ml10m.jsonl'
    crosswalk = [json.loads(line) for line in crosswalk_path.read_text(encoding='utf-8').splitlines()]
    original_ids = {r['catalog_id']: r['raw_movie_id'] for r in crosswalk if r['status']=='PASS'}
    shared = {r['raw_movie_id']: r for r in identity['shared_metadata_rows']}
    optimized_ids = {r['optimized_item_id']: raw for raw,r in shared.items()
                     if r['optimized_item_id'] is not None}
    for raw,r in shared.items():
        r['title'] = rows[r['original_catalog_id']]['title']
        r['year'] = truth[raw]['year']
        r['genres'] = truth[raw]['genres']
    return truth, shared, original_ids, optimized_ids, module, metadata


def make_cases(shared, *, calibration=False):
    candidates = {raw:r for raw,r in shared.items() if r['in_optimized_training_candidates']}
    rng = random.Random(731 if calibration else 20261012)
    cases = []
    genres = ['Comedy','Drama','Action','Thriller','Romance']
    periods = [(1960,1989),(1990,1995),(1996,2000),(1970,2000)]
    for number in range(2 if calibration else 20):
        genre = genres[(number+2 if calibration else number)%len(genres)]
        low,high = (1982,1998) if calibration else periods[number//5]
        feasible = [raw for raw,r in candidates.items() if genre in r['genres'] and low<=r['year']<=high]
        assert len(feasible)>=10
        message = f'Recommend exactly 10 {genre.lower()} movies released from {low} through {high}, inclusive. Give only their movie titles.'
        cases.append({'category':'constraints','public':{'messages':[message], 'catalog_context':[]},
                      'labels':[{'k':10,'genres':[genre],'year_min':low,'year_max':high,
                                 'excluded_raw_ids':[], 'relevant_raw_ids':[]}]})

    data = ROOT/'optimized/data/processed/ml-1m-v1'
    train = parquet.read_table(data/'train.parquet').to_pylist()
    positive = defaultdict(list)
    train_seen = defaultdict(set)
    for row in train:
        user, raw = row['raw_user_id'],row['raw_item_id']
        train_seen[user].add(raw)
        if row['rating']>=4 and raw in candidates:
            positive[user].append((row['timestamp'],raw))
    targets = defaultdict(set)
    for row in parquet.read_table(data/'eval_test.parquet').to_pylist():
        if row['is_target'] and row['segment']=='WARM' and row['raw_item_id'] in candidates:
            if row['raw_item_id'] not in train_seen[row['raw_user_id']]:
                targets[row['raw_user_id']].add(row['raw_item_id'])
    users = sorted(u for u in targets if len(positive[u])>=5)
    rng.shuffle(users)
    # Calibration and formal users are disjoint; no system predictions are used.
    users = users[:2] if calibration else [u for u in users if u not in calibration_users(positive, targets)][:40]
    assert len(users)==(2 if calibration else 40)
    for number,user in enumerate(users):
        liked = [raw for _,raw in sorted(positive[user])[-5:]]
        assert len(liked)==len(set(liked))
        context = [{'title':candidates[raw]['title'], 'year':candidates[raw]['year'],
                    'item_id':candidates[raw]['optimized_item_id']} for raw in liked]
        names = '; '.join(f"{candidates[raw]['title']} ({candidates[raw]['year']})" for raw in liked)
        message = f'I liked these movies and have already watched them: {names}. Recommend exactly 10 other movies based on these preferences. Exclude the movies I listed. Give only movie titles.'
        label = {'k':10,'genres':[],'year_min':None,'year_max':None,
                 'excluded_raw_ids':liked,'relevant_raw_ids':sorted(targets[user])}
        case = {'category':'session_quality', 'public':{'messages':[message], 'catalog_context':context},
                'labels':[label], 'evaluator_raw_user_id':user}
        if (not calibration and number>=20) or (calibration and number==1):
            low,high = (1990,2000) if not calibration else (1985,1999)
            message2 = f'Keep my previous preferences and exclude those already watched movies. This time recommend exactly 10 drama movies released from {low} through {high}, inclusive. Give only movie titles.'
            case['category']='feedback'
            case['public']['messages'].append(message2)
            label2 = dict(label,genres=['Drama'],year_min=low,year_max=high,relevant_raw_ids=[])
            case['labels'].append(label2)
            assert sum('Drama' in r['genres'] and low<=r['year']<=high and raw not in liked
                       for raw,r in candidates.items())>=10
        cases.append(case)
    prefix = 'cal' if calibration else 'formal'
    for number,case in enumerate(cases):
        case['episode_id']=f'{prefix}-{number:03d}'
    return cases


def calibration_users(positive, targets):
    users = sorted(u for u in targets if len(positive[u])>=5)
    random.Random(731).shuffle(users)
    return set(users[:2])


def prepare(directory):
    directory = Path(directory).resolve()
    if not directory.is_relative_to((ROOT/'optimized/artifacts/comparison').resolve()):
        raise ValueError('Keep private evaluation inputs in ignored comparison artifacts')
    if directory.exists():
        raise FileExistsError('Refusing to overwrite a frozen evaluation')
    truth,shared,original_ids,optimized_ids,_,metadata = load_assets()
    calibration = make_cases(shared,calibration=True)
    formal = make_cases(shared)
    assert not ({r['evaluator_raw_user_id'] for r in calibration if 'evaluator_raw_user_id' in r}
                & {r['evaluator_raw_user_id'] for r in formal if 'evaluator_raw_user_id' in r})
    files = {}
    for phase,cases in [('calibration',calibration),('formal',formal)]:
        public = [{'episode_id':r['episode_id'],**r['public']} for r in cases]
        private = [{k:v for k,v in r.items() if k!='public'} for r in cases]
        for suffix,values in [('public',public),('private',private)]:
            path=directory/f'{phase}_{suffix}.json'
            write_new(path,values)
            files[path.name]=sha(path)
    write_new(directory/'identity.json',{'truth':truth,'original_ids':original_ids,'optimized_ids':optimized_ids})
    files['identity.json']=sha(directory/'identity.json')
    source = ROOT/'optimized/AgenticRec/src/agenticrec'
    sources={str(p.relative_to(ROOT)):sha(p) for p in sorted(source.rglob('*.py'))}
    worker=ROOT/'original/reproduction/scripts/comparison_app_worker.py'
    sources[str(worker.relative_to(ROOT))]=sha(worker)
    # Preserve bytes as well as hashes for uncommitted calibration sources.
    for name in sources:
        path=directory/'source_snapshot'/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes((ROOT/name).read_bytes())
    checkpoints={}
    for report in (ROOT/'optimized/reports/rec_baselines/frozen_model_validation_20261011.json',):
        value=json.loads(report.read_text(encoding='utf-8'))
        checkpoints['lightgcn_seed42']=value['runs']['lightgcn_42']['checkpoint_sha256']
    protocol={
        'schema_version':1, 'scope':'actual_two_versions_shared_text_tasks',
        'calibration_episodes':len(calibration),'formal_episodes':len(formal),
        'formal_turns_per_system':sum(len(r['labels']) for r in formal),
        'model_seed':42,'temperature':0,'max_output_tokens':512,'network_retries':0,
        'provider':'deepseek','llm_model':'deepseek-flash','thinking':'disabled',
        'spending_policy':'account_balance','authorization':'existing user authorization to run real LLM evaluation; conserve funds; stop on insufficient balance',
        'internal_ledger_reservation_note':'Uses existing historical conservative reservation solely to satisfy positive-cost ledger contract; total reservation is not a spending authorization or a current price estimate. Provider balance guard controls actual stopping.',
        'upstream_commit':'0959ecb05b0794748426e73e6efc1b6b35ec433d',
        'original':{'entry':'original app.py','demo_mode':'zero','reflection':False,
                    'shortening':False,'reply_style':'concise','resources':'original unchanged pretrained resources'},
        'optimized':{'model':'lightgcn','profile':'validated-20261011','fusion_mode':'rrf',
                     'agent':'O_router','max_replans':1,'public_title_id_context':True},
        'candidate_space':'each system retains its actual full catalog; requests feasible on shared training candidates; no oracle intersection filtering of outputs',
        'metadata_truth':'official MovieLens10M title/year/genres; differences from system-local metadata retained',
        'scoring':'strict final-answer ordered movie identities; ambiguous displayed titles may use a unique observed Map ID for disambiguation, never replace omitted final titles with tool IDs; unresolved identities stay in denominator',
        'quality_cohort':'20 session_quality episodes; warm historical test users, five explicit train-positive liked/watched movies; anonymous public identity for both systems',
        'quality_limits':['previously exposed test','old model training split unknown','pretrained row semantics unverified','anonymous session mode does not isolate LightGCN known-user gain'],
        'primary_metric':'strict_episode_success', 'secondary_metrics':['constraint_precision','Hit@10','Recall@10','NDCG@10','requests','tokens','tool_calls','latency_p50_p95'],
        'aggregation':'all 60 formal episodes; paired bootstrap by episode, 10000 resamples; quality only 20 session_quality users',
        'run_order':'alternate original-first/optimized-first by episode index; serial, persistent prewarmed models',
        'files_sha256':files,'sources_sha256':sources,'checkpoint_sha256':checkpoints,
        'metadata_sha256':sha(metadata),'train_sha256':sha(ROOT/'optimized/data/processed/ml-1m-v1/train.parquet'),
        'test_sha256':sha(ROOT/'optimized/data/processed/ml-1m-v1/eval_test.parquet'),
    }
    write_new(directory/'protocol.json',protocol)
    return protocol
