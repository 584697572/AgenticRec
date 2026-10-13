"""Append-only C1 preference study: prepare, diagnose, develop, freeze, live, report."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
import random
import subprocess
import time

import pyarrow.parquet as pq
import torch

from .comparison_cases import ROOT, write_new, sha
from .comparison_run import append, bootstrap, quantile
from .frontier_experiment import DATA, FrontierSystem
from .frontier_state import empty_state, feasible_ids, verify_delivery
from .three_way_comparison import prior_users, MatchedJournal, MODELS, score
from .preference_ranker import PreferenceRanker, PreferenceSystem, baseline_sources, fuse, constrained_ridge
from ..fusion_pipeline import FusionPipeline
from ..runtime.secrets import read_api_key

PREVIOUS = ROOT/'optimized/artifacts/comparison/three_way_20261013'
SYSTEMS = ('C1', 'candidate')
LAMBDAS = (10, 100, 1000)
MIXES = ((1,1,1), (1,0,0), (1,.25,0), (1,.25,.1), (1,.5,.1), (1,0,.1), (1,.5,.25), (1,1,.25))


def assets():
    catalog = {int(k):v for k,v in json.loads((PREVIOUS/'catalog.json').read_text())['items'].items()}
    base = FusionPipeline.from_frozen('lightgcn',42,mode='rrf',profile='validated-20261011').retriever
    return catalog, base


def state_for(public):
    state = empty_state()
    state['liked_item_ids'] = [r['item_id'] for r in public['catalog_context']]
    state['seen_item_ids'] = state['liked_item_ids'][:]
    return state


def metrics(ids, relevant):
    relevant = set(relevant)
    ranks = [n for n,i in enumerate(ids[:10],1) if i in relevant]
    import math
    ideal = sum(1/math.log2(n+1) for n in range(1,min(10,len(relevant))+1))
    return {'hit':float(bool(ranks)), 'recall':len(ranks)/max(1,len(relevant)),
            'ndcg':sum(1/math.log2(n+1) for n in ranks)/ideal if ideal else 0.,
            'mrr':1/ranks[0] if ranks else 0.}


def prepare(directory):
    if directory.exists() and any(p.name != 'diagnosis.json' for p in directory.iterdir()):
        raise FileExistsError('Earlier evidence cannot be overwritten')
    catalog, _ = assets()
    by_raw = {r['raw_id']:r for r in catalog.values()}
    positive, seen = defaultdict(list), defaultdict(set)
    for r in pq.read_table(DATA/'train.parquet').to_pylist():
        u, raw = r['raw_user_id'], r['raw_item_id']
        seen[u].add(raw)
        if r['rating'] >= 4 and raw in by_raw:
            positive[u].append((r['timestamp'],raw))
    targets = {}
    for phase in ('valid','test'):
        targets[phase] = defaultdict(set)
        for r in pq.read_table(DATA/f'eval_{phase}.parquet').to_pylist():
            u, raw = r['raw_user_id'], r['raw_item_id']
            if r['is_target'] and r['segment']=='WARM' and raw in by_raw and raw not in seen[u]:
                targets[phase][u].add(raw)
    excluded = prior_users()
    for phase in ('formal','calibration'):
        excluded.update(r['evaluator_raw_user_id'] for r in json.loads((PREVIOUS/f'{phase}_private.json').read_text()) if 'evaluator_raw_user_id' in r)
    valid_users = sorted(u for u in targets['valid'] if len(positive[u])>=10 and u not in excluded)
    test_users = sorted(u for u in targets['test'] if len(positive[u])>=10 and u not in excluded and u not in valid_users)
    rng = random.Random(2026101302)
    rng.shuffle(valid_users); rng.shuffle(test_users)
    if len(valid_users)<150 or len(test_users)<210:
        raise ValueError(f'Insufficient independent users: {len(valid_users)}, {len(test_users)}')

    def quality(u, h, phase, episode):
        liked = [i for _,i in sorted(positive[u])[-h:]]
        context = [{k:by_raw[i][k] for k in ('title','year','item_id')} for i in liked]
        titles = '; '.join(f"{by_raw[i]['title']} ({by_raw[i]['year']})" for i in liked)
        public = {'episode_id':episode, 'catalog_context':context,
                  'messages':[f'I have watched and liked these movies: {titles}. Please recommend exactly 10 other movies matching my taste. Exclude all the watched movies listed. Give only titles.']}
        label = {'k':10,'genres':[],'exclude_genres':[],'year_min':None,'year_max':None,
                 'excluded_raw_ids':liked,'relevant_raw_ids':sorted(targets[phase][u]),'exclude_previous':False,'feasible':True}
        private = {'episode_id':episode,'category':'session_quality','history_n':h,'scenario':f'history:{h}',
                   'evaluator_raw_user_id':u,'labels':[label]}
        return public, private

    dev = [quality(u,(3,5,10)[n%3],'valid',f'dev-hit-{n:03d}') for n,u in enumerate(valid_users[:150])]
    formal = [quality(u,(3,5,10)[n%3],'test',f'formal-hit-{n:03d}') for n,u in enumerate(test_users[:180])]
    old = json.loads((PREVIOUS/'formal_private.json').read_text())
    old_public = {r['episode_id']:r for r in json.loads((PREVIOUS/'formal_public.json').read_text())}
    # Same preregistered six feedback forms; replace all private users/history.
    templates = [next(c for c in old if c.get('scenario')==f'feedback:{m}') for m in range(6)]
    for n,u in enumerate(test_users[180:210]):
        mode = n%6
        public, private = quality(u,(3,5,10)[n%3],'test',f'formal-feedback-{n:03d}')
        template = templates[mode]
        labels = deepcopy(template['labels'])
        for label in labels:
            label['excluded_raw_ids'] = private['labels'][0]['excluded_raw_ids'][:]
            label['relevant_raw_ids'] = []
        first = public['messages'][0]
        if mode in (1,2,3,4):
            first += ' Focus on action movies released from 1985 through 2000, inclusive.'
        public['messages'] = [first] + old_public[template['episode_id']]['messages'][1:]
        private.update(category='feedback',scenario=f'feedback:{mode}',labels=labels)
        formal.append((public,private))
    for category, count in [('constraints',24), ('unanswerable',10)]:
        for n,c in enumerate([r for r in old if r['category']==category][:count]):
            episode = f'formal-{category}-{n:03d}'
            pub = deepcopy(old_public[c['episode_id']]);pub['episode_id'] = episode
            formal.append((pub,dict(episode_id=episode,category=category,history_n=None,scenario=c['scenario'],labels=c['labels'])))
    dev_ids = {p['evaluator_raw_user_id'] for _,p in dev}
    formal_ids = {p['evaluator_raw_user_id'] for _,p in formal if 'evaluator_raw_user_id' in p}
    assert len(dev_ids)==150 and len(formal_ids)==210 and not dev_ids&formal_ids and not (dev_ids|formal_ids)&excluded
    assert len(formal)==244 and sum(len(p['labels']) for _,p in formal)==284
    for phase, cases in [('development',dev), ('formal',formal)]:
        write_new(directory/f'{phase}_public.json',[p for p,_ in cases])
        write_new(directory/f'{phase}_private.json',[p for _,p in cases])
    write_new(directory/'prepared.json',{'previous_users_excluded':len(excluded),'valid_candidate_users':len(valid_users),
               'test_candidate_users':len(test_users),'development_users':150,'formal_quality_users':180,
               'formal_user_count':210,'formal_tasks':244,'formal_turns_per_system':284,
               'historical_data_exposure':True,'selection_uses':'eval_valid only',
               'data_sha256':{n:sha(DATA/n) for n in ('train.parquet','train_positive.parquet','eval_valid.parquet','eval_test.parquet','id_map.json')},
               'catalog_sha256':sha(PREVIOUS/'catalog.json')})
    return {'prepared':244,'development':150,'api_requests':0}


def diagnose(directory):
    catalog, base = assets()
    labels = {r['episode_id']:r for r in json.loads((PREVIOUS/'formal_private.json').read_text())}
    records = [json.loads(l) for l in (PREVIOUS/'formal_run/episodes.jsonl').read_text().splitlines()]
    counts = Counter();routes = defaultdict(Counter)
    for record in records:
        if record['system']!='C1' or record['category']!='session_quality':
            continue
        turn = record['turns'][0]
        eligible, sources = baseline_sources(base,catalog,turn['state'])
        predicted = fuse(base,eligible,sources,{'cf':1.,'content':1.,'popularity':1.})
        if predicted!=turn['mapped_optimized_ids']:
            raise ValueError('Refactored ranking differs from original C1')
        relevant = set(labels[record['episode_id']]['labels'][0]['relevant_raw_ids'])
        hit = lambda ids:bool({catalog[i]['raw_id'] for i in ids}&relevant)
        counts['users']+=1;counts['c1_hit10']+=hit(predicted);counts['feasible_oracle_hit']+=hit(eligible)
        union = set().union(*map(set,sources.values()))
        counts['union100_oracle_hit']+=hit(union)
        if not hit(predicted):
            counts['miss_has_relevant_in_union']+=hit(union)
            counts['miss_no_relevant_in_union']+=not hit(union)
        for name,ids in sources.items():
            routes[name]['hit10']+=hit(ids[:10]);routes[name]['oracle_hit100']+=hit(ids)
    if counts['users']!=180 or counts['c1_hit10']!=63:
        raise ValueError('Historical cohort no longer matches verified C1 evidence')
    result = {'counts':dict(counts),'routes':dict(routes),'baseline_exact_equivalence_users':180,
              'scope':'historical diagnostic only; not parameter selection','api_requests':0}
    write_new(directory/'diagnosis.json',result)
    return result


def training_statistics(n_items, masked=False):
    rows = pq.read_table(DATA/'train_positive.parquet').to_pylist()
    n_users = max(r['user_id'] for r in rows)
    if not masked:
        x = torch.zeros((n_users,n_items),dtype=torch.float64)
        x[[r['user_id']-1 for r in rows],[r['item_id']-1 for r in rows]]=1
        gram = x.T @ x
        return gram,gram,{'training_users':n_users,'training_pairs':int(x.sum()),'simulated_rows':0}
    sequences = defaultdict(list)
    for r in rows:
        sequences[r['user_id']].append((r['timestamp'],r['item_id']))
    pairs = []
    for u,seq in sorted(sequences.items()):
        seq = sorted(seq);cut = int(len(seq)*.8)
        if cut<10 or cut>=len(seq):
            continue
        targets = {i for _,i in seq[cut:]}
        for h in (3,5,10):
            liked = {i for _,i in seq[cut-h:cut]}
            if liked&targets:
                raise ValueError('Training time-mask input/target overlap')
            pairs.append((liked,targets))
    gram = torch.zeros((n_items,n_items),dtype=torch.float64)
    cross = torch.zeros_like(gram)
    for offset in range(0,len(pairs),128):
        batch = pairs[offset:offset+128]
        z = torch.zeros((len(batch),n_items),dtype=torch.float64);y = torch.zeros_like(z)
        for n,(liked,targets) in enumerate(batch):
            z[n,[i-1 for i in liked]]=1;y[n,[i-1 for i in targets]]=1
        gram += z.T @ z;cross += z.T @ y
    return gram,cross,{'training_users':len(pairs)//3,'training_pairs':len(rows),'simulated_rows':len(pairs),
                       'history_lengths':[3,5,10],'mask':'80% ordinal chronological prefix, remainder target, all inside train'}


def develop(directory):
    catalog, base = assets()
    public = json.loads((directory/'development_public.json').read_text())
    private = json.loads((directory/'development_private.json').read_text())
    states = [state_for(p) for p in public]
    snapshots = [baseline_sources(base,catalog,s) for s in states]
    configurations = [{'name':f'rrf-{n}','weights':dict(zip(('cf','content','popularity'),mix))} for n,mix in enumerate(MIXES)]
    checkpoints = {}
    model_dir = directory/'models';model_dir.mkdir(exist_ok=False)
    for model in ('ease','short_history_ridge'):
        clock = time.perf_counter()
        gram,cross,stats = training_statistics(base.catalog.n_items,model!='ease')
        stats_seconds = time.perf_counter()-clock
        for regularization in LAMBDAS:
            clock = time.perf_counter();b,residual = constrained_ridge(gram,cross,regularization)
            saved = {'weights':b,'item_ids':list(range(1,base.catalog.n_items+1)),
                     'model':model,'regularization':regularization,'statistics':stats,
                     'statistics_seconds':stats_seconds,'solve_seconds':time.perf_counter()-clock,
                     'off_diagonal_relative_residual':residual,
                     'train_sha256':sha(DATA/'train_positive.parquet'),'id_map_sha256':sha(DATA/'id_map.json')}
            path = model_dir/f'{model}_{regularization}.pt';torch.save(saved,path)
            checkpoints[(model,regularization)] = saved
            for cf_weight in (0,.25):
                configurations.append({'name':f'{model}-{regularization}-cf{cf_weight}',
                      'model':model,'regularization':regularization,'checkpoint':str(path.relative_to(directory)),
                      'weights':{'learned':1.,'cf':cf_weight,'content':0.,'popularity':0.}})
            print(json.dumps({'stage':'trained','model':model,'lambda':regularization,'solve_seconds':saved['solve_seconds']}),flush=True)
        del gram,cross
    summaries = [];predictions = []
    for config in configurations:
        checkpoint = checkpoints.get((config.get('model'),config.get('regularization')))
        ranker = PreferenceRanker(base,catalog,config,checkpoint)
        values = []
        for s,snapshot,p in zip(states,snapshots,private):
            ids = ranker.rank(s,snapshot)
            if verify_delivery(ids,catalog,s):
                raise ValueError('Development ranking violates unchanged hard gates')
            raw = [catalog[i]['raw_id'] for i in ids]
            value = metrics(raw,p['labels'][0]['relevant_raw_ids']);values.append(value)
            predictions.append({'configuration':config['name'],'episode_id':p['episode_id'],'raw_ids':raw,'metrics':value})
        summary = {'configuration':config,'n':len(values),**{m:sum(v[m] for v in values)/len(values) for m in values[0]}}
        summaries.append(summary)
        print(json.dumps({'stage':'development','configuration':config['name'],'hit':summary['hit'],'ndcg':summary['ndcg']}),flush=True)
    winner = max(range(len(summaries)),key=lambda i:(summaries[i]['hit'],summaries[i]['ndcg'],-i))
    write_new(directory/'development_predictions.json',predictions)
    write_new(directory/'selection.json',{'selected':summaries[winner]['configuration'],'summaries':summaries,
                         'selection_data':'eval_valid only; equal 50 users per public history length',
                         'test_predictions_created':False,'api_requests':0})
    return {'selected':summaries[winner],'baseline':summaries[0],'api_requests':0}


def freeze(directory):
    files = {str(p.relative_to(directory)):sha(p) for p in directory.rglob('*') if p.is_file()}
    sources = {str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'optimized/AgenticRec/src/agenticrec').rglob('*.py')}
    resources = {str(p.relative_to(ROOT)):sha(p) for p in MODELS.glob('*.pt')}
    resources.update({str(PREVIOUS.relative_to(ROOT)/n):sha(PREVIOUS/n) for n in ('catalog.json','truth_private.json')})
    facts = json.loads((directory/'prepared.json').read_text())
    resources.update({str((DATA/n).relative_to(ROOT)):v for n,v in facts['data_sha256'].items()})
    resources['optimized/reports/c1_hit_protocol_20261013.md'] = sha(ROOT/'optimized/reports/c1_hit_protocol_20261013.md')
    protocol = {'status':'FROZEN','files':files,'sources':sources,'resources':resources,
                'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                'systems':list(SYSTEMS),'model':'deepseek-flash','output_limit':1024,'retries':0,
                'max_requests':284*2*3,'selected':json.loads((directory/'selection.json').read_text())['selected']}
    write_new(directory/'protocol.json',protocol)
    return {'status':'FROZEN','files':len(files),'sources':len(sources),'max_requests':protocol['max_requests']}


def verify_frozen(directory):
    protocol = json.loads((directory/'protocol.json').read_text())
    for group,root in (('files',directory),('sources',ROOT),('resources',ROOT)):
        for name,expected in protocol[group].items():
            if sha(root/name)!=expected:
                raise ValueError('Frozen artifact changed: '+name)
    return protocol


def load_selected(directory, base, catalog, config):
    checkpoint = torch.load(directory/config['checkpoint'],map_location='cpu') if config.get('model') else None
    if checkpoint:
        if checkpoint['train_sha256']!=sha(DATA/'train_positive.parquet') or checkpoint['id_map_sha256']!=sha(DATA/'id_map.json'):
            raise ValueError('Checkpoint training contract changed')
    return PreferenceRanker(base,catalog,config,checkpoint)


def offline(directory):
    protocol = verify_frozen(directory)
    catalog,base = assets()
    selected = load_selected(directory,base,catalog,protocol['selected'])
    baseline = PreferenceRanker(base,catalog,{'weights':{'cf':1.,'content':1.,'popularity':1.}})
    public = json.loads((directory/'formal_public.json').read_text())
    private = json.loads((directory/'formal_private.json').read_text())
    records = []
    for pub,lab in zip(public,private):
        if lab['category']!='session_quality':continue
        s = state_for(pub);snapshot = baseline_sources(base,catalog,s)
        for name,ranker in [('C1',baseline),('candidate',selected)]:
            ids = ranker.rank(s,snapshot)
            if verify_delivery(ids,catalog,s):raise ValueError('Offline hard-gate violation')
            raw = [catalog[i]['raw_id'] for i in ids]
            records.append({'episode_id':pub['episode_id'],'system':name,'history_n':lab['history_n'],
                            'raw_ids':raw,'metrics':metrics(raw,lab['labels'][0]['relevant_raw_ids'])})
    write_new(directory/'offline_predictions.json',records)
    return quality_summary(records)


def quality_summary(records):
    by = {s:[r for r in records if r['system']==s] for s in SYSTEMS}
    if any(len(v)!=180 for v in by.values()):raise ValueError('Incomplete quality cohort')
    result = {}
    for s,rows in by.items():
        result[s] = {'n':180,**{m:sum(r['metrics'][m] for r in rows)/180 for m in ('hit','recall','ndcg','mrr')},
                     'history':{str(h):{'n':60,'hit':sum(r['metrics']['hit'] for r in rows if r['history_n']==h)/60} for h in (3,5,10)}}
    b = {r['episode_id']:r for r in by['C1']};c = {r['episode_id']:r for r in by['candidate']}
    if b.keys()!=c.keys():raise ValueError('Unpaired users')
    result['paired_candidate_minus_C1'] = {m:bootstrap([c[i]['metrics'][m]-b[i]['metrics'][m] for i in sorted(b)],random.Random(2026101303)) for m in ('hit','recall','ndcg','mrr')}
    return result


def live(directory):
    protocol = verify_frozen(directory)
    run_dir = directory/'live_run';run_dir.mkdir(exist_ok=False)
    catalog,base = assets()
    journal = MatchedJournal(read_api_key(),run_dir)
    systems = {'C1':FrontierSystem('C1',journal,catalog,base,MODELS),
               'candidate':PreferenceSystem(journal,catalog,base,MODELS,load_selected(directory,base,catalog,protocol['selected']))}
    public = json.loads((directory/'formal_public.json').read_text())
    private = json.loads((directory/'formal_private.json').read_text())
    truth = json.loads((PREVIOUS/'truth_private.json').read_text())
    failure = None;completed = 0
    try:
        journal.guard.refresh()
        if journal.guard.should_stop():raise RuntimeError(journal.guard.reason)
        for n,(pub,lab) in enumerate(zip(public,private)):
            order = SYSTEMS if n%2==0 else SYSTEMS[::-1]
            for name in order:
                turns = [];previous = []
                for t,label in enumerate(lab['labels']):
                    if journal.guard.should_stop():raise RuntimeError(journal.guard.reason)
                    if len(journal.records)>=protocol['max_requests']:raise RuntimeError('Finite workload request bound exhausted')
                    before = len(journal.records);clock = time.perf_counter()
                    journal.context = {'system':name,'episode_id':pub['episode_id'],'turn':t+1}
                    try:
                        result = systems[name].turn(pub,t)
                    except ValueError as error:
                        # Preserve parser errors as failed cases, never retry the episode.
                        result = {'status':'ERROR','answer':'','raw_ids':[],'extraction_verified':False,
                                  'tool_calls':0,'error_type':type(error).__name__,'error':str(error)}
                    used = journal.records[before:]
                    result.update(episode_id=pub['episode_id'],turn=t+1,
                                  elapsed_seconds=time.perf_counter()-clock,requests=len(used),
                                  input_tokens=sum(r['usage']['prompt_tokens'] for r in used) if all(r.get('usage') for r in used) else None,
                                  output_tokens=sum(r['usage']['completion_tokens'] for r in used) if all(r.get('usage') for r in used) else None)
                    result['scores'] = score(result,label,truth,previous)
                    previous = result['raw_ids'];turns.append(result)
                record = {'episode_id':pub['episode_id'],'system':name,'category':lab['category'],
                          'history_n':lab.get('history_n'),'turns':turns,'strict_success':all(t['scores']['strict_success'] for t in turns)}
                append(run_dir/'episodes.jsonl',record);completed+=1
            if (n+1)%10==0:
                print(json.dumps({'stage':'live','episodes':n+1,'of':len(public),'requests':len(journal.records)}),flush=True)
    except Exception as error:
        failure = type(error).__name__+': '+str(error).replace(journal._key,'[REDACTED]')
    finally:
        try:journal.guard.refresh()
        except Exception:pass
        write_new(run_dir/'execution.json',{'status':'COMPLETE' if completed==488 and failure is None else 'BLOCKED',
                    'completed_system_tasks':completed,'remote_requests':len(journal.records),'failure':failure,
                    'protocol_sha256':sha(directory/'protocol.json')})
    if failure:raise RuntimeError(failure)
    return {'status':'COMPLETE','completed_system_tasks':completed,'requests':len(journal.records)}


def report(directory):
    verify_frozen(directory)
    execution = json.loads((directory/'live_run/execution.json').read_text())
    if execution['status']!='COMPLETE':raise ValueError('Incomplete live workload')
    records = [json.loads(l) for l in (directory/'live_run/episodes.jsonl').read_text().splitlines()]
    private = {r['episode_id']:r for r in json.loads((directory/'formal_private.json').read_text())}
    truth = json.loads((PREVIOUS/'truth_private.json').read_text())
    quality = [];systems = {}
    for s in SYSTEMS:
        rows = [r for r in records if r['system']==s]
        main = [r for r in rows if r['category']!='unanswerable']
        assert len(main)==234 and len(rows)==244
        for r in rows:
            previous = []
            for t,label in zip(r['turns'],private[r['episode_id']]['labels']):
                assert score(t,label,truth,previous)==t['scores']
                previous = t['raw_ids']
            assert all(t['scores']['strict_success'] for t in r['turns'])==r['strict_success']
            if r['category']=='session_quality':
                quality.append({'episode_id':r['episode_id'],'system':s,'history_n':r['history_n'],'metrics':r['turns'][0]['scores']})
        turns = [t for r in rows for t in r['turns']]
        systems[s] = {'strict_n':234,'strict_passed':sum(r['strict_success'] for r in main),
                      'strict_success':sum(r['strict_success'] for r in main)/234,
                      'requests':sum(t['requests'] for t in turns),
                      'input_tokens':sum(t['input_tokens'] for t in turns),'output_tokens':sum(t['output_tokens'] for t in turns),
                      'turn_latency_p50':quantile([t['elapsed_seconds'] for t in turns],.5),
                      'turn_latency_p95':quantile([t['elapsed_seconds'] for t in turns],.95),
                      'errors':dict(Counter(t.get('error','') for t in turns if t['status']!='OK')),
                      'structural_refusal_passed':sum(r['turns'][0]['scores']['correct_refusal'] for r in rows if r['category']=='unanswerable')}
    main_by = {s:{r['episode_id']:r['strict_success'] for r in records if r['system']==s and r['category']!='unanswerable'} for s in SYSTEMS}
    strict = bootstrap([int(main_by['candidate'][i])-int(main_by['C1'][i]) for i in sorted(main_by['C1'])],random.Random(2026101304))
    events = [json.loads(l) for l in (directory/'live_run/provider.jsonl').read_text().splitlines()]
    completed = [r for r in events if r['status']=='COMPLETED']
    assert len(completed)==execution['remote_requests']==sum(v['requests'] for v in systems.values())
    assert len({r['request_id'] for r in completed})==len(completed)
    assert all(r['http_status']==200 and r['usage'] and r['request']['max_tokens']==1024 for r in completed)
    assert len([r for r in events if r['status']=='STARTED'])==len(completed)
    assert not any(r['status']=='FAILED' for r in events)
    for s in SYSTEMS:
        own = [r for r in completed if r['system']==s]
        assert sum(r['usage']['prompt_tokens'] for r in own)==systems[s]['input_tokens']
        assert sum(r['usage']['completion_tokens'] for r in own)==systems[s]['output_tokens']
    balance = [json.loads(l) for l in (directory/'live_run/balance.jsonl').read_text().splitlines()]
    def cny_total(row):
        return sum((Decimal(r['total_balance']) for r in row['balance']['balance_infos'] if r['currency']=='CNY'),Decimal(0))
    observed_cost = str(cny_total(balance[0])-cny_total(balance[-1])) if len(balance)>=2 else None
    result = {'execution':execution,'selected':json.loads((directory/'selection.json').read_text())['selected'],
              'diagnosis':json.loads((directory/'diagnosis.json').read_text()),
              'development':json.loads((directory/'selection.json').read_text())['summaries'],
              'offline_quality':quality_summary(json.loads((directory/'offline_predictions.json').read_text())),
              'live_quality':quality_summary(quality),'live_systems':systems,
              'strict_paired':strict,'strict_noninferiority_established':strict['ci95'][0]>=-.02,
              'audit':{'recomputed_turns':568,'http200_known_usage':len(completed),'frozen_hashes_verified':True},
              'balance_observations':len(balance),'observed_balance_delta_cny':observed_cost,'provider_invoice_cost_cny':None}
    write_new(directory/'final_report.json',result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','diagnose','develop','freeze','offline','live','report'))
    parser.add_argument('--directory',type=Path,required=True)
    args = parser.parse_args();torch.set_num_threads(2)
    result = globals()[args.phase](args.directory)
    print(json.dumps(result,ensure_ascii=True,indent=2),flush=True)


if __name__=='__main__':
    main()
