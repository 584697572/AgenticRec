"""Finite six-system real-API evaluation of four accuracy-oriented method probes.

CLI steps: prepare, train, calibrate, freeze, formal, report. Outputs never replace
an earlier run; public requests and evaluator labels have separate boundaries.
"""
import argparse
from collections import defaultdict, Counter
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import pyarrow.feather as feather
import pyarrow.parquet as pq
import torch

from .comparison_cases import ROOT, load_assets, make_cases, sha, write_new
from .comparison_run import HTTPJournal, OriginalWorker, OptimizedSystem, append, quantile, bootstrap
from .comparison_scoring import AnswerIdentity, score_turn
from .frontier_state import (GENRES, PROMPT, empty_state, validate_delta, merge_state,
                             feasible_ids, verify_delivery,current_obligation_errors)
from .frontier_models import (train_policy,train_generator,train_ranker,load_models,
                              ranking_features,intent_vector)
from ..adapters.account_balance import BalanceStop
from ..adapters.openai_compatible import OpenAICompatibleTransport
from ..runtime.errors import LLMTransportError
from ..runtime.secrets import read_api_key


SYSTEMS = ('original','optimized','C1','C2','C3','C4')
SEEDS = (42,2024,2025)
DATA = ROOT/'optimized/data/processed/ml-1m-v1'


def compile_catalog():
    truth, shared, _, _, module, metadata = load_assets()
    scripts=ROOT/'optimized/AgenticRec/scripts'
    sys.path.insert(0,str(scripts))
    spec=importlib.util.spec_from_file_location('frontier_identity_helper',scripts/'correct_comparison_identity.py')
    helper=importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    original, optimized, statuses=helper.identity_maps(
        feather.read_table(ROOT/'original/data/raw/upstream_movie/movies.ftr').to_pylist(),
        pq.read_table(DATA/'items.parquet').to_pylist(),module.MetadataIndex(list(truth.values())),
        module.split_title_year,module.source_title_forms)
    catalog={i:{'item_id':i,'raw_id':raw,'title':truth[raw]['title'],
                'year':truth[raw]['year'],'genres':truth[raw]['genres']}
             for i,raw in optimized.items()}
    assert len(set(r['raw_id'] for r in catalog.values()))==len(catalog)
    return truth,catalog,original,optimized,shared,sha(metadata),statuses


def make_new_cases(catalog, shared, *, calibration=False):
    raw_rows={v['raw_id']:v for v in catalog.values() if v['raw_id'] in shared
              and shared[v['raw_id']]['in_optimized_training_candidates']}
    rng=random.Random(419 if calibration else 20261013)
    cases=[]
    n=2 if calibration else 40
    options=['Comedy','Drama','Action','Thriller','Romance','Adventure','Crime','Sci-Fi']
    periods=[(1960,1985),(1986,1993),(1994,2000),(1970,1999),(1990,2000)]
    attempt=0
    while len(cases)<n:
        attempt+=1
        if attempt>2000:raise ValueError('Cannot construct feasible public tasks')
        genre=rng.choice(options); low,high=rng.choice(periods)
        other=rng.choice([g for g in options if g!=genre]) if len(cases)%4==3 else None
        excluded=['Horror'] if len(cases)%3==1 else []
        include=[genre]+([other] if other else [])
        feasible=[raw for raw,r in raw_rows.items() if set(include)<=set(r['genres'])
                  and not set(excluded)&set(r['genres']) and low<=r['year']<=high]
        if len(feasible)<10:continue
        phrase=' and '.join(g.lower() for g in include)
        negative=' Do not include horror movies.' if excluded else ''
        msg=f'Please give exactly 10 {phrase} movies, each released between {low} and {high}, inclusive.{negative} Give only titles.'
        label={'k':10,'genres':include,'exclude_genres':excluded,'year_min':low,'year_max':high,
               'excluded_raw_ids':[],'relevant_raw_ids':[],'exclude_previous':False}
        cases.append({'category':'constraints','public':{'messages':[msg],'catalog_context':[]},'labels':[label]})
    positive=defaultdict(list); seen=defaultdict(set); targets=defaultdict(set)
    for row in pq.read_table(DATA/'train.parquet').to_pylist():
        u,raw=row['raw_user_id'],row['raw_item_id']; seen[u].add(raw)
        if row['rating']>=4 and raw in raw_rows:positive[u].append((row['timestamp'],raw))
    for row in pq.read_table(DATA/'eval_test.parquet').to_pylist():
        u,raw=row['raw_user_id'],row['raw_item_id']
        if row['is_target'] and row['segment']=='WARM' and raw in raw_rows and raw not in seen[u]:targets[u].add(raw)
    historical=make_cases(shared)+make_cases(shared,calibration=True)
    blocked={c['evaluator_raw_user_id'] for c in historical if 'evaluator_raw_user_id' in c}
    users=sorted(u for u in targets if len(positive[u])>=5 and u not in blocked)
    random.Random(20261013).shuffle(users)
    selected=users[80:84] if calibration else users[:80]
    assert len(selected)==(4 if calibration else 80)
    for number,u in enumerate(selected):
        liked=[raw for _,raw in sorted(positive[u])[-5:]]
        context=[{'title':raw_rows[r]['title'],'year':raw_rows[r]['year'],'item_id':raw_rows[r]['item_id']} for r in liked]
        names='; '.join(f"{raw_rows[r]['title']} ({raw_rows[r]['year']})" for r in liked)
        msg=f'I have watched and liked these movies: {names}. Please recommend exactly 10 other movies matching my taste. Exclude all the watched movies listed. Give only titles.'
        label={'k':10,'genres':[],'exclude_genres':[],'year_min':None,'year_max':None,
               'excluded_raw_ids':liked,'relevant_raw_ids':sorted(targets[u]),'exclude_previous':False}
        case={'category':'session_quality','evaluator_raw_user_id':u,
              'public':{'messages':[msg],'catalog_context':context},'labels':[label]}
        if number >= (2 if calibration else 40):
            variant=(number-40)%4 if not calibration else number-2
            if variant==0:
                second='Keep my previous preferences. Now give exactly 10 drama movies from 1991 through 2000, inclusive. Do not include horror movies. Exclude the already watched movies.'
                second_label=dict(label,genres=['Drama'],exclude_genres=['Horror'],year_min=1991,year_max=2000,relevant_raw_ids=[])
            else:
                msg+=' Focus on action movies released from 1985 through 2000, inclusive.'
                case['public']['messages'][0]=msg
                case['labels'][0]=dict(label,genres=['Action'],year_min=1985,year_max=2000,relevant_raw_ids=[])
                if variant==1:
                    second='Keep the previous year limits and action requirement, and also require thriller. Give exactly 10 movies. Exclude those I already watched.'
                    second_label=dict(label,genres=['Action','Thriller'],year_min=1985,year_max=2000,relevant_raw_ids=[])
                elif variant==2:
                    second='Switch to comedy instead of action, but keep the previous year limits and my preferences. Give exactly 10 movies and exclude those already watched.'
                    second_label=dict(label,genres=['Comedy'],year_min=1985,year_max=2000,relevant_raw_ids=[])
                else:
                    second='Discard all previous genre and year conditions. Give exactly 10 romance movies released from 1970 through 1999, inclusive. Keep my preferences and exclude the watched movies.'
                    second_label=dict(label,genres=['Romance'],year_min=1970,year_max=1999,relevant_raw_ids=[])
            second+=' Give only titles.'
            case['category']='feedback'; case['public']['messages'].append(second); case['labels'].append(second_label)
            for lab in case['labels']:
                assert sum(set(lab['genres'])<=set(r['genres']) and not set(lab['exclude_genres'])&set(r['genres'])
                           and (lab['year_min'] is None or r['year']>=lab['year_min'])
                           and (lab['year_max'] is None or r['year']<=lab['year_max'])
                           and raw not in liked for raw,r in raw_rows.items())>=10
        cases.append(case)
    prefix='development' if calibration else 'frontier'
    for number,c in enumerate(cases):c['episode_id']=f'{prefix}-{number:03d}'
    return cases,blocked


def prepare(directory):
    if directory.exists():raise FileExistsError('Never overwrite an experiment')
    truth,catalog,original,optimized,shared,metadata_sha,statuses=compile_catalog()
    formal,old_users=make_new_cases(catalog,shared)
    calibration,_=make_new_cases(catalog,shared,calibration=True)
    formal_users={c['evaluator_raw_user_id'] for c in formal if 'evaluator_raw_user_id' in c}
    cal_users={c['evaluator_raw_user_id'] for c in calibration if 'evaluator_raw_user_id' in c}
    assert not formal_users&cal_users and not formal_users&old_users
    for phase,cases in [('calibration',calibration),('formal',formal)]:
        write_new(directory/f'{phase}_public.json',[dict(c['public'],episode_id=c['episode_id']) for c in cases])
        write_new(directory/f'{phase}_private.json',[{k:v for k,v in c.items() if k!='public'} for c in cases])
    write_new(directory/'catalog.json',{'items':catalog,'metadata_sha256':metadata_sha,'status_counts':statuses,
                                      'original_ids':original,'optimized_ids':optimized})
    write_new(directory/'truth_private.json',truth)
    write_new(directory/'training_excluded_users.json',sorted(formal_users|cal_users))
    write_new(directory/'prepared.json',{'formal_episodes':len(formal),'formal_turns':sum(len(c['labels']) for c in formal),
                                        'catalog_items':len(catalog),'users_disjoint_from_previous':True,
                                        'metadata_sha256':metadata_sha,'old_users_excluded':len(old_users)})
    return json.loads((directory/'prepared.json').read_text())


def assets(directory):
    public=json.loads((directory/'catalog.json').read_text())
    return public,{int(k):v for k,v in public['items'].items()}


def train(directory):
    from ..fusion_pipeline import FusionPipeline
    torch.set_num_threads(2)
    _,catalog=assets(directory)
    excluded=set(json.loads((directory/'training_excluded_users.json').read_text()))
    models=directory/'models'; models.mkdir(exist_ok=False)
    base=FusionPipeline.from_frozen('lightgcn',42,mode='rrf',profile='validated-20261011')
    result={'status':'RUNNING','policy':[],'generator':[],'ranker':[],'train_sha256':sha(DATA/'train.parquet')}
    started=time.perf_counter()
    for seed in SEEDS:
        result['policy'].append(train_policy(models,seed))
        result['generator'].append(train_generator(models,catalog,base.retriever.catalog.n_items,excluded,seed))
        result['ranker'].append(train_ranker(models,base.retriever,catalog,base.retriever.catalog.n_items,excluded,seed))
        write_new(models/f'training_seed_{seed}.json',{k:v[-1] for k,v in result.items() if isinstance(v,list)})
        print(json.dumps({'stage':'training_seed_complete','seed':seed,'seconds':time.perf_counter()-started}),flush=True)
    result.update(status='COMPLETED',elapsed_seconds=time.perf_counter()-started)
    write_new(models/'training.json',result)
    return result


class FrontierSystem:
    def __init__(self,variant,journal,catalog,base,models,seed=42,*,policy_stage='rl',generator_stage='rl',align=True,stateful=True,hard_gate=True,verifier=True):
        self.variant,self.journal,self.catalog,self.base=variant,journal,catalog,base
        self.policy,self.sft_policy,self.generators,self.ranker,self.mean,self.scale=load_models(models,catalog,base.catalog.n_items,seed)
        self.policy_stage,self.generator_stage=policy_stage,generator_stage
        self.align,self.stateful=align,stateful
        self.hard_gate,self.verifier=hard_gate,verifier
        self.transport=OpenAICompatibleTransport(api_base='https://api.deepseek.com',api_key=journal._key,sender=journal.sender) if journal else None
        self.state=empty_state(); self.history=[]

    def parse(self,public,index):
        message=public['messages'][index]
        envelope={'message':message,'previous_state':self.state,'genres':GENRES,
                  'catalog_items':public['catalog_context'],'conversation':self.history[-4:]}
        errors=[]
        for attempt in range(3):
            messages=[{'role':'system','content':PROMPT},{'role':'user','content':json.dumps(envelope,separators=(',',':'))}]
            if errors:messages.append({'role':'user','content':'Previous extraction invalid: '+errors[-1]+'. Correct JSON using the same public evidence.'})
            reply=self.transport.chat(messages,model='deepseek-flash',max_output_tokens=512,timeout_seconds=60,sdk_max_retries=0)
            try:
                delta=validate_delta(json.loads(reply.content),message,public['catalog_context'])
                omission=current_obligation_errors(merge_state(self.state,delta),message,public['catalog_context'])
                if omission:raise ValueError(','.join(omission))
                return delta,errors
            except (ValueError,TypeError) as error:errors.append(str(error))
        raise ValueError('Three semantic parses failed: '+errors[-1])

    def execute(self,public,index,delta):
        from ..pipeline import FixedRequest
        from ..ranking.constraints import Constraints
        if index==0:self.state=empty_state();self.history=[]
        previous=deepcopy(self.state) if self.stateful else empty_state()
        policy=self.sft_policy if self.policy_stage=='sft' else self.policy
        action=policy.action(previous,delta,public['messages'][index]) if self.variant in ('C2','C3') else 0
        state=merge_state(previous,delta,action)
        self.state=state
        eligible=feasible_ids(self.catalog,state)
        if not self.hard_gate:eligible=sorted(self.catalog)
        if not self.align:
            # Same identities, original local metadata: isolates public fact alignment.
            local={i:dict(v,genres=list(self.base.catalog.items[i].genres),year=self.base.catalog.items[i].year)
                   for i,v in self.catalog.items()}
            eligible=feasible_ids(local,state)
        if self.variant=='C4':
            ids=self.generators[self.generator_stage].generate(state['liked_item_ids'],intent_vector(state),eligible,state['k'])
        elif self.variant=='C3':
            features=torch.tensor(ranking_features(self.base,self.generators['sft'],self.catalog,state,eligible))
            with torch.no_grad():scores=self.ranker((features-self.mean)/self.scale).tolist() if len(eligible) else []
            ids=[i for _,i in sorted(zip(scores,eligible),key=lambda row:(-row[0],row[1]))][:state['k']]
        else:
            from ..retrieval.adaptive import _cf_scores
            cf,_=_cf_scores(self.base,eligible,state['liked_item_ids'],None,False)
            query=defaultdict(float)
            for seed in state['liked_item_ids']:
                for term,w in self.base.content_vectors.get(seed,{}).items():query[term]+=w/max(1,len(state['liked_item_ids']))
            for genre in state['include_genres']:query['genre:'+genre.casefold()]+=1
            sources={'popularity':sorted(eligible,key=lambda i:(-self.base.popularity[i],i))[:100],
                     'content':sorted(eligible,key=lambda i:(-sum(query.get(t,0)*w for t,w in self.base.content_vectors[i].items()),i))[:100]}
            if cf:sources['cf']=sorted(cf,key=lambda i:(-cf[i],i))[:100]
            scores=defaultdict(float)
            for ids_source in sources.values():
                for rank,i in enumerate(ids_source,1):scores[i]+=1/(60+rank)
            ids=sorted(eligible,key=lambda i:(-scores[i],-self.base.popularity[i],i))[:state['k']]
        verification=verify_delivery(ids,self.catalog,state) if self.verifier else []
        answer='\n'.join(f"{rank}. {self.catalog[i]['title']} ({self.catalog[i]['year']})" for rank,i in enumerate(ids,1))
        self.state['previous_item_ids']=ids
        self.history.extend([{'role':'user','content':public['messages'][index]},{'role':'assistant','content':answer}])
        return {'status':'OK' if not verification else 'CONSTRAINT_FAILURE','answer':answer,
                'mapped_optimized_ids':ids,'raw_ids':[self.catalog[i]['raw_id'] for i in ids],
                'extraction_verified':True,'tool_calls':2,'delta':delta,'state':deepcopy(state),
                'policy_action':action,'eligible_count':len(eligible),'verification_errors':verification}

    def turn(self,public,index):
        if index==0:self.state=empty_state();self.history=[]
        delta,repairs=self.parse(public,index)
        result=self.execute(public,index,delta)
        result['parse_repair_errors']=repairs
        return result


def score(result,label,truth,previous_raw_ids=()):
    # Same original scorer plus preregistered negative genre / previous-list rules.
    label=deepcopy(label)
    if label.get('exclude_previous'):label['excluded_raw_ids']+=list(previous_raw_ids)
    value=score_turn(result,label,truth)
    forbidden=set(label.get('exclude_genres',[]))
    if forbidden:
        good=[]
        for raw in result['raw_ids']:
            item=truth.get(str(raw))
            good.append(bool(item and set(label['genres'])<=set(item['genres']) and not forbidden&set(item['genres'])
                             and (label['year_min'] is None or item['year']>=label['year_min'])
                             and (label['year_max'] is None or item['year']<=label['year_max'])
                             and raw not in label['excluded_raw_ids']))
        value['constraint_precision']=sum(good)/len(good) if good else 0
        value['strict_success']=value['strict_success'] and all(good)
    return value


def freeze(directory):
    formal_dir=directory/'formal_run'
    if formal_dir.exists():raise ValueError('Cannot freeze after formal execution')
    names=['formal_public.json','formal_private.json','calibration_public.json','calibration_private.json',
           'catalog.json','truth_private.json','training_excluded_users.json','prepared.json']
    files={name:sha(directory/name) for name in names}
    files.update({str(p.relative_to(directory)):sha(p) for p in sorted((directory/'models').glob('*')) if p.is_file()})
    sources={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'optimized/AgenticRec/src/agenticrec').rglob('*.py')}
    sources['original/reproduction/scripts/comparison_app_worker.py']=sha(ROOT/'original/reproduction/scripts/comparison_app_worker.py')
    for name in sources:
        dest=directory/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/name).read_bytes())
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    protocol={'status':'FROZEN','systems':SYSTEMS,'seeds':SEEDS,'online_seed':42,
              'method_scope':'small genuinely trained methods; not industrial LLM paper reproduction',
              'model':'deepseek-flash','max_output_tokens':512,'temperature':0,'network_retries':0,
              'stopping':'all four combinations evaluated or insufficient provider account balance',
              'quality_cohort':'40 new users disjoint from prior experiment; historically exposed MovieLens test',
              'metadata_scope':'source-confirmed canonical identity/year/genres used by NEW systems; B0/B1 untouched',
              'files_sha256':files,'sources_sha256':sources,'commit':commit,
              'train_sha256':sha(DATA/'train.parquet'),'test_sha256':sha(DATA/'eval_test.parquet')}
    write_new(directory/'protocol.json',protocol)
    return protocol


def verify_frozen(directory):
    protocol=json.loads((directory/'protocol.json').read_text())
    for name,digest in protocol['files_sha256'].items():
        if sha(directory/name)!=digest:raise ValueError('Frozen input changed: '+name)
    for name,digest in protocol['sources_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen source changed: '+name)
    return protocol


def run(directory,phase):
    torch.set_num_threads(2)
    if phase=='formal':verify_frozen(directory)
    run_dir=directory/(phase+'_run');run_dir.mkdir(exist_ok=False)
    journal=HTTPJournal(read_api_key(),run_dir)
    worker=None;records=[];failure=None
    systems=SYSTEMS if phase=='formal' else SYSTEMS[1:]
    pub_assets,catalog=assets(directory)
    public=json.loads((directory/f'{phase}_public.json').read_text())
    labels=json.loads((directory/f'{phase}_private.json').read_text())
    truth=json.loads((directory/'truth_private.json').read_text())
    identity=AnswerIdentity(truth)
    original_ids={int(k):v for k,v in pub_assets['original_ids'].items()}
    optimized_ids={int(k):v for k,v in pub_assets['optimized_ids'].items()}
    try:
        balance=journal.guard.refresh()
        if journal.guard.should_stop():raise BalanceStop(journal.guard.reason)
        baseline=OptimizedSystem(journal,balance)
        variants={s:FrontierSystem(s,journal,catalog,baseline.pipeline.retriever,directory/'models') for s in systems if s.startswith('C')}
        if 'original' in systems:worker=OriginalWorker(run_dir,journal)
        write_new(run_dir/'started.json',{'source_protocol_sha256':sha(directory/'protocol.json') if phase=='formal' else None,
                                          'systems':systems,'original_ready':worker.ready if worker else None,
                                          'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
        for episode_number,(pub,private) in enumerate(zip(public,labels)):
            assert pub['episode_id']==private['episode_id']
            # Balanced rotating execution order; priority is implemented during
            # calibration and construction, formal order avoids time confounding.
            order=list(systems[episode_number%len(systems):]+systems[:episode_number%len(systems)])
            for system in order:
                if journal.guard.should_stop():raise BalanceStop(journal.guard.reason)
                turns=[];previous_raw=[]
                for number,label in enumerate(private['labels']):
                    journal.context={'system':system,'episode_id':pub['episode_id'],'turn':number+1}
                    before=len(journal.records);clock=time.perf_counter()
                    try:
                        if system=='original':
                            result=worker.turn(pub,number)
                            result.update(identity.extract(result['answer'],[original_ids.get(i) for i in result['mapped_catalog_ids']]))
                            result['tool_calls']=len(result['tool_trace'])
                        elif system=='optimized':
                            result=baseline.turn(pub,number)
                            result.update(raw_ids=[optimized_ids.get(i) for i in result['mapped_optimized_ids']],extraction_verified=True)
                        else:result=variants[system].turn(pub,number)
                    except (ValueError,KeyError,TypeError) as error:
                        result={'status':'EXECUTION_FAILURE','error_type':type(error).__name__,'error':str(error),
                                'answer':'','raw_ids':[],'extraction_verified':False,'tool_calls':0}
                    result['end_to_end_seconds']=time.perf_counter()-clock
                    calls=journal.records[before:]
                    result['requests']=len(calls)
                    result['input_tokens']=sum(c['usage']['prompt_tokens'] for c in calls) if all(c['usage'] for c in calls) else None
                    result['output_tokens']=sum(c['usage']['completion_tokens'] for c in calls) if all(c['usage'] for c in calls) else None
                    result['scores']=score(result,label,truth,previous_raw)
                    previous_raw=[i for i in result['raw_ids'] if i is not None]
                    turns.append(result)
                record={'episode_id':pub['episode_id'],'system':system,'category':private['category'],
                        'strict_success':all(t['scores']['strict_success'] for t in turns),'turns':turns}
                append(run_dir/'episodes.jsonl',record);records.append(record)
                print(json.dumps({'stage':'episode_completed','phase':phase,'episode':pub['episode_id'],
                                  'system':system,'success':record['strict_success'],'completed':len(records),
                                  'planned':len(public)*len(systems),'requests':len(journal.records)}),flush=True)
        journal.guard.refresh()
    except Exception as error:
        failure={'error_type':type(error).__name__,'reason':str(error).replace(journal._key,'[REDACTED]')}
        write_new(run_dir/'failure.json',failure)
    finally:
        if worker is not None:worker.close()
    summary=summarize(records,public,journal.records,systems)
    summary.update(status='COMPLETED' if failure is None and len(records)==len(public)*len(systems) else 'BLOCKED',blocker=failure)
    write_new(run_dir/'summary.json',summary)
    print(json.dumps({'stage':'run_complete','phase':phase,'status':summary['status'],'requests':len(journal.records)}),flush=True)
    return summary


def summarize(records,public,provider,systems=SYSTEMS):
    summaries={}
    for system in systems:
        rows=[r for r in records if r['system']==system];turns=[t for r in rows for t in r['turns']]
        calls=[c for c in provider if c['system']==system]
        quality=[r['turns'][0]['scores'] for r in rows if r['category']=='session_quality']
        elapsed=[sum(t['end_to_end_seconds'] for t in r['turns']) for r in rows]
        summaries[system]={'completed_episodes':len(rows),'planned_episodes':len(public),
                           'strict_success':sum(r['strict_success'] for r in rows)/len(public) if len(rows)==len(public) else None,
                           'constraint_precision':sum(t['scores']['constraint_precision'] for t in turns)/len(turns) if turns else None,
                           'quality_users':len(quality),'quality_metrics':{m:sum(q[m] for q in quality)/len(quality) if quality else None for m in ('hit','recall','ndcg')},
                           'categories':{c:{'n':sum(r['category']==c for r in rows),'success':sum(r['strict_success'] for r in rows if r['category']==c)} for c in ('constraints','session_quality','feedback')},
                           'requests':len(calls),'completed_http':sum(c['status']=='COMPLETED' for c in calls),
                           'input_tokens':sum(c['usage']['prompt_tokens'] for c in calls) if all(c['usage'] for c in calls) else None,
                           'output_tokens':sum(c['usage']['completion_tokens'] for c in calls) if all(c['usage'] for c in calls) else None,
                           'latency_p50_seconds':quantile(elapsed,.5) if elapsed else None,
                           'latency_p95_seconds':quantile(elapsed,.95) if elapsed else None,
                           'tool_calls':sum(t['tool_calls'] for t in turns),
                           'identity_errors':sum(t['scores']['identity_errors'] for t in turns),
                           'parse_repairs':sum(len(t.get('parse_repair_errors',[])) for t in turns),
                           'state_actions':dict(Counter(str(t['policy_action']) for t in turns if 'policy_action' in t))}
    by_id=defaultdict(dict)
    for row in records:by_id[row['episode_id']][row['system']]=row
    differences={}
    for baseline in ('original','optimized','C1','C2','C3'):
        for system in ('C1','C2','C3','C4'):
            if system==baseline:continue
            paired=[p for p in by_id.values() if system in p and baseline in p]
            if not paired:continue
            rng=random.Random(42)
            strict=[int(p[system]['strict_success'])-int(p[baseline]['strict_success']) for p in paired]
            result={'strict_success':dict(bootstrap(strict,rng),n=len(paired))}
            quality=[p for p in paired if p[system]['category']=='session_quality']
            for metric in ('hit','recall','ndcg'):
                delta=[p[system]['turns'][0]['scores'][metric]-p[baseline]['turns'][0]['scores'][metric] for p in quality]
                if delta:result[metric]=dict(bootstrap(delta,rng),n=len(delta))
            # Exact paired discordance sign test (McNemar binomial), then Holm
            # correction below for the 4 preregistered comparisons with B1.
            wins=sum(d==1 for d in strict);losses=sum(d==-1 for d in strict);n=wins+losses
            p=min(1.,2*sum(math_comb(n,k) for k in range(min(wins,losses)+1))/(2**n)) if n else 1.
            result['paired_discordance']={'wins':wins,'losses':losses,'two_sided_p':p}
            differences[system+'-minus-'+baseline]=result
    tests=sorted((v['paired_discordance']['two_sided_p'],k) for k,v in differences.items() if k.endswith('-minus-optimized'))
    previous=0.
    for number,(p,k) in enumerate(tests):
        previous=max(previous,min(1.,p*(len(tests)-number)))
        differences[k]['paired_discordance']['holm_adjusted_p']=previous
    return {'systems':summaries,'paired_differences':differences,'generation_requests':len(provider),
            'completed_http':sum(c['status']=='COMPLETED' for c in provider),
            'actual_invoice_cost_cny':None,'quality_limit':'historically exposed MovieLens test; new disjoint users',
            'industrial_paper_model_reproduction':False}


def math_comb(n,k):
    import math
    return math.comb(n,k)


def report(directory):
    protocol=verify_frozen(directory)
    run_dir=directory/'formal_run'
    records=[json.loads(s) for s in (run_dir/'episodes.jsonl').read_text().splitlines()]
    public=json.loads((directory/'formal_public.json').read_text())
    labels={r['episode_id']:r for r in json.loads((directory/'formal_private.json').read_text())}
    truth=json.loads((directory/'truth_private.json').read_text())
    for row in records:
        previous=[]
        for turn,label in zip(row['turns'],labels[row['episode_id']]['labels']):
            if score(turn,label,truth,previous)!=turn['scores']:raise ValueError('Metric recomputation mismatch')
            previous=[i for i in turn['raw_ids'] if i is not None]
        assert row['strict_success']==all(t['scores']['strict_success'] for t in row['turns'])
    latest={};started=set()
    for line in (run_dir/'provider.jsonl').read_text().splitlines():
        event=json.loads(line);latest[event['request_id']]=event
        if event['status']=='STARTED':started.add(event['request_id'])
    assert started==set(latest) and all(v['status']!='STARTED' for v in latest.values())
    actual=summarize(records,public,list(latest.values()))
    prior=json.loads((run_dir/'summary.json').read_text())
    for key,value in actual.items():assert prior[key]==value,(key,'summary differs')
    totals={'requests':len(latest),'input_tokens':sum(r['usage']['prompt_tokens'] for r in latest.values() if r['usage']),
            'output_tokens':sum(r['usage']['completion_tokens'] for r in latest.values() if r['usage'])}
    assert sum(t['requests'] for r in records for t in r['turns'])==totals['requests']
    assert sum(t['input_tokens'] for r in records for t in r['turns'])==totals['input_tokens']
    assert sum(t['output_tokens'] for r in records for t in r['turns'])==totals['output_tokens']
    evidence={'recomputed_episodes':len(records),'recomputed_turns':sum(len(r['turns']) for r in records),
              'provider_totals':totals,'unresolved_intents':0,'protocol_sha256':sha(directory/'protocol.json'),
              'episodes_sha256':sha(run_dir/'episodes.jsonl'),'provider_sha256':sha(run_dir/'provider.jsonl'),
              'source_commit':protocol['commit'],'source_inputs_verified':True}
    write_new(ROOT/'optimized/reports/frontier_validation_20261012.json',{'summary':prior,'verification':evidence})
    write_new(directory/'verification.json',evidence)
    return evidence


def replay(directory):
    """Controlled same-real-parses ablations, never new online success claims."""
    from ..fusion_pipeline import FusionPipeline
    torch.set_num_threads(2)
    verify_frozen(directory)
    dest=directory/'component_replay';dest.mkdir(exist_ok=False)
    _,catalog=assets(directory)
    base=FusionPipeline.from_frozen('lightgcn',42,mode='rrf',profile='validated-20261011').retriever
    original=[json.loads(s) for s in (directory/'formal_run/episodes.jsonl').read_text().splitlines()]
    pubs={p['episode_id']:p for p in json.loads((directory/'formal_public.json').read_text())}
    labs={p['episode_id']:p for p in json.loads((directory/'formal_private.json').read_text())}
    truth=json.loads((directory/'truth_private.json').read_text())
    variants=[('C1_no_state','C1',42,{'stateful':False}),
              ('C1_local_metadata','C1',42,{'align':False}),
              ('C1_no_verifier','C1',42,{'verifier':False}),
              ('C1_no_gate','C1',42,{'hard_gate':False,'verifier':False}),
              ('C2_sft_only','C2',42,{'policy_stage':'sft'}),
              ('C2_policy_off','C2',42,{'variant':'C1'}),
              ('C3_ranker_off','C3',42,{'variant':'C2'}),
              ('C4_sft_only','C4',42,{'generator_stage':'sft'}),
              ('C4_no_gate','C4',42,{'hard_gate':False,'verifier':False})]
    variants += [(f'{system}_seed{seed}',system,seed,{}) for seed in SEEDS for system in ('C2','C3','C4')]
    reports={}
    for name,source,seed,options in variants:
        options=options.copy();implementation=options.pop('variant',source)
        system=FrontierSystem(implementation,None,catalog,base,directory/'models',seed,**options)
        paired=[];new_rows=[]
        for old in (r for r in original if r['system']==source):
            system.state=empty_state();system.history=[]
            turns=[];previous=[]
            for index,(prior,label) in enumerate(zip(old['turns'],labs[old['episode_id']]['labels'])):
                clock=time.perf_counter()
                if 'delta' in prior:
                    result=system.execute(pubs[old['episode_id']],index,prior['delta'])
                else:
                    result={'status':prior['status'],'answer':'','raw_ids':[],'extraction_verified':False,
                            'tool_calls':0,'parser_failure_preserved':True}
                result['component_seconds']=time.perf_counter()-clock
                result['scores']=score(result,label,truth,previous)
                previous=[i for i in result['raw_ids'] if i is not None]
                turns.append(result)
            row={'episode_id':old['episode_id'],'system':name,'category':old['category'],
                 'strict_success':all(t['scores']['strict_success'] for t in turns),'turns':turns,
                 'scope':'offline component replay of recorded real parses; no new LLM requests'}
            append(dest/'episodes.jsonl',row);new_rows.append(row);paired.append((old,row))
        metrics={'strict_success':sum(r['strict_success'] for r in new_rows)/len(new_rows),
                 'constraint_precision':sum(t['scores']['constraint_precision'] for r in new_rows for t in r['turns'])/sum(len(r['turns']) for r in new_rows)}
        quality=[r['turns'][0]['scores'] for r in new_rows if r['category']=='session_quality']
        metrics.update({m:sum(r[m] for r in quality)/len(quality) for m in ('hit','recall','ndcg')})
        rng=random.Random(42)
        differences={'strict_success':bootstrap([int(new['strict_success'])-int(old['strict_success']) for old,new in paired],rng)}
        quality_pairs=[p for p in paired if p[0]['category']=='session_quality']
        for metric in ('hit','recall','ndcg'):
            differences[metric]=bootstrap([new['turns'][0]['scores'][metric]-old['turns'][0]['scores'][metric] for old,new in quality_pairs],rng)
        if name in ('C2_seed42','C3_seed42','C4_seed42'):
            assert all(old['strict_success']==new['strict_success'] and
                       all(a['raw_ids']==b['raw_ids'] for a,b in zip(old['turns'],new['turns'])) for old,new in paired)
        reports[name]={'source_system':source,'seed':seed,'metrics':metrics,'paired_difference_from_source':differences,
                       'episodes':len(new_rows),'online_costs_and_latency':'NOT EVALUATED',
                       'component_seconds':sum(t['component_seconds'] for r in new_rows for t in r['turns'])}
        print(json.dumps({'stage':'component_replay_complete','variant':name,'metrics':metrics}),flush=True)
    write_new(dest/'summary.json',{'scope':'same recorded real LLM parses; component-only counterfactual, not fresh online trials',
                                  'new_api_requests':0,'reports':reports,'episodes_sha256':sha(dest/'episodes.jsonl')})
    write_new(ROOT/'optimized/reports/frontier_component_replay_20261012.json',json.loads((dest/'summary.json').read_text()))
    return {'status':'COMPLETED','variants':len(reports),'new_api_requests':0}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','train','calibrate','freeze','formal','report','replay'))
    parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args();directory=args.directory.resolve()
    if not directory.is_relative_to((ROOT/'optimized/artifacts/comparison').resolve()):raise ValueError('Private outputs outside workspace')
    operation={'prepare':prepare,'train':train,'freeze':freeze,'report':report,'replay':replay,
               'calibrate':lambda d:run(d,'calibration'),'formal':lambda d:run(d,'formal')}[args.command]
    value=operation(directory)
    if args.command in ('prepare','report'):print(json.dumps(value,indent=2))
    return 2 if value.get('status')=='BLOCKED' else 0


if __name__=='__main__':raise SystemExit(main())
