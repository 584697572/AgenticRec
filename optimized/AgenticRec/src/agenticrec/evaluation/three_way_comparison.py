"""Finite, append-only comparison of actual original, T19 no_content, and C1.

CLI: prepare -> calibrate -> freeze -> formal -> report. No training or model
selection; source versions are reused unchanged behind a common HTTP limit.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from decimal import Decimal
import importlib.util
import json
import math
from pathlib import Path
import random
import subprocess
import time

import pyarrow.parquet as pq
import torch

from .comparison_cases import ROOT, write_new
from .comparison_run import HTTPJournal, OriginalWorker, append, bootstrap, quantile
from .comparison_scoring import AnswerIdentity
from .frontier_experiment import FrontierSystem, compile_catalog, score as feasible_score, DATA
from .cost_limits import PER_REQUEST_COST_CNY
from .system_factory import build_benchmark_agent_loop
from .variants import build_pipeline_variant
from ..pipeline import FixedRecommendationPipeline, FixedRequest
from ..fusion_pipeline import FusionPipeline
from ..agent.router import RoutingRequest, fixed_request_payload
from ..adapters.llm import ChatAdapter
from ..adapters.openai_compatible import OpenAICompatibleTransport
from ..config import LLMBudget
from ..runtime.errors import LLMTransportError
from ..runtime.secrets import read_api_key
from ..data import digest as sha

SYSTEMS = ('original', 'historical_best', 'C1')
MODELS = ROOT/'optimized/artifacts/comparison/frontier_20261012/models'
NAMES = {'original':'微软原配置运行版', 'historical_best':'历史86% no_content版本', 'C1':'C1'}


def prior_users():
    users=set()
    for dirname in ('actual_versions_20261012','frontier_20261012'):
        for phase in ('formal','calibration'):
            path=ROOT/'optimized/artifacts/comparison'/dirname/f'{phase}_private.json'
            if path.exists():
                users.update(r['evaluator_raw_user_id'] for r in json.loads(path.read_text()) if 'evaluator_raw_user_id' in r)
    # T19 uses mapped user IDs; resolve the frozen map rather than guessing.
    path=DATA/'id_map.json'
    mapping=json.loads(path.read_text())
    lookup={int(k):int(v) for k,v in mapping['model_to_user'].items()}
    for phase in ('test','development'):
        path=ROOT/'optimized/data/eval_private/interactive-v1'/f'{phase}.public.jsonl'
        for line in path.read_text().splitlines():
            group=json.loads(line)['group_id']
            users.add(lookup[int(group.split(':')[1])])
    return users


def build_cases(catalog, shared):
    eligible={v['raw_id']:v for v in catalog.values() if v['raw_id'] in shared
              and shared[v['raw_id']]['in_optimized_training_candidates']}
    positive=defaultdict(list);seen=defaultdict(set);targets=defaultdict(set)
    for r in pq.read_table(DATA/'train.parquet').to_pylist():
        u,raw=r['raw_user_id'],r['raw_item_id'];seen[u].add(raw)
        if r['rating']>=4 and raw in eligible:positive[u].append((r['timestamp'],raw))
    for r in pq.read_table(DATA/'eval_test.parquet').to_pylist():
        u,raw=r['raw_user_id'],r['raw_item_id']
        if r['is_target'] and r['segment']=='WARM' and raw in eligible and raw not in seen[u]:targets[u].add(raw)
    excluded=prior_users()
    users=sorted(u for u in targets if len(positive[u])>=10 and u not in excluded)
    rng=random.Random(2026101301);rng.shuffle(users)
    if len(users)<246:raise ValueError(f'Only {len(users)} untouched eligible users, need 246')
    def label(**kw):
        return dict({'k':10,'genres':[],'exclude_genres':[],'year_min':None,'year_max':None,
                     'excluded_raw_ids':[],'relevant_raw_ids':[],'exclude_previous':False,'feasible':True},**kw)
    def case(category,messages,labels,context=(),**extra):
        return {'category':category,'public':{'messages':messages,'catalog_context':list(context)},'labels':labels,**extra}
    def count(lab):
        return sum(set(lab['genres'])<=set(r['genres']) and not set(lab['exclude_genres'])&set(r['genres'])
                   and (lab['year_min'] is None or r['year']>=lab['year_min'])
                   and (lab['year_max'] is None or r['year']<=lab['year_max'])
                   and raw not in lab['excluded_raw_ids'] for raw,r in eligible.items())
    constraints=[];used=set();attempts=0
    genres=['Comedy','Drama','Action','Thriller','Romance','Adventure','Crime','Sci-Fi']
    periods=[(1950,1980),(1960,1990),(1970,1995),(1980,2000),(1990,1999),(1994,2000)]
    while len(constraints)<62:
        attempts+=1
        if attempts>20000:raise ValueError('Not enough feasible unique constraints')
        genre=rng.choice(genres);low,high=rng.choice(periods);n=len(constraints)
        include=[genre]+([rng.choice([g for g in genres if g!=genre])] if n%3==1 else [])
        no=['Horror'] if n%3==2 else []
        lab=label(genres=include,exclude_genres=no,year_min=low,year_max=high)
        signature=(tuple(include),tuple(no),low,high)
        if count(lab)<10 or signature in used:continue
        used.add(signature)
        text=' and '.join(g.lower() for g in include)
        forms=[f'Please give exactly 10 {text} movies, each released between {low} and {high}, inclusive.',
               f'I want exactly 10 {text} movies released from {low} through {high}, inclusive.',
               f'Choose exactly 10 {text} movies released from {low} to {high}, inclusive.']
        msg=forms[n%3]+(' Do not include horror movies.' if no else '')+' Give only titles.'
        constraints.append(case('constraints',[msg],[lab],scenario='constraint:'+str(n%3)))
    quality=[];feedback=[]
    for n,u in enumerate(users[:246]):
        history_n=(3,5,10)[n%3]
        liked=[raw for _,raw in sorted(positive[u])[-history_n:]]
        context=[{'title':eligible[r]['title'],'year':eligible[r]['year'],'item_id':eligible[r]['item_id']} for r in liked]
        titles='; '.join(f"{eligible[r]['title']} ({eligible[r]['year']})" for r in liked)
        msg=f'I have watched and liked these movies: {titles}. Please recommend exactly 10 other movies matching my taste. Exclude all the watched movies listed. Give only titles.'
        lab=label(excluded_raw_ids=liked,relevant_raw_ids=sorted(targets[u]))
        if n<180 or n>=240:
            quality.append(case('session_quality',[msg],[lab],context,evaluator_raw_user_id=u,history_n=history_n,scenario='history:'+str(history_n)))
            continue
        mode=(n-180)%6
        first=deepcopy(lab);first['relevant_raw_ids']=[]
        if mode in (1,2,3,4):
            msg+=' Focus on action movies released from 1985 through 2000, inclusive.'
            first.update(genres=['Action'],year_min=1985,year_max=2000)
        second=deepcopy(first)
        if mode==0:
            text='Keep my previous preferences. Give exactly 10 drama movies released from 1990 through 2000, inclusive. Do not include horror movies. Exclude those already watched.'
            second.update(genres=['Drama'],exclude_genres=['Horror'],year_min=1990,year_max=2000)
        elif mode==1:
            text='Keep the previous year limits and action requirement, and also require thriller. Give exactly 10 movies and exclude those already watched.'
            second.update(genres=['Action','Thriller'])
        elif mode in (2,4):
            text='Switch to comedy instead of action, but keep the previous year limits and my preferences. Give exactly 10 movies and exclude those already watched.'
            second.update(genres=['Comedy'])
        elif mode==3:
            text='Discard all previous genre and year conditions. Give exactly 10 romance movies released from 1970 through 1999, inclusive. Keep my preferences and exclude the watched movies.'
            second.update(genres=['Romance'],year_min=1970,year_max=1999)
        else:
            text='Keep my previous preferences. Give exactly 10 comedy movies released from 1970 through 2000, inclusive. Exclude those already watched.'
            second.update(genres=['Comedy'],year_min=1970,year_max=2000)
        messages=[msg,text+' Give only titles.'];labels=[first,second]
        if mode==4:
            messages.append('Discard all previous genre and year conditions. Give exactly 10 drama movies released from 1960 through 1985, inclusive. Keep my preferences and exclude the watched movies. Give only titles.')
            labels.append(dict(second,genres=['Drama'],exclude_genres=[],year_min=1960,year_max=1985))
        elif mode==5:
            messages.append('Keep the previous year limits and comedy requirement. Give exactly 10 different movies in another batch. Exclude the previous recommendations and all already watched movies. Give only titles.')
            labels.append(dict(second,exclude_previous=True))
        assert all(count(l)>=30 for l in labels)
        feedback.append(case('feedback',messages,labels,context,evaluator_raw_user_id=u,history_n=history_n,scenario='feedback:'+str(mode)))
    refusal=[]
    for n in range(21):
        if n<10:
            genre=['Drama','Comedy','Action','Romance','Thriller','Crime','Adventure','Sci-Fi','Horror','Documentary'][n]
            text=f'Give exactly 10 {genre.lower()} movies. Do not include {genre.lower()} movies. If no movies satisfy both requirements, explain that you cannot fulfill the request.'
            lab=label(genres=[genre],exclude_genres=[genre],feasible=False)
        else:
            low=2026+n-10;high=low+1
            text=f'Give exactly 10 movies released from {low} through {high}, inclusive. If no movies satisfy the request, explain that you cannot fulfill it.'
            lab=label(year_min=low,year_max=high,feasible=False)
        assert count(lab)==0
        refusal.append(case('unanswerable',[text],[lab],scenario='refusal:'+str(n%2)))
    formal=constraints[:60]+quality[:180]+feedback+refusal[:20]
    # Three quality histories, two constraints, two/three-turn feedback, refusal.
    cal_feedback=deepcopy([feedback[0],feedback[4],feedback[5]])
    for i,c in enumerate(cal_feedback):
        q=quality[180+i]
        c['public']['catalog_context']=q['public']['catalog_context']
        c['public']['messages'][0]=q['public']['messages'][0]
        if i==1:c['public']['messages'][0]+=' Focus on action movies released from 1985 through 2000, inclusive.'
        c['evaluator_raw_user_id']=q['evaluator_raw_user_id']
        for lab in c['labels']:lab['excluded_raw_ids']=q['labels'][0]['excluded_raw_ids']
        assert all(count(l)>=30 for l in c['labels'])
    calibration=constraints[60:]+quality[183:186]+cal_feedback+refusal[20:]
    for phase,cases in [('formal',formal),('calibration',calibration)]:
        for n,c in enumerate(cases):c['episode_id']=f'{phase}-three-{n:03d}'
    assert len(formal)==320 and sum(len(c['labels']) for c in formal)==400
    assert len(calibration)==9
    a={c['evaluator_raw_user_id'] for c in formal if 'evaluator_raw_user_id' in c}
    b={c['evaluator_raw_user_id'] for c in calibration if 'evaluator_raw_user_id' in c}
    assert len(a)==240 and len(b)==6 and not a&b and not (a|b)&excluded
    return formal,calibration,excluded


def prepare(directory):
    if directory.exists():raise FileExistsError('Refuse to overwrite earlier evidence')
    truth,catalog,original,optimized,shared,metadata_sha,statuses=compile_catalog()
    formal,calibration,excluded=build_cases(catalog,shared)
    for phase,cases in [('formal',formal),('calibration',calibration)]:
        write_new(directory/f'{phase}_public.json',[dict(c['public'],episode_id=c['episode_id']) for c in cases])
        write_new(directory/f'{phase}_private.json',[{k:v for k,v in c.items() if k!='public'} for c in cases])
    write_new(directory/'catalog.json',{'items':catalog,'original_ids':original,'optimized_ids':optimized,
                                      'metadata_sha256':metadata_sha,'status_counts':statuses})
    write_new(directory/'truth_private.json',truth)
    write_new(directory/'prepared.json',{'formal_episodes':320,'main_feasible_episodes':300,'quality_users':180,
                                        'feedback_users':60,'formal_turns_per_system':400,'calibration_episodes':9,
                                        'past_users_excluded':len(excluded),'catalog_items':len(catalog),
                                        'new_unique_users':240,'training':False,'download':False})
    return json.loads((directory/'prepared.json').read_text())


class MatchedJournal(HTTPJournal):
    """Override only the provider output limit; actual HTTP payload is journaled."""
    def sender(self,url,headers,body,timeout):
        payload=json.loads(body)
        requested=payload.get('max_tokens')
        if (url!='https://api.deepseek.com/chat/completions' or payload.get('model')!='deepseek-flash'
                or requested not in (512,1024) or payload.get('temperature')!=0
                or payload.get('thinking')!={'type':'disabled'} or payload.get('stream',False) or len(body)>100000):
            raise ValueError('Request outside common comparison settings')
        payload['max_tokens']=1024
        body=json.dumps(payload).encode('utf-8')
        if self.guard.should_stop():raise RuntimeError(self.guard.reason)
        record={'request_id':len(self.records)+1,**self.context,'request':payload,
                'adapter_requested_max_tokens':requested,'status':'STARTED','usage':None,'reply':None,'http_status':None}
        self.records.append(record);append(self.directory/'provider.jsonl',record)
        clock=time.perf_counter()
        try:
            def send():
                reply=OpenAICompatibleTransport._stdlib_send(url,headers,body,timeout)
                if reply.status_code>=400:raise LLMTransportError('Provider error',status_code=reply.status_code,retryable=False)
                return reply
            reply=self.guard.call(send)
            decoded=json.loads(reply.body.decode('utf-8').replace(self._key,'[REDACTED]'))
            record.update(status='COMPLETED',http_status=reply.status_code,usage=decoded.get('usage'),reply=decoded)
            return reply
        except Exception as error:
            record.update(status='FAILED',error_type=type(error).__name__,http_status=getattr(error,'status_code',None))
            raise
        finally:
            record['elapsed_seconds']=time.perf_counter()-clock;append(self.directory/'provider.jsonl',record)


class HistoricalBest:
    """T19 actual no_content pipeline/AgentLoop; retain only executed public state."""
    def __init__(self,journal,balance):
        self.pipeline=build_pipeline_variant(FixedRecommendationPipeline.from_frozen('lightgcn',42),'no_content')
        transport=OpenAICompatibleTransport(api_base='https://api.deepseek.com',api_key=journal._key,sender=journal.sender)
        budget=LLMBudget(allow_paid_api=True,provider='deepseek',model_id='deepseek-flash',api_request_cap=5000,
                         max_output_tokens=1024,money_budget=5000*PER_REQUEST_COST_CNY,max_retries=0,
                         per_request_timeout_seconds=60,round_deadline_seconds=180)
        self.loop=build_benchmark_agent_loop('no_content',fixed_pipeline=self.pipeline,
                    planner=ChatAdapter(transport,budget),planned_cost_ceiling=PER_REQUEST_COST_CNY)
        self.history=[];self.previous=None

    def turn(self,public,index):
        if index==0:self.history=[];self.previous=None
        envelope={'mode':'text','message':public['messages'][index],'user_id':None,'history_authorized':False,
                  'catalog_items':public['catalog_context'],'conversation':deepcopy(self.history)}
        if self.previous is not None:envelope['previous_executed_request']=fixed_request_payload(self.previous)
        report=self.loop.run(RoutingRequest(f'{public["episode_id"]}:{index+1}',envelope,agent_need_score=1.,needs_planning=True))
        for execution in report.executions:
            for event in execution.traces:
                raw=event.arguments.get('request')
                if event.result.ok and isinstance(raw,dict):self.previous=FixedRequest.parse(raw)
        value=report.to_dict();response=value.get('response') or {}
        rows=response.get('recommendations',[])
        answer='\n'.join(f'{n}. {r["title"]}' for n,r in enumerate(rows,1))
        self.history.extend([{'role':'user','content':public['messages'][index]},{'role':'assistant','content':answer or response.get('reason','')}])
        return {'status':response.get('status',value['status']),'answer':answer,
                'mapped_optimized_ids':[r['item_id'] for r in rows],'extraction_verified':True,
                'tool_calls':value['tool_calls'],'trace':value,'response_reason':response.get('reason','')}


def score(result,label,truth,previous=()):
    result=deepcopy(result);label=deepcopy(label)
    scores=feasible_score(result,label,truth,previous)
    ranks=[];credited=set()
    for n,i in enumerate(result['raw_ids'][:label['k']],1):
        if i in label['relevant_raw_ids'] and i not in credited:ranks.append(n);credited.add(i)
    scores['mrr']=1/ranks[0] if ranks and result['status']=='OK' and result['extraction_verified'] else 0.
    if not label['feasible']:
        import re
        empty=not result['raw_ids']
        text=result.get('answer','')+' '+result.get('response_reason','')
        explained=bool(re.search(r'cannot|can\W*t|no (?:movies?|films?|titles?|items?|results?)|conflict|contradict|not possible|unable|incompatible|zero movies',text,re.I))
        structured=result['status'] in ('CLARIFY_CONFLICT','CLARIFY','NO_CANDIDATES','NO_FEASIBLE_ITEMS','EMPTY_CANDIDATES','MISSING_FIELD')
        verified_gate=(result['status']=='CONSTRAINT_FAILURE' and result.get('eligible_count')==0 and 'delta' in result)
        scores.update(safe_non_delivery=empty,correct_refusal=empty and (explained or structured or verified_gate),
                      explanatory_refusal=empty and explained)
        scores['strict_success']=scores['correct_refusal']
    return scores


def freeze(directory):
    if (directory/'formal_run').exists():raise ValueError('Cannot freeze after prediction')
    paths=['formal_public.json','formal_private.json','calibration_public.json','calibration_private.json',
           'catalog.json','truth_private.json','prepared.json','calibration_audit.json']
    source=ROOT/'optimized/AgenticRec/src/agenticrec'
    sources={str(p.relative_to(ROOT)):sha(p) for p in sorted(source.rglob('*.py'))}
    sources['original/reproduction/scripts/comparison_app_worker.py']=sha(ROOT/'original/reproduction/scripts/comparison_app_worker.py')
    sources['optimized/reports/three_way_protocol_20261013.md']=sha(ROOT/'optimized/reports/three_way_protocol_20261013.md')
    for name in ('optimized/AgenticRec/scripts/audit_frontier_validation.py',
                 'optimized/AgenticRec/scripts/correct_comparison_identity.py',
                 'original/reproduction/scripts/catalog_identity.py',
                 'original/reproduction/scripts/live_app_single_turn.py'):
        sources[name]=sha(ROOT/name)
    weights={str(p.relative_to(ROOT)):sha(p) for p in MODELS.glob('*.pt')}
    old=json.loads((ROOT/'optimized/reports/rec_baselines/lightgcn_seed_42.json').read_text())
    layer=old['selection']['selected_layers'];p=ROOT/f'optimized/artifacts/models/lightgcn/seed_42_layer_{layer}.pt'
    weights[str(p.relative_to(ROOT))]=sha(p)
    assert sha(p)==old['trials'][str(layer)]['checkpoint_sha256']
    from ..frozen_model_validation import trial_directory
    validated=(trial_directory('lightgcn',42)/'best.pt').resolve()
    evidence=json.loads((ROOT/'optimized/reports/rec_baselines/frozen_model_validation_20261011.json').read_text())
    weights[str(validated.relative_to(ROOT))]=sha(validated)
    assert sha(validated)==evidence['runs']['lightgcn_42']['checkpoint_sha256']
    for name in sources:
        dest=directory/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/name).read_bytes())
    facts={str(p.relative_to(ROOT)):sha(p) for p in [DATA/'train.parquet',DATA/'eval_test.parquet',DATA/'items.parquet',
          DATA/'id_map.json',ROOT/'original/data/raw/upstream_movie/movies.ftr',ROOT/'original/data/raw/metadata_audit/ml-10m/movies.dat']}
    manifest=json.loads((ROOT/'original/reproduction/resource_manifest.json').read_text())
    for resource in manifest['resources']:
        path=ROOT/'original'/resource['path']
        assert sha(path)==resource['sha256']
        facts[str(path.relative_to(ROOT))]=resource['sha256']
    for path in (ROOT/'original/reproduction/resource_manifest.json',DATA/'split_manifest.json',
                 ROOT/'optimized/AgenticRec/configs/lightgcn.yaml',
                 ROOT/'optimized/AgenticRec/configs/recommendation_search_selected_20261010.json'):
        facts[str(path.relative_to(ROOT))]=sha(path)
    protocol={'status':'FROZEN','commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'systems':SYSTEMS,'model':'deepseek-flash','max_output_tokens':1024,'temperature':0,'thinking':'disabled','network_retries':0,
              'files_sha256':{name:sha(directory/name) for name in paths},'sources_sha256':sources,'weights_sha256':weights,
              'data_sha256':facts,'old_no_content_model_seed':42,'main_feasible_episodes':300,'quality_users':180,
              'quality_scope':'Previously Exposed Test; public anonymous history only; not paper/algorithm superiority',
              'upstream_commit':'0959ecb05b0794748426e73e6efc1b6b35ec433d'}
    write_new(directory/'protocol.json',protocol)
    return {'status':'FROZEN','sources':len(sources),'weights':len(weights),'protocol_sha256':sha(directory/'protocol.json')}


def verify(directory):
    p=json.loads((directory/'protocol.json').read_text())
    for name,digest in p['files_sha256'].items():assert sha(directory/name)==digest,name
    for field in ('sources_sha256','weights_sha256','data_sha256'):
        for name,digest in p[field].items():assert sha(ROOT/name)==digest,name
    return p


def run(directory,phase):
    torch.set_num_threads(2)
    if phase=='formal':verify(directory)
    dest=directory/(phase+'_run');dest.mkdir(exist_ok=False)
    journal=MatchedJournal(read_api_key(),dest);worker=None;rows=[];failure=None
    catalog_blob=json.loads((directory/'catalog.json').read_text());catalog={int(k):v for k,v in catalog_blob['items'].items()}
    pub=json.loads((directory/f'{phase}_public.json').read_text());private=json.loads((directory/f'{phase}_private.json').read_text())
    truth=json.loads((directory/'truth_private.json').read_text());parser=AnswerIdentity(truth)
    try:
        balance=journal.guard.refresh()
        if journal.guard.should_stop():raise RuntimeError(journal.guard.reason)
        old=HistoricalBest(journal,balance)
        base=FusionPipeline.from_frozen('lightgcn',42,mode='rrf',profile='validated-20261011').retriever
        c1=FrontierSystem('C1',journal,catalog,base,MODELS)
        worker=OriginalWorker(dest,journal)
        write_new(dest/'started.json',{'original_ready':worker.ready,'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                                      'actual_max_output_tokens':1024,'protocol_sha256':sha(directory/'protocol.json') if phase=='formal' else None})
        for n,(public,hidden) in enumerate(zip(pub,private)):
            assert public['episode_id']==hidden['episode_id']
            order=SYSTEMS[n%3:]+SYSTEMS[:n%3]
            for system in order:
                if journal.guard.should_stop():raise RuntimeError(journal.guard.reason)
                turns=[];previous=[]
                for number,lab in enumerate(hidden['labels']):
                    if journal.guard.should_stop():raise RuntimeError(journal.guard.reason)
                    journal.context={'system':system,'episode_id':public['episode_id'],'turn':number+1}
                    before=len(journal.records);clock=time.perf_counter()
                    try:
                        if system=='original':
                            result=worker.turn(public,number)
                            result.update(parser.extract(result['answer'],[catalog_blob['original_ids'].get(str(i)) for i in result['mapped_catalog_ids']]))
                            result['tool_calls']=len(result['tool_trace'])
                        elif system=='historical_best':
                            result=old.turn(public,number)
                            result['raw_ids']=[catalog_blob['optimized_ids'].get(str(i)) for i in result['mapped_optimized_ids']]
                        else:result=c1.turn(public,number)
                    except (ValueError,TypeError,KeyError) as error:
                        result={'status':'EXECUTION_FAILURE','error_type':type(error).__name__,'error':str(error),
                                'answer':'','raw_ids':[],'extraction_verified':False,'tool_calls':0}
                    result['end_to_end_seconds']=time.perf_counter()-clock
                    calls=journal.records[before:];result['requests']=len(calls)
                    for field,usage in [('input_tokens','prompt_tokens'),('output_tokens','completion_tokens')]:
                        result[field]=sum(c['usage'][usage] for c in calls) if all(c['usage'] for c in calls) else None
                    result['scores']=score(result,lab,truth,previous);previous=[i for i in result['raw_ids'] if i is not None];turns.append(result)
                row={'episode_id':public['episode_id'],'system':system,'category':hidden['category'],
                     'scenario':hidden['scenario'],'history_n':hidden.get('history_n'),'turns':turns,
                     'strict_success':all(t['scores']['strict_success'] for t in turns)}
                append(dest/'episodes.jsonl',row);rows.append(row)
                print(json.dumps({'stage':'episode','phase':phase,'completed':len(rows),'planned':len(pub)*3,
                                  'episode':row['episode_id'],'system':system,'success':row['strict_success'],'requests':len(journal.records)}),flush=True)
        journal.guard.refresh()
    except Exception as error:
        failure={'error_type':type(error).__name__,'reason':str(error).replace(journal._key,'[REDACTED]')}
        write_new(dest/'failure.json',failure)
    finally:
        if worker is not None:worker.close()
    value={'status':'COMPLETED' if failure is None and len(rows)==len(pub)*3 else 'BLOCKED','blocker':failure,
           'episode_executions':len(rows),'planned_executions':len(pub)*3,'generation_requests':len(journal.records)}
    write_new(dest/'execution.json',value);print(json.dumps(value),flush=True)
    return value


def summarize(rows,provider):
    systems={}
    for system in SYSTEMS:
        own=[r for r in rows if r['system']==system];main=[r for r in own if r['category']!='unanswerable']
        turns=[t for r in own for t in r['turns']];calls=[c for c in provider if c['system']==system]
        quality=[r for r in own if r['category']=='session_quality'];refusal=[r['turns'][0]['scores'] for r in own if r['category']=='unanswerable']
        def qmean(items,metric):return sum(r['turns'][0]['scores'][metric] for r in items)/len(items) if items else None
        elapsed=[sum(t['end_to_end_seconds'] for t in r['turns']) for r in own]
        systems[system]={'completed':len(own),'main_n':len(main),'main_successes':sum(r['strict_success'] for r in main),
                         'strict_success':sum(r['strict_success'] for r in main)/300 if len(main)==300 else None,
                         'quality_n':len(quality),'quality':{m:qmean(quality,m) for m in ('hit','recall','ndcg','mrr')},
                         'quality_by_history':{str(h):{'n':sum(r['history_n']==h for r in quality),**{m:qmean([r for r in quality if r['history_n']==h],m) for m in ('hit','ndcg')}} for h in (3,5,10)},
                         'categories':{cat:{'n':sum(r['category']==cat for r in own),'success':sum(r['strict_success'] for r in own if r['category']==cat)} for cat in ('constraints','session_quality','feedback','unanswerable')},
                         'feedback_by_turns':{str(k):{'n':sum(len(r['turns'])==k for r in own if r['category']=='feedback'),
                                              'success':sum(r['strict_success'] for r in own if r['category']=='feedback' and len(r['turns'])==k)} for k in (2,3)},
                         'refusal':{m:sum(r[m] for r in refusal)/len(refusal) if refusal else None for m in ('safe_non_delivery','correct_refusal','explanatory_refusal')},
                         'constraint_precision':sum(t['scores']['constraint_precision'] for r in main for t in r['turns'])/sum(len(r['turns']) for r in main) if main else None,
                         'identity_errors':sum(t['scores']['identity_errors'] for t in turns),
                         'quantity_wrong_turns':sum(not t['scores']['quantity_correct'] for r in main for t in r['turns']),
                         'duplicate_items':sum(t['scores']['duplicate_count'] for t in turns),
                         'requests':len(calls),'input_tokens':sum(c['usage']['prompt_tokens'] for c in calls) if all(c['usage'] for c in calls) else None,
                         'output_tokens':sum(c['usage']['completion_tokens'] for c in calls) if all(c['usage'] for c in calls) else None,
                         'latency_p50_seconds':quantile(elapsed,.5),'latency_p95_seconds':quantile(elapsed,.95),
                         'failure_statuses':dict(Counter(t['status'] for r in main for t in r['turns'] if not t['scores']['strict_success']))}
    paired=defaultdict(dict)
    for r in rows:paired[r['episode_id']][r['system']]=r
    comparisons={}
    for baseline in ('original','historical_best'):
        complete=[v for v in paired.values() if 'C1' in v and baseline in v and v['C1']['category']!='unanswerable']
        rng=random.Random(42)
        delta=[int(v['C1']['strict_success'])-int(v[baseline]['strict_success']) for v in complete]
        wins=delta.count(1);losses=delta.count(-1);n=wins+losses
        p=min(1.,2*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/(2**n)) if n else 1.
        v={'strict_success':dict(bootstrap(delta,rng),n=len(delta)),
           'discordance':{'wins':wins,'losses':losses,'p':p},'by_category':{}}
        for cat in ('constraints','session_quality','feedback'):
            items=[x for x in complete if x['C1']['category']==cat]
            v['by_category'][cat]=dict(bootstrap([int(x['C1']['strict_success'])-int(x[baseline]['strict_success']) for x in items],rng),n=len(items))
        q=[x for x in complete if x['C1']['category']=='session_quality']
        for m in ('hit','recall','ndcg','mrr'):
            v[m]=dict(bootstrap([x['C1']['turns'][0]['scores'][m]-x[baseline]['turns'][0]['scores'][m] for x in q],rng),n=len(q))
        comparisons['C1-minus-'+baseline]=v
    prior=0.
    for i,(p,name) in enumerate(sorted((v['discordance']['p'],k) for k,v in comparisons.items())):
        prior=max(prior,min(1.,p*(2-i)));comparisons[name]['discordance']['holm_p']=prior
    return {'systems':systems,'paired':comparisons,'actual_invoice_cost_cny':None}


def report(directory):
    protocol=verify(directory);dest=directory/'formal_run'
    execution=json.loads((dest/'execution.json').read_text())
    if execution['status']!='COMPLETED':raise ValueError('Incomplete batch; preserve partial evidence')
    rows=[json.loads(s) for s in (dest/'episodes.jsonl').read_text().splitlines()]
    labels={c['episode_id']:c for c in json.loads((directory/'formal_private.json').read_text())}
    truth=json.loads((directory/'truth_private.json').read_text());catalog=json.loads((directory/'catalog.json').read_text())
    answer_parser=AnswerIdentity(truth)
    assert len(rows)==960 and len({(r['system'],r['episode_id']) for r in rows})==960
    spec=importlib.util.spec_from_file_location('three_way_independent',ROOT/'optimized/AgenticRec/scripts/audit_frontier_validation.py')
    independent=importlib.util.module_from_spec(spec);spec.loader.exec_module(independent)
    checked=0
    for r in rows:
        private=labels[r['episode_id']];assert len(r['turns'])==len(private['labels'])
        previous=[]
        for t,lab in zip(r['turns'],private['labels']):
            assert score(t,lab,truth,previous)==t['scores']
            if lab['feasible']:
                copy=deepcopy(lab)
                if copy.get('exclude_previous'):copy['excluded_raw_ids']+=previous
                for key,value in independent.independent_scores(t,copy,truth).items():
                    assert math.isclose(float(t['scores'][key]),float(value),abs_tol=1e-12)
            if r['system']=='C1' and t.get('mapped_optimized_ids'):
                ids=t['mapped_optimized_ids'];items=catalog['items']
                assert t['raw_ids']==[items[str(i)]['raw_id'] for i in ids]
                assert t['answer']=='\n'.join(f'{n}. {items[str(i)]["title"]} ({items[str(i)]["year"]})' for n,i in enumerate(ids,1))
            elif r['system']=='historical_best':
                assert t['raw_ids']==[catalog['optimized_ids'].get(str(i)) for i in t.get('mapped_optimized_ids',[])]
            elif r['system']=='original':
                backed=[catalog['original_ids'].get(str(i)) for i in t.get('mapped_catalog_ids',[])]
                assert answer_parser.extract(t['answer'],backed)['raw_ids']==t['raw_ids']
            previous=[i for i in t['raw_ids'] if i is not None];checked+=1
        assert r['strict_success']==all(t['scores']['strict_success'] for t in r['turns'])
    latest={};counts=Counter()
    for line in (dest/'provider.jsonl').read_text().splitlines():
        e=json.loads(line);latest[e['request_id']]=e;counts[(e['request_id'],e['status'])]+=1
        req=e['request'];assert req['max_tokens']==1024 and req['temperature']==0 and req['thinking']=={'type':'disabled'}
        assert req['model']=='deepseek-flash'
        assert not any(k in json.dumps(req) for k in ('relevant_raw_ids','evaluator_raw_user_id','formal_private.json','eval_test.parquet'))
    assert all(counts[(k,'STARTED')]==1 and e['status'] in ('COMPLETED','FAILED') for k,e in latest.items())
    by_turn=defaultdict(list)
    for e in latest.values():by_turn[e['system'],e['episode_id'],e['turn']].append(e)
    for r in rows:
        for n,t in enumerate(r['turns'],1):
            calls=by_turn[r['system'],r['episode_id'],n];assert len(calls)==t['requests']
            known=all(c['usage'] for c in calls)
            assert (sum(c['usage']['prompt_tokens'] for c in calls) if known else None)==t['input_tokens']
            assert (sum(c['usage']['completion_tokens'] for c in calls) if known else None)==t['output_tokens']
    assert len(latest)==sum(t['requests'] for r in rows for t in r['turns']) and checked==1200
    result=summarize(rows,list(latest.values()))
    balances=[json.loads(s) for s in (dest/'balance.jsonl').read_text().splitlines()]
    first=next(r for r in balances[0]['balance']['balance_infos'] if r['currency']=='CNY')
    last=next(r for r in balances[-1]['balance']['balance_infos'] if r['currency']=='CNY')
    verification={'status':'VERIFIED','task_executions':len(rows),'turns_recomputed':checked,
                  'requests':len(latest),'all_usage_reconciled':True,'unresolved_intents':0,
                  'provider_failed_attempts':sum(e['status']=='FAILED' for e in latest.values()),
                  'unknown_usage_attempts':sum(not e['usage'] for e in latest.values()),
                  'frozen_hashes_verified':True,'protocol_sha256':sha(directory/'protocol.json'),
                  'episodes_sha256':sha(dest/'episodes.jsonl'),'provider_sha256':sha(dest/'provider.jsonl'),
                  'observed_formal_balance_decrease_cny':str(Decimal(first['total_balance'])-Decimal(last['total_balance'])),
                  'source_commit':protocol['commit'],'private_targets_field_scan':'PASS','actual_invoice_cost_cny':None}
    write_new(directory/'verification.json',verification)
    public=ROOT/'optimized/reports/three_way_comparison_20261013.json'
    write_new(public,{'status':'COMPLETED','summary':result,'verification':verification})
    markdown_report(directory,result,verification)
    return verification


def markdown_report(directory,summary,verification):
    lines=['# C1、历史最佳消融版、微软运行版同场比较','',
           '2026-10-13。320同题任务×3实际系统，960次执行/1200轮，300个可行任务为严格成功主分母；180质量用户、60条件、60反馈、20不可满足任务。全部完成并逐条核验。','',
           '| 版本 | 严格成功/300 | Hit@10/180用户 | NDCG@10 | MRR@10 | 请求 | 总token | 任务p50/p95秒 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for k in SYSTEMS:
        v=summary['systems'][k];q=v['quality']
        tokens=format(v['input_tokens']+v['output_tokens'],',') if v['input_tokens'] is not None and v['output_tokens'] is not None else 'UNKNOWN'
        lines.append(f'| {NAMES[k]} | {v["strict_success"]:.2%} ({v["main_successes"]}/300) | {q["hit"]:.2%} | {q["ndcg"]:.6f} | {q["mrr"]:.6f} | {v["requests"]} | {tokens} | {v["latency_p50_seconds"]:.2f}/{v["latency_p95_seconds"]:.2f} |')
    lines+=['','严格成功是每轮完整按条件交付，不要求命中未来喜欢；Hit是前10部至少命中一个未来4/5星合格目标用户占比。全失败保留分母，不是线上喜欢概率。','',
            '| 版本 | 条件/60通过 | 偏好交付/180通过 | 两轮/40通过 | 三轮/20通过 | 平均约束精度 | 身份未知项 | 数量错误轮 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
    for k in SYSTEMS:
        v=summary['systems'][k]
        lines.append(f'| {NAMES[k]} | {v["categories"]["constraints"]["success"]} | {v["categories"]["session_quality"]["success"]} | {v["feedback_by_turns"]["2"]["success"]} | {v["feedback_by_turns"]["3"]["success"]} | {v["constraint_precision"]:.2%} | {v["identity_errors"]} | {v["quantity_wrong_turns"]} |')
    lines+=['','| 版本 | 3部历史Hit（60用户） | 5部历史Hit（60用户） | 10部历史Hit（60用户） | 正确拒绝/20 | 给出拒绝解释/20 |',
            '|---|---:|---:|---:|---:|---:|']
    for k in SYSTEMS:
        v=summary['systems'][k];h=v['quality_by_history'];r=v['refusal']
        lines.append(f'| {NAMES[k]} | {h["3"]["hit"]:.2%} | {h["5"]["hit"]:.2%} | {h["10"]["hit"]:.2%} | {r["correct_refusal"]:.2%} | {r["explanatory_refusal"]:.2%} |')
    lines+=['','不可满足任务不加入300题主成功率。正确拒绝允许明确错误/冲突/无候选结构状态或回答明确说明无匹配；普通执行失败/空回答不自动算正确拒绝。C1可用eligible_count=0的已验证约束失败状态，但其空回答不算给出解释。','',
            '| C1对照差值 | 严格成功百分点/95%区间 | Holm p | Hit百分点/95%区间 | NDCG差值/95%区间 |',
            '|---|---|---:|---|---|']
    def ci(v,scale=1):return f'{v["mean"]*scale:+.4f} [{v["ci95"][0]*scale:+.4f}, {v["ci95"][1]*scale:+.4f}]'
    for k,v in summary['paired'].items():
        lines.append(f'| {k} | {ci(v["strict_success"],100)} | {v["discordance"]["holm_p"]:.3g} | {ci(v["hit"],100)} | {ci(v["ndcg"])} |')
    lines+=['','每个独立任务/质量用户配对bootstrap10000次；条件/反馈/偏好分层区间与Recall/MRR见JSON。主指标对两基线另做精确discordance二项检验/Holm校正。合成条件模板仍相关，不把所有轮次视作独立用户，也不以区间外推任意自然语言准确率。','',
            '## 版本、费用与边界','',
            '历史版本是真实T19 no_content推荐器、原模型seed42和原AgentLoop工具契约；相同自然语言输入通过对照接线，历史状态只取本系统真正执行并验证的请求。并非上轮B1匿名RRF，也不是重新训练的替代。86%是此前150任务/三seed的结果，这次300可行任务/单seed重新实跑，不能要求重复86%。匿名公开历史模式不能衡量已知用户完整GCN能力。',
            '原版是固定App+原资源zero-demo/无reflection/DeepSeek兼容运行配置；不是原论文GPT4条件或原系统能力上限。C1未修上批继承年份增量失败，没有按正式结果改业务/评分。三者各用真实目录；C1来源确认目录/元数据是系统差异，不能全部归到算法。',
            '三个版本自然语言消息、公开电影标题和年份相同。优化两端沿用其工具接口所需的公开标题→本地item_id关联；原App仍自行调用原检索工具识别标题，不传不兼容的优化ID。身份识别/接线能力仍是系统差异，因此不是纯算法对照。',
            '同一厂商/model/temperature0/非思考/实际1024输出上限/零网络重试；C1及原worker内部512请求仅传输边界统一为1024，实际请求记录可核验。校准9例与正式分离，版本序列轮换，串行预热；缓存及时段影响延迟，不称线上QPS。工具粒度不同，不横比工具阶段数。',
            '历史MovieLens test已暴露；新240用户与此前两批和T19开发/正式用户隔离，但不是新数据集盲测。原权重行映射/训练仍未知，因此本报告支持这些实际系统配置在此批任务的对照，不能称无泄漏算法提升或超越作者论文。',
            f'正式{verification["requests"]}次真实请求完成账目核对，provider失败{verification["provider_failed_attempts"]}、未知usage {verification["unknown_usage_attempts"]}次；未知费用/usage不假设为0，含未知usage的系统token总量为UNKNOWN。观察正式余额窗口减少{verification["observed_formal_balance_decrease_cny"]}元，独立账单未查，不能当每版实际费用。校准请求另存execution.json。输入/源码/权重SHA通过，执行源码commit `{verification["source_commit"]}`。',
            '原始预测、请求/响应、余额、用户标签和源快照仅留忽略的 `optimized/artifacts/comparison/three_way_20261013/`；新训练/下载均0，原文件不覆盖。','',
            '## 核验与运行','',
            '依次运行 `python -m agenticrec.evaluation.three_way_comparison prepare|calibrate|freeze|formal|report --directory optimized/artifacts/comparison/<fresh_directory>`，正式目录拒绝覆盖；未来目标从不进入planner。协议、专项测试、校准、正式逐条评分/身份/provider对账保留。',
            '本轮有限三版本比较完成，停止新增付费调用；实际结论按上述同场差值报告，不能拼接历史最优点估计。','']
    path=ROOT/'optimized/reports/three_way_comparison_20261013.md'
    with path.open('x',encoding='utf-8',newline='\n') as stream:stream.write('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','calibrate','freeze','formal','report'))
    parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args();directory=args.directory.resolve()
    if not directory.is_relative_to((ROOT/'optimized/artifacts/comparison').resolve()):raise ValueError('Private output outside workspace')
    operation={'prepare':prepare,'calibrate':lambda d:run(d,'calibration'),'freeze':freeze,'formal':lambda d:run(d,'formal'),'report':report}
    value=operation[args.command](directory)
    print(json.dumps(value,indent=2),flush=True)
    return 2 if value.get('status')=='BLOCKED' else 0


if __name__=='__main__':raise SystemExit(main())
