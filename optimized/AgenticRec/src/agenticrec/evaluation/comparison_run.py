"""Run actual original and optimized systems on a frozen shared text workload.

No retraining, target injection, failed-case replay or old-result overwriting.
Private provider requests/responses and evaluator labels stay in ignored artifacts.
"""
import argparse
from collections import defaultdict
from decimal import Decimal
import importlib.util
import json
import os
from pathlib import Path
import queue
import random
import subprocess
import sys
import threading
import time

import torch

from .comparison_cases import ROOT, prepare, sha, write_new
from .comparison_scoring import AnswerIdentity, score_turn
from .cost_limits import PER_REQUEST_COST_CNY
from .system_factory import build_benchmark_agent_loop
from ..adapters.account_balance import BalanceClient, BalanceGuard, BalanceStop
from ..adapters.llm import ChatAdapter
from ..adapters.openai_compatible import OpenAICompatibleTransport
from ..agent.router import RoutingRequest
from ..config import LLMBudget
from ..fusion_pipeline import FusionPipeline
from ..runtime.errors import LLMTransportError
from ..runtime.secrets import read_api_key


def append(path,value):
    with Path(path).open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(value,ensure_ascii=True,sort_keys=True)+'\n')
        stream.flush()
        os.fsync(stream.fileno())


class HTTPJournal:
    """Shared no-retry provider boundary; durable intent precedes transmission."""
    def __init__(self,key,directory):
        self._key=key
        self.directory=directory
        self.context={}
        self.records=[]
        self.guard=BalanceGuard(BalanceClient(key),refresh_every=20,
            observer=lambda row:append(directory/'balance.jsonl',row))

    def sender(self,url,headers,body,timeout):
        if url!='https://api.deepseek.com/chat/completions':
            raise ValueError('Non-allowlisted API URL')
        payload=json.loads(body)
        if (payload.get('model')!='deepseek-flash' or payload.get('max_tokens')!=512
                or payload.get('thinking')!={'type':'disabled'} or payload.get('temperature')!=0
                or payload.get('stream',False) or len(body)>100000):
            raise ValueError('Provider request outside frozen runtime settings')
        if self.guard.should_stop():
            raise BalanceStop(self.guard.reason)
        number=len(self.records)+1
        record={'request_id':number,**self.context,'request':payload,
                'status':'STARTED','http_status':None,'usage':None,'reply':None}
        self.records.append(record)
        append(self.directory/'provider.jsonl',record)
        clock=time.perf_counter()
        try:
            def send():
                reply=OpenAICompatibleTransport._stdlib_send(url,headers,body,timeout)
                if reply.status_code>=400:
                    raise LLMTransportError('Provider error',status_code=reply.status_code,retryable=False)
                return reply
            reply=self.guard.call(send)
            text=reply.body.decode('utf-8').replace(self._key,'[REDACTED]')
            decoded=json.loads(text)
            record.update(status='COMPLETED',http_status=reply.status_code,
                          usage=decoded.get('usage'),reply=decoded)
            return reply
        except Exception as error:
            record.update(status='FAILED',error_type=type(error).__name__,
                          http_status=getattr(error,'status_code',None))
            raise
        finally:
            record['elapsed_seconds']=time.perf_counter()-clock
            append(self.directory/'provider.jsonl',record)

    def original(self,body):
        payload=json.dumps(body).encode('utf-8')
        try:
            reply=self.sender('https://api.deepseek.com/chat/completions',
                {'Authorization':'Bearer '+self._key,'Content-Type':'application/json'},payload,60)
            return {'status_code':reply.status_code,'content':reply.body.decode('utf-8').replace(self._key,'[REDACTED]')}
        except Exception as error:
            return {'transport_error':type(error).__name__}


class OriginalWorker:
    def __init__(self,directory,journal):
        self.journal=journal
        self.events=queue.Queue()
        # Reuse the existing keyless offline resource environment allowlist.
        scripts=ROOT/'original/reproduction/scripts'
        spec=importlib.util.spec_from_file_location('comparison_legacy_environment',scripts/'live_app_single_turn.py')
        module=importlib.util.module_from_spec(spec)
        sys.path.insert(0,str(scripts))
        spec.loader.exec_module(module)
        self.resource_evidence=module.resource_checks()
        self.stderr=(directory/'original_worker.stderr.log').open('wb')
        self.process=subprocess.Popen([str(ROOT/'original/reproduction/.venv-legacy/Scripts/python.exe'),
             str(scripts/'comparison_app_worker.py')],cwd=ROOT/'original',env=module.worker_environment(),
             stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.stderr,text=True,encoding='utf-8',bufsize=1)
        def reader():
            for line in self.process.stdout:
                try:
                    self.events.put(json.loads(line))
                except ValueError:
                    self.events.put({'event':'invalid_wire','text':line})
            self.events.put({'event':'worker_eof'})
        self.reader=threading.Thread(target=reader,daemon=True)
        self.reader.start()
        self.ready=self.receive('ready',600)

    def receive(self,expected,timeout):
        start=time.monotonic()
        heartbeat=start
        while time.monotonic()-start<timeout:
            try:
                event=self.events.get(timeout=1)
            except queue.Empty:
                if time.monotonic()-heartbeat>=30:
                    print(json.dumps({'stage':'waiting_original_worker','expected':expected,
                                      'seconds':round(time.monotonic()-start)}),flush=True)
                    heartbeat=time.monotonic()
                continue
            if event['event']==expected:
                return event
            if event['event']=='request':
                if event['episode_id']!=self.journal.context['episode_id'] or event['turn']!=self.journal.context['turn']:
                    raise ValueError('Worker request context drift')
                self.process.stdin.write(json.dumps(self.journal.original(event['body']))+'\n')
                self.process.stdin.flush()
            elif event['event']=='progress':
                print(json.dumps(event),flush=True)
            else:
                raise RuntimeError('Unexpected original worker event: '+event['event'])
        raise TimeoutError('Original worker deadline exceeded')

    def turn(self,public,index):
        command={'command':'turn','episode_id':public['episode_id'],'turn':index+1,
                 'message':public['messages'][index],'reset':index==0,'seed':42}
        self.process.stdin.write(json.dumps(command)+'\n')
        self.process.stdin.flush()
        return self.receive('turn_result',180)

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write('{"command":"stop"}\n')
                self.process.stdin.flush()
                self.process.wait(timeout=10)
            except Exception:
                self.process.kill()
                self.process.wait()
        self.stderr.close()


class OptimizedSystem:
    def __init__(self,journal,initial_balance):
        torch.set_num_threads(2)
        started=time.perf_counter()
        self.pipeline=FusionPipeline.from_frozen('lightgcn',42,mode='rrf',profile='validated-20261011')
        transport=OpenAICompatibleTransport(api_base='https://api.deepseek.com',api_key=journal._key,sender=journal.sender)
        # Existing ledger schema requires numeric limits. The finite workload cap
        # is an engineering bound, not new authorization; balance policy governs.
        config=LLMBudget(allow_paid_api=True,provider='deepseek',model_id='deepseek-flash',
                         api_request_cap=1000,max_output_tokens=512,money_budget=1000*PER_REQUEST_COST_CNY,
                         max_retries=0,per_request_timeout_seconds=60,round_deadline_seconds=180)
        adapter=ChatAdapter(transport,config)
        self.loop=build_benchmark_agent_loop('O',fixed_pipeline=self.pipeline,planner=adapter,
                                             planned_cost_ceiling=PER_REQUEST_COST_CNY)
        self.cold_start_seconds=time.perf_counter()-started
        self.history=[]

    def turn(self,public,index):
        if index==0:
            self.history=[]
        envelope={'mode':'text','message':public['messages'][index],
                  'user_id':None,'history_authorized':False,
                  'catalog_items':public['catalog_context'], 'conversation':self.history.copy()}
        request=RoutingRequest(f"{public['episode_id']}:{index+1}",envelope,
                               agent_need_score=1.0,needs_planning=True)
        report=self.loop.run(request).to_dict()
        rows=(report.get('response') or {}).get('recommendations',[])
        # This structured response is what the optimized application delivers;
        # no hidden LLM summarization cost or oracle evaluation post-filtering.
        answer='\n'.join(f"{n}. {r['title']}" for n,r in enumerate(rows,start=1))
        self.history.extend([{'role':'user','content':public['messages'][index]},
                             {'role':'assistant','content':answer}])
        return {'status':report['status'],'answer':answer,
                'mapped_optimized_ids':[r['item_id'] for r in rows],
                'tool_calls':report['tool_calls']+(1 if report['planner_calls']==0 and rows else 0),
                'trace':report,'elapsed_seconds':report['elapsed_ms']/1000}


def verify(directory):
    protocol=json.loads((directory/'protocol.json').read_text(encoding='utf-8'))
    for name,expected in protocol['files_sha256'].items():
        if sha(directory/name)!=expected:
            raise ValueError('Frozen evaluation input changed: '+name)
    for name,expected in protocol['sources_sha256'].items():
        if sha(ROOT/name)!=expected:
            raise ValueError('Frozen evaluation source changed: '+name)
    return protocol


def run(directory,phase,systems=('original','optimized')):
    directory=Path(directory).resolve()
    if not directory.is_relative_to((ROOT/'optimized/artifacts/comparison').resolve()):
        raise ValueError('Comparison outputs must stay private')
    protocol=verify(directory)
    run_dir=directory/(phase+'_run')
    run_dir.mkdir(exist_ok=False)
    write_new(run_dir/'started.json',{'protocol_sha256':sha(directory/'protocol.json'),
                                    'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    public=json.loads((directory/f'{phase}_public.json').read_text(encoding='utf-8'))
    labels=json.loads((directory/f'{phase}_private.json').read_text(encoding='utf-8'))
    assert [r['episode_id'] for r in public]==[r['episode_id'] for r in labels]
    identity=json.loads((directory/'identity.json').read_text(encoding='utf-8'))
    original_ids={int(k):v for k,v in identity['original_ids'].items()}
    optimized_ids={int(k):v for k,v in identity['optimized_ids'].items()}
    index=AnswerIdentity(identity['truth'])
    journal=HTTPJournal(read_api_key(),run_dir)
    worker=None
    records=[]
    failure=None
    try:
        balance=journal.guard.refresh()
        if journal.guard.should_stop():
            raise BalanceStop(journal.guard.reason)
        optimized=OptimizedSystem(journal,balance)
        if 'original' in systems:
            worker=OriginalWorker(run_dir,journal)
        write_new(run_dir/'warmup.json',{'original':worker.ready if worker else None,
                                      'original_resource_evidence':worker.resource_evidence if worker else None,
                                      'optimized_cold_start_seconds':optimized.cold_start_seconds})
        for episode_number,(pub,private) in enumerate(zip(public,labels)):
            for system in (['original','optimized'] if episode_number%2==0 else ['optimized','original']):
                if system not in systems:
                    continue
                if journal.guard.should_stop():
                    raise BalanceStop(journal.guard.reason)
                turn_records=[]
                for turn,label in enumerate(private['labels']):
                    journal.context={'system':system,'episode_id':pub['episode_id'],'turn':turn+1}
                    before=len(journal.records)
                    clock=time.perf_counter()
                    if system=='original':
                        result=worker.turn(pub,turn)
                        mapped=[original_ids.get(i) for i in result['mapped_catalog_ids']]
                        extracted=index.extract(result['answer'],mapped)
                        result.update(extracted)
                        result['tool_calls']=len(result['tool_trace'])
                    else:
                        result=optimized.turn(pub,turn)
                        result.update(raw_ids=[optimized_ids.get(i) for i in result['mapped_optimized_ids']],
                                      extraction_verified=True)
                    result['end_to_end_seconds']=time.perf_counter()-clock
                    calls=journal.records[before:]
                    result['requests']=len(calls)
                    result['input_tokens']=sum(r['usage']['prompt_tokens'] for r in calls) if all(r['usage'] for r in calls) else None
                    result['output_tokens']=sum(r['usage']['completion_tokens'] for r in calls) if all(r['usage'] for r in calls) else None
                    result['scores']=score_turn(result,label,identity['truth'])
                    turn_records.append(result)
                record={'episode_id':pub['episode_id'],'system':system,'category':private['category'],
                        'strict_success':all(t['scores']['strict_success'] for t in turn_records),
                        'turns':turn_records}
                append(run_dir/'episodes.jsonl',record)
                records.append(record)
                print(json.dumps({'stage':'episode_completed','phase':phase,'system':system,
                                  'episode_id':pub['episode_id'],'success':record['strict_success'],
                                  'completed':len(records),'total':len(public)*len(systems),
                                  'requests':len(journal.records)}),flush=True)
        journal.guard.refresh()
    except Exception as error:
        failure={'error_type':type(error).__name__,'reason':str(error).replace(journal._key,'[REDACTED]')}
        write_new(run_dir/'failure.json',failure)
        print(json.dumps({'stage':'stopped',**failure}),flush=True)
    finally:
        if worker is not None:
            worker.close()
    summary=summarize(records,public,journal.records,phase)
    summary['blocker']=failure
    summary['executed_systems']=list(systems)
    summary['status']='COMPLETED' if failure is None and len(records)==len(public)*len(systems) else 'BLOCKED'
    write_new(run_dir/'summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)
    return 0 if summary['status']=='COMPLETED' else 2


def summarize(records,public,provider,phase):
    summaries={}
    for system in ('original','optimized'):
        selected=[r for r in records if r['system']==system]
        turns=[t for r in selected for t in r['turns']]
        calls=[r for r in provider if r['system']==system]
        quality=[r['turns'][0]['scores'] for r in selected if r['category']=='session_quality']
        elapsed=[sum(t['end_to_end_seconds'] for t in r['turns']) for r in selected]
        summaries[system]={
            'completed_episodes':len(selected),'planned_episodes':len(public),
            'strict_success':sum(r['strict_success'] for r in selected)/len(public) if len(selected)==len(public) else None,
            'observed_partial_success':sum(r['strict_success'] for r in selected)/len(selected) if selected else None,
            'constraint_precision':sum(t['scores']['constraint_precision'] for t in turns)/len(turns) if turns else None,
            'categories':{c:{'n':sum(r['category']==c for r in selected),
                              'success':sum(r['strict_success'] for r in selected if r['category']==c)}
                          for c in sorted({r['category'] for r in selected})},
            'quality_users':len(quality),
            'quality_metrics':{m:sum(r[m] for r in quality)/len(quality) if quality else None for m in ('hit','recall','ndcg')},
            'remote_requests':len(calls),'http_successes':sum(r['status']=='COMPLETED' for r in calls),
            'usage_complete':all(r['usage'] for r in calls),
            'input_tokens':sum(r['usage']['prompt_tokens'] for r in calls) if all(r['usage'] for r in calls) else None,
            'output_tokens':sum(r['usage']['completion_tokens'] for r in calls) if all(r['usage'] for r in calls) else None,
            'tool_calls':sum(t['tool_calls'] for t in turns),
            'latency_p50_seconds':quantile(elapsed,.5) if elapsed else None,
            'latency_p95_seconds':quantile(elapsed,.95) if elapsed else None,
            'identity_errors':sum(t['scores']['identity_errors'] for t in turns),
            'unparsed_final_answers':sum(not t.get('extraction_verified',False) for t in turns),
        }
    intervals={}
    paired={r['episode_id']:{} for r in records}
    for r in records:
        paired[r['episode_id']][r['system']]=r
    full=[pair for pair in paired.values() if set(pair)=={'original','optimized'}]
    rng=random.Random(42)
    if full:
        delta=[int(p['optimized']['strict_success'])-int(p['original']['strict_success']) for p in full]
        intervals['strict_success_difference']={**bootstrap(delta,rng),'paired_episodes':len(full)}
        quality=[p for p in full if p['original']['category']=='session_quality']
        for metric in ('hit','recall','ndcg'):
            delta=[p['optimized']['turns'][0]['scores'][metric]-p['original']['turns'][0]['scores'][metric] for p in quality]
            if len(delta):
                intervals[metric+'_difference']={**bootstrap(delta,rng),'paired_users':len(delta)}
    return {'phase':phase,'scope':'actual two versions, synthetic tasks plus previously exposed anonymous session quality',
            'systems':summaries,'paired_intervals':intervals,
            'actual_invoice_cost_cny':None,'algorithm_only_gain':'NOT EVALUATED',
            'original_pretrained_row_semantics':'NOT VERIFIED',
            'provider_requests':len(provider),'completed_http':sum(r['status']=='COMPLETED' for r in provider)}


def quantile(values,q):
    ordered=sorted(values)
    position=(len(ordered)-1)*q
    low=int(position)
    high=min(low+1,len(ordered)-1)
    return ordered[low]+(ordered[high]-ordered[low])*(position-low)


def bootstrap(delta,rng):
    samples=[sum(rng.choices(delta,k=len(delta)))/len(delta) for _ in range(10000)]
    return {'mean':sum(delta)/len(delta),'ci95':[quantile(samples,.025),quantile(samples,.975)]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--phase',choices=('calibration','formal'))
    parser.add_argument('--systems',nargs='+',choices=('original','optimized'),default=['original','optimized'])
    args=parser.parse_args()
    if args.prepare:
        protocol=prepare(args.directory)
        print(json.dumps({k:protocol[k] for k in ('calibration_episodes','formal_episodes','formal_turns_per_system')},indent=2))
    if args.phase:
        if args.phase=='formal' and set(args.systems)!={'original','optimized'}:
            parser.error('Formal comparison must execute both actual systems')
        return run(args.directory,args.phase,args.systems)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
