"""Small real-provider development check; never reads test evaluator targets."""
import json
import argparse
from pathlib import Path
from functools import partial
import torch

from agenticrec.adapters.account_balance import BalanceClient, BalanceGuard, GuardedTransport
from agenticrec.adapters.providers import build_live_chat_adapter
from agenticrec.adapters.upstream_bridge import UpstreamWorkerSession, run_upstream_plan
from agenticrec.adapters.model import RouteResult
from agenticrec.runtime.secrets import read_api_key
from agenticrec.config import LLMBudget
from agenticrec.pipeline import FixedRecommendationPipeline
from agenticrec.agent.text_parser import TextRequestParser
from agenticrec.evaluation.benchmark import ROOT, _sha256
from agenticrec.evaluation.episodes import load_public_episodes, load_private_episodes
from agenticrec.evaluation.system_executor import FeedbackSchedule, BenchmarkSystemExecutor
from agenticrec.evaluation.system_factory import build_benchmark_agent_loop
from agenticrec.evaluation.upstream_turn import UpstreamRebuiltTurn
from agenticrec.evaluation.live_benchmark import AuditedTransport, append_json, write_json
from agenticrec.evaluation.cost_limits import PER_REQUEST_COST_CNY


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', default='dev_preflight_20261010_v2')
    args = parser.parse_args()
    if not args.run_id.startswith('dev_preflight_') or not all(
        character.isalnum() or character == '_' for character in args.run_id
    ):
        raise ValueError('run-id must be a safe development preflight directory name')
    torch.set_num_threads(1)
    directory=ROOT/'artifacts/runs/t19'/args.run_id
    report_path=directory/'report.json'
    if report_path.exists():
        print(report_path.read_text(encoding='utf-8'))
        return 0
    directory.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'artifacts/eval_manifest.json').read_text(encoding='utf-8'))
    paths={name:ROOT/manifest['files'][name]['path'] for name in ('development_public','development_private')}
    for name,path in paths.items():
        if _sha256(path)!=manifest['files'][name]['sha256']:
            raise ValueError('development file hash mismatch')
    public=load_public_episodes(paths['development_public'])
    private=load_private_episodes(paths['development_private'])
    schedule=FeedbackSchedule.from_private_episodes(private,expected_episode_ids={e.episode_id for e in public})
    cases=[('F',next(e for e in public if e.layer=='text' and e.scenario_family=='explicit_filter' and not e.multi_turn)),
           ('A',next(e for e in public if e.layer=='structured' and e.scenario_family=='personalized' and not e.multi_turn)),
           ('U1',next(e for e in public if e.layer=='text' and e.scenario_family=='seed_similar' and not e.multi_turn)),
           ('O',next(e for e in public if e.layer=='text' and e.scenario_family=='exclusion_feedback' and e.multi_turn))]
    write_json(directory/'cases.json',[{'system':s,'episode':e.to_dict()} for s,e in cases],immutable=True)
    client=BalanceClient(read_api_key())
    guard=BalanceGuard(client,observer=lambda row:append_json(directory/'balance.jsonl',row))
    initial=guard.refresh()
    pipeline=FixedRecommendationPipeline.from_frozen('lightgcn',42)
    rows=[]
    for system,episode in cases:
        if guard.should_stop(): break
        config=LLMBudget(allow_paid_api=True,provider='deepseek',model_id='deepseek-flash',api_request_cap=16,
                         max_output_tokens=1024,money_budget=16*PER_REQUEST_COST_CNY,max_retries=0)
        adapter=build_live_chat_adapter(config,seed=42)
        audit=AuditedTransport(adapter.transport,directory/(system+'_llm.jsonl'))
        audit.current_episode=episode.episode_id
        adapter.transport=GuardedTransport(audit,guard)
        kwargs={'ledger':adapter.ledger,'planned_cost_ceiling':PER_REQUEST_COST_CNY,
                'feedback_source':schedule,'should_stop':lambda:guard.halted}
        session=None
        if system=='F':
            kwargs.update(fixed_pipeline=pipeline,text_parser=TextRequestParser(adapter))
        elif system=='U1':
            session=UpstreamWorkerSession()
            warm=RouteResult([1,2,3],[0.0,0.0,0.0],[1,2],'cold_start',None)
            run_upstream_plan(warm,{i:pipeline.retriever.catalog.items[i].title for i in (1,2,3)},top_k=2,session=session)
            kwargs['upstream_turn']=UpstreamRebuiltTurn(adapter,pipeline,planned_cost_ceiling=PER_REQUEST_COST_CNY,
                                                       bridge=partial(run_upstream_plan,session=session))
        else:
            kwargs.update(fixed_pipeline=pipeline,agent_loop=build_benchmark_agent_loop(system,
                fixed_pipeline=pipeline,planner=adapter,planned_cost_ceiling=PER_REQUEST_COST_CNY))
        executor=BenchmarkSystemExecutor(system,**kwargs)
        try:
            attempt=executor(episode)
        finally:
            write_json(directory/(system+'_ledger.json'),adapter.ledger.snapshot())
            if session: session.close()
        append_json(directory/'traces.jsonl',executor.last_trace)
        row={'system':system,'episode_id':episode.episode_id,'status':attempt.status,
             'requests':attempt.request_count,'input_tokens':attempt.input_tokens,'output_tokens':attempt.output_tokens,
             'tool_calls':attempt.tool_calls,'turns':attempt.turns}
        rows.append(row)
        print(json.dumps(row),flush=True)
        if attempt.status!='OK': break
    final=guard.refresh()
    report={'status':'VERIFIED' if len(rows)==4 and all(r['status']=='OK' for r in rows) else 'BLOCKED_PREFLIGHT',
            'scope':'development-only connection/schema/tool/state check; not quality metrics',
            'remote_generation_requests':guard.requests,'initial_balance':initial,'final_balance':final,
            'rows':rows,'stopped_reason':guard.reason,
            'actual_cost':None,'actual_cost_reason':'provider_bill_not_queried'}
    write_json(report_path,report)
    print(json.dumps(report),flush=True)
    return 0 if report['status']=='VERIFIED' else 3


if __name__=='__main__':
    raise SystemExit(main())
