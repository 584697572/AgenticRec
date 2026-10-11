"""Rebuild upgrade reports from immutable raw predictions; makes zero API calls."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import statistics

from agenticrec.data import digest
from agenticrec.evaluation.metrics import score_ranking, FIELDS
from agenticrec.frozen_model_validation import clustered_pair
from agenticrec.fusion_experiment import user_means, reconstruct
from agenticrec.fusion_pipeline import predict_snapshot
from agenticrec.retrieval.adaptive import weighted_fusion
from agenticrec.training import ROOT


def stable_write(path,value):
    text=value if isinstance(value,str) else json.dumps(value,indent=2,sort_keys=True)+'\n'
    path=Path(path)
    if path.exists():
        if path.read_text(encoding='utf-8')!=text: raise ValueError('Refuse non-identical report replacement: '+str(path))
    else: path.write_text(text,encoding='utf-8')


def verify_records(directory):
    completion=json.loads((directory/'completion.json').read_text(encoding='utf-8'))
    if completion['status']!='COMPLETED': raise ValueError('Incomplete run')
    for name,sha in completion['files'].items():
        if digest(directory/name)!=sha: raise ValueError('Run output hash changed: '+name)
    records=json.loads((directory/'predictions.json').read_text(encoding='utf-8'))
    checked=0
    for row in records:
        request,snapshot=reconstruct(row)
        for name,ids in row['predictions'].items():
            if len(ids)!=len(set(ids)) or not set(ids)<=set(snapshot['eligible_ids']):
                raise ValueError('Prediction violates ID/seen/hard gate')
            score=score_ranking(ids,row['relevant'],snapshot['eligible_ids'],10) if row['relevant'] else None
            if score!=row['metrics'][name]: raise ValueError('Raw metric does not rebuild exactly')
            checked+=1
    return records,checked


def add_matched_control(records):
    """Diagnostic source-only ablation, no new fitting or test selection.

    Same RRF final ranker, same popularity weight and same candidate caps.
    Needed because primary no_content/pop.1 and rrf/pop1 differ in weights.
    """
    for r in records:
        request,snap=reconstruct(r)
        rankings={s:ids for s,ids in snap['rankings'].items() if s!='content'}
        fused=weighted_fusion(rankings,{s:1 for s in rankings},universe=snap['eligible_ids'])
        ids=fused.candidate_ids[:10]
        r['predictions']['matched_rrf_no_content']=ids
        r['metrics']['matched_rrf_no_content']=score_ranking(ids,r['relevant'],snap['eligible_ids'],10) if r['relevant'] else None


def aggregated(records):
    names=[n for n in records[0]['metrics'] if not n.startswith('trial_')]
    output={}
    for name in names:
        by_user={f:user_means(records,name,f) for f in FIELDS}
        latencies=sorted(r['latency_seconds'][name] for r in records if name in r['latency_seconds'])
        outputs=sum(len(r['predictions'][name]) for r in records)
        expected=sum(min(10,len(r['snapshot']['eligible_ids'])) for r in records)
        output[name]={'users':len(by_user['ndcg']),'queries':len(records),
            'queries_without_relevance':sum(r['metrics'][name] is None for r in records),
            **{f:statistics.mean(v.values()) if v else None for f,v in by_user.items()},
            'coverage':len({i for r in records for i in r['predictions'][name]}),
            'item_validity':1 if outputs else 0,'constraint_precision':1 if outputs else 0,
            'fill_of_feasible_capacity':outputs/expected if expected else None,
            'empty_outputs':sum(not r['predictions'][name] for r in records),
            'component_fusion_rank_latency_p50_seconds':statistics.median(latencies) if latencies else None,
            'component_fusion_rank_latency_p95_seconds':latencies[min(len(latencies)-1,int(.95*len(latencies)))] if latencies else None}
    return output


def paired(records,a,b,field):
    if a=='learned_mean':
        seeds=[user_means(records,f'learned_{s}_full',field) for s in (7,42,2026)]
        first={u:{field:statistics.mean(v[u] for v in seeds)} for u in seeds[0]}
    else: first={u:{field:v} for u,v in user_means(records,a,field).items()}
    second={u:{field:v} for u,v in user_means(records,b,field).items()}
    return clustered_pair(first,second,field)


def source_analysis(records):
    quality=defaultdict(list);old_quality=defaultdict(list)
    losses=[];negative_changes=0
    for row in records:
        rel=set(row['relevant'])
        if rel:
            for s,ids in row['snapshot']['rankings'].items(): quality[s].append(len(set(ids)&rel)/len(rel))
            ranks=row['legacy_diagnostics']['candidate_rows']
            for s in row['legacy_diagnostics']['source_counts']:
                ids={r['item_id'] for r in ranks if s in r['source_ranks']}
                old_quality[s].append(len(ids&rel)/len(rel))
        diag=row['diagnostics'];cf=next((ids for s,ids in row['snapshot']['rankings'].items() if s.startswith('cf_')),[])
        kept={r['item_id'] for r in diag['candidate_rows'] if r['retained']}
        losses.append(len(set(cf[:10])-kept))
        if row['state']=='local_negative' and row['predictions']['legacy_full']!=row['predictions']['negative_penalty_off']:
            negative_changes+=1
    return {'positive_source_recall_at_100':{s:statistics.mean(v) for s,v in quality.items()},
            'legacy_source_recall_at_100':{s:statistics.mean(v) for s,v in old_quality.items()},
            'rrf_cf_top10_candidates_lost_total':sum(losses),
            'rrf_cf_top10_loss_queries':sum(x>0 for x in losses),
            'local_negative_queries_changed_by_removing_genre_penalty':negative_changes,
            'candidate_trace_scope':'Raw source scores/ranks, overlap, cap retention and final flags for every recalled candidate'}


def t19_group_audit():
    from agenticrec.evaluation.episodes import load_public_episodes,load_private_episodes,EpisodeAttempt,score_episode
    from agenticrec.evaluation.runner import _load_events
    from agenticrec.ranking.constraints import Catalog
    manifest=json.loads((ROOT/'artifacts/eval_manifest.json').read_text(encoding='utf-8'))
    for k in ('test_public','test_private'):
        v=manifest['files'][k]
        if digest(ROOT/v['path'])!=v['sha256']: raise ValueError('Historical test changed')
    public={r.episode_id:r for r in load_public_episodes(ROOT/manifest['files']['test_public']['path'])}
    private={r.episode_id:r for r in load_private_episodes(ROOT/manifest['files']['test_private']['path'])}
    catalog=Catalog.from_frozen(); scores={}; provenance={}
    for name in ('O','A','no_content'):
        scores[name]={i:[] for i in public}
        for seed in (7,42,2026):
            p=ROOT/f'artifacts/runs/t19/live_20261010/{name}/seed_{seed}/predictions.jsonl'
            provenance[str(p.relative_to(ROOT))]=digest(p)
            events=_load_events(p); completed=[r for r in events if r['event']=='COMPLETED']
            if len(completed)!=len(public): raise ValueError('Incomplete historical journal')
            for event in completed:
                v=dict(event['attempt']);v['item_ids']=tuple(v['item_ids'])
                attempt=EpisodeAttempt(**v)
                score=score_episode(public[attempt.episode_id],private[attempt.episode_id],attempt,catalog)
                scores[name][attempt.episode_id].append(float(score.strict_success))
    comparisons={}
    for a,b in (('O','A'),('O','no_content')):
        group=defaultdict(list)
        for episode,p in public.items():
            group[p.template_group_id].append(statistics.mean(scores[a][episode])-statistics.mean(scores[b][episode]))
        values=[(sum(xs),len(xs)) for xs in group.values()]
        rng=random.Random(42);boot=[]
        for _ in range(2000):
            chosen=rng.choices(values,k=len(values));boot.append(sum(v for v,n in chosen)/sum(n for v,n in chosen))
        boot.sort()
        comparisons[a+'_minus_'+b]={'difference':sum(v for v,n in values)/sum(n for v,n in values),
            'template_cluster_ci95':[boot[50],boot[1950]],'template_groups':len(values),'distinct_episodes':len(public),
            'bootstrap_samples':2000,'seed':42,'limit':'Small number of related templates; not general population inference'}
    return {'historical_test':'Previously Exposed Test','comparisons':comparisons,'raw_sha256':provenance,
            'new_api_requests':0,'parser_confound':'F/U1 interface differences remain; O/A shared-interface comparison retained',
            'replanning_quality':'NOT EVALUATED: original formal runs had zero replans'}


def main():
    run=ROOT/'artifacts/runs/fusion_research/fusion_20261011'
    dev,checked_dev=verify_records(run/'develop'); external,checked_external=verify_records(run/'external')
    add_matched_control(dev);add_matched_control(external)
    supplemental=ROOT/'artifacts/runs/upgrade_20261011/matched_control_predictions.json'
    stable_write(supplemental,{split:[{'user':r['user'],'state':r['state'],
        'prediction':r['predictions']['matched_rrf_no_content'],
        'metrics':r['metrics']['matched_rrf_no_content']} for r in records]
        for split,records in [('development',dev),('external',external)]})
    groups=json.loads((ROOT/'data/eval_private/fusion-development-20261011/user_groups.json').read_text(encoding='utf-8'))
    audit=[r for r in dev if r['user'] in groups['audit']]
    selected=json.loads((run/'develop/selection_frozen.json').read_text(encoding='utf-8'))
    public={'schema_version':1,'date':'2026-10-11','scope':'Structured anonymous train-seed recommendation, no new LLM/planning benchmark',
        'metrics':'Mean per user after averaging states with relevance. Empty valid predictions count zero; no-relevance states counted separately.',
        'development':{g:aggregated([r for r in dev if r['user'] in ids]) for g,ids in groups.items()},
        'external':aggregated(external),'selected_fixed_weight':selected['fixed_weight'],
        'fit_gate_positive_labels':selected['fit_label_counts'],
        'selected_gates':{key:{'regularization':value['regularization'],'select_ndcg':value['select_ndcg'],
            'gate_sha256':hashlib.sha256(json.dumps(value['gate'],sort_keys=True).encode()).hexdigest()}
            for key,value in selected['gates'].items()},
        'source_diagnostics':{'development_audit':source_analysis(audit),'external':source_analysis(external)},
        'paired':{},'strata':{},'remote_api_requests':0,
        'verification':{'raw_metrics_rebuilt_exact':checked_dev+checked_external,
                        'formal_predictions_sha256':{s:digest(run/s/'predictions.json') for s in ('develop','external')},
                        'method_freeze_sha256':digest(run/'develop/selection_frozen.json'),
                        'matched_control_predictions_sha256':digest(supplemental),
                        'report_generator_sha256':digest(Path(__file__)),
                        'primary_method_source_sha256':selected['code_sha256']},
        'matched_control_status':'Supplemental single-variable diagnostic, no training/search/adoption selection; not a new blind confirmatory protocol',
        'timing_boundary':'Fusion/rank component on precomputed sources; excludes recall/loading/LLM. Ordered cached timings are descriptive, not causal speed gain.',
        'limits':['ML1M audit is development audit after historical model tuning, not blind model test',
                  'External release has only 51 relevant warm users; 694 no-relevance users separate',
                  'No guarantee of independent real persons across MovieLens releases',
                  'No external valid tuning; source graph rebuilt only from external train',
                  'No new live Agent planning, token-cost or online CTR claim']}
    for split,records in [('development_audit',audit),('external',external)]:
        public['paired'][split]={}
        for a,b in [('learned_mean','no_content'),('learned_mean','legacy_no_content'),('rrf','legacy_full'),
                    ('rrf','legacy_no_content'),('rrf','matched_rrf_no_content'),('negative_penalty_off','legacy_full'),
                    ('recall_only_off','legacy_full'),('rrf','no_content')]:
            public['paired'][split][a+'_minus_'+b]={field:paired(records,a,b,field) for field in ('ndcg','hit')}
        public['strata'][split]={}
        for state in ('positive_seeds','local_negative','genre_constraint'):
            public['strata'][split][state]=aggregated([r for r in records if r['state']==state])
        for name,lo,hi in [('activity_under100',0,100),('activity_100_299',100,300),('activity_ge300',300,float('inf'))]:
            subset=[r for r in records if lo<=r['train_activity']<hi]
            public['strata'][split][name]=aggregated(subset) if subset else {'users':0}
    gate_names=[f'learned_{s}_full' for s in (7,42,2026)]
    public['learned_seed_summary']={split:{f:{'mean':statistics.mean(table[n][f] for n in gate_names),
        'std':statistics.stdev(table[n][f] for n in gate_names)} for f in ('ndcg','hit')}
        for split,table in [('development_audit',public['development']['audit']),('external',public['external'])]}
    public['gate_diagnostics']={split:{'queries':len(records),'content_enabled_by_seed':{
        str(seed):sum(r['gate_decisions'][f'learned_{seed}_full']['weights'].get('content',0)>0 for r in records)
        for seed in (7,42,2026)}} for split,records in [('development_audit',audit),('external',external)]}
    public['core_gate_conclusion']='NO VERIFIED GAIN' if public['paired']['external']['learned_mean_minus_no_content']['ndcg']['ci95'][0]<=0 else 'External gain vs no_content; assess all stronger baselines before adoption'
    before=json.loads((ROOT/'artifacts/runs/upgrade_20261011/preservation_before.json').read_text(encoding='utf-8'))
    changed=[p for p,sha in before.items() if digest(ROOT/p)!=sha]
    if changed: raise ValueError('Historical evidence modified: '+str(changed))
    public['verification']['preserved_historical_files']=len(before)
    stable_write(ROOT/'reports/fusion_research_20261011.json',public)
    stable_write(ROOT/'reports/t19_template_cluster_audit_20261011.json',t19_group_audit())
    lines=['# 候选融合研究结果（2026-10-11）','',
        '真实结构化组件评测；没有新LLM调用。主学习门控结论：**'+public['core_gate_conclusion']+'**。',
        '','## 实验及分母','',
        'ML1M valid用户288/96/97分fit/select/audit；每用户三个训练期状态，先在用户内平均，不当三个独立用户。外部100K先冻结、门控随后冻结再评测；51相关暖用户、694无相关用户。历史ML1M/交互test是Previously Exposed Test。',
        '','| 方法 | 开发审计 NDCG@10 | 外部 NDCG@10 | 外部 Hit@10 |',
        '|---|---:|---:|---:|']
    for name in ('popularity','legacy_full','legacy_no_content','recall_only_off','negative_penalty_off','no_content','matched_rrf_no_content','fixed_best','rrf','conditional','learned_42_full','learned_42_no_quality','learned_42_no_protection'):
        a=public['development']['audit'][name];e=public['external'][name]
        lines.append(f"| {name} | {a['ndcg']:.6f} | {e['ndcg']:.6f} | {e['hit']:.6f} |")
    lines+=['','主四项指标/三seed均值标准差、配对用户CI、状态/活跃度层及来源诊断均在机器报告。匹配RRF无内容为后补单变量诊断，不训练或按它选择参数；不能称另一次盲测。',
        '','## 验证与结论','',f"全部 {public['verification']['raw_metrics_rebuilt_exact']} 个原始预测指标精确重算；{len(before)} 个选定历史文件SHA不变。",'']
    for key in ('learned_mean_minus_no_content','rrf_minus_legacy_no_content','rrf_minus_matched_rrf_no_content','rrf_minus_legacy_full','negative_penalty_off_minus_legacy_full','recall_only_off_minus_legacy_full'):
        v=public['paired']['external'][key]['ndcg'];lines.append(f"- {key}：NDCG差值 {v['difference']:.6f}，95%用户聚类区间 {v['ci95']}。")
    lines+=['','学习门控若退化为禁用内容或区间跨0，保留负结果，不默认上线。去源质量/候选保护消融即使无差异也完整保留，不声称每个组件有贡献。旧genre排序与一致融合排序的差异，不能冒称LightGCN或LLM规划提升。',
        '','## 工程及限制','',
        '新FusionPipeline是显式实验注入入口，旧CLI/模型默认保持稳定。门控10个可学习参数、每个候选四正则×三个seed、最多200步；原始候选/来源分数/重叠/保留率/最终标记可在本地diagnose查看。',
        '','指标不是满意度或CTR。外部用户少、元数据粗、未新测LLM解析；component latency只含预计算来源后的融合/排序，不能写在线QPS或纯算法加速。失败和计数修复均保留。',
        '','## 复现','',
        '```powershell',
        '& AgenticRec/.venv-t20/Scripts/python.exe -m agenticrec.fusion_experiment develop --run-id <new_id>',
        '& AgenticRec/.venv-t20/Scripts/python.exe -m agenticrec.fusion_experiment external --run-id <same_id>',
        '& AgenticRec/.venv-t20/Scripts/python.exe AgenticRec/scripts/report_upgrade_research.py',
        '```','',
        '训练/数据协议见optimization_proposal与fusion_protocol_20261011.json。正式run要求源码已提交，重复目录拒绝覆盖；报告脚本重建本轮固定run且只允许字节一致写回。']
    stable_write(ROOT/'reports/fusion_research_20261011.md','\n'.join(lines)+'\n')
    print(json.dumps({'raw_metrics_rebuilt':checked_dev+checked_external,'historical_files_unchanged':len(before),
                      'core_gate_conclusion':public['core_gate_conclusion']}))


if __name__=='__main__': main()
