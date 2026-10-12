"""Rebuild explicit-feedback study evidence from predictions and frozen weights."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import statistics

import torch

from agenticrec.data import digest
from agenticrec.evaluation.metrics import score_ranking, evaluate_cohort
from agenticrec.explicit_low_training import BASE, ROOT, code_identity, load_verified, plan_and_groups, load_train_contract, load_frozen_cohort, load_lows
from agenticrec.frozen_model_validation import average_rows, clustered_pair, strata
from agenticrec.lightgcn_diagnostic import validation_only_files
from agenticrec.lightgcn_optimization import subset_cohort
from agenticrec.recommendation_search import evaluate

PLAN=ROOT/'AgenticRec/configs/lightgcn_explicit_low_20261012.json'
FIELDS=('recall','hit','mrr','ndcg')


def exact(a,b,label):
    if a != b:
        raise ValueError('Evidence mismatch: '+label)


def fixed_train_fit(model,kind,samples):
    """Post-hoc mechanism check, reusing train-only pairs fixed before this study."""
    tuples=[]
    for row in samples:
        tuples.extend((row['user_id'],p,n) for p,n in zip(row['positive_ids'],row['low_ids']))
    users=torch.tensor([u for u,p,n in tuples]);pos=torch.tensor([p for u,p,n in tuples])
    neg=torch.tensor([n for u,p,n in tuples])
    with torch.no_grad():
        ue,ie=model.propagate() if kind=='lightgcn' else (model.users.weight[1:],model.items.weight[1:])
        margins=((ue[users-1]*ie[pos-1]).sum(1)-(ue[users-1]*ie[neg-1]).sum(1)).tolist()
    by_user={}
    for (u,p,n),margin in zip(tuples,margins):
        counts=by_user.setdefault(str(u),[0,0]);counts[0]+=int(margin<=0);counts[1]+=1
    rows={u:{'inversions':v[0],'pairs':v[1],'inversion_rate':v[0]/v[1]} for u,v in by_user.items()}
    return rows,{'users':len(rows),'pairs':len(tuples),
                 'user_macro_inversion_rate':statistics.mean(r['inversion_rate'] for r in rows.values())}


def verify(tag):
    plan,groups=plan_and_groups(PLAN)
    summary=json.loads((BASE/'audit/summary.json').read_text(encoding='utf-8'))
    frozen=json.loads((BASE/'selection_frozen.json').read_text(encoding='utf-8'))
    exact(summary['code_sha256'],code_identity(),'source')
    exact(summary['plan_sha256'],digest(PLAN),'plan')
    exact(summary['selection_frozen_sha256'],digest(BASE/'selection_frozen.json'),'selection freeze')
    names=[t['name'] for t in plan['trials']]
    preserved=json.loads((BASE/'preservation_before.json').read_text(encoding='utf-8'))
    for name,expected in preserved.items():
        exact(digest(ROOT/name),expected,'protected file '+name)
    all_rows={}; rebuilt=0; reloads=0; old_baselines={};fitting={}
    torch.set_num_threads(2)
    with validation_only_files():
        contract=load_train_contract()
        cohort=load_frozen_cohort('valid',ROOT/'artifacts/data_manifest.json')
        popularity=Counter(i for _,i in contract[4])
        sample_dir=ROOT/'artifacts/runs/lightgcn_training_diagnostics_20261011/diagnosis_deterministic'
        sample_source=json.loads((sample_dir/'result.json').read_text(encoding='utf-8'))
        exact(digest(sample_dir/'samples.json'),sample_source['samples_sha256'],'fixed train-only samples')
        samples=json.loads((sample_dir/'samples.json').read_text(encoding='utf-8'))
        lows=load_lows(contract);positives=set(contract[4])
        for row in samples:
            u=row['user_id']
            if any((u,i) not in positives for i in row['positive_ids']) or any(i not in lows[u] for i in row['low_ids']):
                raise ValueError('Diagnostic pair not in train')
        cohorts={'selection':groups['fit']+groups['select'],'audit':groups['audit'],'all_valid':cohort['user_ids']}
        for trial in plan['trials']:
            for seed in plan['seeds']:
                name=trial['name']; key=f'{name}_{seed}'
                directory=BASE/f'seed_{seed}'/name
                model,result=load_verified(directory,trial,contract,digest(PLAN),seed)
                reloads+=1
                fit_rows,fit_summary=fixed_train_fit(model,trial['model'],samples)
                fitting[key]=fit_summary
                fit_dir=BASE/'low_feedback_fit';fit_dir.mkdir(exist_ok=True)
                fit_path=fit_dir/f'{key}.json'
                serialized=(json.dumps(fit_rows,sort_keys=True,indent=2)+'\n').encode('utf-8')
                if fit_path.exists():exact(fit_path.read_bytes(),serialized,'locked train fit '+key)
                else:
                    with fit_path.open('xb') as f:f.write(serialized)
                exact(result['identity']['git_dirty'],False,'clean training start '+key)
                exact(summary['runs'][key]['source_result_sha256'],digest(directory/'result.json'),'training result '+key)
                for group,users in cohorts.items():
                    subset=subset_cohort(cohort,users)
                    path=BASE/'audit'/f'{name}_{seed}_{group}_predictions.json'
                    exact(digest(path),summary['runs'][key][group+'_predictions_sha256'],'prediction SHA '+key+group)
                    raw=json.loads(path.read_text(encoding='utf-8'))
                    expected_ids={str(u) for u in users if subset['relevant_by_user'][str(u)]}
                    exact(set(raw),expected_ids,'user coverage '+key+group)
                    for u,row in raw.items():
                        computed=score_ranking(row['prediction'],subset['relevant_by_user'][u],
                            subset['candidate_ids'],10,subset['seen_by_user'][u])
                        for field in FIELDS:
                            exact(computed[field],row[field],'raw '+key+group+u+field);rebuilt+=1
                    computed=evaluate_cohort(subset['user_ids'],
                        {u:subset['relevant_by_user'][str(u)] for u in users},
                        {int(u):r['prediction'] for u,r in raw.items()},subset['candidate_ids'],10,
                        {u:subset['seen_by_user'][str(u)] for u in users})
                    exact(computed,summary['runs'][key]['metrics'][group],'cohort '+key+group)
                    metrics,reloaded_rows,diversity=evaluate(model,subset,popularity)
                    exact(reloaded_rows,raw,'real checkpoint predictions '+key+group)
                    exact(metrics,computed,'real checkpoint metrics '+key+group)
                    exact(diversity,summary['runs'][key]['diversity'][group],'diversity '+key+group)
                    if group=='selection':
                        exact(metrics,result['selection_metrics'],'selected checkpoint '+key)
                    all_rows[name,seed,group]=raw
                if name.endswith('uniform'):
                    old=json.loads((ROOT/'artifacts/runs/lightgcn_hard_negatives_20261011'/f'seed_{seed}'/name/'result.json').read_text(encoding='utf-8'))
                    old_baselines[key]={'same_initial_state':old['identity']['initial_state_sha256']==result['identity']['initial_state_sha256'],
                        'same_epoch_selection_scores':[r['selection_ndcg'] for r in old['curve']]==[r['selection_ndcg'] for r in result['curve']],
                        'same_final_selection_metrics':old['selection_metrics']==result['selection_metrics']}
        comparisons=[('gcn_low25','gcn_uniform'),('bpr_low25','bpr_uniform'),
                     (frozen['selected']['lightgcn'],frozen['selected']['bpr'])]
        for group in cohorts:
            averaged={name:average_rows([all_rows[name,s,group] for s in plan['seeds']]) for name in names}
            for name in names:
                for field in FIELDS:
                    values=[summary['runs'][f'{name}_{s}']['metrics'][group][field] for s in plan['seeds']]
                    exact({'mean':statistics.mean(values),'std':statistics.stdev(values)},
                          summary['aggregate'][group][name][field],'aggregate '+group+name+field)
                exact(strata(averaged[name],contract[3]),summary['activity_strata'][group][name],'strata '+group+name)
            for a,b in comparisons:
                exact(clustered_pair(averaged[a],averaged[b],'ndcg'),summary['paired_ndcg'][group][a+'_minus_'+b],'CI '+group+a+b)
        for kind in ('gcn','bpr'):
            for seed in plan['seeds']:
                a=json.loads((BASE/f'seed_{seed}'/(kind+'_uniform')/'result.json').read_text(encoding='utf-8'))
                b=json.loads((BASE/f'seed_{seed}'/(kind+'_low25')/'result.json').read_text(encoding='utf-8'))
                exact(a['identity']['initial_state_sha256'],b['identity']['initial_state_sha256'],'matched init')
                for x,y in zip(a['curve'],b['curve']):exact(x['paired_stream_sha256'],y['paired_stream_sha256'],'matched samples')
    result={'status':'PASS','protected_files':len(preserved),'real_checkpoint_reloads':reloads,
            'raw_metric_values_rebuilt':rebuilt,'paired_random_streams_verified':6,
            'source_sha256':code_identity(),'plan_sha256':digest(PLAN),
            'old_uniform_replay_checks':old_baselines,'test_labels_read':False,
            'posthoc_train_feedback_fit':fitting,'posthoc_train_samples_sha256':digest(sample_dir/'samples.json')}
    path=BASE/f'verification_{tag}.json'
    with path.open('x',encoding='utf-8',newline='\n') as f:json.dump(result,f,sort_keys=True,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)
    return result


def test_count(path):
    raw=path.read_bytes()
    text=raw.decode('utf-16' if raw.startswith(b'\xff\xfe') else 'utf-8')
    if re.search(r'\b[1-9]\d* (?:failed|errors?)\b',text):raise ValueError('Failed tests: '+str(path))
    matches=re.findall(r'(\d+) passed',text)
    if not matches:raise ValueError('Missing test result: '+str(path))
    return int(matches[-1])


def uniform_parameters():
    """Report numerical replay limits against preserved prior matched controls."""
    checks={}
    for prefix in ('gcn','bpr'):
        for seed in (7,42,2026):
            old=ROOT/'artifacts/runs/lightgcn_hard_negatives_20261011'/f'seed_{seed}'/(prefix+'_uniform')
            new=BASE/f'seed_{seed}'/(prefix+'_uniform')
            prior=json.loads((old/'result.json').read_text(encoding='utf-8'))
            current=json.loads((new/'result.json').read_text(encoding='utf-8'))
            for directory,result in ((old,prior),(new,current)):
                exact(digest(directory/'best.pt'),result['checkpoint_sha256'],'uniform replay checkpoint')
            a=torch.load(old/'best.pt',map_location='cpu',weights_only=True)['state_dict']
            b=torch.load(new/'best.pt',map_location='cpu',weights_only=True)['state_dict']
            checks[f'{prefix}_{seed}']={
                'max_best_weight_difference':max(float((a[k]-b[k]).abs().max()) for k in a if a[k].is_floating_point()),
                'best_epoch_old':prior['best_epoch'],'best_epoch_new':current['best_epoch']}
    exact(checks,json.loads((BASE/'uniform_replay_parameter_check.json').read_text(encoding='utf-8')),'uniform replay diagnostics')
    return checks


def report():
    plan,_=plan_and_groups(PLAN)
    torch.set_num_threads(plan['torch_threads'])
    summary=json.loads((BASE/'audit/summary.json').read_text(encoding='utf-8'))
    checks={tag:json.loads((BASE/f'verification_{tag}.json').read_text(encoding='utf-8')) for tag in ('main','locked')}
    for value in checks.values():
        if value['status']!='PASS':raise ValueError('Verification incomplete')
    exits={tag:json.loads((BASE/f'exit_{tag}_exit.json').read_text(encoding='utf-8')) for tag in ('main','locked')}
    for tag,entry in exits.items():
        exact(entry['exit_code'],0,'actual Python verification exit '+tag)
        exact(json.loads((BASE/f'verification_exit_{tag}.json').read_text(encoding='utf-8')),checks[tag],'verification exit recheck '+tag)
    tests={tag:test_count(BASE/filename) for tag,filename in
           [('full','tests_full_final.txt'),('locked','tests_locked_final.txt')]}
    exact(json.loads((BASE/'tests_full_final_exit.json').read_text(encoding='utf-8'))['exit_code'],0,'full test process exit')
    totals={'actual_epochs':sum(r['actual_epochs'] for r in summary['runs'].values()),
            'optimizer_steps':sum(json.loads((BASE/f'seed_{seed}'/t['name']/'result.json').read_text(encoding='utf-8'))['optimizer_steps']
                                  for seed in plan['seeds'] for t in plan['trials'])}
    commits=sorted({json.loads((BASE/f'seed_{seed}'/t['name']/'result.json').read_text(encoding='utf-8'))['identity']['git_commit']
                     for seed in plan['seeds'] for t in plan['trials']})
    outcome={}
    for prefix in ('gcn','bpr'):
        a,b=prefix+'_low25',prefix+'_uniform'
        selection=summary['paired_ndcg']['selection'][a+'_minus_'+b]
        audit=summary['paired_ndcg']['audit'][a+'_minus_'+b]
        outcome[prefix]={'development_quality':'POSITIVE EXPLORATORY SIGNAL' if selection['ci95'][0]>0 and audit['ci95'][0]>0 else 'NO VERIFIED GAIN',
                         'selection_difference':selection,'audit_difference':audit,
                         'independent_quality':'NOT EVALUATED'}
    metadata=json.loads((BASE/'hetrec_metadata_read.json').read_text(encoding='utf-8'))
    exact(digest(BASE/'hetrec_readme.txt'),metadata['sha256'],'future source README')
    output={**summary,'verification':checks,'tests_passed':tests,'totals':totals,
            'training_commits':commits,'quality':outcome,'dataset_or_weight_downloads':0,
            'source_metadata_read':metadata,'serving_default_changed':False,
            'verification_process_exits':exits,'uniform_replay_parameters':uniform_parameters(),
            'verification_script_sha256':digest(Path(__file__)),
            'summary_sha256':digest(BASE/'audit/summary.json'),
            'external_protocol':'reports/lightgcn_explicit_low_protocol_20261012.md'}
    lines=['# LG-LF01：显式低分监督真实训练（2026-10-12）','',
           '单变量：训练低分1/2星的独立排序监督，权重0/.25；原未知采样、图、正则不变。强BPR同步训练，alpha0是移除低分监督的消融。','',
           f"12训练×三seed（7/42/2026），实际{totals['actual_epochs']} epoch / {totals['optimizer_steps']}更新。",
           '384人开发选择/早停，seed42按NDCG冻结，97人最后开发审计；全valid有481名相关暖用户，其余4870无相关不进Hit分母。两组早停epoch允许不同，共同上限32、patience8。','',
           '**已有valid已经暴露，本轮没有读取旧test；这不是新盲测。独立推荐质量 NOT EVALUATED，不声称微软论文或线上业务提升。**','',
           '| 方法 | 选择NDCG | 审计NDCG ± seed标准差 | 审计Hit@10 | 全valid NDCG | 全valid Hit@10 |',
           '|---|---:|---:|---:|---:|---:|']
    for trial in plan['trials']:
        name=trial['name'];a=summary['aggregate']['audit'][name];s=summary['aggregate']['selection'][name];v=summary['aggregate']['all_valid'][name]
        lines.append(f"| {name} | {s['ndcg']['mean']:.6f} | {a['ndcg']['mean']:.6f} ± {a['ndcg']['std']:.6f} | {a['hit']['mean']:.2%} | {v['ndcg']['mean']:.6f} | {v['hit']['mean']:.2%} |")
    lines.extend(['',f"seed42冻结选择：{json.dumps(summary['selected'],sort_keys=True)}；不按审计改赢家。",'',
                  '## 配对差异与消融','',
                  '在用户内平均三seed后2000次user bootstrap；95%区间未经多重校正，仅为开发探索。'])
    for group in ('selection','audit'):
        for name,entry in summary['paired_ndcg'][group].items():
            lines.append(f"- {group} {name}：差{entry['difference']:+.6f}，CI[{entry['ci95'][0]:+.6f}, {entry['ci95'][1]:+.6f}]。")
    lines.extend(['','## 训练机制诊断（事后，仅train）','',
                  '使用前轮已固定的正例—低分配对，对best checkpoint计算低分得分不低于正例的用户宏平均比例；这只说明拟合训练偏好，不代表未见电影推荐提升，也不用于改alpha或重新选模型。','',
                  '| 方法 | train偏好倒置比例（三seed平均） |','|---|---:|'])
    for trial in plan['trials']:
        name=trial['name']
        rate=statistics.mean(checks['main']['posthoc_train_feedback_fit'][f'{name}_{seed}']['user_macro_inversion_rate'] for seed in plan['seeds'])
        lines.append(f'| {name} | {rate:.2%} |')
    lines.extend(['','## 结论与验证','',
                  f"GCN {outcome['gcn']['development_quality']}；BPR {outcome['bpr']['development_quality']}。独立质量 NOT EVALUATED，默认模型不替换。",
                  f"完整工程{tests['full']}项、已有锁环境{tests['locked']}专项通过；两个环境各12checkpoint的原始预测/指标完全一致。",
                  f"重算{checks['main']['raw_metric_values_rebuilt']}项用户指标并验证聚合/区间/分层；6组初始化与采样哈希一致，{checks['main']['protected_files']}份旧文件/权重SHA不变。",
                  f"训练commits {commits}，所有训练启动Git clean。源码SHA {code_identity()}；plan SHA {digest(PLAN)}。",
                  '六组uniform最佳epoch与最终选择指标均与前轮一致；best权重最大差1.788e-6，不声称bit-identical重放。BPR seed42的非最佳第32轮NDCG有0.000166差异，原因未作因果验证，不用于选择新参数。',
                  '两个环境复用当前checkout和冻结数据，不是全新机器重训；CPU并发时间只记录成本，不是隔离速度基准。',
                  '初始缺模块失败及系统tmp无权限错误保留，工作区tmp测试通过；全套测试可使用历史评测fixture，本轮训练/审计/报告不读test。API0、数据包/权重下载0；另只读归档官方HetRec README 8105bytes，SHA及非商业许可记录保留。','',
                  '交付时先回归复现新目录缺少初始化，再修补运行目录与历史保护ledger自动创建；未重训或修改已完成算法。PowerShell将Torch stderr警告转成NativeCommandError的日志保留；直接捕获两个环境Python真实退出码均0、验证JSON完全一致。','',
                  '## 复现','',
                  '在已完成LG-D02/LG-HN01、拥有冻结ML1M数据/开发组/训练样本和已有环境的工作区运行；新checkout先按项目README重建前置数据与环境。初始化自动创建目录和保护ledger，现有结果仅允许校验后续跑，不覆盖。','',
                  '```powershell',
                  '& .\\AgenticRec\\.venv-t20\\Scripts\\python.exe .\\AgenticRec\\scripts\\run_explicit_low_study.py',
                  '& .\\AgenticRec\\.venv-t20\\Scripts\\python.exe .\\AgenticRec\\scripts\\verify_explicit_low_study.py --tag rerun',
                  '```','',
                  '未暴露HetRec2011确认协议已预登记；本次不因开发观测无限扩alpha或重复旧audit，不以该数据包尚未取得冒称外部结论。'])
    outputs={ROOT/'reports/rec_baselines/lightgcn_explicit_low_20261012.json':json.dumps(output,sort_keys=True,indent=2,allow_nan=False)+'\n',
             ROOT/'reports/rec_baselines/lightgcn_explicit_low_20261012.md':'\n'.join(lines)+'\n'}
    for path,value in outputs.items():
        raw=value.encode('utf-8')
        if path.exists():exact(path.read_bytes(),raw,'byte-identical report rebuild')
        else:
            with path.open('xb') as f:f.write(raw)
    print(json.dumps({'status':'PASS','quality':outcome,'totals':totals,'tests':tests}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag',default='main');parser.add_argument('--report',action='store_true')
    args=parser.parse_args()
    if args.report:report()
    else:
        if not re.fullmatch('[a-z0-9_]+',args.tag):raise ValueError('Unsafe verification tag')
        verify(args.tag)
