"""Rebuild all formal metrics from predictions; refuse to overwrite reports."""
from collections import Counter
import json
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET

from agenticrec.data import digest
from agenticrec.evaluation.metrics import evaluate_cohort, score_ranking
from agenticrec.frozen_model_validation import average_rows, clustered_pair, strata
from agenticrec.hard_negative_training import BASE,ROOT,code_identity,plan_and_groups
from agenticrec.lightgcn_diagnostic import validation_only_files
from agenticrec.training import load_frozen_cohort,load_train_contract


def test_count(path):
    suite=ET.parse(path).getroot().find('testsuite')
    if int(suite.get('errors')) or int(suite.get('failures')):raise ValueError('Required tests failed')
    return int(suite.get('tests'))-int(suite.get('skipped',0))


def main():
    plan_path=ROOT/'AgenticRec/configs/lightgcn_hard_negatives_20261011.json'
    plan,groups=plan_and_groups(plan_path)
    summary=json.loads((BASE/'audit/summary.json').read_text(encoding='utf-8'))
    if summary['code_sha256']!=code_identity() or summary['plan_sha256']!=digest(plan_path) or (
            summary['selection_frozen_sha256']!=digest(BASE/'selection_frozen.json')):
        raise ValueError('Report source or frozen selection changed')
    checked,epochs,steps,dirty,commits=0,0,0,[],set()
    raw={}; originals={}
    with validation_only_files():
        contract=load_train_contract(); cohort=load_frozen_cohort('valid',ROOT/'artifacts/data_manifest.json')
        cohort_groups={'selection':groups['fit']+groups['select'],'audit':groups['audit'],'all_valid':cohort['user_ids']}
        popularity=Counter(i for _,i in contract[4]); hot=set(sorted(popularity,key=lambda i:(-popularity[i],i))[:100])
        for trial in plan['trials']:
            for seed in plan['seeds']:
                name=trial['name']; directory=BASE/f'seed_{seed}'/name
                original=json.loads((directory/'result.json').read_text(encoding='utf-8'))
                originals[name,seed]=original; run=summary['runs'][f'{name}_{seed}']
                if (digest(directory/'best.pt')!=run['checkpoint_sha256'] or
                        digest(directory/'result.json')!=run['source_result_sha256'] or
                        original['identity']['code_sha256']!=summary['code_sha256'] or
                        original['identity']!=json.loads((directory/'identity.json').read_text(encoding='utf-8'))):
                    raise ValueError('Checkpoint/result/source/identity mismatch')
                commits.add(original['identity']['git_commit']); epochs+=original['actual_epochs'];steps+=original['optimizer_steps']
                if original['identity']['git_dirty']:dirty.append(f'{name}_{seed}')
                for group,users in cohort_groups.items():
                    path=BASE/'audit'/f'{name}_{seed}_{group}_predictions.json'
                    if digest(path)!=run[group+'_predictions_sha256']:raise ValueError('Prediction SHA changed')
                    rows=json.loads(path.read_text(encoding='utf-8'))
                    if set(rows)!={str(u) for u in users if cohort['relevant_by_user'][str(u)]}:raise ValueError('Denominator changed')
                    for u,row in rows.items():
                        scores=score_ranking(row['prediction'],cohort['relevant_by_user'][u],cohort['candidate_ids'],10,cohort['seen_by_user'][u])
                        for field in ('recall','hit','mrr','ndcg'):
                            if scores[field]!=row[field]:raise ValueError('Per-user metric changed')
                            checked+=1
                    measured=evaluate_cohort(sorted(users),{u:cohort['relevant_by_user'][str(u)] for u in users},
                        {int(u):r['prediction'] for u,r in rows.items()},cohort['candidate_ids'],10,
                        {u:cohort['seen_by_user'][str(u)] for u in users})
                    if measured!=run['metrics'][group]:raise ValueError('Aggregate mismatch')
                    items=[i for r in rows.values() for i in r['prediction']]
                    diversity={'distinct_recommended_items':len(set(items)),'train_hot100_slot_share':sum(i in hot for i in items)/len(items)}
                    if diversity!=run['diversity'][group]:raise ValueError('Diversity changed')
                    raw[name,seed,group]=rows
        for group in cohort_groups:
            averaged={t['name']:average_rows([raw[t['name'],s,group] for s in plan['seeds']]) for t in plan['trials']}
            for comparison,value in summary['paired_ndcg'][group].items():
                a,b=comparison.split('_minus_')
                if clustered_pair(averaged[a],averaged[b],'ndcg')!=value:raise ValueError('Paired CI changed')
            for name,rows in averaged.items():
                if strata(rows,contract[3])!=summary['activity_strata'][group][name]:raise ValueError('Strata changed')
                for field in ('recall','hit','mrr','ndcg'):
                    values=[summary['runs'][f'{name}_{s}']['metrics'][group][field] for s in plan['seeds']]
                    if {'mean':statistics.mean(values),'std':statistics.stdev(values)}!=summary['aggregate'][group][name][field]:raise ValueError('Mean/std changed')
    protected=json.loads((BASE/'preservation_before.json').read_text(encoding='utf-8'))
    if any(digest(ROOT/p)!=sha for p,sha in protected.items()):raise ValueError('Historical evidence changed')
    locked=json.loads((BASE/'locked_reload.json').read_text(encoding='utf-8'))
    if dirty or locked['status']!='VERIFIED':raise ValueError('Formal source or locked reload not verified')
    full=test_count(BASE/'full_tests.xml'); focused=test_count(BASE/'locked_tests.xml')
    verdict={}
    for prefix in ('gcn','bpr'):
        pair=summary['paired_ndcg']['audit'][f'{prefix}_mixed25_minus_{prefix}_uniform']
        verdict[prefix]='POSITIVE DEVELOPMENT EVIDENCE; independent quality NOT EVALUATED' if pair['ci95'][0]>0 else 'NO VERIFIED GAIN'
    public={**summary,'conclusion':verdict,'actual_epochs':epochs,'optimizer_steps':steps,
            'verification':{'raw_metric_values_rebuilt':checked,'historical_files_preserved':len(protected),
                            'full_tests_passed':full,'locked_tests_passed':focused,'locked_reload':locked,
                            'training_commits':sorted(commits),'dirty_training_runs':dirty,
                            'report_script_sha256':digest(Path(__file__))}}
    lines=['# LG-HN01：混合困难负样本真实训练（2026-10-11）','',
           '仅改变未知负样本选择：uniform与25%最高分池+75%随机；GCN/BPR均等预算。原上游/数据划分/指标/服务默认不改，低分和时间监督未加入。', '',
           f'12次真实训练，实际{epochs}epoch/{steps}次更新；四条件×seed7/42/2026。每epoch462887训练三元组，227次更新，最多32epoch/patience8，batch2048/lr0.001/reg0.0001。GCN1层64维、BPR32维沿用各自强配置，维度不同不宣称纯结构效率。', '',
           '384人用于早停与seed42选择，冻结后补全其他seed，97人最后做开发审计。此前valid已暴露，这不是独立测试。全量valid为5351暖用户，其中481人有合格目标、4870人无目标单列；无目标不作为Hit分母。', '',
           '## 所有条件（主NDCG@10）', '',
           '| 方法 | 选择组NDCG均值 | 审计NDCG均值±标准差 | 审计Hit@10 |',
           '|---|---:|---:|---:|']
    for trial in plan['trials']:
        name=trial['name']; a=summary['aggregate']['audit'][name]; s=summary['aggregate']['selection'][name]
        lines.append(f"| {name} | {s['ndcg']['mean']:.6f} | {a['ndcg']['mean']:.6f} ± {a['ndcg']['std']:.6f} | {100*a['hit']['mean']:.2f}% |")
    lines += ['', f"seed42冻结选择：{json.dumps(summary['selected'],sort_keys=True)}。不根据审计更换赢家，不扩大比例。", '',
              '## 配对差异与消融', '', '用户内先平均三个seed，再bootstrap2000/seed42；探索性95%区间未多重校正。uniform是移除困难选择的单变量消融。', '']
    for comparison,pair in summary['paired_ndcg']['audit'].items():
        lines.append(f"- {comparison}：NDCG差{pair['difference']:+.6f}，区间[{pair['ci95'][0]:+.6f}, {pair['ci95'][1]:+.6f}]。")
    lines += ['',f"结论：GCN {verdict['gcn']}；BPR {verdict['bpr']}。独立推荐质量NOT EVALUATED，不能写微软论文或线上提升。默认模型没有替换。",'',
              '## 训练信号及公平性', '',
              '每seed的初始化、正边顺序、8候选池与随机掩码逐epoch核验哈希一致，所有条件均评分候选池。最终仅一个负片参与原BPR数据loss和ego正则；选择detach不反传。候选互异、排除所有train评分与padding，不查看未来喜欢标签。', '',
              'Torch候选流与独立正边shuffle不同于旧共用Python随机流，因此重新训练匹配uniform基线；本次对照不是旧checkpoint原样重放。训练并行时间仅记录开销，不作为孤立性能实验。', '',
              '| 方法 | 最佳epoch处容易题比例（seed均值） | 实际困难选择比例 |',
              '|---|---:|---:|']
    for trial in plan['trials']:
        name=trial['name']; best=[originals[name,s]['curve'][originals[name,s]['best_epoch']-1] for s in plan['seeds']]
        lines.append(f"| {name} | {100*statistics.mean(r['easy_fraction'] for r in best):.2f}% | {100*statistics.mean(r['actual_hard_fraction'] for r in best):.2f}% |")
    lines += ['', '训练信号强弱不是验证精度；未知高分电影可能是真喜欢，困难负例风险尚不能用未来标签修正。所有条件、活动度分层和热门集中度保留在JSON。', '',
              '## 验证', '',
              f'{checked}项原始用户指标/均值/区间/分层/热门集中度重算；12个checkpoint重载与选择指标一致；训练commits {sorted(commits)}，dirty runs {dirty}。完整{full}项及既有锁环境{focused}专项通过，锁环境12个真实模型重载指标完全一致，{len(protected)}历史文件SHA不变。既有环境/共享数据，不是全新checkout重训。', '',
              '完整工程测试可能使用历史评测fixture；本轮训练/审计/报告没有打开test标签。API0、下载0，未改旧权重。', '',
              '## 复现', '', '```powershell',
              '& .\\AgenticRec\\.venv-t20\\Scripts\\python.exe AgenticRec/scripts/run_hard_negative_study.py',
              '& .\\AgenticRec\\.venv-t20\\Scripts\\python.exe AgenticRec/scripts/report_hard_negative_study.py',
              '```', '', '需要原冻结数据及开发分组。每个run拒绝覆盖；runner只恢复身份和SHA完全一致的成功结果，部分失败需要另行记录处理。报告仅首次创建或完全一致重建。后续显式低分监督需单独登记，不能据本次audit继续调参。','']
    outputs={'reports/rec_baselines/lightgcn_hard_negatives_20261011.json':json.dumps(public,sort_keys=True,indent=2,allow_nan=False)+'\n',
             'reports/rec_baselines/lightgcn_hard_negatives_20261011.md':'\n'.join(lines)}
    for relative,text in outputs.items():
        path=ROOT/relative; data=text.encode('utf-8')
        if path.exists():
            if path.read_bytes()!=data:raise ValueError('Refuse to replace measured report')
        else:path.write_bytes(data)
    print(json.dumps({'status':'VERIFIED','epochs':epochs,'raw_metrics_rebuilt':checked,'conclusion':verdict}))


if __name__=='__main__':main()
