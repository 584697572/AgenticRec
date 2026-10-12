"""Rebuild public aggregate search evidence; private predictions remain local."""
from collections import Counter
import json
from pathlib import Path
import random
import xml.etree.ElementTree as ET

from agenticrec.data import digest
from agenticrec.evaluation.metrics import score_ranking
from agenticrec.lightgcn_diagnostic import validation_only_files
from agenticrec.recommendation_search import validate_plan
from agenticrec.training import ROOT, load_frozen_cohort


def paired(base, treatment):
    if set(base) != set(treatment) or not base:
        raise ValueError('Paired users differ')
    output = {}
    for field in ('ndcg', 'hit', 'recall', 'mrr'):
        diffs = [treatment[u][field]-base[u][field] for u in sorted(base, key=int)]
        rng = random.Random(42)
        means = sorted(sum(rng.choices(diffs, k=len(diffs)))/len(diffs) for _ in range(2000))
        output[field] = {'mean': sum(diffs)/len(diffs), 'ci95': [means[49], means[1949]],
                         'paired_users': len(diffs), 'bootstrap_samples': 2000,
                         'scope': 'exploratory validation; no multiple-testing correction'}
    return output


def build():
    plan_path = ROOT/'AgenticRec/configs/recommendation_search_20261010.json'
    plan = validate_plan(json.loads(plan_path.read_text(encoding='utf-8')))
    base_dir = ROOT/'artifacts/runs/recommendation_search'/plan['run_id']
    results, predictions = {}, {}
    for trial in plan['trials']:
        out = base_dir/trial['name']
        result = json.loads((out/'result.json').read_text(encoding='utf-8'))
        if (result['trial'] != trial or result['identity']['plan_sha256'] != digest(plan_path)
                or result['status'] != 'VERIFIED' or result['remote_api_requests'] != 0
                or result['test_evaluation'] != 'NOT EVALUATED'):
            raise ValueError('Trial identity mismatch')
        if (digest(ROOT/result['checkpoint_path']) != result['checkpoint_sha256']
                or digest(out/'valid_predictions.json') != result['predictions_sha256']):
            raise ValueError('Trial artifact digest mismatch')
        if result['curve'] != [json.loads(line) for line in (out/'curve.jsonl').read_text().splitlines()]:
            raise ValueError('Raw training curve differs')
        results[trial['name']] = result
        predictions[trial['name']] = json.loads((out/'valid_predictions.json').read_text())
    if len({r['identity']['code_sha256'] for r in results.values()}) != 1:
        raise ValueError('Trial code differs')
    selected = {model: max((t['name'] for t in plan['trials'] if t['model'] == model),
                          key=lambda name: results[name]['valid_metrics']['ndcg'])
                for model in ('lightgcn', 'bpr')}
    with validation_only_files():
        valid = load_frozen_cohort('valid', ROOT/'artifacts/data_manifest.json')
        old_path = ROOT/'artifacts/runs/lightgcn_diagnostics/lightgcn_diagnostic_layer1_20261010/smaller_batch_valid_predictions.json'
        old_rows = json.loads(old_path.read_text())
        previous = {u: score_ranking(row['prediction'], valid['relevant_by_user'][u],
                     valid['candidate_ids'], 10, valid['seen_by_user'][u]) for u, row in old_rows.items()}
        # Independent regeneration of every public aggregate from private rankings.
        for name, rows in predictions.items():
            if set(rows) != set(previous):
                raise ValueError('Frozen evaluation denominator differs')
            computed = Counter()
            for u, row in rows.items():
                score = score_ranking(row['prediction'], valid['relevant_by_user'][u],
                                      valid['candidate_ids'], 10, valid['seen_by_user'][u])
                if any(row[field] != value for field, value in score.items()):
                    raise ValueError('Per-user metrics differ from ranking')
                computed.update(score)
            if any(abs(computed[f]/len(rows)-results[name]['valid_metrics'][f]) > 1e-14
                   for f in ('ndcg', 'hit', 'recall', 'mrr')):
                raise ValueError('Rebuilt aggregate differs')
    preserved_path = ROOT/'artifacts/runs/recommendation_search/logs/preservation_before.json'
    preserved = json.loads(preserved_path.read_text())
    changed = [p for p, h in preserved.items() if digest(ROOT/p) != h]
    if changed:
        raise ValueError('Historical files changed: '+repr(changed))
    green_path = ROOT/'artifacts/runs/recommendation_search/logs/green.xml'
    suites = ET.parse(green_path).getroot().findall('testsuite')
    tests = {field: sum(int(s.get(field, '0')) for s in suites)
             for field in ('tests', 'errors', 'failures', 'skipped')}
    full_path = ROOT/'artifacts/runs/recommendation_search/logs/full_tests.xml'
    full_suites = ET.parse(full_path).getroot().findall('testsuite')
    full_tests = {field: sum(int(s.get(field, '0')) for s in full_suites)
                  for field in ('tests', 'errors', 'failures', 'skipped')}
    if any(full_tests[field] for field in ('errors', 'failures')):
        raise ValueError('Full engineering tests failed')
    report = {'status': 'VERIFIED', 'scope': plan['purpose'], 'plan': plan,
              'plan_sha256': digest(plan_path), 'results': results, 'selected': selected,
              'comparisons': {'selected_gcn_minus_previous_gcn': paired(previous, predictions[selected['lightgcn']]),
                              'selected_gcn_minus_tuned_bpr': paired(predictions[selected['bpr']], predictions[selected['lightgcn']])},
              'verification': {'targeted_tests': tests, 'all_prediction_metrics_rebuilt': True,
                               'full_engineering_tests': full_tests,
                               'raw_curves_exact': True,
                               'original_curve_prefix_audit': json.loads((ROOT/'artifacts/runs/recommendation_search/logs/prefix_audit.json').read_text()),
                               'unchanged_historical_files': len(preserved), 'changed_historical_files': changed},
              'test_evaluation': 'NOT EVALUATED', 'three_training_seed_evaluation': 'NOT EVALUATED',
              'serving_model_replaced': False, 'remote_api_requests': 0, 'downloads': 0,
              'limitations': ['One training seed and reused validation set after historical test exposure.',
                             'More LightGCN candidates due to depth parameter; total search counts differ.',
                             'NDCG selects configuration and epoch; HitRate is secondary, not guaranteed to rise.',
                             'Longer budget and patience are changed together compared to historical models.',
                             'Concurrent local CPU training times are not isolated latency benchmarks.',
                             'Python open guard is not an OS/native I/O sandbox.']}
    target = ROOT/'reports/rec_baselines/recommendation_search_20261010.json'
    target.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    freeze = {'schema_version': 1, 'purpose': 'Frozen LG-S01 validation selection for subsequent three-seed comparison',
              'selection_metric': plan['selection_metric'], 'source_plan_sha256': digest(plan_path),
              'source_report_sha256': digest(target), 'seeds': [7, 42, 2026],
              'selected': {model: results[name]['trial'] for model, name in selected.items()},
              'historical_test_already_exposed': True, 'new_test_evaluation': 'NOT EVALUATED',
              'three_seed_evaluation': 'NOT EVALUATED', 'remote_api_requests': 0,
              'serving_model_replaced': False}
    frozen_path = ROOT/'AgenticRec/configs/recommendation_search_selected_20261010.json'
    frozen_bytes = (json.dumps(freeze, indent=2, sort_keys=True)+'\n').encode('utf-8')
    if frozen_path.exists() and frozen_path.read_bytes() != frozen_bytes:
        raise ValueError('Refusing to change frozen selection')
    frozen_path.write_bytes(frozen_bytes)
    gc = results[selected['lightgcn']]
    bp = results[selected['bpr']]
    previous_metrics = {field: sum(r[field] for r in previous.values())/len(previous)
                        for field in ('ndcg', 'hit', 'recall', 'mrr')}
    delta = report['comparisons']['selected_gcn_minus_previous_gcn']
    control = report['comparisons']['selected_gcn_minus_tuned_bpr']
    lines = ['# LightGCN 有限联合搜索（2026-10-10）', '',
             '单seed开发搜索已执行；新模型test/三seed NOT EVALUATED。原正式模型未替换。', '',
             '## Completed', '',
             '- 完成预登记六个训练对照：1/2层×32/64维LightGCN、32/64维BPR；相同batch2048/lr.001/正则1e-4/max32/patience8。',
             '- 仅valid NDCG@10选模型/epoch；独立重载checkpoint并从私密预测复算四项指标。', '',
             '## Files Created / Modified', '',
             '- 独立recommendation_search模块、搜索/冻结配置、专项测试、报告生成脚本、本报告和机器报告。',
             '- STATUS、TASKS与ADR-037；旧模型、数据、训练数学实现、正式T19报告不变。', '',
             '## Commands / Tests', '',
             '```powershell',
             '& AgenticRec/.venv-t20/Scripts/python.exe -m agenticrec.recommendation_search --plan AgenticRec/configs/recommendation_search_20261010.json --trial <registered_trial>',
             '& AgenticRec/.venv-t20/Scripts/python.exe AgenticRec/scripts/report_recommendation_search.py',
             '```', '',
             '重复run/trial拒绝覆盖；私密预测、权重、训练曲线及stdout/stderr仅在ignored artifacts/runs/recommendation_search。', '',
             '## Verification Results', '',
             f"专项 {tests['tests']} 项；完整工程回归 {full_tests['tests']} 项，失败 {full_tests['failures']}，错误 {full_tests['errors']}；六个checkpoint重载、原始曲线和预测复算通过；{len(preserved)}个历史文件SHA不变。", '',
             '| 模型 | 层数 | 维度 | 实际轮数 | 最优轮 | valid NDCG@10 | valid HitRate@10 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for trial in plan['trials']:
        r=results[trial['name']]; m=r['valid_metrics']
        lines.append(f"| {trial['model']} | {trial['layers']} | {trial['embedding_dim']} | {r['actual_epochs']} | {r['best_epoch']} | {m['ndcg']:.6f} | {m['hit']:.6f} |")
    lines += ['', '所有模型valid主均值分母481人，无相关记录4870人另列，全集3469物品，历史已评分排除规则不变。', '',
              '## Findings', '',
              f"- 选中LightGCN：{selected['lightgcn']}，最优epoch{gc['best_epoch']}；选中增强BPR：{selected['bpr']}，epoch{bp['best_epoch']}。",
              f"- 相对上一轮1层/32维/8轮，NDCG {previous_metrics['ndcg']:.6f} → {gc['valid_metrics']['ndcg']:.6f}（相对{100*(gc['valid_metrics']['ndcg']/previous_metrics['ndcg']-1):+.2f}%），HitRate {previous_metrics['hit']:.2%} → {gc['valid_metrics']['hit']:.2%}（{100*delta['hit']['mean']:+.2f}个百分点）。",
              f"- 与上一轮比较，配对NDCG差值95%探索区间={delta['ndcg']['ci95']}；HitRate差值区间={delta['hit']['ci95']}。",
              f"- 与增强BPR比较，配对NDCG差值={control['ndcg']['mean']:+.6f}，95%探索区间={control['ndcg']['ci95']}；HitRate差值={100*control['hit']['mean']:+.2f}个百分点，区间={control['hit']['ci95']}。",
              '- BPR32原前三轮验证曲线与旧运行完全相同：原patience2在第3轮停止，本轮第6轮恢复并继续改善。GCN32前八轮验证NDCG也精确一致。早停/训练预算确实是值得共同检查的缺口，但本实验同时改变上限与patience，不能单独归因。',
              '- 不能将valid收益直接外推为test HitRate或真实用户喜欢概率。配对区间为探索性统计，未做多重比较校正。',
              '- 公共维度和停止策略一致，GCN额外搜索层数；不声称两种模型调参总预算相等。', '',
              '## Blockers', '', '本轮：None。正式改进模型三seed/test仍NOT EVALUATED；T04旧资源ID/许可阻塞沿用。', '',
              '## Deviations from Spec', '',
              '用户明确要求T20后继续优化LightGCN，先LG-S01、T21暂待。遵守原选模指标及数据协议；历史test已暴露，后续开发不称新盲测。', '',
              '## Current Project State', '',
              '有限开发搜索已执行，基于valid NDCG选中的两模型配置已冻结到recommendation_search_selected_20261010.json。旧线上/演示模型未自动替换，API请求0、下载0；全部坏结果保留。', '',
              '## Next Tasks', '',
              '冻结选中配置再跑seed7/42/2026；按既往测试暴露事实报告正式全集指标和配对区间。若仅开发收益，保持实验性。']
    target.with_suffix('.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'status':'VERIFIED', 'selected': selected,
                      'selected_metrics': {m:results[n]['valid_metrics'] for m,n in selected.items()},
                      'report_sha256': digest(target)}, sort_keys=True))
    return report


if __name__ == '__main__':
    build()
