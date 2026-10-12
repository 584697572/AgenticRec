"""Rebuild LG-O01 public results from private predictions, not typed metrics."""
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import subprocess
import xml.etree.ElementTree as ET

from agenticrec.data import digest
from agenticrec.evaluation.metrics import evaluate_cohort, score_ranking
from agenticrec.frozen_model_validation import average_rows, clustered_pair, strata
from agenticrec.lightgcn_diagnostic import validation_only_files
from agenticrec.lightgcn_optimization import BASE, plan_and_groups
from agenticrec.training import ROOT, load_frozen_cohort, load_train_contract


def stable_write(path, content):
    data = content.encode('utf-8')
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('Refuse to replace measured report: '+path.name)
    else:
        path.write_bytes(data)


def main():
    plan_path = ROOT/'AgenticRec/configs/lightgcn_optimization_20261011.json'
    plan, groups = plan_and_groups(plan_path)
    summary_path = BASE/'audit/summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    selection = json.loads((BASE/'selection_frozen.json').read_text(encoding='utf-8'))
    if summary['plan_sha256'] != digest(plan_path) or summary['selection_frozen_sha256'] != digest(BASE/'selection_frozen.json'):
        raise ValueError('Frozen report identity mismatch')
    checked, checkpoints, raw, measured_commits, dirty = 0, 0, {}, set(), []
    with validation_only_files():
        contract = load_train_contract()
        cohort = load_frozen_cohort('valid', ROOT/'artifacts/data_manifest.json')
        universe = cohort['candidate_ids']
        cohorts = {'selection': groups['fit']+groups['select'], 'audit': groups['audit'], 'all_valid': cohort['user_ids']}
        popularity = Counter(i for _, i in contract[4])
        hot = set(sorted(popularity, key=lambda i: (-popularity[i], i))[:100])
        for trial in plan['trials']:
            name = trial['name']
            for seed in plan['seeds']:
                directory = BASE/f'seed_{seed}'/name
                original = json.loads((directory/'result.json').read_text(encoding='utf-8'))
                run = summary['runs'][f'{name}_{seed}']
                if digest(directory/'best.pt') != original['checkpoint_sha256'] or digest(directory/'result.json') != run['source_result_sha256']:
                    raise ValueError('Checkpoint/result digest mismatch')
                if original['identity']['plan_sha256'] != digest(plan_path) or original['identity']['code_sha256'] != summary['code_sha256']:
                    raise ValueError('Training source identity mismatch')
                measured_commits.add(original['identity']['git_commit'])
                if original['identity']['git_dirty']:
                    dirty.append(f'{name}_{seed}')
                checkpoints += 1
                for group, users in cohorts.items():
                    path = BASE/'audit'/f'{name}_{seed}_{group}_predictions.json'
                    if digest(path) != run[group+'_predictions_sha256']:
                        raise ValueError('Raw prediction digest mismatch')
                    rows = json.loads(path.read_text(encoding='utf-8'))
                    expected_users = {str(u) for u in users if cohort['relevant_by_user'][str(u)]}
                    if set(rows) != expected_users:
                        raise ValueError('Prediction denominator changed')
                    for u, row in rows.items():
                        scores = score_ranking(row['prediction'], cohort['relevant_by_user'][u], universe, 10, cohort['seen_by_user'][u])
                        for field in ('recall', 'hit', 'mrr', 'ndcg'):
                            if scores[field] != row[field]:
                                raise ValueError('Per-user metric mismatch')
                            checked += 1
                    metrics = evaluate_cohort(sorted(users), {u: cohort['relevant_by_user'][str(u)] for u in users},
                        {int(u): r['prediction'] for u, r in rows.items()}, universe, 10,
                        {u: cohort['seen_by_user'][str(u)] for u in users})
                    if metrics != run['metrics'][group]:
                        raise ValueError('Aggregate metric mismatch')
                    predicted = [i for r in rows.values() for i in r['prediction']]
                    diversity = {'distinct_recommended_items': len(set(predicted)),
                                 'train_hot100_slot_share': sum(i in hot for i in predicted)/len(predicted)}
                    if diversity != run['diversity'][group]:
                        raise ValueError('Diversity mismatch')
                    raw[name, seed, group] = rows
        for group in cohorts:
            means = {t['name']: average_rows([raw[t['name'], s, group] for s in plan['seeds']]) for t in plan['trials']}
            for comparison, value in summary['paired_ndcg'][group].items():
                a, b = comparison.split('_minus_')
                if clustered_pair(means[a], means[b], 'ndcg') != value:
                    raise ValueError('Paired interval mismatch')
            for name, rows in means.items():
                if strata(rows, contract[3]) != summary['activity_strata'][group][name]:
                    raise ValueError('Activity stratification mismatch')
                for field in ('recall', 'hit', 'mrr', 'ndcg'):
                    values = [summary['runs'][f'{name}_{s}']['metrics'][group][field] for s in plan['seeds']]
                    if {'mean': statistics.mean(values), 'std': statistics.stdev(values)} != summary['aggregate'][group][name][field]:
                        raise ValueError('Seed mean/std mismatch')
    preserved = json.loads((ROOT/'artifacts/runs/upgrade_20261011/preservation_before.json').read_text(encoding='utf-8'))
    if any(digest(ROOT/path) != sha for path, sha in preserved.items()):
        raise ValueError('Historical evidence changed')
    models = json.loads((BASE/'historical_models_before.json').read_text(encoding='utf-8'))
    if any(digest(ROOT/path) != sha for path, sha in models.items()):
        raise ValueError('Historical weights changed')
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT/'RecAI', text=True).strip() != '0959ecb05b0794748426e73e6efc1b6b35ec433d':
        raise ValueError('Upstream revision changed')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT/'RecAI'):
        raise ValueError('Upstream tree changed')
    test_xml = BASE/'full_tests.xml'
    suite = ET.parse(test_xml).getroot().find('testsuite')
    if int(suite.attrib['failures']) or int(suite.attrib['errors']):
        raise ValueError('Full regression did not pass')
    selected_g = summary['selected']['lightgcn']
    selected_b = summary['selected']['bpr']
    vs_g = summary['paired_ndcg']['audit'][selected_g+'_minus_gcn_edge_uniform']
    vs_b = summary['paired_ndcg']['audit'][selected_g+'_minus_'+selected_b]
    verdict = 'NO VERIFIED GAIN'
    if vs_g['ci95'][0] > 0 and vs_b['ci95'][0] > 0:
        verdict = 'Positive development-audit gain; independent confirmation NOT EVALUATED'
    profile = json.loads((BASE/'kernel_profile_final.json').read_text(encoding='utf-8'))
    epoch_profile = json.loads((BASE/'epoch_profile.json').read_text(encoding='utf-8'))
    sampling = json.loads((BASE/'sampling_intervention.json').read_text(encoding='utf-8'))
    fresh = json.loads((BASE/'fresh_smoke.json').read_text(encoding='utf-8'))
    fresh_suite = ET.parse(BASE/'fresh_tests.xml').getroot().find('testsuite')
    if dirty or int(fresh_suite.attrib['failures']) or int(fresh_suite.attrib['errors']) or fresh['status'] != 'PASS':
        raise ValueError('Dirty formal training or independent environment failure')
    for measured, source in [(profile, 'profile_lightgcn_kernel.py'), (epoch_profile, 'profile_lightgcn_epoch.py')]:
        if measured['status'] != 'PASS' or measured['source_sha256'] != digest(ROOT/'AgenticRec/scripts'/source):
            raise ValueError('Profiler measurement/source identity mismatch')
        if measured['model_source_sha256'] != digest(ROOT/'AgenticRec/src/agenticrec/models/lightgcn_residual.py'):
            raise ValueError('Profiled model source changed')
    original_time, optimized_time = [r['elapsed_seconds'] for r in epoch_profile['results']]
    epoch_speedup = original_time/optimized_time
    epoch_reduction = 1-optimized_time/original_time
    public = {**summary, 'conclusion': verdict,
              'verification': {'raw_metric_values_rebuilt': checked, 'checkpoints': checkpoints,
                               'historical_files_preserved': len(preserved), 'historical_weights_preserved': len(models),
                               'full_tests_passed': int(suite.attrib['tests'])-int(suite.attrib.get('skipped', 0)),
                               'full_test_xml_sha256': digest(test_xml), 'training_commits': sorted(measured_commits),
                               'dirty_training_runs': dirty, 'kernel_profile_sha256': digest(BASE/'kernel_profile_final.json')},
              'kernel_profile': profile, 'epoch_profile': epoch_profile,
              'epoch_loop_speedup': epoch_speedup, 'epoch_loop_time_reduction': epoch_reduction,
              'sampling_intervention': sampling, 'independent_environment': fresh,
              'fresh_tests_passed': int(fresh_suite.attrib['tests'])-int(fresh_suite.attrib.get('skipped', 0)),
              'report_generator_sha256': digest(Path(__file__)),
              'raw_summary_sha256': digest(summary_path), 'selection': selection,
              'timing_limit': 'Concurrent development training; not isolated model speed comparison',
              'serving_model_replaced': False, 'downloads': 0,
              'reproduction_command': 'python -m agenticrec.lightgcn_optimization {train|freeze|audit} --plan AgenticRec/configs/lightgcn_optimization_20261011.json'}
    stable_write(ROOT/'reports/rec_baselines/lightgcn_optimization_20261011.json', json.dumps(public, indent=2, sort_keys=True)+'\n')
    lines = ['# LightGCN采样与层聚合优化（2026-10-11）', '',
             '有界开发研究完成。结论：**'+verdict+'**。不改变原服务默认，不打开历史test，不称微软论文/线上提升。', '',
             '## 实验与选择', '',
             '固定MovieLens1M训练图与全集候选；384个开发用户选择/early stop，97个开发用户在冻结后审计。此前valid已用过，audit不是独立盲测。6条件×3seed；每epoch462887三元组，batch2048，lr0.001，L2=0.0001，至多32轮/patience8。GCN1层64维，BPR32维沿用上轮强配置。', '',
             'seed42开发选择GCN='+selected_g+'，BPR='+selected_b+'；另两seed及audit后不重新挑赢家。', '',
             '| 方法 | 选择集NDCG@10（3seed均值） | 审计NDCG@10（均值±std） | 审计Hit@10 |',
             '|---|---:|---:|---:|']
    for trial in plan['trials']:
        name = trial['name']; a = summary['aggregate']['audit'][name]; s = summary['aggregate']['selection'][name]
        lines.append(f"| {name} | {s['ndcg']['mean']:.6f} | {a['ndcg']['mean']:.6f}±{a['ndcg']['std']:.6f} | {a['hit']['mean']:.2%} |")
    lines += ['', '## 消融与不确定性', '',
              'edge/user是边遍历/均匀用户正片采样；uniform/residual是ego/邻居0.5/0.5与0.8/0.2。单变量控制保持训练更新上限相同。按用户先平均seed再bootstrap；下面是探索性、未多重校正的95%区间，不作为普遍显著提升。', '']
    for comparison, row in summary['paired_ndcg']['audit'].items():
        lines.append(f"- {comparison}：NDCG差值{row['difference']:.6f}，区间[{row['ci95'][0]:.6f}, {row['ci95'][1]:.6f}]。")
    lines += ['', '## 验证、成本与限制', '',
              f'真实训练{len(summary["runs"])}次、实际{sum(r["actual_epochs"] for r in summary["runs"].values())}epoch；{checkpoints}权重重载，{checked}原始用户指标重算，{len(preserved)}历史文件及{len(models)}历史权重SHA保持。完整回归'+str(public['verification']['full_tests_passed'])+'项通过。新增API0、下载0、云费用0；本地CPU墙钟不是线上latency。', '',
              '前置测试首次因系统临时目录写权限发生5个setup error；指定工作区新basetemp后通过。缺可选NumPy、TypedStorage弃用及实验CSR beta警告保留。原始失败日志不删。', '',
              '原生CSR更慢，直接缓存转置只小幅变化；固定CSR的sum内核配合显式固定图反向通过独立损失/梯度和真实图检查。kernel报告只测相同batch计算，不将比例冒称全训练/在线加速；GPU/Linux未验证。', '',
              f'串行相同初始权重、样本、227次Adam更新的一轮训练循环：原COO {original_time:.2f}秒，缓存CSR sum {optimized_time:.2f}秒，观测{epoch_speedup:.2f}倍/耗时减少{epoch_reduction:.2%}。包含采样、张量、loss/backward/optimizer，不含数据/模型准备、valid及冷启动；仅一次固定顺序配对，不是完整训练或线上加速。最大参数差异{epoch_profile["results"][1]["parameter_max_abs"]:.3g}，最大batch loss差异{epoch_profile["results"][1]["max_batch_loss_difference"]:.3g}。', '',
              f'采样干预检查：最活跃约10%用户占原边{sampling["edge_high_activity_share"]:.2%}，均匀用户采样占{sampling["uniform_user_high_activity_share"]:.2%}；样本数量相同且全来自train。改变分布真实生效，不代表准确率必然提高。', '',
              '既有独立锁定环境16项新专项通过，真实残差GCN seed7重载和384人选择指标精确一致；共享checkout/数据/权重，未新建环境或再次从零重训。', '',
              '本轮未跑新的真实LLM、外部或历史test，正式独立质量提升仍NOT EVALUATED。三seed复用相同用户，不等于三倍独立样本。发现负结果后不扩大搜索或自动接入。原T04阻塞保持。', '',
              '## 复现', '', '```powershell',
              '& AgenticRec/.venv-t20/Scripts/python.exe -m agenticrec.lightgcn_optimization train --plan AgenticRec/configs/lightgcn_optimization_20261011.json --trial <registered_name> --seed 42',
              '# 完成全部6个seed42后freeze；再跑每条件seed7和2026，最后audit。已有run拒绝覆盖。',
              '& AgenticRec/.venv-t20/Scripts/python.exe -m agenticrec.lightgcn_optimization freeze --plan AgenticRec/configs/lightgcn_optimization_20261011.json',
              '& AgenticRec/.venv-t20/Scripts/python.exe -m agenticrec.lightgcn_optimization audit --plan AgenticRec/configs/lightgcn_optimization_20261011.json',
              '& AgenticRec/.venv-t20/Scripts/python.exe AgenticRec/scripts/report_lightgcn_optimization.py', '```', '',
              '层加权依据[原论文](https://arxiv.org/abs/2002.02126)，均匀用户采样依据[固定作者代码](https://github.com/gusye1234/LightGCN-PyTorch/blob/947ca2b3b1d2d3545b114145710cb06c4e57b3d2/code/utils.py)。属于经典方法的本项目验证，不称算法首创。', '']
    stable_write(ROOT/'reports/rec_baselines/lightgcn_optimization_20261011.md', '\n'.join(lines))
    print(json.dumps({'status': 'VERIFIED', 'conclusion': verdict, 'raw_metrics_rebuilt': checked,
                      'checkpoints': checkpoints, 'historical_preserved': len(preserved)}))


if __name__ == '__main__':
    main()
