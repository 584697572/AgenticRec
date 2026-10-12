"""Publish verified six-checkpoint results; never train, tune, or call APIs."""
import json
from pathlib import Path
import statistics

from agenticrec.data import digest
from agenticrec.evaluation.metrics import score_ranking, evaluate_cohort, FIELDS
from agenticrec.training import ROOT, load_frozen_cohort
from report_upgrade_research import stable_write


def main():
    directory = ROOT/'artifacts/runs/upgrade_20261011/model_validation_v2'
    summary_path = directory/'summary.json'
    report = json.loads(summary_path.read_text(encoding='utf-8'))
    executed_source = ROOT/'artifacts/runs/upgrade_20261011/executed_frozen_model_validation.py'
    if report['status'] != 'VERIFIED' or digest(executed_source) != report['code_sha256']:
        raise ValueError('Completed result or executed-source evidence mismatch')
    checked = 0
    for split in ('valid', 'test'):
        cohort = load_frozen_cohort(split, ROOT/'artifacts/data_manifest.json')
        for kind in ('bpr', 'lightgcn'):
            for seed in report['seeds']:
                entry = report['runs'][f'{kind}_{seed}']
                path = directory/f'{kind}_seed_{seed}_{split}_predictions.json'
                if digest(path) != entry[split+'_predictions_sha256']:
                    raise ValueError('Frozen predictions modified')
                rows = json.loads(path.read_text(encoding='utf-8'))
                expected = {str(u) for u in cohort['user_ids'] if cohort['relevant_by_user'][str(u)]}
                if set(rows) != expected:
                    raise ValueError('Prediction cohort mismatch')
                for user, row in rows.items():
                    scores = score_ranking(row['prediction'], cohort['relevant_by_user'][user],
                        cohort['candidate_ids'], 10, cohort['seen_by_user'][user])
                    if scores != {k: row[k] for k in FIELDS}:
                        raise ValueError('Raw metrics do not rebuild exactly')
                    checked += 1
                metrics = evaluate_cohort(cohort['user_ids'],
                    {u: cohort['relevant_by_user'][str(u)] for u in cohort['user_ids']},
                    {int(u): row['prediction'] for u, row in rows.items()}, cohort['candidate_ids'], 10,
                    {u: cohort['seen_by_user'][str(u)] for u in cohort['user_ids']})
                if metrics != entry['metrics'][split]:
                    raise ValueError('Macro metrics differ from raw reconstruction')
            for field in FIELDS:
                values = [report['runs'][f'{kind}_{seed}']['metrics'][split][field] for seed in report['seeds']]
                if report['aggregate'][split][kind][field] != {'mean': statistics.mean(values), 'std': statistics.stdev(values)}:
                    raise ValueError('Seed summary mismatch')
    report['verification'] = {'raw_user_predictions_rebuilt_exact': checked,
        'raw_summary_sha256': digest(summary_path), 'generator_sha256': digest(Path(__file__)),
        'execution_base_commit': 'e1cb622',
        'execution_worktree': 'Output-path bug fix was uncommitted at launch; exact executed source SHA preserved',
        'new_training_base_commit': '6150227',
        'seed42_training': 'Reused immutable 2026-10-10 search outputs',
        'new_training_runs': 4, 'total_checked_checkpoints': 6,
        'selection_config': 'AgenticRec/configs/recommendation_search_selected_20261010.json'}
    report['limits'] = ['Historical test is Previously Exposed Test; cannot infer an independent blind gain',
        'Configs selected only by validation on 2026-10-10; no test-driven search in this validation',
        'Paired bootstrap unit is user after averaging seeds; not three independent user populations',
        '64-dim GCN versus 32-dim BPR; parameter counts and concurrent timing are disclosed',
        'No new Microsoft paper metric, LLM planning improvement, or online CTR claim']
    destination = ROOT/'reports/rec_baselines/frozen_model_validation_20261011'
    stable_write(destination.with_suffix('.json'), report)
    lines = ['# 固定强基线三 seed 验证（2026-10-11）', '',
        '只补齐已冻结配置的 seed 7/2026，复用 seed 42；没有新网格搜索或测试选参。', '',
        '| 模型 | valid NDCG@10 mean ± std | 暴露过的 test NDCG@10 mean ± std | test Hit@10 | 参数量 |',
        '|---|---:|---:|---:|---:|']
    for kind in ('bpr', 'lightgcn'):
        valid = report['aggregate']['valid'][kind]['ndcg']
        test = report['aggregate']['test'][kind]['ndcg']
        hit = report['aggregate']['test'][kind]['hit']['mean']
        lines.append(f"| {kind} | {valid['mean']:.6f} ± {valid['std']:.6f} | {test['mean']:.6f} ± {test['std']:.6f} | {hit:.6f} | {report['runs'][kind+'_42']['parameter_count']} |")
    paired = report['paired']['test']['ndcg']
    lines += ['', f"GCN−BPR test NDCG差值 {paired['difference']:.6f}，用户聚类95%区间 {paired['ci95']}；**NO VERIFIED GAIN**。", '',
        f"{checked} 条用户预测四项指标精确重算，六 checkpoint 重载 valid 指标与训练记录完全一致。valid/test 主分母为481/954相关暖用户，无相关用户4870/4397另计。", '',
        '配置：GCN一层64维、BPR32维；batch2048/lr0.001/reg0.0001/max32 epochs/patience8；seed7/42/2026。图、负采样和映射均仅使用train，不用未来target过滤。', '',
        '所有逐seed指标、最佳/实际epoch、checkpoint SHA、活跃度层、覆盖度、热门占比、推理时间在机器报告中。训练计时来自并行开发运行，不能比较公平速度；推理计时包含整空间评分和Python排序，不能冒充在线QPS。', '',
        '首次验证已经完成评分，但在报告路径转换时失败；原输出与错误日志保留。回归先复现相对路径错误后修复，第二次完整验证完成。执行时基于e1cb622，路径修复尚未提交；执行源码SHA及副本保留，不能声称当时工作树clean。', '',
        '新的完整测试是历史 Previously Exposed Test。此处比旧弱配置提高不等于原创算法收益，也不是相对Microsoft原论文提升。', '',
        '```powershell',
        'foreach ($trial in @("bpr_l0_d32", "lightgcn_l1_d64")) {',
        '  & AgenticRec/.venv-t20/Scripts/python.exe -m agenticrec.recommendation_search --plan AgenticRec/configs/frozen_model_validation_20261011_seed_7.json --trial $trial',
        '  & AgenticRec/.venv-t20/Scripts/python.exe -m agenticrec.recommendation_search --plan AgenticRec/configs/frozen_model_validation_20261011_seed_2026.json --trial $trial',
        '}',
        '# 若运行目录已存在须在副本配置中使用新 run_id；禁止覆盖历史权重',
        '& AgenticRec/.venv-t20/Scripts/python.exe -m agenticrec.frozen_model_validation --output artifacts/runs/<new_validation_id>',
        '& AgenticRec/.venv-t20/Scripts/python.exe AgenticRec/scripts/report_frozen_validation.py',
        '```', '']
    stable_write(destination.with_suffix('.md'), '\n'.join(lines))
    print(json.dumps({'status': 'VERIFIED', 'raw_predictions_rebuilt': checked, 'api_requests': 0}))


if __name__ == '__main__':
    main()
