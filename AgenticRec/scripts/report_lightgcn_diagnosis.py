"""Rebuild aggregate-only exploratory diagnostics from private validation runs."""
from collections import Counter
import json
import math
from pathlib import Path
import platform
import random
import sys
import xml.etree.ElementTree as ET

import pyarrow
import torch

from agenticrec.data import digest
from agenticrec.lightgcn_diagnostic import bootstrap_difference, validation_only_files
from agenticrec.models.lightgcn import LightGCN, normalized_bipartite_graph
from agenticrec.training import ROOT, load_train_contract


def main():
    base = ROOT / 'artifacts/runs/lightgcn_diagnostics'
    run_ids = ['lightgcn_diagnostic_20261010', 'lightgcn_diagnostic_layer1_20261010',
               'lightgcn_diagnostic_patience_20261010']
    reports = {run_id: json.loads((base / run_id / 'summary.json').read_text()) for run_id in run_ids}
    for run_id, report in reports.items():
        if (report['status'] != 'VERIFIED' or report['test_evaluation'] != 'NOT EVALUATED'
                or report['remote_api_requests'] != 0 or report['plan']['seed'] != 42
                or report['identity']['plan_sha256'] != digest(base / run_id / 'plan.json')):
            raise ValueError('Incomplete or incompatible diagnostic run')
    if len({report['identity']['valid_cohort_sha256'] for report in reports.values()}) != 1:
        raise ValueError('Validation cohorts differ')
    if len({report['identity']['training_code_sha256'] for report in reports.values()}) != 1:
        raise ValueError('Underlying training math changed')
    torch.set_num_threads(4)
    with validation_only_files():
        manifest, nu, ni, _seen, edges = load_train_contract()
        graph = normalized_bipartite_graph(nu, ni, edges)
        pop = Counter(i for _, i in edges)
        hot = set(sorted(pop, key=lambda i: (-pop[i], i))[:100])
        degree = torch.bincount(graph.indices()[0], minlength=nu + ni).float()
        q = degree.sqrt()
        q = q / q.norm()
        rng = random.Random(42)
        pairs = [rng.sample(range(nu), 2) for _ in range(2000)]
        left = torch.tensor([a for a, _ in pairs])
        right = torch.tensor([b for _, b in pairs])
        trials, private = {}, {}
        for run_id, report in reports.items():
            for name, row in report['trials'].items():
                key = str(report['plan']['layers']) + 'layer_' + name
                folder = base / run_id
                checkpoint = folder / (name + '.pt')
                if digest(checkpoint) != row['checkpoint_sha256']:
                    raise ValueError('Diagnostic checkpoint hash mismatch')
                prediction_path = folder / (name + '_valid_predictions.json')
                predictions = json.loads(prediction_path.read_text())
                private[key] = predictions
                slots = [i for user in predictions.values() for i in user['prediction']]
                payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
                if payload['identity'] != report['identity'] or payload['plan'] != report['plan']:
                    raise ValueError('Checkpoint diagnostic identity mismatch')
                model = LightGCN(nu, ni, report['plan']['embedding_dim'], report['plan']['layers'], graph)
                model.load_state_dict(payload['state_dict'])
                with torch.no_grad():
                    user_embeddings, item_embeddings = model.propagate()
                    embeddings = torch.cat([user_embeddings, item_embeddings])
                    mode_fraction = float((q @ embeddings).square().sum() / embeddings.square().sum())
                    user_cosine = float(torch.nn.functional.cosine_similarity(
                        user_embeddings[left], user_embeddings[right], dim=1).mean())
                trials[key] = {**row, 'layers': report['plan']['layers'],
                               'steps_per_epoch': math.ceil(len(edges) / row['trial']['batch_size']),
                               'validation_predictions_sha256': digest(prediction_path),
                               'distinct_recommended_items': len(set(slots)),
                               'top100_train_popularity_slot_share': sum(i in hot for i in slots) / len(slots),
                               'global_sqrt_degree_mode_energy_fraction': mode_fraction,
                               'random_distinct_user_pair_cosine_mean': user_cosine,
                               'sampled_user_pairs': 2000, 'pair_sampling_seed': 42}
    comparisons = {}
    for baseline, treatment in [('3layer_original', '3layer_longer_training'),
                                ('3layer_original', '3layer_smaller_batch'),
                                ('1layer_original', '1layer_smaller_batch'),
                                ('3layer_smaller_batch', '1layer_smaller_batch'),
                                ('3layer_original', '1layer_smaller_batch'),
                                ('3layer_smaller_batch', '3layer_relaxed_patience'),
                                ('3layer_original', '3layer_relaxed_patience'),
                                ('3layer_relaxed_patience', '1layer_smaller_batch')]:
        comparisons[treatment + '_minus_' + baseline] = bootstrap_difference(private[baseline], private[treatment])
    interaction = {}
    for user in private['3layer_original']:
        interaction[user] = {'ndcg': (
            private['1layer_smaller_batch'][user]['ndcg'] - private['1layer_original'][user]['ndcg']
            - private['3layer_smaller_batch'][user]['ndcg'] + private['3layer_original'][user]['ndcg'])}
    comparisons['layer_by_batch_interaction'] = bootstrap_difference(
        {user: {'ndcg': 0.0} for user in interaction}, interaction)
    before = json.loads((base / 'preservation_before.json').read_text())
    changed = [name for name, sha in before.items() if digest(ROOT / name) != sha]
    if changed:
        raise ValueError('Historical baseline changed: ' + ', '.join(changed))
    audit = json.loads((base / 'static_audit.json').read_text())
    checks = json.loads((base / 'baseline_reproduction_verified.json').read_text())
    historical_layer1 = json.loads((ROOT / 'reports/rec_baselines/lightgcn_seed_42.json').read_text())['trials']['1']
    checks['layer1_valid_metrics_exact'] = trials['1layer_original']['valid_metrics'] == historical_layer1['valid_metrics']
    checks['layer1_valid_curve_exact'] = all(a['valid_ndcg_at_10'] == b['valid_ndcg_at_10']
                                           for a, b in zip(trials['1layer_original']['curve'], historical_layer1['curve']))
    checks['patience_control_prefix_valid_curve_exact'] = all(
        a['valid_ndcg_at_10'] == b['valid_ndcg_at_10'] and a['optimizer_steps'] == b['optimizer_steps']
        for a, b in zip(trials['3layer_smaller_batch']['curve'], trials['3layer_relaxed_patience']['curve']))
    if checks['status'] != 'PASS' or not checks['layer1_valid_metrics_exact'] or not checks['layer1_valid_curve_exact']:
        raise ValueError('Baseline reproduction failed')
    if not checks['patience_control_prefix_valid_curve_exact']:
        raise ValueError('Patience control diverged before stopping boundary')
    junit = base / 'diagnostic_tests_final.xml'
    suite = ET.parse(junit).getroot().find('testsuite')
    test_counts = {key: int(suite.attrib[key]) for key in ('tests', 'failures', 'errors', 'skipped')}
    if test_counts['tests'] != 25 or any(test_counts[key] for key in ('failures', 'errors', 'skipped')):
        raise ValueError('Diagnostic tests not verified')
    report = {'status': 'VERIFIED', 'scope': 'Post-T19 exploratory LightGCN diagnosis, validation-only, seed42',
              'environment': {'python': platform.python_version(), 'torch': str(torch.__version__),
                              'pyarrow': pyarrow.__version__, 'platform': sys.platform, 'device': 'cpu', 'torch_threads': 4},
              'training_source_commits': {run_ids[0]: 'a3f4f12', run_ids[1]: 'a3f4f12', run_ids[2]: '27e4095'},
              'report_code_sha256': digest(Path(__file__)),
              'run_summary_sha256': {name: digest(base / name / 'summary.json') for name in run_ids},
              'source_identities': {name: record['identity'] for name, record in reports.items()},
              'verification': {'targeted_tests': test_counts, 'junit_sha256': digest(junit),
                               'full_current_258_suite': 'NOT VERIFIED in this diagnosis turn',
                               'prior_t20_full_suite': '253 passed before diagnostic additions',
                               'guard_regression': '2 RED failures before fix, final GREEN',
                               'first_test_setup': '20 passed and 3 setup errors; missing basetemp parent, retained',
                               'repeated_run': 'refuses existing output directory before training'},
              'static_audit': audit, 'baseline_reproduction': checks, 'preserved_historical_files': len(before),
              'changed_historical_files': changed, 'trials': trials, 'comparisons': comparisons,
              'bpr_validation_audit': json.loads((base / 'bpr_diversity_audit.json').read_text()),
              'test_evaluation': 'NOT EVALUATED', 'remote_api_requests': 0, 'resource_downloads': 0,
              'limitations': ['One training seed; reused validation set; historical test already exposed.',
                              'Adaptive second phase and multiple comparisons; exploratory intervals only.',
                              'Batch change also changes update count and gradient noise; these effects are not isolated.',
                              'Longer training changes both epoch cap and early-stopping patience.',
                              'Graph-mode energy and pairwise cosine are descriptive, not causal proof of oversmoothing.',
                              'Python-open guard is an accidental-access check, not an OS sandbox for native I/O.',
                              'No architecture/config/checkpoint replaced in original serving pipeline.']}
    output = ROOT / 'reports/rec_baselines/lightgcn_diagnosis_20261010.json'
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps({'status': report['status'], 'report': output.relative_to(ROOT).as_posix(),
                      'valid_ndcg': {name: row['valid_metrics']['ndcg'] for name, row in trials.items()},
                      'preserved_historical_files': len(before)}, sort_keys=True))


if __name__ == '__main__':
    main()
