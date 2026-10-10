"""Separate, append-only LightGCN train/valid diagnostics; no test evaluator.

The original trainers, configs, checkpoints and T19 reports remain untouched.
Results are exploratory: the historical test comparison has already been seen.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import json
import math
from pathlib import Path
import random
import time

import torch

from .data import digest
from .evaluation.metrics import rank_candidates, score_ranking
from .lightgcn_training import evaluate_lightgcn, _code_digest
from .models.bpr import TrainNegativeSampler
from .models.lightgcn import LightGCN, lightgcn_loss
from .training import ROOT, load_frozen_cohort, load_train_contract


def validate_plan(plan):
    expected = {'schema_version', 'run_id', 'purpose', 'seed', 'layers',
                'embedding_dim', 'learning_rate', 'regularization', 'torch_threads',
                'top_k', 'selection_metric', 'test_access', 'remote_api_requests', 'trials'}
    if set(plan) != expected or plan['schema_version'] != 1:
        raise ValueError('Invalid diagnostic plan schema')
    if plan['test_access'] is not False or plan['remote_api_requests'] != 0:
        raise ValueError('Diagnosis is train/valid only, with zero API requests')
    if plan['selection_metric'] != 'valid_ndcg_at_10' or plan['top_k'] != 10:
        raise ValueError('Do not change the frozen selection metric')
    if (not isinstance(plan['run_id'], str) or not plan['run_id']
            or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in plan['run_id'])):
        raise ValueError('Invalid run ID')
    for key in ('seed', 'layers', 'embedding_dim', 'torch_threads'):
        if type(plan[key]) is not int or plan[key] < (0 if key == 'seed' else 1):
            raise ValueError('Invalid ' + key)
    for key in ('learning_rate', 'regularization'):
        value = plan[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value < 1:
            raise ValueError('Invalid ' + key)
    if plan['learning_rate'] == 0:
        raise ValueError('Learning rate must be positive')
    if not isinstance(plan['trials'], list) or not plan['trials']:
        raise ValueError('Empty trial plan')
    names = set()
    for trial in plan['trials']:
        if set(trial) != {'name', 'batch_size', 'max_epochs', 'patience'}:
            raise ValueError('Invalid trial schema')
        name = trial['name']
        if (not isinstance(name, str) or not name or name in names
                or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for c in name)):
            raise ValueError('Duplicate or invalid trial name')
        names.add(name)
        for key in ('batch_size', 'max_epochs', 'patience'):
            if type(trial[key]) is not int or trial[key] < 1:
                raise ValueError('Invalid trial ' + key)
    return plan


@contextmanager
def validation_only_files():
    """Deny Python file opens of all canonical test labels, even indirectly."""
    import builtins
    import io
    real_open, real_io_open = builtins.open, io.open

    def guarded(opener):
        def open_file(file, *args, **kwargs):
            if isinstance(file, (str, bytes, Path)):
                path = str(file).replace('\\', '/').lower()
                if (path.endswith('/eval_test.parquet') or path.endswith('/test.parquet')
                        or ('/data/eval_private/' in path and path.endswith('/test.json'))
                        or '/data/interactive/' in path and 'test' in Path(path).name):
                    raise PermissionError('Diagnostic may not open test labels')
            return opener(file, *args, **kwargs)
        return open_file
    builtins.open, io.open = guarded(real_open), guarded(real_io_open)
    try:
        yield
    finally:
        builtins.open, io.open = real_open, real_io_open


def valid_rows(model, cohort, positive_degree):
    """Private validation predictions and aggregate train-degree diagnostics."""
    users = [u for u in cohort['user_ids'] if cohort['relevant_by_user'][str(u)]]
    with torch.no_grad():
        ue, ie = model.propagate()
        scores = (ue[torch.tensor(users) - 1] @ ie.T).tolist()
    rows = {}
    for user, values in zip(users, scores):
        key = str(user)
        predicted = rank_candidates(dict(zip(cohort['candidate_ids'], values)),
                                    cohort['candidate_ids'], 10, cohort['seen_by_user'][key])
        rows[key] = {'ndcg': score_ranking(predicted, cohort['relevant_by_user'][key],
                                         cohort['candidate_ids'], 10,
                                         cohort['seen_by_user'][key])['ndcg'],
                     'train_positive_degree': positive_degree[user], 'prediction': predicted}
    bands = {}
    for name, lower, upper in [('5-19', 5, 20), ('20-99', 20, 100), ('100+', 100, math.inf)]:
        values = [row['ndcg'] for row in rows.values()
                  if lower <= row['train_positive_degree'] < upper]
        bands[name] = {'users': len(values), 'ndcg': sum(values) / len(values) if values else None}
    return rows, bands


def bootstrap_difference(baseline, treatment):
    if set(baseline) != set(treatment) or not baseline:
        raise ValueError('Paired validation users differ')
    diffs = [treatment[key]['ndcg'] - baseline[key]['ndcg'] for key in sorted(baseline, key=int)]
    rng = random.Random(42)
    samples = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(2000))
    return {'mean': sum(diffs) / len(diffs), 'ci95': [samples[49], samples[1949]],
            'paired_users': len(diffs), 'bootstrap_samples': 2000, 'seed': 42,
            'scope': 'exploratory same-validation-set, one training seed; not held-out confirmation'}


def run(plan_path):
    plan_path = Path(plan_path)
    plan = validate_plan(json.loads(plan_path.read_text(encoding='utf-8')))
    out = ROOT / 'artifacts/runs/lightgcn_diagnostics' / plan['run_id']
    # mkdir, not overwrite/recover: a rerun must use an independently registered ID.
    out.mkdir(parents=True, exist_ok=False)
    (out / 'plan.json').write_bytes(plan_path.read_bytes())
    identity = {'plan_sha256': digest(plan_path), 'training_code_sha256': _code_digest(),
                'diagnostic_code_sha256': digest(Path(__file__)), 'torch_version': str(torch.__version__)}
    torch.set_num_threads(plan['torch_threads'])
    results, all_rows = {}, {}
    with validation_only_files():
        manifest, n_users, n_items, seen, frozen_edges = load_train_contract()
        from .models.lightgcn import normalized_bipartite_graph
        graph = normalized_bipartite_graph(n_users, n_items, frozen_edges)
        valid = load_frozen_cohort('valid', ROOT / 'artifacts/data_manifest.json')
        degree = Counter(u for u, _ in frozen_edges)
        identity.update({'id_map_sha256': manifest['id_map_sha256'],
                         'train_positive_sha256': manifest['files']['train_positive.parquet']['sha256'],
                         'data_manifest_sha256': digest(ROOT / 'artifacts/data_manifest.json'),
                         'valid_cohort_sha256': digest(ROOT / 'data/eval_private/ml-1m-v1/valid.json')})
        for trial in plan['trials']:
            torch.manual_seed(plan['seed'])
            rng = random.Random(plan['seed'])
            edges = list(frozen_edges)
            sampler = TrainNegativeSampler(n_items, seen, rng)
            model = LightGCN(n_users, n_items, plan['embedding_dim'], plan['layers'], graph)
            optimizer = torch.optim.Adam(model.parameters(), lr=plan['learning_rate'])
            curve, best, best_epoch, stale, best_state = [], -1.0, 0, 0, None
            steps = 0
            started = time.perf_counter()
            checkpoint = out / (trial['name'] + '.pt')
            for epoch in range(1, trial['max_epochs'] + 1):
                rng.shuffle(edges)
                model.train()
                total_loss = 0.0
                for offset in range(0, len(edges), trial['batch_size']):
                    batch = edges[offset:offset + trial['batch_size']]
                    users = torch.tensor([u for u, _ in batch], dtype=torch.long)
                    pos = torch.tensor([i for _, i in batch], dtype=torch.long)
                    neg = torch.tensor([sampler.sample(u) for u, _ in batch], dtype=torch.long)
                    optimizer.zero_grad()
                    loss = lightgcn_loss(model, users, pos, neg, plan['regularization'])
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * len(batch)
                    steps += 1
                model.eval()
                metrics = evaluate_lightgcn(model, valid, plan['top_k'])
                row = {'trial': trial['name'], 'epoch': epoch,
                       'mean_train_loss': total_loss / len(edges), 'optimizer_steps': steps,
                       'valid_ndcg_at_10': metrics['ndcg'],
                       'elapsed_seconds': time.perf_counter() - started}
                curve.append(row)
                with (out / 'curve.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(row, sort_keys=True) + '\n')
                print(json.dumps(row, sort_keys=True), flush=True)
                if metrics['ndcg'] > best:
                    best, best_epoch, stale = metrics['ndcg'], epoch, 0
                    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
                    torch.save({'state_dict': best_state, 'identity': identity,
                                'plan': plan, 'trial': trial, 'epoch': epoch}, checkpoint)
                else:
                    stale += 1
                    if stale >= trial['patience']:
                        break
            model.load_state_dict(best_state)
            metrics = evaluate_lightgcn(model, valid, plan['top_k'])
            private_rows, bands = valid_rows(model, valid, degree)
            all_rows[trial['name']] = private_rows
            (out / (trial['name'] + '_valid_predictions.json')).write_text(
                json.dumps(private_rows, sort_keys=True) + '\n', encoding='utf-8')
            results[trial['name']] = {'trial': trial, 'best_epoch': best_epoch,
                                      'best_optimizer_steps': best_epoch * math.ceil(len(edges) / trial['batch_size']),
                                      'total_optimizer_steps': steps, 'curve': curve,
                                      'valid_metrics': metrics, 'train_degree_bands': bands,
                                      'checkpoint_sha256': digest(checkpoint),
                                      'elapsed_seconds': time.perf_counter() - started,
                                      'early_stopped': stale >= trial['patience']}
            (out / (trial['name'] + '_result.json')).write_text(
                json.dumps(results[trial['name']], indent=2, sort_keys=True) + '\n', encoding='utf-8')
    baseline = next(iter(all_rows))
    report = {'status': 'VERIFIED', 'scope': plan['purpose'], 'identity': identity, 'plan': plan,
              'graph_edges': len(frozen_edges), 'warm_users': n_users, 'candidate_items': n_items,
              'graph_zero_degree_nodes': int((torch.bincount(graph.indices()[0], minlength=n_users+n_items) == 0).sum()),
              'trials': results, 'comparisons': {name: bootstrap_difference(all_rows[baseline], rows)
                                               for name, rows in all_rows.items() if name != baseline},
              'test_evaluation': 'NOT EVALUATED', 'test_label_file_opens_allowed': False,
              'remote_api_requests': 0, 'baseline_weights_modified': False,
              'caution': 'One-seed validation diagnosis after historical test exposure. No new test or online improvement claim.'}
    (out / 'summary.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'VERIFIED', 'output': out.relative_to(ROOT).as_posix(),
                      'valid_ndcg': {name: row['valid_metrics']['ndcg'] for name, row in results.items()}}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    run(parser.parse_args().plan)
