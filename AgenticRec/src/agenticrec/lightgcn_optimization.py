"""Finite train/development-only sampling x aggregation experiment.

All historical test labels are excluded. Audit results require frozen selection.
No API calls, downloads, serving replacement, or overwrite of existing runs.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import statistics
import subprocess
import time

import torch

from .data import digest
from .frozen_model_validation import average_rows, clustered_pair, strata
from .lightgcn_diagnostic import validation_only_files
from .lightgcn_training import _code_digest
from .models.bpr import BPRMF, TrainNegativeSampler, bpr_loss
from .models.lightgcn import lightgcn_loss, normalized_bipartite_graph
from .models.lightgcn_residual import ResidualLightGCN, validate_weights
from .recommendation_search import evaluate
from .training import ROOT, load_frozen_cohort, load_train_contract


BASE = ROOT / 'artifacts/runs/lightgcn_optimization_20261011'


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write('\n')


def validate_trial(trial):
    keys = {'name', 'model', 'layers', 'embedding_dim', 'layer_weights', 'sampling',
            'learning_rate', 'regularization', 'batch_size', 'max_epochs', 'patience'}
    if set(trial) != keys or not isinstance(trial['name'], str) or not trial['name'] or any(
            c not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for c in trial['name']):
        raise ValueError('Invalid trial schema/name')
    if trial['model'] not in ('bpr', 'lightgcn') or trial['sampling'] not in ('edge', 'uniform_user'):
        raise ValueError('Invalid model or training-only sampler')
    layers = trial['layers']
    if type(layers) is not int or (trial['model'] == 'bpr' and layers != 0) or (
            trial['model'] == 'lightgcn' and layers not in (1, 2, 3)):
        raise ValueError('Invalid layers')
    validate_weights(trial['layer_weights'], layers)
    for field in ('embedding_dim', 'batch_size', 'max_epochs', 'patience'):
        if type(trial[field]) is not int or trial[field] < 1:
            raise ValueError('Invalid ' + field)
    for field in ('learning_rate', 'regularization'):
        v = trial[field]
        if type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v < 1:
            raise ValueError('Invalid ' + field)
    if trial['learning_rate'] == 0:
        raise ValueError('Learning rate must be positive')
    return trial


def epoch_pairs(edges, sampling, rng, *, count=None):
    if sampling == 'edge':
        if count is not None:
            raise ValueError('Edge traversal must preserve its original count')
        pairs = list(edges)
        rng.shuffle(pairs)
        return pairs
    if sampling != 'uniform_user' or not edges:
        raise ValueError('Invalid sampler or empty training positives')
    positive = defaultdict(list)
    for u, i in edges:
        positive[u].append(i)
    users = sorted(positive)
    return [(u, rng.choice(positive[u])) for u in
            (rng.choice(users) for _ in range(len(edges) if count is None else count))]


def subset_cohort(cohort, users):
    users = sorted(set(users))
    if not users or not set(users) <= set(cohort['user_ids']):
        raise ValueError('Development users must belong to frozen validation cohort')
    return {**cohort, 'user_ids': users,
            'relevant_by_user': {str(u): cohort['relevant_by_user'][str(u)] for u in users},
            'seen_by_user': {str(u): cohort['seen_by_user'][str(u)] for u in users}}


def code_identity():
    source = Path(__file__).resolve().parent
    return hashlib.sha256((_code_digest() + ''.join(digest(source / name) for name in (
        'lightgcn_optimization.py', 'models/lightgcn_residual.py', 'recommendation_search.py',
        'lightgcn_diagnostic.py', 'frozen_model_validation.py'))).encode()).hexdigest()


def plan_and_groups(path):
    plan = json.loads(Path(path).read_text(encoding='utf-8'))
    expected = {'schema_version', 'purpose', 'seeds', 'torch_threads', 'selection_metric',
                'test_access', 'remote_api_requests', 'group_sha256', 'trials'}
    if set(plan) != expected or plan['schema_version'] != 1 or plan['seeds'] != [7, 42, 2026]:
        raise ValueError('Invalid study identity/seeds')
    if plan['test_access'] is not False or plan['remote_api_requests'] != 0 or (
            plan['selection_metric'] != 'development_selection_ndcg_at_10'):
        raise ValueError('Only development selection and zero remote calls are allowed')
    if type(plan['torch_threads']) is not int or not 1 <= plan['torch_threads'] <= 4:
        raise ValueError('Invalid CPU threads')
    names = []
    for trial in plan['trials']:
        validate_trial(trial)
        names.append(trial['name'])
    if not names or len(names) != len(set(names)) or len(names) > 6:
        raise ValueError('Finite study permits at most six unique trials')
    group_path = ROOT / 'data/eval_private/fusion-development-20261011/user_groups.json'
    if digest(group_path) != plan['group_sha256']:
        raise ValueError('Development group identity mismatch')
    groups = json.loads(group_path.read_text(encoding='utf-8'))
    selection = groups['fit'] + groups['select']
    if len(set(selection)) != len(selection) or set(selection) & set(groups['audit']):
        raise ValueError('Development selection/audit overlap')
    return plan, groups


def build_model(trial, contract):
    _, nu, ni, _, edges = contract
    if trial['model'] == 'bpr':
        return BPRMF(nu, ni, trial['embedding_dim'])
    graph = normalized_bipartite_graph(nu, ni, edges)
    return ResidualLightGCN(nu, ni, trial['embedding_dim'], trial['layers'], graph, trial['layer_weights'])


def load_verified(directory, trial, contract, plan_sha, seed):
    result = json.loads((directory / 'result.json').read_text(encoding='utf-8'))
    manifest, nu, ni, _, _ = contract
    required = {'plan_sha256': plan_sha, 'code_sha256': code_identity(), 'seed': seed,
                'data_manifest_sha256': digest(ROOT/'artifacts/data_manifest.json'),
                'id_map_sha256': manifest['id_map_sha256'],
                'train_positive_sha256': manifest['files']['train_positive.parquet']['sha256'],
                'valid_sha256': digest(ROOT/'data/eval_private/ml-1m-v1/valid.json'),
                'torch_version': str(torch.__version__)}
    if result['status'] != 'VERIFIED' or result['trial'] != trial or any(
            result['identity'].get(k) != v for k, v in required.items()):
        raise ValueError('Checkpoint training identity mismatch')
    if digest(directory/'best.pt') != result['checkpoint_sha256']:
        raise ValueError('Checkpoint SHA mismatch')
    payload = torch.load(directory/'best.pt', map_location='cpu', weights_only=True)
    if payload['identity'] != result['identity'] or payload['trial'] != trial or (
            payload['dimensions'] != [nu, ni] or payload['best_epoch'] != result['best_epoch']):
        raise ValueError('Checkpoint payload/config mismatch')
    model = build_model(trial, contract)
    model.load_state_dict(payload['state_dict'], strict=True)
    model.eval()
    return model, result


def train(plan_path, name, seed):
    plan, groups = plan_and_groups(plan_path)
    matches = [t for t in plan['trials'] if t['name'] == name]
    if len(matches) != 1 or seed not in plan['seeds']:
        raise ValueError('Unregistered trial/seed')
    if seed != 42:
        frozen = json.loads((BASE/'selection_frozen.json').read_text(encoding='utf-8'))
        if frozen['plan_sha256'] != digest(plan_path) or frozen['code_sha256'] != code_identity():
            raise ValueError('Selection must be frozen before independent seeds')
    trial = matches[0]
    out = BASE / f'seed_{seed}' / name
    out.mkdir(parents=True, exist_ok=False)
    (out/'plan.json').write_bytes(Path(plan_path).read_bytes())
    torch.set_num_threads(plan['torch_threads'])
    torch.manual_seed(seed)
    rng = random.Random(seed)
    started = time.perf_counter()
    with validation_only_files():
        contract = load_train_contract()
        manifest, nu, ni, seen, edges = contract
        cohort = load_frozen_cohort('valid', ROOT/'artifacts/data_manifest.json')
        selection = subset_cohort(cohort, groups['fit'] + groups['select'])
        identity = {'seed': seed, 'plan_sha256': digest(plan_path), 'code_sha256': code_identity(),
                    'data_manifest_sha256': digest(ROOT/'artifacts/data_manifest.json'),
                    'id_map_sha256': manifest['id_map_sha256'],
                    'train_positive_sha256': manifest['files']['train_positive.parquet']['sha256'],
                    'valid_sha256': digest(ROOT/'data/eval_private/ml-1m-v1/valid.json'),
                    'torch_version': str(torch.__version__), 'python': platform.python_version(),
                    'platform': platform.platform(), 'git_commit': subprocess.check_output(
                        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    'git_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT))}
        write(out/'identity.json', identity)
        model = build_model(trial, contract)
        sampler = TrainNegativeSampler(ni, seen, rng)
        popularity = Counter(i for _, i in edges)
        optimizer = torch.optim.Adam(model.parameters(), lr=trial['learning_rate'])
        loss_fn = lightgcn_loss if trial['model'] == 'lightgcn' else bpr_loss
        best, best_epoch, stale, steps, curve = -1.0, 0, 0, 0, []
        for epoch in range(1, trial['max_epochs'] + 1):
            pairs = epoch_pairs(edges, trial['sampling'], rng)
            if trial['sampling'] == 'edge':
                # Preserve legacy repeated in-place permutation across epochs.
                edges = pairs
            model.train()
            total = 0.0
            for offset in range(0, len(pairs), trial['batch_size']):
                batch = pairs[offset:offset + trial['batch_size']]
                users = torch.tensor([u for u, _ in batch], dtype=torch.long)
                pos = torch.tensor([i for _, i in batch], dtype=torch.long)
                neg = torch.tensor([sampler.sample(u) for u, _ in batch], dtype=torch.long)
                optimizer.zero_grad()
                loss = loss_fn(model, users, pos, neg, trial['regularization'])
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite loss')
                loss.backward()
                optimizer.step()
                total += loss.item()*len(batch)
                steps += 1
            model.eval()
            metrics, _, _ = evaluate(model, selection, popularity)
            row = {'epoch': epoch, 'train_loss': total/len(pairs), 'optimizer_steps': steps,
                   'selection_ndcg': metrics['ndcg'], 'elapsed_seconds': time.perf_counter()-started}
            curve.append(row)
            with (out/'curve.jsonl').open('a', encoding='utf-8', newline='\n') as f:
                f.write(json.dumps(row, sort_keys=True)+'\n')
            print(json.dumps({'trial': name, 'seed': seed, **row}), flush=True)
            if metrics['ndcg'] > best:
                best, best_epoch, stale = metrics['ndcg'], epoch, 0
                torch.save({'state_dict': model.state_dict(), 'trial': trial, 'identity': identity,
                            'dimensions': [nu, ni], 'best_epoch': epoch}, out/'best.pt')
            else:
                stale += 1
                if stale >= trial['patience']:
                    break
        payload = torch.load(out/'best.pt', map_location='cpu', weights_only=True)
        reloaded = build_model(trial, contract)
        reloaded.load_state_dict(payload['state_dict'], strict=True)
        reloaded.eval()
        metrics, rows, diversity = evaluate(reloaded, selection, popularity)
        if metrics['ndcg'] != best:
            raise ValueError('Reloaded development NDCG mismatch')
        write(out/'selection_predictions.json', rows)
        result = {'status': 'VERIFIED', 'trial': trial, 'identity': identity, 'best_epoch': best_epoch,
                  'actual_epochs': len(curve), 'curve': curve, 'optimizer_steps': steps,
                  'selection_metrics': metrics, 'diversity': diversity,
                  'checkpoint_sha256': digest(out/'best.pt'),
                  'predictions_sha256': digest(out/'selection_predictions.json'),
                  'parameter_count': sum(p.numel() for p in model.parameters()),
                  'elapsed_seconds': time.perf_counter()-started, 'test_evaluation': 'NOT EVALUATED',
                  'audit_evaluation': 'NOT EVALUATED', 'remote_api_requests': 0,
                  'checkpoint_reload_exact': True}
        write(out/'result.json', result)
    return result


def freeze(plan_path):
    plan, groups = plan_and_groups(plan_path)
    trials = {}
    for trial in plan['trials']:
        directory = BASE/'seed_42'/trial['name']
        result = json.loads((directory/'result.json').read_text(encoding='utf-8'))
        if result['trial'] != trial or result['identity']['plan_sha256'] != digest(plan_path) or (
                result['identity']['code_sha256'] != code_identity() or result['status'] != 'VERIFIED'):
            raise ValueError('Development results are inconsistent')
        trials[trial['name']] = {'ndcg': result['selection_metrics']['ndcg'],
                                'result_sha256': digest(directory/'result.json')}
    selected = {kind: max(sorted(t['name'] for t in plan['trials'] if t['model'] == kind),
                         key=lambda name: trials[name]['ndcg']) for kind in ('lightgcn', 'bpr')}
    value = {'plan_sha256': digest(plan_path), 'code_sha256': code_identity(), 'selected': selected,
             'all_trials': trials, 'selection_users': len(groups['fit'])+len(groups['select']),
             'audit_users': len(groups['audit']), 'audit_used_for_selection': False,
             'test_access': False, 'remote_api_requests': 0}
    write(BASE/'selection_frozen.json', value)
    print(json.dumps(value), flush=True)
    return value


def audit(plan_path):
    plan, groups = plan_and_groups(plan_path)
    frozen = json.loads((BASE/'selection_frozen.json').read_text(encoding='utf-8'))
    if frozen['plan_sha256'] != digest(plan_path) or frozen['code_sha256'] != code_identity():
        raise ValueError('Frozen selection identity mismatch')
    out = BASE/'audit'
    out.mkdir(exist_ok=False)
    torch.set_num_threads(plan['torch_threads'])
    with validation_only_files():
        contract = load_train_contract()
        cohort = load_frozen_cohort('valid', ROOT/'artifacts/data_manifest.json')
        groups_to_evaluate = {'selection': groups['fit']+groups['select'], 'audit': groups['audit'],
                              'all_valid': cohort['user_ids']}
        popularity = Counter(i for _, i in contract[4])
        runs, raw = {}, {}
        for trial in plan['trials']:
            name = trial['name']
            for seed in plan['seeds']:
                directory = BASE/f'seed_{seed}'/name
                model, original = load_verified(directory, trial, contract, digest(plan_path), seed)
                run = {'metrics': {}, 'diversity': {}, 'inference_seconds': {},
                       'training_elapsed_seconds': original['elapsed_seconds'],
                       'parameter_count': original['parameter_count'], 'best_epoch': original['best_epoch'],
                       'actual_epochs': original['actual_epochs'], 'source_result_sha256': digest(directory/'result.json')}
                for group, users in groups_to_evaluate.items():
                    tick = time.perf_counter()
                    metrics, rows, diversity = evaluate(model, subset_cohort(cohort, users), popularity)
                    run['inference_seconds'][group] = time.perf_counter()-tick
                    if group == 'selection' and metrics != original['selection_metrics']:
                        raise ValueError('Selection metrics changed during audit')
                    run['metrics'][group], run['diversity'][group] = metrics, diversity
                    raw[name, seed, group] = rows
                    path = out/f'{name}_{seed}_{group}_predictions.json'
                    write(path, rows)
                    run[group+'_predictions_sha256'] = digest(path)
                runs[f'{name}_{seed}'] = run
        aggregate, paired, bands = {}, {}, {}
        comparisons = [('gcn_user_uniform', 'gcn_edge_uniform'),
                       ('gcn_edge_residual', 'gcn_edge_uniform'),
                       ('gcn_user_residual', 'gcn_user_uniform'),
                       ('gcn_user_residual', 'gcn_edge_residual'),
                       (frozen['selected']['lightgcn'], 'gcn_edge_uniform'),
                       (frozen['selected']['lightgcn'], frozen['selected']['bpr'])]
        for group in groups_to_evaluate:
            averaged = {t['name']: average_rows([raw[t['name'], s, group] for s in plan['seeds']]) for t in plan['trials']}
            aggregate[group] = {t['name']: {field: {'mean': statistics.mean(
                runs[f"{t['name']}_{s}"]['metrics'][group][field] for s in plan['seeds']),
                'std': statistics.stdev(runs[f"{t['name']}_{s}"]['metrics'][group][field] for s in plan['seeds'])}
                for field in ('recall', 'hit', 'mrr', 'ndcg')} for t in plan['trials']}
            paired[group] = {a+'_minus_'+b: clustered_pair(averaged[a], averaged[b], 'ndcg') for a, b in comparisons}
            bands[group] = {name: strata(rows, contract[3]) for name, rows in averaged.items()}
        summary = {'status': 'VERIFIED', 'scope': 'Exploratory train/valid-only LightGCN optimization',
                   'plan_sha256': digest(plan_path), 'code_sha256': code_identity(),
                   'selection_frozen_sha256': digest(BASE/'selection_frozen.json'),
                   'selected': frozen['selected'], 'seeds': plan['seeds'], 'runs': runs,
                   'aggregate': aggregate, 'paired_ndcg': paired, 'activity_strata': bands,
                   'audit_limit': 'Previously used validation users; development audit, not independent blind test',
                   'multiple_comparisons': 'Exploratory unadjusted 95% intervals, not confirmatory claims',
                   'test_evaluation': 'NOT EVALUATED', 'remote_api_requests': 0}
        write(out/'summary.json', summary)
    print(json.dumps({'status': 'VERIFIED', 'selected': summary['selected'], 'aggregate': aggregate['audit']}), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('train', 'freeze', 'audit'))
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--trial')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.action == 'train':
        train(args.plan, args.trial, args.seed)
    elif args.action == 'freeze':
        freeze(args.plan)
    else:
        audit(args.plan)
