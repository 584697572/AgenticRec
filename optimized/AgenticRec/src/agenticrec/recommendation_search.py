"""Append-only, validation-only BPR/LightGCN search after historical test exposure.

Uses unchanged baseline models, sampling, graph and metric implementations.
Each preregistered trial owns a new directory; never replaces serving weights.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import time

import pyarrow
import torch

from .data import digest
from .evaluation.metrics import evaluate_cohort, rank_candidates, score_ranking
from .lightgcn_diagnostic import validation_only_files
from .lightgcn_training import _code_digest
from .models.bpr import BPRMF, TrainNegativeSampler, bpr_loss
from .models.lightgcn import LightGCN, lightgcn_loss, normalized_bipartite_graph
from .training import ROOT, load_frozen_cohort, load_train_contract


def validate_plan(plan):
    keys = {'schema_version', 'run_id', 'purpose', 'seed', 'torch_threads', 'top_k',
            'selection_metric', 'test_access', 'remote_api_requests', 'trials'}
    if set(plan) != keys or type(plan['schema_version']) is not int or plan['schema_version'] != 1:
        raise ValueError('Invalid search schema')
    if plan['test_access'] is not False or type(plan['remote_api_requests']) is not int or plan['remote_api_requests'] != 0:
        raise ValueError('Search permits only train/valid with zero API calls')
    if plan['selection_metric'] != 'valid_ndcg_at_10' or type(plan['top_k']) is not int or plan['top_k'] != 10:
        raise ValueError('Frozen selection metric is valid NDCG@10')
    if not isinstance(plan['purpose'], str) or not plan['purpose']:
        raise ValueError('Purpose is required')
    def name(value):
        return isinstance(value, str) and value and all(c in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in value)
    if not name(plan['run_id']):
        raise ValueError('Invalid run ID')
    for field in ('seed', 'torch_threads'):
        if type(plan[field]) is not int or plan[field] < (0 if field == 'seed' else 1):
            raise ValueError('Invalid ' + field)
    if not isinstance(plan['trials'], list) or not plan['trials']:
        raise ValueError('No trials')
    names = set()
    trial_keys = {'name', 'model', 'layers', 'embedding_dim', 'learning_rate',
                  'regularization', 'batch_size', 'max_epochs', 'patience'}
    for trial in plan['trials']:
        if set(trial) != trial_keys or not name(trial['name']) or trial['name'] in names:
            raise ValueError('Invalid or duplicate trial')
        names.add(trial['name'])
        if trial['model'] not in ('bpr', 'lightgcn') or type(trial['layers']) is not int:
            raise ValueError('Invalid model')
        if (trial['model'] == 'bpr' and trial['layers'] != 0
                or trial['model'] == 'lightgcn' and trial['layers'] not in (1, 2, 3)):
            raise ValueError('Invalid layer count')
        for field in ('embedding_dim', 'batch_size', 'max_epochs', 'patience'):
            if type(trial[field]) is not int or trial[field] < 1:
                raise ValueError('Invalid ' + field)
        for field in ('learning_rate', 'regularization'):
            value = trial[field]
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value < 1:
                raise ValueError('Invalid ' + field)
        if trial['learning_rate'] == 0:
            raise ValueError('Zero learning rate')
    return plan


def evaluate(model, cohort, train_popularity):
    """Same full-universe ranking, with four private per-user metric values."""
    users = [u for u in cohort['user_ids'] if cohort['relevant_by_user'][str(u)]]
    with torch.no_grad():
        if isinstance(model, LightGCN):
            ue, ie = model.propagate()
            scores = (ue[torch.tensor(users, dtype=torch.long) - 1] @ ie.T).tolist()
        else:
            scores = (model.users.weight[users] @ model.items.weight[1:].T).tolist()
    predictions, rows = {}, {}
    hot = set(sorted(train_popularity, key=lambda i: (-train_popularity[i], i))[:100])
    hot_slots, slots, unique = 0, 0, set()
    for user, values in zip(users, scores):
        key = str(user)
        prediction = rank_candidates(dict(zip(cohort['candidate_ids'], values)),
                                     cohort['candidate_ids'], 10, cohort['seen_by_user'][key])
        predictions[user] = prediction
        rows[key] = {**score_ranking(prediction, cohort['relevant_by_user'][key],
                                    cohort['candidate_ids'], 10, cohort['seen_by_user'][key]),
                     'prediction': prediction}
        hot_slots += sum(item in hot for item in prediction)
        slots += len(prediction)
        unique.update(prediction)
    metrics = evaluate_cohort(cohort['user_ids'],
        {u: cohort['relevant_by_user'][str(u)] for u in cohort['user_ids']},
        predictions, cohort['candidate_ids'], 10,
        {u: cohort['seen_by_user'][str(u)] for u in cohort['user_ids']})
    return metrics, rows, {'distinct_recommended_items': len(unique),
                          'train_hot100_slot_share': hot_slots / slots if slots else None}


def run(plan_path, trial_name):
    plan_path = Path(plan_path)
    plan = validate_plan(json.loads(plan_path.read_text(encoding='utf-8')))
    matches = [t for t in plan['trials'] if t['name'] == trial_name]
    if len(matches) != 1:
        raise ValueError('Trial must be registered before execution')
    trial = matches[0]
    out = ROOT / 'artifacts/runs/recommendation_search' / plan['run_id'] / trial_name
    out.mkdir(parents=True, exist_ok=False)
    (out / 'plan.json').write_bytes(plan_path.read_bytes())
    torch.set_num_threads(plan['torch_threads'])
    torch.manual_seed(plan['seed'])
    rng = random.Random(plan['seed'])
    started = time.perf_counter()
    with validation_only_files():
        manifest, n_users, n_items, seen, edges = load_train_contract()
        valid = load_frozen_cohort('valid', ROOT / 'artifacts/data_manifest.json')
        if valid['candidate_ids'] != list(range(1, n_items+1)) or valid['user_ids'] != list(range(1, n_users+1)):
            raise ValueError('Frozen valid IDs do not match training contract')
        identity = {'seed': plan['seed'], 'plan_sha256': digest(plan_path),
                    'code_sha256': hashlib.sha256((_code_digest() + digest(Path(__file__))
                        + digest(Path(__file__).with_name('lightgcn_diagnostic.py'))).encode()).hexdigest(),
                    'id_map_sha256': manifest['id_map_sha256'],
                    'split_sha256': manifest['files']['split_manifest.json']['sha256'],
                    'train_positive_sha256': manifest['files']['train_positive.parquet']['sha256'],
                    'data_manifest_sha256': digest(ROOT / 'artifacts/data_manifest.json'),
                    'valid_cohort_sha256': digest(ROOT / 'data/eval_private/ml-1m-v1/valid.json'),
                    'torch_version': str(torch.__version__), 'pyarrow_version': pyarrow.__version__,
                    'python_version': platform.python_version(), 'platform': platform.platform()}
        (out / 'identity.json').write_text(json.dumps(identity, indent=2, sort_keys=True)+'\n', encoding='utf-8')
        graph = normalized_bipartite_graph(n_users, n_items, edges) if trial['model'] == 'lightgcn' else None
        def build():
            if graph is None:
                return BPRMF(n_users, n_items, trial['embedding_dim'])
            return LightGCN(n_users, n_items, trial['embedding_dim'], trial['layers'], graph)
        model = build()
        sampler = TrainNegativeSampler(n_items, seen, rng)
        popularity = Counter(i for _, i in edges)
        optimizer = torch.optim.Adam(model.parameters(), lr=trial['learning_rate'])
        loss_fn = bpr_loss if graph is None else lightgcn_loss
        best, best_epoch, stale, steps, curve = -1.0, 0, 0, 0, []
        checkpoint = out / 'best.pt'
        for epoch in range(1, trial['max_epochs']+1):
            rng.shuffle(edges)
            model.train()
            total_loss = 0.0
            for offset in range(0, len(edges), trial['batch_size']):
                batch = edges[offset:offset+trial['batch_size']]
                users = torch.tensor([u for u, _ in batch], dtype=torch.long)
                positive = torch.tensor([i for _, i in batch], dtype=torch.long)
                negative = torch.tensor([sampler.sample(u) for u, _ in batch], dtype=torch.long)
                optimizer.zero_grad()
                loss = loss_fn(model, users, positive, negative, trial['regularization'])
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                loss.backward()
                optimizer.step()
                total_loss += loss.item()*len(batch)
                steps += 1
            model.eval()
            metrics, _rows, _diversity = evaluate(model, valid, popularity)
            row = {'epoch': epoch, 'mean_train_loss': total_loss/len(edges),
                   'optimizer_steps': steps, 'valid_metrics': metrics,
                   'elapsed_seconds': time.perf_counter()-started}
            curve.append(row)
            with (out / 'curve.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, sort_keys=True)+'\n')
            print(json.dumps({'trial': trial_name, **row}, sort_keys=True), flush=True)
            if metrics['ndcg'] > best:
                best, best_epoch, stale = metrics['ndcg'], epoch, 0
                torch.save({'state_dict': model.state_dict(), 'identity': identity,
                            'trial': trial, 'best_epoch': best_epoch,
                            'dimensions': [n_users, n_items]}, checkpoint)
            else:
                stale += 1
                if stale >= trial['patience']:
                    break
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if payload['identity'] != identity or payload['trial'] != trial:
            raise ValueError('Checkpoint provenance differs')
        reloaded = build()
        reloaded.load_state_dict(payload['state_dict'], strict=True)
        reloaded.eval()
        metrics, rows, diversity = evaluate(reloaded, valid, popularity)
        if metrics != curve[best_epoch-1]['valid_metrics']:
            raise ValueError('Reloaded best checkpoint metrics differ')
        prediction_path = out / 'valid_predictions.json'
        prediction_path.write_text(json.dumps(rows, sort_keys=True)+'\n', encoding='utf-8')
        result = {'status': 'VERIFIED', 'scope': plan['purpose'], 'identity': identity,
                  'trial': trial, 'seed': plan['seed'], 'best_epoch': best_epoch,
                  'actual_epochs': len(curve), 'early_stopped': stale >= trial['patience'],
                  'optimizer_steps': steps, 'curve': curve, 'valid_metrics': metrics,
                  'diversity': diversity, 'checkpoint_path': checkpoint.relative_to(ROOT).as_posix(),
                  'checkpoint_sha256': digest(checkpoint), 'predictions_sha256': digest(prediction_path),
                  'checkpoint_reload_metrics_exact': True, 'train_positive_edges': len(edges),
                  'elapsed_seconds': time.perf_counter()-started,
                  'test_evaluation': 'NOT EVALUATED', 'remote_api_requests': 0}
        (out / 'result.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    print(json.dumps({'status': 'VERIFIED', 'trial': trial_name, 'best_epoch': best_epoch,
                      'valid_metrics': metrics}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--trial', required=True)
    args = parser.parse_args()
    run(args.plan, args.trial)
