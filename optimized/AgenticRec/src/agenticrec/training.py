"""T08 train-only BPR-MF runner; holdout labels enter evaluator functions only."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import platform
import random
import time

import pyarrow
import pyarrow.parquet as pq
import torch

from .data import digest
from .evaluation.metrics import evaluate_cohort, rank_candidates
from .models.baselines import popularity_scores, random_ranking
from .models.bpr import BPRMF, TrainNegativeSampler, bpr_loss, load_checkpoint, save_checkpoint


ROOT = Path(__file__).resolve().parents[3]
MODEL_KEYS = {'schema_version', 'dataset', 'embedding_dim', 'learning_rate',
              'regularization', 'batch_size', 'max_epochs', 'early_stopping_patience',
              'torch_threads', 'top_k', 'selection_metric'}


def load_train_contract():
    """Only training files and mapping are visible here; no holdout path exists."""
    data_dir = ROOT / 'data/processed/ml-1m-v1'
    manifest_path = ROOT / 'artifacts/data_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    required = ('train.parquet', 'train_positive.parquet', 'id_map.json', 'split_manifest.json')
    for name in required:
        if digest(data_dir / name) != manifest['files'][name]['sha256']:
            raise ValueError('Frozen train artifact hash mismatch: ' + name)
    mapping = json.loads((data_dir / 'id_map.json').read_text(encoding='utf-8'))
    users = {int(raw): int(model) for raw, model in mapping['user_to_model'].items()}
    items = {int(raw): int(model) for raw, model in mapping['item_to_model'].items()}
    n_users, n_items = len(users), len(items)
    if sorted(users.values()) != list(range(1, n_users + 1)) or sorted(items.values()) != list(range(1, n_items + 1)):
        raise ValueError('Noncontiguous frozen IDs')
    observed = defaultdict(set)
    train = pq.read_table(data_dir / 'train.parquet', columns=['raw_user_id', 'raw_item_id'])
    for raw_user, raw_item in zip(train['raw_user_id'].to_pylist(), train['raw_item_id'].to_pylist()):
        if raw_user in users and raw_item in items:
            observed[users[raw_user]].add(items[raw_item])
    positives = pq.read_table(data_dir / 'train_positive.parquet', columns=['user_id', 'item_id'])
    edges = list(zip(positives['user_id'].to_pylist(), positives['item_id'].to_pylist()))
    if len(edges) != manifest['split']['train_counts']['graph_edges'] or any(i not in observed[u] for u, i in edges):
        raise ValueError('Train-positive graph inconsistent with rated history')
    return manifest, n_users, n_items, observed, edges


def load_frozen_cohort(split, manifest_path):
    """Evaluator-only label loader, checked against the sealed T07 summary."""
    if split not in ('valid', 'test'):
        raise ValueError('Unknown evaluation split')
    summary = json.loads((ROOT / 'reproduction/native_runs/20261006_metrics/cohort_summary.json').read_text(encoding='utf-8'))
    if summary['data_manifest_sha256'] != digest(manifest_path):
        raise ValueError('Evaluation cohort split hash mismatch')
    entry = summary['splits'][split]
    path = ROOT / entry['private_path']
    if digest(path) != entry['sha256']:
        raise ValueError('Frozen evaluation cohort hash mismatch')
    cohort = json.loads(path.read_text(encoding='utf-8'))
    if cohort['split'] != split or cohort['id_map_sha256'] != summary['id_map_sha256'] or not cohort['evaluator_only']:
        raise ValueError('Evaluation cohort identity mismatch')
    return cohort


def evaluate_model(model, cohort, k):
    """All candidate scores are generated before train-seen filtering."""
    candidates = cohort['candidate_ids']
    user_ids = [u for u in cohort['user_ids'] if cohort['relevant_by_user'][str(u)]]
    with torch.no_grad():
        item_tensor = model.items.weight[1:]
        score_rows = (model.users.weight[user_ids] @ item_tensor.T).tolist()
    predictions = {}
    for u, row in zip(user_ids, score_rows):
        scores = dict(zip(candidates, row))
        predictions[u] = rank_candidates(scores, candidates, k, cohort['seen_by_user'][str(u)])
    return evaluate_cohort(cohort['user_ids'],
                           {u: cohort['relevant_by_user'][str(u)] for u in cohort['user_ids']},
                           predictions, candidates, k,
                           {u: cohort['seen_by_user'][str(u)] for u in cohort['user_ids']})


def evaluate_simple_baselines(cohort, train_popularity, k, seed):
    candidates = cohort['candidate_ids']
    scores = popularity_scores(candidates, train_popularity)
    rankings = {'Random': {}, 'Popularity': {}}
    for u in cohort['user_ids']:
        if not cohort['relevant_by_user'][str(u)]:
            continue
        seen = cohort['seen_by_user'][str(u)]
        rankings['Random'][u] = random_ranking(candidates, set(seen), k, seed, u)
        rankings['Popularity'][u] = rank_candidates(scores, candidates, k, seen)
    relevant = {u: cohort['relevant_by_user'][str(u)] for u in cohort['user_ids']}
    observed = {u: cohort['seen_by_user'][str(u)] for u in cohort['user_ids']}
    return {name: evaluate_cohort(cohort['user_ids'], relevant, preds, candidates, k, observed)
            for name, preds in rankings.items()}


def validate_config(path):
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    if set(config) != MODEL_KEYS or config['schema_version'] != 1 or config['dataset'] != 'ml-1m-v1':
        raise ValueError('Invalid T08 config keys or dataset')
    if config['selection_metric'] != 'valid_ndcg_at_10' or config['top_k'] != 10:
        raise ValueError('T07 valid NDCG@10 is the frozen selection metric')
    for field in ('embedding_dim', 'batch_size', 'max_epochs', 'early_stopping_patience', 'torch_threads'):
        if type(config[field]) is not int or config[field] < 1:
            raise ValueError(field + ' must be a positive integer')
    if type(config['learning_rate']) not in (int, float) or not 0 < config['learning_rate'] < 1:
        raise ValueError('Invalid learning rate')
    if type(config['regularization']) not in (int, float) or not 0 <= config['regularization'] < 1:
        raise ValueError('Invalid regularization')
    return config


def _code_digest():
    paths = ['models/bpr.py', 'models/baselines.py', 'training.py', 'evaluation/metrics.py']
    source = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for relative in paths:
        h.update(relative.encode('utf-8'))
        h.update((source / relative).read_bytes())
    return h.hexdigest()


def train(config_path, seed):
    if type(seed) is not int or seed < 0:
        raise ValueError('Seed must be a nonnegative integer')
    config_path = Path(config_path)
    config = validate_config(config_path)
    torch.set_num_threads(config['torch_threads'])
    torch.manual_seed(seed)
    rng = random.Random(seed)
    manifest, n_users, n_items, seen, edges = load_train_contract()
    sampler = TrainNegativeSampler(n_items, seen, rng)
    manifest_path = ROOT / 'artifacts/data_manifest.json'
    valid = load_frozen_cohort('valid', manifest_path)
    if valid['candidate_ids'] != list(range(1, n_items + 1)) or valid['user_ids'] != list(range(1, n_users + 1)):
        raise ValueError('Valid cohort dimension mismatch')
    identity = {'id_map_sha256': manifest['id_map_sha256'],
                'split_sha256': manifest['files']['split_manifest.json']['sha256'],
                'data_manifest_sha256': digest(manifest_path), 'config_sha256': digest(config_path),
                'code_sha256': _code_digest(), 'seed': seed, 'torch_version': str(torch.__version__),
                'pyarrow_version': pyarrow.__version__, 'python_version': platform.python_version()}
    model_dir = ROOT / 'artifacts/models/bpr'
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = model_dir / ('seed_' + str(seed) + '.pt')
    result_path = ROOT / 'reports/rec_baselines' / ('seed_' + str(seed) + '.json')
    if checkpoint.exists() or result_path.exists():
        raise FileExistsError('Seed outputs already exist; refusing to overwrite evidence')
    model = BPRMF(n_users, n_items, config['embedding_dim'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    curve, best_ndcg, best_epoch, stale = [], -1.0, 0, 0
    start = time.perf_counter()
    for epoch in range(1, config['max_epochs'] + 1):
        rng.shuffle(edges)
        model.train()
        total_loss, count = 0.0, 0
        for offset in range(0, len(edges), config['batch_size']):
            batch = edges[offset:offset + config['batch_size']]
            u = torch.tensor([x[0] for x in batch], dtype=torch.long)
            pos = torch.tensor([x[1] for x in batch], dtype=torch.long)
            neg = torch.tensor([sampler.sample(x[0]) for x in batch], dtype=torch.long)
            optimizer.zero_grad()
            loss = bpr_loss(model, u, pos, neg, config['regularization'])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
            count += len(batch)
        model.eval()
        metrics = evaluate_model(model, valid, config['top_k'])
        step = {'epoch': epoch, 'mean_train_loss': total_loss / count,
                'valid_ndcg_at_10': metrics['ndcg'], 'valid_users_evaluated': metrics['users_evaluated'],
                'elapsed_seconds': time.perf_counter() - start}
        curve.append(step)
        print(json.dumps(step, sort_keys=True), flush=True)
        if metrics['ndcg'] > best_ndcg:
            best_ndcg, best_epoch, stale = metrics['ndcg'], epoch, 0
            save_checkpoint(checkpoint, model, identity)
        else:
            stale += 1
            if stale >= config['early_stopping_patience']:
                break
    best_model = load_checkpoint(checkpoint, n_users, n_items, config['embedding_dim'], identity)
    valid_metrics = evaluate_model(best_model, valid, config['top_k'])
    # Test is first read after valid-only model selection and checkpoint freeze.
    test = load_frozen_cohort('test', manifest_path)
    if test['candidate_ids'] != valid['candidate_ids'] or test['user_ids'] != valid['user_ids']:
        raise ValueError('Test cohort dimension mismatch')
    pop_count = Counter(item for _user, item in edges)
    counts_path = model_dir / ('seed_' + str(seed) + '_popularity.json')
    counts_path.write_text(json.dumps({str(i): pop_count[i] for i in range(1, n_items + 1)}, sort_keys=True) + '\n', encoding='utf-8')
    baselines = {split: evaluate_simple_baselines(cohort, pop_count, config['top_k'], seed)
                 for split, cohort in (('valid', valid), ('test', test))}
    output = {'status': 'PASS', 'task': 'T08', 'scope': 'MovieLens1M independent warm full-universe',
              'seed': seed, 'config': config, 'identity': identity,
              'train_positive_edges': len(edges), 'train_seen_users': len(seen),
              'candidate_items': n_items, 'warm_users': n_users,
              'negative_sampling': 'uniform train candidate IDs excluding every train-rated item; future labels not consulted',
              'regularization_normalization': 'mean per triple of (user+positive+negative embedding squared L2)/2',
              'selection': {'metric': 'valid_ndcg_at_10', 'best_epoch': best_epoch,
                            'early_stopped': stale >= config['early_stopping_patience']},
              'curve': curve, 'checkpoint_sha256': digest(checkpoint),
              'popularity_counts_sha256': digest(counts_path),
              'cohort_sha256': {s: digest(ROOT / ('data/eval_private/ml-1m-v1/' + s + '.json')) for s in ('valid', 'test')},
              'metrics': {'valid': {**baselines['valid'], 'BPR-MF': valid_metrics},
                          'test': {**baselines['test'], 'BPR-MF': evaluate_model(best_model, test, config['top_k'])}},
              'elapsed_seconds': time.perf_counter() - start,
              'test_used_for_selection': False, 'remote_api_requests': 0}
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(output, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'result': result_path.relative_to(ROOT).as_posix(),
                      'best_epoch': best_epoch, 'metrics': output['metrics']}, sort_keys=True), flush=True)
    return output
