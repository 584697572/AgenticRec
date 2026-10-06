"""T09 LightGCN: train graph only, valid-only selection, sealed test once."""
import hashlib
import json
from pathlib import Path
import platform
import random
import time

import pyarrow
import torch

from .data import digest
from .evaluation.metrics import evaluate_cohort, rank_candidates
from .models.bpr import TrainNegativeSampler
from .models.lightgcn import (LightGCN, lightgcn_loss, load_lightgcn_checkpoint,
                              normalized_bipartite_graph, save_lightgcn_checkpoint)
from .training import ROOT, load_frozen_cohort, load_train_contract


CONFIG_KEYS = {'schema_version', 'model', 'dataset', 'embedding_dim', 'layers_to_try',
               'learning_rate', 'regularization', 'batch_size', 'max_epochs',
               'early_stopping_patience', 'torch_threads', 'top_k', 'selection_metric'}
AUTHOR_REFERENCE = '947ca2b3b1d2d3545b114145710cb06c4e57b3d2'


def validate_config(path):
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    if set(config) != CONFIG_KEYS or config['schema_version'] != 1 or config['model'] != 'lightgcn' or config['dataset'] != 'ml-1m-v1':
        raise ValueError('Invalid LightGCN config identity or keys')
    if config['selection_metric'] != 'valid_ndcg_at_10' or config['top_k'] != 10:
        raise ValueError('Frozen T07 selection metric must be valid NDCG@10')
    if config['layers_to_try'] != [1, 2, 3]:
        raise ValueError('T09 initial layer experiment must compare 1/2/3 layers')
    for name in ('embedding_dim', 'batch_size', 'max_epochs', 'early_stopping_patience', 'torch_threads'):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(name + ' must be positive integer')
    if type(config['learning_rate']) not in (int, float) or not 0 < config['learning_rate'] < 1:
        raise ValueError('Invalid learning rate')
    if type(config['regularization']) not in (int, float) or not 0 <= config['regularization'] < 1:
        raise ValueError('Invalid regularization')
    return config


def _code_digest():
    source = Path(__file__).resolve().parent
    paths = ['models/lightgcn.py', 'models/bpr.py', 'lightgcn_training.py',
             'training.py', 'evaluation/metrics.py']
    h = hashlib.sha256()
    for relative in paths:
        h.update(relative.encode('utf-8'))
        h.update((source / relative).read_bytes())
    return h.hexdigest()


def evaluate_lightgcn(model, cohort, k):
    candidates = cohort['candidate_ids']
    user_ids = [u for u in cohort['user_ids'] if cohort['relevant_by_user'][str(u)]]
    with torch.no_grad():
        users, items = model.propagate()
        score_rows = (users[torch.tensor(user_ids) - 1] @ items.T).tolist()
    predictions = {}
    for u, row in zip(user_ids, score_rows):
        predictions[u] = rank_candidates(dict(zip(candidates, row)), candidates,
                                         k, cohort['seen_by_user'][str(u)])
    return evaluate_cohort(cohort['user_ids'],
                           {u: cohort['relevant_by_user'][str(u)] for u in cohort['user_ids']},
                           predictions, candidates, k,
                           {u: cohort['seen_by_user'][str(u)] for u in cohort['user_ids']})


def train(config_path, seed):
    if type(seed) is not int or seed < 0:
        raise ValueError('Seed must be nonnegative integer')
    config_path = Path(config_path)
    config = validate_config(config_path)
    torch.set_num_threads(config['torch_threads'])
    manifest, n_users, n_items, train_seen, frozen_edges = load_train_contract()
    graph = normalized_bipartite_graph(n_users, n_items, frozen_edges)
    if graph._nnz() != 2 * len(frozen_edges):
        raise ValueError('Graph edge count mismatch')
    manifest_path = ROOT / 'artifacts/data_manifest.json'
    valid = load_frozen_cohort('valid', manifest_path)
    if valid['candidate_ids'] != list(range(1, n_items + 1)) or valid['user_ids'] != list(range(1, n_users + 1)):
        raise ValueError('Valid cohort does not match train graph IDs')
    model_dir = ROOT / 'artifacts/models/lightgcn'
    result_path = ROOT / 'reports/rec_baselines' / ('lightgcn_seed_' + str(seed) + '.json')
    checkpoints = {layers: model_dir / ('seed_' + str(seed) + '_layer_' + str(layers) + '.pt')
                   for layers in config['layers_to_try']}
    baseline_path = ROOT / 'reports/rec_baselines' / ('seed_' + str(seed) + '.json')
    if result_path.exists() or any(path.exists() for path in checkpoints.values()):
        raise FileExistsError('T09 seed outputs already exist; refusing to overwrite evidence')
    if not baseline_path.exists():
        raise FileNotFoundError('Same-seed T08 baseline report required before T09 comparison')
    model_dir.mkdir(parents=True, exist_ok=True)
    identity = {'id_map_sha256': manifest['id_map_sha256'],
                'split_sha256': manifest['files']['split_manifest.json']['sha256'],
                'train_positive_sha256': manifest['files']['train_positive.parquet']['sha256'],
                'data_manifest_sha256': digest(manifest_path), 'config_sha256': digest(config_path),
                'code_sha256': _code_digest(), 'seed': seed,
                'torch_version': str(torch.__version__), 'pyarrow_version': pyarrow.__version__,
                'python_version': platform.python_version()}
    start = time.perf_counter()
    trials = {}
    for layers in config['layers_to_try']:
        torch.manual_seed(seed)
        rng = random.Random(seed)
        edges = list(frozen_edges)
        sampler = TrainNegativeSampler(n_items, train_seen, rng)
        model = LightGCN(n_users, n_items, config['embedding_dim'], layers, graph)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        curve, best_ndcg, best_epoch, stale = [], -1.0, 0, 0
        for epoch in range(1, config['max_epochs'] + 1):
            rng.shuffle(edges)
            model.train()
            total_loss, count = 0.0, 0
            for offset in range(0, len(edges), config['batch_size']):
                batch = edges[offset:offset + config['batch_size']]
                users = torch.tensor([u for u, _ in batch], dtype=torch.long)
                positives = torch.tensor([i for _, i in batch], dtype=torch.long)
                negatives = torch.tensor([sampler.sample(u) for u, _ in batch], dtype=torch.long)
                optimizer.zero_grad()
                loss = lightgcn_loss(model, users, positives, negatives, config['regularization'])
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch)
                count += len(batch)
            model.eval()
            metrics = evaluate_lightgcn(model, valid, config['top_k'])
            step = {'layers': layers, 'epoch': epoch, 'mean_train_loss': total_loss / count,
                    'valid_ndcg_at_10': metrics['ndcg'], 'train_edges_seen': count,
                    'elapsed_seconds': time.perf_counter() - start}
            curve.append(step)
            print(json.dumps(step, sort_keys=True), flush=True)
            if metrics['ndcg'] > best_ndcg:
                best_ndcg, best_epoch, stale = metrics['ndcg'], epoch, 0
                save_lightgcn_checkpoint(checkpoints[layers], model, identity, config)
            else:
                stale += 1
                if stale >= config['early_stopping_patience']:
                    break
        best_model = load_lightgcn_checkpoint(checkpoints[layers], graph, n_users, n_items,
                                              config['embedding_dim'], layers, identity, config)
        best_metrics = evaluate_lightgcn(best_model, valid, config['top_k'])
        trials[str(layers)] = {'best_epoch': best_epoch, 'early_stopped': stale >= config['early_stopping_patience'],
                               'valid_metrics': best_metrics, 'curve': curve,
                               'checkpoint_sha256': digest(checkpoints[layers])}
    # Choose layers only from valid; ties prefer fewer layers. No test was loaded above.
    selected = max(config['layers_to_try'], key=lambda layer: (trials[str(layer)]['valid_metrics']['ndcg'], -layer))
    chosen = load_lightgcn_checkpoint(checkpoints[selected], graph, n_users, n_items,
                                       config['embedding_dim'], selected, identity, config)
    test = load_frozen_cohort('test', manifest_path)
    if test['candidate_ids'] != valid['candidate_ids'] or test['user_ids'] != valid['user_ids']:
        raise ValueError('Test cohort dimension mismatch')
    test_metrics = evaluate_lightgcn(chosen, test, config['top_k'])
    baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
    if (baseline['seed'] != seed or baseline['identity']['id_map_sha256'] != identity['id_map_sha256']
            or baseline['identity']['split_sha256'] != identity['split_sha256']):
        raise ValueError('T08 baseline used a different frozen protocol')
    result = {'status': 'PASS', 'task': 'T09', 'scope': 'MovieLens1M independent warm full-universe',
              'config': config, 'seed': seed, 'identity': identity, 'graph': {
                  'source': 'train_positive.parquet only', 'edge_count': len(frozen_edges),
                  'normalized_sparse_nonzeros': graph._nnz(), 'shape': list(graph.shape),
                  'layout': 'sparse_coo', 'degree_zero_nodes': int((torch.bincount(graph.indices()[0], minlength=n_users+n_items) == 0).sum()),
                  'author_reference_commit': AUTHOR_REFERENCE},
              'trials': trials, 'selection': {'metric': 'valid_ndcg_at_10',
                                               'selected_layers': selected,
                                               'best_epoch': trials[str(selected)]['best_epoch']},
              'test_metrics': test_metrics, 'baseline_report_sha256': digest(baseline_path),
              'baseline_test_metrics': baseline['metrics']['test'],
              'cohort_sha256': {'valid': digest(ROOT / 'data/eval_private/ml-1m-v1/valid.json'),
                                'test': digest(ROOT / 'data/eval_private/ml-1m-v1/test.json')},
              'test_used_for_training_or_selection': False,
              'elapsed_seconds': time.perf_counter() - start, 'remote_api_requests': 0}
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'result': result_path.relative_to(ROOT).as_posix(),
                      'selected_layers': selected, 'test_metrics': test_metrics}, sort_keys=True), flush=True)
    return result
