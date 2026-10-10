"""Freeze evaluation labels apart from model training input."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from agenticrec.evaluation.metrics import evaluate_cohort


ROOT = Path(__file__).resolve().parents[4]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fixed_cohort(user_to_model, item_to_model, train_rows, holdout_rows):
    """Training supplies users/candidates/seen; holdout supplies labels only."""
    users = sorted(user_to_model.values())
    candidates = sorted(item_to_model.values())
    if users != list(range(1, len(users) + 1)) or candidates != list(range(1, len(candidates) + 1)):
        raise ValueError('Noncontiguous frozen model IDs')
    seen = defaultdict(set)
    for row in train_rows:
        user = user_to_model.get(str(row['raw_user_id']))
        item = item_to_model.get(str(row['raw_item_id']))
        if user is not None and item is not None:
            seen[user].add(item)
    relevant = defaultdict(set)
    for row in holdout_rows:
        if row['rating'] < 4:
            continue
        user = user_to_model.get(str(row['raw_user_id']))
        item = item_to_model.get(str(row['raw_item_id']))
        if user is None or item is None:
            continue
        if item in seen[user] or item in relevant[user]:
            raise ValueError('Holdout item overlaps train history or repeats')
        relevant[user].add(item)
    result = {'schema_version': 1, 'candidate_ids': candidates, 'user_ids': users,
              'seen_by_user': {str(u): sorted(seen[u]) for u in users},
              'relevant_by_user': {str(u): sorted(relevant[u]) for u in users}}
    zeros = {user: [] for user in users}
    metrics = evaluate_cohort(users, {u: relevant[u] for u in users}, zeros,
                              candidates, 10, seen_by_user=seen)
    result['coverage_only'] = {'warm_users': len(users),
                               'users_with_relevance': metrics['users_evaluated'],
                               'users_without_relevance': metrics['users_without_relevance'],
                               'relevant_items': sum(len(x) for x in relevant.values()),
                               'candidate_items': len(candidates)}
    return result


def freeze():
    import pyarrow.parquet as pq

    data_dir = ROOT / 'data/processed/ml-1m-v1'
    manifest = ROOT / 'artifacts/data_manifest.json'
    out = ROOT / 'data/eval_private/ml-1m-v1'
    public = ROOT / 'reproduction/native_runs/20261006_metrics/cohort_summary.json'
    if out.exists():
        raise FileExistsError('Frozen evaluation cohorts already exist')
    m = json.loads(manifest.read_text(encoding='utf-8'))
    mapping_path = data_dir / 'id_map.json'
    assert sha(mapping_path) == m['id_map_sha256']
    for name in ('train.parquet', 'eval_valid.parquet', 'eval_test.parquet'):
        assert sha(data_dir / name) == m['files'][name]['sha256']
    ids = json.loads(mapping_path.read_text(encoding='utf-8'))
    train = pq.read_table(data_dir / 'train.parquet').to_pylist()
    encoded_cohorts = {}
    summary = {'status': 'PASS', 'protocol': 'fixed MovieLens1M warm full-universe cohorts',
               'data_manifest_sha256': sha(manifest), 'id_map_sha256': m['id_map_sha256'],
               'test_used_for_training_or_tuning': False, 'model_metrics': None,
               'splits': {}}
    for split in ('valid', 'test'):
        labels = pq.read_table(data_dir / ('eval_' + split + '.parquet')).to_pylist()
        cohort = fixed_cohort(ids['user_to_model'], ids['item_to_model'], train, labels)
        cohort['split'] = split
        cohort['data_manifest_sha256'] = sha(manifest)
        cohort['id_map_sha256'] = m['id_map_sha256']
        cohort['holdout_sha256'] = m['files']['eval_' + split + '.parquet']['sha256']
        cohort['evaluator_only'] = True
        path = out / (split + '.json')
        encoded = (json.dumps(cohort, separators=(',', ':'), sort_keys=True) + '\n').encode('utf-8')
        encoded_cohorts[path] = encoded
        summary['splits'][split] = {'private_path': path.relative_to(ROOT).as_posix(),
                                    'sha256': hashlib.sha256(encoded).hexdigest(), **cohort['coverage_only']}
    # Public summaries ship in Git; private labels do not. Restore only the
    # exact sealed labels, validate both splits before writing, preserve summary.
    if public.exists() and json.loads(public.read_text(encoding='utf-8')) != summary:
        raise ValueError('Rebuilt cohorts differ from the published frozen summary')
    out.mkdir(parents=True)
    for path, encoded in encoded_cohorts.items():
        with path.open('xb') as stream:
            stream.write(encoded)
    if not public.exists():
        public.parent.mkdir(parents=True, exist_ok=True)
        with public.open('x', encoding='utf-8', newline='\n') as f:
            json.dump(summary, f, indent=2, sort_keys=True)
            f.write('\n')
    print(json.dumps(summary, sort_keys=True))


if __name__ == '__main__':
    freeze()
