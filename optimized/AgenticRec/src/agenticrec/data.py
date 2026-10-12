"""Frozen MovieLens 1M data contract. No model or holdout-derived IDs."""
from collections import Counter, defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile


ROOT = Path(__file__).resolve().parents[3]
ALLOWED_KEYS = {'schema_version', 'seed', 'source_url', 'source_zip_sha256',
                'source_readme_sha256', 'rating_positive_at_least',
                'train_fraction', 'valid_fraction', 'min_train_positives',
                'movie_title_encoding', 'output_subdir'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def choose_time_boundaries(rows, train_fraction=0.8, valid_fraction=0.1):
    """The timestamp at each rank starts the next window, including all ties."""
    if not (0 < train_fraction < 1 and 0 < valid_fraction < 1 - train_fraction):
        raise ValueError('Invalid global time fractions')
    times = sorted(r[3] for r in rows)
    if len(times) < 3:
        raise ValueError('At least three interactions required')
    low = times[int(len(times) * train_fraction)]
    high = times[int(len(times) * (train_fraction + valid_fraction))]
    if low >= high or times[0] >= low or high > times[-1]:
        raise ValueError('Time boundaries collapse after keeping ties together')
    return low, high


def split_interactions(rows, valid_start, test_start):
    train, valid, test = [], [], []
    for row in rows:
        timestamp = row[3]
        (train if timestamp < valid_start else valid if timestamp < test_start else test).append(row)
    if not train or not valid or not test:
        raise ValueError('Empty temporal window')
    if not (max(r[3] for r in train) < min(r[3] for r in valid)
            <= max(r[3] for r in valid) < min(r[3] for r in test)):
        raise ValueError('Time leakage across windows')
    return train, valid, test


def build_contract(train, valid, test, min_train_positives=5, positive_at_least=4):
    """Maps, graph edges, popularity and observed history depend on train only."""
    if min_train_positives < 1 or positive_at_least < 1:
        raise ValueError('Invalid train-only filters')
    positives = [r for r in train if r[2] >= positive_at_least]
    counts = Counter(r[0] for r in positives)
    users = {raw: i for i, raw in enumerate(sorted(u for u, n in counts.items()
             if n >= min_train_positives), 1)}
    items = {raw: i for i, raw in enumerate(sorted({r[1] for r in positives}), 1)}
    if not users or not items:
        raise ValueError('No warm users or candidate items in train')
    edges = [r for r in positives if r[0] in users and r[1] in items]
    observed = defaultdict(set)
    for user, item, _rating, _timestamp in train:
        observed[user].add(item)

    def labels(rows):
        output = []
        for user, item, rating, timestamp in rows:
            warm_user, warm_item = user in users, item in items
            segment = ('WARM' if warm_user and warm_item else 'COLD_USER'
                       if not warm_user and warm_item else 'COLD_ITEM'
                       if warm_user and not warm_item else 'COLD_BOTH')
            output.append({'raw_user_id': user, 'raw_item_id': item,
                           'rating': rating, 'timestamp': timestamp,
                           'user_id': users.get(user), 'item_id': items.get(item),
                           'segment': segment, 'is_target': rating >= positive_at_least})
        return output

    return {'users': users, 'items': items, 'train_edges': edges,
            'observed_by_user': dict(observed), 'valid_labels': labels(valid),
            'test_labels': labels(test), 'train_popularity': dict(Counter(r[1] for r in edges))}


def parse_archive(path, movie_encoding):
    if movie_encoding != 'iso-8859-1':
        raise ValueError('The verified MovieLens1M movie encoding is iso-8859-1')
    with zipfile.ZipFile(path) as archive:
        expected = {'ml-1m/ratings.dat', 'ml-1m/movies.dat', 'ml-1m/users.dat', 'ml-1m/README'}
        if not expected <= set(archive.namelist()):
            raise ValueError('Official MovieLens1M members missing')
        movies_raw = archive.read('ml-1m/movies.dat')
        ratings_raw = archive.read('ml-1m/ratings.dat')
        users_raw = archive.read('ml-1m/users.dat')
        readme_raw = archive.read('ml-1m/README')
    movies = {}
    for line in movies_raw.decode('iso-8859-1').splitlines():
        parts = line.split('::')
        if len(parts) != 3:
            raise ValueError('Invalid movie record')
        raw_id = int(parts[0])
        if raw_id <= 0 or raw_id in movies:
            raise ValueError('Duplicate or invalid movie ID')
        match = re.fullmatch(r'(.+) \((\d{4})\)', parts[1])
        movies[raw_id] = {'raw_item_id': raw_id, 'title': parts[1],
                          'genres': parts[2].split('|'),
                          'year': int(match[2]) if match else None,
                          'metadata_source': 'official_movielens_1m_movies.dat'}
    user_ids = set()
    for line in users_raw.decode('ascii').splitlines():
        parts = line.split('::')
        if len(parts) != 5:
            raise ValueError('Invalid user record')
        user_ids.add(int(parts[0]))
    ratings, keys = [], set()
    for line in ratings_raw.decode('ascii').splitlines():
        parts = line.split('::')
        if len(parts) != 4:
            raise ValueError('Invalid rating record')
        row = tuple(map(int, parts))
        user, item, rating, timestamp = row
        if user not in user_ids or item not in movies or not 1 <= rating <= 5 or timestamp <= 0:
            raise ValueError('Rating references unknown ID or invalid value')
        key = (user, item)
        if key in keys:
            raise ValueError('Duplicate user-item interaction key')
        keys.add(key)
        ratings.append(row)
    return movies, ratings, {'movies.dat': len(movies_raw), 'ratings.dat': len(ratings_raw),
                              'users.dat': len(users_raw), 'README': len(readme_raw),
                              'users': len(user_ids)}


def arrow_interactions(rows):
    import pyarrow as pa
    return pa.table({'raw_user_id': pa.array((r[0] for r in rows), type=pa.int32()),
                     'raw_item_id': pa.array((r[1] for r in rows), type=pa.int32()),
                     'rating': pa.array((r[2] for r in rows), type=pa.int8()),
                     'timestamp': pa.array((r[3] for r in rows), type=pa.int64())})


def save_json(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write('\n')


def prepare(config_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if set(config) != ALLOWED_KEYS or config['schema_version'] != 1 or config['seed'] != 42:
        raise ValueError('Invalid frozen data config')
    if config['source_url'] != 'https://files.grouplens.org/datasets/movielens/ml-1m.zip':
        raise ValueError('Unexpected data source')
    if (config['rating_positive_at_least'] != 4 or config['min_train_positives'] != 5
            or config['train_fraction'] != 0.8 or config['valid_fraction'] != 0.1
            or config['output_subdir'] != 'ml-1m-v1'):
        raise ValueError('Protocol differs from frozen T06 configuration')
    out = ROOT / 'data/processed' / config['output_subdir']
    manifest_path = ROOT / 'artifacts/data_manifest.json'
    if out.exists() or manifest_path.exists():
        raise FileExistsError('Frozen data outputs exist; refusing overwrite')
    zip_path = ROOT / 'data/raw/ml-1m/ml-1m.zip'
    terms = ROOT / 'data/raw/ml-1m/ml-1m-README.txt'
    checksum = ROOT / 'data/raw/ml-1m/ml-1m.zip.md5'
    if digest(zip_path) != config['source_zip_sha256'] or digest(terms) != config['source_readme_sha256']:
        raise ValueError('Official source or terms digest mismatch')
    match = re.fullmatch(r'MD5 \(ml-1m\.zip\) = ([0-9a-f]{32})\s*',
                         checksum.read_text(encoding='ascii'))
    if not match or hashlib.md5(zip_path.read_bytes()).hexdigest() != match[1]:
        raise ValueError('Official MovieLens checksum mismatch')

    movies, rows, member_info = parse_archive(zip_path, config['movie_title_encoding'])
    valid_start, test_start = choose_time_boundaries(rows, config['train_fraction'],
                                                       config['valid_fraction'])
    train, valid, test = split_interactions(rows, valid_start, test_start)
    train_keys = {(r[0], r[1]) for r in train}
    valid_keys = {(r[0], r[1]) for r in valid}
    test_keys = {(r[0], r[1]) for r in test}
    if train_keys & valid_keys or train_keys & test_keys or valid_keys & test_keys:
        raise ValueError('Duplicate interaction across windows')
    contract = build_contract(train, valid, test, config['min_train_positives'],
                              config['rating_positive_at_least'])
    out.mkdir(parents=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    file_info = {}

    def write_parquet(name, table):
        path = out / name
        pq.write_table(table, path, compression='zstd')
        file_info[name] = {'sha256': digest(path), 'bytes': path.stat().st_size,
                           'rows': table.num_rows}

    write_parquet('interactions.parquet', arrow_interactions(rows))
    for name, group in (('train', train), ('valid', valid), ('test', test)):
        write_parquet(name + '.parquet', arrow_interactions(group))
    write_parquet('train_positive.parquet', pa.table({
        'raw_user_id': pa.array((r[0] for r in contract['train_edges']), type=pa.int32()),
        'raw_item_id': pa.array((r[1] for r in contract['train_edges']), type=pa.int32()),
        'user_id': pa.array((contract['users'][r[0]] for r in contract['train_edges']), type=pa.int32()),
        'item_id': pa.array((contract['items'][r[1]] for r in contract['train_edges']), type=pa.int32()),
        'rating': pa.array((r[2] for r in contract['train_edges']), type=pa.int8()),
        'timestamp': pa.array((r[3] for r in contract['train_edges']), type=pa.int64())}))
    metadata = [movies[i] for i in sorted(movies)]
    write_parquet('items.parquet', pa.table({
        'raw_item_id': pa.array((m['raw_item_id'] for m in metadata), type=pa.int32()),
        'item_id': pa.array((contract['items'].get(m['raw_item_id']) for m in metadata), type=pa.int32()),
        'title': pa.array((m['title'] for m in metadata), type=pa.string()),
        'genres': pa.array((m['genres'] for m in metadata), type=pa.list_(pa.string())),
        'year': pa.array((m['year'] for m in metadata), type=pa.int16()),
        'metadata_source': pa.array((m['metadata_source'] for m in metadata), type=pa.string())}))
    for name in ('valid', 'test'):
        labels = contract[name + '_labels']
        write_parquet('eval_' + name + '.parquet', pa.table({
            'raw_user_id': pa.array((x['raw_user_id'] for x in labels), type=pa.int32()),
            'raw_item_id': pa.array((x['raw_item_id'] for x in labels), type=pa.int32()),
            'user_id': pa.array((x['user_id'] for x in labels), type=pa.int32()),
            'item_id': pa.array((x['item_id'] for x in labels), type=pa.int32()),
            'rating': pa.array((x['rating'] for x in labels), type=pa.int8()),
            'timestamp': pa.array((x['timestamp'] for x in labels), type=pa.int64()),
            'segment': pa.array((x['segment'] for x in labels), type=pa.string()),
            'is_target': pa.array((x['is_target'] for x in labels), type=pa.bool_())}))
    mapping = {'schema_version': 1, 'padding_id': 0,
               'source_zip_sha256': digest(zip_path), 'train_parquet_sha256': file_info['train.parquet']['sha256'],
               'train_time_end_exclusive': valid_start,
               'user_to_model': {str(k): v for k, v in contract['users'].items()},
               'model_to_user': {str(v): k for k, v in contract['users'].items()},
               'item_to_model': {str(k): v for k, v in contract['items'].items()},
               'model_to_item': {str(v): k for k, v in contract['items'].items()}}
    id_path = out / 'id_map.json'
    save_json(id_path, mapping)
    file_info['id_map.json'] = {'sha256': digest(id_path), 'bytes': id_path.stat().st_size}

    def label_counts(name):
        labels = contract[name + '_labels']
        return {'all_observed': len(labels), 'positive': sum(x['is_target'] for x in labels),
                'positive_segments': dict(sorted(Counter(x['segment'] for x in labels
                    if x['is_target']).items())),
                'main_warm_positive': sum(x['segment'] == 'WARM' and x['is_target'] for x in labels),
                'excluded_nonpositive': sum(not x['is_target'] for x in labels)}

    split = {'schema_version': 1, 'source_zip_sha256': digest(zip_path),
             'source_readme_sha256': digest(terms), 'official_md5': match[1],
             'config_sha256': digest(config_path), 'seed': config['seed'],
             'rating_positive_at_least': config['rating_positive_at_least'],
             'min_train_positives_for_warm_user': config['min_train_positives'],
             'candidate_rule': 'at least one positive in train only',
             'training_graph_rule': 'warm user positive rows in train only',
             'train_history_rule': 'all train rated items, including ratings below positive threshold',
             'retrain_with_valid': False,
             'train_fraction_requested': 0.8, 'valid_fraction_requested': 0.1,
             'test_fraction_requested': 0.1, 'valid_start_timestamp': valid_start,
             'test_start_timestamp': test_start,
             'train_max_timestamp': max(r[3] for r in train),
             'valid_max_timestamp': max(r[3] for r in valid),
             'test_max_timestamp': max(r[3] for r in test),
             'actual_counts': {'train': len(train), 'valid': len(valid), 'test': len(test)},
             'actual_fractions': {'train': len(train) / len(rows), 'valid': len(valid) / len(rows),
                                  'test': len(test) / len(rows)},
             'raw_counts': {'ratings': len(rows), 'movies': len(movies),
                            'users': member_info['users']},
             'train_counts': {'warm_users': len(contract['users']),
                              'cold_users_with_train_positive': len({r[0] for r in train if r[2] >= 4}) - len(contract['users']),
                              'candidate_items': len(contract['items']),
                              'graph_edges': len(contract['train_edges']),
                              'nonpositive_observed': sum(r[2] < 4 for r in train)},
             'holdout_coverage': {'valid': label_counts('valid'), 'test': label_counts('test')},
             'train_parquet_sha256': file_info['train.parquet']['sha256'],
             'id_map_sha256': file_info['id_map.json']['sha256'],
             'tie_policy': 'timestamp equal to cutoff enters later window'}
    save_json(out / 'split_manifest.json', split)
    file_info['split_manifest.json'] = {'sha256': digest(out / 'split_manifest.json'),
                                        'bytes': (out / 'split_manifest.json').stat().st_size}
    manifest = {'schema_version': 1, 'status': 'PASS', 'protocol': 'T06 independent MovieLens1M global-time',
                'source_url': config['source_url'], 'source_sha256': digest(zip_path),
                'source_official_md5': match[1], 'terms_sha256': digest(terms),
                'config_sha256': digest(config_path), 'seed': config['seed'],
                'encoding': {'movies.dat': config['movie_title_encoding'],
                             'ratings.dat': 'ascii', 'users.dat': 'ascii'},
                'raw_counts': {'ratings': len(rows), 'movies': len(movies),
                               'users': member_info['users'], 'archive_members': member_info},
                'split': split, 'id_map_sha256': file_info['id_map.json']['sha256'],
                'train_positive_threshold': config['rating_positive_at_least'],
                'min_warm_train_positives': config['min_train_positives'],
                'train_users_with_any_positive': len({r[0] for r in train if r[2] >= 4}),
                'warm_train_users': len(contract['users']), 'candidate_items': len(contract['items']),
                'training_graph_edges': len(contract['train_edges']),
                'training_observed_rows': len(train),
                'train_nonpositive_observed': sum(r[2] < 4 for r in train),
                'valid': label_counts('valid'), 'test': label_counts('test'),
                'all_train_only': ['model IDs', 'training graph', 'observed history', 'candidate items',
                                   'popularity counts', 'sampling eligibility'],
                'evaluation': 'NOT EVALUATED', 'files': file_info,
                'dataset_redistributed': False,
                'unknown_attributes': ['plot', 'duration', 'price', 'emotion']}
    save_json(manifest_path, manifest)
    print(json.dumps({'status': 'PASS', 'source_sha256': manifest['source_sha256'],
                      'ratings': len(rows), 'split': split['actual_counts'],
                      'warm_users': len(contract['users']), 'candidate_items': len(contract['items']),
                      'graph_edges': len(contract['train_edges']),
                      'valid': manifest['valid'], 'test': manifest['test'],
                      'evaluation': 'NOT EVALUATED'}, sort_keys=True))


def main(argv=None):
    parser = argparse.ArgumentParser(description='Freeze the official MovieLens1M T06 contract')
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args(argv)
    prepare(args.config)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
