"""Read-only comparison readiness audit; no inference, training or API calls.

The crosswalk labels catalog metadata only, never pretrained matrix/checkpoint
rows. Shared IDs are checked against titles and years across both metadata sets.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path

import pyarrow.feather as feather
import pyarrow.parquet as parquet


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root):
    root = root.resolve()
    identity_path = root / 'original/reproduction/scripts/catalog_identity.py'
    spec = importlib.util.spec_from_file_location('catalog_identity', identity_path)
    identity = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(identity)
    paths = {
        'original_catalog': root / 'original/data/raw/upstream_movie/movies.ftr',
        'crosswalk': root / 'original/data/metadata_private/catalog_identity_20261005/catalog_to_ml10m.jsonl',
        'crosswalk_manifest': root / 'original/data/metadata_private/catalog_identity_20261005/manifest.json',
        'optimized_catalog': root / 'optimized/data/processed/ml-1m-v1/items.parquet',
        'train_positive': root / 'optimized/data/processed/ml-1m-v1/train_positive.parquet',
        'split_manifest': root / 'optimized/data/processed/ml-1m-v1/split_manifest.json',
        'identity_source': identity_path,
    }
    hashes = {name: digest(path) for name, path in paths.items()}
    manifest = json.loads(paths['crosswalk_manifest'].read_text(encoding='utf-8'))
    assert hashes['original_catalog'] == manifest['catalog_sha256']
    expected = next(a['sha256'] for a in manifest['artifacts']
                    if a['path'].endswith('/catalog_to_ml10m.jsonl'))
    assert hashes['crosswalk'] == expected
    original = feather.read_table(paths['original_catalog']).to_pylist()
    optimized = parquet.read_table(paths['optimized_catalog']).to_pylist()
    candidates = set(parquet.read_table(paths['train_positive'], columns=['item_id'])
                     .column('item_id').to_pylist())
    split = json.loads(paths['split_manifest'].read_text(encoding='utf-8'))
    assert len(candidates) == split['train_counts']['candidate_items']
    crosswalk = [json.loads(line) for line in paths['crosswalk'].read_text(encoding='utf-8').splitlines()]
    assert len(original) == len(crosswalk)
    by_catalog = {r['catalog_id']: r for r in crosswalk}
    assert len(by_catalog) == len(crosswalk)
    by_raw = {int(r['raw_item_id']): r for r in optimized}
    assert len(by_raw) == len(optimized)

    def genres(values):
        return {"children" if g.casefold() == "children's" else g.casefold() for g in values}

    statuses, shared = Counter(), []
    for row in original:
        link = by_catalog[int(row['id'])]
        if link['status'] != 'PASS':
            statuses['original_crosswalk_not_pass'] += 1
            continue
        item = by_raw.get(int(link['raw_movie_id']))
        if item is None:
            statuses['not_in_optimized_catalog'] += 1
            continue
        title, year = identity.split_title_year(item['title'])
        if year != int(row['release_date']) or not (
                identity.source_title_forms(title) & identity.source_title_forms(row['title'])):
            statuses['same_raw_id_but_identity_not_confirmed'] += 1
            continue
        equal_genres = genres(row['tags']) == genres(item['genres'])
        item_id = int(item['item_id']) if item['item_id'] is not None else None
        shared.append({'original_catalog_id': int(row['id']),
                       'raw_movie_id': int(link['raw_movie_id']),
                       'optimized_item_id': item_id,
                       'in_optimized_training_candidates': item_id in candidates,
                       'genre_sets_equal': equal_genres})
        statuses['shared_identity_confirmed'] += 1
    assert sum(statuses.values()) == len(original)
    assert len({r['raw_movie_id'] for r in shared}) == len(shared)
    return {
        'scope': 'catalog_metadata_readiness_only',
        'status': 'VERIFIED',
        'original_catalog_items': len(original),
        'optimized_catalog_items': len(optimized),
        'optimized_training_candidate_items': len(candidates),
        'identity_counts': dict(sorted(statuses.items())),
        'shared_training_candidates': sum(r['in_optimized_training_candidates'] for r in shared),
        'shared_items_equal_genre_sets': sum(r['genre_sets_equal'] for r in shared),
        'shared_items_different_genre_sets': sum(not r['genre_sets_equal'] for r in shared),
        'source_sha256': hashes,
        'catalog_crosswalk_is_pretrained_row_map': False,
        'recommendation_quality': 'NOT EVALUATED',
        'exact_original_vs_optimized_comparison': 'NOT EVALUATED',
        'remote_api_requests': 0,
        'new_training_runs': 0,
        'shared_metadata_rows': shared,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    output = args.output.resolve()
    if not output.is_relative_to((root / 'optimized/artifacts').resolve()):
        raise ValueError('Private metadata crosswalk must stay in ignored artifacts')
    if output.exists():
        raise FileExistsError('Refusing to overwrite earlier evidence')
    result = audit(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=True, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ('shared_metadata_rows', 'source_sha256')}, indent=2))


if __name__ == '__main__':
    main()
