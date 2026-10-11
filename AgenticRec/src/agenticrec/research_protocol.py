"""Append-only data freezing for the 2026-10-11 fusion research.

External labels are prepared once by a data-only process, never by serving.
Existing MovieLens1M train/valid/test and interactive-v1 remain unchanged.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import zipfile

from .data import digest
from .training import ROOT


def split_ratings(rows, train_fraction=.8, valid_fraction=.1):
    rows = sorted(rows, key=lambda r: (r[3], r[0], r[1]))
    if len({(r[0], r[1]) for r in rows}) != len(rows):
        raise ValueError('duplicate rating key')
    if len(rows) < 10 or not 0 < train_fraction < train_fraction + valid_fraction < 1:
        raise ValueError('Invalid time split')
    a = rows[int(len(rows) * train_fraction)][3]
    b = rows[int(len(rows) * (train_fraction + valid_fraction))][3]
    if a >= b:
        raise ValueError('Ties collapse validation window')
    return ([r for r in rows if r[3] < a], [r for r in rows if a <= r[3] < b],
            [r for r in rows if r[3] >= b])


def train_space(train, min_positives=5):
    counts = Counter(r[0] for r in train if r[2] >= 4)
    users = {u: i for i, u in enumerate(sorted(u for u in counts if counts[u] >= min_positives), 1)}
    items = {v: i for i, v in enumerate(sorted({r[1] for r in train if r[2] >= 4}), 1)}
    return users, items


def user_partition(users):
    users = list(users)
    if len(users) != len(set(users)):
        raise ValueError('Duplicate users')
    ordered = sorted(users, key=lambda u: (hashlib.sha256(('fusion-20261011:'+str(u)).encode()).hexdigest(), u))
    a, b = int(.6 * len(ordered)), int(.8 * len(ordered))
    return {'fit': sorted(ordered[:a]), 'select': sorted(ordered[a:b]), 'audit': sorted(ordered[b:])}


def _write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True)+'\n')


def prepare_external(raw_dir, output):
    """Only data preparation reads raw full ratings; no method or evaluation."""
    raw_dir, output = Path(raw_dir), Path(output)
    archive, terms = raw_dir/'ml-100k.zip', raw_dir/'ml-100k-README.txt'
    if not archive.is_file() or not terms.is_file():
        raise ValueError('Official archive and terms required')
    output.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        bad = z.testzip()
        if bad is not None:
            raise ValueError('Archive CRC failure: '+bad)
        rows = [tuple(map(int, line.split())) for line in z.read('ml-100k/u.data').decode('ascii').splitlines()]
        if len(rows) != 100000 or any(len(r) != 4 or not 1 <= r[2] <= 5 for r in rows):
            raise ValueError('Official 100K rating schema/count differs')
        genre_names = [line.split('|')[0] for line in z.read('ml-100k/u.genre').decode('ascii').splitlines() if line]
        raw_items = {}
        for line in z.read('ml-100k/u.item').decode('latin-1').splitlines():
            parts = line.split('|')
            if len(parts) != 5 + len(genre_names):
                raise ValueError('Official item schema differs')
            match = re.search(r'\((\d{4})\)\s*$', parts[1])
            raw_items[int(parts[0])] = {'title': parts[1], 'year': int(match[1]) if match else None,
                'genres': [g for g, value in zip(genre_names, parts[5:]) if value == '1' and g != 'unknown']}
    train, valid, test = split_ratings(rows)
    users, items = train_space(train)
    if not users or not items:
        raise ValueError('No training universe')
    mapped_train = [(users[r[0]], items[r[1]], r[2], r[3]) for r in train if r[0] in users and r[1] in items]
    edges = [(u,i) for u,i,r,_ in mapped_train if r >= 4]
    seen = defaultdict(set)
    for u,i,_r,_t in mapped_train: seen[u].add(i)
    catalog = {str(model): {'item_id':model, 'raw_item_id':raw, **raw_items[raw]}
               for raw,model in items.items()}
    _write(output/'catalog.json', catalog)
    _write(output/'train.json', mapped_train)
    _write(output/'train_positive.json', edges)
    _write(output/'id_map.json', {'users':users, 'items':items, 'padding':0})
    coverage = {}
    for name, subset in [('valid',valid), ('test',test)]:
        relevant = defaultdict(set)
        excluded = Counter()
        for u,i,r,_t in subset:
            if r < 4: excluded['nonpositive'] += 1; continue
            if u not in users: excluded['cold_user'] += 1; continue
            if i not in items: excluded['cold_item'] += 1; continue
            mu, mi = users[u], items[i]
            if mi in seen[mu]: excluded['train_seen'] += 1; continue
            relevant[mu].add(mi)
        cohort = {'schema_version':1,'split':name,'evaluator_only':True,
                  'candidate_ids':list(range(1,len(items)+1)), 'user_ids':list(range(1,len(users)+1)),
                  'seen_by_user':{str(u):sorted(seen[u]) for u in range(1,len(users)+1)},
                  'relevant_by_user':{str(u):sorted(relevant[u]) for u in range(1,len(users)+1)}}
        _write(output/(name+'.json'), cohort)
        nonempty = sum(bool(ids) for ids in relevant.values())
        coverage[name] = {'observed_rows':len(subset), 'users_with_relevance':nonempty,
                          'users_without_relevance':len(users)-nonempty, 'excluded':dict(excluded)}
    files = {p.name:digest(p) for p in output.iterdir() if p.is_file()}
    manifest = {'schema_version':1,'dataset':'MovieLens100K official release',
        'source_url':'https://files.grouplens.org/datasets/movielens/ml-100k.zip',
        'source_zip_sha256':digest(archive),'terms_sha256':digest(terms),
        'download_bytes':archive.stat().st_size, 'source_crc_verified':True,
        'dataset_redistributed':False, 'raw_rows':len(rows),
        'split_counts':{'train':len(train),'valid':len(valid),'test':len(test)},
        'time_bounds':{n:[min(r[3] for r in rs),max(r[3] for r in rs)] for n,rs in [('train',train),('valid',valid),('test',test)]},
        'warm_users':len(users),'candidate_items':len(items),'train_edges':len(edges),
        'coverage':coverage,'files':files,
        'method_development_test_access':False,
        'independence_limit':'New external release; no guarantee of distinct real persons across MovieLens releases'}
    _write(output/'manifest.json', manifest)
    return manifest


def freeze_development(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cohort_path = ROOT/'data/eval_private/ml-1m-v1/valid.json'
    cohort = json.loads(cohort_path.read_text(encoding='utf-8'))
    users = [u for u in cohort['user_ids'] if cohort['relevant_by_user'][str(u)]]
    groups = user_partition(users)
    _write(output/'user_groups.json', groups)
    manifest = {'schema_version':1, 'protocol':'fusion-20261011',
        'development':'MovieLens1M original valid only', 'historical_test':'Previously Exposed Test',
        'valid_sha256':digest(cohort_path),'data_manifest_sha256':digest(ROOT/'artifacts/data_manifest.json'),
        'user_groups_sha256':digest(output/'user_groups.json'),
        'group_counts':{k:len(v) for k,v in groups.items()},
        'query_states':['positive_seeds','local_negative','genre_constraint'],
        'positive_seeds':'last three train rating>=4 items, timestamp/ID stable',
        'local_negative':'last train rating<=2 item, if any, concrete item denial only',
        'constraints':'first genre of most recent positive seed; train/metadata only',
        'no_relevant_after_constraint':'count separately, never fabricate acceptance',
        'audit_limit':'Original model hyperparameters used all valid; audit is development audit, not blind model test',
        'seeds':[7,42,2026], 'remote_api_requests':0}
    _write(output/'manifest.json',manifest)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['external','development'])
    parser.add_argument('--raw-dir',type=Path,default=ROOT/'data/raw/ml-100k-upgrade')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    value = prepare_external(args.raw_dir,args.output) if args.action=='external' else freeze_development(args.output)
    print(json.dumps(value,indent=2,sort_keys=True))
