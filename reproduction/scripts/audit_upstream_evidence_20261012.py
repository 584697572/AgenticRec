"""Offline, bounded evidence audit. Never load weights, execute notebooks or read keys."""
import argparse
import hashlib
import json
from pathlib import Path
import pickletools
import re
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'artifacts/runs/upstream_evidence_20261012'
UPSTREAM = ROOT / 'RecAI'
PIN = '0959ecb05b0794748426e73e6efc1b6b35ec433d'
MUTABLE = {'STATUS.md', 'AgenticRec/docs/STATUS.md', 'TASKS.yaml', 'DECISIONS.md', 'reports/spec_issues.md'}

def sha(data):
    return hashlib.sha256(data).hexdigest()

def git(*args):
    return subprocess.check_output(['git', '-C', str(UPSTREAM), *args])

def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def write_new(path, obj):
    data = (json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode('utf-8')
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('Refuse to overwrite evidence: ' + path.name)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

def freeze():
    paths = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    protected = {p: sha((ROOT / p).read_bytes()) for p in paths if p and p not in MUTABLE}
    # Include all five original resources, and all twelve new formal checkpoints.
    resources = load(ROOT / 'reproduction/resource_manifest.json')['resources']
    for resource in resources:
        p = resource.get('local_path') or resource.get('path')
        if p and (ROOT / p).is_file():
            protected[p] = sha((ROOT / p).read_bytes())
    for path in (ROOT / 'artifacts/runs/lightgcn_hard_negatives_20261011').glob('seed_*/*/best.pt'):
        protected[path.relative_to(ROOT).as_posix()] = sha(path.read_bytes())
    write_new(BASE / 'preservation_before.json', protected)
    print(json.dumps({'frozen_files': len(protected)}))

def audit():
    assert git('rev-parse', 'HEAD').decode().strip() == PIN
    assert not git('status', '--porcelain')
    protected = load(BASE / 'preservation_before.json')
    assert all(sha((ROOT / p).read_bytes()) == h for p, h in protected.items())
    ledger = load(BASE / 'public_discussions/ledger.json')
    assert len(ledger['requests']) <= 18 and ledger['body_bytes'] <= 4194304
    assert ledger['body_bytes'] == sum(x['body_bytes'] for x in ledger['requests'])
    sources = []
    for record in ledger['requests']:
        matches = list(BASE.glob('public*/' + record['file'])) if record['file'] else []
        assert len(matches) == 1
        assert matches[0].stat().st_size == record['body_bytes']
        assert sha(matches[0].read_bytes()) == record['sha256']
        sources.append({k: record[k] for k in ('name', 'url', 'status', 'complete', 'body_bytes', 'sha256', 'error')})
    brief = load(BASE / 'public_supplement/latest_commit_brief.body')[0]
    tree = load(BASE / 'public_supplement/current_tree.body')
    assert brief['sha'] == tree['sha'] == PIN and not tree['truncated']
    current_paths = [r['path'] for r in tree['tree'] if r['type'] == 'blob' and r['path'].startswith('InteRecAgent/')]
    assert sorted(current_paths) == sorted(git('ls-tree', '-r', '--name-only', PIN, 'InteRecAgent').decode().splitlines())
    for name, path in [('current_readme', 'InteRecAgent/README.md'), ('current_license', 'LICENSE.txt')]:
        assert (BASE / f'public_supplement/{name}.body').read_bytes() == git('show', PIN + ':' + path)
    issue = load(BASE / 'public/issue_110.body')
    comments = load(BASE / 'public/issue_110_comments.body')
    assert issue['comments'] == len(comments) == 0
    searches = {}
    for name in ('interecagent_issues', 'mapping_issues', 'license_issues'):
        data = load(BASE / f'public_focused/{name}.body')
        assert not data['incomplete_results'] and data['total_count'] == len(data['items'])
        searches[name] = {'total_count': data['total_count'], 'items': [{'number': x['number'], 'title': x['title'], 'url': x['html_url']} for x in data['items']]}
    discussions = {}
    for number in (75, 107, 106):
        values = load(BASE / f'public_discussions/issue_{number}_comments.body')
        assert len(values) < 100
        discussions[str(number)] = [{'url': x['html_url'], 'login': x['user']['login'], 'association': x['author_association'], 'created_at': x['created_at'], 'body_sha256': sha(x['body'].encode()), 'urls': re.findall(r'https?://[^\s)>]+', x['body'])} for x in values]
    # Historical file inventory and unique text blobs; inspect only this subproject.
    lines = git('rev-list', '--objects', '--all', '--', 'InteRecAgent').decode().splitlines()
    objects = {x.split(' ', 1)[0]: x.split(' ', 1)[1] for x in lines if ' ' in x and x.split(' ', 1)[1].startswith('InteRecAgent/')}
    batch = subprocess.run(['git', '-C', str(UPSTREAM), 'cat-file', '--batch-check=%(objectname) %(objecttype) %(objectsize)'], input=('\n'.join(objects)+'\n').encode(), capture_output=True, check=True)
    blobs = []
    text_blobs = []
    notebooks = []
    for line in batch.stdout.decode().splitlines():
        oid, kind, size = line.split()
        if kind != 'blob':
            continue
        path = objects[oid]
        blobs.append({'git_blob': oid, 'path': path, 'bytes': int(size)})
        if Path(path).suffix.lower() not in ('.py', '.md', '.sh', '.json', '.jsonl', '.yaml', '.yml', '.txt', '.ipynb'):
            continue
        if int(size) > 3145728:
            raise ValueError('Historical text blob exceeds audit bound')
        body = git('cat-file', 'blob', oid)
        text = body.decode('utf-8')
        text_blobs.append({'git_blob': oid, 'path': path, 'sha256': sha(body), 'terms_present': sorted(set(re.findall(r'item_id_map|user_id_map|ml-latest|ml-10m|test_data_50|license|licence', text, flags=re.I)))})
        if path.endswith('/movies.ipynb') or path.endswith('/ItemCF.ipynb') or path.endswith('/preprocess_redial.ipynb'):
            nb = json.loads(text)
            cells = [(i, ''.join(c.get('source', []))) for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code']
            relevant = [{'cell': i, 'lines': [s.strip() for s in code.splitlines() if any(term in s for term in ('item_id_map', 'user_id_map', 'unique()', 'to_feather', 'to_csv', 'ratings.csv', 'movies.csv', 'ml-latest', 'ml-10m', '[:50]', '[:500]', 'test_data_50', 'data_path', 'read_csv', 'k_core_filter'))]} for i, code in cells]
            notebooks.append({'git_blob': oid, 'path': path, 'sha256': sha(body), 'code_cells': len(cells), 'relevant_cells': [r for r in relevant if r['lines']]})
    write_new(BASE / 'history_inventory.json', {'blobs': blobs, 'text_blobs': text_blobs, 'notebooks': notebooks})
    checkpoint = ROOT / 'data/raw/upstream_movie/SASRec-SASRec.pth'
    with zipfile.ZipFile(checkpoint) as z:
        payload = z.read(next(p for p in z.namelist() if p.endswith('/data.pkl')))
        strings = [arg for op, arg, _ in pickletools.genops(payload) if op.name in ('BINUNICODE', 'SHORT_BINUNICODE', 'UNICODE')]
    # Static key presence cannot certify provenance; no pickle execution.
    mapping_words = [s for s in strings if re.search(r'item.*(map|title)|movie.*(map|title)|id2|2id|position|use_pos|has_pos', s, flags=re.I)]
    archive = load(ROOT / 'reproduction/archive_inventory.json')
    version_text = (BASE / 'public/paper_abstract.body').read_text(encoding='utf-8')
    versions = sorted(set(re.findall(r'2308\.16505v\d+', version_text)))
    assert versions == ['2308.16505v1', '2308.16505v2', '2308.16505v3']
    paper_text = (BASE / 'paper_v3_text.txt').read_text(encoding='utf-8')
    for fact in ('without position embedding', 'gpt-4-0613', '27,042,493', '496', '19 negative items', 'training and validation sets'):
        assert fact in paper_text, fact
    summary = {
        'audit_date': '2026-10-12', 'audit_status': 'VERIFIED', 'scope': 'bounded evidence search; no reproduction experiment',
        'upstream_pin': PIN, 'current_public_head': brief['sha'], 'current_tree_paths': len(tree['tree']), 'current_interecagent_files': len(current_paths),
        'local_commits_available': int(git('rev-list', '--all', '--count')), 'historical_distinct_paths': len({x['path'] for x in blobs}),
        'historical_blobs': len(blobs), 'historical_text_blobs_inspected': len(text_blobs), 'notebook_versions_inspected': len(notebooks),
        'sources': sources, 'http_requests': len(ledger['requests']), 'archived_http_body_bytes': ledger['body_bytes'],
        'network_scope': 'collector only; web rendering/source verification is additional and not measured as HTTP bytes',
        'partial_responses_excluded': [x['name'] for x in sources if not x['complete']], 'http_errors': [x['name'] for x in sources if isinstance(x['status'], int) and x['status'] >= 400],
        'issue110': {'state': issue['state'], 'comments': len(comments), 'updated_at': issue['updated_at']},
        'issue_searches': searches, 'discussions': discussions,
        'checkpoint_static': {'sha256': sha(checkpoint.read_bytes()), 'mapping_or_position_string_keys': sorted(set(mapping_words)), 'unpickled': False, 'executed': False},
        'archive_member_count': len(archive), 'archive_names': [x['name'] for x in archive],
        'paper_version': '2308.16505v3', 'paper_protocol': {'llm': 'gpt-4-0613', 'ranker': 'SASRec without positional embedding', 'soft_retrieval': 'ItemCF', 'split': 'leave-one-out; tools trained on train+valid', 'k_core': 5, 'dynamic_demos': 3, 'hard_candidates': 1000, 'soft_top_fraction': 0.05, 'initial_one_turn_samples': 500, 'movielens_one_turn_rows_table_A1': 496, 'simulator_samples': 1000, 'lifelong_samples': 100, 'retrieval_metric': 'Recall@5', 'ranking_metric': 'NDCG@20; 19 sampled negatives + 1 positive', 'movielens_table_users': 298074, 'movielens_table_items': 36255, 'movielens_table_interactions': 27042493},
        'paper_data_release': 'NOT VERIFIED; 10M link/label conflicts with 27,042,493 interactions and CSV/latest preprocessing',
        'blockers': {'checkpoint_title_mapping': 'NOT VERIFIED', 'matrix_row_mapping': 'NOT VERIFIED', 'original_resource_independent_license': 'NOT VERIFIED', 'canonical_movie_eval_files': 'NOT VERIFIED', 'A2_acceptance': 'BLOCKED'},
        'preserved_files': len(protected), 'models_trained': 0, 'paid_llm_requests': 0, 'messages_sent': 0,
        'report_generation_script_sha256': sha(Path(__file__).read_bytes()), 'collector_sha256': sha((ROOT / 'reproduction/scripts/evidence_public_20261012.ps1').read_bytes()),
    }
    write_new(BASE / 'audit_summary.json', summary)
    write_new(ROOT / 'reports/upstream_evidence_20261012.json', summary)
    print(json.dumps({k: summary[k] for k in ('audit_status', 'historical_distinct_paths', 'historical_blobs', 'notebook_versions_inspected', 'http_requests', 'archived_http_body_bytes', 'preserved_files', 'blockers')}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    freeze() if args.freeze else audit()
