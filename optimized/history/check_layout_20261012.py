"""Read-only relocation and staged-byte verification; no model/API execution."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify(staged=False):
    inventory = json.loads((ROOT/'optimized/history/layout_20261012.json').read_text(encoding='utf-8'))
    failures = []
    for entry in inventory['tracked_files']:
        path = ROOT/entry.get('preserved_at',entry['new'])
        if not path.is_file() or sha(path) != entry['sha256']:
            failures.append(entry['old'])
    for entry in inventory['private_binary_files']:
        # Old paths resolve to the same canonical resources through local links.
        path = ROOT/entry['old']
        if not path.is_file() or path.stat().st_size != entry['bytes'] or sha(path) != entry['sha256']:
            failures.append(entry['old'])
    upstream = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT/'original/RecAI',text=True).strip()
    dirty = subprocess.check_output(['git','status','--porcelain'],cwd=ROOT/'original/RecAI')
    if upstream != inventory['upstream_commit'] or dirty:
        failures.append('original/RecAI')
    result = {'status':'PASS' if not failures else 'FAIL','failures':failures,
              'preserved_tracked_inputs':len(inventory['tracked_files']),
              'preserved_private_binaries':len(inventory['private_binary_files']),
              'upstream_commit':upstream,'upstream_clean':not bool(dirty),
              'new_training_runs':0,'remote_api_requests':0,'downloads':0}
    if staged:
        rows = subprocess.check_output(['git','ls-files','--stage','-z'],cwd=ROOT).decode('utf-8').split('\0')
        parts, checked = set(), 0
        for row in filter(None,rows):
            metadata,name = row.split('\t',1)
            mode,oid,_stage = metadata.split()
            if '/' in name:
                parts.add(name.split('/')[0])
            if mode == '160000':
                if name != 'original/RecAI' or oid != upstream:
                    failures.append('incorrect_upstream_gitlink')
                continue
            content = (ROOT/name).read_bytes()
            computed = hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
            if computed != oid:
                failures.append('staged_bytes:'+name)
            if any(p in ('.env','data','artifacts') or p.startswith('.venv') for p in Path(name).parts):
                failures.append('private_path_staged:'+name)
            checked += 1
        if parts != {'original','optimized'}:
            failures.append('extra_public_parts:'+repr(sorted(parts)))
        result.update(public_parts=sorted(parts),staged_files_checked=checked)
    result['failures']=failures
    result['status']='PASS' if not failures else 'FAIL'
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--staged',action='store_true')
    args=parser.parse_args()
    result=verify(args.staged)
    print(json.dumps(result,ensure_ascii=True,indent=2))
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':
    raise SystemExit(main())
