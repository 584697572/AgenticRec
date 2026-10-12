"""One-time read-only inventory of the pre-migration checkout."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BASELINE_PREFIXES = ('a1_', 'catalog_identity_', 'deepseek_setup_', 'g0_', 'live_',
                     'local_key_', 'multiturn_', 'progress_', 'public_repository_',
                     'reproduction_', 't03_', 'upstream_evidence_', 'upstream_material_')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    files = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode('utf-8').split('\0')
    records = []
    baseline = []
    for name in filter(None, files):
        if name.startswith('AgenticRec/'):
            new = 'optimized/' + name
        elif name.startswith('reproduction/'):
            new = 'original/' + name
        elif name.startswith('reports/'):
            is_original = '/' not in name[len('reports/'):] and Path(name).name.startswith(BASELINE_PREFIXES)
            new = ('original/' if is_original else 'optimized/') + name
            if is_original:
                baseline.append(Path(name).name)
        elif name in ('README.md', 'STATUS.md', 'DECISIONS.md', 'TASKS.yaml') or name.startswith('AgenticRec_'):
            new = 'optimized/history/workspace/' + name
        else:
            new = name
        records.append({'old': name, 'new': new, 'sha256': sha(ROOT / name)})
    # Record immutable data and checkpoints, not reinstallable environments.
    private = []
    for directory in ('data', 'artifacts'):
        for path in sorted((ROOT / directory).rglob('*')):
            if not path.is_file() or path.suffix not in ('.pt', '.pth', '.npy', '.zip', '.ftr', '.parquet'):
                continue
            private.append({'old': path.relative_to(ROOT).as_posix(), 'sha256': sha(path),
                            'bytes': path.stat().st_size})
    value = {
        'date': '2026-10-12',
        'before_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'upstream_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT/'RecAI', text=True).strip(),
        'baseline_report_names': sorted(baseline),
        'canonical_report_paths': [r['new'] for r in records if r['old'].startswith('reports/')],
        'tracked_files': records,
        'private_binary_files': private,
        'private_contents_in_manifest': False,
        'original_rules_binding': False,
    }
    destination = ROOT / 'optimized/history/layout_20261012.json'
    with destination.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'tracked_files': len(records), 'private_binary_files': len(private),
                      'baseline_reports': len(baseline), 'status': 'CAPTURED'}))


if __name__ == '__main__':
    main()
