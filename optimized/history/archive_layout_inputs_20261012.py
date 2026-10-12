"""Preserve replaced navigation documents, and restore canonical file visibility."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    path = ROOT / 'optimized/history/layout_20261012.json'
    inventory = json.loads(path.read_text(encoding='utf-8'))
    replaced = {'.gitignore', '.gitattributes', 'AgenticRec/README.md', 'AgenticRec/docs/REPRODUCE.md'}
    for record in inventory['tracked_files']:
        if record['old'] not in replaced:
            continue
        if record['old'].startswith('AgenticRec/'):
            before = (ROOT / record['new']).read_bytes()
        else:
            before = subprocess.check_output(['git', 'show', inventory['before_commit'] + ':' + record['old']], cwd=ROOT)
            if hashlib.sha256(before).hexdigest() != record['sha256']:
                before = before.replace(b'\n', b'\r\n')
        if hashlib.sha256(before).hexdigest() != record['sha256']:
            raise ValueError('Cannot preserve historical document bytes: ' + record['old'])
        target = ROOT / 'optimized/history/navigation_before_20261012' / record['old']
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(before)
        record['preserved_at'] = target.relative_to(ROOT).as_posix()
    path.write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    if os.name == 'nt':
        import ctypes
        # The first prepare used hiding on hardlinks; hardlink attributes are
        # shared with their source. Restore all canonical document visibility.
        for record in inventory['tracked_files']:
            target = ROOT / record.get('preserved_at', record['new'])
            if target.exists():
                flags = ctypes.windll.kernel32.GetFileAttributesW(str(target))
                ctypes.windll.kernel32.SetFileAttributesW(str(target), flags & ~2)
        for target in (ROOT/'LICENSE.txt', ROOT/'.env'):
            if target.exists():
                flags = ctypes.windll.kernel32.GetFileAttributesW(str(target))
                ctypes.windll.kernel32.SetFileAttributesW(str(target), flags & ~2)
    print(json.dumps({'status': 'ARCHIVED', 'navigation_inputs': len(replaced)}))


if __name__ == '__main__':
    main()
