"""Archive two credential locators before changing their checkout-root lookup."""
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def main():
    manifest=ROOT/'optimized/history/layout_20261012.json'
    inventory=json.loads(manifest.read_text(encoding='utf-8'))
    selected={'reproduction/scripts/local_api_key.py','AgenticRec/src/agenticrec/runtime/secrets.py'}
    for record in inventory['tracked_files']:
        if record['old'] not in selected:
            continue
        content=(ROOT/record['new']).read_bytes()
        if hashlib.sha256(content).hexdigest()!=record['sha256']:
            raise ValueError('Credential locator was already changed')
        archived=ROOT/'optimized/history/navigation_before_20261012'/record['old']
        archived.parent.mkdir(parents=True,exist_ok=True)
        with archived.open('xb') as stream:
            stream.write(content)
        record['preserved_at']=archived.relative_to(ROOT).as_posix()
    inventory['runtime_changes']=['credential files locate the shared repository-root .env directly']
    manifest.write_text(json.dumps(inventory,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    canonical=ROOT/'.env'
    before=canonical.stat()
    for part in ('original','optimized'):
        alias=ROOT/part/'.env'
        if alias.exists():
            if not os.path.samefile(alias,canonical):
                raise ValueError('Refusing to remove a non-alias credential file')
            alias.unlink()  # Only the extra hardlink, never the root .env.
    after=canonical.stat()
    assert before.st_ino==after.st_ino and before.st_size==after.st_size
    print(json.dumps({'archived_locators':2,'canonical_env_unchanged':True,'secret_contents_read':False}))


if __name__=='__main__':
    main()
