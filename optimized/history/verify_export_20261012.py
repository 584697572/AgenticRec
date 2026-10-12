"""Exercise only the public staged source, without copying private data or envs."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'optimized/artifacts/verification/two_part_20261012'
EXPORT=OUT/'source_export'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory',choices=('source_export','source_export_final'),default='source_export')
    args=parser.parse_args()
    export=OUT/args.directory
    export.mkdir()
    rows=subprocess.check_output(['git','ls-files','--stage','-z'],cwd=ROOT).decode('utf-8').split('\0')
    count=0
    for row in filter(None,rows):
        metadata,name=row.split('\t',1)
        if metadata.startswith('160000 '):
            continue  # Upstream is fetched explicitly by git submodule update.
        target=export/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(subprocess.check_output(['git','show',':'+name],cwd=ROOT))
        count+=1
    spec=importlib.util.spec_from_file_location('exported_workspace',export/'workspace.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prepared=module.prepare(export)
    broken=[]
    documents=['README.md','original/README.md','optimized/README.md','optimized/AgenticRec/README.md',
               'optimized/AgenticRec/docs/REPRODUCE.md']
    for name in documents:
        path=export/name
        for destination in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if destination.startswith(('https://','http://','#','app://')):
                continue
            target=(path.parent/destination.split('#')[0]).resolve()
            if not target.is_relative_to(export) or not target.exists():
                broken.append({'document':name,'link':destination})
    python=ROOT/'optimized/AgenticRec/.venv-t20/Scripts/python.exe'
    source=export/'optimized/AgenticRec/src'
    code=('import sys;sys.path.insert(0,'+repr(str(source))+');import agenticrec;'
          'assert agenticrec.__file__.startswith('+repr(str(source))+');'
          'from agenticrec.cli import main;sys.exit(main(["fixture"]))')
    result=subprocess.run([str(python),'-c',code],cwd=export,capture_output=True,
                          env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8'),timeout=60)
    (OUT/'source_export_fixture.stdout.log').write_bytes(result.stdout)
    (OUT/'source_export_fixture.stderr.log').write_bytes(result.stderr)
    value={'status':'PASS' if result.returncode==0 and not broken else 'FAIL',
           'exported_public_files':count,'broken_current_navigation_links':broken,
           'fixture_exit_code':result.returncode,'fresh_source_imported':result.returncode==0,
           'prepare_status':prepared['status'],'new_dependency_install':False,
           'upstream_checkout_in_export':False,'remote_api_requests':0,'downloads':0,
           'scope':'fresh public source with existing runtime; not a full new-machine rebuild'}
    (OUT/'source_export_check.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(value,ensure_ascii=True,indent=2))
    return 0 if value['status']=='PASS' else 1


if __name__=='__main__':
    raise SystemExit(main())
