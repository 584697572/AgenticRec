"""Capture new migration verification outputs without overwriting old evidence."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'optimized/artifacts/verification/two_part_20261012'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--name',required=True)
    parser.add_argument('--cwd',default='.')
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    if not args.name.replace('_','').isalnum():
        parser.error('name must be alphanumeric with underscores')
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    cwd=(ROOT/args.cwd).resolve()
    if not cwd.is_relative_to(ROOT):
        parser.error('cwd must be inside checkout')
    executable=(cwd/command[0]).resolve()
    if executable.is_file():
        command[0]=str(executable)
    OUT.mkdir(parents=True,exist_ok=True)
    environment=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8',HF_HUB_OFFLINE='1',
                     TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',TOKENIZERS_PARALLELISM='false')
    start=time.perf_counter()
    with (OUT/(args.name+'.stdout.log')).open('xb') as stdout, (OUT/(args.name+'.stderr.log')).open('xb') as stderr:
        result=subprocess.run(command,cwd=cwd,env=environment,stdout=stdout,stderr=stderr,timeout=600)
    summary={'name':args.name,'command':command,'cwd':args.cwd,'exit_code':result.returncode,
             'elapsed_seconds':round(time.perf_counter()-start,3),'logs_are_new':True,
             'remote_api_requests':0,'new_training_runs':0}
    with (OUT/(args.name+'.json')).open('x',encoding='utf-8') as stream:
        json.dump(summary,stream,ensure_ascii=True,indent=2)
        stream.write('\n')
    print(json.dumps(summary,ensure_ascii=True))
    return result.returncode


if __name__=='__main__':
    raise SystemExit(main())
