"""Two CPU workers, preregistered phases, append-only logs, no paid calls."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys

from agenticrec.data import digest
from agenticrec.explicit_low_training import BASE, ROOT, code_identity, plan_and_groups


PLAN=ROOT/'AgenticRec/configs/lightgcn_explicit_low_20261012.json'


def initialize():
    """Create an append-only run root and protection ledger without manual setup."""
    BASE.mkdir(parents=True,exist_ok=True)
    ledger=BASE/'preservation_before.json'
    if ledger.exists():return
    paths=set(subprocess.check_output(['git','-c','core.quotepath=false','ls-files'],
                                     cwd=ROOT,text=True,encoding='utf-8').splitlines())
    previous=ROOT/'artifacts/runs/upstream_evidence_20261012/preservation_before.json'
    if previous.exists():paths.update(json.loads(previous.read_text(encoding='utf-8')))
    paths.update(p.relative_to(ROOT).as_posix() for p in (ROOT/'artifacts').rglob('best.pt'))
    paths-={'STATUS.md','AgenticRec/docs/STATUS.md','TASKS.yaml','DECISIONS.md','reports/spec_issues.md'}
    protected={name:digest(ROOT/name) for name in sorted(paths)
               if (ROOT/name).is_file() and not Path(name).name.startswith('.env')}
    with ledger.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(protected,f,indent=2,sort_keys=True);f.write('\n')


def execute(trial,seed):
    target=BASE/f'seed_{seed}'/trial
    if (target/'result.json').exists():
        value=json.loads((target/'result.json').read_text(encoding='utf-8'))
        if (value['status']!='VERIFIED' or value['identity']['code_sha256']!=code_identity()
                or value['identity']['plan_sha256']!=digest(PLAN)
                or digest(target/'best.pt')!=value['checkpoint_sha256']):
            raise ValueError('Cannot resume changed result')
        return value
    command=[sys.executable,'-u','-m','agenticrec.explicit_low_training','train',
             '--plan',str(PLAN),'--trial',trial,'--seed',str(seed)]
    prefix=BASE/f'{trial}_seed_{seed}'
    with Path(str(prefix)+'_stdout.txt').open('x',encoding='utf-8') as stdout, Path(str(prefix)+'_stderr.txt').open('x',encoding='utf-8') as stderr:
        result=subprocess.run(command,cwd=ROOT,stdout=stdout,stderr=stderr)
    Path(str(prefix)+'_exit.json').write_text(json.dumps({'exit_code':result.returncode})+'\n',encoding='utf-8')
    if result.returncode:raise RuntimeError(f'Training failed: {trial} seed {seed}; original logs retained')
    return json.loads((target/'result.json').read_text(encoding='utf-8'))


def phase(jobs):
    with ThreadPoolExecutor(max_workers=2) as executor:
        pending={executor.submit(execute,name,seed):(name,seed) for name,seed in jobs}
        for future in as_completed(pending):
            name,seed=pending[future]; value=future.result()
            print(json.dumps({'completed':name,'seed':seed,'epochs':value['actual_epochs'],
                              'best_epoch':value['best_epoch'],'selection':value['selection_metrics']}),flush=True)


def action(name):
    prefix=BASE/name
    with Path(str(prefix)+'_stdout.txt').open('x',encoding='utf-8') as stdout, Path(str(prefix)+'_stderr.txt').open('x',encoding='utf-8') as stderr:
        result=subprocess.run([sys.executable,'-u','-m','agenticrec.explicit_low_training',name,'--plan',str(PLAN)],
                              cwd=ROOT,stdout=stdout,stderr=stderr)
    Path(str(prefix)+'_exit.json').write_text(json.dumps({'exit_code':result.returncode})+'\n',encoding='utf-8')
    if result.returncode:raise RuntimeError('Study phase failed: '+name)
    print(json.dumps({'phase':name,'status':'VERIFIED'}),flush=True)


def main():
    plan,_=plan_and_groups(PLAN)
    initialize()
    phase([(t['name'],42) for t in plan['trials']])
    if not (BASE/'selection_frozen.json').exists():action('freeze')
    frozen=json.loads((BASE/'selection_frozen.json').read_text(encoding='utf-8'))
    if frozen['code_sha256']!=code_identity() or frozen['plan_sha256']!=digest(PLAN):raise ValueError('Frozen selection changed')
    phase([(t['name'],seed) for seed in (7,2026) for t in plan['trials']])
    if not (BASE/'audit/summary.json').exists():action('audit')
    print(json.dumps({'status':'VERIFIED','training_runs':12,'scope':'bounded development study'}),flush=True)


if __name__=='__main__':main()
