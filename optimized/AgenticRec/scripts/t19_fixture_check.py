"""Exercise real frozen tools with mechanical public-only replies; no metrics."""
import contextlib
import hashlib
import io
import json
from pathlib import Path

from agenticrec.evaluation.live_benchmark import run_batch


def main():
    batch = Path('../artifacts/runs/t19/fixture_audit_20261010_v3')
    rows = []
    for condition in ('F', 'A', 'O', 'no_user_model', 'no_content', 'no_collaborative',
                      'no_explicit_preference_state', 'no_replanning'):
        # Traces remain in local artifacts; stdout carries only aggregate checks.
        with contextlib.redirect_stdout(io.StringIO()):
            run_batch('configs/benchmark.yaml', live=False, batch_dir=batch,
                      condition=condition, seed=42)
        directory = batch / condition / 'seed_42'
        journal = directory / 'predictions.jsonl'
        events = [json.loads(line) for line in journal.read_text(encoding='utf-8').splitlines()]
        attempts = [event['attempt'] for event in events if event['event'] == 'COMPLETED']
        digest = hashlib.sha256(journal.read_bytes()).hexdigest()
        with contextlib.redirect_stdout(io.StringIO()):
            run_batch('configs/benchmark.yaml', live=False, batch_dir=batch,
                      condition=condition, seed=42)
        if hashlib.sha256(journal.read_bytes()).hexdigest() != digest:
            raise RuntimeError('completed resume changed raw attempts')
        ledger = json.loads((directory / 'ledger.json').read_text(encoding='utf-8'))
        if len(attempts) != 150 or sum(a['request_count'] for a in attempts) != 0:
            raise RuntimeError('fixture denominator or no-network contract failed')
        row = {'condition': condition, 'episodes': len(attempts),
               'remote_requests': 0, 'fixture_planner_calls': ledger['attempts'],
               'tool_calls': sum(a['tool_calls'] for a in attempts),
               'three_turn': sum(a['turns'] == 3 for a in attempts),
               'resume_sha_unchanged': True, 'journal_sha256': digest}
        rows.append(row)
        print(json.dumps(row), flush=True)
    print(json.dumps({'status': 'MECHANICAL_CHECK_PASSED', 'runs': rows,
                      'quality': 'NOT EVALUATED', 'paid_calls': 0}))


if __name__ == '__main__':
    main()
