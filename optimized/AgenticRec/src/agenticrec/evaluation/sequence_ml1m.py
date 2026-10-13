"""Fixed short-sequence retraining and post-hoc replay of the original 40% cohort."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import time

import pyarrow.parquet as pq
import torch

from .simple_baselines import ROOT, load, write_new, digest, verify_source, quality, paired_interval
from .rank_upgrade_models import ShortSequence, padded_histories, multi_positive_loss, mixed_scores, ordered_ids
from ..training import load_train_contract

SOURCE = ROOT/'optimized/artifacts/comparison/c1_hit_20261013'
CONTROLS = ROOT/'optimized/artifacts/comparison/simple_baselines_20261013'
DATA = ROOT/'optimized/data/processed/ml-1m-v1'
PROTOCOL = ROOT/'optimized/reports/sequence_ml1m_protocol_20261013.md'
SEEDS = (7, 42, 2026)
METRICS = ('hit', 'recall', 'ndcg', 'mrr')


def training_examples(rows):
    sequences = defaultdict(list)
    for r in rows:
        if r['rating'] < 4:
            raise ValueError('Only train positives allowed')
        sequences[r['user_id']].append((r['timestamp'], r['item_id']))
    result = []
    for u, seq in sorted(sequences.items()):
        seq = sorted(seq)
        if len(seq) != len({i for _, i in seq}):
            raise ValueError('Duplicate training item')
        cut = int(len(seq)*.8)
        if cut < 10 or cut >= len(seq):
            continue
        targets = sorted(i for _, i in seq[cut:])
        for h in (3, 5, 10):
            liked = [i for _, i in seq[cut-h:cut]]
            if set(liked) & set(targets):
                raise ValueError('History/target overlap')
            result.append({'liked_item_ids': liked, 'relevant_item_ids': targets})
    return result


def verify(directory):
    contract = load(directory/'contract.json')
    for name, sha in contract['files'].items():
        if digest(ROOT/name) != sha:
            raise ValueError('Frozen file changed: '+name)
    return contract


def prepare(directory):
    if directory.exists():
        raise FileExistsError('New directory required')
    verify_source(SOURCE)
    _, _, n_items, _, _ = load_train_contract()
    old = load(CONTROLS/'report.json')
    if old['systems']['candidate']['hits'] != 72 or old['systems']['popular']['hits'] != 72:
        raise ValueError('Wrong original 40% study')
    assert load(CONTROLS/'protocol.json')['source_directory'] == str(SOURCE)
    assert old['predictions_sha256'] == digest(CONTROLS/'predictions.json')
    catalog = load(ROOT/'optimized/artifacts/comparison/three_way_20261013/catalog.json')['items']
    public = load(SOURCE/'formal_public.json')
    # Fixed original IDs; no test targets inspected by training preparation.
    public = [r for r in public if r['episode_id'].startswith('formal-hit-')]
    assert len(public) == 180
    rows = pq.read_table(DATA/'train_positive.parquet').to_pylist()
    examples = training_examples(rows)
    directory.mkdir(parents=True)
    write_new(directory/'train_examples.json', examples)
    write_new(directory/'public.json', public)
    paths = [PROTOCOL, Path(__file__), ROOT/'optimized/AgenticRec/src/agenticrec/evaluation/rank_upgrade_models.py',
             ROOT/'optimized/AgenticRec/src/agenticrec/evaluation/simple_baselines.py',
             ROOT/'optimized/AgenticRec/scripts/audit_sequence_ml1m.py',
             ROOT/'optimized/AgenticRec/scripts/audit_three_way_comparison.py',
             ROOT/'optimized/artifacts/comparison/three_way_20261013/catalog.json',
             directory/'train_examples.json', directory/'public.json']
    paths += [DATA/n for n in ('train.parquet', 'train_positive.parquet', 'id_map.json', 'eval_test.parquet')]
    paths += [SOURCE/n for n in ('protocol.json', 'formal_public.json', 'formal_private.json', 'live_run/episodes.jsonl')]
    paths += [CONTROLS/n for n in ('protocol.json', 'predictions.json', 'scores.json', 'report.json')]
    write_new(directory/'contract.json', {'status': 'FROZEN', 'scope': 'post_hoc_original_180',
        'items': len(catalog), 'model_items': n_items, 'training_examples': len(examples), 'training_users': len(examples)//3,
        'epochs': 10, 'alpha': .5, 'seeds': list(SEEDS), 'primary_seed': 42, 'parameter_search': False,
        'api_requests': 0, 'files': {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}})
    return {'status': 'PREPARED', 'training_examples': len(examples), 'items': len(catalog)}


def train(directory):
    contract = verify(directory)
    if (directory/'training.json').exists() or (directory/'models').exists():
        raise FileExistsError('Training evidence cannot be overwritten')
    (directory/'models').mkdir()
    torch.set_num_threads(2)
    examples = load(directory/'train_examples.json')
    histories = padded_histories(examples)
    targets = torch.zeros((len(examples), contract['model_items']))
    for n, row in enumerate(examples):
        targets[n, [i-1 for i in row['relevant_item_ids']]] = 1
    logs = []
    started = time.perf_counter()
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = ShortSequence(contract['model_items'])
        optimizer = torch.optim.Adam(model.parameters(), lr=.002, weight_decay=1e-4)
        curve = []
        for epoch in range(1, 11):
            model.train()
            total = 0.
            for batch in torch.randperm(len(examples)).split(32):
                loss = multi_positive_loss(model(histories[batch]), targets[batch], histories[batch])
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                total += float(loss.detach())*len(batch)
            step = {'seed': seed, 'epoch': epoch, 'mean_loss': total/len(examples)}
            curve.append(step)
            print(json.dumps(step), flush=True)
        path = directory/'models'/f'seed_{seed}.pt'
        torch.save({'state_dict': model.state_dict(), 'seed': seed, 'epoch': 10,
                    'contract_sha256': digest(directory/'contract.json')}, path)
        logs.append({'seed': seed, 'curve': curve, 'checkpoint_sha256': digest(path)})
    result = {'status': 'TRAINED', 'runs': logs, 'elapsed_seconds': time.perf_counter()-started,
              'valid_test_used_for_training_or_selection': False, 'api_requests': 0}
    write_new(directory/'training.json', result)
    return result


def summarize(records):
    by = defaultdict(dict)
    for row in records:
        if row['episode_id'] in by[row['system']]:
            raise ValueError('Duplicate scored episode')
        by[row['system']][row['episode_id']] = row
    keys = sorted(by['candidate'])
    if len(keys) != 180 or any(sorted(rows) != keys for rows in by.values()) or len(by) != 7:
        raise ValueError('Incomplete paired study')
    result = {'systems': {}, 'comparisons': {}}
    for name, rows in by.items():
        result['systems'][name] = {'n': 180, 'hits': int(sum(r['metrics']['hit'] for r in rows.values())),
            **{m: sum(r['metrics'][m] for r in rows.values())/180 for m in METRICS},
            'history': {str(h): {'n': sum(r['history_n']==h for r in rows.values()),
                         'hits': int(sum(r['metrics']['hit'] for r in rows.values() if r['history_n']==h))}
                        for h in (3, 5, 10)}}
    for bn, baseline in enumerate(('candidate', 'popular', 'itemcf')):
        a, b = by['sequence42'], by[baseline]
        wins = sum(a[i]['metrics']['hit'] > b[i]['metrics']['hit'] for i in keys)
        losses = sum(a[i]['metrics']['hit'] < b[i]['metrics']['hit'] for i in keys)
        n = wins+losses
        p = min(1., 2*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/2**n)
        result['comparisons'][baseline] = {'wins': wins, 'losses': losses, 'hit_exact_p': p,
            'metrics': {m: paired_interval([a[i]['metrics'][m]-b[i]['metrics'][m] for i in keys],
                                           2026101314+bn*10+mn) for mn,m in enumerate(METRICS)}}
    previous = 0.
    for rank, name in enumerate(sorted(result['comparisons'], key=lambda k: result['comparisons'][k]['hit_exact_p'])):
        row = result['comparisons'][name]
        previous = max(previous, min(1., row['hit_exact_p']*(3-rank)))
        row['hit_holm_p'] = previous
    main = result['comparisons']['candidate']
    result['conclusion'] = ('GAIN_ON_EXPOSED_COHORT' if main['metrics']['hit']['ci95'][0]>0 and main['hit_holm_p']<.05
                            else 'NO VERIFIED QUALITY GAIN')
    result['sequence_three_seed_mean_hit'] = sum(result['systems']['sequence'+str(s)]['hit'] for s in SEEDS)/3
    return result


def evaluate(directory):
    contract = verify(directory)
    if (directory/'predictions.json').exists():
        raise FileExistsError('Predictions cannot be overwritten')
    torch.set_num_threads(2)
    public = load(directory/'public.json')
    catalog = {int(k): v for k,v in load(ROOT/'optimized/artifacts/comparison/three_way_20261013/catalog.json')['items'].items()}
    rows = pq.read_table(DATA/'train_positive.parquet', columns=['item_id']).column(0).to_pylist()
    pop = torch.zeros(contract['model_items'], dtype=torch.float64)
    for item in rows: pop[item-1] += 1
    histories = [{'liked_item_ids': [r['item_id'] for r in p['catalog_context']]} for p in public]
    x = padded_histories(histories)
    predictions = []
    training = load(directory/'training.json')
    for seed in SEEDS:
        path = directory/'models'/f'seed_{seed}.pt'
        assert digest(path) == next(r['checkpoint_sha256'] for r in training['runs'] if r['seed']==seed)
        checkpoint = torch.load(path, map_location='cpu')
        assert checkpoint['contract_sha256'] == digest(directory/'contract.json')
        model = ShortSequence(contract['model_items']); model.load_state_dict(checkpoint['state_dict']); model.eval()
        with torch.no_grad(): logits = model(x)
        for n, pub in enumerate(public):
            liked = histories[n]['liked_item_ids']
            values = mixed_scores(logits[n], pop, .5)
            ids = sorted(set(catalog)-set(liked), key=lambda i: (-float(values[i-1]),i))[:10]
            predictions.append({'system': 'sequence'+str(seed), 'episode_id': pub['episode_id'],
                'history_n': len(liked), 'item_ids': ids, 'raw_ids': [catalog[i]['raw_id'] for i in ids]})
    controls = [r for r in load(CONTROLS/'predictions.json') if r['system'] != 'content']
    assert len(predictions) == 540 and len(controls) == 720
    write_new(directory/'predictions.json', predictions+controls)
    # Future labels first enter this scoring phase, after all predictions persisted.
    labels = {r['episode_id']: r for r in load(SOURCE/'formal_private.json')}
    scored = []
    for r in predictions+controls:
        label = labels[r['episode_id']]
        assert label['history_n'] == r['history_n']
        scored.append(dict(r, metrics=quality(r['raw_ids'], label['labels'][0]['relevant_raw_ids'])))
    write_new(directory/'scores.json', scored)
    result = {'status': 'EVALUATED', 'scope': 'post_hoc_original_180', 'held_out_blind_test': False,
              'api_requests': 0, 'downloads': 0, 'default_changed': False,
              'strict_agent_tasks_evaluated': False, 'predictions_sha256': digest(directory/'predictions.json'),
              'contract_sha256': digest(directory/'contract.json'), **summarize(scored)}
    write_new(directory/'report.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'train', 'evaluate'))
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(globals()[args.phase](args.directory.resolve()), indent=2), flush=True)


if __name__ == '__main__':
    main()
