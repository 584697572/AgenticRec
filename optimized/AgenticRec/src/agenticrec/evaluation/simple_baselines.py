"""Post-hoc, offline controls on the frozen C1 cohort; never calls an LLM."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random

from .frontier_state import empty_state, feasible_ids, verify_delivery

ROOT = Path(__file__).resolve().parents[5]
CONTROLS = ('popular', 'itemcf', 'content')
SYSTEMS = (*CONTROLS, 'C1', 'candidate')
METRICS = ('hit', 'recall', 'ndcg', 'mrr')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as out:
        json.dump(value, out, indent=2, sort_keys=True, ensure_ascii=True)
        out.write('\n')


def verify_source(source):
    old = load(source / 'protocol.json')
    if old['status'] != 'FROZEN':
        raise ValueError('Source study is not frozen')
    for group, base in [('files', source), ('sources', ROOT), ('resources', ROOT)]:
        for name, expected in old[group].items():
            if digest(base / name) != expected:
                raise ValueError('Changed source evidence: ' + name)
    final = load(source / 'final_report.json')
    if load(source / 'live_run/execution.json')['status'] != 'COMPLETE':
        raise ValueError('Source live study is incomplete')
    if final['live_quality']['C1']['n'] != 180:
        raise ValueError('Wrong source cohort')
    return old


def public_state(public):
    ids = [r['item_id'] for r in public['catalog_context']]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate public history')
    state = empty_state()
    state['liked_item_ids'] = ids
    state['seen_item_ids'] = ids[:]
    return state


class SimpleRanker:
    """Binary train ItemCF, popular counts, or original train TF-IDF content.

    ItemCF uses all neighbors: sum_j |U_i & U_j| / sqrt(|U_i| |U_j|).
    No user ID, labels, hidden history, nearest-neighbor tuning or RRF input.
    Ties use item ID, without a popularity blend. No supported seed -> popular.
    """
    def __init__(self, catalog, train_edges, content_vectors):
        self.catalog, self.content_vectors = catalog, content_vectors
        self.item_users, self.user_items = defaultdict(set), defaultdict(set)
        edges = set()
        for user, item in train_edges:
            if (user, item) in edges or user <= 0 or item <= 0:
                raise ValueError('Invalid or duplicate training edge')
            edges.add((user, item))
            self.item_users[item].add(user)
            self.user_items[user].add(item)
        self.popularity = Counter({i: len(users) for i, users in self.item_users.items()})
        self.cooccurrences = {}

    def scores(self, eligible, state, method):
        if method not in CONTROLS:
            raise ValueError('Unknown control')
        liked = state['liked_item_ids']
        if len(liked) != len(set(liked)) or any(i not in self.catalog for i in liked):
            raise ValueError('Public likes must be unique confirmed items')
        if method == 'popular':
            return {i: float(self.popularity[i]) for i in eligible}
        scores = defaultdict(float)
        if method == 'itemcf':
            supported = [i for i in liked if self.popularity[i]]
            if not supported:
                return self.scores(eligible, state, 'popular')
            for seed in supported:
                if seed not in self.cooccurrences:
                    counts = Counter()
                    for user in sorted(self.item_users[seed]):
                        counts.update(sorted(self.user_items[user]))
                    self.cooccurrences[seed] = counts
                for item, count in self.cooccurrences[seed].items():
                    scores[item] += count / math.sqrt(self.popularity[seed] * self.popularity[item]) / len(liked)
        else:
            if not liked:
                return self.scores(eligible, state, 'popular')
            query = defaultdict(float)
            for seed in liked:
                for term, weight in self.content_vectors[seed].items():
                    query[term] += weight / len(liked)
            for genre in state['include_genres']:
                query['genre:' + genre.casefold()] += 1
            for item in eligible:
                scores[item] = sum(query.get(term, 0) * w for term, w in self.content_vectors[item].items())
        return {i: scores[i] for i in eligible}

    def rank(self, state, method):
        eligible = feasible_ids(self.catalog, state)
        scores = self.scores(eligible, state, method)
        return sorted(eligible, key=lambda i: (-scores[i], i))[:state['k']]


def quality(ids, relevant):
    if len(ids) != 10 or len(ids) != len(set(ids)):
        raise ValueError('Expected exactly ten unique recommendations')
    targets = set(relevant)
    if not targets:
        raise ValueError('Quality user has no positive targets')
    ranks = [n for n, item in enumerate(ids, 1) if item in targets]
    ideal = sum(1 / math.log2(n + 1) for n in range(1, min(10, len(targets)) + 1))
    return {'hit': float(bool(ranks)), 'recall': len(ranks) / len(targets),
            'ndcg': sum(1 / math.log2(n + 1) for n in ranks) / ideal,
            'mrr': 1 / ranks[0] if ranks else 0.}


def paired_interval(delta, seed):
    rng = random.Random(seed)
    samples = sorted(sum(rng.choices(delta, k=len(delta))) / len(delta) for _ in range(10000))
    def percentile(q):
        pos = (len(samples) - 1) * q
        lo = int(pos)
        return samples[lo] + (samples[min(lo + 1, len(samples) - 1)] - samples[lo]) * (pos - lo)
    return {'mean': sum(delta) / len(delta), 'ci95': [percentile(.025), percentile(.975)]}


def summarize(records):
    by = {s: {r['episode_id']: r for r in records if r['system'] == s} for s in SYSTEMS}
    if len(records) != 900 or any(len(v) != 180 for v in by.values()):
        raise ValueError('Incomplete or duplicated cohort')
    keys = sorted(by['C1'])
    if any(sorted(v) != keys for v in by.values()):
        raise ValueError('Unpaired cohort')
    result = {'systems': {}, 'comparisons': {}}
    for s, rows in by.items():
        if Counter(r['history_n'] for r in rows.values()) != Counter({3: 60, 5: 60, 10: 60}):
            raise ValueError('Wrong history strata')
        result['systems'][s] = {'n': 180, 'hits': int(sum(r['metrics']['hit'] for r in rows.values())),
            **{m: sum(r['metrics'][m] for r in rows.values()) / 180 for m in METRICS},
            'history': {str(h): {'n': 60, 'hits': int(sum(r['metrics']['hit'] for r in rows.values() if r['history_n'] == h))}
                        for h in (3, 5, 10)}}
    for ai, a in enumerate(('C1', 'candidate')):
        for bi, b in enumerate(CONTROLS):
            values = {m: paired_interval([by[a][i]['metrics'][m] - by[b][i]['metrics'][m] for i in keys],
                                         2026101305 + ai * 100 + bi * 10 + mi)
                      for mi, m in enumerate(METRICS)}
            wins = sum(by[a][i]['metrics']['hit'] > by[b][i]['metrics']['hit'] for i in keys)
            losses = sum(by[a][i]['metrics']['hit'] < by[b][i]['metrics']['hit'] for i in keys)
            n = wins + losses
            p = min(1., 2 * sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n)
            result['comparisons'][a + '_minus_' + b] = {'n': 180, 'wins': wins, 'losses': losses,
                'metrics': values, 'hit_exact_p': p}
    # Holm-adjust the six fixed Hit comparisons; all other metrics are descriptive.
    ordered = sorted(result['comparisons'], key=lambda k: result['comparisons'][k]['hit_exact_p'])
    adjusted = 0.
    for rank, name in enumerate(ordered):
        row = result['comparisons'][name]
        adjusted = max(adjusted, min(1., row['hit_exact_p'] * (len(ordered) - rank)))
        row['hit_holm_p'] = adjusted
    return result


def prepare(source, directory):
    if directory.exists():
        raise FileExistsError('Use a fresh evidence directory')
    old = verify_source(source)
    anchored = ['protocol.json', 'formal_public.json', 'formal_private.json',
                'offline_predictions.json', 'live_run/episodes.jsonl', 'final_report.json']
    new_files = [Path(__file__), ROOT/'optimized/reports/simple_baselines_protocol_20261013.md']
    write_new(directory/'protocol.json', {'status': 'FROZEN', 'scope': 'post_hoc_offline_controls',
        'source_directory': str(source.resolve()), 'source_sha256': {n: digest(source/n) for n in anchored},
        'implementation_sha256': {str(p.relative_to(ROOT)): digest(p) for p in new_files},
        'source_selected': old['selected'], 'systems': list(SYSTEMS),
        'quality_users': 180, 'top_k': 10, 'bootstrap_samples': 10000,
        'api_requests': 0, 'training': 'existing train_positive only',
        'parameter_selection': 'none', 'held_out_blind_test': False})
    return {'status': 'FROZEN', 'api_requests': 0}


def evaluate(directory):
    protocol = load(directory/'protocol.json')
    source = Path(protocol['source_directory'])
    if (directory/'predictions.json').exists():
        raise FileExistsError('Predictions cannot be overwritten')
    for name, expected in protocol['source_sha256'].items():
        if digest(source/name) != expected:
            raise ValueError('Source changed: ' + name)
    for name, expected in protocol['implementation_sha256'].items():
        if digest(ROOT/name) != expected:
            raise ValueError('Implementation changed: ' + name)
    verify_source(source)
    from ..retrieval.sources import CandidateRetriever
    from .preference_ranker import PreferenceRanker, baseline_sources
    from ..training import load_train_contract
    _, _, _, _, edges = load_train_contract()
    base = CandidateRetriever.from_frozen()  # No known-user model/checkpoint needed.
    catalog_blob = load(ROOT/'optimized/artifacts/comparison/three_way_20261013/catalog.json')
    catalog = {int(k): v for k, v in catalog_blob['items'].items()}
    ranker = SimpleRanker(catalog, edges, base.content_vectors)
    if ranker.popularity != base.popularity or ranker.user_items != base.user_items:
        raise ValueError('Training graph differs from C1')
    controls = {'C1': PreferenceRanker(base, catalog, {'weights': {'cf': 1., 'content': 1., 'popularity': 1.}}),
                'candidate': PreferenceRanker(base, catalog, protocol['source_selected'])}
    private = {r['episode_id']: r for r in load(source/'formal_private.json')}
    public = load(source/'formal_public.json')
    quality_ids = {i for i, r in private.items() if r['category'] == 'session_quality'}
    if len(quality_ids) != 180 or len({private[i]['evaluator_raw_user_id'] for i in quality_ids}) != 180:
        raise ValueError('Wrong number of independent users')
    records = []
    for pub in public:
        if pub['episode_id'] not in quality_ids:
            continue
        # Scoring targets never enter a ranker: only publicly stated history.
        state = public_state(pub)
        snapshot = baseline_sources(base, catalog, state)
        for method in SYSTEMS:
            ids = ranker.rank(state, method) if method in CONTROLS else controls[method].rank(state, snapshot)
            if verify_delivery(ids, catalog, state):
                raise ValueError('Invalid public delivery')
            records.append({'episode_id': pub['episode_id'], 'system': method,
                'history_n': len(state['liked_item_ids']), 'item_ids': ids,
                'raw_ids': [catalog[i]['raw_id'] for i in ids]})
    # Persist all predictions before inspecting any future quality target.
    write_new(directory/'predictions.json', records)
    old_predictions = {(r['system'], r['episode_id']): r for r in load(source/'offline_predictions.json')}
    live = {(r['system'], r['episode_id']): r['turns'][0]['raw_ids']
            for r in (json.loads(line) for line in (source/'live_run/episodes.jsonl').read_text(encoding='utf-8').splitlines())
            if r['category'] == 'session_quality'}
    for r in records:
        label = private[r['episode_id']]
        if label['history_n'] != r['history_n']:
            raise ValueError('History length differs')
        r['metrics'] = quality(r['raw_ids'], label['labels'][0]['relevant_raw_ids'])
        if r['system'] in controls:
            previous = old_predictions[(r['system'], r['episode_id'])]
            if previous['raw_ids'] != r['raw_ids'] or live[(r['system'], r['episode_id'])] != r['raw_ids']:
                raise ValueError('C1/candidate replay differs from actual live study')
            if any(not math.isclose(r['metrics'][m], previous['metrics'][m], abs_tol=1e-12) for m in METRICS):
                raise ValueError('Independent scoring differs from old study')
    write_new(directory/'scores.json', records)
    result = {'status': 'VERIFIED', 'scope': 'post_hoc_offline_controls',
              'api_requests': 0, 'new_training': False, 'downloads': 0,
              'candidate_tuning': False, 'strict_agent_tasks_evaluated': False,
              'source_frozen_hashes_verified': True, 'live_rankings_replayed': 360,
              'catalog_items': len(catalog), 'train_positive_edges': len(edges),
              'all_predictions_valid': True, 'predictions_sha256': digest(directory/'predictions.json'),
              'protocol_sha256': digest(directory/'protocol.json'), **summarize(records)}
    write_new(directory/'report.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'evaluate'))
    parser.add_argument('--source', type=Path, default=ROOT/'optimized/artifacts/comparison/c1_hit_20261013')
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    value = prepare(args.source, args.directory) if args.command == 'prepare' else evaluate(args.directory)
    print(json.dumps(value, indent=2))


if __name__ == '__main__':
    main()
