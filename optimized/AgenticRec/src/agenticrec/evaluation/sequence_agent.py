"""Frozen short-sequence ranking hooked into the actual C1 text/state runtime."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
import random
import time

import torch

from .simple_baselines import ROOT, load, digest, write_new, paired_interval
from .rank_upgrade import (ARCHIVE, verify_frozen as verify_ranking, archive_rows,
                           temporal_cases, environment, load_model)
from .rank_upgrade_models import mixed_scores, padded_histories
from .frontier_experiment import FrontierSystem
from .frontier_state import empty_state, merge_state, feasible_ids, verify_delivery
from .preference_ranker import PreferenceRanker
from .three_way_comparison import MatchedJournal, score
from .comparison_run import append, quantile
from ..adapters.openai_compatible import OpenAICompatibleTransport
from ..runtime.secrets import read_api_key

SOURCE = ROOT/'optimized/artifacts/comparison/rank_upgrade_20261013'
SYSTEMS = ('sequence', 'mixed', 'itemcf', 'popular')
PROTOCOL = ROOT/'optimized/reports/sequence_agent_protocol_20261013.md'


class PublicRanker:
    """Full eligible catalog, frozen ranking; never receives evaluator labels."""
    def __init__(self, system, env, model=None):
        if system not in SYSTEMS:
            raise ValueError('Unknown ranking system')
        self.system, self.env, self.model = system, env, model
        if system == 'sequence' and model is None:
            raise ValueError('Frozen sequence checkpoint required')
        self.mixed = PreferenceRanker(env['base'], env['catalog'],
            {'weights': {'cf': 1., 'content': .5, 'popularity': .25}}) if system == 'mixed' else None

    def rank(self, state):
        eligible = feasible_ids(self.env['catalog'], state)
        if self.mixed is not None:
            return self.mixed.rank(state)
        liked = state['liked_item_ids']
        index = self.env['index']
        if any(i not in self.env['catalog'] for i in liked) or len(liked) != len(set(liked)):
            raise ValueError('Only unique public catalog likes allowed')
        if self.system == 'popular' or not liked:
            scores = index.popularity
        elif self.system == 'itemcf':
            scores = index.cosine(liked)
        else:
            self.model.eval()
            with torch.no_grad():
                logits = self.model(padded_histories([{'liked_item_ids': liked[-10:]}]))[0]
            scores = mixed_scores(logits, index.popularity, .5)
        if not torch.isfinite(scores).all():
            raise ValueError('Nonfinite frozen ranking')
        values = scores.tolist()
        return sorted(eligible, key=lambda i: (-values[i-1], i))[:state['k']]


class SequenceAgentSystem(FrontierSystem):
    """Reuse actual C1 parse/turn; all four systems share the execution gate."""
    def __init__(self, journal, ranker):
        self.journal, self.ranking = journal, ranker
        self.catalog = ranker.env['catalog']
        self.transport = (OpenAICompatibleTransport(api_base='https://api.deepseek.com',
            api_key=journal._key, sender=journal.sender) if journal else None)
        self.state, self.history = empty_state(), []

    def execute(self, public, index, delta):
        if index == 0:
            self.state, self.history = empty_state(), []
        self.state = merge_state(self.state, delta, 0)
        eligible = feasible_ids(self.catalog, self.state)
        clock = time.perf_counter()
        ids = self.ranking.rank(self.state)
        rank_seconds = time.perf_counter()-clock
        errors = verify_delivery(ids, self.catalog, self.state)
        answer = '\n'.join(f"{n}. {self.catalog[i]['title']} ({self.catalog[i]['year']})"
                           for n, i in enumerate(ids, 1))
        self.state['previous_item_ids'] = ids
        self.history.extend([{'role': 'user', 'content': public['messages'][index]},
                             {'role': 'assistant', 'content': answer}])
        return {'status': 'OK' if not errors else 'CONSTRAINT_FAILURE', 'answer': answer,
                'mapped_optimized_ids': ids, 'raw_ids': [self.catalog[i]['raw_id'] for i in ids],
                'extraction_verified': True, 'tool_calls': 2, 'delta': delta,
                'state': deepcopy(self.state), 'policy_action': 0, 'eligible_count': len(eligible),
                'verification_errors': errors, 'ranking_seconds': rank_seconds}


def assets():
    verify_ranking(SOURCE)
    env = environment(SOURCE)
    if any(v['year'] is None for v in env['catalog'].values()):
        raise ValueError('Shared legacy year gate requires known metadata')
    model = load_model(SOURCE, 'route3', 42, {'epoch': 10, 'alpha': .5}, len(env['catalog']))
    return env, model


def label(k=10, genres=(), excluded=(), lo=None, hi=None, relevant=(), previous=False, feasible=True):
    return {'k': k, 'genres': list(genres), 'exclude_genres': list(excluded), 'year_min': lo,
            'year_max': hi, 'excluded_raw_ids': [], 'relevant_raw_ids': list(relevant),
            'exclude_previous': previous, 'feasible': feasible}


def user_case(catalog, histories, targets, user, h, episode, mode=None):
    liked = [i for _, i in histories[user][-h:]]
    context = [{k: catalog[i][k] for k in ('title', 'year', 'item_id')} for i in liked]
    names = '; '.join(f"{catalog[i]['title']} ({catalog[i]['year']})" for i in liked)
    first = f'I have watched and liked these movies: {names}. The list is chronological, from oldest to most recent. Please recommend exactly 10 other movies matching my taste. Exclude all the watched movies listed. Give only titles.'
    messages, labels = [first], [label(relevant=[catalog[i]['raw_id'] for i in sorted(targets[user])])]
    if mode is not None:
        labels[0]['relevant_raw_ids'] = []
        if mode in (1, 2, 3, 4):
            messages[0] += ' Focus on action movies released from 1985 through 2000, inclusive.'
            labels[0].update(genres=['Action'], year_min=1985, year_max=2000)
        if mode == 0:
            messages.append('Keep my previous preferences. Now give exactly 10 drama movies from 1991 through 2000, inclusive. Do not include horror movies. Exclude the already watched movies. Give only titles.')
            labels.append(label(genres=['Drama'], excluded=['Horror'], lo=1991, hi=2000))
        elif mode == 1:
            messages.append('Keep the previous year limits and action requirement, and also require thriller. Give exactly 10 movies. Exclude those I already watched. Give only titles.')
            labels.append(label(genres=['Action', 'Thriller'], lo=1985, hi=2000))
        elif mode == 2:
            messages.append('Switch to comedy instead of action, but keep the previous year limits and my preferences. Give exactly 10 movies and exclude those already watched. Give only titles.')
            labels.append(label(genres=['Comedy'], lo=1985, hi=2000))
        elif mode == 3:
            messages.append('Discard all previous genre and year conditions. Give exactly 10 romance movies released from 1970 through 1999, inclusive. Keep my preferences and exclude the watched movies. Give only titles.')
            labels.append(label(genres=['Romance'], lo=1970, hi=1999))
        elif mode == 4:
            messages.extend(['Keep the action requirement and my preferences. Narrow the year limits to from 1991 through 1998, inclusive. Give exactly 10 movies and exclude the watched movies. Give only titles.',
                             'Keep all previous conditions and my preferences. Give exactly 10 different movies, excluding the previous recommendations and the watched movies. Give only titles.'])
            labels.extend([label(genres=['Action'], lo=1991, hi=1998), label(genres=['Action'], lo=1991, hi=1998, previous=True)])
        else:
            messages.extend(['Keep my preferences. Give exactly 10 different movies, excluding the previous recommendations and the watched movies. Give only titles.',
                             'Discard all previous genre and year conditions. Now give exactly 5 comedy movies from 1990 through 2005, inclusive. Keep my preferences and exclude the watched movies. Give only titles.'])
            labels.extend([label(previous=True), label(k=5, genres=['Comedy'], lo=1990, hi=2005)])
    excluded = [catalog[i]['raw_id'] for i in liked]
    for lab in labels:
        lab['excluded_raw_ids'] = excluded[:]
        state = empty_state()
        state.update(k=lab['k'], include_genres=lab['genres'], exclude_genres=lab['exclude_genres'],
                     year_min=lab['year_min'], year_max=lab['year_max'], seen_item_ids=liked)
        if len(feasible_ids(catalog, state)) < lab['k']+(20 if lab['exclude_previous'] else 0):
            raise ValueError('Feedback infeasible without model-independent margin')
    return ({'episode_id': episode, 'catalog_context': context, 'messages': messages},
            {'episode_id': episode, 'category': 'session_quality' if mode is None else 'feedback',
             'history_n': h, 'scenario': 'history:'+str(h) if mode is None else 'feedback:'+str(mode),
             'evaluator_raw_user_id': user, 'labels': labels})


def extra_cases(catalog, prefix, n_constraints, n_refusals):
    cases, rng = [], random.Random(2026101315 if prefix == 'formal' else 2026101314)
    while len(cases) < n_constraints:
        genre = rng.choice(('Comedy', 'Drama', 'Action', 'Thriller', 'Romance', 'Adventure', 'Crime', 'Sci-Fi'))
        lo, hi = rng.choice(((1960, 1985), (1986, 1993), (1994, 2000), (1970, 1999), (1990, 2005)))
        extra = rng.choice(('Action', 'Drama', 'Thriller')) if len(cases)%4 == 3 else None
        include = list(dict.fromkeys([genre]+([extra] if extra else [])))
        exclude = ['Horror'] if len(cases)%3 == 1 else []
        lab = label(genres=include, excluded=exclude, lo=lo, hi=hi)
        state = empty_state(); state.update(include_genres=include, exclude_genres=exclude, year_min=lo, year_max=hi)
        if len(feasible_ids(catalog, state)) < 10:
            continue
        episode = f'{prefix}-constraints-{len(cases):03d}'
        msg = f"Please give exactly 10 {' and '.join(g.lower() for g in include)} movies released between {lo} and {hi}, inclusive."
        if exclude: msg += ' Do not include horror movies.'
        cases.append(({'episode_id': episode, 'catalog_context': [], 'messages': [msg+' Give only titles.']},
                      {'episode_id': episode, 'category': 'constraints', 'history_n': None, 'scenario': 'explicit_constraints', 'labels': [lab]}))
    for n in range(n_refusals):
        episode = f'{prefix}-unanswerable-{n:03d}'
        if n%2:
            msg = 'Give exactly 10 comedy movies. Do not include comedy movies. Give only titles.'
            lab = label(genres=['Comedy'], excluded=['Comedy'], feasible=False)
        else:
            lo, hi = 2030+n, 2031+n
            msg = f'Give exactly 10 movies released from {lo} through {hi}, inclusive. Give only titles.'
            lab = label(lo=lo, hi=hi, feasible=False)
        cases.append(({'episode_id': episode, 'catalog_context': [], 'messages': [msg]},
                      {'episode_id': episode, 'category': 'unanswerable', 'history_n': None, 'scenario': 'infeasible', 'labels': [lab]}))
    return cases


def prepare(directory):
    if directory.exists(): raise FileExistsError('Fresh study directory required')
    env, model = assets(); catalog = env['catalog']
    _, ratings, _ = archive_rows()
    mapping = {r['raw_id']: i for i, r in catalog.items()}
    t85 = load(SOURCE/'prepared.json')['cutoffs_ms'][2]
    histories, targets, eligible = temporal_cases(ratings, mapping, t85, 10**20)
    excluded = set(load(SOURCE/'training_users_private.json'))
    for phase in ('development', 'test'):
        excluded.update(r['evaluator_raw_user_id'] for r in load(SOURCE/(phase+'_private.json')))
    users = sorted(set(eligible)-excluded); random.Random(2026101313).shuffle(users)
    if len(users) < 159: raise ValueError('Insufficient entirely unused users')
    directory.mkdir(parents=True)
    for phase, quality_users, feedback_users, nc, nr in [
            ('calibration', users[:3], users[3:9], 2, 1),
            ('formal', users[9:129], users[129:159], 24, 8)]:
        cases = [user_case(catalog, histories, targets, u, (3, 5, 10)[n%3], f'{phase}-quality-{n:03d}')
                 for n, u in enumerate(quality_users)]
        cases += [user_case(catalog, histories, targets, u, (3, 5, 10)[n%3], f'{phase}-feedback-{n:03d}', n%6)
                  for n, u in enumerate(feedback_users)]
        cases += extra_cases(catalog, phase, nc, nr)
        write_new(directory/(phase+'_public.json'), [p for p, q in cases])
        write_new(directory/(phase+'_private.json'), [q for p, q in cases])
    write_new(directory/'catalog.json', catalog)
    write_new(directory/'truth_private.json', {r['raw_id']: r for r in catalog.values()})
    write_new(directory/'excluded_users_private.json', sorted(excluded))
    result = {'status': 'PREPARED', 'source_directory': str(SOURCE), 'source_frozen_sha256': digest(SOURCE/'frozen.json'),
              'available_unused_users': len(users), 'calibration_users': 9, 'formal_users': 150,
              'formal_quality_users': 120, 'calibration_tasks': 12, 'formal_tasks': 182,
              'calibration_turns_per_system': 20, 'formal_turns_per_system': 222,
              'systems': list(SYSTEMS), 'sequence_config': {'epoch': 10, 'alpha': .5, 'seed': 42},
              'new_training': False, 'new_download_bytes': 0, 'default_changed': False, 'api_requests': 0}
    write_new(directory/'prepared.json', result)
    return result


def regression(directory):
    env, model = assets()
    old_predictions = {(r['system'], r['seed'], r['episode_id']): r['item_ids'] for r in load(SOURCE/'test_predictions.json')}
    rankers = {s: PublicRanker(s, env, model if s == 'sequence' else None) for s in SYSTEMS}
    aliases = {'sequence': ('route3', 42), 'mixed': ('c1_weighted', None), 'popular': ('popular', None), 'itemcf': ('itemcf', None)}
    counts = Counter()
    for p in load(SOURCE/'test_public.json'):
        state = empty_state(); state.update(liked_item_ids=p['liked_item_ids'], seen_item_ids=p['liked_item_ids'])
        for s, ranker in rankers.items():
            expected = old_predictions[(*aliases[s], p['episode_id'])]
            actual = ranker.rank(state)
            if actual != expected:
                raise ValueError('Unconstrained ranking regression: '+s+':'+p['episode_id'])
            counts[s] += 1
    result = {'status': 'VERIFIED', 'unchanged_previous_rankings': dict(counts), 'api_requests': 0, 'labels_used': False}
    write_new(directory/'ranking_regression.json', result)
    return result


def verify(directory):
    value = load(directory/'frozen.json')
    verify_ranking(SOURCE)
    for group, base in [('inputs', directory), ('sources', ROOT), ('resources', ROOT)]:
        for name, expected in value[group].items():
            if digest(base/name) != expected: raise ValueError('Frozen evidence changed: '+name)
    return value


def freeze(directory):
    verify_ranking(SOURCE)
    if load(directory/'ranking_regression.json')['status'] != 'VERIFIED': raise ValueError('Missing regression')
    if load(directory/'calibration_run/execution.json')['status'] != 'COMPLETE': raise ValueError('Incomplete calibration')
    if (directory/'formal_run').exists(): raise ValueError('Cannot freeze after formal execution')
    result = {'status': 'FROZEN', 'inputs': {str(p.relative_to(directory)): digest(p) for p in directory.rglob('*') if p.is_file()},
              'sources': {str(p.relative_to(ROOT)): digest(p) for p in (ROOT/'optimized/AgenticRec/src/agenticrec').rglob('*.py')},
              'resources': {str((SOURCE/n).relative_to(ROOT)): digest(SOURCE/n) for n in
                  ('frozen.json', 'graph70.parquet', 'cache/index70.pt', 'catalog.json',
                   'route3/seed_42_epoch_10.pt', 'route3/seed_7_epoch_10.pt', 'route3/seed_2026_epoch_10.pt')},
              'formal_labels_scored': False, 'api_settings': {'provider': 'deepseek', 'model': 'deepseek-flash',
                  'max_tokens': 1024, 'temperature': 0, 'thinking': 'disabled', 'network_retries': 0},
              'max_formal_requests': 2664}
    result['sources'][str(PROTOCOL.relative_to(ROOT))] = digest(PROTOCOL)
    write_new(directory/'frozen.json', result)
    return {'status': 'FROZEN', 'sources': len(result['sources']), 'formal_labels_scored': False}


def offline(directory):
    verify(directory); env, model = assets()
    public = [r for r in load(directory/'formal_public.json') if '-quality-' in r['episode_id']]
    rows = []
    for s in SYSTEMS:
        for seed in ((42, 7, 2026) if s == 'sequence' else (None,)):
            selected = model if seed == 42 else load_model(SOURCE, 'route3', seed, {'epoch': 10, 'alpha': .5}, len(env['catalog'])) if s == 'sequence' else None
            ranker = PublicRanker(s, env, selected)
            for p in public:
                ids = [r['item_id'] for r in p['catalog_context']]
                state = empty_state(); state.update(liked_item_ids=ids, seen_item_ids=ids)
                result = ranker.rank(state)
                rows.append({'system': s, 'seed': seed, 'episode_id': p['episode_id'],
                             'item_ids': result, 'raw_ids': [env['catalog'][i]['raw_id'] for i in result]})
    write_new(directory/'offline_predictions.json', rows)
    private = {r['episode_id']: r for r in load(directory/'formal_private.json')}
    truth = load(directory/'truth_private.json')
    for r in rows:
        r['scores'] = score(dict(r, status='OK', extraction_verified=True), private[r['episode_id']]['labels'][0], truth)
    write_new(directory/'offline_scores.json', rows)
    return {'status': 'EVALUATED', 'predictions': len(rows), 'api_requests': 0, 'scope': 'direct public-input ranking component'}


def run(directory, phase):
    if phase not in ('calibration', 'formal'): raise ValueError('Unknown live phase')
    if phase == 'formal': verify(directory)
    else:
        verify_ranking(SOURCE)
        if load(directory/'ranking_regression.json')['status'] != 'VERIFIED': raise ValueError('Missing regression')
    public = load(directory/(phase+'_public.json')); private = load(directory/(phase+'_private.json'))
    run_dir = directory/(phase+'_run'); run_dir.mkdir(exist_ok=False)
    if phase == 'calibration':
        write_new(run_dir/'contract.json', {
            'sources': {str(p.relative_to(ROOT)): digest(p) for p in (ROOT/'optimized/AgenticRec/src/agenticrec').rglob('*.py')},
            'inputs': {n: digest(directory/n) for n in ('calibration_public.json', 'calibration_private.json', 'catalog.json', 'ranking_regression.json')},
            'source_frozen_sha256': digest(SOURCE/'frozen.json'), 'api_output_limit': 1024, 'network_retries': 0})
    journal = MatchedJournal(read_api_key(), run_dir)
    clock = time.perf_counter(); env, model = assets()
    systems = {s: SequenceAgentSystem(journal, PublicRanker(s, env, model if s == 'sequence' else None)) for s in SYSTEMS}
    setup_seconds = time.perf_counter()-clock
    # Warm all ranking paths without a provider call or target access.
    warm = empty_state(); warm.update(liked_item_ids=[1, 2, 3], seen_item_ids=[1, 2, 3])
    for system in systems.values(): system.ranking.rank(warm)
    truth = load(directory/'truth_private.json')
    failure, completed = None, 0
    max_requests = (20 if phase == 'calibration' else 222)*4*3
    try:
        journal.guard.refresh()
        if journal.guard.should_stop(): raise RuntimeError(journal.guard.reason)
        for n, (p, q) in enumerate(zip(public, private)):
            shift = n%4; order = SYSTEMS[shift:]+SYSTEMS[:shift]
            for s in order:
                turns, previous = [], []
                for t, lab in enumerate(q['labels']):
                    if journal.guard.should_stop(): raise RuntimeError(journal.guard.reason)
                    if len(journal.records) >= max_requests: raise RuntimeError('Finite workload request bound exhausted')
                    before = len(journal.records); clock = time.perf_counter()
                    journal.context = {'system': s, 'episode_id': p['episode_id'], 'turn': t+1}
                    try:
                        value = systems[s].turn(p, t)
                    except (ValueError, TypeError) as error:
                        value = {'status': 'ERROR', 'answer': '', 'raw_ids': [], 'extraction_verified': False,
                                 'tool_calls': 0, 'error_type': type(error).__name__, 'error': str(error)}
                    used = journal.records[before:]
                    value.update(turn=t+1, elapsed_seconds=time.perf_counter()-clock, requests=len(used),
                         input_tokens=sum(r['usage']['prompt_tokens'] for r in used) if all(r.get('usage') for r in used) else None,
                         output_tokens=sum(r['usage']['completion_tokens'] for r in used) if all(r.get('usage') for r in used) else None)
                    value['scores'] = score(value, lab, truth, previous)
                    previous = value['raw_ids']; turns.append(value)
                append(run_dir/'episodes.jsonl', {'episode_id': p['episode_id'], 'system': s, 'category': q['category'],
                       'history_n': q['history_n'], 'turns': turns, 'strict_success': all(v['scores']['strict_success'] for v in turns)})
                completed += 1
            print(json.dumps({'phase': phase, 'tasks': n+1, 'of': len(public), 'requests': len(journal.records)}), flush=True)
    except Exception as error:
        failure = type(error).__name__+': '+str(error).replace(journal._key, '[REDACTED]')
    finally:
        try: journal.guard.refresh()
        except Exception: pass
        result = {'status': 'COMPLETE' if completed == len(public)*4 and failure is None else 'BLOCKED',
                  'system_tasks_completed': completed, 'remote_requests': len(journal.records), 'failure': failure,
                  'setup_seconds': setup_seconds, 'expected_tasks': len(public)*4,
                  'expected_turns': sum(len(r['labels']) for r in private)*4}
        write_new(run_dir/'execution.json', result)
    if failure: raise RuntimeError(failure)
    return result


def statistics(directory):
    verify(directory)
    execution = load(directory/'formal_run/execution.json')
    if execution['status'] != 'COMPLETE': raise ValueError('Live evaluation incomplete')
    records = [json.loads(line) for line in (directory/'formal_run/episodes.jsonl').read_text().splitlines()]
    private = {r['episode_id']: r for r in load(directory/'formal_private.json')}
    truth = load(directory/'truth_private.json')
    summaries, by = {}, {}
    for s in SYSTEMS:
        own = [r for r in records if r['system'] == s]
        if len(own) != 182: raise ValueError('Incomplete system tasks')
        for r in own:
            previous = []
            for t, lab in zip(r['turns'], private[r['episode_id']]['labels']):
                if score(t, lab, truth, previous) != t['scores']: raise ValueError('Scoring mismatch')
                previous = t['raw_ids']
        feasible = [r for r in own if r['category'] != 'unanswerable']
        quality = [r for r in own if r['category'] == 'session_quality']
        turns = [t for r in own for t in r['turns']]
        if any(t['input_tokens'] is None or t['output_tokens'] is None for t in turns):
            raise ValueError('Unknown actual token usage')
        by[s] = {r['episode_id']: r for r in own}
        summaries[s] = {'strict_n': len(feasible), 'strict_passed': sum(r['strict_success'] for r in feasible),
            'strict_success': sum(r['strict_success'] for r in feasible)/len(feasible),
            'quality_n': len(quality), 'quality_hits': int(sum(r['turns'][0]['scores']['hit'] for r in quality)),
            'quality': {m: sum(r['turns'][0]['scores'][m] for r in quality)/len(quality) for m in ('hit', 'recall', 'ndcg', 'mrr')},
            'history': {str(h): {'n': 40, 'hits': int(sum(r['turns'][0]['scores']['hit'] for r in quality if r['history_n'] == h))} for h in (3, 5, 10)},
            'requests': sum(t['requests'] for t in turns), 'input_tokens': sum(t['input_tokens'] for t in turns),
            'output_tokens': sum(t['output_tokens'] for t in turns), 'semantic_repair_requests': sum(max(0, t['requests']-1) for t in turns),
            'turn_latency_p50': quantile([t['elapsed_seconds'] for t in turns], .5),
            'turn_latency_p95': quantile([t['elapsed_seconds'] for t in turns], .95),
            'ranking_latency_p50': quantile([t['ranking_seconds'] for t in turns if 'ranking_seconds' in t], .5),
            'ranking_latency_p95': quantile([t['ranking_seconds'] for t in turns if 'ranking_seconds' in t], .95),
            'tool_calls': sum(t['tool_calls'] for t in turns),
            'correct_refusals': sum(r['strict_success'] for r in own if r['category'] == 'unanswerable'),
            'explained_refusals': sum(r['turns'][0]['scores'].get('explanatory_refusal', False) for r in own if r['category'] == 'unanswerable'),
            'categories': {cat: {'n': sum(r['category'] == cat for r in own), 'passed': sum(r['strict_success'] for r in own if r['category'] == cat)} for cat in ('session_quality', 'feedback', 'constraints', 'unanswerable')},
            'errors': dict(Counter(t.get('error', '') for t in turns if t['status'] == 'ERROR'))}
    comparisons = {}
    for n, baseline in enumerate(('mixed', 'itemcf', 'popular')):
        keys = sorted(k for k, r in by['sequence'].items() if r['category'] == 'session_quality')
        differences = {m: [by['sequence'][k]['turns'][0]['scores'][m]-by[baseline][k]['turns'][0]['scores'][m] for k in keys] for m in ('hit', 'recall', 'ndcg', 'mrr')}
        cis = {m: paired_interval(v, 2026101316+n*10+mn) for mn, (m, v) in enumerate(differences.items())}
        delta = differences['hit']; observed = abs(sum(delta)); rng = random.Random(2026101320+n)
        extreme = sum(abs(sum(d*rng.choice((-1, 1)) for d in delta)) >= observed-1e-12 for _ in range(10000))
        comparisons['sequence_minus_'+baseline] = {'quality': cis, 'quality_paired_users': len(keys), 'hit_permutation_p': (extreme+1)/10001}
    running = 0.
    for n, key in enumerate(sorted(comparisons, key=lambda k: comparisons[k]['hit_permutation_p'])):
        running = max(running, min(1., (3-n)*comparisons[key]['hit_permutation_p']))
        comparisons[key]['hit_holm_p'] = running
    keys = sorted(k for k, r in by['sequence'].items() if r['category'] != 'unanswerable')
    strict = paired_interval([int(by['sequence'][k]['strict_success'])-int(by['mixed'][k]['strict_success']) for k in keys], 2026101324)
    balances = [json.loads(line) for line in (directory/'formal_run/balance.jsonl').read_text().splitlines()]
    def total(row): return sum((Decimal(r['total_balance']) for r in row['balance']['balance_infos'] if r['currency'] == 'CNY'), Decimal(0))
    result = {'status': 'EVALUATED', 'systems': summaries, 'comparisons': comparisons, 'strict_sequence_minus_mixed': strict,
              'strict_noninferiority_established': strict['ci95'][0] >= -.02,
              'verified_quality_gain_over_mixed': comparisons['sequence_minus_mixed']['quality']['hit']['ci95'][0] > 0 and comparisons['sequence_minus_mixed']['hit_holm_p'] < .05,
              'formal_requests': execution['remote_requests'], 'formal_tasks': 728, 'formal_turns': 888,
              'observed_formal_balance_delta_cny': str(total(balances[0])-total(balances[-1])) if len(balances) >= 2 else None,
              'provider_invoice_cost_cny': None, 'new_download_bytes': 0, 'new_training': False, 'default_changed': False,
              'frozen_sha256': digest(directory/'frozen.json'), 'primary_sequence_seed': 42,
              'scope': 'four ranking hooks inside same actual C1 text/state runtime on new HetRec users'}
    write_new(directory/'report.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'regression', 'calibration', 'freeze', 'offline', 'formal', 'report'))
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args(); torch.set_num_threads(4)
    operations = {'prepare': prepare, 'regression': regression, 'calibration': lambda d: run(d, 'calibration'),
                  'freeze': freeze, 'offline': offline, 'formal': lambda d: run(d, 'formal'), 'report': statistics}
    print(json.dumps(operations[args.phase](args.directory), indent=2), flush=True)


if __name__ == '__main__': main()
