"""Read-only raw-cohort, actual-delivery and provider audit; makes no API calls."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
import io
import json
import math
from pathlib import Path
import random
import zipfile

import torch

from audit_three_way_comparison import assess, digest

SYSTEMS = ('sequence', 'mixed', 'itemcf', 'popular')
METRICS = ('hit', 'recall', 'ndcg', 'mrr')


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def lines(path): return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines()]


def close(a, b):
    assert math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, abs_tol=1e-12, rel_tol=0), (a, b)


def paired(delta, seed):
    n = len(delta); rng = random.Random(seed)
    samples = sorted(sum(delta[int(rng.random()*n)] for _ in range(n))/n for _ in range(10000))
    intervals = []
    for q in (.025, .975):
        position = (len(samples)-1)*q; lo = int(position)
        intervals.append(samples[lo]+(samples[lo+1]-samples[lo])*(position-lo))
    return {'mean': sum(delta)/n, 'ci95': intervals}


def check_cohort(directory, source, archive):
    catalog = read(directory/'catalog.json')
    assert catalog == read(source/'catalog.json') and len(catalog) == 7541
    raw_to_model = {r['raw_id']: int(i) for i, r in catalog.items()}
    excluded = set(read(source/'training_users_private.json'))
    for phase in ('development', 'test'):
        excluded.update(r['evaluator_raw_user_id'] for r in read(source/(phase+'_private.json')))
    assert sorted(excluded) == read(directory/'excluded_users_private.json')
    cutoff = read(source/'prepared.json')['cutoffs_ms'][2]
    with zipfile.ZipFile(archive) as z:
        ratings = sorted((int(r['timestamp']), int(r['userID']), int(r['movieID']), float(r['rating']))
                         for r in csv.DictReader(io.StringIO(z.read('user_ratedmovies-timestamps.dat').decode('latin-1')), delimiter='\t'))
    assert len(ratings) == 855598
    histories, seen, targets = defaultdict(list), defaultdict(set), defaultdict(set)
    for ts, user, raw, rating in ratings:
        if ts < cutoff:
            seen[user].add(raw)
            if rating >= 4 and raw in raw_to_model:
                histories[user].append(raw)
    for ts, user, raw, rating in ratings:
        if ts >= cutoff and rating >= 4 and raw in raw_to_model and raw not in seen[user]:
            targets[user].add(raw)
    users = sorted(u for u in targets if len(histories[u]) >= 10 and u not in excluded)
    assert len(users) == 164
    random.Random(2026101313).shuffle(users)
    selected = {'calibration': users[:9], 'formal': users[9:159]}
    used = {}
    for phase, n, quality_n, turns in [('calibration', 12, 3, 20), ('formal', 182, 120, 222)]:
        private = read(directory/(phase+'_private.json')); public = read(directory/(phase+'_public.json'))
        assert len(private) == len(public) == n and sum(len(r['labels']) for r in private) == turns
        assert [r['episode_id'] for r in private] == [r['episode_id'] for r in public]
        evaluator_users = [r['evaluator_raw_user_id'] for r in private if 'evaluator_raw_user_id' in r]
        assert evaluator_users == selected[phase]
        assert len(set(evaluator_users)) == len(evaluator_users) and not set(evaluator_users)&excluded
        used[phase] = set(evaluator_users)
        qrows = [r for r in private if r['category'] == 'session_quality']
        assert len(qrows) == quality_n and Counter(r['history_n'] for r in qrows) == {3: quality_n//3, 5: quality_n//3, 10: quality_n//3}
        for p, q in zip(public, private):
            if 'evaluator_raw_user_id' in q:
                user = q['evaluator_raw_user_id']; h = q['history_n']
                expected_raw = histories[user][-h:]
                expected_context = [{k: catalog[str(raw_to_model[raw])][k] for k in ('title', 'year', 'item_id')} for raw in expected_raw]
                assert p['catalog_context'] == expected_context
                assert all(r['title'] in p['messages'][0] for r in expected_context)
                for lab in q['labels']:
                    assert lab['excluded_raw_ids'] == expected_raw
                if q['category'] == 'session_quality':
                    assert set(q['labels'][0]['relevant_raw_ids']) == targets[user]
                    assert not set(expected_raw)&targets[user]
    assert not used['calibration']&used['formal']
    return {'unused_eligible_users': 164, 'formal_unique_users': 150, 'calibration_unique_users': 9,
            'formal_quality_users': 120, 'source_identity_and_temporal_cohort_verified': True}


def independent_merge(previous, delta):
    state = deepcopy(previous)
    if delta['reset_constraints']:
        state['include_genres'] = []; state['exclude_genres'] = []
        state['year_min'] = state['year_max'] = None
    for name in ('include_genres', 'exclude_genres', 'year_min', 'year_max', 'k'):
        if delta[name] is not None:
            state[name] = list(dict.fromkeys(state[name]+delta[name])) if name == 'include_genres' and delta['genre_mode'] == 'add' else deepcopy(delta[name])
    for name in ('liked_item_ids', 'seen_item_ids'):
        state[name] = list(dict.fromkeys(state[name]+delta[name]))
    if delta['exclude_previous']:
        state['excluded_previous_ids'] = list(dict.fromkeys(state['excluded_previous_ids']+state['previous_item_ids']))
    return state


def verify_provider(events, expected_requests):
    started = [r for r in events if r['status'] == 'STARTED']
    complete = [r for r in events if r['status'] == 'COMPLETED']
    assert len(events) == 2*expected_requests and len(started) == len(complete) == expected_requests
    assert sorted(r['request_id'] for r in started) == list(range(1, expected_requests+1))
    assert sorted(r['request_id'] for r in complete) == list(range(1, expected_requests+1))
    starts = {r['request_id']: r for r in started}
    for r in complete:
        request, usage = r['request'], r['usage']
        assert r['http_status'] == 200 and usage is not None
        assert request == starts[r['request_id']]['request']
        assert request['max_tokens'] == 1024 and request['model'] == 'deepseek-flash'
        assert request['thinking'] == {'type': 'disabled'} and request['temperature'] == 0
        assert request['response_format'] == {'type': 'json_object'}
        assert usage == r['reply']['usage']
        assert all(type(usage[k]) is int and usage[k] >= 0 for k in ('prompt_tokens', 'completion_tokens', 'total_tokens'))
        assert usage['completion_tokens'] <= 1024 and usage['total_tokens'] == usage['prompt_tokens']+usage['completion_tokens']
    return complete


def verify_offline(directory, catalog, truth):
    public = {r['episode_id']: r for r in read(directory/'formal_public.json') if '-quality-' in r['episode_id']}
    private = {r['episode_id']: r for r in read(directory/'formal_private.json') if r['category'] == 'session_quality'}
    expected = {(s, seed, e) for s in SYSTEMS for seed in ((42, 7, 2026) if s == 'sequence' else (None,)) for e in public}
    predictions, scores = read(directory/'offline_predictions.json'), read(directory/'offline_scores.json')
    saved = {(r['system'], r['seed'], r['episode_id']): r for r in predictions}
    assert len(predictions) == len(scores) == len(saved) == 720 and set(saved) == expected
    used, totals = set(), defaultdict(Counter)
    for r in scores:
        key = (r['system'], r['seed'], r['episode_id'])
        assert key in expected and key not in used; used.add(key)
        assert {k: v for k, v in r.items() if k != 'scores'} == saved[key]
        assert r['raw_ids'] == [catalog[i]['raw_id'] for i in r['item_ids']]
        values, errors = assess({'raw_ids': r['raw_ids'], 'status': 'OK', 'extraction_verified': True},
                               private[r['episode_id']]['labels'][0], truth, [])
        assert values['strict_success'] and not any(errors.values())
        for m in METRICS:
            close(values[m], r['scores'][m]); totals[(r['system'], r['seed'])][m] += values[m]
        totals[(r['system'], r['seed'])]['n'] += 1
    assert used == expected and all(v['n'] == 120 for v in totals.values())
    assert (directory/'offline_predictions.json').stat().st_mtime_ns >= (directory/'frozen.json').stat().st_mtime_ns
    assert (directory/'offline_scores.json').stat().st_mtime_ns >= (directory/'offline_predictions.json').stat().st_mtime_ns
    return {s+':'+str(seed): {'n': v['n'], 'hits': int(v['hit']),
                             'metrics': {m: v[m]/v['n'] for m in METRICS}}
            for (s, seed), v in totals.items()}, saved


def verify_phase(directory, phase, catalog, truth, rankers=None):
    from agenticrec.evaluation.frontier_state import empty_state, validate_delta, feasible_ids
    run = directory/(phase+'_run'); execution = read(run/'execution.json')
    assert execution['status'] == 'COMPLETE'
    private = {r['episode_id']: r for r in read(directory/(phase+'_private.json'))}
    public = {r['episode_id']: r for r in read(directory/(phase+'_public.json'))}
    records = lines(run/'episodes.jsonl')
    expected = {(s, p) for s in SYSTEMS for p in private}
    actual = {(r['system'], r['episode_id']) for r in records}
    assert actual == expected and len(records) == len(expected) == execution['system_tasks_completed']
    receipts = verify_provider(lines(run/'provider.jsonl'), execution['remote_requests'])
    by_turn = defaultdict(list)
    for r in receipts:
        by_turn[(r['system'], r['episode_id'], r['turn'])].append(r)
    accounted, turn_count, replayed, errors = set(), 0, 0, 0
    for record in records:
        p, q = public[record['episode_id']], private[record['episode_id']]
        assert record['category'] == q['category'] and record['history_n'] == q['history_n']
        assert len(record['turns']) == len(q['labels'])
        previous_raw, state, history = [], empty_state(), []
        for index, (t, lab) in enumerate(zip(record['turns'], q['labels']), 1):
            turn_count += 1
            used = by_turn[(record['system'], record['episode_id'], index)]
            assert 1 <= len(used) == t['requests'] <= 3
            for r in used:
                assert r['request_id'] not in accounted; accounted.add(r['request_id'])
                envelope = json.loads(r['request']['messages'][1]['content'])
                assert envelope['message'] == p['messages'][index-1] and envelope['catalog_items'] == p['catalog_context']
                assert envelope['previous_state'] == state and envelope['conversation'] == history[-4:]
                assert not set(envelope)&{'user_id', 'evaluator_raw_user_id', 'relevant_raw_ids', 'relevant_item_ids'}
            assert t['input_tokens'] == sum(r['usage']['prompt_tokens'] for r in used)
            assert t['output_tokens'] == sum(r['usage']['completion_tokens'] for r in used)
            assert math.isfinite(t['elapsed_seconds']) and t['elapsed_seconds'] >= sum(r['elapsed_seconds'] for r in used)-.01
            if 'delta' in t:
                delta = validate_delta(json.loads(used[-1]['reply']['choices'][0]['message']['content']), p['messages'][index-1], p['catalog_context'])
                assert delta == t['delta']
                state = independent_merge(state, delta)
                assert len(feasible_ids(catalog, state)) == t['eligible_count']
                if rankers:
                    assert rankers[record['system']].rank(state) == t['mapped_optimized_ids']
                    replayed += 1
                assert t['raw_ids'] == [catalog[i]['raw_id'] for i in t['mapped_optimized_ids']]
                expected_answer = '\n'.join(f"{n}. {catalog[i]['title']} ({catalog[i]['year']})" for n, i in enumerate(t['mapped_optimized_ids'], 1))
                assert t['answer'] == expected_answer and t['extraction_verified']
                state['previous_item_ids'] = t['mapped_optimized_ids']
                assert t['state'] == state
                history.extend([{'role': 'user', 'content': p['messages'][index-1]}, {'role': 'assistant', 'content': expected_answer}])
            else:
                assert t['status'] == 'ERROR' and t['raw_ids'] == [] and not t['extraction_verified']
                errors += 1
            expected_metrics, violations = assess({'raw_ids': t['raw_ids'], 'status': t['status'],
                'extraction_verified': t['extraction_verified']}, lab, truth, previous_raw)
            for m in METRICS: close(t['scores'][m], expected_metrics[m])
            if lab['feasible']:
                assert t['scores']['strict_success'] == expected_metrics['strict_success']
            else:
                # All predefined refusals have no candidates; extraction errors
                # never count as a correct structured refusal.
                correct = not t['raw_ids'] and t['status'] == 'CONSTRAINT_FAILURE' and t.get('eligible_count') == 0 and 'delta' in t
                assert t['scores']['correct_refusal'] == correct and t['scores']['strict_success'] == correct
            previous_raw = t['raw_ids']
        assert record['strict_success'] == all(t['scores']['strict_success'] for t in record['turns'])
    assert len(accounted) == len(receipts) and turn_count == execution['expected_turns']
    return records, {'system_tasks': len(records), 'turns': turn_count, 'http200_known_usage': len(receipts),
                     'actual_ranking_replays': replayed, 'preserved_parse_errors': errors}


def audit(directory, output, calibration_only=False):
    from agenticrec.evaluation.sequence_agent import SOURCE, ARCHIVE, assets, PublicRanker
    root = Path(__file__).resolve().parents[3]
    torch.set_num_threads(4)
    catalog = {int(k): v for k, v in read(directory/'catalog.json').items()}
    truth = read(directory/'truth_private.json')
    assert truth == {str(r['raw_id']): r for r in catalog.values()}
    env, model = assets()
    rankers = {s: PublicRanker(s, env, model if s == 'sequence' else None) for s in SYSTEMS}
    cal = read(directory/'calibration_run/contract.json')
    for name, expected in cal['sources'].items(): assert digest(root/name) == expected
    for name, expected in cal['inputs'].items(): assert digest(directory/name) == expected
    if calibration_only:
        records, details = verify_phase(directory, 'calibration', catalog, truth, rankers)
        result = {'status': 'VERIFIED', 'phase': 'calibration', **details,
                  'strict_passed': {s: sum(r['strict_success'] for r in records if r['system'] == s) for s in SYSTEMS},
                  'no_failed_case_replay': True, 'api_requests_added_by_audit': 0, 'auditor_sha256': digest(Path(__file__))}
    else:
        frozen = read(directory/'frozen.json')
        for group, base in [('inputs', directory), ('sources', root), ('resources', root)]:
            for name, expected in frozen[group].items(): assert digest(base/name) == expected, name
        cohort = check_cohort(directory, SOURCE, ARCHIVE)
        offline, direct_predictions = verify_offline(directory, catalog, truth)
        records, details = verify_phase(directory, 'formal', catalog, truth, rankers)
        report = read(directory/'report.json')
        by = {s: {r['episode_id']: r for r in records if r['system'] == s} for s in SYSTEMS}
        for s, users in by.items():
            summary = report['systems'][s]
            quality = [r for r in users.values() if r['category'] == 'session_quality']
            feasible = [r for r in users.values() if r['category'] != 'unanswerable']
            turns = [t for r in users.values() for t in r['turns']]
            assert summary['quality_n'] == len(quality) == 120 and summary['strict_n'] == len(feasible) == 174
            assert summary['strict_passed'] == sum(r['strict_success'] for r in feasible)
            close(summary['strict_success'], summary['strict_passed']/174)
            assert summary['quality_hits'] == sum(r['turns'][0]['scores']['hit'] for r in quality)
            for m in METRICS: close(summary['quality'][m], sum(r['turns'][0]['scores'][m] for r in quality)/120)
            for key in ('requests', 'input_tokens', 'output_tokens', 'tool_calls'):
                assert summary[key] == sum(t[key] for t in turns)
            assert summary['semantic_repair_requests'] == sum(t['requests']-1 for t in turns)
        assert set(report['comparisons']) == {'sequence_minus_'+b for b in ('mixed', 'itemcf', 'popular')}
        quality_keys = sorted(k for k, r in by['sequence'].items() if r['category'] == 'session_quality')
        for n, b in enumerate(('mixed', 'itemcf', 'popular')):
            comp = report['comparisons']['sequence_minus_'+b]
            assert comp['quality_paired_users'] == 120
            for mn, m in enumerate(METRICS):
                delta = [by['sequence'][k]['turns'][0]['scores'][m]-by[b][k]['turns'][0]['scores'][m] for k in quality_keys]
                expected = paired(delta, 2026101316+n*10+mn)
                close(comp['quality'][m]['mean'], expected['mean'])
                for x, y in zip(comp['quality'][m]['ci95'], expected['ci95']): close(x, y)
            delta = [by['sequence'][k]['turns'][0]['scores']['hit']-by[b][k]['turns'][0]['scores']['hit'] for k in quality_keys]
            observed = abs(sum(delta)); rng = random.Random(2026101320+n)
            extreme = sum(abs(sum(d*((-1, 1)[rng.randrange(2)]) for d in delta)) >= observed-1e-12 for _ in range(10000))
            close(comp['hit_permutation_p'], (extreme+1)/10001)
        running = 0.
        for n, k in enumerate(sorted(report['comparisons'], key=lambda k: report['comparisons'][k]['hit_permutation_p'])):
            running = max(running, min(1., (3-n)*report['comparisons'][k]['hit_permutation_p']))
            close(report['comparisons'][k]['hit_holm_p'], running)
        keys = sorted(k for k, r in by['sequence'].items() if r['category'] != 'unanswerable')
        expected = paired([int(by['sequence'][k]['strict_success'])-int(by['mixed'][k]['strict_success']) for k in keys], 2026101324)
        assert report['strict_sequence_minus_mixed'] == expected
        assert report['strict_noninferiority_established'] == (expected['ci95'][0] >= -.02)
        comp = report['comparisons']['sequence_minus_mixed']
        assert report['verified_quality_gain_over_mixed'] == (comp['quality']['hit']['ci95'][0] > 0 and comp['hit_holm_p'] < .05)
        assert report['formal_requests'] == details['http200_known_usage']
        assert report['default_changed'] is False and report['new_training'] is False and report['new_download_bytes'] == 0
        result = {'status': 'VERIFIED', **cohort, **details, 'frozen_hashes_verified': True,
                  'offline_independent_scores': 720, 'offline_quality': offline,
                  'live_quality_matches_direct_ranking': {s: sum(
                      r['turns'][0]['raw_ids'] == direct_predictions[(s, 42 if s == 'sequence' else None, r['episode_id'])]['raw_ids']
                      for r in by[s].values() if r['category'] == 'session_quality') for s in SYSTEMS},
                  'independent_metrics_and_statistics_verified': True, 'provider_usage_reconciled': True,
                  'auditor_sha256': digest(Path(__file__)), 'scoring_helper_sha256': digest(Path(__file__).with_name('audit_three_way_comparison.py')),
                  'report_sha256': digest(directory/'report.json'), 'api_requests_added_by_audit': 0}
    with output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2); f.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--calibration-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(audit(args.directory, args.output, args.calibration_only), indent=2))
