import copy
import importlib.util
from pathlib import Path

import pytest


def auditor(monkeypatch):
    scripts = Path(__file__).resolve().parents[1]/'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('sequence_agent_audit_tests', scripts/'audit_sequence_agent.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def receipts():
    request = {'max_tokens': 1024, 'model': 'deepseek-flash', 'thinking': {'type': 'disabled'},
               'temperature': 0, 'response_format': {'type': 'json_object'}}
    usage = {'prompt_tokens': 12, 'completion_tokens': 3, 'total_tokens': 15}
    start = {'status': 'STARTED', 'request_id': 1, 'request': request}
    end = {'status': 'COMPLETED', 'request_id': 1, 'request': request, 'usage': usage,
           'reply': {'usage': usage}, 'http_status': 200}
    return [start, end]


def test_provider_audit_accepts_actual_usage_receipt(monkeypatch):
    module = auditor(monkeypatch)
    assert len(module.verify_provider(receipts(), 1)) == 1


@pytest.mark.parametrize('fault', ['unknown_usage', 'duplicate_id', 'wrong_cap', 'invented_usage'])
def test_provider_audit_rejects_incomplete_or_invented_usage(monkeypatch, fault):
    module = auditor(monkeypatch); events = copy.deepcopy(receipts())
    if fault == 'unknown_usage': events[-1]['usage'] = None
    elif fault == 'duplicate_id': events[-1]['request_id'] = 2
    elif fault == 'wrong_cap': events[-1]['request']['max_tokens'] = 512
    else: events[-1]['usage'] = dict(events[-1]['usage'], total_tokens=999)
    with pytest.raises(AssertionError): module.verify_provider(events, 1)


def test_independent_state_reset_preserves_public_likes_and_previous_exclusion(monkeypatch):
    from agenticrec.evaluation.frontier_state import empty_state
    module = auditor(monkeypatch); previous = empty_state()
    previous.update(include_genres=['Action'], year_min=1985, liked_item_ids=[1, 2],
                    seen_item_ids=[1, 2], previous_item_ids=[3, 4])
    delta = {'reset_constraints': True, 'include_genres': ['Comedy'], 'exclude_genres': None,
             'year_min': None, 'year_max': None, 'k': 5, 'genre_mode': 'replace',
             'liked_item_ids': [], 'seen_item_ids': [], 'exclude_previous': True}
    state = module.independent_merge(previous, delta)
    assert state['include_genres'] == ['Comedy'] and state['year_min'] is None
    assert state['liked_item_ids'] == [1, 2] and state['excluded_previous_ids'] == [3, 4] and state['k'] == 5


def test_incomplete_execution_cannot_be_audited_as_complete(tmp_path, monkeypatch):
    import json
    module = auditor(monkeypatch); run = tmp_path/'formal_run'; run.mkdir()
    (run/'execution.json').write_text(json.dumps({'status': 'BLOCKED'}), encoding='utf-8')
    with pytest.raises(AssertionError): module.verify_phase(tmp_path, 'formal', {}, {})


@pytest.mark.parametrize('fault', ['invented_hit', 'missing_seed'])
def test_offline_audit_rejects_fabricated_scores_and_missing_seed(tmp_path, monkeypatch, fault):
    import json
    module = auditor(monkeypatch)
    public = [{'episode_id': f'formal-quality-{n:03d}', 'catalog_context': []} for n in range(120)]
    labels = [{'episode_id': r['episode_id'], 'category': 'session_quality', 'labels': [{
        'k': 10, 'genres': [], 'exclude_genres': [], 'year_min': None, 'year_max': None,
        'excluded_raw_ids': [120], 'relevant_raw_ids': [300], 'exclude_previous': False}]} for r in public]
    predictions = [{'system': s, 'seed': seed, 'episode_id': r['episode_id'],
                    'item_ids': list(range(1, 11)), 'raw_ids': list(range(101, 111))}
                   for s in module.SYSTEMS for seed in ((42, 7, 2026) if s == 'sequence' else (None,)) for r in public]
    scores = [dict(r, scores={'hit': 0., 'recall': 0., 'ndcg': 0., 'mrr': 0.}) for r in predictions]
    if fault == 'invented_hit': scores[0]['scores']['hit'] = 1.
    else: predictions.pop(); scores.pop()
    files = {'formal_public.json': public, 'formal_private.json': labels, 'frozen.json': {},
             'offline_predictions.json': predictions, 'offline_scores.json': scores}
    for name, value in files.items(): (tmp_path/name).write_text(json.dumps(value), encoding='utf-8')
    catalog = {i: {'raw_id': 100+i, 'genres': [], 'year': 1990} for i in range(1, 21)}
    truth = {str(v['raw_id']): v for v in catalog.values()}
    with pytest.raises(AssertionError): module.verify_offline(tmp_path, catalog, truth)
