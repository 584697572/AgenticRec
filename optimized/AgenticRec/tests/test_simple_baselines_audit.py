import importlib.util
from pathlib import Path
import sys

import pytest


def auditor():
    scripts = Path(__file__).resolve().parents[1]/'scripts'
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location('simple_baselines_auditor', scripts/'audit_simple_baselines.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def test_independent_audit_rejects_incomplete_output():
    with pytest.raises(AssertionError): auditor().verify_scores([], [], {}, {})


def test_independent_audit_rejects_fabricated_metrics():
    row = {'system': 'popular', 'episode_id': 'e', 'history_n': 3,
           'raw_ids': list(range(1, 11)), 'metrics': {'hit': 1., 'recall': 1., 'ndcg': 1., 'mrr': 1.}}
    prediction = {k: v for k, v in row.items() if k != 'metrics'}
    label = {'k': 10, 'genres': [], 'exclude_genres': [], 'year_min': None, 'year_max': None,
             'excluded_raw_ids': [], 'relevant_raw_ids': [11], 'exclude_previous': False, 'feasible': True}
    labels = {'e': {'category': 'session_quality', 'history_n': 3, 'labels': [label]}}
    truth = {str(i): {'genres': [], 'year': 2000} for i in range(1, 12)}
    # Cardinality passes; independent scoring must catch the invented hit first.
    predictions = [dict(prediction, episode_id='e'+str(i)) for i in range(900)]
    records = [dict(row, episode_id='e'+str(i)) for i in range(900)]
    labels = {'e'+str(i): labels['e'] for i in range(900)}
    with pytest.raises(AssertionError): auditor().verify_scores(records, predictions, labels, truth)
