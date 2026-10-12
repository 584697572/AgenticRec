import json

from agenticrec.evaluation import benchmark
from test_benchmark import _fixture


def test_money_shortfall_blocks_even_with_sufficient_request_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, 'ROOT', tmp_path)
    path = _fixture(tmp_path)
    ledger_path = tmp_path / 'ledger.json'
    ledger = json.loads(ledger_path.read_text())
    ledger['budget'].update(api_request_cap=1000, money_budget=1, allow_paid_api=True)
    ledger['estimated_cost_cny_peak_ceiling'] = 0
    ledger_path.write_text(json.dumps(ledger))
    plan = benchmark.build_dry_run(path)
    assert plan['authorization']['request_shortfall'] == 0
    assert plan['status'] == 'BLOCKED_AUTHORIZATION'
    assert plan['live_run_authorized'] is False
    assert plan['authorization']['money_shortfall_cny'] > 0


def test_engineering_only_profile_never_authorizes_paid_batch(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, 'ROOT', tmp_path)
    path = _fixture(tmp_path)
    ledger_path = tmp_path / 'ledger.json'
    ledger = json.loads(ledger_path.read_text())
    ledger['budget'].update(api_request_cap=1000, money_budget=100, allow_paid_api=False)
    ledger['estimated_cost_cny_peak_ceiling'] = 0
    ledger_path.write_text(json.dumps(ledger))
    plan = benchmark.build_dry_run(path)
    assert plan['live_run_authorized'] is False
    assert plan['authorization']['paid_calls_allowed'] is False
