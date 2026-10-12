import json
import pytest

from agenticrec.adapters.account_balance import BalanceClient, BalanceGuard
from agenticrec.adapters.openai_compatible import HttpResponse
from agenticrec.runtime.errors import LLMTransportError


def response(value='10.00', available=True):
    return HttpResponse(200, json.dumps({'is_available': available, 'balance_infos': [
        {'currency': 'CNY', 'total_balance': value, 'granted_balance': '0',
         'topped_up_balance': value}]}).encode())


def test_balance_lookup_uses_get_endpoint_and_never_repr_key():
    calls=[]
    def sender(url, headers, timeout):
        calls.append((url, headers, timeout))
        return response()
    client=BalanceClient('private-fixture-key', sender=sender)
    assert client.fetch()['balance_infos'][0]['total_balance']=='10.00'
    assert calls[0][0]=='https://api.deepseek.com/user/balance'
    assert 'private-fixture-key' not in repr(client)


@pytest.mark.parametrize('value', ['NaN','Infinity','-1','bad'])
def test_balance_rejects_invalid_amounts(value):
    client=BalanceClient('fixture',sender=lambda *_:response(value))
    with pytest.raises(ValueError): client.fetch()


@pytest.mark.parametrize('status,reason', [
    (402, 'ACCOUNT_BALANCE_INSUFFICIENT'),
    (403, 'PROVIDER_CONFIGURATION_ERROR'),
])
def test_terminal_provider_error_latches_stop_and_prevents_later_transport_calls(status,reason):
    client=BalanceClient('fixture',sender=lambda *_:response())
    guard=BalanceGuard(client, refresh_every=100)
    calls=[]
    def failing():
        calls.append(1)
        raise LLMTransportError('fixture',status_code=status)
    with pytest.raises(LLMTransportError): guard.call(failing)
    assert guard.should_stop() and guard.reason==reason
    with pytest.raises(RuntimeError): guard.call(failing)
    assert len(calls)==1


def test_empty_account_does_not_send_generation():
    guard=BalanceGuard(BalanceClient('fixture',sender=lambda *_:response('0',False)))
    calls=[]
    with pytest.raises(RuntimeError): guard.call(lambda:calls.append(1))
    assert calls==[] and guard.should_stop()


def test_runner_stops_before_next_intent_and_resumes_without_duplicate(tmp_path,monkeypatch):
    from agenticrec.evaluation import runner
    from test_benchmark_runner import _spec, _attempt
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    spec=_spec(tmp_path)
    path=tmp_path/'attempts.jsonl'
    calls=[]
    def executor(episode):
        calls.append(episode.episode_id)
        return _attempt(episode.episode_id)
    report=runner.run_benchmark(spec,path,executor,should_stop=lambda:bool(calls))
    assert report['status']=='STOPPED' and report['completed_episodes']==1
    assert len(path.read_text().splitlines())==3
    report=runner.run_benchmark(spec,path,executor)
    assert report['status']=='COMPLETE' and calls==['test-001','test-002']
