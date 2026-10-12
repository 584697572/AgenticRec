from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from agenticrec.adapters.llm import TransportReply, TokenUsage
from agenticrec.evaluation.episodes import EpisodeAttempt
from agenticrec.evaluation.live_benchmark import AuditedTransport
from agenticrec.evaluation.system_reports import verify_provider_evidence
from agenticrec.runtime.errors import LLMTransportError


class Transport:
    def __init__(self, fixture=False):
        self.calls = 0
        self.fixture = fixture

    def chat(self, messages, **options):
        self.calls += 1
        return TransportReply('{}', TokenUsage(12, 1, 13), None,
                              None, 'provider_bill_not_queried', is_fixture=self.fixture)


def attempt(**kwargs):
    return EpisodeAttempt('test-001', 'OK', (1,), 1, 1, 10, 12, 1, 1, **kwargs)


def test_audit_reconciles_actual_attempt_usage_and_rejects_fake_delivery(tmp_path):
    path = tmp_path / 'llm_attempts.jsonl'
    audit = AuditedTransport(Transport(), path)
    audit.current_episode = 'test-001'
    audit.chat([{'role': 'user', 'content': 'fixture'}])
    assert verify_provider_evidence(path, [attempt()])['requests'] == 1
    with pytest.raises(ValueError, match='request count'):
        verify_provider_evidence(path, [replace(attempt(), request_count=2)])
    with pytest.raises(ValueError, match='token usage'):
        verify_provider_evidence(path, [replace(attempt(), input_tokens=99)])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1]['reply']['is_fixture'] = True
    path.write_text('\n'.join(json.dumps(row) for row in rows) + '\n')
    with pytest.raises(ValueError, match='fixture'):
        verify_provider_evidence(path, [attempt()])


def test_provider_failure_remains_in_denominator_with_unknown_usage(tmp_path):
    class Failed:
        def chat(self, *_args, **_kwargs):
            raise LLMTransportError('fixture', error_type='timeout', retryable=False)
    path = tmp_path / 'llm_attempts.jsonl'
    audit = AuditedTransport(Failed(), path)
    audit.current_episode = 'test-001'
    with pytest.raises(LLMTransportError):
        audit.chat([{'role': 'user', 'content': 'fixture'}])
    row = replace(attempt(), status='LLM_FAILED', item_ids=(), input_tokens=None, output_tokens=None)
    assert verify_provider_evidence(path, [row])['failed_requests'] == 1
    path.write_text(path.read_text().splitlines()[0] + '\n')
    with pytest.raises(ValueError, match='unresolved'):
        verify_provider_evidence(path, [row])


def test_fault_counts_include_u1_fallback_in_addition_to_original_tools():
    from agenticrec.evaluation.faults import FaultBoundExecutor, FaultSchedule
    pipeline = SimpleNamespace(invocation_count=0, fault_count=0)
    from contextlib import contextmanager
    @contextmanager
    def activate(_flag):
        yield
    pipeline.activate = activate
    class Executor:
        feedback_sha256 = None
        last_trace = {}
        def __call__(self, _episode):
            pipeline.invocation_count += 2
            pipeline.fault_count += 1
            return replace(attempt(), tool_calls=3)
    bound = FaultBoundExecutor(Executor(), FaultSchedule({'test-001': 'ranking_timeout'}), pipeline)
    result = bound(SimpleNamespace(episode_id='test-001'))
    assert result.tool_calls == 4
    assert bound.executor.last_trace['fault_execution']['provider_fault_traffic'] == 0
