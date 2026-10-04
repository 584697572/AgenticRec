import json
import socket
import pytest

from agenticrec.cli import main
from agenticrec.config import LLMBudget, parse_config
from agenticrec.protocols import ToolResult
from agenticrec.testing import AttemptBudget, FakeLLM


def test_defaults_deny_live_calls():
    config = parse_config({})
    with pytest.raises(RuntimeError):
        AttemptBudget(config.llm).reserve()


@pytest.mark.parametrize("data", [
    {"unknown": 1}, {"schema_version": True}, {"domain": "game"},
    {"llm": {"api_key": "fixture-secret"}}, {"llm": {"api_request_cap": True}},
    {"llm": {"money_budget": float("nan")}}, {"llm": {"allow_paid_api": True}},
    {"llm": {"max_output_tokens": -1}}, {"llm": {"provider": []}},
])
def test_invalid_config_fails(data):
    with pytest.raises((ValueError, TypeError)):
        parse_config(data)


def test_failed_attempt_is_still_spent():
    budget = AttemptBudget(LLMBudget(True, "offline-test", "offline-model", 1, 128, 1, "CNY"))
    assert budget.reserve() == 1
    # No refund on errors: next attempt must fail without network access.
    with pytest.raises(RuntimeError):
        budget.reserve()


def test_fake_replay_and_exhaustion():
    model = FakeLLM(["first", "second"])
    messages = [{"role": "user", "content": "fixture"}]
    assert model.chat(messages).content == "first"
    reply = model.chat(messages)
    assert reply.content == "second" and reply.usage is None and reply.remote_api_requests == 0
    with pytest.raises(RuntimeError):
        model.chat(messages)


def test_offline_doctor_never_connects(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("offline doctor attempted network access")
    monkeypatch.setattr(socket, "socket", forbidden)
    assert main(["doctor", "--offline"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["remote_api_requests"] == 0
    assert report["config"]["llm"]["allow_paid_api"] is False


def test_unknown_cli_option_rejected():
    with pytest.raises(SystemExit) as error:
        main(["doctor", "--offline", "--misspelled"])
    assert error.value.code == 2


def test_tool_result_requires_consistent_status():
    assert ToolResult(False, error_code="NO_FEASIBLE_ITEMS").ok is False
    with pytest.raises(ValueError):
        ToolResult(False)
    with pytest.raises(ValueError):
        ToolResult(True, error_code="ERROR")
