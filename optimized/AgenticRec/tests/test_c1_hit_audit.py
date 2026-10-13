"""The independent auditor must refuse changed evidence and partial completion."""
import importlib.util
import json
from pathlib import Path

import pytest


def auditor(monkeypatch):
    scripts = Path(__file__).resolve().parents[1]/'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('c1_hit_auditor_tests',scripts/'audit_c1_hit.py')
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


def test_audit_refuses_changed_frozen_evidence(tmp_path,monkeypatch):
    module = auditor(monkeypatch)
    (tmp_path/'frozen.json').write_text('changed',encoding='utf-8')
    write(tmp_path/'protocol.json',{'files':{'frozen.json':'0'*64},'sources':{},'resources':{}})
    with pytest.raises(AssertionError):module.audit(tmp_path,tmp_path/'result.json')
    assert not (tmp_path/'result.json').exists()


def test_audit_refuses_blocked_execution(tmp_path,monkeypatch):
    module = auditor(monkeypatch)
    write(tmp_path/'protocol.json',{'files':{},'sources':{},'resources':{}})
    write(tmp_path/'live_run/execution.json',{'status':'BLOCKED'})
    with pytest.raises(AssertionError):module.audit(tmp_path,tmp_path/'result.json')
    assert not (tmp_path/'result.json').exists()


def test_audit_refuses_complete_label_with_missing_predictions(tmp_path,monkeypatch):
    module = auditor(monkeypatch)
    write(tmp_path/'protocol.json',{'files':{},'sources':{},'resources':{}})
    write(tmp_path/'live_run/execution.json',{'status':'COMPLETE'})
    write(tmp_path/'formal_private.json',[])
    (tmp_path/'live_run/episodes.jsonl').write_text('',encoding='utf-8')
    with pytest.raises(AssertionError):module.audit(tmp_path,tmp_path/'result.json')
    assert not (tmp_path/'result.json').exists()
