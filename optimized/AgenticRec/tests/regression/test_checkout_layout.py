"""A fresh two-part checkout must preserve paths without clobbering user files."""
import importlib.util
import os
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location('workspace_layout', Path(__file__).resolve().parents[4]/'workspace.py')
layout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(layout)


def skeleton(root):
    (root/'original/reproduction/scripts').mkdir(parents=True)
    (root/'optimized/AgenticRec/src').mkdir(parents=True)
    (root/'original/reports').mkdir()
    (root/'optimized/reports').mkdir()
    (root/'original/reports/original.md').write_text('original evidence', encoding='utf-8')
    (root/'optimized/reports/optimized.md').write_text('optimized evidence', encoding='utf-8')


def test_fresh_prepare_is_idempotent_and_preserves_both_parts(tmp_path):
    skeleton(tmp_path)
    first = layout.prepare(tmp_path)
    second = layout.prepare(tmp_path)
    assert first['downloads'] == second['remote_api_requests'] == 0
    assert os.path.samefile(tmp_path/'AgenticRec', tmp_path/'optimized/AgenticRec')
    assert os.path.samefile(tmp_path/'optimized/reproduction', tmp_path/'original/reproduction')
    assert (tmp_path/'original/reports/original.md').read_text() == 'original evidence'
    assert (tmp_path/'optimized/reports/optimized.md').read_text() == 'optimized evidence'
    assert all(row['status'] != 'created' for row in second['compatibility_links'])


def test_existing_user_directory_is_never_replaced(tmp_path):
    skeleton(tmp_path)
    (tmp_path/'AgenticRec').mkdir()
    marker = tmp_path/'AgenticRec/my-file.txt'
    marker.write_text('keep', encoding='utf-8')
    with pytest.raises(ValueError, match='Refusing to replace'):
        layout.prepare(tmp_path)
    assert marker.read_text() == 'keep'


def test_link_target_must_stay_within_workspace(tmp_path):
    root = tmp_path/'checkout'
    root.mkdir()
    outside = tmp_path/'outside.txt'
    outside.write_text('keep', encoding='utf-8')
    with pytest.raises(ValueError, match='outside checkout'):
        layout.link(root, 'alias.txt', '../outside.txt')
    assert not (root/'alias.txt').exists()


def test_both_credential_locators_use_root_env_without_reading_it():
    from agenticrec.runtime.secrets import WORKSPACE_ENV_FILE
    root=Path(layout.__file__).parent
    spec=importlib.util.spec_from_file_location('layout_legacy_keys',root/'original/reproduction/scripts/local_api_key.py')
    legacy=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    assert WORKSPACE_ENV_FILE==legacy.WORKSPACE_ENV_FILE==root/'.env'
    assert not (root/'original/.env').exists()
    assert not (root/'optimized/.env').exists()
