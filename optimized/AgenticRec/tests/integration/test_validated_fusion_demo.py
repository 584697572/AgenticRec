"""Checkpoint profiles and candidate diagnostics must stay opt-in and local."""
import pytest

from agenticrec.adapters.validated import validate_runtime_identity


def test_frozen_runtime_mismatch_is_rejected():
    import platform,pyarrow,torch
    identity={'python_version':platform.python_version(),'pyarrow_version':pyarrow.__version__,
              'torch_version':str(torch.__version__)}
    validate_runtime_identity(identity)
    with pytest.raises(ValueError): validate_runtime_identity({**identity,'pyarrow_version':'different'})


def test_diagnostics_output_cannot_overwrite_a_file_or_escape_artifacts(tmp_path,monkeypatch):
    import agenticrec.fusion_demo as demo
    monkeypatch.setattr(demo,'ROOT',tmp_path)
    permitted=tmp_path/'artifacts/trace.json'
    demo.write_diagnostics(permitted,{'status':'OK'})
    with pytest.raises(FileExistsError): demo.write_diagnostics(permitted,{})
    with pytest.raises(ValueError): demo.write_diagnostics(tmp_path/'private.json',{})


def test_modified_local_result_cannot_replace_public_checkpoint_evidence(tmp_path,monkeypatch):
    import json
    import agenticrec.adapters.validated as validated
    from agenticrec.data import digest
    directory=tmp_path/'trial'; directory.mkdir()
    (directory/'result.json').write_text('{"checkpoint_sha256":"recorded"}')
    public=tmp_path/'reports/rec_baselines/frozen_model_validation_20261011.json'
    public.parent.mkdir(parents=True)
    public.write_text(json.dumps({'status':'VERIFIED','checkpoint_reload_exact':True,
        'runs':{'bpr_42':{'source_result_sha256':digest(directory/'result.json'),
                          'checkpoint_sha256':'recorded'}}}))
    monkeypatch.setattr(validated,'ROOT',tmp_path)
    validated.validate_public_evidence(directory,'bpr',42,{'checkpoint_sha256':'recorded'})
    (directory/'result.json').write_text('{"checkpoint_sha256":"changed"}')
    with pytest.raises(ValueError):
        validated.validate_public_evidence(directory,'bpr',42,{'checkpoint_sha256':'changed'})
