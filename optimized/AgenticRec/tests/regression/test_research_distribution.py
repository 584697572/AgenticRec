"""Git publication must preserve the byte identity of measured research evidence."""
import subprocess

import pytest

from agenticrec.training import ROOT


@pytest.mark.parametrize('path',[
    'reports/fusion_research_20261011.json',
    'reports/rec_baselines/frozen_model_validation_20261011.json',
    'AgenticRec/configs/frozen_model_validation_20261011_seed_7.json',
])
def test_research_evidence_is_not_rewritten_by_git_line_filters(path):
    value=b'{"measured":true}\r\n'
    filtered=subprocess.check_output(['git','hash-object','--stdin','--path',path],
                                     input=value,cwd=ROOT)
    raw=subprocess.check_output(['git','hash-object','--stdin','--no-filters'],
                                input=value,cwd=ROOT)
    assert filtered==raw, 'Publication would change recorded evidence bytes'
