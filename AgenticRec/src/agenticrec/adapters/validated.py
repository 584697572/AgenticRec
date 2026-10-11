"""Opt-in loading of the frozen 2026-10-10 search profile, without test labels."""
import json
import platform

import pyarrow
import torch

from .model import KnownUserScorer
from ..data import digest
from ..frozen_model_validation import load_search_model, trial_directory
from ..training import ROOT, load_train_contract


def validate_runtime_identity(identity):
    expected = {'python_version': platform.python_version(),
                'pyarrow_version': pyarrow.__version__, 'torch_version': str(torch.__version__)}
    if any(identity.get(key) != value for key, value in expected.items()):
        raise ValueError('Frozen profile requires the exact recorded Python/PyArrow/Torch runtime')


def validate_public_evidence(directory,kind,seed,result):
    report=json.loads((ROOT/'reports/rec_baselines/frozen_model_validation_20261011.json')
                      .read_text(encoding='utf-8'))
    if report['status']!='VERIFIED' or report['checkpoint_reload_exact'] is not True:
        raise ValueError('Public validation has not completed')
    evidence=report['runs'][f'{kind}_{seed}']
    if (digest(directory/'result.json')!=evidence['source_result_sha256']
            or result['checkpoint_sha256']!=evidence['checkpoint_sha256']):
        raise ValueError('Local checkpoint metadata differs from public validation evidence')


def load_validated_scorer(kind, seed=42):
    if kind not in ('bpr', 'lightgcn') or type(seed) is not int or seed not in (7, 42, 2026):
        raise ValueError('Validated profile supports bpr/lightgcn and seeds 7/42/2026')
    directory = trial_directory(kind, seed)
    result = json.loads((directory/'result.json').read_text(encoding='utf-8'))
    validate_public_evidence(directory,kind,seed,result)
    validate_runtime_identity(result['identity'])
    config = json.loads((ROOT/'AgenticRec/configs/recommendation_search_selected_20261010.json')
                        .read_text(encoding='utf-8'))
    contract = load_train_contract()
    model, _result = load_search_model(directory, config['selected'][kind], seed, contract)
    _manifest, n_users, n_items, seen, _edges = contract
    return KnownUserScorer(model, kind, n_users, n_items, seen)
