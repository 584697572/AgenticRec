"""Model identity fails before deserialization; statistics preserve user units."""
import json

import pytest

from agenticrec.frozen_model_validation import load_search_model, average_rows, clustered_pair


def test_modified_checkpoint_is_rejected_before_loading(tmp_path,monkeypatch):
    trial={'model':'bpr'}
    (tmp_path/'result.json').write_text(json.dumps({'status':'VERIFIED','trial':trial,'seed':7,'checkpoint_sha256':'0'*64}))
    (tmp_path/'best.pt').write_bytes(b'modified checkpoint')
    import torch
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('Unverified checkpoint was loaded'))
    with pytest.raises(ValueError,match='SHA'):
        load_search_model(tmp_path,trial,7,({},1,1,{},[]))


def test_seed_averaging_preserves_users_and_rejects_missing_runs():
    def row(v): return {f:v for f in ('recall','hit','mrr','ndcg')}
    result=average_rows([{'1':row(0),'2':row(1)},{'1':row(1),'2':row(0)}])
    assert result=={'1':row(.5),'2':row(.5)}
    with pytest.raises(ValueError): average_rows([{'1':row(0)},{'2':row(0)}])


def test_user_cluster_bootstrap_has_expected_constant_difference():
    result=clustered_pair({'1':{'ndcg':.75},'2':{'ndcg':.5}},
                          {'1':{'ndcg':.5},'2':{'ndcg':.25}},'ndcg',samples=100)
    assert result['difference']==.25 and result['ci95']==[.25,.25]
    assert result['users']==2
