"""Independent diagnostic oracles; fixture numbers are not experiment results."""
import json
import math
import random

import pytest

from agenticrec.training_signal_diagnostics import (
    pair_summary, choose_negatives, genre_distance, temporal_row,
    sample_users, aggregate_rows, train_only_files, verified_model,
    validate_plan, write_once,
)


def test_margin_loss_and_gradient_use_independent_formula():
    row = pair_summary([2.0, -1.0], [0.0, 1.0], 0.05)
    assert row['loss'] == pytest.approx((math.log1p(math.exp(-2))+math.log1p(math.exp(2)))/2)
    assert row['gradient'] == pytest.approx(0.5)
    assert row['margin'] == 0
    assert row['negative_outranks_positive'] == 0.5
    assert row['easy_fraction'] == 0
    assert pair_summary([1000.], [-1000.], .05)['loss'] == 0
    assert pair_summary([-1000.], [1000.], .05)['loss'] == 2000
    assert pair_summary([], [], .05) is None
    with pytest.raises(ValueError):
        pair_summary([float('nan')], [0.], .05)


def test_negative_choices_are_deterministic_and_pool_first_is_random_control():
    assert choose_negatives([3, 1, 2, 4], [0., 2., 2., -1.]) == {
        'uniform': 3, 'median_pool': 3, 'hardest_pool': 2}
    assert choose_negatives([5], [2.]) == dict.fromkeys(['uniform','median_pool','hardest_pool'], 5)


def test_sampling_never_uses_train_rated_low_or_neutral_as_unknown():
    seen = {1: {1,2,3,4}, 2: {2,3}}
    positive = {1: [1,2], 2: [2]}
    low = {1: [3], 2: []}
    a = sample_users(6, seen, positive, low, 2, 3, 2, 42)
    assert a == sample_users(6, seen, positive, low, 2, 3, 2, 42)
    for user in a:
        for pool in user['pools']:
            assert set(pool).isdisjoint(seen[user['user_id']])
            assert len(pool) == len(set(pool))
        assert set(user['positive_ids']) <= set(positive[user['user_id']])
    assert a[0]['low_ids'] == [3]
    with pytest.raises(ValueError):
        sample_users(2, {1:{1,2}}, {1:[1]}, {}, 1, 1, 1, 42)


def test_genre_total_variation_has_known_extremes_and_equal_item_mass():
    genres={1:['A'],2:['B'],3:['A','B']}
    assert genre_distance([1],[2],genres) == 1
    assert genre_distance([1,2],[3],genres) == 0
    assert genre_distance([1],[1],genres) == 0


def test_genre_distance_is_byte_stable_across_python_hash_seeds():
    import os
    import subprocess
    import sys
    code = """from agenticrec.training_signal_diagnostics import genre_distance
import json
genres={i:['g'+str((i+j)%19) for j in range(1+i%5)] for i in range(1,61)}
print(json.dumps([genre_distance(list(range(1,n)),list(range(n,61)),genres) for n in range(10,40)]))
"""
    outputs=[subprocess.check_output([sys.executable,'-c',code],
              env={**os.environ,'PYTHONHASHSEED':str(seed)}) for seed in (1,17)]
    assert outputs[0] == outputs[1]


def test_temporal_ties_are_kept_whole_without_item_id_pseudo_sequence():
    records=[(i, 4, 100) for i in range(1,21)]
    row=temporal_row(1,records,{i:['A'] for i in range(1,21)},20,5,42)
    assert row['split_status'] == 'TIMESTAMP_TIES_NOT_SEPARABLE'
    assert row['span_days'] == 0 and row['largest_day_fraction'] == 1
    assert row['genre_tv'] is None


def test_temporal_genre_change_is_not_confounded_by_same_day_ties():
    records=[(i,4,i*86400) for i in range(1,21)]
    genres={i:['A'] if i<=10 else ['B'] for i in range(1,21)}
    row=temporal_row(1,records,genres,20,5,42)
    assert row['split_status'] == 'SEPARABLE'
    assert row['genre_tv'] == 1
    assert row['early_count'] == row['recent_count'] == 5
    assert row['early_max_timestamp'] < row['recent_min_timestamp']
    assert row == temporal_row(1,list(reversed(records)),genres,20,5,42)


def test_aggregation_is_per_user_and_low_denominator_excludes_absent():
    sample=pair_summary([1],[0],.05)
    rows=[{'user_id':1,'positive_count':5,'negative':{'uniform':sample},'low':None},
          {'user_id':2,'positive_count':105,'negative':{'uniform':sample},'low':{
              'pair':sample,'high_score_fraction':.25,'score_percentile':.75,'count':4}}]
    result=aggregate_rows(rows)
    assert result['users'] == 2 and result['low']['users'] == 1
    assert result['low']['high_score_fraction'] == .25
    assert result['strata']['5_19']['users'] == 1
    assert result['strata']['100_plus']['users'] == 1


def test_train_only_guard_protects_pathlib_builtin_arrow_and_digest(tmp_path):
    import pyarrow.parquet as pq
    from agenticrec.data import digest
    allowed=tmp_path/'train.json'; allowed.write_text('{}')
    blocked=tmp_path/'valid.json'; blocked.write_text('{}')
    with train_only_files([allowed]) as reads:
        assert json.loads(allowed.read_text()) == {}
        digest(allowed)
        for operation in (lambda: blocked.read_text(),lambda: open(blocked),
                          lambda: digest(blocked),lambda: pq.read_table(blocked)):
            with pytest.raises(PermissionError): operation()
    assert sorted(reads) == [str(allowed.resolve())]
    assert blocked.read_text() == '{}'


def test_checkpoint_tamper_is_rejected_before_loading(tmp_path):
    result={'status':'VERIFIED','checkpoint_sha256':'0'*64}
    (tmp_path/'result.json').write_text(json.dumps(result))
    (tmp_path/'best.pt').write_bytes(b'bad weights')
    with pytest.raises(ValueError, match='Checkpoint SHA'):
        verified_model(tmp_path, {}, (None,0,0,None,None), 'irrelevant', 42)


def test_plan_forbids_training_holdout_and_unknown_parameters():
    from pathlib import Path
    plan=json.loads((Path(__file__).parents[2]/'configs/lightgcn_training_diagnostics_20261011.json').read_text())
    validate_plan(plan)
    for key,value in [('train_only',False),('training_runs',1),('remote_api_requests',1),('low_rating_max',3)]:
        with pytest.raises(ValueError): validate_plan({**plan,key:value})
    with pytest.raises(ValueError): validate_plan({**plan,'test_access':True})


def test_evidence_outputs_refuse_overwrite(tmp_path):
    path=tmp_path/'result.json'
    write_once(path,{'status':'fixture'})
    with pytest.raises(FileExistsError): write_once(path,{'status':'modified'})
    assert json.loads(path.read_text())['status'] == 'fixture'


def test_low_scores_are_diagnostic_only_and_model_is_immutable():
    import torch
    from agenticrec.models.bpr import BPRMF
    from agenticrec.training_signal_diagnostics import model_rows
    model=BPRMF(1,6,1)
    with torch.no_grad():
        model.users.weight.fill_(1)
        model.items.weight.copy_(torch.tensor([[0.],[2.],[3.],[5.],[4.],[1.],[6.]]))
    before={k:v.clone() for k,v in model.state_dict().items()}
    sample=[{'user_id':1,'positive_ids':[1,2],'pools':[[5,6],[5,6]],'low_ids':[3]}]
    rows=model_rows(model,'bpr',sample,{1:[1,2]},{1:[3]},{1:{1,2,3,4}},6,
                    {'score_batch_size':128,'easy_gradient_below':.05,'top_k':10})
    assert rows[0]['negative']['uniform']['margin'] == 1.5
    assert rows[0]['negative']['hardest_pool']['margin'] == -3.5
    assert rows[0]['low']['pair']['margin'] == -3
    assert rows[0]['low']['score_percentile'] == .5
    for k,v in model.state_dict().items(): assert torch.equal(v,before[k])


def test_small_complete_run_has_no_holdout_access_and_replays(tmp_path,monkeypatch):
    import torch
    import pyarrow as pa
    import pyarrow.parquet as pq
    import agenticrec.training_signal_diagnostics as diag
    from agenticrec.data import digest
    from agenticrec.lightgcn_optimization import build_model
    from pathlib import Path
    plan_path=tmp_path/'plan.json'
    plan=json.loads((Path(__file__).parents[2]/'configs/lightgcn_training_diagnostics_20261011.json').read_text())
    write_once(plan_path,plan)
    data=tmp_path/'data/processed/ml-1m-v1'; data.mkdir(parents=True)
    pq.write_table(pa.table({'raw_user_id':[1,1,1,2,2], 'raw_item_id':[1,2,3,2,4],
                             'rating':[4,5,1,5,2],'timestamp':[10,20,30,40,50]}),data/'train.parquet')
    pq.write_table(pa.table({'item_id':[1,2,3,4], 'genres':[['A'],['B'],['A'],['B']]}),data/'items.parquet')
    write_once(data/'id_map.json',{'user_to_model':{'1':1,'2':2},'item_to_model':{str(i):i for i in range(1,5)}})
    manifest={'train_positive_threshold':4,'id_map_sha256':digest(data/'id_map.json'),
              'files':{'items.parquet':{'sha256':digest(data/'items.parquet')},'train_positive.parquet':{'sha256':'fixture'}}}
    artifact=tmp_path/'artifacts'; artifact.mkdir()
    write_once(artifact/'data_manifest.json',manifest)
    contract=(manifest,2,4,{1:{1,2,3},2:{2,4}},[(1,1),(1,2),(2,2)])
    monkeypatch.setattr(diag,'ROOT',tmp_path)
    monkeypatch.setattr(diag,'load_train_contract',lambda:contract)
    monkeypatch.setattr(diag,'code_identity',lambda:'fixture-source')
    source=tmp_path/plan['source_plan']; source.parent.mkdir(parents=True)
    original_plan=json.loads((Path(__file__).parents[2]/'configs/lightgcn_optimization_20261011.json').read_text())
    write_once(source,original_plan)
    base=artifact/'runs/lightgcn_optimization_20261011'; base.mkdir(parents=True)
    write_once(base/'selection_frozen.json',{'selected':{'lightgcn':'gcn_edge_uniform','bpr':'bpr_edge'},
                                            'plan_sha256':digest(source),'code_sha256':'fixture-source'})
    # Real model serialization plus production verification, never fake quality.
    for name in plan['models']:
        trial=next(t for t in original_plan['trials'] if t['name']==name)
        for seed in plan['model_seeds']:
            directory=base/f'seed_{seed}'/name; directory.mkdir(parents=True)
            torch.manual_seed(seed); model=build_model(trial,contract)
            identity={'seed':seed,'plan_sha256':digest(source),'code_sha256':'fixture-source',
                      'data_manifest_sha256':digest(artifact/'data_manifest.json'),
                      'id_map_sha256':manifest['id_map_sha256'],'train_positive_sha256':'fixture',
                      'torch_version':str(torch.__version__)}
            torch.save({'state_dict':model.state_dict(),'trial':trial,'identity':identity,
                        'dimensions':[2,4],'best_epoch':1},directory/'best.pt')
            write_once(directory/'result.json',{'status':'VERIFIED','trial':trial,'identity':identity,
                       'best_epoch':1,'checkpoint_sha256':digest(directory/'best.pt')})
    # Actual module/dependency identities are permitted separately from fixture root.
    monkeypatch.setattr(diag,'source_paths',lambda:list((Path(diag.__file__).parent).rglob('*.py')))
    result=diag.run(plan_path,tmp_path/'run1')
    other=diag.run(plan_path,tmp_path/'run2')
    assert result == other
    assert result['holdout_labels_read'] is False and result['training_runs'] == 0
    assert result['train_users']==2 and len(result['models'])==6
    for item in result['models'].values():
        assert digest(tmp_path/'run1'/item['rows_file']) == item['rows_sha256']
    with pytest.raises(FileExistsError): diag.run(plan_path,tmp_path/'run1')
    first=base/'seed_42/gcn_edge_uniform'
    original=(first/'result.json').read_text()
    bad=json.loads(original); bad['identity']['id_map_sha256']='tampered'
    (first/'result.json').write_text(json.dumps(bad))
    trial=next(t for t in original_plan['trials'] if t['name']=='gcn_edge_uniform')
    with pytest.raises(ValueError,match='identity mismatch'):
        diag.verified_model(first,trial,contract,digest(source),42)
