"""Independent explicit-feedback objective and train-only contract checks."""
import json
from pathlib import Path

import pytest
import torch

from agenticrec.explicit_low_training import LowSampler, explicit_loss, mapped_lows, validate_plan
from agenticrec.models.bpr import BPRMF, bpr_loss
from agenticrec.models.lightgcn import lightgcn_loss, normalized_bipartite_graph
from agenticrec.models.lightgcn_residual import ResidualLightGCN


@pytest.mark.parametrize('kind', ['bpr', 'lightgcn'])
@pytest.mark.parametrize('weight', [0., .25])
def test_loss_and_all_gradients_against_dense_oracle(kind, weight):
    torch.manual_seed(42)
    model = BPRMF(3, 5, 4) if kind == 'bpr' else ResidualLightGCN(
        3, 5, 4, 1, normalized_bipartite_graph(3, 5, [(1, 1), (2, 2), (3, 3)]), [.5, .5])
    users = torch.tensor([1, 2, 3]); positive = torch.tensor([1, 2, 3])
    unknown = torch.tensor([3, 4, 5]); low = torch.tensor([2, 5, 0])
    loss, stats = explicit_loss(model, kind, users, positive, unknown, low, weight, .0001)
    loss.backward(); actual = [p.grad.clone() for p in model.parameters()]
    eu = model.users.weight.detach().clone().requires_grad_()
    ei = model.items.weight.detach().clone().requires_grad_()
    if kind == 'lightgcn':
        ego = torch.cat((eu[1:], ei[1:])); out = (ego + model.graph.to_dense() @ ego) / 2
        ue, ie = out[:3], out[3:]
    else:
        ue, ie = eu[1:], ei[1:]
    expected = 0.
    for row, user in enumerate(users.tolist()):
        p, n, l = positive[row].item(), unknown[row].item(), low[row].item()
        ps = (ue[user-1] * ie[p-1]).sum(); ns = (ue[user-1] * ie[n-1]).sum()
        expected = expected + torch.nn.functional.softplus(ns-ps) / 3
        expected = expected + .0001*(eu[user].square().sum()+ei[p].square().sum()+ei[n].square().sum())/6
        if l:
            ls = (ue[user-1] * ie[l-1]).sum()
            expected = expected + weight*torch.nn.functional.softplus(ls-ps)/3
    expected.backward()
    torch.testing.assert_close(loss, expected, atol=1e-7, rtol=0)
    for a, b in zip(actual, (eu.grad, ei.grad)):
        torch.testing.assert_close(a, b, atol=1e-7, rtol=0)
    assert stats['low_pairs'] == 2
    if weight == 0:
        original = lightgcn_loss if kind == 'lightgcn' else bpr_loss
        torch.testing.assert_close(loss, original(model, users, positive, unknown, .0001), atol=1e-7, rtol=0)


def test_empty_low_users_have_no_padding_gradient_or_loss():
    model = BPRMF(1, 3, 2)
    u, p, n = (torch.tensor([v]) for v in (1, 1, 3))
    loss, stats = explicit_loss(model, 'bpr', u, p, n, torch.tensor([0]), .25, .0001)
    torch.testing.assert_close(loss, bpr_loss(model, u, p, n, .0001))
    loss.backward(); assert torch.equal(model.items.weight.grad[0], torch.zeros(2))
    assert stats['low_pairs'] == 0 and stats['low_loss'] == 0


def test_low_sampler_reproducible_independent_and_empty_safe():
    torch.manual_seed(7); initial = torch.random.get_rng_state().clone()
    a = LowSampler(3, 5, {1: [2, 4], 2: [5], 3: []}, 42)
    b = LowSampler(3, 5, {1: [2, 4], 2: [5], 3: []}, 42)
    users = torch.tensor([1, 2, 3]*100)
    draws = a.sample(users); assert torch.equal(draws, b.sample(users))
    assert set(draws[::3].tolist()) == {2, 4}
    assert set(draws[1::3].tolist()) == {5} and set(draws[2::3].tolist()) == {0}
    assert torch.equal(initial, torch.random.get_rng_state())
    for bad in ({1:[0]}, {4:[2]}, {1:[2,2]}, {1:[6]}):
        with pytest.raises(ValueError): LowSampler(3,5,bad,42)
    with pytest.raises(ValueError): a.sample(torch.tensor([0]))


def test_low_rows_only_train_threshold_and_train_mapping():
    rows = [dict(raw_user_id=u,raw_item_id=i,rating=r) for u,i,r in
            [(10,20,1),(10,21,2),(10,22,3),(10,23,4),(99,20,1),(10,999,1)]]
    lows = mapped_lows(rows, {'10':1}, {'20':1,'21':2,'22':3,'23':4},
                       {1:{1,2,3,4}}, [(1,4)])
    assert lows == {1:[1,2]}
    with pytest.raises(ValueError): mapped_lows(rows, {'10':1}, {'20':1,'21':2}, {1:{1}}, [])
    with pytest.raises(ValueError): mapped_lows(rows+rows[:1], {'10':1}, {'20':1,'21':2}, {1:{1,2}}, [])


def test_plan_is_fixed_and_excludes_test_and_hard_negative_stack():
    plan = json.loads((Path(__file__).parents[2]/'configs/lightgcn_explicit_low_20261012.json').read_text())
    validate_plan(plan)
    for key,value in [('test_access',True),('negative_pool_size',16),('seeds',[42]),('remote_api_requests',1)]:
        with pytest.raises(ValueError): validate_plan({**plan,key:value})
    altered = json.loads(json.dumps(plan)); altered['trials'][1]['low_weight'] = .5
    with pytest.raises(ValueError): validate_plan(altered)


def test_toy_training_freeze_reloads_and_refuses_test_or_overwrite(tmp_path, monkeypatch):
    import agenticrec.explicit_low_training as module
    monkeypatch.setattr(module,'ROOT',tmp_path)
    monkeypatch.setattr(module,'BASE',tmp_path/'runs')
    (tmp_path/'artifacts').mkdir(); (tmp_path/'artifacts/data_manifest.json').write_text('fixture')
    path=tmp_path/'data/eval_private/ml-1m-v1/valid.json'; path.parent.mkdir(parents=True)
    path.write_text('fixture development')
    manifest={'id_map_sha256':'fixture','files':{'train_positive.parquet':{'sha256':'fixture'},
                'train.parquet':{'sha256':'fixture'}}}
    contract=(manifest,2,12,{1:{1,2},2:{2,3}},[(1,1),(2,2)])
    trials=[{'name':name,'model':'bpr','layers':0,'embedding_dim':3,'layer_weights':[1.],
              'sampling':'edge','learning_rate':.001,'regularization':.0001,'batch_size':2,
              'max_epochs':2,'patience':2,'low_weight':weight}
              for name,weight in [('bpr_uniform',0.),('bpr_low25',.25)]]
    plan={'torch_threads':1,'negative_pool_size':8,'seeds':[7,42,2026],'trials':trials}
    plan_path=tmp_path/'plan.json'; plan_path.write_text(json.dumps(plan))
    monkeypatch.setattr(module,'plan_and_groups',lambda p:(plan,{'fit':[1],'select':[],'audit':[2]}))
    monkeypatch.setattr(module,'load_train_contract',lambda:contract)
    monkeypatch.setattr(module,'load_lows',lambda c:{1:[2],2:[3]})
    reads=[]
    def cohort(split,path):
        reads.append(split); assert split=='valid'
        return {'user_ids':[1,2],'candidate_ids':list(range(1,13)),
                'relevant_by_user':{'1':[3],'2':[1]},'seen_by_user':{'1':[1,2],'2':[2,3]}}
    monkeypatch.setattr(module,'load_frozen_cohort',cohort)
    results=[module.train(plan_path,t['name'],42) for t in trials]
    assert all(r['status']=='VERIFIED' and r['optimizer_steps']==2 for r in results)
    assert results[0]['identity']['initial_state_sha256']==results[1]['identity']['initial_state_sha256']
    assert [r['paired_stream_sha256'] for r in results[0]['curve']]==[r['paired_stream_sha256'] for r in results[1]['curve']]
    directory=module.BASE/'seed_42/bpr_low25'
    _,loaded=module.load_verified(directory,trials[1],contract,module.digest(plan_path),42)
    assert loaded['checkpoint_sha256']==results[1]['checkpoint_sha256'] and reads==['valid','valid']
    with pytest.raises(FileExistsError):module.train(plan_path,'bpr_low25',42)
    with pytest.raises(FileNotFoundError):module.train(plan_path,'bpr_low25',7)
    (directory/'best.pt').write_bytes(b'tampered')
    with pytest.raises(ValueError,match='SHA'):module.load_verified(directory,trials[1],contract,module.digest(plan_path),42)


def test_train_low_loader_rejects_changed_train_artifact(tmp_path,monkeypatch):
    import agenticrec.explicit_low_training as module
    monkeypatch.setattr(module,'ROOT',tmp_path)
    path=tmp_path/'data/processed/ml-1m-v1/train.parquet';path.parent.mkdir(parents=True)
    path.write_bytes(b'tampered')
    contract=({'files':{'train.parquet':{'sha256':'wrong'}}},1,2,{1:{1}},[(1,1)])
    with pytest.raises(ValueError,match='SHA'):module.load_lows(contract)
