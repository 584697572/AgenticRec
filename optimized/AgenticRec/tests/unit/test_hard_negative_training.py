"""Independent loss/gradient and train-only sampling checks."""
import json
from pathlib import Path

import pytest
import torch

from agenticrec.hard_negative_training import (
    PoolSampler, pooled_loss, validate_plan, paired_streams, train, load_verified,
)
from agenticrec.models.bpr import BPRMF, bpr_loss
from agenticrec.models.lightgcn import lightgcn_loss, normalized_bipartite_graph
from agenticrec.models.lightgcn_residual import ResidualLightGCN


@pytest.mark.parametrize('kind',['bpr','lightgcn'])
@pytest.mark.parametrize('fraction',[0.0,.25,1.0])
def test_pooled_selection_loss_and_all_gradients_match_independent_oracle(kind,fraction):
    torch.manual_seed(42)
    model=BPRMF(2,4,3) if kind=='bpr' else ResidualLightGCN(2,4,3,1,
        normalized_bipartite_graph(2,4,[(1,1),(2,2)]),[.5,.5])
    users=torch.tensor([1,2]); pos=torch.tensor([1,2]); pools=torch.tensor([[2,3,4],[1,3,4]])
    coins=torch.tensor([.1,.8])
    loss,negative,diagnostic=pooled_loss(model,kind,users,pos,pools,coins,fraction,.0001)
    loss.backward()
    actual=[p.grad.clone() for p in model.parameters()]
    eu=model.users.weight.detach().clone().requires_grad_(); ei=model.items.weight.detach().clone().requires_grad_()
    if kind=='lightgcn':
        ego=torch.cat((eu[1:],ei[1:])); out=(ego+model.graph.to_dense()@ego)/2
        ue,ie=out[:2],out[2:]
    else: ue,ie=eu[1:],ei[1:]
    expected_neg=[]
    for row,u in enumerate(users.tolist()):
        ids=pools[row].tolist()
        values=[(ue[u-1]*ie[i-1]).sum().detach().item() for i in ids]
        idx=max(range(len(ids)),key=lambda i:values[i]) if coins[row]<fraction else 0
        expected_neg.append(ids[idx])
    expected_neg=torch.tensor(expected_neg)
    ps=(ue[users-1]*ie[pos-1]).sum(1); ns=(ue[users-1]*ie[expected_neg-1]).sum(1)
    expected=torch.nn.functional.softplus(ns-ps).mean()
    expected += .0001*(eu[users].square()+ei[pos].square()+ei[expected_neg].square()).sum()/4
    expected.backward()
    assert torch.equal(negative,expected_neg)
    torch.testing.assert_close(loss,expected,atol=1e-7,rtol=0)
    for a,b in zip(actual,(eu.grad,ei.grad)): torch.testing.assert_close(a,b,atol=1e-7,rtol=0)
    assert diagnostic['hard_selected'] == int((coins<fraction).sum())
    if fraction==0:
        original=lightgcn_loss if kind=='lightgcn' else bpr_loss
        torch.testing.assert_close(loss,original(model,users,pos,pools[:,0],.0001),atol=1e-7,rtol=0)


def test_pool_sampler_excludes_all_rated_and_padding_and_is_reproducible():
    seen={1:{1,2,3},2:{3,4}}
    a=PoolSampler(2,12,seen,8,42); b=PoolSampler(2,12,seen,8,42)
    users=torch.tensor([1,2]*100)
    pool=a.sample(users); assert torch.equal(pool,b.sample(users))
    for user,row in zip(users.tolist(),pool.tolist()):
        assert len(set(row))==8 and set(row).isdisjoint(seen[user]) and min(row)>0 and max(row)<=12
    with pytest.raises(ValueError): PoolSampler(1,10,{1:{1,2,3}},8,42)
    with pytest.raises(ValueError): PoolSampler(1,10,{1:{0}},8,42)
    with pytest.raises(ValueError): a.sample(torch.tensor([0]))


def test_paired_random_streams_share_candidate_draws_and_masks_without_global_rng():
    torch.manual_seed(42); before=torch.random.get_rng_state().clone()
    edge_rng, pool_seed,mask=paired_streams(42)
    a=torch.rand(10000,generator=mask)
    assert .23<float((a<.25).float().mean())<.27
    other,other_seed,other_mask=paired_streams(42)
    assert pool_seed==other_seed and edge_rng.random()==other.random()
    assert torch.equal(a,torch.rand(10000,generator=other_mask))
    assert torch.equal(before,torch.random.get_rng_state())


def test_plan_rejects_expanding_experiment_or_using_test():
    plan=json.loads((Path(__file__).parents[2]/'configs/lightgcn_hard_negatives_20261011.json').read_text())
    validate_plan(plan)
    for key,value in [('test_access',True),('negative_pool_size',16),('seeds',[42]),('remote_api_requests',1)]:
        with pytest.raises(ValueError):validate_plan({**plan,key:value})
    altered=json.loads(json.dumps(plan)); altered['trials'][1]['hard_fraction']=.5
    with pytest.raises(ValueError):validate_plan(altered)


def test_toy_training_checkpoint_reload_and_no_test_access(tmp_path,monkeypatch):
    import agenticrec.hard_negative_training as module
    monkeypatch.setattr(module,'ROOT',tmp_path)
    monkeypatch.setattr(module,'BASE',tmp_path/'runs')
    (tmp_path/'artifacts').mkdir(); (tmp_path/'artifacts/data_manifest.json').write_text('fixture')
    cohort_path=tmp_path/'data/eval_private/ml-1m-v1/valid.json'; cohort_path.parent.mkdir(parents=True)
    cohort_path.write_text('fixture development labels')
    manifest={'id_map_sha256':'fixture','files':{'train_positive.parquet':{'sha256':'fixture'}}}
    contract=(manifest,2,12,{1:{1,2},2:{2,3}},[(1,1),(1,2),(2,2),(2,3)])
    trial={'name':'tiny','model':'lightgcn','layers':1,'embedding_dim':4,'layer_weights':[.5,.5],
           'sampling':'edge','learning_rate':.001,'regularization':.0001,'batch_size':2,'max_epochs':2,
           'patience':2,'hard_fraction':.25}
    plan={'torch_threads':1,'negative_pool_size':8,'seeds':[7,42,2026],'trials':[trial]}
    plan_path=tmp_path/'plan.json'; plan_path.write_text(json.dumps(plan))
    monkeypatch.setattr(module,'plan_and_groups',lambda p:(plan,{'fit':[1],'select':[],'audit':[2]}))
    monkeypatch.setattr(module,'load_train_contract',lambda:contract)
    reads=[]
    def cohort(split,path):
        reads.append(split); assert split=='valid'
        return {'user_ids':[1,2],'candidate_ids':list(range(1,13)),
                'relevant_by_user':{'1':[3],'2':[1]},'seen_by_user':{'1':[1,2],'2':[2,3]}}
    monkeypatch.setattr(module,'load_frozen_cohort',cohort)
    result=train(plan_path,'tiny',42)
    assert result['status']=='VERIFIED' and result['actual_epochs']==2
    assert result['optimizer_steps']==4 and result['test_evaluation']=='NOT EVALUATED'
    directory=module.BASE/'seed_42/tiny'
    model,loaded=load_verified(directory,trial,contract,module.digest(plan_path),42)
    assert loaded['checkpoint_sha256']==result['checkpoint_sha256']
    assert reads==['valid']
    with pytest.raises(FileExistsError):train(plan_path,'tiny',42)
    with pytest.raises(FileNotFoundError):train(plan_path,'tiny',7)
    (directory/'best.pt').write_bytes(b'tampered')
    with pytest.raises(ValueError,match='SHA'):load_verified(directory,trial,contract,module.digest(plan_path),42)
