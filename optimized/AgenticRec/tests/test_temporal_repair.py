import pytest
import torch

from agenticrec.evaluation.temporal_repair_models import temporal_examples,corrected_positive_loss,any_positive_loss,group_weights


def test_temporal_targets_use_real_gap_and_complete_timestamp_batch():
    rows = [{'user_id':1,'item_id':i+1,'timestamp':(0 if i<15 else 40*86400),'rating':4} for i in range(20)]
    cases = temporal_examples(rows,set(range(1,21)))
    assert len(cases)==3
    assert all(r['anchor']==0 and r['gap_days']==30 and r['masked_item_ids']==list(range(1,16)) for r in cases)
    assert all(r['relevant_item_ids']==list(range(16,21)) for r in cases)
    assert not temporal_examples([dict(r,timestamp=0) for r in rows],set(range(1,21)))


def test_past_positive_mask_has_zero_gradient_and_no_target_collision():
    logits = torch.tensor([[2.,1.,3.,4.]],requires_grad=True)
    targets = torch.tensor([[0.,0.,1.,0.]])
    mask = torch.tensor([[True,True,False,False]])
    corrected_positive_loss(logits,targets,mask).backward()
    assert logits.grad[0,:2].eq(0).all() and logits.grad[0,2]<0 and logits.grad[0,3]>0
    with pytest.raises(ValueError,match='overlaps'):
        corrected_positive_loss(logits,targets,torch.ones_like(mask))


def test_any_positive_objective_rejects_masked_target_and_respects_valid_pool():
    logits = torch.tensor([[1.,2.,3.]],requires_grad=True)
    positive = torch.tensor([[False,True,False]])
    allowed = torch.tensor([[True,True,False]])
    any_positive_loss(logits,positive,allowed).backward()
    assert logits.grad[0,2]==0 and logits.grad[0,1]<0
    with pytest.raises(ValueError):any_positive_loss(logits,positive,torch.zeros_like(allowed))


def test_sampling_mass_is_group_and_user_balanced():
    examples = [dict(kind='ordinal',training_user_id=1)]*3+[dict(kind='long',training_user_id=2)]*6+[dict(kind='long',training_user_id=3)]
    weights = group_weights(examples)
    assert float(weights[:3].sum())==pytest.approx(.25)
    assert float(weights[3:9].sum())==pytest.approx(.375)
    assert float(weights[9])==pytest.approx(.375)
