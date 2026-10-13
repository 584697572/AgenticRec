import pytest

from agenticrec.evaluation.sequence_ml1m import training_examples


def test_chronological_train_prefix_and_future_targets():
    rows = [{'user_id':1,'item_id':i,'timestamp':i//2,'rating':4} for i in range(15,0,-1)]
    examples = training_examples(rows)
    assert [e['liked_item_ids'] for e in examples] == [list(range(10,13)),list(range(8,13)),list(range(3,13))]
    assert all(e['relevant_item_ids']==[13,14,15] for e in examples)
    assert not training_examples(rows[:10])


def test_duplicate_and_nonpositive_feedback_rejected():
    rows = [{'user_id':1,'item_id':1,'timestamp':1,'rating':4}]*2
    with pytest.raises(ValueError,match='Duplicate'): training_examples(rows)
    with pytest.raises(ValueError,match='positives'):
        training_examples([dict(rows[0],rating=3)])
