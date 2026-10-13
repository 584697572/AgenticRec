import pytest

from agenticrec.evaluation.ranking_diagnosis import describe, rank_bucket, rotate_inputs


@pytest.mark.parametrize('rank,expected',[(1,'1-10'),(10,'1-10'),(11,'11-20'),(100,'51-100'),
                                       (101,'101-500'),(500,'101-500'),(501,'501+'),(None,'no_feasible_target')])
def test_target_rank_boundaries(rank,expected):
    assert rank_bucket(rank)==expected


def test_rotation_preserves_history_length_and_uses_only_other_public_history():
    public = [{'catalog_context':[{'item_id':i} for i in row],'private_targets':[999]} for row in ([1,2,3],[4,5,6],[1]*5,[2]*5)]
    assert rotate_inputs(public)==[[4,5,6],[1,2,3],[2]*5,[1]*5]
    with pytest.raises(ValueError,match='singleton'): rotate_inputs(public[:1])


def test_descriptive_quantiles_interpolate_actual_values():
    assert describe([])=={'n':0}
    stats = describe([10,0,20])
    assert stats['median']==10 and stats['mean']==10 and stats['p90']==18
