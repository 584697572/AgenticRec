"""Statistics count users, and frozen experiment records retain public schemas."""
from dataclasses import asdict
import json

from agenticrec.fusion_experiment import user_means, reconstruct, requests_for_users
from agenticrec.pipeline import FixedRequest
from agenticrec.ranking.constraints import Catalog, Item, Constraints


def test_repeated_states_and_seeds_do_not_multiply_user_denominator():
    records=[{'user':1,'metrics':{'x':{'ndcg':1.0}}},
             {'user':1,'metrics':{'x':{'ndcg':0.0}}},
             {'user':2,'metrics':{'x':{'ndcg':0.25}}},
             {'user':3,'metrics':{'x':None}}]
    assert user_means(records,'x')=={1:.5,2:.25}


def test_serialized_request_and_snapshot_reconstruction_has_no_targets():
    req=FixedRequest(Constraints(k=10),liked_item_ids=(1,),disliked_item_ids=(2,))
    record={'request':json.loads(json.dumps(asdict(req))),
            'snapshot':{'scores':{'content':{'3':.5}},'rankings':{'content':[3]},
                        'eligible_ids':[3],'profile_source':'cold_start','per_route':100},
            'relevant':[999]}
    actual,snap=reconstruct(record)
    assert actual==req and snap['scores']['content']=={3:.5}
    assert 'relevant' not in asdict(actual) and 'relevant' not in snap


def test_seed_feedback_and_genre_are_derived_from_train_rows_only():
    catalog=Catalog([Item(i,str(i),frozenset({'Comedy'}),1995) for i in range(1,8)])
    rows=[(1,1,5,1),(1,2,5,2),(1,3,5,3),(1,4,5,4),(1,5,1,5)]
    reqs=requests_for_users(rows,catalog,[1])
    assert len(reqs)==3
    assert reqs[0][2].liked_item_ids==(4,3,2)
    assert reqs[1][2].disliked_item_ids==(5,)
    assert reqs[2][2].constraints.include_genres==('Comedy',)
    assert 6 not in reqs[0][2].seen_item_ids and 7 not in reqs[0][2].liked_item_ids
