import copy
import pytest
import torch

from agenticrec.evaluation.frontier_state import (empty_state, merge_state, feasible_ids,
                                                 verify_delivery, validate_delta,current_obligation_errors)
from agenticrec.evaluation.frontier_models import CatalogGenerator, metadata_tensor, intent_vector, policy_training_data
from agenticrec.evaluation.frontier_experiment import score, summarize


def delta(**kwargs):
    value=dict(include_genres=None,exclude_genres=None,year_min=None,year_max=None,k=None,
               liked_item_ids=[],seen_item_ids=[],exclude_previous=False,reset_constraints=False,
               genre_mode='replace',evidence={})
    value.update(kwargs)
    return value


CATALOG={1:{'raw_id':101,'title':'A','year':1989,'genres':['Action','Thriller']},
         2:{'raw_id':102,'title':'B','year':1995,'genres':['Drama','Horror']},
         3:{'raw_id':103,'title':'C','year':1998,'genres':['Drama']}}


def test_feedback_keeps_unmentioned_years_and_liked_movies():
    state=merge_state(empty_state(),delta(include_genres=['Action'],year_min=1980,year_max=2000,
                                          liked_item_ids=[1],seen_item_ids=[1]))
    updated=merge_state(state,delta(include_genres=['Drama']))
    assert updated['include_genres']==['Drama']
    assert updated['year_min']==1980 and updated['year_max']==2000
    assert updated['liked_item_ids']==[1] and updated['seen_item_ids']==[1]
    assert state['include_genres']==['Action']


def test_add_and_reset_have_different_effects():
    previous=dict(empty_state(),include_genres=['Action'],year_min=1980)
    added=merge_state(previous,delta(include_genres=['Thriller'],genre_mode='add'))
    assert added['include_genres']==['Action','Thriller']
    reset=merge_state(previous,delta(include_genres=['Drama'],reset_constraints=True))
    assert reset['include_genres']==['Drama'] and reset['year_min'] is None


def test_previous_output_exclusion_is_explicit_and_persistent():
    state=dict(empty_state(),previous_item_ids=[1,2])
    assert merge_state(state,delta())['excluded_previous_ids']==[]
    assert merge_state(state,delta(exclude_previous=True))['excluded_previous_ids']==[1,2]


def test_hard_gate_intersects_genres_and_excludes_seen():
    state=dict(empty_state(),include_genres=['Drama'],exclude_genres=['Horror'],k=1)
    assert feasible_ids(CATALOG,state)==[3]
    assert verify_delivery([3],CATALOG,state)==[]
    assert verify_delivery([2],CATALOG,state)==['PUBLIC_CONSTRAINT_VIOLATION']
    state['seen_item_ids']=[3]
    assert feasible_ids(CATALOG,state)==[]


def test_no_constraint_relaxation_on_shortfall():
    state=dict(empty_state(),include_genres=['Drama'],exclude_genres=['Horror'],k=10)
    assert verify_delivery([3],CATALOG,state)==['WRONG_QUANTITY']
    assert feasible_ids(CATALOG,dict(state,year_min=2001,year_max=1999))==[]


def test_delta_rejects_hallucinated_ids_and_hidden_fields():
    with pytest.raises(ValueError):validate_delta(delta(liked_item_ids=[77]),'Liked movie',[])
    with pytest.raises(ValueError):validate_delta(dict(delta(),relevant_raw_ids=[101]),'msg',[])


def test_evidence_rejects_unsupported_year_and_reset():
    with pytest.raises(ValueError):validate_delta(delta(year_min=1990,evidence={'year_min':'Please recommend movies'}),'Please recommend movies',[])
    with pytest.raises(ValueError):validate_delta(delta(reset_constraints=True),'Keep previous conditions',[])
    with pytest.raises(ValueError):validate_delta(delta(exclude_previous=True),'Exclude watched movies',[])


def test_empty_unsupported_genre_delta_cannot_erase_previous_requirement():
    cleaned=validate_delta(delta(include_genres=[],exclude_genres=[]),'Keep previous conditions',[])
    assert cleaned['include_genres'] is None
    previous=dict(empty_state(),include_genres=['Action'])
    assert merge_state(previous,cleaned)['include_genres']==['Action']


def test_generator_cannot_emit_padding_duplicates_or_disallowed_ids():
    torch.manual_seed(42)
    model=CatalogGenerator(3,metadata_tensor(3,CATALOG))
    assert model.generate([1],intent_vector(empty_state()),[2,3],10) in ([2,3],[3,2])
    assert model.generate([1],intent_vector(empty_state()),[],10)==[]


def test_policy_environment_rewards_actual_complete_state():
    x,rewards=policy_training_data(42,60)
    assert x.shape==(60,10) and rewards.shape==(60,3)
    assert set(rewards.flatten().tolist())=={0.,1.}
    assert (rewards.max(1).values==1).all()
    assert (rewards[:,0]==0).any()


def test_extended_score_preserves_denominator_and_negative_genre():
    truth={str(v['raw_id']):v for v in CATALOG.values()}
    result={'raw_ids':[102,103],'status':'OK','extraction_verified':True}
    label={'k':2,'genres':['Drama'],'exclude_genres':['Horror'],'year_min':None,'year_max':None,
           'excluded_raw_ids':[],'relevant_raw_ids':[103]}
    value=score(result,label,truth)
    assert not value['strict_success'] and value['constraint_precision']==.5 and value['hit']==1
    assert not score(dict(result,raw_ids=[None,103]),label,truth)['strict_success']


def test_original_success_does_not_imply_preference_hit():
    truth={'103':CATALOG[3]}
    result={'raw_ids':[103],'status':'OK','extraction_verified':True}
    label={'k':1,'genres':['Drama'],'year_min':None,'year_max':None,'excluded_raw_ids':[],'relevant_raw_ids':[101]}
    value=score(result,label,truth)
    assert value['strict_success'] and value['hit']==0


def test_incomplete_summary_never_reports_complete_accuracy():
    value=summarize([], [{'episode_id':'not-run'}],[],systems=('C1',))
    assert value['systems']['C1']['strict_success'] is None
    assert value['systems']['C1']['quality_metrics']['hit'] is None


def test_public_verifier_catches_silent_omission_without_private_labels():
    message='Give exactly 10 drama movies from 1991 through 2000. Do not include horror movies.'
    errors=current_obligation_errors(empty_state(),message,[])
    assert 'EXPLICIT_YEAR_RANGE_OMITTED' in errors
    assert 'EXPLICIT_GENRE_OMITTED:Drama' in errors
    assert 'EXPLICIT_NEGATIVE_GENRE_OMITTED:Horror' in errors
    state=dict(empty_state(),include_genres=['Drama'],exclude_genres=['Horror'],year_min=1991,year_max=2000)
    assert current_obligation_errors(state,message,[])==[]


def test_movie_title_year_is_not_a_requested_year_range():
    message='I have watched and liked these movies: A (1989). Recommend exactly 10 other movies.'
    context=[dict(CATALOG[1],item_id=1)]
    assert current_obligation_errors(empty_state(),message,context)==['EXPLICIT_WATCHED_ITEMS_OMITTED','EXPLICIT_LIKED_ITEMS_OMITTED']


def test_switch_does_not_reimpose_rejected_genre():
    state=dict(empty_state(),include_genres=['Comedy'],year_min=1985,year_max=2000)
    assert current_obligation_errors(state,'Switch to comedy instead of action. Keep previous year limits.',[])==[]
    assert current_obligation_errors(state,'Keep the action requirement and also require thriller.',[])==[
        'EXPLICIT_GENRE_OMITTED:Thriller','EXPLICIT_RETAINED_GENRE_OMITTED:Action']
