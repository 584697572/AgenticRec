import json
import pytest

from agenticrec.evaluation.three_way_comparison import score, prior_users, DATA, MatchedJournal
from agenticrec.adapters.openai_compatible import OpenAICompatibleTransport, HttpResponse


@pytest.mark.skipif(not (DATA/'id_map.json').exists(),reason='Requires frozen local experiment data')
def test_prior_users_resolves_actual_frozen_map_without_nonexistent_users_table():
    users=prior_users()
    assert len(users)>=150 and all(type(u) is int and u>0 for u in users)


def test_unanswerable_execution_error_is_not_correct_refusal():
    label={'k':10,'genres':[],'exclude_genres':[],'year_min':2026,'year_max':2027,
           'excluded_raw_ids':[],'relevant_raw_ids':[],'exclude_previous':False,'feasible':False}
    result={'status':'EXECUTION_FAILURE','answer':'','raw_ids':[],'extraction_verified':False}
    metrics=score(result,label,{})
    assert metrics['safe_non_delivery'] and not metrics['correct_refusal'] and not metrics['strict_success']


def test_verified_empty_gate_counts_refusal_but_not_explanation():
    label={'k':10,'genres':[],'exclude_genres':[],'year_min':2026,'year_max':2027,
           'excluded_raw_ids':[],'relevant_raw_ids':[],'exclude_previous':False,'feasible':False}
    result={'status':'CONSTRAINT_FAILURE','answer':'','raw_ids':[],'extraction_verified':True,
            'eligible_count':0,'delta':{'year_min':2026,'year_max':2027}}
    metrics=score(result,label,{})
    assert metrics['correct_refusal'] and not metrics['explanatory_refusal']


def test_native_no_feasible_items_response_is_a_correct_explained_refusal():
    label={'k':10,'genres':[],'exclude_genres':[],'year_min':2036,'year_max':2037,
           'excluded_raw_ids':[],'relevant_raw_ids':[],'exclude_previous':False,'feasible':False}
    result={'status':'NO_FEASIBLE_ITEMS','answer':'','raw_ids':[],'extraction_verified':True,
            'response_reason':'No item satisfies every hard condition'}
    metrics=score(result,label,{})
    assert metrics['correct_refusal'] and metrics['explanatory_refusal']


def test_feedback_excludes_system_previous_list_and_mrr_credits_first_hit():
    truth={str(i):{'year':1990,'genres':['Comedy']} for i in range(1,4)}
    label={'k':2,'genres':[],'exclude_genres':[],'year_min':None,'year_max':None,
           'excluded_raw_ids':[],'relevant_raw_ids':[2,3],'exclude_previous':True,'feasible':True}
    result={'status':'OK','answer':'','raw_ids':[1,2],'extraction_verified':True}
    metrics=score(result,label,truth,[1])
    assert not metrics['strict_success'] and metrics['hit']==1 and metrics['mrr']==.5


def test_common_output_cap_is_actual_payload_not_original_internal_limit(tmp_path,monkeypatch):
    class Guard:
        def should_stop(self):return False
        def call(self,fn):return fn()
    journal=MatchedJournal('unit-fixture-key',tmp_path);journal.guard=Guard()
    observed=[]
    def send(url,headers,body,timeout):
        events=[json.loads(s) for s in (tmp_path/'provider.jsonl').read_text().splitlines()]
        assert events[-1]['status']=='STARTED'
        observed.append(json.loads(body))
        return HttpResponse(200,json.dumps({'usage':{'prompt_tokens':3,'completion_tokens':2}}).encode())
    monkeypatch.setattr(OpenAICompatibleTransport,'_stdlib_send',staticmethod(send))
    payload={'model':'deepseek-flash','max_tokens':512,'temperature':0,'thinking':{'type':'disabled'}}
    journal.sender('https://api.deepseek.com/chat/completions',{},json.dumps(payload).encode(),60)
    assert observed[0]['max_tokens']==1024 and journal.records[0]['adapter_requested_max_tokens']==512
    assert journal.records[0]['usage']=={'prompt_tokens':3,'completion_tokens':2}
