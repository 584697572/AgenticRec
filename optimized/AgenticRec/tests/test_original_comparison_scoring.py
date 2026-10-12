from agenticrec.evaluation.comparison_scoring import AnswerIdentity, score_turn

TRUTH={1:{'title':'Alien','year':1979,'genres':['Horror']},
       2:{'title':'Aliens','year':1986,'genres':['Action']},
       3:{'title':'Sabrina','year':1954,'genres':['Comedy']},
       4:{'title':'Sabrina','year':1995,'genres':['Comedy']}}


def test_no_fuzzy_substring_identity():
    parsed=AnswerIdentity(TRUTH).extract('1. Aliens (1986)',[2])
    assert parsed['raw_ids']==[2]
    assert parsed['raw_ids']!=[1]


def test_ambiguous_remake_needs_year():
    index=AnswerIdentity(TRUTH)
    assert index.resolve('Sabrina') is None
    assert index.resolve('Sabrina (1995)')==4


def test_verified_map_disambiguates_displayed_title_but_not_wrong_year():
    index=AnswerIdentity(TRUTH)
    assert index.extract('1. Sabrina',[4])['raw_ids']==[4]
    assert index.extract('1. Sabrina',[3,4])['raw_ids']==[None]
    assert index.extract('1. Sabrina (2000)',[4])['raw_ids']==[None]


def test_unknown_distinct_titles_are_not_reported_as_duplicates():
    result={'status':'OK','extraction_verified':True,'raw_ids':[None,None]}
    label={'k':2,'genres':[],'year_min':None,'year_max':None,
           'excluded_raw_ids':[],'relevant_raw_ids':[]}
    score=score_turn(result,label,TRUTH)
    assert score['duplicate_count']==0 and score['identity_errors']==2


def test_final_answer_omission_and_unknown_are_retained():
    value=AnswerIdentity(TRUTH).extract('1. Alien (1979)\n2. Imaginary Movie (2000)',[1,2])
    assert value['raw_ids']==[1,None]
    assert value['unresolved_entries']==1


def test_tools_without_delivered_titles_do_not_pass():
    value=AnswerIdentity(TRUTH).extract('Here are some suggestions.',[1,2])
    assert value['raw_ids']==[]
    assert not value['extraction_verified']


def test_duplicates_do_not_inflate_quality():
    result={'status':'OK','extraction_verified':True,'raw_ids':[2,2]}
    label={'k':2,'genres':['Action'],'year_min':None,'year_max':None,
           'excluded_raw_ids':[],'relevant_raw_ids':[2]}
    score=score_turn(result,label,TRUTH)
    assert not score['strict_success']
    assert score['recall']==1 and score['ndcg']==1


def test_error_delivery_stays_quality_denominator_as_zero():
    result={'status':'ERROR','extraction_verified':True,'raw_ids':[2]}
    label={'k':1,'genres':[],'year_min':None,'year_max':None,
           'excluded_raw_ids':[],'relevant_raw_ids':[2]}
    score=score_turn(result,label,TRUTH)
    assert score['quality_evaluable'] and score['hit']==0
    assert not score['strict_success']


def test_year_and_seen_constraints_fail():
    result={'status':'OK','extraction_verified':True,'raw_ids':[2]}
    label={'k':1,'genres':[],'year_min':1990,'year_max':2000,
           'excluded_raw_ids':[2],'relevant_raw_ids':[]}
    assert score_turn(result,label,TRUTH)['constraint_precision']==0


def test_bold_order_preserves_actual_rank():
    value=AnswerIdentity(TRUTH).extract('Try **Aliens** followed by **Alien**.',[1,2])
    assert value['raw_ids']==[2,1]


def test_unstarted_benchmark_is_not_a_zero_success_result():
    from agenticrec.evaluation.comparison_run import summarize
    result=summarize([], [{'episode_id':'not-started'}], [], 'calibration')
    assert result['systems']['original']['strict_success'] is None
    assert result['systems']['optimized']['strict_success'] is None


def test_provider_intent_precedes_delivery_and_no_headers_are_saved(tmp_path,monkeypatch):
    import json
    from agenticrec.evaluation.comparison_run import HTTPJournal
    from agenticrec.adapters.openai_compatible import OpenAICompatibleTransport, HttpResponse
    journal=HTTPJournal('unit-test-key',tmp_path)
    journal.context={'system':'original','episode_id':'test','turn':1}
    monkeypatch.setattr(journal.guard.client,'fetch',lambda:{'is_available':True,'balance_infos':[]})

    def sender(url,headers,body,timeout):
        intent=json.loads((tmp_path/'provider.jsonl').read_text().splitlines()[0])
        assert intent['status']=='STARTED'
        assert 'headers' not in intent
        return HttpResponse(200,json.dumps({'choices':[{'message':{'content':'OK'}}],
            'usage':{'prompt_tokens':2,'completion_tokens':1,'total_tokens':3}}).encode())

    monkeypatch.setattr(OpenAICompatibleTransport,'_stdlib_send',staticmethod(sender))
    body=json.dumps({'model':'deepseek-flash','max_tokens':512,'temperature':0,
                     'thinking':{'type':'disabled'},'messages':[]}).encode()
    journal.sender('https://api.deepseek.com/chat/completions',{},body,60)
    assert journal.records[0]['usage']['total_tokens']==3
    assert 'unit-test-key' not in (tmp_path/'provider.jsonl').read_text()


def test_insufficient_balance_latches_without_retry(tmp_path,monkeypatch):
    import json
    import pytest
    from agenticrec.evaluation.comparison_run import HTTPJournal
    from agenticrec.adapters.account_balance import BalanceStop
    from agenticrec.adapters.openai_compatible import OpenAICompatibleTransport, HttpResponse
    from agenticrec.runtime.errors import LLMTransportError
    journal=HTTPJournal('unit-test-key',tmp_path)
    journal.context={'system':'original','episode_id':'test','turn':1}
    monkeypatch.setattr(journal.guard.client,'fetch',lambda:{'is_available':True,'balance_infos':[]})
    monkeypatch.setattr(OpenAICompatibleTransport,'_stdlib_send',staticmethod(lambda *args:HttpResponse(402,b'{}')))
    body=json.dumps({'model':'deepseek-flash','max_tokens':512,'temperature':0,
                     'thinking':{'type':'disabled'},'messages':[]}).encode()
    with pytest.raises(LLMTransportError):
        journal.sender('https://api.deepseek.com/chat/completions',{},body,60)
    with pytest.raises(BalanceStop):
        journal.sender('https://api.deepseek.com/chat/completions',{},body,60)
    assert len(journal.records)==1
    assert journal.records[0]['http_status']==402


def test_token_limit_rejects_before_network_intent(tmp_path):
    import json
    import pytest
    from agenticrec.evaluation.comparison_run import HTTPJournal
    journal=HTTPJournal('unit-test-key',tmp_path)
    body=json.dumps({'model':'deepseek-flash','max_tokens':1025,'temperature':0,
                     'thinking':{'type':'disabled'},'messages':[]}).encode()
    with pytest.raises(ValueError):
        journal.sender('https://api.deepseek.com/chat/completions',{},body,60)
    assert not journal.records
    assert not (tmp_path/'provider.jsonl').exists()


def test_optimized_wiring_reaches_transport_with_positive_reservation(tmp_path,monkeypatch):
    import json
    from agenticrec.evaluation.comparison_run import HTTPJournal, OptimizedSystem
    from agenticrec.adapters.openai_compatible import OpenAICompatibleTransport, HttpResponse
    from agenticrec.fusion_pipeline import FusionPipeline
    from agenticrec.pipeline import FixedResponse

    class Pipeline:
        def recommend(self,request):
            return FixedResponse('OK',None,[{'item_id':1,'title':'Alien (1979)',
                'genres':['Horror'],'year':1979}], 'session',None,{},[],0)

    monkeypatch.setattr(FusionPipeline,'from_frozen',classmethod(lambda cls,*args,**kwargs:Pipeline()))
    journal=HTTPJournal('unit-test-key',tmp_path)
    journal.context={'system':'optimized','episode_id':'test','turn':1}
    balance={'is_available':True,'balance_infos':[]}
    monkeypatch.setattr(journal.guard.client,'fetch',lambda:balance)
    request={'schema_version':1,'user_id':None,'history_authorized':False,
             'liked_item_ids':[],'disliked_item_ids':[],'seen_item_ids':[],
             'constraints':{'include_genres':[],'exclude_genres':[],'year_min':None,
                           'year_max':None,'excluded_item_ids':[],'exclude_seen':True,
                           'k':1,'required_fields':[]}}
    plan={'plan':[{'step_id':'1','tool_name':'recommend','arguments':{'request':request}}]}
    reply={'choices':[{'message':{'content':json.dumps(plan)}}],
           'usage':{'prompt_tokens':2,'completion_tokens':1,'total_tokens':3}}
    monkeypatch.setattr(OpenAICompatibleTransport,'_stdlib_send',staticmethod(lambda *args:HttpResponse(200,json.dumps(reply).encode())))
    system=OptimizedSystem(journal,balance)
    result=system.turn({'episode_id':'test','messages':['Recommend one movie.'],'catalog_context':[]},0)
    assert result['status']=='OK'
    assert len(journal.records)==1
    assert system.loop.planner.ledger.snapshot()['attempts']==1
