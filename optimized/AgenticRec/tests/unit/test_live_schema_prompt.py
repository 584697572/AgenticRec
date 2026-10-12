import json
from agenticrec.adapters.llm import ChatAdapter
from agenticrec.agent.text_parser import TextRequestParser
from agenticrec.config import LLMBudget
from agenticrec.testing import FakeLLM
from agenticrec.runtime.errors import ResponseValidationError
import pytest


def test_parser_declares_version_type_and_same_exclude_seen_default_as_agent():
    fake=FakeLLM(['{}'])
    parser=TextRequestParser(ChatAdapter(fake,LLMBudget()))
    with pytest.raises(ResponseValidationError):
        parser.parse({'mode':'text','template_id':'development-1','message':'Suggest movies',
                      'user_id':None,'history_authorized':False},planned_cost_ceiling=0)
    prompt=fake.calls[0][0][1]
    assert 'schema_version is integer 1' in prompt
    assert 'exclude_seen true' in prompt
