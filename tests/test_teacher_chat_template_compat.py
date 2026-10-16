"""Use the actual pinned vendor renderer, without loading model weights."""
import hashlib
import json
from pathlib import Path

import pytest
from transformers.utils.chat_template_utils import _compile_jinja_template
from scripts.coreweave.chat_template_compat import preserve_ordered_system_messages

FIXTURE=Path(__file__).parent/'fixtures/qwen38-flash-next-chat-template.jinja'


def render(template,messages,**kwargs):
    return _compile_jinja_template(template).render(messages=messages,add_generation_prompt=True,
        **kwargs)


@pytest.mark.parametrize('thinking',[False,True])
@pytest.mark.parametrize('tools',[None,[{'type':'function','function':{'name':'return_result',
    'parameters':{'type':'object','properties':{'value':{'type':'string'}}}}}]])
@pytest.mark.parametrize('system',[None,'','Original system contract.'])
def test_every_originally_valid_rendering_is_byte_identical(thinking,tools,system):
    source=FIXTURE.read_text();provenance=json.loads(FIXTURE.with_suffix('.source.json').read_text())
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest()==provenance['sha256']
    messages=([{'role':'system','content':system}] if system is not None else [])+[
        {'role':'user','content':'Original user question Ω.'},
        {'role':'assistant','content':'','reasoning_content':'Actual recorded reasoning.',
         'tool_calls':[{'function':{'name':'return_result','arguments':{'value':'actual value'}}}]},
        {'role':'tool','content':'Actual tool observation.'}]
    derived=preserve_ordered_system_messages(source)
    assert render(source,messages,tools=tools,enable_thinking=thinking)==render(
        derived,messages,tools=tools,enable_thinking=thinking)


@pytest.mark.parametrize('position',[1,2,3,4])
def test_later_system_messages_keep_exact_roles_text_and_position(position):
    source=FIXTURE.read_text();messages=[{'role':'system','content':'Initial contract.'},
        {'role':'user','content':'User question.'},{'role':'assistant','content':'Attempt.'},
        {'role':'tool','content':'Observation.'}]
    messages.insert(position,{'role':'system','content':[{'type':'text','text':'Recovery Ω.'}]})
    with pytest.raises(Exception,match='System message must be at the beginning'):
        render(source,messages,enable_thinking=False)
    actual=render(preserve_ordered_system_messages(source),messages,enable_thinking=False)
    assert actual.count('<|im_start|>system\nRecovery Ω.<|im_end|>\n')==1
    offsets=[actual.index(text) for text in [
        item['content'] if isinstance(item['content'],str) else item['content'][0]['text']
        for item in messages]]
    assert offsets==sorted(offsets)


def test_system_content_validation_is_preserved_and_unknown_source_fails():
    source=FIXTURE.read_text()
    with pytest.raises(ValueError,match='pinned later-system guard'):
        preserve_ordered_system_messages(source.replace('System message must','Other message must'))
    with pytest.raises(Exception,match='System message cannot contain images'):
        render(preserve_ordered_system_messages(source),[
            {'role':'user','content':'Actual user question.'},
            {'role':'system','content':[{'type':'image','image':'irrelevant'}]}])
