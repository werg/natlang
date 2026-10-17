"""Malformed self-analysis headers must not invent an external dispatch."""
import copy
import hashlib
import pytest

from tests.test_acquired_agent_sources import fixture
from neuralese_data import agents,acquired_agents
from neuralese_data.common import Reject


def raw_messages():
    return [{'role':'user','content':'Research the original question.'},
        {'role':'assistant','recipient':'assistant<|channel|>analysis','channel':'browser.open','content':'{"cursor":77,"id":11}'},
        {'role':'assistant','channel':'analysis','content':'That attempt did not open; try the original link.'},
        {'role':'assistant','recipient':'browser.open','channel':'analysis','content':'{"cursor":77,"id":11}'},
        {'role':'tool','name':'browser.open','content':'Exact original observation.'},
        {'role':'assistant','channel':'final','content':'Source answer.'}]


def test_self_analysis_preserves_text_original_positions_and_true_dispatch_identity():
    raw=raw_messages();before=copy.deepcopy(raw);proof=[]
    messages,_=agents.harmony_messages(raw,normalization_proof=proof)
    assert raw==before and len(messages)==len(raw)
    assert messages[1]=={'role':'assistant','content':None,'reasoning':raw[1]['content']}
    assert messages[3]['tool_calls'][0]['id']=='original-call-3'
    assert messages[4]['tool_call_id']=='original-call-3'
    assert messages[4]['content']=='Exact original observation.'
    assert proof==[{'index':1,'kind':'encoded-self-analysis-recipient',
        'original':{'recipient':'assistant<|channel|>analysis','channel':'browser.open'},
        'normalized':{'recipient':'assistant','channel':'analysis'},
        'content_text_sha256':hashlib.sha256(raw[1]['content'].encode()).hexdigest(),
        'original_content_canonical_sha256':hashlib.sha256(agents.canonical_json(raw[1]['content']).encode()).hexdigest()}]


def test_source_adapter_retains_normalization_witness_and_original_channel_metadata(tmp_path,monkeypatch):
    raw=raw_messages();row={'qid':11,'chunk_idx':3,'num_chunks':8,'status':'success','messages':raw}
    fixture(tmp_path,monkeypatch,'D32',[row])
    _,trajectory=next(acquired_agents.openresearcher_structured_acquired())
    assert not isinstance(trajectory,Reject)
    notes=trajectory.lineage['notes']['original_research_channels']
    assert notes['messages'][1]['recipient']=='assistant<|channel|>analysis'
    assert notes['messages'][1]['channel']=='browser.open'
    assert notes['header_normalizations'][0]['index']==1
    assert trajectory.outcome['checked']is None and trajectory.outcome['details']['executed_locally']is False
    assert len(trajectory.messages)==len(raw)


@pytest.mark.parametrize('change',['orphan','conflicting-explicit-call','wrong-observation'])
def test_true_missing_or_conflicting_dispatches_remain_rejected(change):
    raw=raw_messages()
    if change=='orphan':del raw[3]
    elif change=='conflicting-explicit-call':raw[1]['tool_calls']=[{'function':{'name':'browser.open','arguments':'{}'}}]
    else:raw[4]['name']='browser.search'
    with pytest.raises(Reject):agents.harmony_messages(raw)


def test_other_recipients_and_ordinary_analysis_do_not_acquire_header_repair():
    raw=raw_messages();raw[1]={'role':'assistant','channel':'analysis','content':'ordinary source reasoning'}
    proof=[];messages,_=agents.harmony_messages(raw,normalization_proof=proof)
    assert proof==[] and messages[1]['reasoning']=='ordinary source reasoning'
    assert messages[3]['tool_calls'][0]['function']['name']=='browser.open'


def test_structured_source_content_is_bound_separately_from_rendered_text():
    raw=raw_messages();text=raw[1]['content']
    raw[1]['content']=[{'type':'text','text':text,'unused_original_metadata':None}]
    proof=[];messages,_=agents.harmony_messages(raw,normalization_proof=proof)
    assert messages[1]['reasoning']==text
    assert proof[0]['content_text_sha256']==hashlib.sha256(text.encode()).hexdigest()
    assert proof[0]['original_content_canonical_sha256']==hashlib.sha256(agents.canonical_json(raw[1]['content']).encode()).hexdigest()
    assert proof[0]['content_text_sha256']!=proof[0]['original_content_canonical_sha256']
