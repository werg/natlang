"""Exact auxiliary source events must never become positive actions."""
import copy
import hashlib
import json

import pytest

from scripts.neuralese_data.acquired_granted import swe_raw_contexts
from scripts.neuralese_data.common import Reject


def snapshot():
    return {'type':'response_item','payload':{'type':'ghost_snapshot','ghost_commit':{
        'id':'a'*40,'parent':'b'*40,'preexisting_untracked_files':['literal.txt'],
        'preexisting_untracked_dirs':[]}}}


def records(aux):
    return list(enumerate([
        {'type':'response_item','payload':{'type':'message','role':'user','content':'Original request'}},
        aux,
        {'type':'response_item','payload':{'type':'function_call','call_id':'call','name':'inspect','arguments':'{}'}},
        {'type':'response_item','payload':{'type':'function_call_output','call_id':'call','output':'Observed exact value'}},
        {'type':'response_item','payload':{'type':'message','role':'assistant','content':'Original answer'}}]))


def test_snapshot_is_bound_auxiliary_evidence_not_target_or_tool_result():
    rows=records(snapshot());before=copy.deepcopy(rows)
    representative,messages,evidence=next(swe_raw_contexts(rows,source_id='exact-source'))
    assert rows==before and representative==4
    assert [m['role'] for m in messages]==['user','assistant','tool','assistant']
    assert messages[-1]['content']=='Original answer'
    assert messages[1]['tool_calls'][0]['id']=='call'
    assert messages[2]['content']=='Observed exact value'
    assert evidence['original_message_rows']==[0,2,3,4]
    assert evidence['missing_ancestry']=={}
    assert evidence['auxiliary_source_rows']==[{'row':1,'kind':'ghost_snapshot',
        'payload_sha256':hashlib.sha256(json.dumps(rows[1][1]['payload'],sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()}]


@pytest.mark.parametrize('kind',['thread_rolled_back','turn_aborted'])
def test_history_boundary_remains_unresolved_for_existing_positive_quality_gate(kind):
    _,messages,evidence=next(swe_raw_contexts(records({'type':'event_msg','payload':{'type':kind,'num_turns':1}}),source_id='exact-source'))
    assert messages[-1]['content']=='Original answer'
    assert evidence['missing_ancestry']['unresolved_codex_history_events'][0]['kind']==kind
    assert evidence['original_message_rows']==[0,2,3,4]


@pytest.mark.parametrize('mutation',[
    lambda p:p.update(content='Never silently discard model-visible content'),
    lambda p:p['ghost_commit'].update(id='bad'),
    lambda p:p['ghost_commit'].update(parent=123),
    lambda p:p['ghost_commit'].update(preexisting_untracked_files='file'),
    lambda p:p['ghost_commit'].update(preexisting_untracked_dirs=[{}]),
    lambda p:p['ghost_commit'].update(unrecognized='value'),
])
def test_malformed_or_extended_snapshot_is_not_silently_ignored(mutation):
    aux=snapshot();mutation(aux['payload'])
    with pytest.raises(Reject,match='SWE Codex ghost'):
        next(swe_raw_contexts(records(aux),source_id='exact-source'))


def test_unknown_response_item_still_rejected():
    with pytest.raises(Reject,match='unsupported original SWE Codex response item'):
        next(swe_raw_contexts(records({'type':'response_item','payload':{'type':'unknown'}}),source_id='exact-source'))
