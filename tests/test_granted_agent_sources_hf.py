"""Pinned actual-source full-content examples through both production tokenizers.

The inspection records retain complete rows and source digests. They are
format/renderer qualification inputs, not an acquisition/admission certificate.
"""
import base64
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest
from natlang_neuralese.common.paths import resolve
from scripts.neuralese_data import acquired_agents, acquired_granted as granted
from scripts.neuralese_data import build_acquired_full_session_graphs as builder
from scripts.neuralese_data.common import Reject

INSPECTION=resolve('data_hdd','natlang-development-data/granted-format-inspection-20261016/inspection.json')
BASES=[resolve('data_nvme','dgxops/qad-drift-probe/base'),resolve('models','mellum21-12b-a2.5b-thinking')]


def actual_examples(monkeypatch):
    receipt=json.loads((INSPECTION.parent/'transfer-receipt.json').read_text())
    raw=INSPECTION.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==receipt['source']['sha256']
    assert len(raw)==receipt['source']['bytes']
    inspection=json.loads(raw);examples=[]
    for record in inspection['records']:
        path=record['catalog_file']['path'];ref=record['ref']
        source={'repo':record['repo'],'revision':record['revision'],'card':{'license':'source-pinned'}}
        if 'full_raw_bytes_base64' in record:
            original=base64.b64decode(record['full_raw_bytes_base64'])
            assert len(original)==record['source_bytes']
            assert hashlib.sha256(original).hexdigest()==record['source_sha256']
            events=[(index,json.loads(line)) for index,line in enumerate(original.splitlines()) if line.strip()]
            contexts=list(granted.swe_raw_contexts(events))
            assert contexts,'actual full raw transcript must be supported'
            index,messages,evidence=min(contexts,key=lambda item:len(json.dumps(item[1])))
            t=granted._session(ref,source,path+':'+str(index),{},messages)
            t.messages=messages;t.lineage['notes']['original_swe_context']=evidence
            examples.append((record,path+':'+str(index),t))
        elif record.get('records'):
            row=record['records'][0]
            monkeypatch.setattr(acquired_agents,'acquired_rows',lambda actual,limit=None,r=record,v=row,s=source:iter([(s,r['catalog_file']['path']+':0',v)]) if actual==r['ref'] else iter([]))
            adapter={'D24':granted.miroverse_acquired,'D39':granted.midtool_acquired}.get(ref)
            if adapter is None:continue
            for rid,t in adapter():
                assert not isinstance(t,Reject),str(t)
                if t.lineage['notes'].get('source_quality_flags'):continue
                examples.append((record,rid,t))
    additional=INSPECTION.parent/'swe-additional-inspection.json'
    if additional.is_file():
        encoded=additional.read_bytes()
        assert hashlib.sha256(encoded).hexdigest()=='1967af1380935ee8916fb8bc87ff703de2a09eca23b844e991f66c8d6b6b2a04'
        extra=json.loads(encoded)
        for record in extra['examples']:
            if 'full_raw_bytes_base64' not in record:continue
            original=base64.b64decode(record['full_raw_bytes_base64'])
            assert len(original)==record['bytes'] and hashlib.sha256(original).hexdigest()==record['sha256']
            events=[(index,json.loads(line)) for index,line in enumerate(original.splitlines()) if line.strip()]
            if record['kind']=='codex':
                with pytest.raises(Reject):list(granted.swe_raw_contexts(events,source_id=record['file']))
                continue  # Actual upstream two-row test fixture has no request.
            contexts=list(granted.swe_raw_contexts(events,source_id=record['file']))
            assert contexts
            index,messages,evidence=min(contexts,key=lambda item:len(json.dumps(item[1])))
            if evidence.get('redacted_context_rows'):
                assert record['kind']=='transcripts_subagent'
                continue  # Missing encrypted source task is retained nonpositive.
            source={'repo':extra['repo'],'revision':extra['revision'],'card':{'license':'odc-by'}}
            rid=record['file']+':'+str(index)
            t=granted._session('D20',source,rid,{},messages);t.messages=messages
            adapted={'ref':'D20','catalog_file':{'path':record['file']},'source_sha256':record['sha256'],
                     'source_bytes':record['bytes'],'revision':extra['revision']}
            examples.append((adapted,rid,t))
    assert {record['ref'] for record,_,_ in examples}=={'D20','D24','D39'}
    assert {record['catalog_file']['path'].split('/')[0] for record,_,_ in examples if record['ref']=='D39'}=={'code','pdf','web','native-agent-traj'}
    return examples


@pytest.mark.parametrize('base',BASES,ids=['lfm','mellum'])
def test_actual_granted_examples_preserve_full_native_views_both_ports(base,monkeypatch):
    if not INSPECTION.is_file() or not (base/'tokenizer.json').is_file():
        pytest.skip('pinned inspection bundle or production tokenizer unavailable')
    prepared=builder.prepare_tokenizer(base)
    for source,rid,trajectory in actual_examples(monkeypatch):
        value={'schema':'natlang.normalized-agent-trajectory/1','whole_trajectory':True,'windowed':False,
            'positive_candidate':True,'source_row':rid,'source_file':{'path':source['catalog_file']['path'],
              'sha256':source['source_sha256'],'bytes':source['source_bytes']},
            'source_revision':source['revision'],'trajectory':asdict(trajectory)}
        graphs=list(builder.session_graphs(value,prepared.tokenizer,
            source_fingerprint=prepared.source_fingerprint,
            expanded_fingerprint=prepared.expanded_fingerprint,split=trajectory.split))
        coverage=graphs[0]['conversion']['acquired_session']
        assert graphs and coverage['original_messages']==len(trajectory.messages)
        assert coverage['original_assistant_turns']==sum(m['role']=='assistant' for m in trajectory.messages)
        assert graphs[0]['nodes'][0]['token_ids']
        assert any(w>0 for w in graphs[0]['nodes'][0]['target_weights'])
        assert all(graph['conversion']['acquired_session']['original_target_ids']==builder.source_target_ids(value) for graph in graphs)
