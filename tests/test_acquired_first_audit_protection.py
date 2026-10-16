"""Actual both-HF old numeric receipts receive a new complete source audit."""
import json
from pathlib import Path
import pytest
from natlang_neuralese.common.hashing import sha256_file_hex
from natlang_neuralese.common.jsonio import write_jsonl_text
from scripts.neuralese_data import build_acquired_full_session_graphs as builder
from scripts.neuralese_data import prepare_acquired_session_admission as admission
from scripts.neuralese_data.continue_acquired_protection import certify, verify_certificate
from scripts.neuralese_data.close_acquired_source_protection import project
from scripts.neuralese_data.qualify_acquired_session_samples import qualify
from test_prepared_acquired_tokenizers import normalized, BASES
from test_acquired_protection_continuation import fixture, write


def test_first_complete_audit_uses_current_closure_and_exact_old_numeric_both_ports(tmp_path,monkeypatch):
    if any(not (base/'tokenizer.json').is_file() for base in BASES):pytest.skip('actual pinned tokenizers unavailable')
    _,old,closed,_=fixture(tmp_path,monkeypatch)
    protection=admission.current_protection(old)
    original=tmp_path/'original'/'source.json';original.parent.mkdir();original.write_text('{}\n')
    original_binding={'path':'source.json','sha256':sha256_file_hex(original),'bytes':original.stat().st_size}
    rows=[normalized('source-session'),normalized('failed-session')]
    for row in rows:row['source_file']=original_binding
    rows[0]['trajectory']['split_groups']=['SAFE']
    rows[1]['trajectory']['split_groups']=['FAIL'];rows[1]['positive_candidate']=False
    rows[1]['trajectory']['outcome']['label']='failed'
    exports=tmp_path/'exports'/'D15';source=exports/'file-one'/'full-trajectories.jsonl.zst';source.parent.mkdir(parents=True)
    with write_jsonl_text(source) as stream:
        for row in rows:stream.write(json.dumps(row)+'\n')
    export=write(source.parent/'export.json',{'schema':'natlang.normalized-agent-source-file/1',
        'contract':{'source':original_binding,'revision':'pinned'},'outputs':{source.name:sha256_file_hex(source)},'complete_sessions':2})
    write(exports/'complete.json',{'files':[{'status':'exported','receipt':str(export)}]})
    current=project(closed,tmp_path/'current',exports=[exports])
    graphroot=tmp_path/'graphs';holds=sorted((Path(builder.__file__).resolve().parents[2]/'training/decision-holds').glob('*.jsonl'))
    monkeypatch.setattr(builder,'current_protection',lambda path:protection)
    models=dict(zip(('lfm25','mellum21'),BASES))
    for model,base in models.items():
        argv=['--export-receipt',str(export),'--base-tokenizer',str(base),'--protection',str(old),
              '--output',str(graphroot/'file-one'/model)]
        for hold in holds:argv+=['--decision-hold',str(hold)]
        builder.main(argv)
    numeric={str(path):sha256_file_hex(path) for path in graphroot.rglob('*') if path.is_file() and path.name!='.conversion.lock'}
    common=['--exports',str(exports),'--native-root',str(graphroot),'--registry-path','runs/test-D15',
            '--original-root',str(original.parent)]
    for model,base in models.items():common+=['--model',model+'='+str(base)]
    # No prior audit is fabricated, and the ordinary current-closure gate stays strict.
    with pytest.raises(ValueError,match='connected-group closure'):
        admission.main(common+['--protection',str(old),'--id','old-is-not-current'])
    admission.main(common+['--protection',str(current),'--conversion-protection',str(old),'--id','current-eligible'])
    directory=graphroot/'admission-current-eligible';audit=directory/'audit.json';value=json.loads(audit.read_text())
    transition=value['conversion_protection_transition']
    assert transition['all_original_source_records_checked']==2
    assert transition['prior_conversion_protection']==protection['proof']
    assert transition['eligibility_protection']==admission.current_protection(current)['proof']
    assert not transition['numeric_rerendered'] and transition['original_conversion_receipts_unchanged']
    assert value['normalized_sessions']==2 and value['train_held_group_intersections']==0
    assert all(item['conversion_protection']==protection['proof'] for item in value['conversions'])
    assert {path:sha256_file_hex(path) for path in numeric}==numeric
    sample=qualify(audit,models,directory/'source-samples.json')
    assert sample['passed'] and len(sample['samples'])==2
    assert all(report['exact_entire_graph_equal'] for report in sample['samples'])
    # Future continuation understands old numerical protection under the new audit.
    certificate=directory/'continued.json';certify(audit,current,current,certificate,decision_paths=holds)
    assert verify_certificate(certificate)['passed']
    # Even an omitted failed source must retain its original split, not only positive rows.
    current_state=admission.current_protection(current)
    changed={**current_state,'held':current_state['held']|{'FAIL'}}
    ids,bindings=admission.load_decision_holds(holds)
    with pytest.raises(ValueError,match='original source split'):
        admission.verify_conversion(graphroot/'file-one'/'lfm25',export,protection=changed,
            conversion_protection=protection,held_sources=ids,decision_bindings=bindings,source_rows=rows)
