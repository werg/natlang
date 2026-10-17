"""Transitive source siblings remain protected across exports and machines."""
import json
import os
from pathlib import Path
import shutil

import pytest

from natlang_neuralese.common.hashing import sha256_file_hex
from natlang_neuralese.common.jsonio import write_jsonl_text
from scripts.neuralese_data import build_verified_teacher_trajectory_graphs as teacher
from scripts.neuralese_data.close_acquired_source_protection import project
from scripts.neuralese_data import prepare_acquired_session_admission as admission
from scripts.neuralese_data.prepare_acquired_session_admission import verify_global_source_closure
from scripts.neuralese_data.continue_acquired_protection import verify_protection_transition
from scripts.neuralese_data import write_acquired_source_closure_receipt as closure_writer
from scripts.neuralese_data.conversion_output_lease import acquire_output_lease
from scripts.neuralese_data.splits import close_group_sets


def export(root, rows):
    directory=root/'one';directory.mkdir(parents=True)
    source=directory/'full-trajectories.jsonl.zst'
    with write_jsonl_text(source) as stream:
        for groups,split in rows:
            stream.write(json.dumps({'trajectory':{'split':split,'split_groups':groups}})+'\n')
    receipt=directory/'export.json'
    receipt.write_text(json.dumps({'outputs':{source.name:sha256_file_hex(source)},'complete_sessions':len(rows)}))
    (root/'complete.json').write_text(json.dumps({'files':[{'status':'reused','receipt':str(receipt)}]}))
    return {'path':str(receipt),'sha256':sha256_file_hex(receipt),'source_sha256':sha256_file_hex(source)}


def base(tmp_path, monkeypatch):
    review=tmp_path/'reviews';review.mkdir();(review/'holds.jsonl').write_text('')
    monkeypatch.setattr(teacher.source_review,'REGISTRY',review)
    monkeypatch.setattr(teacher.source_review,'pending_unpinned_source_groups',lambda:{'P1'})
    root=tmp_path/'base';root.mkdir();files=[]
    for relative,groups in [('held-union/seeds.jsonl',['C']),('quality-groups.jsonl',['Q1'])]:
        path=root/relative;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps({'source_groups':groups})+'\n')
        files.append({'path':relative,'sha256':sha256_file_hex(path),'bytes':path.stat().st_size})
    binding=export(tmp_path/'sources'/'D16',[
        (['A','B'],'train'),(['B','C'],'train'),(['C'],'test'),
        (['Q1','Q2'],'train'),(['P1','P2'],'train')])
    path=root/'protection-manifest.json';path.write_text(json.dumps({
        'schema':'natlang.training-protection-projection/1','files':files,
        'held_groups':1,'quality_groups':1,'source_review_sha256':sha256_file_hex(review/'holds.jsonl'),
        'normalized_export_bindings':[binding]}))
    return path


def test_closure_is_order_independent_and_covers_each_protected_set():
    rows=[['A','B'],['B','C'],['Q1','Q2'],['P1','P2']]
    seeds={'held':{'C'},'quality':{'Q1'},'pending':{'P1'}}
    closed,components=close_group_sets(rows,seeds)
    assert closed=={'held':{'A','B','C'},'quality':{'Q1','Q2'},'pending':{'P1','P2'}}
    assert close_group_sets(reversed(rows),seeds)==(closed,components)


def test_publish_then_extend_without_original_normalized_text(tmp_path,monkeypatch):
    original=base(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='connected-group closure'):
        verify_global_source_closure(original,teacher.current_protection(original))
    first=project(original,tmp_path/'closed')
    protected=teacher.current_protection(first)
    assert protected['held']=={'A','B','C'} and protected['quality']=={'Q1','Q2'}
    assert protected['pending']=={'P1','P2'}
    assert verify_global_source_closure(first,protected)['normalized_sessions']==5
    extra=tmp_path/'new-machine'/'D17'
    export(extra,[(['B','D'],'train'),(['P2','P3'],'train')])
    # The qualified prior component witness, not copied text, carries history.
    shutil.rmtree(tmp_path/'sources')
    second=project(first,tmp_path/'extended',exports=[extra])
    projected=teacher.current_protection(second)
    assert projected['held']=={'A','B','C','D'} and projected['pending']=={'P1','P2','P3'}
    assert verify_global_source_closure(second,projected)['normalized_sessions']==2
    monkeypatch.setattr(teacher.source_review,'pending_unpinned_source_groups',lambda:{'B'})
    assert teacher.current_protection(second)['pending']=={'A','B','C','D'}
    with pytest.raises(ValueError,match='immutable'):project(first,tmp_path/'extended',exports=[extra])
    component=second.parent/'inherited-components.jsonl';component.write_text('{}\n')
    with pytest.raises(ValueError,match='differs'):teacher.current_protection(second)


def test_closure_receipt_writer_force_mode_hashes_without_cache(tmp_path,monkeypatch):
    projection=project(base(tmp_path,monkeypatch),tmp_path/'closed')
    closure_calls=[]
    artifact_calls=[]
    real_closure=admission.acquired_source_closure
    real_artifacts=admission._source_closure_artifacts
    def observe_closure(path,protection,*,use_cache=True):
        closure_calls.append(use_cache)
        return real_closure(path,protection,use_cache=use_cache)
    def observe_artifacts(path,manifest,*,use_cache=True):
        artifact_calls.append(use_cache)
        return real_artifacts(path,manifest,use_cache=use_cache)
    monkeypatch.setattr(admission,'acquired_source_closure',observe_closure)
    monkeypatch.setattr(admission,'_source_closure_artifacts',observe_artifacts)
    output=tmp_path/'closure-receipt.json'
    protection_calls=[]
    real_current=closure_writer.current_protection
    def observe_current(path):
        protection_calls.append(path)
        return real_current(path)
    monkeypatch.setattr(closure_writer,'current_protection',observe_current)
    with acquire_output_lease(tmp_path):
        with pytest.raises(RuntimeError,match='already owns'):
            closure_writer.main(['--protection',str(projection),'--output',str(output),
                                 '--force-full-validation'])
    assert protection_calls==[]
    closure_writer.main(['--protection',str(projection),'--output',str(output),
                         '--force-full-validation'])
    assert protection_calls==[projection]
    assert closure_calls==[False]
    assert artifact_calls==[False]
    receipt=json.loads(output.read_text())
    assert receipt['training_admission'] is False
    assert receipt['accounting']['normalized_sessions']==5
    before=sha256_file_hex(output)
    with pytest.raises(ValueError,match='immutable'):
        closure_writer.main(['--protection',str(projection),'--output',str(output),
                             '--force-full-validation'])
    assert sha256_file_hex(output)==before


def test_extension_resolves_declared_receipts_through_verified_mount_alias(tmp_path,monkeypatch):
    original=base(tmp_path,monkeypatch)
    first=project(original,tmp_path/'closed')
    extra=tmp_path/'new-machine'/'D17'
    export(extra,[(['B','D'],'train'),(['P2','P3'],'train')])
    actual=extra/'one'/'export.json'
    alias=tmp_path/'normalized-alias'
    alias.symlink_to(extra,target_is_directory=True)
    complete=json.loads((extra/'complete.json').read_text())
    complete['files'][0]['receipt']=str(alias/'one'/'export.json')
    (extra/'complete.json').write_text(json.dumps(complete))

    second=project(first,tmp_path/'extended',exports=[extra])
    projected=teacher.current_protection(second)
    assert projected['held']=={'A','B','C','D'}
    assert projected['pending']=={'P1','P2','P3'}
    assert verify_global_source_closure(second,projected)['normalized_sessions']==2


def test_durable_source_closure_receipt_reuses_only_exact_current_inputs(tmp_path,monkeypatch):
    original=base(tmp_path,monkeypatch)
    first=project(original,tmp_path/'closed')
    protected=teacher.current_protection(first)
    receipt=tmp_path/'closure-receipt.json'
    expected=verify_global_source_closure(first,protected,receipt_path=receipt)
    assert receipt.exists()
    monkeypatch.setattr(admission,'sessions',lambda *_:pytest.fail('receipt hit must not reparse global sessions'))
    assert verify_global_source_closure(first,protected,receipt_path=receipt)==expected
    assert verify_protection_transition(first,first,closure_receipt_path=receipt)[2]==expected
    source=tmp_path/'sources'/'D16'/'one'/'full-trajectories.jsonl.zst'
    with source.open('ab') as stream:stream.write(b'changed')
    with pytest.raises(ValueError,match='immutable|input inventory|acquired closure artifact changed'):
        verify_global_source_closure(first,protected,receipt_path=receipt)


def test_force_full_closure_validation_ignores_cached_receipt(tmp_path,monkeypatch):
    original=base(tmp_path,monkeypatch)
    first=project(original,tmp_path/'closed')
    protected=teacher.current_protection(first)
    receipt=tmp_path/'closure-receipt.json'
    expected=verify_global_source_closure(first,protected,write_receipt_path=receipt)
    value=json.loads(receipt.read_text());value['accounting']['normalized_sessions']=999
    receipt.write_text(json.dumps(value))
    monkeypatch.setattr(admission,'sha256_file_hex_cached',lambda *_args,**_kwargs:pytest.fail('forced validation must not consult hash cache'))
    assert verify_global_source_closure(first,protected,receipt_path=receipt,
        force_full_validation=True)==expected
    with pytest.raises(ValueError,match='receipts are immutable'):
        verify_global_source_closure(first,protected,write_receipt_path=receipt,
            force_full_validation=True)


def test_cached_closure_receipt_rejects_same_size_source_rewrite_with_restored_mtime(tmp_path,monkeypatch):
    original=base(tmp_path,monkeypatch)
    first=project(original,tmp_path/'closed')
    protected=teacher.current_protection(first)
    receipt=tmp_path/'closure-receipt.json'
    verify_global_source_closure(first,protected,write_receipt_path=receipt)
    source=tmp_path/'sources'/'D16'/'one'/'full-trajectories.jsonl.zst'
    status=source.stat();data=source.read_bytes()
    replacement=bytes([data[0]^1])+data[1:]
    source.write_bytes(replacement)
    os.utime(source,ns=(status.st_atime_ns,status.st_mtime_ns))
    with pytest.raises(ValueError,match='acquired closure artifact changed|input inventory'):
        verify_global_source_closure(first,protected,receipt_path=receipt)


def test_durable_source_closure_receipt_rejects_changed_current_hold_sets(tmp_path,monkeypatch):
    original=base(tmp_path,monkeypatch)
    first=project(original,tmp_path/'closed')
    protected=teacher.current_protection(first)
    receipt=tmp_path/'closure-receipt.json'
    verify_global_source_closure(first,protected,write_receipt_path=receipt)
    changed={**protected,'held':protected['held']|{'new-held'}}
    with pytest.raises(ValueError,match='stale for current protection'):
        verify_global_source_closure(first,changed,receipt_path=receipt)


def test_durable_source_closure_receipt_pins_implementation_dependencies(tmp_path,monkeypatch):
    original=base(tmp_path,monkeypatch)
    first=project(original,tmp_path/'closed')
    protected=teacher.current_protection(first)
    receipt=tmp_path/'closure-receipt.json'
    verify_global_source_closure(first,protected,write_receipt_path=receipt)
    value=json.loads(receipt.read_text())
    value['implementation_files']['connected_group_closure']['sha256']='0'*64
    receipt.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='implementation dependency changed'):
        verify_global_source_closure(first,protected,receipt_path=receipt)
