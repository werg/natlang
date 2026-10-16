"""Transitive source siblings remain protected across exports and machines."""
import json
from pathlib import Path
import shutil

import pytest

from natlang_neuralese.common.hashing import sha256_file_hex
from natlang_neuralese.common.jsonio import write_jsonl_text
from scripts.neuralese_data import build_verified_teacher_trajectory_graphs as teacher
from scripts.neuralese_data.close_acquired_source_protection import project
from scripts.neuralese_data.prepare_acquired_session_admission import verify_global_source_closure
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
