import hashlib,json
from pathlib import Path
import pytest
from natlang_neuralese.common import custody
from natlang_neuralese.common.runtime_inventory import assemble_protected_runtime_inputs


def test_assembly_inherits_only_selected_packets_and_checks_controls(tmp_path, monkeypatch):
    old=tmp_path/'old'; new=tmp_path/'new'
    for root in (old,new):
        (root/'training').mkdir(parents=True)
        (root/'training/protected-worlds.json').write_text('{}')
    registry={'corpora':[{'id':'held','admission':'evaluation-only-never-training','path':'runs/held','include':['cases.jsonl']},
        {'id':'other','admission':'train','path':'runs/other','include':['cases.jsonl']}]}
    (new/'training/neuralese_corpora.json').write_text(json.dumps(registry))
    (old/'runs/held').mkdir(parents=True)
    (old/'runs/held/cases.jsonl').write_text('held rows')
    control=tmp_path/'prompt.json';control.write_text('exact prompt')
    digest=hashlib.sha256(control.read_bytes()).hexdigest()
    class Worlds:
        digest='worlds'
    monkeypatch.setattr(custody.ProtectedWorlds,'load',lambda **kwargs:Worlds())
    result=assemble_protected_runtime_inputs(old,new,bound_files=[{'path':str(control),'sha256':digest}])
    assert (new/'runs/held/cases.jsonl').read_text()=='held rows'
    assert not (new/'runs/other').exists()
    assert result['protected_worlds_sha256']=='worlds' and len(result['evaluation_packets'])==1
    (new/'runs/held/cases.jsonl').write_text('altered')
    with pytest.raises(ValueError,match='differs'):
        assemble_protected_runtime_inputs(old,new)


def test_assembly_rejects_packet_path_escape(tmp_path, monkeypatch):
    old=tmp_path/'old';new=tmp_path/'new';old.mkdir();(new/'training').mkdir(parents=True)
    (new/'training/neuralese_corpora.json').write_text(json.dumps({'corpora':[{'id':'bad','admission':'evaluation-only-never-training','path':'../escape','include':['cases.jsonl']}]}))
    with pytest.raises(ValueError,match='safe runtime-relative'):
        assemble_protected_runtime_inputs(old,new)


def test_assembly_copies_packets_of_every_evaluation_packet_registry_in_the_union(tmp_path, monkeypatch):
    old=tmp_path/'old'; new=tmp_path/'new'
    (new/'training').mkdir(parents=True)
    (new/'training/neuralese_corpora.json').write_text('{"corpora": []}')
    (new/'training/memetic-held-programs.json').write_text(json.dumps({'corpora':[
        {'id':'memetic','admission':'evaluation-only-never-training','path':'runs/memetic','include':['cases.jsonl']}]}))
    (new/'training/protected-worlds.json').write_text(json.dumps({'sources':[
        {'id':'a','kind':'evaluation-packets','registry':'training/neuralese_corpora.json'},
        {'id':'b','kind':'evaluation-packets','registry':'training/memetic-held-programs.json'}]}))
    (old/'runs/memetic').mkdir(parents=True)
    (old/'runs/memetic/cases.jsonl').write_text('held programs')
    class Worlds:
        digest='worlds'
    monkeypatch.setattr(custody.ProtectedWorlds,'load',lambda **kwargs:Worlds())
    result=assemble_protected_runtime_inputs(old,new)
    assert (new/'runs/memetic/cases.jsonl').read_text()=='held programs'
    assert [item['id'] for item in result['evaluation_packets']]==['memetic']
