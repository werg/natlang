import json
from pathlib import Path
import pytest
from natlang_neuralese.train.native_source_preparation import NativeSourcePreparation
from natlang_neuralese.train.native_resident import binding


def row(path, name, split='train'):
    path.write_text(json.dumps({'schema':'natlang.neuralese-native-graph/1','id':name,
        'split':split,'source_groups':[name], 'tokenizer_fingerprint':'tok',
        'nodes':[{'name':'text','token_ids':[1,2],'target_weights':[0.,1.]}]})+'\n')
    return binding(path)


def manifest(path, children):
    path.write_text(json.dumps({'schema':'natlang.native-graph-sources/1',
        'sources':[{'kind':'native-graphs',**child} for child in children]})+'\n')
    return binding(path)


def test_nested_old_root_and_geometry_views_reuse_the_actual_validated_objects(tmp_path,monkeypatch):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'cache'))
    old=manifest(tmp_path/'old.json',[{**row(tmp_path/'a.jsonl','a'),'training_priority':3}])
    extra=row(tmp_path/'b.jsonl','b')
    new=manifest(tmp_path/'new.json',[old,extra])
    pool=NativeSourcePreparation('tok',())
    train=pool.load(new,split='train')
    previous=pool.load(old,split='train')
    assert previous is train.children[0]
    assert pool.load(old,split='train') is previous
    assert pool.loaded_roots==1 and pool.reused_roots==2
    assert previous.entries==train.entries[:1]
    assert train.entries[0]['training_priority']==3
    assert previous.bindings=={'a':train.bindings['a']}
    assert pool.load(extra,split='train') is train.children[1]


def test_reused_manifest_still_verifies_every_immutable_dependency(tmp_path,monkeypatch):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'cache'))
    leaf=tmp_path/'a.jsonl';child=row(leaf,'a')
    old=manifest(tmp_path/'old.json',[child]);new=manifest(tmp_path/'new.json',[old])
    pool=NativeSourcePreparation('tok',());pool.load(new,split='train')
    row(leaf,'changed')
    with pytest.raises(ValueError,match='reused preparation source changed'):
        pool.load(old,split='train')


def test_absent_subtree_uses_existing_loader_and_split_is_never_reinterpreted(tmp_path,monkeypatch):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'cache'))
    a=row(tmp_path/'a.jsonl','a');b=row(tmp_path/'b.jsonl','b')
    pool=NativeSourcePreparation('tok',())
    first=pool.load(a,split='train');second=pool.load(b,split='train')
    assert first is not second and pool.loaded_roots==2
    with pytest.raises(ValueError,match='expected test'):
        pool.load(a,split='test')
    assert pool.reused_roots==0
    with pytest.raises(ValueError):
        pool.load(a,split='train',pair_model='other')
    assert pool.reused_roots==0
