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


def test_preparation_reuses_preloaded_children_before_reading_the_next_leaf(tmp_path,monkeypatch):
    from natlang_neuralese.data.native_sources import NativeGraphSource
    from natlang_neuralese.common.hashing import canonical_json_sha256_hex
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'cache'))
    leaves=[row(tmp_path/f'{name}.jsonl',name) for name in ['a','b','c']]
    old=manifest(tmp_path/'old.json',[{**leaves[0],'training_priority':3},leaves[1]])
    new=manifest(tmp_path/'new.json',[old,leaves[2]])
    calls=[]
    class FrozenLeafLoader(NativeGraphSource):
        def __init__(self,path,*args,**kwargs):
            assert Path(path).suffix!='.json', 'preparation must own recursive manifest loading'
            calls.append(Path(path).name)
            super().__init__(path,*args,**kwargs)
    pool=NativeSourcePreparation('tok',());pool.source_type=FrozenLeafLoader
    previous=pool.load(old,split='train');source=pool.load(new,split='train')
    assert calls==['a.jsonl','b.jsonl','c.jsonl']
    assert source.children[0] is previous
    assert source.entries[0] is previous.entries[0]
    assert source.entries[1] is previous.entries[1]
    ordinary=NativeGraphSource(new['path'],'tok',required_split='train')
    assert source.entries==ordinary.entries
    assert canonical_json_sha256_hex(source.entries)==canonical_json_sha256_hex(ordinary.entries)
    assert [entry['training_priority'] for entry in source.entries]==[3,0,0]
    overridden=manifest(tmp_path/'override.json',[{**new,'training_priority':7}])
    changed=pool.load(overridden,split='train')
    assert [entry['training_priority'] for entry in changed.entries]==[7,7,7]
    assert changed.entries[0] is not source.entries[0]
    assert source.entries[0]['training_priority']==3
    with pytest.raises(ValueError,match='cycles'):
        pool.load(old,split='train',_parents=(Path(old['path']).resolve(),))


def test_equal_read_only_group_metadata_is_retained_once_across_leaves(tmp_path,monkeypatch):
    from natlang_neuralese.data.native_sources import NativeGraphSource
    from natlang_neuralese.common.hashing import canonical_json_sha256_hex
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'cache'))
    group='shared-source-group-'+('fresh-words-'*30)
    leaves=[]
    for index in range(3):
        path=tmp_path/f'leaf-{index}.jsonl';row(path,str(index))
        value=json.loads(path.read_text());value['source_groups']=[group]
        path.write_text(json.dumps(value)+'\n');leaves.append(binding(path))
    root=manifest(tmp_path/'root.json',leaves)
    baseline=NativeGraphSource(root['path'],'tok',required_split='train')
    expected=canonical_json_sha256_hex(baseline.entries)
    pool=NativeSourcePreparation('tok',());actual=pool.load(root,split='train')
    assert actual.entries==baseline.entries
    assert canonical_json_sha256_hex(actual.entries)==expected
    groups=[child.entries[0]['groups'] for child in actual.children]
    assert all(group is groups[0] for group in groups)
    assert all(next(iter(child.groups)) is groups[0][0] for child in actual.children)
    assert len(pool._group_strings)==1 and len(pool._group_bundles)==1
    assert pool.loaded_sources==4
    duplicate=manifest(tmp_path/'duplicate.json',[leaves[0],leaves[0]])
    with pytest.raises(ValueError,match='repeats a graph identity'):
        pool.load(duplicate,split='train')
