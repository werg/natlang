"""Identical operational copies preserve immutable source meaning and stream cursors."""
import json
import os
from pathlib import Path
import pytest
from natlang_neuralese.data import native_sources as sources
from natlang_neuralese.data.native_sources import NativeGraphSource
from natlang_neuralese.train.native_source_preparation import NativeSourcePreparation
from natlang_neuralese.train.native_resident import binding
from test_native_state_sft import graph


def source(tmp_path,kind):
    leaf=tmp_path/'rows.jsonl'
    leaf.write_text(''.join(json.dumps(graph(name))+'\n' for name in ('a','b','c')))
    path=leaf
    if kind=='manifest':
        path=tmp_path/'sources.json';path.write_text(json.dumps({'schema':'natlang.native-graph-sources/1',
            'sources':[{'kind':'native-graphs',**binding(leaf)}]}))
    value=NativeGraphSource(path,'tokenizer-sha',required_split='train')
    return path,value


def replace(path,kind):
    if kind=='touch':os.utime(path,ns=(path.stat().st_atime_ns,path.stat().st_mtime_ns+1000000))
    else:
        temporary=path.with_suffix('.replacement');temporary.write_bytes(path.read_bytes());temporary.replace(path)


@pytest.mark.parametrize('kind',['leaf','manifest'])
@pytest.mark.parametrize('change',['touch','replace'])
def test_identical_change_before_and_during_stream_preserves_exact_resume(tmp_path,monkeypatch,kind,change):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'indexes'))
    path,value=source(tmp_path,kind);entries=value.entries.copy();bindings=dict(value.bindings)
    before=value.file_binding;replace(path,change)
    assert [row.id for row in value.iter_from(1)]==['b','c']
    assert value.file_binding!=before and value.entries==entries and value.bindings==bindings
    stream=value.iter_from();assert next(stream).id=='a'
    replace(path,change)
    assert [row.id for row in stream]==['b','c']
    assert [row.id for row in value.iter_indices([2,0])]==['c','a']
    preparation=NativeSourcePreparation('tokenizer-sha',());preparation.retain(value)
    replace(path,change)
    assert preparation.load(binding(path),split='train') is value


@pytest.mark.parametrize('kind',['leaf','manifest'])
def test_changed_bytes_even_with_restored_size_and_mtime_reject(tmp_path,monkeypatch,kind):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'indexes'))
    path,value=source(tmp_path,kind);status=path.stat()
    data=path.read_bytes();changed=data.replace(b'"a"',b'"z"',1) if kind=='leaf' else data.replace(b'"sources"',b'"changed"',1)
    assert changed!=data and len(changed)==len(data)
    path.write_bytes(changed);os.utime(path,ns=(status.st_atime_ns,status.st_mtime_ns))
    with pytest.raises(ValueError,match='source changed after index validation'):list(value)


@pytest.mark.parametrize('kind',['leaf','manifest'])
def test_hash_race_rejects_even_when_content_returns_to_identical_bytes(tmp_path,monkeypatch,kind):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'indexes'))
    path,value=source(tmp_path,kind);replace(path,'touch')
    original=sources.sha256_file_hex
    def racing(target):
        result=original(target);replace(Path(target),'replace');return result
    monkeypatch.setattr(sources,'sha256_file_hex',racing)
    with pytest.raises(ValueError,match='source changed after index validation'):list(value)


def test_cached_constructor_identical_copy_during_load_revalidates_without_parsing(tmp_path,monkeypatch):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'indexes'))
    path,before=source(tmp_path,'leaf')
    from natlang_neuralese.common import row_index
    load=row_index.load
    def replaced(cache):
        value=load(cache);replace(path,'replace');return value
    monkeypatch.setattr(row_index,'load',replaced)
    monkeypatch.setattr(NativeGraphSource,'_parse',lambda *args:pytest.fail('cache must remain valid'))
    after=NativeGraphSource(path,'tokenizer-sha',required_split='train')
    assert after.entries==before.entries


def test_initial_leaf_parse_rejects_changed_then_restored_bytes(tmp_path,monkeypatch):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'indexes'))
    path=tmp_path/'rows.jsonl';original=json.dumps(graph('a'))+'\n';other=json.dumps(graph('z'))+'\n'
    path.write_text(original)
    def changed_rows(target):
        path.write_text(other)
        yield json.loads(path.read_text())
        path.write_text(original)
    monkeypatch.setattr(sources,'_jsonl',changed_rows)
    with pytest.raises(ValueError,match='source changed during streaming graph validation'):
        NativeGraphSource(path,'tokenizer-sha',required_split='train')
    assert path.read_text()==original


def test_initial_manifest_parse_rejects_changed_then_restored_bytes(tmp_path,monkeypatch):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'indexes'))
    path,_=source(tmp_path,'manifest');original=path.read_text()
    other=json.loads(original);other['sources'][0]['training_priority']=7
    read_text=Path.read_text
    def changed_read(target,*args,**kwargs):
        if target==path:
            path.write_text(json.dumps(other));value=read_text(target,*args,**kwargs)
            path.write_text(original);return value
        return read_text(target,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',changed_read)
    with pytest.raises(ValueError,match='native graph manifest changed during validation'):
        NativeGraphSource(path,'tokenizer-sha',required_split='train')
    assert read_text(path)==original
