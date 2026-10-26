import json
from pathlib import Path

import pytest

from natlang_neuralese.common.hashing import sha256_file_hex
from natlang_neuralese.train.native_foundation_recovery import native_graph_bindings, _source_map_call
from natlang_neuralese.train import native_foundation_recovery
from natlang_neuralese.data import native_source_tree


def bind(path):
    return {'path': str(path), 'sha256': sha256_file_hex(path), 'bytes': path.stat().st_size}


def header(path, payload, declared):
    path.write_text(json.dumps({'schema': 'natlang.native-graph-sources/1',
        'sources': [{'kind': 'semantic-pairs', 'path': declared,
                     'sha256': sha256_file_hex(payload), 'model': 'fixture'}]}))
    return bind(path)


def test_relative_payload_keeps_canonical_resolution_and_records_consulted_identity_map(tmp_path):
    payload=tmp_path/'pairs.jsonl';payload.write_text('exact payload')
    staged=tmp_path/'staged.jsonl';staged.write_bytes(payload.read_bytes())
    mapping=tmp_path/'.natlang-source-map.json'
    mapping.write_text(json.dumps({'schema':'natlang.staged-source-map/1',
        'paths':{str(payload):str(staged)}}))
    reference=header(tmp_path/'source.json',payload,'pairs.jsonl')
    paths={item['path']for item in native_graph_bindings(reference)}
    assert paths=={reference['path'],str(payload),str(staged),str(mapping)}


def test_identity_mapping_is_bound_even_when_it_does_not_change_the_path(tmp_path):
    payload=tmp_path/'pairs.jsonl';payload.write_text('exact payload')
    mapping=tmp_path/'.natlang-source-map.json'
    mapping.write_text(json.dumps({'schema':'natlang.staged-source-map/1','paths':{str(payload):str(payload)}}))
    reference=header(tmp_path/'source.json',payload,str(payload))
    assert str(mapping)in{item['path']for item in native_graph_bindings(reference)}


def test_payload_reverse_identity_map_is_part_of_consumed_custody(tmp_path):
    manifest_dir=tmp_path/'manifests';manifest_dir.mkdir()
    payload_dir=tmp_path/'payloads';payload_dir.mkdir()
    payload=payload_dir/'pairs.jsonl';payload.write_text('exact payload')
    mapping=payload_dir/'.natlang-source-map.json'
    original='/pruned/original/pairs.jsonl'
    mapping.write_text(json.dumps({'schema':'natlang.staged-source-map/1','paths':{original:str(payload)}}))
    assert native_source_tree._source_identity(payload)==Path(original)
    reference=header(manifest_dir/'source.json',payload,str(payload))
    assert str(mapping)in{item['path']for item in native_graph_bindings(reference)}


def test_map_custody_rejects_mutation_during_its_canonical_read(tmp_path,monkeypatch):
    mapping=tmp_path/'.natlang-source-map.json'
    mapping.write_text(json.dumps({'schema':'natlang.staged-source-map/1','paths':{}}))
    actual=native_foundation_recovery.sha256_file_hex_cached
    def changed(path):
        mapping.write_text(json.dumps({'schema':'natlang.staged-source-map/1','paths':{},'changed':True}))
        return actual(path)
    monkeypatch.setattr(native_foundation_recovery,'sha256_file_hex_cached',changed)
    with pytest.raises(ValueError,match='changed while reading'):
        _source_map_call(native_source_tree._source_identity,tmp_path/'payload',maps=[])


def test_recipe_dependency_walk_captures_a_map_created_after_identity_walk(tmp_path, monkeypatch):
    from natlang_neuralese.train import recipe
    payload=tmp_path/'pairs.jsonl';payload.write_text('exact payload')
    staged=tmp_path/'staged.jsonl';staged.write_bytes(payload.read_bytes())
    reference=header(tmp_path/'source.json',payload,'pairs.jsonl')
    mapping=tmp_path/'.natlang-source-map.json'
    original=recipe.native_source_dependencies
    def changed(path, **kwargs):
        mapping.write_text(json.dumps({'schema':'natlang.staged-source-map/1',
            'paths':{str(payload):str(staged)}}))
        return original(path, **kwargs)
    monkeypatch.setattr(recipe,'native_source_dependencies',changed)
    paths={item['path']for item in native_graph_bindings(reference)}
    assert {str(mapping),str(staged),str(payload)} <= paths
