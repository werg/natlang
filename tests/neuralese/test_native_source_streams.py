import contextlib
import json

import pytest

from natlang_neuralese.data.native_sources import NativeGraphSource
from natlang_neuralese.train.native_source_streams import stream_pending_batches
from natlang_neuralese.train.native_state_sft import pending_exposure_batches
from natlang_neuralese.common.hashing import sha256_file_hex
from test_native_state_sft import graph, authenticated_pair, source_manifest


def compressed_source(tmp_path):
    from natlang_neuralese.common.jsonio import write_jsonl_text
    ordinary = [dict(graph(f'ordinary-{i}'), tokenizer_fingerprint='1'*64) for i in range(6)]
    pairs = [authenticated_pair(f'pair-{i}') for i in range(6)]
    manifest = source_manifest(tmp_path, ordinary, pairs)
    declaration = json.loads(manifest.read_text())
    for binding in declaration['sources']:
        old = tmp_path/binding['path']; target = old.with_suffix('.jsonl.zst')
        with write_jsonl_text(target) as stream:
            stream.write(old.read_text())
        binding.update(path=target.name, sha256=sha256_file_hex(target))
        if binding['kind'] == 'native-graphs': binding['training_priority'] = 1
    manifest.write_text(json.dumps(declaration))
    return NativeGraphSource(manifest, '1'*64, required_split='train', role_ids=(58,59,60,61))


def count_readers(monkeypatch):
    import natlang_neuralese.data.native_sources as module
    original = module.open_jsonl_text
    counts, active = {}, set()
    @contextlib.contextmanager
    def opened(path):
        counts[str(path)] = counts.get(str(path), 0)+1
        marker = object(); active.add(marker)
        try:
            with original(path) as stream: yield stream
        finally: active.remove(marker)
    monkeypatch.setattr(module, 'open_jsonl_text', opened)
    return counts, active


@pytest.mark.parametrize('nested', [False, True])
def test_interleaving_reads_each_compressed_leaf_once(tmp_path, monkeypatch, nested):
    source = compressed_source(tmp_path)
    if nested:
        path = tmp_path/'outer.json'
        path.write_text(json.dumps({'schema':'natlang.native-graph-sources/1', 'sources':[
            {'kind':'native-graphs', 'path':source.path.name, 'sha256':sha256_file_hex(source.path)}]}))
        source = NativeGraphSource(path, '1'*64, required_split='train', role_ids=(58,59,60,61))
    _, batches = pending_exposure_batches(source.entries, {}, epochs=1, batch_graphs=2)
    expected = [source.entries[index]['id'] for _, indices, _ in batches for index in indices]
    counts, active = count_readers(monkeypatch)
    reference = [row.id for row in source.iter_indices([i for _, indices, _ in batches for i in indices])]
    assert reference == expected and sum(counts.values()) > 2 and not active
    counts.clear()
    actual = [example.id for _, _, _, batch in stream_pending_batches(source, batches) for example in batch]
    assert actual == expected
    assert len(counts) == 2 and set(counts.values()) == {1}
    assert not active


def test_early_close_releases_all_independent_decompressors(tmp_path, monkeypatch):
    source = compressed_source(tmp_path)
    _, batches = pending_exposure_batches(source.entries, {}, epochs=1, batch_graphs=1)
    counts, active = count_readers(monkeypatch)
    stream = stream_pending_batches(source, batches)
    next(stream); next(stream)
    assert len(active) == 2
    stream.close()
    assert not active and set(counts.values()) == {1}


def test_resumed_multivisit_streams_keep_exact_remaining_credit_order(tmp_path, monkeypatch):
    source = compressed_source(tmp_path)
    credits = {source.entries[0]['id']:2, source.entries[7]['id']:1}
    _, batches = pending_exposure_batches(source.entries, credits, epochs=2, batch_graphs=2)
    counts, active = count_readers(monkeypatch)
    result = list(stream_pending_batches(source, batches))
    assert [(v,i,end) for v,i,end,_ in result] == batches
    assert [row.id for _,_,_,batch in result for row in batch] == [source.entries[i]['id'] for _,indices,_ in batches for i in indices]
    assert set(counts.values()) == {2} and not active


def test_interleaved_reader_checks_mutation_before_next_rank(tmp_path):
    source = compressed_source(tmp_path)
    _, batches = pending_exposure_batches(source.entries, {}, epochs=1, batch_graphs=1)
    stream = stream_pending_batches(source, batches)
    next(stream)
    path = source.children[0].path
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='source changed'):
        next(stream)
    stream.close()
