import importlib.util
import json
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('compact_hash_cache',Path(__file__).parents[1]/'scripts/compact_hash_cache.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_compaction_keeps_live_signatures_and_first_digest_with_pipe_names(tmp_path):
    source=tmp_path/'live|source';source.write_text('data');s=source.stat()
    legacy=f'/source-fixture/live|source|{s.st_size}|{s.st_mtime_ns}'
    strict=f'content|/source-fixture/live|source|{s.st_dev}|{s.st_ino}|{s.st_size}|{s.st_mtime_ns}|{s.st_ctime_ns}'
    dead='/tmp/pytest-of-owner/removed|0|1'
    rows=[{'key':key,'sha256':digest*64} for key,digest in [(legacy,'a'),(legacy,'b'),(strict,'c'),(dead,'d')]]
    cache=tmp_path/'cache.jsonl';cache.write_text(''.join(json.dumps(row)+'\n' for row in rows)+'incomplete')
    original=source.read_bytes();result=module.compact(cache,threshold=0)
    assert result['retained_entries']==2
    assert [json.loads(s)['sha256'] for s in cache.read_text().splitlines()]==['a'*64,'c'*64]
    assert source.read_bytes()==original


def test_replaced_source_signature_is_never_reused_and_symlink_cache_is_refused(tmp_path,monkeypatch):
    source=tmp_path/'source';source.write_text('old');s=source.stat()
    key=f'content|{source}|{s.st_dev}|{s.st_ino}|{s.st_size}|{s.st_mtime_ns}|{s.st_ctime_ns}'
    cache=tmp_path/'cache';cache.write_text(json.dumps({'key':key,'sha256':'a'*64})+'\n')
    replacement=tmp_path/'replacement';replacement.write_text('new');replacement.replace(source)
    assert module.compact(cache,threshold=0)['retained_entries']==0
    monkeypatch.setenv('NATLANG_HASH_CACHE',str(cache))
    from natlang_neuralese.common.hashing import sha256_file_hex_cached,sha256_file_hex
    assert sha256_file_hex_cached(source,trust_archive_receipt=False)==sha256_file_hex(source)
    assert sha256_file_hex(source)!='a'*64
    link=tmp_path/'link';link.symlink_to(cache)
    with pytest.raises(ValueError):module.compact(link,threshold=0)


def test_compaction_collapses_old_signatures_and_discards_test_scratch(tmp_path):
    source=tmp_path/'source';cache=tmp_path/'cache'
    rows=[{'key':key,'sha256':digest*64} for key,digest in [
        ('/source-fixture/key|1|1','a'),('/source-fixture/key|1|2','b'),('/source-fixture/key|1|1','c'),
        ('/tmp/pytest-of-owner/old-file|1|1','d')]]
    cache.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    result=module.compact(cache,threshold=0)
    assert result['retained_entries']==1 and result['discarded_disposable_entries']==1
    assert json.loads(cache.read_text())['sha256']=='a'*64
