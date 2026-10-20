from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path

import pytest

from scripts.neuralese_data import prepare_acquired_session_admission as producer
from scripts.neuralese_data import source_closure_metadata_cache as cache


@pytest.fixture
def closure(tmp_path, monkeypatch):
    monkeypatch.setenv('NATLANG_HASH_CACHE', str(tmp_path / 'hashes.jsonl'))
    cohort = tmp_path / 'cohort'
    row = cohort / 'row'
    row.mkdir(parents=True)
    source = row / 'full-trajectories.jsonl.zst'
    source.write_bytes(b'original source bytes')
    export = row / 'export.json'
    export.write_text(json.dumps({'outputs': {source.name: sha256(source.read_bytes()).hexdigest()}}))
    complete = cohort / 'complete.json'
    complete.write_text(json.dumps({'files': [{'status': 'exported', 'receipt': str(export)}]}))
    manifest = {'files': [], 'normalized_export_bindings': [{
        'path': str(export), 'sha256': sha256(export.read_bytes()).hexdigest(),
        'source_sha256': sha256(source.read_bytes()).hexdigest()}]}
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    return path, manifest, source, export, complete


def test_real_producer_reuses_without_reopening_metadata(closure, monkeypatch):
    path, manifest, _, export, complete = closure
    original = producer._source_closure_artifacts
    with cache.reuse_authenticated_closure_metadata(producer) as memo:
        expected = producer._source_closure_artifacts(path, manifest)
        read_text = Path.read_text

        def forbid_metadata_read(self, *args, **kwargs):
            assert self not in (export, complete), 'authenticated metadata reopened'
            return read_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, 'read_text', forbid_metadata_read)
        assert producer._source_closure_artifacts(path, manifest) == expected
        assert memo.producer_calls == 1 and memo.reuses == 1
        assert memo.artifact_authentications == 2 * len(expected)
    assert producer._source_closure_artifacts is original


@pytest.mark.parametrize('index', [2, 3, 4])
def test_rewritten_referenced_artifact_same_mtime_reaches_original_rejection(closure, index):
    path, manifest, *_ = closure
    artifact = closure[index]
    with cache.reuse_authenticated_closure_metadata(producer) as memo:
        producer._source_closure_artifacts(path, manifest)
        st = artifact.stat()
        data = artifact.read_bytes()
        changed = bytes([data[0] ^ 1]) + data[1:]
        artifact.write_bytes(changed)
        os.utime(artifact, ns=(st.st_atime_ns, st.st_mtime_ns))
        with pytest.raises((ValueError, json.JSONDecodeError)):
            producer._source_closure_artifacts(path, manifest)
        assert memo.reuses == 0 and memo.producer_calls == 2


def test_return_values_do_not_mutate_memo(closure):
    path, manifest, *_ = closure
    with cache.reuse_authenticated_closure_metadata(producer) as memo:
        first = producer._source_closure_artifacts(path, manifest)
        expected = json.loads(json.dumps(first))
        first[0]['sha256'] = '0' * 64
        second = producer._source_closure_artifacts(path, manifest)
        assert second == expected
        second.clear()
        assert producer._source_closure_artifacts(path, manifest) == expected
        assert memo.reuses == 2


def test_force_full_validation_still_reads_real_metadata(closure, monkeypatch):
    path, manifest, _, export, _ = closure
    reads = []
    read_text = Path.read_text

    def observed(self, *args, **kwargs):
        if self == export:
            reads.append(self)
        return read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', observed)
    with cache.reuse_authenticated_closure_metadata(producer) as memo:
        expected = producer._source_closure_artifacts(path, manifest)
        assert producer._source_closure_artifacts(path, manifest, use_cache=False) == expected
        assert producer._source_closure_artifacts(path, manifest, use_cache=False) == expected
        assert len(reads) == 3 and memo.producer_calls == 3 and memo.reuses == 0


def test_manifest_bytes_and_supplied_value_both_bind_key(closure):
    path, manifest, *_ = closure
    with cache.reuse_authenticated_closure_metadata(producer) as memo:
        first = producer._source_closure_artifacts(path, manifest)
        path.write_text(json.dumps(manifest, indent=2))
        assert producer._source_closure_artifacts(path, manifest) == first
        supplied = {**manifest, 'files': [{'path': 'missing', 'sha256': '0' * 64}]}
        with pytest.raises(FileNotFoundError):
            producer._source_closure_artifacts(path, supplied)
        assert memo.producer_calls == 3 and memo.reuses == 0


def test_relocated_symlink_returns_changed_inventory_for_standard_verifier(closure):
    path, manifest, source, *_ = closure
    with cache.reuse_authenticated_closure_metadata(producer) as memo:
        first = producer._source_closure_artifacts(path, manifest)
        target = source.parent / 'relocated' / source.name
        target.parent.mkdir()
        source.rename(target)
        source.symlink_to(target)
        after = producer._source_closure_artifacts(path, manifest)
        assert after != first and str(target) in {r['path'] for r in after}
        assert memo.producer_calls == 2 and memo.reuses == 0


def test_removal_does_not_reuse_old_inventory(closure):
    path, manifest, source, *_ = closure
    with cache.reuse_authenticated_closure_metadata(producer) as memo:
        producer._source_closure_artifacts(path, manifest)
        source.unlink()
        with pytest.raises(FileNotFoundError):
            producer._source_closure_artifacts(path, manifest)
        assert memo.reuses == 0


def test_hash_stat_race_fails_closed(closure, monkeypatch):
    path, _, source, *_ = closure
    original = cache.sha256_file_hex_cached

    def racing(candidate, **kwargs):
        digest = original(candidate, **kwargs)
        if candidate == source:
            candidate.write_bytes(b'changed')
        return digest

    monkeypatch.setattr(cache, 'sha256_file_hex_cached', racing)
    with pytest.raises(RuntimeError, match='changed during'):
        cache._authenticated(source)


def test_context_restores_original_after_failure_and_rejects_nested_cache(closure):
    original = producer._source_closure_artifacts
    with pytest.raises(LookupError):
        with cache.reuse_authenticated_closure_metadata(producer):
            with pytest.raises(ValueError, match='already installed'):
                with cache.reuse_authenticated_closure_metadata(producer):
                    pass
            raise LookupError('actual caller failure')
    assert producer._source_closure_artifacts is original


@pytest.mark.parametrize('kind', ['export', 'source', 'complete', 'declared-receipt', 'manifest-file'])
def test_preexisting_alias_retarget_never_reuses_intact_old_tree(closure, kind):
    path, manifest, source, export, complete = closure
    root = path.parent
    if kind == 'export':
        other = root / 'other-cohort'
        other_row = other / 'row'
        other_row.mkdir(parents=True)
        (other_row / source.name).write_bytes(source.read_bytes())
        other_export = other_row / export.name
        other_export.write_bytes(export.read_bytes())
        (other / 'complete.json').write_text(json.dumps({'files': [
            {'status': 'exported', 'receipt': str(other_export)}]}))
        alias = root / 'cohort-alias'
        alias.symlink_to(export.parent.parent, target_is_directory=True)
        manifest['normalized_export_bindings'][0]['path'] = str(alias / 'row' / export.name)
        new_target = other
    elif kind == 'source':
        targets = root / 'targets'
        targets.mkdir()
        first = targets / 'first' / source.name
        second = targets / 'second' / source.name
        first.parent.mkdir()
        second.parent.mkdir()
        first.write_bytes(source.read_bytes())
        second.write_bytes(source.read_bytes())
        source.unlink()
        source.symlink_to(first)
        alias, new_target = source, second
    elif kind == 'complete':
        first, second = root / 'complete-first.json', root / 'complete-second.json'
        first.write_bytes(complete.read_bytes())
        second.write_bytes(complete.read_bytes())
        complete.unlink()
        complete.symlink_to(first)
        alias, new_target = complete, second
    elif kind == 'declared-receipt':
        other = root / 'other-export.json'
        other.write_bytes(export.read_bytes())
        alias = root / 'receipt-alias.json'
        alias.symlink_to(export)
        complete.write_text(json.dumps({'files': [
            {'status': 'exported', 'receipt': str(alias)}]}))
        new_target = other
    else:
        first, second = root / 'first.bin', root / 'second.bin'
        first.write_bytes(b'identical body')
        second.write_bytes(b'identical body')
        alias = root / 'artifact-alias'
        alias.symlink_to(first)
        manifest['files'] = [{'path': alias.name,
                              'sha256': sha256(first.read_bytes()).hexdigest()}]
        new_target = second
    path.write_text(json.dumps(manifest))
    with cache.reuse_authenticated_closure_metadata(producer) as memo:
        before = producer._source_closure_artifacts(path, manifest)
        # Every old physical target remains present and byte-identical.
        old_bindings = [(Path(r['path']), Path(r['path']).read_bytes()) for r in before]
        alias.unlink()
        alias.symlink_to(new_target, target_is_directory=new_target.is_dir())
        assert all(p.read_bytes() == body for p, body in old_bindings)
        if kind == 'declared-receipt':
            with pytest.raises(ValueError, match='each complete'):
                producer._source_closure_artifacts(path, manifest)
        else:
            after = producer._source_closure_artifacts(path, manifest)
            assert after != before
        assert memo.producer_calls == 2 and memo.reuses == 0


def test_alias_retarget_during_first_producer_call_is_not_cached(closure, monkeypatch):
    path, manifest, source, *_ = closure
    first = source.parent / 'first' / source.name
    second = source.parent / 'second' / source.name
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(source.read_bytes())
    second.write_bytes(source.read_bytes())
    source.unlink()
    source.symlink_to(first)
    original = producer._source_closure_artifacts

    def racing(*args, **kwargs):
        result = original(*args, **kwargs)
        source.unlink()
        source.symlink_to(second)
        return result

    monkeypatch.setattr(producer, '_source_closure_artifacts', racing)
    with cache.reuse_authenticated_closure_metadata(producer) as memo:
        with pytest.raises(ValueError, match='retargeted'):
            producer._source_closure_artifacts(path, manifest)
        assert memo.key is None and memo.reuses == 0


def test_complete_metadata_rewrite_cannot_leave_uncaptured_receipt_edges(closure, monkeypatch):
    path, manifest, _, export, complete = closure
    alias_a = path.parent / 'receipt-a.json'
    alias_b = path.parent / 'receipt-b.json'
    alias_a.symlink_to(export)
    alias_b.symlink_to(export)
    complete.write_text(json.dumps({'files': [
        {'status': 'exported', 'receipt': str(alias_a)}]}))
    original = producer._source_closure_artifacts
    calls = 0

    def racing(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            # Dependency capture saw A. The actual producer now sees B,
            # which initially resolves to exactly the same declared export.
            complete.write_text(json.dumps({'files': [
                {'status': 'exported', 'receipt': str(alias_b)}]}))
        return original(*args, **kwargs)

    monkeypatch.setattr(producer, '_source_closure_artifacts', racing)
    with cache.reuse_authenticated_closure_metadata(producer) as memo:
        with pytest.raises(ValueError, match='parsed closure metadata'):
            producer._source_closure_artifacts(path, manifest)
        assert memo.key is None and memo.reuses == 0
        other = path.parent / 'other-export.json'
        other.write_bytes(export.read_bytes())
        alias_b.unlink()
        alias_b.symlink_to(other)
        with pytest.raises(ValueError, match='each complete'):
            producer._source_closure_artifacts(path, manifest)
        assert memo.reuses == 0 and memo.producer_calls == 2
