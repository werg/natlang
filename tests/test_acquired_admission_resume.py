"""Exact restart semantics for completed per-model admission checks."""

import json
import os

import pytest

from natlang_neuralese.common.jsonio import canonical_json_bytes
from scripts.neuralese_data.admission_resume import (
    load_or_compute_model_checkpoint,
    load_model_checkpoint,
    model_checkpoint_identity,
    save_model_checkpoint,
)


def identity(path, *, policy='current', verifier='v1'):
    code = path.parent / f'{verifier}.py'
    code.write_text(verifier)
    return model_checkpoint_identity(
        source={'export_sha256': 'a' * 64, 'source_sha256': 'b' * 64},
        model={'name': 'lfm25', 'tokenizer_inventory': {'config': 'c' * 64}},
        policy={'current_protection': policy},
        artifacts=[path], verifier_files=[code],
    )


def test_completed_model_result_resumes_but_unfinished_peer_does_not(tmp_path):
    source = tmp_path / 'source.zst'; source.write_bytes(b'source')
    checkpoint_root = tmp_path / 'checkpoints'
    first = identity(source)
    result = {'proof_rows': [{'id': 'graph-1', 'sha256': 'd' * 64}],
              'counts': {'graphs:train': 1}}
    save_model_checkpoint(checkpoint_root, first, result)

    # The completed first model survives interruption; a different peer/model
    # has no checkpoint and must execute its verifier.
    assert load_model_checkpoint(checkpoint_root, first) == result
    peer = {**first, 'model': {'name': 'mellum21'}}
    assert load_model_checkpoint(checkpoint_root, peer) is None


def test_interrupted_pair_resumes_completed_side_and_verifies_missing_side(tmp_path):
    source = tmp_path / 'source.zst'; source.write_bytes(b'source')
    cache = tmp_path / 'checkpoints'
    lf = identity(source)
    mellum = {**lf, 'model': {'name': 'mellum21'}}
    calls = []

    first, _, reused = load_or_compute_model_checkpoint(
        cache, lf, lambda: calls.append('lf') or {'verified': 'lf'})
    assert first == {'verified': 'lf'} and not reused
    # Simulate process interruption before Mellum is validated.
    restarted_lf, _, reused = load_or_compute_model_checkpoint(
        cache, lf, lambda: calls.append('lf-again') or {'verified': 'wrong'})
    assert restarted_lf == first and reused
    restarted_mellum, _, reused = load_or_compute_model_checkpoint(
        cache, mellum, lambda: calls.append('mellum') or {'verified': 'mellum'})
    assert restarted_mellum == {'verified': 'mellum'} and not reused
    assert calls == ['lf', 'mellum']


def test_policy_source_and_verifier_changes_invalidate_saved_result(tmp_path):
    source = tmp_path / 'source.zst'; source.write_bytes(b'original')
    cache = tmp_path / 'cache'
    original = identity(source)
    result = {'proof_rows': [], 'counts': {}}
    save_model_checkpoint(cache, original, result)

    assert load_model_checkpoint(cache, identity(source, policy='new-current-holds')) is None
    assert load_model_checkpoint(cache, identity(source, verifier='v2')) is None

    # Same-size content replacement with restored mtime is still a cache miss:
    # the shared strong cache identity includes ctime and inode metadata.
    before = source.stat()
    source.write_bytes(b'mutated!')
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
    changed = identity(source)
    assert changed != original
    assert load_model_checkpoint(cache, changed) is None


def test_checkpoint_content_address_and_atomic_immutability(tmp_path):
    source = tmp_path / 'source.zst'; source.write_bytes(b'source')
    current = identity(source)
    cache = tmp_path / 'cache'
    first = {'proof_rows': [{'id': 'a'}], 'counts': {'graphs:train': 1}}
    path = save_model_checkpoint(cache, current, first)
    assert load_model_checkpoint(cache, current) == first
    assert save_model_checkpoint(cache, current, first) == path
    with pytest.raises(ValueError, match='conflicting acquired admission results'):
        save_model_checkpoint(cache, current, {'proof_rows': [], 'counts': {}})

    # A changed cache payload cannot impersonate the filename's content hash.
    text = path.read_text()
    path.write_text(text.replace('"graphs:train":1', '"graphs:train":2'))
    with pytest.raises(ValueError, match='content identity mismatch'):
        load_model_checkpoint(cache, current)


def test_json_round_trip_preserves_tuple_based_identity_and_result_checkpoint(tmp_path):
    source = tmp_path / 'source.zst'; source.write_bytes(b'source')
    current = identity(source)
    # decision_contract currently returns tuple rows, which JSON persists as
    # arrays.  A later process reconstructs the same JSON identity as lists.
    current['policy']['decision_holds'] = [('hold.jsonl', 'a' * 64, 17, 'decision-hold')]
    result = {'groups': {'train': ('group-a', 'group-b')}}
    cache = tmp_path / 'cache'
    path = save_model_checkpoint(cache, current, result)
    reloaded_identity = json.loads(canonical_json_bytes(current))

    calls = []
    loaded, loaded_path, reused = load_or_compute_model_checkpoint(
        cache, reloaded_identity, lambda: calls.append('recomputed') or {'wrong': True})
    assert reused is True
    assert loaded_path == path
    assert loaded == json.loads(canonical_json_bytes(result))
    assert calls == []

    # The original process-local tuple identity also loads the JSON array
    # representation directly; this is the path used by a same-process retry.
    assert load_model_checkpoint(cache, current) == json.loads(canonical_json_bytes(result))

    # The original process-local tuple form must also see the same immutable
    # JSON checkpoint and permit an idempotent save.
    assert save_model_checkpoint(cache, current, result) == path


def test_concurrent_publication_of_same_tuple_checkpoint_is_idempotent(tmp_path, monkeypatch):
    source = tmp_path / 'source.zst'; source.write_bytes(b'source')
    current = identity(source)
    current['policy']['decision_holds'] = [('hold.jsonl', 'a' * 64, 17, 'decision-hold')]
    result = {'groups': {'train': ('group-a', 'group-b')}}
    real_link = os.link

    def publish_same_payload_then_report_race(source_path, destination):
        # Model another publisher committing the identical canonical payload
        # between our exists check and link().
        real_link(source_path, destination)
        raise FileExistsError(destination)

    monkeypatch.setattr('scripts.neuralese_data.admission_resume.os.link',
                        publish_same_payload_then_report_race)
    checkpoint = save_model_checkpoint(tmp_path / 'cache', current, result)
    assert load_model_checkpoint(tmp_path / 'cache', current) == json.loads(canonical_json_bytes(result))
    assert checkpoint.is_file()
