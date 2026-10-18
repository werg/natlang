"""Exact restart semantics for completed per-model admission checks."""

import json
import hashlib
import os
from pathlib import Path
import subprocess

import pytest

from natlang_neuralese.common.jsonio import canonical_json_bytes
from scripts.neuralese_data.admission_resume import (
    load_or_compute_model_checkpoint,
    load_model_checkpoint,
    model_checkpoint_identity,
    save_model_checkpoint,
)
from scripts.neuralese_data.promote_admission_checkpoint import (
    SHARD_SCHEMA,
    promote_checkpoint_file,
    promote_checkpoint_set,
    verify_pinned_verifier_revisions,
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


def test_exact_verifier_checkpoint_promotion_preserves_result_and_old_file(tmp_path):
    old_verifier = {'path': str(tmp_path / 'admission_resume.py'),
                    'sha256': 'a' * 64, 'bytes': 5784}
    new_verifier = {'path': old_verifier['path'], 'sha256': 'b' * 64, 'bytes': 6380}
    identity = {
        'schema': 'natlang.acquired-admission-shard-checkpoint-identity/1',
        'model_identity': {'model': {'name': 'lfm25'}, 'artifacts': [],
                           'verifier_files': [old_verifier, {'path': 'other.py',
                               'sha256': 'c' * 64, 'bytes': 23}]},
        'range': {'start': 0, 'stop': 50, 'source_rows_sha256': 'd' * 64},
        'artifacts': [],
    }
    result = {'proofs': [{'id': 'source-row-1', 'sha256': 'e' * 64}],
              'groups': {'train': ['group-a'], 'test': []}}
    old_root = tmp_path / 'old'; old = save_model_checkpoint(old_root, identity, result)
    old_bytes = old.read_bytes()

    promoted = promote_checkpoint_file(old, tmp_path / 'promoted',
                                       old_binding=old_verifier, new_binding=new_verifier)
    new_payload = json.loads(Path(promoted['new_checkpoint']['path']).read_text())
    old_payload = json.loads(old.read_text())
    expected = json.loads(canonical_json_bytes(identity))
    expected['model_identity']['verifier_files'][0] = new_verifier
    assert new_payload['identity'] == expected
    assert new_payload['result'] == old_payload['result']
    assert old.read_bytes() == old_bytes
    assert promoted['training_admission'] is False
    assert promoted['old_identity_sha256'] != promoted['new_identity_sha256']

    with pytest.raises(ValueError, match='old verifier binding'):
        promote_checkpoint_file(old, tmp_path / 'rejected',
                                old_binding={**old_verifier, 'sha256': 'f' * 64},
                                new_binding=new_verifier)


def test_checkpoint_set_promotion_resumes_partial_and_is_idempotent(tmp_path):
    export = tmp_path / 'export.json'
    export.write_text('{"complete_sessions":2}')
    old_verifier = {'path': str(tmp_path / 'admission_resume.py'),
                    'sha256': 'a' * 64, 'bytes': 5784}
    new_verifier = {'path': old_verifier['path'],
                    'sha256': 'b' * 64, 'bytes': 6380}
    normalized_sha = 'c' * 64
    checkpoint_root = tmp_path / 'checkpoints'
    result = {'verified': [{'id': 'source-0', 'sha256': 'd' * 64}]}
    old_paths = []
    for start in (0, 1):
        identity = {
            'schema': SHARD_SCHEMA,
            'model_identity': {
                'model': {'name': 'lfm25'},
                'source': {'export_path': str(export.resolve()),
                           'source_output_sha256': normalized_sha},
                'artifacts': [], 'verifier_files': [old_verifier],
            },
            'artifacts': [{'path': str(export.resolve()),
                           'sha256': hashlib.sha256(export.read_bytes()).hexdigest(),
                           'bytes': export.stat().st_size}],
            'range': {'start': start, 'stop': start + 1},
        }
        old_paths.append(save_model_checkpoint(
            checkpoint_root / 'shards' / f'{start:04d}', identity, result))

    # Simulate a process interruption after only the first shard was promoted.
    promote_checkpoint_file(old_paths[0], old_paths[0].parent,
                            old_binding=old_verifier, new_binding=new_verifier)
    args = dict(checkpoint_root=checkpoint_root, export_path=export,
                model_name='lfm25', old_binding=old_verifier,
                new_binding=new_verifier, expected_ranges={(0, 1), (1, 2)},
                normalized_source_sha256=normalized_sha)
    first = promote_checkpoint_set(**args)
    assert [row['range'] for row in first] == [
        {'start': 0, 'stop': 1}, {'start': 1, 'stop': 2}]
    assert all(row['training_admission'] is False for row in first)
    assert all(row['new_checkpoint'] for row in first)

    # A complete rerun authenticates the old/new pairs and reuses their exact
    # promoted checkpoint bytes without creating another identity or result.
    second = promote_checkpoint_set(**args)
    assert second == first

    # A new-binding checkpoint without its immutable old parent cannot be
    # treated as an authorized promotion just because its content hash checks.
    old_paths[0].unlink()
    with pytest.raises(ValueError, match='lacks its immutable old parent'):
        promote_checkpoint_set(**args)


def test_pinned_verifier_allows_unrelated_ancestor_advance_but_rejects_module_change(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()

    def git(*args):
        return subprocess.run(['git', '-C', str(repo), *args], check=True,
                              stdout=subprocess.PIPE, text=True).stdout.strip()

    git('init', '-q')
    git('config', 'user.email', 'test@example.invalid')
    git('config', 'user.name', 'Admission test')
    module = repo / 'scripts/neuralese_data/admission_resume.py'
    module.parent.mkdir(parents=True)
    old_bytes = b'old verifier implementation\n'
    module.write_bytes(old_bytes)
    git('add', '.')
    git('commit', '-qm', 'old verifier')
    old_revision = git('rev-parse', 'HEAD')

    new_bytes = b'canonical-json verifier implementation\n'
    module.write_bytes(new_bytes)
    git('add', '.')
    git('commit', '-qm', 'approved verifier')
    new_revision = git('rev-parse', 'HEAD')

    (repo / 'unrelated.txt').write_text('unrelated data update')
    git('add', '.')
    git('commit', '-qm', 'unrelated registry advance')
    bindings = dict(old_revision=old_revision, new_revision=new_revision,
                    old_sha256=hashlib.sha256(old_bytes).hexdigest(), old_bytes=len(old_bytes),
                    new_sha256=hashlib.sha256(new_bytes).hexdigest(), new_bytes=len(new_bytes))
    old_binding, new_binding, current = verify_pinned_verifier_revisions(repo, **bindings)
    assert current == git('rev-parse', 'HEAD') and current != new_revision
    assert old_binding['sha256'] == bindings['old_sha256']
    assert new_binding['sha256'] == bindings['new_sha256']

    changed_bytes = b'unreviewed verifier change\n'
    module.write_bytes(changed_bytes)
    with pytest.raises(ValueError, match='checkout verifier bytes differ'):
        verify_pinned_verifier_revisions(repo, **bindings)
