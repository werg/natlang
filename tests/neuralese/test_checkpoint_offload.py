"""Asynchronous checkpoint offload to the external disk (train/checkpoint_offload.py), with tiny files."""
import hashlib
import json
import os
from pathlib import Path

import pytest
import torch

from natlang_neuralese.train import checkpoint_offload as co
from natlang_neuralese.train import checkpoint_policy as cp


@pytest.fixture
def layout(tmp_path, monkeypatch):
    repo, hdd = tmp_path / 'repo', tmp_path / 'hdd'
    out = repo / 'runs' / 'lineage' / 'run-v1' / 'stage'
    out.mkdir(parents=True)
    hdd.mkdir()
    monkeypatch.setenv('NATLANG_REPO', str(repo))
    monkeypatch.setenv('NATLANG_CHECKPOINT_OFFLOAD', '1')
    monkeypatch.setenv('NATLANG_CHECKPOINT_MIRROR', str(hdd / 'natlang-active-lineage'))
    monkeypatch.setenv('NATLANG_CHECKPOINT_OFFLOAD_MIN_GB', '0')
    monkeypatch.setenv('NATLANG_CHECKPOINT_OFFLOAD_HDD_FLOOR_GB', '0')
    monkeypatch.setattr(co, '_mount_of', lambda path: hdd)  # the tmp "external disk"
    monkeypatch.setattr(co, '_same_filesystem', lambda a, b: False)
    monkeypatch.setattr(co, '_OFFLOADER', None)
    monkeypatch.setattr(co, '_MANIFEST_FILES', None)
    return out, hdd / 'natlang-active-lineage' / 'lineage' / 'run-v1' / 'stage'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data=b''):
    path.write_bytes(data or os.urandom(4096))
    return path


def test_the_offload_copy_verifies_and_an_archived_file_becomes_a_link(layout):
    out, mirror_dir = layout
    best = write(out / 'best-checkpoint.pt')
    digest = sha(best)
    event = co.offload_one(best, 'archive', co.OffloadSettings.from_env(), copier=co.local_copy)
    mirror = mirror_dir / 'best-checkpoint.pt'
    assert event['linked'] and event['sha256'] == digest and event['copied']
    assert best.is_symlink() and Path(os.readlink(best)) == mirror and sha(best) == digest
    assert co.read_record(mirror)['sha256'] == digest
    assert not list(mirror_dir.glob('*.offload.part')) and not list(out.glob('*.offload-link'))


def test_the_latest_slot_is_mirrored_but_stays_on_the_nvme(layout):
    out, mirror_dir = layout
    slot = write(out / 'checkpoint.pt')
    event = co.offload_one(slot, 'latest', co.OffloadSettings.from_env(), copier=co.local_copy)
    assert not slot.is_symlink() and 'linked' not in event
    assert sha(mirror_dir / 'checkpoint.pt') == sha(slot)
    again = co.offload_one(slot, 'latest', co.OffloadSettings.from_env(), copier=co.local_copy)
    assert not again['copied']  # the mirror is current: no second copy


@pytest.mark.parametrize('role,name', [('latest', 'checkpoint.pt'), ('archive', 'best-checkpoint.pt')])
def test_atomic_publication_keeps_the_opened_snapshot_but_never_replaces_its_successor(layout, monkeypatch, role, name):
    out, mirror_dir = layout
    slot = write(out / name, b'a' * 4096)
    original_identity = co._identity(slot.stat())
    successor = write(out / (name + '.pending'), b'b' * 4096)
    monkeypatch.setattr(co, 'CHUNK', 1024)
    monkeypatch.setattr(co, 'SYNC_EVERY', 1024)

    def publish_successor(*streams):
        if successor.exists():
            os.replace(successor, slot)

    monkeypatch.setattr(co, '_drop', publish_successor)
    if role == 'latest':
        event = co.offload_one(slot, role, co.OffloadSettings.from_env(), copier=co.local_copy)
        assert event['copied']
    else:
        with pytest.raises(RuntimeError, match='changed before its link swap'):
            co.offload_one(slot, role, co.OffloadSettings.from_env(), copier=co.local_copy)
    mirror = mirror_dir / name
    assert mirror.read_bytes() == b'a' * 4096
    assert co.read_record(mirror)['source_identity'] == original_identity
    assert co.mirror_current(slot, mirror) is None
    assert not slot.is_symlink() and slot.read_bytes() == b'b' * 4096
    assert not list(mirror_dir.glob('*.offload.part'))


def test_in_place_source_mutation_is_rejected_even_when_the_copied_bytes_hash_matches(layout, monkeypatch):
    out, mirror_dir = layout
    slot = write(out / 'checkpoint.pt', b'a' * 4096)
    monkeypatch.setattr(co, 'CHUNK', 1024)
    monkeypatch.setattr(co, 'SYNC_EVERY', 1024)
    changed = False

    def mutate_source(*streams):
        nonlocal changed
        if not changed:
            with slot.open('ab') as stream:
                stream.write(b'b')
            changed = True

    monkeypatch.setattr(co, '_drop', mutate_source)
    with pytest.raises(RuntimeError, match='changed during the copy'):
        co.offload_one(slot, 'latest', co.OffloadSettings.from_env(), copier=co.local_copy)
    assert slot.read_bytes() == b'a' * 4096 + b'b'
    assert not (mirror_dir / slot.name).exists()
    assert not list(mirror_dir.glob('*.offload.part'))


def test_a_copy_that_fails_verification_leaves_the_file_and_no_partial(layout, monkeypatch):
    out, mirror_dir = layout
    best = write(out / 'best-checkpoint.pt')
    monkeypatch.setattr(co, 'file_sha256', lambda path, **_: '0' * 64)  # the copy reads back wrong
    with pytest.raises(RuntimeError, match='verification failed'):
        co.offload_one(best, 'archive', co.OffloadSettings.from_env(), copier=co.local_copy)
    assert best.is_file() and not best.is_symlink()
    assert not (mirror_dir / 'best-checkpoint.pt').exists() and not list(mirror_dir.glob('*.part'))


def test_the_link_swap_never_leaves_a_dangling_link_or_replaces_a_changed_file(tmp_path):
    path = write(tmp_path / 'best-checkpoint.pt')
    with pytest.raises(FileNotFoundError):
        co.replace_with_link(path, tmp_path / 'missing.pt')
    assert path.is_file() and not path.is_symlink()
    dest = write(tmp_path / 'mirror.pt')
    identity = co._identity(path.lstat())
    write(tmp_path / 'best-checkpoint.pt.new')
    os.replace(tmp_path / 'best-checkpoint.pt.new', path)  # a newer best was published meanwhile
    with pytest.raises(RuntimeError, match='changed before its link swap'):
        co.replace_with_link(path, dest, expect=identity)
    assert path.is_file() and not path.is_symlink() and not list(tmp_path.glob('*.offload-link'))
    co.replace_with_link(path, dest, expect=co._identity(path.lstat()))
    assert path.is_symlink() and path.read_bytes() == dest.read_bytes()


def test_a_failing_offload_never_fails_training_and_is_retried_at_the_next_write(layout):
    out, mirror_dir = layout
    calls = []

    def flaky(path, dest, prune=False):
        calls.append(path)
        if len(calls) == 1:
            raise OSError('external disk went away')
        return co.verified_copy(path, dest)

    worker = co.CheckpointOffloader(co.OffloadSettings.from_env(), copier=flaky)
    best = write(out / 'best-checkpoint.pt')
    worker.submit(best, 'archive')  # returns at once; the failure is an event
    assert worker.wait(30)
    assert worker.events[-1]['event'] == 'checkpoint_offload_failed' and best.is_file() and not best.is_symlink()
    assert json.loads((out / 'checkpoint-offload.jsonl').read_text().splitlines()[-1])['event'] == \
        'checkpoint_offload_failed'
    worker.submit(write(out / 'checkpoint.pt'), 'latest')  # the next write retries the failed file
    assert worker.wait(30)
    assert best.is_symlink() and (mirror_dir / 'checkpoint.pt').is_file()


def test_policy_writes_are_offloaded_and_a_best_shares_the_mirror_by_a_hard_link(layout, monkeypatch):
    out, mirror_dir = layout
    monkeypatch.setenv('NATLANG_CHECKPOINT_PRUNE_OPTIMIZER', '0')  # unpruned bests share the slot's mirror
    worker = co.CheckpointOffloader(co.OffloadSettings.from_env(), copier=co.local_copy)
    monkeypatch.setattr(co, '_OFFLOADER', worker)
    policy = cp.CheckpointPolicy(out)
    assert json.loads((out / 'checkpoint-policy.json').read_text())['offload']['mirror'] == str(mirror_dir)
    policy.save({'step': 1, 'w': torch.ones(3)}, metric=1.0)
    assert worker.wait(30)
    slot, best = out / 'checkpoint.pt', out / 'best-checkpoint.pt'
    assert not slot.is_symlink() and best.is_symlink()
    assert (mirror_dir / 'best-checkpoint.pt').stat().st_ino == (mirror_dir / 'checkpoint.pt').stat().st_ino
    policy.save({'step': 2, 'w': torch.zeros(3)}, metric=2.0)  # worse: the best keeps its step-1 mirror
    assert worker.wait(30)
    assert torch.load(best)['step'] == 1 and torch.load(mirror_dir / 'checkpoint.pt')['step'] == 2
    assert json.loads((out / 'checkpoint-policy.json').read_text())['best_metric'] == 1.0


def test_resume_reads_the_mirror_when_the_nvme_slot_is_missing_or_truncated(layout, monkeypatch):
    out, mirror_dir = layout
    monkeypatch.setattr(co, '_OFFLOADER', co.CheckpointOffloader(co.OffloadSettings.from_env(),
                                                                 copier=co.local_copy))
    cp.CheckpointPolicy(out).save({'step': 5, 'w': torch.arange(4.)})
    assert co._OFFLOADER.wait(30)
    slot = out / 'checkpoint.pt'
    with open(slot, 'r+b') as stream:  # a cut-off write
        stream.truncate(slot.stat().st_size // 2)
    policy = cp.CheckpointPolicy(out)
    assert policy.recovered['local'] == 'truncated' and slot.is_symlink()
    assert cp.CheckpointPolicy.load(policy.slot)['step'] == 5
    slot.unlink()
    assert cp.CheckpointPolicy(out).recovered['local'] == 'missing'
    assert cp.CheckpointPolicy.load(slot)['step'] == 5
    fresh = layout[0].parent / 'never-mirrored'
    assert cp.CheckpointPolicy(fresh).recovered is None  # a run without a mirror record starts fresh


def test_stale_partials_are_cleaned_and_another_copiers_partial_is_left_alone(layout):
    out, mirror_dir = layout
    mirror_dir.mkdir(parents=True)
    ours = write(mirror_dir / 'best-checkpoint.pt.offload.part')
    os.utime(ours, (1, 1))
    theirs = write(mirror_dir / 'best-checkpoint.pt.part')
    best = write(out / 'best-checkpoint.pt')
    event = co.offload_one(best, 'archive', co.OffloadSettings.from_env(), copier=co.local_copy)
    assert event['event'] == 'checkpoint_offload_skipped' and 'another copy' in event['reason']
    assert theirs.exists() and best.is_file() and not best.is_symlink()
    theirs.unlink()
    co.offload_one(best, 'archive', co.OffloadSettings.from_env(), copier=co.local_copy)
    assert not ours.exists() and best.is_symlink()


def test_no_mirror_on_the_source_filesystem_or_outside_the_runs(layout, monkeypatch, tmp_path):
    settings = co.OffloadSettings.from_env()
    assert co.mirror_unavailable(tmp_path / 'elsewhere' / 'checkpoint.pt', settings) == 'outside the mirrored trees'
    monkeypatch.setattr(co, '_same_filesystem', lambda a, b: True)
    assert 'source filesystem' in co.mirror_unavailable(layout[0] / 'checkpoint.pt', settings)
    monkeypatch.setattr(co, '_mount_of', lambda path: None)
    assert 'no mounted filesystem' in co.mirror_unavailable(layout[0] / 'checkpoint.pt', settings)


def test_the_default_copier_runs_in_an_idle_priority_child(layout):
    out, mirror_dir = layout
    slot = write(out / 'checkpoint.pt')
    record = co.child_copy(slot, mirror_dir / 'checkpoint.pt')
    assert record['sha256'] == sha(slot) == sha(mirror_dir / 'checkpoint.pt')
    with pytest.raises(RuntimeError, match='offload copy failed'):
        co.child_copy(out / 'missing.pt', mirror_dir / 'missing.pt')


def full_state(step):
    return {'step': step, 'student_parameters': {'w': torch.arange(6, dtype=torch.bfloat16)},
            'optimizer': {'state': {0: {'momentum': torch.ones(6, dtype=torch.bfloat16)}}},
            'optimizer_parameter_names': ['w'], 'rng': {'cpu': torch.zeros(3, dtype=torch.uint8)}}


def test_an_archived_full_state_is_stored_weights_only_and_verified(layout):
    out, mirror_dir = layout
    best = out / 'best-checkpoint.pt'
    torch.save(full_state(3), best)
    event = co.offload_one(best, 'archive', co.OffloadSettings.from_env(), copier=co.local_copy, prune=True)
    assert event['optimizer_pruned'] and event['bytes'] < event['source_bytes'] and best.is_symlink()
    state = cp.CheckpointPolicy.load(best)
    assert state['optimizer_pruned'] is True and 'optimizer' not in state and 'optimizer_parameter_names' not in state
    assert state['optimizer_pruned_keys'] == ['optimizer', 'optimizer_parameter_names']
    assert torch.equal(state['student_parameters']['w'], torch.arange(6, dtype=torch.bfloat16)) and state['step'] == 3
    assert co.read_record(mirror_dir / 'best-checkpoint.pt')['optimizer_pruned'] is True


def test_the_latest_slot_keeps_its_optimizer_in_the_mirror(layout):
    out, mirror_dir = layout
    slot = out / 'checkpoint.pt'
    torch.save(full_state(4), slot)
    co.offload_one(slot, 'latest', co.OffloadSettings.from_env(), copier=co.local_copy)
    assert 'optimizer' in torch.load(mirror_dir / 'checkpoint.pt') and sha(slot) == sha(mirror_dir / 'checkpoint.pt')


def test_a_pruned_copy_that_reads_back_different_is_refused(layout, monkeypatch):
    out, mirror_dir = layout
    best = out / 'best-checkpoint.pt'
    torch.save(full_state(3), best)
    real = co.load_state
    monkeypatch.setattr(co, 'load_state', lambda path: (
        {**real(path), 'step': 99} if str(path).endswith('.offload.part') else real(path)))
    with pytest.raises(RuntimeError, match='differs at state.step'):
        co.offload_one(best, 'archive', co.OffloadSettings.from_env(), copier=co.local_copy, prune=True)
    assert best.is_file() and not best.is_symlink() and not list(mirror_dir.glob('*.part'))


def test_policy_bests_are_weights_only_while_the_slot_keeps_its_optimizer(layout, monkeypatch):
    out, mirror_dir = layout
    worker = co.CheckpointOffloader(co.OffloadSettings.from_env(), copier=co.local_copy)
    monkeypatch.setattr(co, '_OFFLOADER', worker)
    policy = cp.CheckpointPolicy(out)
    policy.save(full_state(1), metric=1.0)
    assert worker.wait(30)
    best = cp.CheckpointPolicy.load(out / 'best-checkpoint.pt')
    assert best['optimizer_pruned'] and best['step'] == 1 and 'optimizer' not in best
    assert 'optimizer' in cp.CheckpointPolicy.load(out / 'checkpoint.pt')
    assert 'optimizer' in torch.load(mirror_dir / 'checkpoint.pt')


def test_superseded_pruned_best_does_not_rebind_successor_receipts(layout):
    out, mirror_dir = layout
    best = out / 'best-checkpoint.pt'
    torch.save(full_state(7), best)
    receipt = out / 'best-checkpoint.json'
    successor_receipt = None

    def publish_after_copy(path, mirror, prune):
        nonlocal successor_receipt
        record = co.local_copy(path, mirror, prune)
        successor = out / 'successor.pt'
        torch.save(full_state(8), successor)
        os.replace(successor, best)
        successor_receipt = {'step': 8, 'files': {'best-checkpoint.pt': {'sha256': sha(best), 'bytes': best.stat().st_size}}}
        receipt.write_text(json.dumps(successor_receipt))
        return record

    with pytest.raises(RuntimeError, match='changed before'):
        co.offload_one(best, 'archive', co.OffloadSettings.from_env(), copier=publish_after_copy, prune=True)
    assert json.loads(receipt.read_text()) == successor_receipt
    assert not best.is_symlink() and cp.CheckpointPolicy.load(best)['step'] == 8


def test_receipts_that_pin_a_pruned_best_are_rebound_to_the_weights_only_file(layout):
    out, mirror_dir = layout
    best = out / 'best-checkpoint.pt'
    torch.save(full_state(7), best)
    receipt = out / 'best-checkpoint.json'
    receipt.write_text(json.dumps({'step': 7, 'files': {'best-checkpoint.pt': {'sha256': sha(best), 'bytes': 1},
                                                        'best-heads.pt': {'sha256': 'abc', 'bytes': 2}}}))
    event = co.offload_one(best, 'archive', co.OffloadSettings.from_env(), copier=co.local_copy, prune=True)
    files = json.loads(receipt.read_text())['files']
    assert event['receipts'] == [str(receipt)]
    assert files['best-checkpoint.pt'] == {'sha256': sha(best), 'bytes': best.stat().st_size, 'optimizer_pruned': True}
    assert files['best-heads.pt'] == {'sha256': 'abc', 'bytes': 2}


def test_a_continuation_refuses_a_weights_only_checkpoint_clearly(tmp_path):
    with pytest.raises(ValueError, match='weights-only checkpoint .* continue from the stage.s latest checkpoint.pt'):
        co.require_optimizer({'optimizer_pruned': True, 'step': 3}, tmp_path / 'best-checkpoint.pt')
    co.require_optimizer(full_state(3), tmp_path / 'checkpoint.pt')


def test_auto_pruning_keeps_the_optimizer_of_writers_that_continue_from_a_best(layout):
    out, mirror_dir = layout
    warmup_best, direct_best = out / 'best-checkpoint.pt', out / 'best-other.pt'
    torch.save(full_state(1), warmup_best)  # no direct schema: a text warm-up best is a continuation parent
    torch.save({**full_state(2), 'schema': 'natlang.neuralese-direct/1'}, direct_best)
    settings = co.OffloadSettings.from_env()
    assert not co.offload_one(warmup_best, 'archive', settings, copier=co.local_copy, prune='auto')['optimizer_pruned']
    assert 'optimizer' in torch.load(warmup_best)
    assert co.offload_one(direct_best, 'archive', settings, copier=co.local_copy, prune='auto')['optimizer_pruned']


def test_files_of_a_published_corpus_snapshot_are_never_linked(layout, monkeypatch):
    out, mirror_dir = layout
    repo = Path(os.environ['NATLANG_REPO'])
    (repo / 'training' / 'corpus-manifests').mkdir(parents=True)
    (repo / 'training' / 'corpus-manifests' / 'held.json').write_text(json.dumps({'path': 'runs/lineage', 'files': [{'path': 'run-v1/stage/best-checkpoint.pt'}]}))
    best = write(out / 'best-checkpoint.pt')
    event = co.offload_one(best, 'archive', co.OffloadSettings.from_env(), copier=co.local_copy)
    assert 'pinned by corpus manifest held.json' in event['reason'] and not best.is_symlink()
