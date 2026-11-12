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
    # The container gate uses HOME=/tmp, which would make every tmp_path an
    # eligible checkpoint source under the home-root policy.
    monkeypatch.setenv('HOME', str(tmp_path / 'home'))
    monkeypatch.setenv('NATLANG_REPO', str(repo))
    monkeypatch.setenv('NATLANG_CHECKPOINT_OFFLOAD', '1')
    monkeypatch.setenv('NATLANG_CHECKPOINT_MIRROR', str(hdd / 'natlang-active-lineage'))
    monkeypatch.setenv('NATLANG_CHECKPOINT_OFFLOAD_MIN_GB', '0')
    monkeypatch.setenv('NATLANG_CHECKPOINT_OFFLOAD_HDD_FLOOR_GB', '0')
    monkeypatch.setenv('NATLANG_CHECKPOINT_OFFLOAD_QUEUE', str(tmp_path / 'offload-queue'))
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


def test_explicitly_cancelled_copy_discards_partial_without_replacing_verified_mirror(tmp_path, monkeypatch):
    source = write(tmp_path / 'checkpoint.pt', b'x' * 4096)
    mirror = write(tmp_path / 'mirror.pt', b'previous-verified')
    old = mirror.read_bytes()
    monkeypatch.setattr(co, 'CHUNK', 1024)
    calls = 0

    def superseded():
        nonlocal calls
        calls += 1
        return calls == 2

    with pytest.raises(co.CheckpointCopySuperseded):
        co.verified_copy(source, mirror, cancel=superseded)
    assert mirror.read_bytes() == old and not list(tmp_path.glob('*.offload.part'))


def test_explicit_copy_cancellation_also_checks_full_readback_hash(tmp_path, monkeypatch):
    source = write(tmp_path / 'checkpoint.pt', b'x' * 4096)
    mirror = write(tmp_path / 'mirror.pt', b'previous-verified')
    old = mirror.read_bytes()
    monkeypatch.setattr(co, 'CHUNK', 1024)
    calls = 0

    def superseded_during_readback():
        nonlocal calls
        calls += 1
        return calls == 6  # four source chunks, source-final check, then first mirror readback chunk

    with pytest.raises(co.CheckpointCopySuperseded):
        co.verified_copy(source, mirror, cancel=superseded_during_readback)
    assert mirror.read_bytes() == old and not list(tmp_path.glob('*.offload.part'))


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
    failure = co._process_queue_once(worker.root, settings=worker.settings, copier=worker.copier)
    assert failure['event'] == 'checkpoint_offload_failed' and best.is_file() and not best.is_symlink()
    assert list((worker.root / 'pending').glob('*.json'))
    assert json.loads((out / 'checkpoint-offload.jsonl').read_text().splitlines()[-1])['event'] == \
        'checkpoint_offload_failed'
    worker.submit(write(out / 'checkpoint.pt'), 'latest')  # the next write retries the failed file
    assert worker.wait(30)
    assert best.is_symlink() and (mirror_dir / 'checkpoint.pt').is_file()
    assert worker.close()


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


def test_owned_copier_runs_without_a_detached_process(layout):
    out, mirror_dir = layout
    slot = write(out / 'checkpoint.pt')
    record = co.owned_copy(slot, mirror_dir / 'checkpoint.pt')
    assert record['sha256'] == sha(slot) == sha(mirror_dir / 'checkpoint.pt')
    with pytest.raises(FileNotFoundError):
        co.owned_copy(out / 'missing.pt', mirror_dir / 'missing.pt')


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


def test_owner_retries_durable_job_after_copy_failure_without_another_checkpoint(layout):
    import threading
    out, mirror_dir = layout
    attempts = []

    def fail_once(path, dest, prune=False):
        attempts.append((path, dest))
        if len(attempts) == 1:
            raise OSError('temporary HDD disconnect')
        return co.verified_copy(path, dest)

    worker = co.CheckpointOffloader(co.OffloadSettings.from_env(), copier=fail_once)
    slot = write(out / 'checkpoint.pt')
    worker.submit(slot, 'latest')
    assert co._process_queue_once(worker.root, settings=worker.settings, copier=fail_once)['event'] == \
        'checkpoint_offload_failed'
    owner = threading.Thread(target=co.run_offload_owner, kwargs={
        'root': worker.root, 'idle_seconds': .05, 'retry_seconds': .01, 'copier': fail_once})
    owner.start()
    owner.join(5)
    assert not owner.is_alive() and len(attempts) == 2
    assert not list((worker.root / 'pending').glob('*.json'))
    assert sha(mirror_dir / slot.name) == sha(slot)
    assert co.read_record(mirror_dir / slot.name)['source_identity'] == co._identity(slot.stat())


def test_ledger_owner_launch_uses_durable_low_priority_service_command(layout, monkeypatch):
    calls = []

    class Result:
        returncode = 0
        stdout = stderr = ''

    def run(args, **kwargs):
        calls.append(args)
        if args[:4] == ['systemctl', '--user', 'is-active', '--quiet']:
            return type('Inactive', (), {'returncode': 1})()
        return Result()

    monkeypatch.setattr(co.subprocess, 'run', run)
    co._launch_owner(layout[0].parent / 'queue')
    command = calls[-1]
    assert command[1].endswith('scripts/memory_ledger.py') and command[2:4] == ['run', '--unit']
    assert command[command.index('--priority') + 1] == 'low'
    assert command[command.index('--class') + 1] == 'service'
    assert command[command.index('--budget-gb') + 1] == '2'
    assert '--detach' in command and '--resumable' in command and '--wait' in command
    assert command[-3:] == ['-m', 'natlang_neuralese.train.checkpoint_offload', '--owner-loop']


def test_submit_waits_for_an_owner_in_idle_exit_before_starting_replacement(layout, monkeypatch):
    root = layout[0].parent / 'queue'
    root.mkdir()
    (root / 'owner-idle.json').write_text('{}\n')
    calls = []
    owner_checks = 0

    class Result:
        stdout = stderr = ''

        def __init__(self, returncode):
            self.returncode = returncode

    def run(args, **kwargs):
        nonlocal owner_checks
        calls.append(args)
        if args[:4] == ['systemctl', '--user', 'is-active', '--quiet']:
            if args[4].endswith('.service') and not args[4].endswith('-waiter.service'):
                owner_checks += 1
                return Result(0 if owner_checks == 1 else 1)
            return Result(1)
        return Result(0)

    monkeypatch.setattr(co.subprocess, 'run', run)
    co._launch_owner(root)
    assert owner_checks >= 2 and any('memory_ledger.py' in str(arg) for arg in calls[-1])


def test_new_rolling_checkpoint_is_selected_before_queued_archive_work(layout):
    out, _ = layout
    worker = co.CheckpointOffloader(co.OffloadSettings.from_env(), copier=co.local_copy)
    slot = write(out / 'checkpoint.pt')
    (out.parent / 'old-stage-a').mkdir()
    (out.parent / 'old-stage-b').mkdir()
    old_a = write(out.parent / 'old-stage-a' / 'checkpoint-58.pt')
    old_b = write(out.parent / 'old-stage-b' / 'checkpoint-49.pt')
    worker.submit(old_a, 'archive', 'auto')
    worker.submit(old_b, 'archive', 'auto')
    worker.submit(slot, 'latest')
    first = co._process_queue_once(worker.root, settings=worker.settings, copier=worker.copier)
    assert first['path'] == str(slot.resolve())
    assert worker.close()


def test_new_checkpoint_replaces_only_pending_generation_and_keeps_exact_latest_pin(layout):
    out, _ = layout
    worker = co.CheckpointOffloader(co.OffloadSettings.from_env(), copier=co.local_copy)
    slot = write(out / 'checkpoint.pt', b'old-generation')
    worker.submit(slot, 'latest')
    descriptor = next((worker.root / 'pending').glob('*.json'))
    old = json.loads(descriptor.read_text())
    replacement = write(out / 'checkpoint.pt.pending', b'new-generation')
    os.replace(replacement, slot)
    worker.submit(slot, 'latest')
    new = json.loads(descriptor.read_text())
    assert new['id'] != old['id'] and Path(new['pin']).read_bytes() == b'new-generation'
    assert not Path(old['pin']).exists()
    assert worker.wait(30)
    assert worker.close()


def test_active_copy_pin_survives_pending_replacement_until_it_returns(layout):
    import threading
    out, _ = layout
    worker = co.CheckpointOffloader(co.OffloadSettings.from_env(), copier=co.local_copy)
    slot = write(out / 'checkpoint.pt', b'active-generation')
    worker.submit(slot, 'latest')
    descriptor = next((worker.root / 'pending').glob('*.json'))
    active = json.loads(descriptor.read_text())
    started, release = threading.Event(), threading.Event()

    def blocked(path, dest, prune=False):
        started.set()
        assert release.wait(5)
        return co.verified_copy(path, dest)

    owner = threading.Thread(target=lambda: co._process_queue_once(
        worker.root, settings=worker.settings, copier=blocked))
    owner.start()
    assert started.wait(5)
    replacement = write(out / 'checkpoint.pt.pending', b'newer-generation')
    os.replace(replacement, slot)
    worker.submit(slot, 'latest')
    pending = json.loads(descriptor.read_text())
    assert pending['id'] != active['id'] and Path(active['pin']).is_file()
    release.set()
    owner.join(5)
    assert not owner.is_alive() and not Path(active['pin']).exists() and descriptor.exists()
    assert worker.wait(30) and not descriptor.exists()
    assert worker.close()


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
