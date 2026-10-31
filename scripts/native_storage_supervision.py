#!/usr/bin/env python3
"""Observe the actual native writers and their filesystem reservations.

Checkpoint admission is enforced by the writer, not by this observer. No saved
PID, retired learner cursor, or historical run directory establishes ownership.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training/neuralese'))
from run_training_pipeline import atomic_json
from natlang_neuralese.common.paths import root as machine_root


def _reservation_slot_status(slot, suffix):
    slot = Path(slot)
    pending = slot.with_name(slot.name + suffix + '.reserved-pending')
    receipt = slot.with_name(slot.name + suffix + '.reservation.json')
    result = {'slot': str(slot), 'pending': str(pending), 'protected': False}
    if slot.is_file():
        result['committed_bytes'] = slot.stat().st_size
    if not pending.exists() and not receipt.exists():
        result['state'] = 'not-yet-reserved'
        return result
    try:
        record = json.loads(receipt.read_text())
        st = pending.lstat()
        expected = {'schema': 'natlang.checkpoint-reservation/1', 'device': st.st_dev, 'inode': st.st_ino}
        if pending.is_symlink() or st.st_nlink != 1 or record != expected:
            raise ValueError('reservation receipt or file identity differs')
        if slot.exists() and slot.samefile(pending):
            raise ValueError('reservation aliases the durable slot')
        allocated = st.st_blocks * 512
        complete = st.st_size > 0 and allocated >= st.st_size
        result.update(state='reserved' if complete else 'partially-allocated', pending_bytes=st.st_size,
                      allocated_bytes=allocated, protected=complete)
    except (OSError, ValueError, TypeError) as error:
        result.update(state='invalid-reservation', error=str(error))
    return result


def native_reservation_status(slot):
    slot = Path(slot)
    primary = _reservation_slot_status(slot, '')
    alternate = _reservation_slot_status(slot, '.alternate')
    result = dict(primary)
    result['slots'] = {'primary': primary, 'alternate': alternate}
    roles_path = slot.with_name(slot.name + '.reservation-roles.json')
    if roles_path.exists():
        try:
            roles = json.loads(roles_path.read_text())
            if roles.get('schema') != 'natlang.checkpoint-reservation-roles/1':
                raise ValueError('invalid reservation role schema')
            for role in ('writer', 'next'):
                identity = roles.get(role)
                if identity is None:
                    continue
                path = Path(identity['path'])
                if str(path) not in (primary['pending'], alternate['pending']):
                    raise ValueError('reservation role names another path')
                st = path.lstat()
                if (identity['device'], identity['inode']) != (st.st_dev, st.st_ino):
                    raise ValueError('reservation role inode changed')
            result['roles'] = roles
            next_identity = roles.get('next')
            if next_identity is not None:
                selected = primary if next_identity['path'] == primary['pending'] else alternate
                result.update(state=selected['state'], protected=selected['protected'])
        except (OSError, ValueError, TypeError, KeyError) as error:
            result['roles'] = {'state': 'stale-or-invalid', 'error': str(error)}
    if alternate['state'] == 'invalid-reservation':
        result.update(state='invalid-reservation', protected=False)
    return result


def current_native_writers(proc_root=Path('/proc')):
    writers = []
    for directory in proc_root.iterdir():
        if not directory.name.isdigit():
            continue
        try:
            args = (directory / 'cmdline').read_bytes().split(b'\0')
            args = [value.decode('utf-8', 'replace') for value in args if value]
            if 'natlang_neuralese.train.native_state_sft' not in args or '--validate-continuation-only' in args:
                continue
            out = args[args.index('--out') + 1]
            environment = (directory / 'environ').read_bytes().split(b'\0')
            unit = next((v.decode().split('=', 1)[1] for v in environment
                         if v.startswith(b'NATLANG_LEDGER_UNIT=')), None)
            record = {'pid': int(directory.name), 'unit': unit, 'out': out,
                      'reservation': native_reservation_status(Path(out) / 'checkpoint.pt')}
            progress = Path(out) / 'progress.json'
            if progress.is_file():
                value = json.loads(progress.read_text())
                record['progress'] = {k: value.get(k) for k in
                    ('global_step', 'step', 'checkpoint_step', 'timestamp', 'forward_tokens_per_second')}
            writers.append(record)
        except (OSError, ValueError, IndexError, StopIteration):
            continue  # Process exited or changed during this point-in-time scan.
    return writers


def native_storage_observation(internal, external):
    usage = shutil.disk_usage(internal)
    result = {'schema': 'natlang.native-storage-supervision/1', 'at': time.time(),
              'internal': {'path': str(internal), 'free_bytes': usage.free, 'total_bytes': usage.total},
              'writers': current_native_writers(),
              'enforcement': 'persistent pending-inode reservation and atomic admission in each writer',
              'scope': 'point-in-time actual process/filesystem observation; no pruning or recovery-completeness claim'}
    if external.is_mount():
        result['external'] = {'path': str(external), 'free_bytes': shutil.disk_usage(external).free}
    else:
        result['external'] = {'path': str(external), 'state': 'not-mounted'}
    result['attention'] = usage.free < 64 * 1024**3 or any(
        w['reservation']['state'] == 'invalid-reservation' for w in result['writers'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--internal', type=Path, default=machine_root('data_nvme'))
    parser.add_argument('--external', type=Path, default=machine_root('data_hdd'))
    parser.add_argument('--status', type=Path, default=Path.home() / '.local/state/natlang/native-storage/status.json')
    args = parser.parse_args()
    result = native_storage_observation(args.internal, args.external)
    atomic_json(args.status, result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
