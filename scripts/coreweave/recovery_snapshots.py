#!/usr/bin/env python3
"""Run an owner's explicit filesystem snapshot schedule for one sandbox.

This is durability plumbing, not training policy. A READY snapshot is an
additional recovery candidate; it is never represented as a verified restore.
Retention removes only this operator's snapshots and requires an independently
recorded successful restore of a newer snapshot before pruning anything.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import time
import uuid

try:
    from .recovery_inputs import durable_json
except ImportError:
    from recovery_inputs import durable_json


def verify_restored_files(source_id, snapshot_id, destination_id, files, *, inspect_destination, execute):
    """Read the real restored mount and bind exact bytes to its snapshot origin."""
    if source_id == destination_id:
        raise ValueError('restore verification requires a separate destination')
    import re
    if not files or any(not isinstance(f, dict) or not str(f.get('path', '')).startswith('/work/')
                        or '..' in Path(f['path']).parts or type(f.get('bytes')) is not int
                        or f['bytes'] <= 0 or not re.fullmatch('[0-9a-f]{64}', str(f.get('sha256', '')))
                        for f in files):
        raise ValueError('restore files need exact absolute work-volume paths and identities')
    actual_snapshot_ids = inspect_destination()
    if snapshot_id not in actual_snapshot_ids:
        raise ValueError('destination was not created from the selected snapshot')
    program = """import hashlib,json,pathlib,sys
items=json.loads(sys.argv[1]);out=[]
for item in items:
 p=pathlib.Path(item['path']);before=p.stat()
 with p.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
 after=p.stat()
 assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
 out.append({'path':str(p),'bytes':after.st_size,'sha256':sha})
print(json.dumps(out))
"""
    actual = execute(['python3', '-c', program, json.dumps(files, sort_keys=True)])
    expected = [{key: f[key] for key in ('path', 'bytes', 'sha256')} for f in files]
    if actual != expected:
        raise ValueError('restored files differ from their expected full identities')
    return {'source_sandbox_id': source_id, 'snapshot_id': snapshot_id,
            'destination_sandbox_id': destination_id, 'passed': True,
            'verified_files': actual, 'verified_at': time.time(),
            'full_training_continuation_verified': False,
            'scope': 'Actual snapshot origin and selected exact restored files; '
                     'full checkpoint decode and optimizer continuation are separate checks.'}


def run_once(directory, sandbox_id, *, capture, inspect, delete, interval=1800,
             keep=2, now=None):
    now = time.time() if now is None else now
    if interval <= 0 or keep < 2:
        raise ValueError('positive interval and at least two recovery snapshots required')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'state.json'
    state = json.loads(path.read_text()) if path.exists() else {
        'schema': 'natlang.recovery-snapshots/1', 'sandbox_id': sandbox_id,
        'snapshots': [], 'restore_verified': False}
    if state['sandbox_id'] != sandbox_id:
        raise ValueError('snapshot state belongs to a different sandbox')
    pending = state.get('pending')
    if not pending and now - state.get('last_capture_at', -interval) >= interval:
        pending = {'request_id': uuid.uuid4().hex, 'requested_at': now}
        state['pending'] = pending
        durable_json(path, state)
    if pending:
        if not pending.get('snapshot_id'):
            # Persist the deduplication key BEFORE the RPC. An ambiguous error
            # retries this key, never allocating another unseen snapshot.
            pending['snapshot_id'] = capture(pending['request_id'])
            durable_json(path, state)
        snapshot = inspect(pending['snapshot_id'])
        if snapshot['source_sandbox_id'] != sandbox_id:
            raise ValueError('snapshot source sandbox differs')
        pending.update(status=snapshot['status'], size_bytes=snapshot.get('size_bytes'))
        if snapshot['status'] in ('READY', 'FAILED'):
            state['snapshots'].append(dict(pending))
            state['last_capture_at'] = now
            state.pop('pending')
        durable_json(path, state)
        if snapshot['status'] == 'FAILED':
            raise RuntimeError('filesystem snapshot failed; independent checkpoint exports still required')

    # No restore receipt, no pruning. This intentionally retains the first
    # recovery candidates until actual destination readback is proven.
    restore_path = directory / 'verified-restore.json'
    if restore_path.exists():
        restore = json.loads(restore_path.read_text())
        ready = [s for s in state['snapshots'] if s.get('status') == 'READY'
                 and not s.get('deleted')]
        verified = next((s for s in ready if s['snapshot_id'] == restore.get('snapshot_id')), None)
        if (restore.get('source_sandbox_id') != sandbox_id or restore.get('passed') is not True
                or not restore.get('destination_sandbox_id') or not restore.get('verified_files')
                or verified is None):
            raise ValueError('restore receipt lacks successful bound destination verification')
        files = restore['verified_files']
        import re
        if not isinstance(files, list) or any(
                not isinstance(f, dict) or not f.get('path') or type(f.get('bytes')) is not int
                or f['bytes'] <= 0 or not re.fullmatch('[0-9a-f]{64}', str(f.get('sha256', '')))
                for f in files):
            raise ValueError('restore receipt lacks exact file identities')
        state['restore_verified'] = True
        protected = {s['snapshot_id'] for s in ready[-keep:]}
        protected.add(verified['snapshot_id'])
        protected.update(restore.get('protected_snapshot_ids', []))
        remove = [s for s in ready if s['snapshot_id'] not in protected
                  and s['requested_at'] < verified['requested_at']]
        durable_json(directory / 'deletion-plan.json', {
            'sandbox_id': sandbox_id, 'protected_snapshot_ids': sorted(protected),
            'delete_snapshot_ids': [s['snapshot_id'] for s in remove], 'at': now})
        for snapshot in remove:
            delete(snapshot['snapshot_id'])  # Adapter verifies actual absence.
            snapshot['deleted'] = True
            durable_json(path, state)
    durable_json(path, state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sandbox-id', required=True)
    parser.add_argument('--state-dir', type=Path, required=True)
    parser.add_argument('--interval', type=float, default=1800)
    parser.add_argument('--keep', type=int, default=2)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--verify-restore', metavar='SNAPSHOT_ID')
    parser.add_argument('--destination-sandbox-id')
    parser.add_argument('--restore-files', type=Path,
                        help='JSON list of exact path/bytes/sha256 bindings on /work')
    args = parser.parse_args()
    if args.interval <= 0 or args.keep < 2:
        parser.error('positive interval and at least two recovery snapshots required')
    try:
        from . import cwctl
    except ImportError:
        import cwctl
    cwctl.load_env_file()
    from cwsandbox import Sandbox
    from cwsandbox.exceptions import SnapshotNotFoundError
    auth = cwctl.sdk_auth()
    args.state_dir.mkdir(parents=True, exist_ok=True)
    with (args.state_dir / 'operator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.verify_restore:
            if not args.destination_sandbox_id or not args.restore_files:
                parser.error('restore verification needs destination and exact file bindings')
            destination = cwctl.read_sandbox(args.destination_sandbox_id, patience=45)

            def inspect_destination():
                # Use the actual backend spec; an operator's declaration or a
                # destination's self-reported environment is not restore proof.
                from cwsandbox._proto import sandbox_pb2
                async def get():
                    await destination._ensure_client()
                    return await destination._stub.GetSandbox(sandbox_pb2.GetSandboxRequest(
                        sandbox_id=args.destination_sandbox_id), timeout=30,
                        metadata=destination._auth_metadata)
                value = destination._loop_manager.run_sync(get())
                return [v.scratch.restore_from_snapshot_id for v in value.spec.volumes
                        if v.HasField('scratch')]

            def execute(argv):
                result = destination.exec(argv, timeout_seconds=3600).result(timeout=3660)
                if result.returncode:
                    raise RuntimeError('restored file verification command failed')
                return json.loads(result.stdout)

            proof = verify_restored_files(args.sandbox_id, args.verify_restore,
                args.destination_sandbox_id, json.loads(args.restore_files.read_text()),
                inspect_destination=inspect_destination, execute=execute)
            proof['expected_files_manifest'] = {
                'path': str(args.restore_files),
                'sha256': hashlib.sha256(args.restore_files.read_bytes()).hexdigest()}
            durable_json(args.state_dir / 'verified-restore.json', proof)
            print(json.dumps(proof))
            return
        sandbox = cwctl.read_sandbox(args.sandbox_id, patience=45)

        def capture(request_id):
            return sandbox.snapshot(wait_for_ready=False, request_id=request_id).result(timeout=45)

        def inspect(snapshot_id):
            s = Sandbox.get_snapshot(snapshot_id, auth=auth, timeout_seconds=30).result(timeout=45)
            return {'status': str(s.status).upper(), 'source_sandbox_id': s.source_sandbox_id,
                    'size_bytes': s.size_bytes}

        def delete(snapshot_id):
            Sandbox.delete_snapshot(snapshot_id, auth=auth, timeout_seconds=30).result(timeout=45)
            try:
                Sandbox.get_snapshot(snapshot_id, auth=auth, timeout_seconds=30).result(timeout=45)
            except SnapshotNotFoundError:
                return
            raise RuntimeError('snapshot deletion has not been independently observed')

        while True:
            try:
                result = run_once(args.state_dir, args.sandbox_id, capture=capture,
                                  inspect=inspect, delete=delete, interval=args.interval, keep=args.keep)
                print(json.dumps({'at': time.time(), 'sandbox_id': args.sandbox_id,
                                  'pending': result.get('pending'),
                                  'restore_verified': result['restore_verified']}), flush=True)
            except Exception as error:
                durable_json(args.state_dir / 'last-error.json', {
                    'at': time.time(), 'error_type': type(error).__name__,
                    'sandbox_id': args.sandbox_id, 'recovery_coverage_claimed': False})
                if args.once:
                    raise
            if args.once:
                return
            time.sleep(30)


if __name__ == '__main__':
    main()
