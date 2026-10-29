#!/usr/bin/env python3
"""Read-only private export of a replicated DiLoCo coordinator's sealed state.

The replicated coordinator (`replica_cli bootstrap`/`peer`) is authoritative as an
immutable generation baseline plus a hash-chained transaction journal; replay of the
sealed journal prefix over the baseline reproduces the exact coordinator state
(fragments, outer momentum/EMA, pending windows, quorum timers, progress, leases).
This exporter never writes into the coordinator: it reads the active generation
pointer, walks the journal chain to its newest sealed head and streams exactly those
immutable files into one archive in the `natlang.diloco-replica-seed/1` layout under
`seed/`, which `replica_seed.verify_seed` / `install_seed` accept after extraction.
No hardlinks or retention leases are taken: a link changes the ctime the coordinator
checks while hashing, and an extra leased generation trips its rollover bound.

Blobs of at least `dedupe_bytes` already contained in an earlier verified export are
referenced instead of copied (the constant seed and journal payloads shared by two
exports of one generation); `export.json` names the exact earlier object and member.

The workstation side persists an intent before upload, uploads through the existing
SHA-bound private run-file exporter, requires its complete owner GET and a local copy,
then verifies that copy member by member (hashes and journal chain) before recording
the export as verified. `restore` reassembles the seed directory from local copies.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import uuid
from pathlib import Path

SEED_SCHEMA = 'natlang.diloco-replica-seed/1'
EXPORT_SCHEMA = 'natlang.diloco-replica-export/1'
JOURNAL_SCHEMA = 'natlang.diloco-replica-export-journal/1'
BLOCK = 8 << 20


def identity(value):
    """Exactly replica_journal._identity: the chain hash of a record or identity."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------- sandbox side

class _HashingWriter(io.RawIOBase):
    def __init__(self, raw):
        self.raw = raw; self.hash = hashlib.sha256(); self.count = 0
    def writable(self): return True
    def tell(self): return self.count
    def write(self, data):
        view = memoryview(data).cast('B'); self.hash.update(view); self.count += len(view)
        return self.raw.write(view)


class _HashingReader:
    def __init__(self, raw):
        self.raw = raw; self.hash = hashlib.sha256(); self.count = 0
    def read(self, size=-1):
        data = self.raw.read(size); self.hash.update(data); self.count += len(data)
        return data


def select_sealed(store):
    """Return the active generation and its newest sealed, chain-verified journal prefix."""
    store = Path(store)
    pointer_path = store / 'primary-active.json'
    if not pointer_path.exists(): pointer_path = store / 'replica-active.json'
    pointer_raw = pointer_path.read_bytes(); pointer = json.loads(pointer_raw)
    baseline = Path(pointer['baseline']); journal = Path(pointer['journal'])
    receipt_raw = (baseline / 'replica-baseline.json').read_bytes()
    if sha256_bytes(receipt_raw) != pointer['baseline_sha256']: raise ValueError('active baseline receipt differs from pointer')
    value = json.loads(receipt_raw)
    if value.get('schema') != 'natlang.diloco-replica-baseline/1': raise ValueError('unsupported baseline')
    identity_raw = (journal / 'identity.json').read_bytes(); ident = json.loads(identity_raw)
    if ident['baseline_sha256'] != pointer['baseline_sha256'] or ident['run_identity'] != value['run_id']:
        raise ValueError('journal identity does not bind the active baseline')
    head = identity(ident)
    if head != pointer['head']: raise ValueError('journal identity differs from active pointer head')
    records = []; previous = head; sequence = 0
    while True:
        directory = journal / f'{sequence + 1:020d}'
        try: raw = (directory / 'record.json').read_bytes()
        except FileNotFoundError: break
        record = json.loads(raw)
        if record.get('schema') != 'natlang.diloco-replica-transaction/1' or record.get('sequence') != sequence + 1 or record.get('previous') != previous:
            raise ValueError('journal chain mismatch at %d' % (sequence + 1))
        sequence += 1; previous = identity(record); records.append((raw, record, previous))
    sealed = len(records) - len(records) % 2
    records = records[:sealed]
    sealed_head = records[-1][2] if records else head
    return {'pointer': pointer, 'pointer_raw': pointer_raw, 'pointer_path': str(pointer_path), 'baseline': baseline,
            'journal': journal, 'baseline_receipt_raw': receipt_raw, 'baseline_value': value, 'identity_raw': identity_raw,
            'records': records, 'sequence': sealed, 'head': sealed_head}


def pack(plan):
    """Stream the sealed seed into a new archive; returns the measured result."""
    from scripts.coreweave.cache_bounded_archive import CacheBoundedWriter
    from scripts.coreweave.private_export_guard import CacheBoundedReader
    owner = plan.get('owner')
    owner_alive = None
    if owner:
        try:
            fields = (Path('/proc') / str(owner['pid']) / 'stat').read_text().rsplit(')', 1)[1].split()
            owner_alive = (Path('/proc/sys/kernel/random/boot_id').read_text().strip() == owner['boot_id']
                           and fields[19] == str(owner['start_ticks']) and fields[0] != 'Z')
        except (OSError, IndexError): owner_alive = False
    s = select_sealed(plan['store']); exclude = plan.get('exclude', {}); dedupe = int(plan.get('dedupe_bytes', 1 << 20))
    out = Path(plan['archive']); out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists(): raise ValueError('archive already exists')
    files = []; members = []  # (arcname, bytes-or-path, size, declared sha)
    def small(relative, raw):
        files.append({'path': relative, 'bytes': len(raw), 'sha256': sha256_bytes(raw)}); members.append(('seed/' + relative, raw, len(raw), None))
    references = {}
    def large(relative, path, size, digest):
        files.append({'path': relative, 'bytes': size, 'sha256': digest})
        if size >= dedupe and digest in exclude: references[relative] = {**exclude[digest], 'sha256': digest, 'bytes': size}
        else: members.append(('seed/' + relative, Path(path), size, digest))
    small('baseline/replica-baseline.json', s['baseline_receipt_raw'])
    for item in s['baseline_value']['files']:
        relative = Path(item['path'])
        if relative.is_absolute() or '..' in relative.parts: raise ValueError('invalid baseline path')
        large('baseline/' + item['path'], s['baseline'] / relative, item['bytes'], item['sha256'])
    small('journal/identity.json', s['identity_raw'])
    for raw, record, _ in s['records']:
        name = f"journal/{record['sequence']:020d}"
        small(name + '/record.json', raw)
        large(name + '/payload', s['journal'] / f"{record['sequence']:020d}" / 'payload', record['payload_bytes'], record['payload_sha256'])
    receipt = {'schema': SEED_SCHEMA, 'run_id': s['baseline_value']['run_id'], 'baseline_sha256': s['pointer']['baseline_sha256'],
               'sequence': s['sequence'], 'head': s['head'], 'lease': None, 'freeze_nonce': 'read-only-export-' + plan['export_id'],
               'files': files}
    receipt_raw = json.dumps(receipt, sort_keys=True, indent=1).encode()
    meta = {'pointer': s['pointer'], 'pointer_path': s['pointer_path']}
    work_root = s['pointer'].get('work_root')
    for name, path in (('replica-session.json', Path(work_root) / 'replica-session.json' if work_root else None),
                       ('generation.json', s['baseline'].parent / 'generation.json')):
        if path is not None and path.is_file(): meta[name] = json.loads(path.read_bytes())
    pending = out.with_name('.' + out.name + '.pending')
    pending.unlink(missing_ok=True)
    stored = 0
    try:
        raw_out = CacheBoundedWriter(pending, window_bytes=64 << 20)
        hashed = _HashingWriter(raw_out)
        with tarfile.TarFile(fileobj=hashed, mode='w', format=tarfile.PAX_FORMAT, copybufsize=BLOCK) as tar:
            def add_bytes(arcname, raw):
                info = tarfile.TarInfo(arcname); info.size = len(raw); info.mtime = 0; info.mode = 0o600
                tar.addfile(info, io.BytesIO(raw))
            add_bytes('seed/seed-receipt.json', receipt_raw)
            for arcname, source, size, digest in members:
                if isinstance(source, bytes): add_bytes(arcname, source); continue
                with CacheBoundedReader(source, release_bytes=64 << 20) as raw:
                    before = os.fstat(raw.fileno())
                    if before.st_size != size: raise ValueError('declared size differs: ' + arcname)
                    reader = _HashingReader(raw); info = tarfile.TarInfo(arcname); info.size = size; info.mtime = 0; info.mode = 0o600
                    tar.addfile(info, reader)
                    after = os.fstat(raw.fileno())
                    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
                        raise ValueError('source changed while archiving: ' + arcname)
                    if reader.count != size or reader.hash.hexdigest() != digest: raise ValueError('content differs from its sealed digest: ' + arcname)
                    stored += size
            export = {'schema': EXPORT_SCHEMA, 'export_id': plan['export_id'], 'seed_receipt_sha256': sha256_bytes(receipt_raw),
                      'run_id': receipt['run_id'], 'baseline_sha256': receipt['baseline_sha256'], 'sequence': s['sequence'],
                      'head': s['head'], 'references': references, 'stored_bytes': stored, 'meta': meta,
                      'owner': owner, 'owner_alive': owner_alive, 'packed_at': time.time()}
            add_bytes('export.json', json.dumps(export, sort_keys=True, indent=1).encode())
        raw_out.close(); os.rename(pending, out)
        fd = os.open(out.parent, os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)
    finally:
        pending.unlink(missing_ok=True)
    st = out.stat()
    if st.st_size != hashed.count: raise ValueError('archive size differs from written bytes')
    return {'path': str(out), 'sha256': hashed.hash.hexdigest(), 'bytes': st.st_size,
            'stat': {k: getattr(st, k) for k in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')},
            'export': export, 'blobs': {f['sha256']: 'seed/' + f['path'] for f in files
                                        if f['bytes'] >= dedupe and f['path'] not in references}}


def verify_archive(path, *, expected_export_id=None):
    """Stream a local copy; check every member hash, the inventory and the journal chain."""
    files = None; export = None; seen = {}; records = {}; ident = None; baseline_raw = None
    with open(path, 'rb') as raw, tarfile.open(fileobj=raw, mode='r|') as tar:
        for member in tar:
            if not member.isfile(): raise ValueError('archive contains a non-file member')
            source = tar.extractfile(member); h = hashlib.sha256(); keep = member.size <= (1 << 20); data = []
            while block := source.read(BLOCK):
                h.update(block)
                if keep: data.append(block)
            raw_bytes = b''.join(data) if keep else None
            if member.name == 'seed/seed-receipt.json':
                receipt_raw = raw_bytes; receipt = json.loads(raw_bytes); files = {f['path']: f for f in receipt['files']}
            elif member.name == 'export.json': export = json.loads(raw_bytes)
            else:
                if not member.name.startswith('seed/') or member.name in seen: raise ValueError('unexpected member ' + member.name)
                seen[member.name[5:]] = (member.size, h.hexdigest())
                relative = member.name[5:]
                if relative == 'journal/identity.json': ident = json.loads(raw_bytes)
                elif relative == 'baseline/replica-baseline.json': baseline_raw = raw_bytes
                elif relative.endswith('/record.json'): records[int(relative.split('/')[1])] = json.loads(raw_bytes)
    if files is None or export is None: raise ValueError('archive lacks its receipt or export record')
    if export['schema'] != EXPORT_SCHEMA or export['seed_receipt_sha256'] != sha256_bytes(receipt_raw): raise ValueError('export record does not bind the seed receipt')
    if expected_export_id is not None and export['export_id'] != expected_export_id: raise ValueError('export id differs')
    for relative, item in files.items():
        if relative in seen:
            if seen[relative] != (item['bytes'], item['sha256']): raise ValueError('member differs from receipt: ' + relative)
        elif relative in export['references']:
            ref = export['references'][relative]
            if (ref['sha256'], ref['bytes']) != (item['sha256'], item['bytes']): raise ValueError('reference differs from receipt: ' + relative)
        else: raise ValueError('receipt file is neither stored nor referenced: ' + relative)
    if set(seen) - set(files): raise ValueError('archive has members outside the receipt')
    if sha256_bytes(baseline_raw) != receipt['baseline_sha256']: raise ValueError('baseline receipt digest differs')
    if ident['baseline_sha256'] != receipt['baseline_sha256']: raise ValueError('journal identity differs')
    baseline = json.loads(baseline_raw)
    expected = {'baseline/' + x['path']: (x['bytes'], x['sha256']) for x in baseline['files']}
    for relative, value in expected.items():
        if relative not in files or (files[relative]['bytes'], files[relative]['sha256']) != value: raise ValueError('baseline inventory differs')
    previous = identity(ident)
    for sequence in range(1, receipt['sequence'] + 1):
        record = records.get(sequence)
        if record is None or record['sequence'] != sequence or record['previous'] != previous: raise ValueError('journal chain mismatch')
        payload = files.get(f'journal/{sequence:020d}/payload')
        if payload is None or (payload['bytes'], payload['sha256']) != (record['payload_bytes'], record['payload_sha256']): raise ValueError('payload binding differs')
        previous = identity(record)
    if previous != receipt['head'] or len(records) != receipt['sequence'] or receipt['sequence'] % 2: raise ValueError('sealed head differs')
    return {'export_id': export['export_id'], 'sequence': receipt['sequence'], 'head': receipt['head'],
            'baseline_sha256': receipt['baseline_sha256'], 'files': len(files), 'stored': len(seen),
            'referenced': len(export['references']), 'seed_receipt_sha256': export['seed_receipt_sha256'], 'export': export}


def restore(journal_path, export_id, destination):
    """Reassemble the seed directory of one verified export from local copies."""
    state = json.loads(Path(journal_path).read_text())
    entries = {e['export_id']: e for e in state['exports'] if e.get('phase') == 'verified'}
    entry = entries[export_id]; destination = Path(destination); destination.mkdir(parents=True, exist_ok=False)
    def extract(local, wanted):
        with open(local, 'rb') as raw, tarfile.open(fileobj=raw, mode='r|') as tar:
            for member in tar:
                if member.name in wanted:
                    target = destination / wanted.pop(member.name); target.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(member) as src, target.open('xb') as dst: shutil.copyfileobj(src, dst, BLOCK)
    export = entry['verification']['export']
    names = None
    with open(entry['local_copy'], 'rb') as raw, tarfile.open(fileobj=raw, mode='r|') as tar:
        names = [m.name for m in tar if m.name.startswith('seed/')]
    extract(entry['local_copy'], {n: n[5:] for n in names})
    by_object = {}
    for relative, ref in export['references'].items(): by_object.setdefault(ref['export_id'], {})[ref['member']] = relative
    for source_id, wanted in by_object.items(): extract(entries[source_id]['local_copy'], wanted)
    receipt = json.loads((destination / 'seed-receipt.json').read_text())
    for item in receipt['files']:
        path = destination / item['path']
        if path.stat().st_size != item['bytes'] or hashlib.file_digest(path.open('rb'), 'sha256').hexdigest() != item['sha256']:
            raise ValueError('restored file differs: ' + item['path'])
    return {'destination': str(destination), 'seed_receipt_sha256': entry['verification']['seed_receipt_sha256'],
            'sequence': receipt['sequence'], 'head': receipt['head'], 'files': len(receipt['files'])}


# ---------------------------------------------------------------- workstation side

CW_PYTHON = Path.home() / '.local/share/natlang/cw-venv/bin/python'
EXPORTER = Path('/home/werg/data/dgxops/coreweave-private-runfile-export-20261014.py')
RECEIPTS = Path('/home/werg/data/dgxops')
CODE = ('scripts/__init__.py', 'scripts/coreweave/diloco_replica_export.py',
        'scripts/coreweave/cache_bounded_archive.py', 'scripts/coreweave/private_export_guard.py')


def _write(path, value):
    from natlang_neuralese.common.jsonio import write_json_durable
    write_json_durable(path, value)


def _remote_code(box, directory):
    repo = Path(__file__).resolve().parents[2]; digests = {}
    for relative in CODE:
        raw = (repo / relative).read_bytes(); compile(raw, relative, 'exec')
        box.write_file(directory + '/' + relative, raw).result(timeout=120); digests[relative] = sha256_bytes(raw)
    return digests


def _exclusions(state):
    """Blobs of verified exports whose local copies and private objects exist."""
    out = {}
    for entry in state['exports']:
        if entry.get('phase') != 'verified' or not Path(entry['local_copy']).is_file(): continue
        for digest, member in entry['pack'].get('blobs', {}).items():
            out.setdefault(digest, {'export_id': entry['export_id'], 'member': member, 'object': entry['object']})
    return out


def cycle(config, state, journal):
    import cwctl
    cwctl.load_env_file()
    export_id = time.strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:8]
    entry = {'export_id': export_id, 'phase': 'packing', 'started_at': time.time()}
    state['exports'].append(entry); _write(journal, state)
    box = cwctl.read_sandbox(config['sandbox_id'], patience=300)
    staging = config['staging'].rstrip('/') + '/' + export_id
    entry['code'] = _remote_code(box, staging + '/code')
    plan = {'store': config['store'], 'archive': staging + '/coordinator.tar', 'export_id': export_id,
            'owner': config.get('owner'), 'dedupe_bytes': config.get('dedupe_bytes', 1 << 20),
            'exclude': _exclusions(state)}
    box.write_file(staging + '/plan.json', json.dumps(plan).encode()).result(timeout=120)
    runner = ('import json,sys;sys.path.insert(0,sys.argv[1]);from scripts.coreweave.diloco_replica_export import pack;'
              'print(json.dumps(pack(json.loads(open(sys.argv[2]).read()))))')
    compile(runner, 'runner', 'exec')
    result = box.exec(['nice', '-n', '19', 'ionice', '-c', '2', '-n', '7', 'env', 'OMP_NUM_THREADS=1', 'python3', '-c', runner,
                       staging + '/code', staging + '/plan.json'], timeout_seconds=4 * 3600).result(timeout=4 * 3600 + 60)
    if result.returncode != 0: raise RuntimeError('pack failed: ' + result.stderr[-1500:])
    packed = json.loads(result.stdout.strip().splitlines()[-1])
    export = packed['export']
    role = f"{config['role_prefix']}-s{export['sequence']}-{export_id[-8:]}"
    intent = {'schema': 'natlang.diloco-replica-export-intent/1', 'export_id': export_id, 'role': role,
              'sandbox_id': config['sandbox_id'], 'path': packed['path'], 'sha256': packed['sha256'], 'bytes': packed['bytes'],
              'stat': packed['stat'], 'run_id': export['run_id'], 'baseline_sha256': export['baseline_sha256'],
              'sequence': export['sequence'], 'head': export['head'], 'generation': export['meta']['pointer'].get('id'),
              'seed_receipt_sha256': export['seed_receipt_sha256'], 'owner': export['owner'], 'owner_alive': export['owner_alive'],
              'references': len(export['references']), 'at': time.time()}
    intent_path = Path(config['state_root']) / f'coordinator-v5-export-{export_id}-intent.json'
    _write(intent_path, intent)
    entry.update(phase='uploading', pack={k: packed[k] for k in ('path', 'sha256', 'bytes', 'stat', 'blobs')},
                 intent=str(intent_path), intent_sha256=hashlib.sha256(intent_path.read_bytes()).hexdigest(), role=role,
                 sequence=export['sequence'], head=export['head'], generation=intent['generation'], owner_alive=export['owner_alive'])
    _write(journal, state)
    local = Path(config['local_root']) / export_id / 'coordinator.tar'
    local.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, 'CW_EXPORT_SANDBOX_ID': config['sandbox_id'], 'CW_EXPORT_EXPECTED_SHA256': packed['sha256'],
           'CW_EXPORT_EXPECTED_BYTES': str(packed['bytes']), 'CW_EXPORT_CACHE_BOUNDED_READS': '1',
           'CW_EXPORT_SEALED_PIN': json.dumps({'path': packed['path'], 'identity': {'sha256': packed['sha256'], 'bytes': packed['bytes']},
                                               'stat': packed['stat']}),
           'CW_EXPORT_VERIFY_ON_OWNER': '1', 'CW_EXPORT_OWNER_DESTINATION': str(local)}
    log = Path(config['state_root']) / f'coordinator-v5-export-{export_id}.log'
    with log.open('ab') as output:
        code = subprocess.run([str(CW_PYTHON), str(EXPORTER), role, packed['path']], env=env, stdout=output, stderr=subprocess.STDOUT).returncode
    receipt_path = RECEIPTS / f'coreweave-production-{role}-private-export-receipt-20261014.json'
    if code != 0 or not receipt_path.exists(): raise RuntimeError(f'private export failed ({code}); see {log}')
    receipt = json.loads(receipt_path.read_text()); check = receipt['independent_verification']
    if (not receipt.get('private') or check.get('readback_location') != 'owner' or check.get('sha256') != packed['sha256']
            or check.get('bytes') != packed['bytes'] or receipt['identity'] != {'sha256': packed['sha256'], 'bytes': packed['bytes']}):
        raise RuntimeError('receipt lacks an exact owner GET')
    copy = check.get('local_copy') or {}
    entry.update(receipt=str(receipt_path), receipt_sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
                 object={'run': receipt['run'], 'name': receipt['name']}, owner_get=True, local_copy_record=copy)
    if not copy.get('committed'):
        entry['phase'] = 'cloud-verified-no-local-copy'; _write(journal, state)
        raise RuntimeError('local copy not committed: ' + json.dumps(copy))
    entry.update(phase='verifying-local', local_copy=copy['path']); _write(journal, state)
    verification = verify_archive(copy['path'], expected_export_id=export_id)
    if (verification['sequence'], verification['head']) != (export['sequence'], export['head']): raise RuntimeError('local copy differs from intent')
    entry.update(phase='verified', verification=verification, verified_at=time.time()); _write(journal, state)
    removed = box.exec(['rm', '-rf', '--', staging], timeout_seconds=300).result(timeout=330)
    entry['staging_removed'] = removed.returncode == 0; _write(journal, state)
    return entry


def run(config_path):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    config = json.loads(Path(config_path).read_text())
    journal = Path(config['state_root']) / config.get('journal_name', 'coordinator-v5-journal.json')
    state = json.loads(journal.read_text()) if journal.exists() else {'schema': JOURNAL_SCHEMA, 'config': config, 'exports': []}
    for entry in state['exports']:
        if entry.get('phase') not in ('verified', 'failed', 'cloud-verified-no-local-copy'):
            entry.update(phase='failed', error='interrupted before verification (supervisor restart)')
    _write(journal, state)
    while True:
        last = max((e['started_at'] for e in state['exports']), default=0)
        failed_recently = state['exports'] and state['exports'][-1].get('phase') != 'verified'
        wait = (config.get('retry_seconds', 600) if failed_recently else config.get('interval_seconds', 2700)) - (time.time() - last)
        if wait > 0: time.sleep(min(wait, 300)); continue
        try:
            entry = cycle(config, state, journal)
            print(json.dumps({'verified': entry['export_id'], 'sequence': entry['sequence'], 'sha256': entry['pack']['sha256']}), flush=True)
        except Exception as error:
            entry = state['exports'][-1]
            if entry.get('phase') != 'cloud-verified-no-local-copy': entry['phase'] = 'failed'
            entry['error'] = f'{type(error).__name__}: {str(error)[:1500]}'; _write(journal, state)
            print('export cycle failed', entry['error'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='action', required=True)
    r = sub.add_parser('run'); r.add_argument('config')
    v = sub.add_parser('verify'); v.add_argument('archive')
    s = sub.add_parser('restore'); s.add_argument('journal'); s.add_argument('export_id'); s.add_argument('destination')
    a = parser.parse_args()
    if a.action == 'run': return run(a.config)
    print(json.dumps(verify_archive(a.archive) if a.action == 'verify' else restore(a.journal, a.export_id, a.destination), indent=1))


if __name__ == '__main__':
    main()
